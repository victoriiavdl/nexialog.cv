"""
Tuning HP vs Baseline — comparaison walk-forward post-COVID.

Protocole (Stratégie 1 — "tune once, apply everywhere") :
  1. Sur le train du cutoff le plus ancien (2022-01 → 2024-01), Optuna TPE
     avec TimeSeriesSplit(3) choisit les meilleurs HP pour CatBoost et
     XGBoost (Ridge/RandomForest restent baseline par design).
  2. HP gelés.
  3. Relance du walk-forward 3 cutoffs (2024-01, 2024-07, 2025-01) avec
     ces HP gelés pour CB et XGB.
  4. Comparaison métrique-par-métrique vs baseline (rolling_origin_postcovid_results.json).

Output -> tuning_vs_baseline_results.json (consommé par le dashboard).

GPU obligatoire (CatBoost task_type=GPU, XGBoost device=cuda).
"""
import io
import json
import sys
import time
import warnings
from pathlib import Path

# Windows console est en cp1252 par défaut et ne peut pas afficher les caractères
# UTF-8 (ex: Δ, €). On force UTF-8 pour éviter UnicodeEncodeError en print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import optuna
import pandas as pd

import vr_pipeline as vp
import rolling_origin_postcovid as ro


N_TRIALS = 15
N_SPLITS_INNER = 3
RANDOM_SEED = 42
BASELINE_PATH = "rolling_origin_postcovid_results.json"
TUNED_PATH = "rolling_origin_postcovid_tuned.json"
TUNING_PATH = "tuning_params_postcovid.json"
OUTPUT_PATH = "tuning_vs_baseline_results.json"

# --------------------------------------------------------------------------- #
# 1. Prépare le train du cutoff le plus ancien (mêmes FE que le WF)
# --------------------------------------------------------------------------- #

def prepare_train_for_tuning() -> tuple[pd.DataFrame, list[str]]:
    um = vp.load_used_market()
    df = vp.prepare_used_market(um)
    cutoff = ro.CUTOFFS[0]
    df_tr = df[
        (df["date de vente"] >= ro.POST_COVID_START)
        & (df["date de vente"] < cutoff)
    ].copy()

    clustering = vp.cluster_high_cardinality_categorical(
        df_tr, column="model", n_clusters=6
    )
    df_tr = vp.apply_clustering(df_tr, clustering, new_col="model_family")
    df_tr = vp.add_derived_ratio_features(df_tr)
    comp_stats = vp.build_comparable_stats(
        df_tr, group_cols=("brand", "model", "fuel_type")
    )
    df_tr = vp.merge_comparable_features(df_tr, comp_stats)

    num_features_ext = (
        list(vp.NUM_FEATURES) + vp.DERIVED_RATIO_FEATURES + vp.COMP_FEATURES
    )
    df_tr = df_tr.sort_values("date de vente").reset_index(drop=True)
    return df_tr, num_features_ext


# --------------------------------------------------------------------------- #
# 2. Optuna tuners (TSSplit(3) via _cv_mape_eur_timeseries)
# --------------------------------------------------------------------------- #

def _tune_one(
    name: str,
    trainer,
    suggest_params,
    df_train: pd.DataFrame,
    num_features_ext: list[str],
    n_trials: int,
) -> dict:
    trials_log: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        try:
            score = vp._cv_mape_eur_timeseries(
                trainer,
                df_train,
                params,
                n_splits=N_SPLITS_INNER,
                num_features=num_features_ext,
            )
        except Exception as exc:
            raise optuna.TrialPruned() from exc
        return score

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        val = trial.value
        trials_log.append({
            "trial": trial.number,
            "state": trial.state.name,
            "cv_mape": None if val is None else round(float(val), 4),
            "params": dict(trial.params),
        })
        if val is not None:
            print(
                f"    [{name:<8}] trial {trial.number:>2}/{n_trials-1}  "
                f"{trial.state.name:<8}  cv_mape={val:.4f}%"
            )
        else:
            print(
                f"    [{name:<8}] trial {trial.number:>2}/{n_trials-1}  "
                f"{trial.state.name:<8}  (pruned)"
            )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
    elapsed = round(time.perf_counter() - t0, 2)

    return {
        "best_params": dict(study.best_params),
        "best_cv_mape": round(float(study.best_value), 4),
        "trials": trials_log,
        "elapsed_seconds": elapsed,
    }


def _suggest_catboost(trial: optuna.Trial) -> dict:
    return {
        "iterations":    trial.suggest_int("iterations", 500, 1500),
        "depth":         trial.suggest_int("depth", 5, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


def _suggest_xgboost(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":  trial.suggest_int("n_estimators", 300, 1000),
        "max_depth":     trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
    }


# --------------------------------------------------------------------------- #
# 3. Diff baseline vs tuned
# --------------------------------------------------------------------------- #

def _compute_diff(baseline: dict, tuned: dict) -> dict:
    """Diff par modèle sur toutes les métriques résumées + par fold."""
    bm = baseline["summary"]["per_model"]
    tm = tuned["summary"]["per_model"]
    per_model = {}
    for name in baseline["models"]:
        b, t = bm[name], tm[name]
        per_model[name] = {
            "mape_eur_baseline":  b["mape_eur_mean"],
            "mape_eur_tuned":     t["mape_eur_mean"],
            "mape_eur_delta":     round(t["mape_eur_mean"] - b["mape_eur_mean"], 3),
            "mape_eur_std_baseline": b["mape_eur_std"],
            "mape_eur_std_tuned":    t["mape_eur_std"],
            "r2_eur_baseline":    b["r2_eur_mean"],
            "r2_eur_tuned":       t["r2_eur_mean"],
            "r2_eur_delta":       round(t["r2_eur_mean"] - b["r2_eur_mean"], 4),
            "r2_logratio_baseline": b["r2_logratio_mean"],
            "r2_logratio_tuned":    t["r2_logratio_mean"],
            "r2_logratio_delta":    round(t["r2_logratio_mean"] - b["r2_logratio_mean"], 4),
        }

    per_fold = []
    for bf, tf in zip(baseline["folds"], tuned["folds"]):
        row = {"cutoff": bf["cutoff"]}
        for name in baseline["models"]:
            b = bf["models"][name]
            t = tf["models"][name]
            row[name] = {
                "mape_eur_baseline":    b["mape_eur"],
                "mape_eur_tuned":       t["mape_eur"],
                "mape_eur_delta":       round(t["mape_eur"] - b["mape_eur"], 3),
                "r2_eur_baseline":      b["r2_eur"],
                "r2_eur_tuned":         t["r2_eur"],
                "r2_eur_delta":         round(t["r2_eur"] - b["r2_eur"], 4),
                "mape_logratio_baseline": b["mape_logratio"],
                "mape_logratio_tuned":    t["mape_logratio"],
                "mape_logratio_delta":    round(t["mape_logratio"] - b["mape_logratio"], 3),
                "r2_logratio_baseline":   b["r2_logratio"],
                "r2_logratio_tuned":      t["r2_logratio"],
                "r2_logratio_delta":      round(t["r2_logratio"] - b["r2_logratio"], 4),
            }
        per_fold.append(row)

    best_b = baseline["summary"]["best_model_global"]
    best_t = tuned["summary"]["best_model_global"]
    return {
        "per_model": per_model,
        "per_fold":  per_fold,
        "best_model_baseline": best_b,
        "best_model_tuned":    best_t,
        "best_mape_baseline":  baseline["summary"]["best_mape_mean"],
        "best_mape_tuned":     tuned["summary"]["best_mape_mean"],
        "best_mape_delta":     round(
            tuned["summary"]["best_mape_mean"]
            - baseline["summary"]["best_mape_mean"], 3
        ),
    }


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #

def _ensure_baseline() -> dict:
    if Path(BASELINE_PATH).exists():
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"  [baseline] absent -> run walk-forward baseline")
    return ro.run_rolling_origin_postcovid(output_path=BASELINE_PATH, tag="baseline")


def _ensure_cv_std(tuning: dict, df_train, num_features_ext) -> dict:
    """Complète tuning_params avec cv_mape_std pour best HP et baseline HP.

    Mesure la stabilité intra-CV : écart-type du MAPE EUR sur les 3 plis
    TimeSeriesSplit pour un même HP. Fait une passe CV par modèle si les
    valeurs sont absentes — critère de décision "modèle le plus stable".
    """
    trainer_map = {"catboost": vp.train_catboost, "xgboost": vp.train_xgboost}
    needs_update = False
    for name, trainer in trainer_map.items():
        entry = tuning.get(name, {})
        if "cv_mape_std_tuned" in entry and "cv_mape_std_baseline" in entry:
            continue
        needs_update = True
        print(f"\n  [cv-std] {name} : 1 passe TSSplit(3) sur best HP + baseline")
        best_folds = vp.cv_mape_fold_values(
            trainer, df_train, entry["best_params"],
            n_splits=N_SPLITS_INNER, num_features=num_features_ext,
        )
        baseline_folds = vp.cv_mape_fold_values(
            trainer, df_train, {},
            n_splits=N_SPLITS_INNER, num_features=num_features_ext,
        )
        entry["cv_mape_folds_tuned"]    = [round(v, 4) for v in best_folds]
        entry["cv_mape_mean_tuned"]     = round(float(np.mean(best_folds)), 4)
        entry["cv_mape_std_tuned"]      = round(float(np.std(best_folds, ddof=1)), 4)
        entry["cv_mape_folds_baseline"] = [round(v, 4) for v in baseline_folds]
        entry["cv_mape_mean_baseline"]  = round(float(np.mean(baseline_folds)), 4)
        entry["cv_mape_std_baseline"]   = round(float(np.std(baseline_folds, ddof=1)), 4)
        tuning[name] = entry
        print(
            f"    tuned    mean={entry['cv_mape_mean_tuned']:.4f}%  "
            f"std={entry['cv_mape_std_tuned']:.4f}pt  folds={entry['cv_mape_folds_tuned']}"
        )
        print(
            f"    baseline mean={entry['cv_mape_mean_baseline']:.4f}%  "
            f"std={entry['cv_mape_std_baseline']:.4f}pt  folds={entry['cv_mape_folds_baseline']}"
        )
    if needs_update:
        with open(TUNING_PATH, "w", encoding="utf-8") as f:
            json.dump(tuning, f, indent=2, ensure_ascii=False)
        print(f"  -> {TUNING_PATH}  (cv_mape_std ajouté)")
    return tuning


def _load_or_run_tuning() -> dict:
    """Charge les résultats Optuna si déjà calculés, sinon lance le tuning."""
    if Path(TUNING_PATH).exists():
        print(f"  [skip] tuning déjà calculé dans {TUNING_PATH}")
        with open(TUNING_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"\n  [train tuning] cutoff={ro.CUTOFFS[0].date()}  "
          f"TSSplit(n={N_SPLITS_INNER})")
    df_train, num_features_ext = prepare_train_for_tuning()
    print(f"                 n_train={len(df_train):>6,}")

    print("\n  [ref] CV MAPE baseline (HP par défaut)...")
    cv_baseline_cb = vp._cv_mape_eur_timeseries(
        vp.train_catboost, df_train, {}, N_SPLITS_INNER, num_features_ext
    )
    print(f"    catboost baseline CV MAPE = {cv_baseline_cb:.4f}%")
    cv_baseline_xgb = vp._cv_mape_eur_timeseries(
        vp.train_xgboost, df_train, {}, N_SPLITS_INNER, num_features_ext
    )
    print(f"    xgboost  baseline CV MAPE = {cv_baseline_xgb:.4f}%")

    print("\n  [optuna] CatBoost tuning...")
    cb_res = _tune_one(
        "catboost", vp.train_catboost, _suggest_catboost,
        df_train, num_features_ext, N_TRIALS,
    )
    print(f"    -> best CB: {cb_res['best_params']}  "
          f"CV MAPE={cb_res['best_cv_mape']:.4f}%  "
          f"({cb_res['elapsed_seconds']:.1f}s)")

    print("\n  [optuna] XGBoost tuning...")
    xgb_res = _tune_one(
        "xgboost", vp.train_xgboost, _suggest_xgboost,
        df_train, num_features_ext, N_TRIALS,
    )
    print(f"    -> best XGB: {xgb_res['best_params']}  "
          f"CV MAPE={xgb_res['best_cv_mape']:.4f}%  "
          f"({xgb_res['elapsed_seconds']:.1f}s)")

    tuning = {
        "catboost": {
            **cb_res,
            "baseline_cv_mape": round(float(cv_baseline_cb), 4),
            "search_space": {
                "iterations": [500, 1500],
                "depth": [5, 9],
                "learning_rate": [0.02, 0.1],
                "l2_leaf_reg": [1.0, 10.0],
            },
        },
        "xgboost": {
            **xgb_res,
            "baseline_cv_mape": round(float(cv_baseline_xgb), 4),
            "search_space": {
                "n_estimators": [300, 1000],
                "max_depth": [4, 8],
                "learning_rate": [0.02, 0.1],
            },
        },
    }
    with open(TUNING_PATH, "w", encoding="utf-8") as f:
        json.dump(tuning, f, indent=2, ensure_ascii=False)
    print(f"\n  -> {TUNING_PATH}  (cache Optuna)")
    return tuning


def _load_or_run_tuned_wf(hp_override: dict) -> dict:
    """Charge le walk-forward tuné si déjà calculé, sinon le relance."""
    if Path(TUNED_PATH).exists():
        print(f"  [skip] walk-forward tuned déjà calculé dans {TUNED_PATH}")
        with open(TUNED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print("\n  [walk-forward] relance 3 folds avec HP tunes...")
    return ro.run_rolling_origin_postcovid(
        hp_override=hp_override,
        output_path=TUNED_PATH,
        tag="tuned",
    )


def _build_verdict(mape_delta: float, best_tuned: str, best_baseline: str,
                   mape_tuned: float, mape_baseline: float) -> str:
    sign = "+" if mape_delta >= 0 else ""
    parts = [
        f"Best tuned : {best_tuned} MAPE mean={mape_tuned:.2f}% "
        f"(baseline best={best_baseline} {mape_baseline:.2f}%, "
        f"delta={sign}{mape_delta:.2f}pt).",
    ]
    if mape_delta < -0.25:
        parts.append("Gain significatif du tuning.")
    elif mape_delta > 0.25:
        parts.append("Degradation -> baseline retenue.")
    else:
        parts.append(
            "Gain non significatif (|delta|<0.25pt) -> baseline robuste."
        )
    return " ".join(parts)


def run_tuning_vs_baseline():
    t_top = time.perf_counter()
    device = "GPU" if vp.gpu_available() else "CPU"
    print("=" * 72)
    print(f"  Tuning vs Baseline  |  device={device}  |  TSSplit(3) + Optuna TPE")
    print(f"  {N_TRIALS} trials par modele  |  Strategy = tune once on oldest train, apply to all folds")
    print("=" * 72)

    # 1) Baseline (load or run)
    baseline = _ensure_baseline()
    print(
        f"  Baseline : best={baseline['summary']['best_model_global']}  "
        f"MAPE mean={baseline['summary']['best_mape_mean']:.2f}%  "
        f"std={baseline['summary']['best_mape_std']:.2f}pt"
    )

    # 2) Tuning (cached or fresh)
    tuning = _load_or_run_tuning()

    # 2-bis) Complète cv_mape_std (stabilité intra-CV) si absent
    if any(
        "cv_mape_std_tuned" not in tuning.get(m, {})
        for m in ("catboost", "xgboost")
    ):
        df_train, num_features_ext = prepare_train_for_tuning()
        tuning = _ensure_cv_std(tuning, df_train, num_features_ext)

    hp_override = {
        "catboost": dict(tuning["catboost"]["best_params"]),
        "xgboost":  dict(tuning["xgboost"]["best_params"]),
    }

    # 3) Walk-forward tuné (cached or fresh)
    tuned = _load_or_run_tuned_wf(hp_override)

    # 4) Diff
    diff = _compute_diff(baseline, tuned)
    verdict = _build_verdict(
        diff["best_mape_delta"],
        diff["best_model_tuned"],
        diff["best_model_baseline"],
        diff["best_mape_tuned"],
        diff["best_mape_baseline"],
    )

    total_sec = round(time.perf_counter() - t_top, 2)
    payload = {
        "protocol": "tune-once-apply-everywhere",
        "inner_cv": f"TimeSeriesSplit(n={N_SPLITS_INNER})",
        "tuning_train_cutoff": str(ro.CUTOFFS[0].date()),
        "n_trials": N_TRIALS,
        "device": device,
        "total_seconds": total_sec,
        "baseline": baseline,
        "tuned": tuned,
        "tuning": tuning,
        "diff": diff,
        "verdict": verdict,
    }
    # Ecrire IMMEDIATEMENT avant tout print (cp1252 fragile sur Windows).
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  -> {OUTPUT_PATH}  (total {total_sec:.1f}s)")
    print(f"  [verdict] {verdict}")
    print("=" * 72)
    return payload


if __name__ == "__main__":
    run_tuning_vs_baseline()
