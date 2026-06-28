"""
CatBoost HP tuning avec TimeSeriesSplit + Optuna.

Protocole :
  1. Split temporel classique (train < 2024-01, test >= 2024-01).
  2. Sur le train SEULEMENT, TimeSeriesSplit(n=3) pour valider les HP en
     respectant l'ordre chronologique (train_fold < valid_fold).
  3. Optuna (TPE sampler + Median pruner) explore l'espace :
       iterations    ~ int  [500, 1500]
       depth         ~ int  [5, 9]
       learning_rate ~ log-uniform [0.02, 0.1]
       l2_leaf_reg   ~ log-uniform [1, 10]
  4. Le meilleur trial est ré-entraîné sur 100% du train temporel et
     évalué sur le test temporel (EUR + log_ratio).
  5. Résultats sérialisés dans optuna_tuning_results.json, consommés
     par le dashboard pour montrer la courbe d'optimisation et le gain.

Gain attendu vs baseline CatBoost non tuné : +1 à +3pt de MAPE temporelle,
car le tuning actuel (`CATBOOST_SEARCH_SPACE` dans vr_pipeline.py) est
validé en KFold stratifié — donc les HP choisis ne résistent pas au drift
temporel.
"""
import json
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

import vr_pipeline as vp


CUTOFF = pd.Timestamp("2024-01-01")
OUTPUT_PATH = "optuna_tuning_results.json"
N_TRIALS = 15
N_SPLITS = 3
RANDOM_SEED = 42


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def _log_ratio_metrics(model, df_test: pd.DataFrame) -> dict:
    y_true = df_test[vp.TARGET].to_numpy()
    y_pred = model.predict(df_test)
    residuals = y_true - y_pred
    return {
        "R² (log_ratio)": round(float(r2_score(y_true, y_pred)), 4),
        "MAE (log_ratio)": round(float(np.mean(np.abs(residuals))), 4),
        "MAPE (log_ratio) %": round(
            float(np.mean(np.abs(residuals / np.where(y_true == 0, 1e-9, y_true))) * 100), 2
        ),
    }


def build_objective(df_train_sorted: pd.DataFrame, num_features_ext: list):
    """Crée la fonction objectif Optuna. Capture df + features en closure."""
    X_index = np.arange(len(df_train_sorted))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations":    trial.suggest_int("iterations", 500, 1500),
            "depth":         trial.suggest_int("depth", 5, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }

        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        fold_scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_index)):
            df_tr = df_train_sorted.iloc[tr_idx]
            df_va = df_train_sorted.iloc[va_idx]

            try:
                model = vp.train_catboost(
                    df_tr,
                    num_features=num_features_ext,
                    iterations=params["iterations"],
                    depth=params["depth"],
                    learning_rate=params["learning_rate"],
                    l2_leaf_reg=params["l2_leaf_reg"],
                )
            except Exception as exc:
                raise optuna.TrialPruned() from exc

            y_va = df_va["prix_vente"].to_numpy()
            y_pred = vp.predict_eur(model, df_va)
            mape = _mape(y_va, y_pred)
            fold_scores.append(mape)

            # Report intermédiaire pour le pruner
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def run_optuna_tuning():
    print("=" * 72)
    print(f"  Optuna HP tuning — CatBoost · TimeSeriesSplit(n={N_SPLITS}) · {N_TRIALS} trials")
    print("=" * 72)

    um = vp.load_used_market()
    df = vp.prepare_used_market(um)
    df_train = df[df["date de vente"] < CUTOFF].copy()
    df_test = df[df["date de vente"] >= CUTOFF].copy()

    print(f"  Train : {len(df_train):>6,}  Test : {len(df_test):>6,}")

    # Feature engineering identique à temporal_split_check
    clustering = vp.cluster_high_cardinality_categorical(
        df_train, column="model", n_clusters=6
    )
    df_train = vp.apply_clustering(df_train, clustering, new_col="model_family")
    df_test = vp.apply_clustering(df_test, clustering, new_col="model_family")
    df_train = vp.add_derived_ratio_features(df_train)
    df_test = vp.add_derived_ratio_features(df_test)
    comp_stats = vp.build_comparable_stats(
        df_train, group_cols=("brand", "model", "fuel_type")
    )
    df_train = vp.merge_comparable_features(df_train, comp_stats)
    df_test = vp.merge_comparable_features(df_test, comp_stats)

    num_features_ext = list(vp.NUM_FEATURES) + vp.DERIVED_RATIO_FEATURES + vp.COMP_FEATURES

    # Baseline non tuné (HP par défaut de vp.train_catboost)
    print("\n  [baseline] train_catboost avec HP par défaut (1200, 8, 0.05)...")
    baseline = vp.train_catboost(df_train, num_features=num_features_ext)
    baseline_eur = vp.evaluate_model(baseline, df_test, model_name="catboost_baseline")
    baseline_lr = _log_ratio_metrics(baseline, df_test)
    print(f"        MAPE={baseline_eur['MAPE (%)']:.2f}%  R²={baseline_eur['R²']:.4f}  "
          f"R²_lr={baseline_lr['R² (log_ratio)']:.4f}")

    # Optuna search
    df_train_sorted = df_train.sort_values("date de vente").reset_index(drop=True)
    print("\n  [optuna] lancement de l'optimisation (TPE + Median pruner)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=1
        ),
    )

    objective = build_objective(df_train_sorted, num_features_ext)
    trial_log = []

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        status = trial.state.name
        val = trial.value if trial.value is not None else float("nan")
        msg = (
            f"        trial {trial.number:>2}/{N_TRIALS - 1}  {status:<8}  "
            f"cv_mape={val:.2f}%"
            if not np.isnan(val)
            else f"        trial {trial.number:>2}/{N_TRIALS - 1}  {status:<8}  (pruned)"
        )
        print(msg)
        trial_log.append({
            "trial": trial.number,
            "state": status,
            "cv_mape": None if np.isnan(val) else round(val, 4),
            "params": trial.params,
        })

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_callback])

    best_params = study.best_params
    best_cv_mape = float(study.best_value)
    print(f"\n  [optuna] best CV MAPE = {best_cv_mape:.2f}%")
    print(f"          best params  = {best_params}")

    # Re-entraîner sur 100% du train temporel + évaluer sur test
    print("\n  [refit] entraînement final avec les meilleurs HP sur 100% train...")
    best_model = vp.train_catboost(
        df_train,
        num_features=num_features_ext,
        iterations=best_params["iterations"],
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
    )
    test_eur = vp.evaluate_model(best_model, df_test, model_name="catboost_optuna")
    test_lr = _log_ratio_metrics(best_model, df_test)
    print(f"        TEST   MAPE={test_eur['MAPE (%)']:.2f}%  R²={test_eur['R²']:.4f}  "
          f"R²_lr={test_lr['R² (log_ratio)']:.4f}")

    delta_mape = round(test_eur["MAPE (%)"] - baseline_eur["MAPE (%)"], 3)
    delta_r2_lr = round(test_lr["R² (log_ratio)"] - baseline_lr["R² (log_ratio)"], 4)
    keep = bool(delta_mape < 0 and delta_r2_lr >= 0)
    reason = (
        f"MAPE Δ={delta_mape:+.3f}pt · R²_log_ratio Δ={delta_r2_lr:+.4f}"
        + (" → on wire les HP en prod" if keep else " → gain non significatif, on garde baseline")
    )
    print(f"  [decision] {reason}")

    result: dict[str, Any] = {
        "cutoff": str(CUTOFF.date()),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_trials": N_TRIALS,
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
        "baseline": {
            "hp": {"iterations": 1200, "depth": 8, "learning_rate": 0.05},
            "mape_eur": baseline_eur["MAPE (%)"],
            "r2_eur": baseline_eur["R²"],
            "mae_eur": baseline_eur["MAE (€)"],
            "mape_logratio": baseline_lr["MAPE (log_ratio) %"],
            "r2_logratio": baseline_lr["R² (log_ratio)"],
        },
        "best": {
            "hp": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best_params.items()},
            "cv_mape": round(best_cv_mape, 4),
            "mape_eur": test_eur["MAPE (%)"],
            "r2_eur": test_eur["R²"],
            "mae_eur": test_eur["MAE (€)"],
            "mape_logratio": test_lr["MAPE (log_ratio) %"],
            "r2_logratio": test_lr["R² (log_ratio)"],
        },
        "delta_mape_pts": delta_mape,
        "delta_r2_logratio": delta_r2_lr,
        "keep_tuning": keep,
        "decision_reason": reason,
        "trials": trial_log,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  -> {OUTPUT_PATH}")
    print("=" * 72)
    return result


if __name__ == "__main__":
    run_optuna_tuning()
