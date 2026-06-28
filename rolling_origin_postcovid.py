"""
Walk-forward post-COVID : rolling-origin backtesting tous modèles.

Protocole :
  - Train = [POST_COVID_START, cutoff) à 3 cutoffs (2024-01, 2024-07, 2025-01).
    Fenêtres train croissantes (24/30/36 mois).
  - Test  = [cutoff, cutoff + HORIZON_MONTHS).
  - Feature engineering (clustering + derived + comp_stats) fit sur
    train-only à chaque fold.
  - Les 4 modèles (Ridge, XGBoost, CatBoost, RandomForest) + baselines naïfs
    (group brand × age_year et global median ratio) sont évalués à chaque fold.
  - GPU auto-détecté pour CatBoost / XGBoost.

Output -> rolling_origin_postcovid_results.json, consommé directement par
le dashboard (section Walk-Forward).
"""
import json
import time
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import r2_score

import vr_pipeline as vp


POST_COVID_START = vp.POST_COVID_TRAIN_START
CUTOFFS = [
    pd.Timestamp("2024-01-01"),
    pd.Timestamp("2024-07-01"),
    pd.Timestamp("2025-01-01"),
]
HORIZON_MONTHS = 6
OUTPUT_PATH = "rolling_origin_postcovid_results.json"
MODELS = list(vp.DEFAULT_MODELS)


def _log_ratio_metrics(model, df_test: pd.DataFrame) -> dict:
    y_true = df_test[vp.TARGET].to_numpy()
    y_pred = model.predict(df_test)
    residuals = y_true - y_pred
    denom = np.where(y_true == 0, 1e-9, y_true)
    return {
        "r2_logratio": round(float(r2_score(y_true, y_pred)), 4),
        "mae_logratio": round(float(np.mean(np.abs(residuals))), 4),
        "mape_logratio": round(float(np.mean(np.abs(residuals / denom)) * 100), 2),
    }


def _eur_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mape_eur": round(mape, 2), "mae_eur": round(mae, 0), "r2_eur": round(r2, 4)}


def naive_baselines(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    train = df_train.copy()
    test = df_test.copy()
    train["_ratio"] = train["prix_vente"] / train["prix_catalogue"]
    train["_age_y"] = (train["age_months"] / 12).round().clip(lower=0, upper=15).astype(int)
    test["_age_y"] = (test["age_months"] / 12).round().clip(lower=0, upper=15).astype(int)

    global_ratio = float(train["_ratio"].median())
    brand_ratio = train.groupby("brand")["_ratio"].median().to_dict()
    group_ratio = train.groupby(["brand", "_age_y"])["_ratio"].median().to_dict()

    def _lookup(row):
        key = (row["brand"], row["_age_y"])
        if key in group_ratio and not pd.isna(group_ratio[key]):
            return group_ratio[key]
        if row["brand"] in brand_ratio and not pd.isna(brand_ratio[row["brand"]]):
            return brand_ratio[row["brand"]]
        return global_ratio

    ratios_group = test.apply(_lookup, axis=1).to_numpy()
    v0 = test["prix_catalogue"].to_numpy()
    y_true = test["prix_vente"].to_numpy()

    pred_global = np.clip(v0 * global_ratio, 500.0, None)
    pred_group = np.clip(v0 * ratios_group, 500.0, None)

    return {
        "naive_global": {
            "ratio": round(global_ratio, 4),
            **_eur_metrics(y_true, pred_global),
        },
        "naive_group": {
            "n_groups": int(len(group_ratio)),
            **_eur_metrics(y_true, pred_group),
        },
    }


def run_fold(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    hp_override: dict[str, dict] | None = None,
) -> dict:
    horizon_end = cutoff + relativedelta(months=HORIZON_MONTHS)
    df_train = df[
        (df["date de vente"] >= POST_COVID_START)
        & (df["date de vente"] < cutoff)
    ].copy()
    df_test = df[
        (df["date de vente"] >= cutoff) & (df["date de vente"] < horizon_end)
    ].copy()

    if len(df_test) == 0 or len(df_train) == 0:
        return {"cutoff": str(cutoff.date()), "skipped": True, "reason": "empty split"}

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

    num_features_ext = (
        list(vp.NUM_FEATURES) + vp.DERIVED_RATIO_FEATURES + vp.COMP_FEATURES
    )

    models_rows: dict[str, dict] = {}
    hp_override = hp_override or {}
    for name in MODELS:
        t0 = time.perf_counter()
        extra = dict(hp_override.get(name, {}))
        model = vp._TRAINERS[name](
            df_train, num_features=num_features_ext, **extra
        )
        train_sec = round(time.perf_counter() - t0, 2)
        eur = vp.evaluate_model(model, df_test, model_name=name)
        lr = _log_ratio_metrics(model, df_test)
        models_rows[name] = {
            "train_seconds": train_sec,
            "mape_eur": eur["MAPE (%)"],
            "r2_eur": eur["R²"],
            "mae_eur": eur["MAE (€)"],
            "mape_logratio": lr["mape_logratio"],
            "r2_logratio": lr["r2_logratio"],
            "mae_logratio": lr["mae_logratio"],
        }

    naive = naive_baselines(df_train, df_test)
    best_name = min(models_rows, key=lambda k: models_rows[k]["mape_eur"])
    best = models_rows[best_name]
    ng = naive["naive_group"]["mape_eur"]
    lift = round(ng - best["mape_eur"], 2) if ng is not None else float("nan")

    return {
        "cutoff": str(cutoff.date()),
        "train_start": str(POST_COVID_START.date()),
        "horizon_end": str(horizon_end.date()),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "models": models_rows,
        "best_model": best_name,
        "best_mape_eur": best["mape_eur"],
        "best_r2_eur": best["r2_eur"],
        "best_r2_logratio": best["r2_logratio"],
        "naive": naive,
        "lift_vs_naive_group_pts": lift,
        "skipped": False,
    }


def run_rolling_origin_postcovid(
    hp_override: dict[str, dict] | None = None,
    output_path: str = OUTPUT_PATH,
    tag: str = "baseline",
):
    print("=" * 72)
    device = "GPU" if vp.gpu_available() else "CPU"
    print(f"  Walk-forward post-COVID [{tag}] | train >= {POST_COVID_START.date()} | "
          f"{len(CUTOFFS)} cutoffs | horizon={HORIZON_MONTHS}m | device={device}")
    print(f"  Modèles : {MODELS}")
    if hp_override:
        for name, params in hp_override.items():
            print(f"    HP override [{name}] : {params}")
    print("=" * 72)

    um = vp.load_used_market()
    df = vp.prepare_used_market(um)
    print(f"  Dataset total : {len(df):>6,} lignes  "
          f"({df['date de vente'].min().date()} -> {df['date de vente'].max().date()})")
    print()

    folds = []
    t_start = time.perf_counter()
    for i, cutoff in enumerate(CUTOFFS, 1):
        print(f"  [fold {i}/{len(CUTOFFS)}] cutoff={cutoff.date()}...", flush=True)
        fold = run_fold(df, cutoff, hp_override=hp_override)
        if fold.get("skipped"):
            print(f"         skipped ({fold['reason']})")
        else:
            print(
                f"         n_train={fold['n_train']:>6,}  "
                f"n_test={fold['n_test']:>5,}  "
                f"best={fold['best_model']} "
                f"MAPE={fold['best_mape_eur']:.2f}%  "
                f"R²={fold['best_r2_eur']:.4f}"
            )
            for name in MODELS:
                row = fold["models"][name]
                print(
                    f"           {name:>14}  MAPE={row['mape_eur']:.2f}%  "
                    f"R²={row['r2_eur']:.4f}  ({row['train_seconds']:.1f}s)"
                )
        folds.append(fold)
    total_sec = round(time.perf_counter() - t_start, 2)

    done = [f for f in folds if not f.get("skipped")]

    # Résumés par modèle (mean / std sur les folds)
    per_model_summary = {}
    for name in MODELS:
        mapes = np.array([f["models"][name]["mape_eur"] for f in done])
        r2s = np.array([f["models"][name]["r2_eur"] for f in done])
        r2s_lr = np.array([f["models"][name]["r2_logratio"] for f in done])
        per_model_summary[name] = {
            "mape_eur_mean": round(float(mapes.mean()), 3),
            "mape_eur_std": round(float(mapes.std(ddof=0)), 3),
            "mape_eur_min": round(float(mapes.min()), 3),
            "mape_eur_max": round(float(mapes.max()), 3),
            "r2_eur_mean": round(float(r2s.mean()), 4),
            "r2_logratio_mean": round(float(r2s_lr.mean()), 4),
        }

    # Best model global (mean MAPE across folds)
    best_global = min(per_model_summary, key=lambda k: per_model_summary[k]["mape_eur_mean"])
    best_sum = per_model_summary[best_global]

    naive_group_mean = float(np.mean([f["naive"]["naive_group"]["mape_eur"] for f in done]))
    naive_global_mean = float(np.mean([f["naive"]["naive_global"]["mape_eur"] for f in done]))

    summary = {
        "per_model": per_model_summary,
        "best_model_global": best_global,
        "best_mape_mean": best_sum["mape_eur_mean"],
        "best_mape_std": best_sum["mape_eur_std"],
        "best_mape_min": best_sum["mape_eur_min"],
        "best_mape_max": best_sum["mape_eur_max"],
        "best_r2_mean": best_sum["r2_eur_mean"],
        "best_r2_logratio_mean": best_sum["r2_logratio_mean"],
        "naive_group_mape_mean": round(naive_group_mean, 3),
        "naive_global_mape_mean": round(naive_global_mean, 3),
        "lift_pts_mean": round(naive_group_mean - best_sum["mape_eur_mean"], 3),
    }

    stable = bool(
        best_sum["mape_eur_std"] < 1.0
        and best_sum["mape_eur_max"] - best_sum["mape_eur_min"] < 2.5
    )
    verdict = (
        f"Best model global : {best_global}. MAPE mean="
        f"{best_sum['mape_eur_mean']:.2f}% std={best_sum['mape_eur_std']:.2f}pt "
        f"(range [{best_sum['mape_eur_min']:.2f}, "
        f"{best_sum['mape_eur_max']:.2f}]%). "
        f"Lift vs naive_group = {summary['lift_pts_mean']:+.2f}pt."
    )
    verdict += " Modèle stable entre folds." if stable else " Variance entre folds non négligeable."

    print()
    print("  " + "-" * 68)
    print(f"  Best global : {best_global}  "
          f"MAPE mean={best_sum['mape_eur_mean']:.2f}%  "
          f"std={best_sum['mape_eur_std']:.2f}pt")
    print(f"  [verdict] {verdict}")
    print(f"  total wall-clock : {total_sec:.1f}s")

    result: dict[str, Any] = {
        "post_covid_start": str(POST_COVID_START.date()),
        "cutoffs": [str(c.date()) for c in CUTOFFS],
        "horizon_months": HORIZON_MONTHS,
        "device": device,
        "models": MODELS,
        "tag": tag,
        "hp_override": hp_override or {},
        "total_seconds": total_sec,
        "folds": folds,
        "summary": summary,
        "stable": stable,
        "verdict": verdict,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  -> {output_path}")
    print("=" * 72)
    return result


if __name__ == "__main__":
    run_rolling_origin_postcovid()
