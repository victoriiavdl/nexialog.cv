"""
Etude du régime COVID sur le split temporel principal (cutoff 2024-01).

Hypothèse : la période 2020-03 → 2022-12 (pénurie de semi-conducteurs +
rebond post-COVID) perturbe la dynamique du log_ratio. Un modèle entraîné
sur l'intégralité du train 2018+ peut sous-performer vs un modèle qui
ignore cette période atypique.

Trois variantes comparées sur le MÊME test set (>= 2024-01) :
  A. baseline_full       : train complet 2018+
  B. train_post_covid    : train restreint >= 2022-01-01
  C. covid_era_flag      : train complet + dummy 0/1 en feature

Décision gate : ne garder une variante que si
  - MAPE_eur baisse d'au moins 0.5pt ET
  - R²_log_ratio augmente d'au moins 0.02.
Sinon on conserve la baseline par parcimonie.

Résultats -> covid_regime_results.json (consommé par dashboard).
"""
import json
import time
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import vr_pipeline as vp


CUTOFF = pd.Timestamp("2024-01-01")
POST_COVID_CUTOFF = pd.Timestamp("2022-01-01")
OUTPUT_PATH = "covid_regime_results.json"
USE_GPU = True


def _eur_metrics(model, df_test) -> dict:
    y_true = df_test["prix_vente"].to_numpy()
    y_pred = vp.predict_eur(model, df_test)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mape_eur": round(mape, 2), "mae_eur": round(mae, 0), "r2_eur": round(r2, 4)}


def _lr_metrics(model, df_test) -> dict:
    y_true = df_test[vp.TARGET].to_numpy()
    y_pred = model.predict(df_test)
    residuals = y_true - y_pred
    return {
        "r2_logratio": round(float(r2_score(y_true, y_pred)), 4),
        "mape_logratio": round(
            float(np.mean(np.abs(residuals / np.where(y_true == 0, 1e-9, y_true))) * 100), 2
        ),
    }


def engineer_features(df_train, df_test, extra_num=()):
    """Feature eng fit sur train only. `extra_num` permet d'ajouter
    covid_era quand on teste la variante C."""
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
        list(vp.NUM_FEATURES) + vp.DERIVED_RATIO_FEATURES + vp.COMP_FEATURES + list(extra_num)
    )
    return df_train, df_test, num_features_ext


def variant_A_full(df_train, df_test):
    df_tr, df_te, nums = engineer_features(df_train.copy(), df_test.copy())
    t0 = time.perf_counter()
    model = vp.train_catboost(
        df_tr, num_features=nums, task_type="GPU" if USE_GPU else "CPU"
    )
    return model, df_te, nums, round(time.perf_counter() - t0, 2), len(df_tr)


def variant_B_post_covid(df_train, df_test):
    df_train = df_train[df_train["date de vente"] >= POST_COVID_CUTOFF].copy()
    df_tr, df_te, nums = engineer_features(df_train, df_test.copy())
    t0 = time.perf_counter()
    model = vp.train_catboost(
        df_tr, num_features=nums, task_type="GPU" if USE_GPU else "CPU"
    )
    return model, df_te, nums, round(time.perf_counter() - t0, 2), len(df_tr)


def variant_C_covid_flag(df_train, df_test):
    def _add_flag(df):
        df = df.copy()
        covid_start = pd.Timestamp("2020-03-01")
        covid_end = pd.Timestamp("2022-12-31")
        df["covid_era"] = (
            (df["date de vente"] >= covid_start) & (df["date de vente"] <= covid_end)
        ).astype(int)
        return df

    df_train = _add_flag(df_train)
    df_test = _add_flag(df_test)
    df_tr, df_te, nums = engineer_features(df_train, df_test, extra_num=["covid_era"])
    t0 = time.perf_counter()
    model = vp.train_catboost(
        df_tr, num_features=nums, task_type="GPU" if USE_GPU else "CPU"
    )
    return model, df_te, nums, round(time.perf_counter() - t0, 2), len(df_tr)


def run_covid_regime_study():
    print("=" * 72)
    device = "GPU" if USE_GPU else "CPU"
    print(f"  COVID regime study | cutoff={CUTOFF.date()} | CatBoost {device}")
    print("=" * 72)

    um = vp.load_used_market()
    df = vp.prepare_used_market(um)
    df_train_full = df[df["date de vente"] < CUTOFF].copy()
    df_test_full = df[df["date de vente"] >= CUTOFF].copy()
    print(f"  Train full : {len(df_train_full):>6,}  Test : {len(df_test_full):>6,}")
    covid_mask = (
        (df_train_full["date de vente"] >= pd.Timestamp("2020-03-01"))
        & (df_train_full["date de vente"] <= pd.Timestamp("2022-12-31"))
    )
    print(f"  Lignes train en zone COVID (2020-03 -> 2022-12) : "
          f"{int(covid_mask.sum()):>6,} ({100*covid_mask.mean():.1f}%)")
    print()

    variants = []

    print("  [A] baseline_full (train complet) ...", flush=True)
    model, df_te, _, dt, n_tr = variant_A_full(df_train_full, df_test_full)
    A_eur = _eur_metrics(model, df_te)
    A_lr = _lr_metrics(model, df_te)
    print(f"      n_train={n_tr:>6,}  MAPE={A_eur['mape_eur']}%  "
          f"R2={A_eur['r2_eur']}  R2_lr={A_lr['r2_logratio']}  ({dt}s)")
    variants.append({
        "name": "baseline_full",
        "label": "Train complet 2018+",
        "n_train": n_tr,
        "train_seconds": dt,
        **A_eur, **A_lr,
    })

    print("\n  [B] train_post_covid (>= 2022-01) ...", flush=True)
    model, df_te, _, dt, n_tr = variant_B_post_covid(df_train_full, df_test_full)
    B_eur = _eur_metrics(model, df_te)
    B_lr = _lr_metrics(model, df_te)
    print(f"      n_train={n_tr:>6,}  MAPE={B_eur['mape_eur']}%  "
          f"R2={B_eur['r2_eur']}  R2_lr={B_lr['r2_logratio']}  ({dt}s)")
    variants.append({
        "name": "train_post_covid",
        "label": "Train >= 2022-01 (exclut shortage)",
        "n_train": n_tr,
        "train_seconds": dt,
        **B_eur, **B_lr,
    })

    print("\n  [C] covid_era_flag ...", flush=True)
    model, df_te, _, dt, n_tr = variant_C_covid_flag(df_train_full, df_test_full)
    C_eur = _eur_metrics(model, df_te)
    C_lr = _lr_metrics(model, df_te)
    print(f"      n_train={n_tr:>6,}  MAPE={C_eur['mape_eur']}%  "
          f"R2={C_eur['r2_eur']}  R2_lr={C_lr['r2_logratio']}  ({dt}s)")
    variants.append({
        "name": "covid_era_flag",
        "label": "Train complet + feature covid_era",
        "n_train": n_tr,
        "train_seconds": dt,
        **C_eur, **C_lr,
    })

    # Décision gate
    base = variants[0]
    decision = []
    best = base
    for v in variants[1:]:
        d_mape = round(v["mape_eur"] - base["mape_eur"], 3)
        d_r2lr = round(v["r2_logratio"] - base["r2_logratio"], 4)
        keep = bool(d_mape <= -0.5 and d_r2lr >= 0.02)
        v["delta_mape_pts"] = d_mape
        v["delta_r2_logratio"] = d_r2lr
        v["keep_vs_baseline"] = keep
        decision.append(f"{v['name']}: dMAPE={d_mape:+.2f}pt dR2_lr={d_r2lr:+.4f} keep={keep}")
        if keep and v["mape_eur"] < best["mape_eur"]:
            best = v

    print("\n  " + "-" * 68)
    for d in decision:
        print(f"  [decision] {d}")
    if best["name"] == "baseline_full":
        verdict = "Aucune variante ne bat la baseline sur les deux criteres -> on garde le train complet."
    else:
        verdict = (
            f"Variante gagnante : {best['name']} "
            f"(MAPE={best['mape_eur']}%, R2_lr={best['r2_logratio']}) -> "
            f"candidate pour remplacer la baseline."
        )
    print(f"  [verdict]  {verdict}")

    result: dict[str, Any] = {
        "cutoff": str(CUTOFF.date()),
        "post_covid_cutoff": str(POST_COVID_CUTOFF.date()),
        "covid_rows_in_train": int(covid_mask.sum()),
        "covid_pct_in_train": round(float(100 * covid_mask.mean()), 2),
        "task_type": "GPU" if USE_GPU else "CPU",
        "variants": variants,
        "best_variant": best["name"],
        "verdict": verdict,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  -> {OUTPUT_PATH}")
    print("=" * 72)
    return result


if __name__ == "__main__":
    run_covid_regime_study()
