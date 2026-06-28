"""
Intervalles de prédiction par split conformal (inductive conformal regression).

Pourquoi ça compte pour le client : une prédiction de VR sans incertitude
ne permet pas de provisionner. Un intervalle calibré à 90% (IC90) qui
contient la vraie VR 9 fois sur 10 est la brique manquante pour le
risque de portefeuille — on peut alors calculer une VaR / CVaR sur les
reprises, pas juste une espérance.

Protocole (split conformal sur log_ratio) :
  1. Train temporel = sales < 2024-01. Calibration = dernier 20% chronologique
     du train (= sales les plus récentes avant 2024-01, distribution la plus
     proche du test).
  2. Fit CatBoost sur le 80% restant.
  3. Résidus absolus sur calibration : r_i = |y_i - ŷ_i| en log_ratio.
  4. Pour α ∈ {0.05, 0.10, 0.20}, quantile empirique
       q_α = quantile(r, (1-α)(n+1)/n)
     (correction de Romano & al. pour avoir E[coverage] >= 1-α).
  5. Intervalle en € : [V0·exp(ŷ - q), V0·exp(ŷ + q)]. Asymétrique en €
     (exp non-linéaire), ce qui est correct physiquement : la borne haute
     s'étale davantage.
  6. Métriques sur test : empirical coverage, width_mean_eur, width_mean_pct,
     Winkler interval score (pénalise largeur + oubli).

Garantie théorique : si (X_cal, y_cal) et (X_test, y_test) sont échangeables,
la couverture marginale >= 1-α. Hypothèse raisonnable ici car calibration
est la fenêtre juste avant le test (pas de drift fort sur 6 mois).
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
# On se cale sur le régime post-COVID validé par covid_regime_study :
# retirer 2018 -> 2021 (shortage de semi-conducteurs) ameliore drastiquement
# le fit, donc aussi la qualite des residus de calibration.
POST_COVID_CUTOFF = pd.Timestamp("2022-01-01")
CAL_FRACTION = 0.20
ALPHAS = [0.05, 0.10, 0.20]
OUTPUT_PATH = "conformal_results.json"
USE_GPU = True


def winkler_score(y, lo, hi, alpha):
    """Score d'intervalle de Winkler : width + pénalité si y hors IC.
    Plus bas = mieux."""
    width = hi - lo
    below = y < lo
    above = y > hi
    penalty = np.where(below, (2 / alpha) * (lo - y), 0.0)
    penalty += np.where(above, (2 / alpha) * (y - hi), 0.0)
    return float(np.mean(width + penalty))


def run_conformal():
    print("=" * 72)
    device = "GPU" if USE_GPU else "CPU"
    print(f"  Split conformal prediction intervals | cutoff={CUTOFF.date()} | CatBoost {device}")
    print("=" * 72)

    um = vp.load_used_market()
    df = vp.prepare_used_market(um)
    # Regime post-COVID pour eviter que le fit s'effondre sur la periode
    # shortage quand on lui retire les donnees recentes pour calibrer.
    df_train_full = (
        df[(df["date de vente"] >= POST_COVID_CUTOFF) & (df["date de vente"] < CUTOFF)]
        .copy()
        .sort_values("date de vente")
    )
    df_test = df[df["date de vente"] >= CUTOFF].copy()

    # Split chronologique train / calibration
    n_total = len(df_train_full)
    n_cal = int(n_total * CAL_FRACTION)
    df_fit = df_train_full.iloc[: n_total - n_cal].copy()
    df_cal = df_train_full.iloc[n_total - n_cal:].copy()

    print(f"  Train fit         : {len(df_fit):>6,} ({df_fit['date de vente'].min().date()} "
          f"to {df_fit['date de vente'].max().date()})")
    print(f"  Calibration       : {len(df_cal):>6,} ({df_cal['date de vente'].min().date()} "
          f"to {df_cal['date de vente'].max().date()})")
    print(f"  Test              : {len(df_test):>6,} ({df_test['date de vente'].min().date()} "
          f"to {df_test['date de vente'].max().date()})")
    print()

    # Feature engineering sur le fit only
    clustering = vp.cluster_high_cardinality_categorical(df_fit, column="model", n_clusters=6)
    df_fit = vp.apply_clustering(df_fit, clustering, new_col="model_family")
    df_cal = vp.apply_clustering(df_cal, clustering, new_col="model_family")
    df_test = vp.apply_clustering(df_test, clustering, new_col="model_family")
    df_fit = vp.add_derived_ratio_features(df_fit)
    df_cal = vp.add_derived_ratio_features(df_cal)
    df_test = vp.add_derived_ratio_features(df_test)
    comp_stats = vp.build_comparable_stats(df_fit, group_cols=("brand", "model", "fuel_type"))
    df_fit = vp.merge_comparable_features(df_fit, comp_stats)
    df_cal = vp.merge_comparable_features(df_cal, comp_stats)
    df_test = vp.merge_comparable_features(df_test, comp_stats)

    num_features_ext = (
        list(vp.NUM_FEATURES) + vp.DERIVED_RATIO_FEATURES + vp.COMP_FEATURES
    )

    # Fit CatBoost
    print("  [fit] CatBoost sur train_fit...", flush=True)
    t0 = time.perf_counter()
    model = vp.train_catboost(
        df_fit, num_features=num_features_ext, task_type="GPU" if USE_GPU else "CPU"
    )
    fit_sec = round(time.perf_counter() - t0, 2)
    print(f"        fait en {fit_sec}s")

    # Résidus absolus sur calibration (log_ratio space)
    y_cal = df_cal[vp.TARGET].to_numpy()
    y_cal_hat = model.predict(df_cal)
    resid_cal = np.abs(y_cal - y_cal_hat)
    n_cal_eff = len(resid_cal)
    print(f"  [calibration] residus |y-yhat| en log_ratio : mean={resid_cal.mean():.4f}  "
          f"q90={np.quantile(resid_cal, 0.9):.4f}  q95={np.quantile(resid_cal, 0.95):.4f}")

    # Predictions test
    y_test = df_test["prix_vente"].to_numpy()
    v0 = df_test["prix_catalogue"].to_numpy()
    yhat_lr = model.predict(df_test)
    yhat_eur = np.clip(v0 * np.exp(yhat_lr), 500.0, None)
    point_mape = float(np.mean(np.abs((y_test - yhat_eur) / y_test)) * 100)
    point_r2 = float(r2_score(y_test, yhat_eur))
    point_r2_lr = float(r2_score(df_test[vp.TARGET].to_numpy(), yhat_lr))

    print()
    print(f"  [point] MAPE={point_mape:.2f}%  R2={point_r2:.4f}  R2_lr={point_r2_lr:.4f}")
    print()
    print()

    # Conformal pour chaque alpha
    alpha_results = []
    for alpha in ALPHAS:
        # Quantile corrigé : on prend le (1-alpha)(n+1)/n quantile empirique
        q_rank = min(1.0, (1 - alpha) * (n_cal_eff + 1) / n_cal_eff)
        q = float(np.quantile(resid_cal, q_rank))
        lo_eur = np.clip(v0 * np.exp(yhat_lr - q), 500.0, None)
        hi_eur = np.clip(v0 * np.exp(yhat_lr + q), 500.0, None)
        covered = (y_test >= lo_eur) & (y_test <= hi_eur)
        coverage = float(covered.mean())
        width_eur = hi_eur - lo_eur
        width_pct = 100 * width_eur / v0
        winkler = winkler_score(y_test, lo_eur, hi_eur, alpha)

        label_target = int(round((1 - alpha) * 100))
        print(
            f"  [alpha={alpha:.2f} IC{label_target}] q={q:.4f}  "
            f"coverage={100*coverage:.2f}% (target {label_target}%)  "
            f"width_mean={width_eur.mean():.0f}EUR ({width_pct.mean():.2f}%V0)  "
            f"winkler={winkler:.0f}"
        )
        alpha_results.append({
            "alpha": alpha,
            "target_coverage_pct": label_target,
            "q_logratio": round(q, 6),
            "empirical_coverage_pct": round(100 * coverage, 2),
            "coverage_gap_pts": round(100 * coverage - label_target, 2),
            "width_mean_eur": round(float(width_eur.mean()), 0),
            "width_median_eur": round(float(np.median(width_eur)), 0),
            "width_mean_pct": round(float(width_pct.mean()), 2),
            "winkler_score": round(winkler, 0),
        })

    result: dict[str, Any] = {
        "cutoff": str(CUTOFF.date()),
        "post_covid_cutoff": str(POST_COVID_CUTOFF.date()),
        "cal_fraction": CAL_FRACTION,
        "n_fit": int(len(df_fit)),
        "n_cal": int(n_cal_eff),
        "n_test": int(len(df_test)),
        "task_type": "GPU" if USE_GPU else "CPU",
        "point_estimate": {
            "mape_eur": round(point_mape, 2),
            "r2_eur": round(point_r2, 4),
            "r2_logratio": round(point_r2_lr, 4),
        },
        "calibration_residuals": {
            "mean": round(float(resid_cal.mean()), 4),
            "q90": round(float(np.quantile(resid_cal, 0.9)), 4),
            "q95": round(float(np.quantile(resid_cal, 0.95)), 4),
        },
        "alphas": alpha_results,
    }

    # Verdict : on vise coverage_gap ~ 0 (dans [-2, +2] pts). Un gap negatif
    # = sous-couverture = IC trop étroit = drift temporel présent.
    gaps = np.array([r["coverage_gap_pts"] for r in alpha_results])
    if np.all(np.abs(gaps) <= 2.0):
        verdict = "Tous les IC sont calibres (ecart <= 2pt) -> garantie conformal tient malgre le split temporel."
    elif np.all(gaps < -2.0):
        verdict = "Sous-couverture systematique -> drift significatif entre calibration et test, envisager CQR ou recalibration glissante."
    else:
        verdict = "Couverture mixte selon alpha -> inspection par tail (queue inferieure vs superieure)."
    result["verdict"] = verdict
    print(f"\n  [verdict] {verdict}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  -> {OUTPUT_PATH}")
    print("=" * 72)
    return result


if __name__ == "__main__":
    run_conformal()
