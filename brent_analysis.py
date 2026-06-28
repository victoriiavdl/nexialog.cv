"""
Innovation — Intégration du Brent Crude Oil comme feature proxy
================================================================

Objectif : tester si l'ajout du Brent Crude (prix du pétrole, données temps-réel
disponibles jusqu'à aujourd'hui) améliore les prédictions de valeur résiduelle,
au-delà de ce que capturent déjà les features HICP.

Motivation économique :
- HICP énergie (BCE) n'est disponible que jusqu'à mars 2025.
- Le Brent (cours du pétrole) est disponible en temps réel (jusqu'à aujourd'hui).
- Corrélation Brent ↔ HICP énergie forte → Brent est un bon proxy temps-réel.
- Pour les véhicules du portefeuille vendus entre avril 2025 et aujourd'hui,
  on gagne 13 mois d'information macroéconomique réelle (vs forward-fill).

Méthodologie :
1. Collecte du Brent via yfinance (Yahoo Finance), mensualisé
2. Validation de la corrélation Brent ↔ HICP énergie
3. Ajout de Brent_YoY comme feature au pipeline
4. Comparaison MAPE / MAE / R² avec et sans Brent

Auteur : Challenge Nexialog 2026
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split

import vr_pipeline as vp

warnings.filterwarnings("ignore")

RESULTS = {}


# ===========================================================================
# 1. Collecte des données Brent Crude
# ===========================================================================

def download_brent(start="2013-01-01"):
    """Télécharge le Brent Crude Oil (BZ=F) et mensualise."""
    print("[1/5] Téléchargement des données Brent Crude (BZ=F)...")
    brent = yf.download("BZ=F", start=start, auto_adjust=True, progress=False)
    if isinstance(brent.columns, pd.MultiIndex):
        brent.columns = [c[0] for c in brent.columns]
    brent = brent[["Close"]].rename(columns={"Close": "brent_price"})
    brent.index = pd.to_datetime(brent.index)

    # Agrégation mensuelle (moyenne)
    brent_m = brent.resample("ME").mean()
    brent_m.index = brent_m.index.to_period("M")
    brent_m.index.name = "month"
    brent_m["brent_yoy"] = brent_m["brent_price"].pct_change(12) * 100

    print(f"    OK : {len(brent_m)} mois, de {brent_m.index[0]} à {brent_m.index[-1]}")
    return brent_m


# ===========================================================================
# 2. Validation de la corrélation Brent ↔ HICP énergie
# ===========================================================================

def validate_correlation(brent_m, hicp):
    """Valide la pertinence du Brent comme proxy."""
    print("\n[2/5] Validation de la corrélation Brent ↔ HICP énergie...")

    # Raw HICP (avant forward-fill) : seulement où on a des données observées
    # On charge directement depuis le CSV pour avoir la vraie dernière observation
    import os
    energy_csv = pd.read_csv("data/external/hicp_energy_yoy.csv", skiprows=[1])
    energy_csv["month"] = pd.to_datetime(energy_csv["DATE"]).dt.to_period("M")
    energy_col = [c for c in energy_csv.columns if "HICP" in c][0]
    energy_csv = energy_csv[["month", energy_col]].rename(
        columns={energy_col: "hicp_energy_yoy_raw"}
    )
    true_last = energy_csv["month"].max()
    print(f"    Dernière observation HICP énergie (non forward-fillée) : {true_last}")

    df = brent_m.reset_index().merge(energy_csv, on="month", how="inner")
    df = df.dropna()

    corr_pearson = df["brent_yoy"].corr(df["hicp_energy_yoy_raw"])
    corr_spearman = df["brent_yoy"].corr(df["hicp_energy_yoy_raw"], method="spearman")

    # Lag analysis
    lag_results = []
    for lag in range(-3, 4):
        d = df.copy()
        d["brent_lag"] = d["brent_yoy"].shift(lag)
        d = d.dropna()
        c = d["brent_lag"].corr(d["hicp_energy_yoy_raw"])
        lag_results.append({"lag_months": lag, "correlation": round(float(c), 3)})
    best = max(lag_results, key=lambda x: abs(x["correlation"]))

    print(f"    Corrélation Pearson  : {corr_pearson:.3f}")
    print(f"    Corrélation Spearman : {corr_spearman:.3f}")
    print(f"    Meilleur lag         : {best['lag_months']} mois (r = {best['correlation']:.3f})")
    print(f"    Nombre d'observations : {len(df)}")

    RESULTS["correlation"] = {
        "pearson": round(float(corr_pearson), 3),
        "spearman": round(float(corr_spearman), 3),
        "best_lag_months": int(best["lag_months"]),
        "best_lag_corr": float(best["correlation"]),
        "lag_analysis": lag_results,
        "n_obs": int(len(df)),
        "hicp_last_observed": str(true_last),
    }
    return df, true_last


# ===========================================================================
# 3. Préparation des features enrichies
# ===========================================================================

def prepare_enriched(um, brent_m):
    """Prépare used_market avec Brent_YoY comme feature supplémentaire."""
    df = vp.prepare_used_market(um)
    df["sale_month"] = pd.to_datetime(df["date de vente"]).dt.to_period("M")

    brent_lookup = brent_m["brent_yoy"].to_dict()
    df["brent_yoy"] = df["sale_month"].map(brent_lookup)
    # Fallback si mois manquant
    df["brent_yoy"] = df["brent_yoy"].fillna(df["brent_yoy"].median())
    df = df.drop(columns=["sale_month"])
    return df


def prepare_portfolio_enriched(po, brent_m):
    """Prépare portfolio avec Brent_YoY (pour chaque date de fin de contrat)."""
    df = vp.prepare_portfolio(po)
    df["sale_month"] = pd.to_datetime(df["current_contract_planned_end_date"]).dt.to_period("M")

    # Pour les dates postérieures à la dernière observation Brent, forward-fill
    last_brent = brent_m.index[brent_m["brent_yoy"].notna()].max()
    last_value = brent_m.loc[last_brent, "brent_yoy"]

    brent_lookup = brent_m["brent_yoy"].to_dict()
    df["brent_yoy"] = df["sale_month"].map(brent_lookup).fillna(last_value)
    df = df.drop(columns=["sale_month"])
    return df


# ===========================================================================
# 4. Impact sur used_market : MAPE avec / sans Brent
# ===========================================================================

def impact_on_used_market(um, brent_m):
    """Compare MAPE des 3 modèles AVEC vs SANS la feature Brent."""
    print("\n[3/5] Impact sur used_market — baseline (sans Brent)...")

    # --- Baseline : pipeline actuel ---
    comp_base = vp.compare_models_on_used_market(um, n_model_families=6, tune=False)
    baseline_rows = comp_base.metrics.to_dict("records")

    print("    Résultats baseline :")
    for r in baseline_rows:
        print(f"      {r['model']:10s} | MAPE {r['MAPE (%)']}% | MAE {r['MAE (€)']} EUR | R² {r['R²']}")

    # --- Avec Brent : recréer split + entraîner les 3 modèles ---
    print("\n[4/5] Impact sur used_market — avec Brent en feature...")
    df_ext = prepare_enriched(um, brent_m)

    bins = pd.qcut(df_ext["prix_vente"], q=10, labels=False, duplicates="drop")
    df_train, df_test = train_test_split(
        df_ext, test_size=0.2, random_state=42, stratify=bins
    )
    clustering = vp.cluster_high_cardinality_categorical(
        df_train, column="model", n_clusters=6
    )
    df_train = vp.apply_clustering(df_train, clustering, new_col="model_family")
    df_test = vp.apply_clustering(df_test, clustering, new_col="model_family")

    num_features_ext = list(vp.NUM_FEATURES) + ["brent_yoy"]
    with_brent = {}

    for name, trainer in [("ridge", vp.train_ridge),
                          ("xgboost", vp.train_xgboost),
                          ("catboost", vp.train_catboost)]:
        m = trainer(df_train, cat_features=vp.CAT_FEATURES, num_features=num_features_ext)
        metrics = vp.evaluate_model(m, df_test, model_name=name)
        with_brent[name] = metrics

    print("    Résultats avec Brent :")
    for name, m in with_brent.items():
        print(f"      {name:10s} | MAPE {m['MAPE (%)']}% | MAE {m['MAE (€)']} EUR | R² {m['R²']}")

    # --- Synthèse ---
    print("\n    === Synthèse Δ MAPE (pp, négatif = amélioration) ===")
    rows_summary = []
    for name in ["ridge", "xgboost", "catboost"]:
        base = next(r for r in baseline_rows if r["model"] == name)
        ext = with_brent[name]
        d_mape = float(ext["MAPE (%)"]) - float(base["MAPE (%)"])
        d_mae = float(ext["MAE (€)"]) - float(base["MAE (€)"])
        d_r2 = float(ext["R²"]) - float(base["R²"])
        print(f"      {name:10s} | Δ MAPE {d_mape:+.2f} pp | Δ MAE {d_mae:+.0f} EUR | Δ R² {d_r2:+.4f}")
        rows_summary.append({
            "model": name,
            "baseline_mape": float(base["MAPE (%)"]),
            "baseline_mae": float(base["MAE (€)"]),
            "baseline_r2": float(base["R²"]),
            "with_brent_mape": float(ext["MAPE (%)"]),
            "with_brent_mae": float(ext["MAE (€)"]),
            "with_brent_r2": float(ext["R²"]),
            "delta_mape_pp": round(d_mape, 3),
            "delta_mae_eur": round(d_mae, 0),
            "delta_r2": round(d_r2, 4),
        })

    RESULTS["used_market_impact"] = rows_summary
    return rows_summary


# ===========================================================================
# 5. Impact sur le portfolio
# ===========================================================================

def impact_on_portfolio(um, po, brent_m):
    """
    Compare les prédictions portfolio :
    - baseline : pipeline actuel (HICP forward-fill)
    - enrichi  : pipeline avec Brent ajouté en feature
    """
    print("\n[5/5] Impact sur le portfolio...")

    # --- Baseline ---
    pipeline = vp.fit_final_pipeline(um, "catboost", n_model_families=6)
    preds_base = vp.predict_portfolio(pipeline, po)
    median_base = float(np.median(preds_base["prediction"]))
    mean_base = float(np.mean(preds_base["prediction"]))

    # --- Enrichi : réentraînement sur 100% de used_market avec Brent ---
    df_ext = prepare_enriched(um, brent_m)
    clustering = vp.cluster_high_cardinality_categorical(
        df_ext, column="model", n_clusters=6
    )
    df_ext = vp.apply_clustering(df_ext, clustering, new_col="model_family")
    num_features_ext = list(vp.NUM_FEATURES) + ["brent_yoy"]
    model_ext = vp.train_catboost(
        df_ext, cat_features=vp.CAT_FEATURES, num_features=num_features_ext
    )

    # Prédiction
    df_po_ext = prepare_portfolio_enriched(po, brent_m)
    df_po_ext = vp.apply_clustering(df_po_ext, clustering, new_col="model_family")
    preds_ext = vp.predict_eur(model_ext, df_po_ext)
    median_ext = float(np.median(preds_ext))
    mean_ext = float(np.mean(preds_ext))

    delta_median = median_ext - median_base
    delta_mean = mean_ext - mean_base

    print(f"    Prédiction médiane baseline : {median_base:.0f} EUR")
    print(f"    Prédiction médiane enrichie : {median_ext:.0f} EUR")
    print(f"    Δ médiane : {delta_median:+.0f} EUR ({100*delta_median/median_base:+.2f}%)")
    print(f"    Prédiction moyenne baseline : {mean_base:.0f} EUR")
    print(f"    Prédiction moyenne enrichie : {mean_ext:.0f} EUR")

    RESULTS["portfolio_impact"] = {
        "baseline_median": round(median_base, 0),
        "enriched_median": round(median_ext, 0),
        "delta_median": round(delta_median, 0),
        "delta_median_pct": round(100 * delta_median / median_base, 2),
        "baseline_mean": round(mean_base, 0),
        "enriched_mean": round(mean_ext, 0),
        "delta_mean": round(delta_mean, 0),
    }

    # Sauvegarde les prédictions enrichies
    out = pd.DataFrame({"id": po["id"] if "id" in po.columns else range(len(po)),
                        "prediction_baseline": preds_base["prediction"].values,
                        "prediction_enriched": preds_ext})
    out.to_csv("data/external/portfolio_comparison_brent.csv", index=False)
    print("    Prédictions comparées sauvegardées : data/external/portfolio_comparison_brent.csv")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("  INNOVATION — Brent Crude Oil en feature proxy temps-réel")
    print("=" * 70)

    # 1. Données
    brent_m = download_brent()

    # 2. HICP
    hicp = vp.load_hicp()

    # 3. Validation
    df_joint, true_last = validate_correlation(brent_m, hicp)

    # 4. Charger datasets
    um = vp.load_used_market()
    po = vp.load_portfolio()

    # 5. Impact used_market
    impact_on_used_market(um, brent_m)

    # 6. Impact portfolio
    impact_on_portfolio(um, po, brent_m)

    # Export
    with open("brent_analysis_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    brent_m.reset_index().to_csv("data/external/brent_monthly.csv", index=False)

    print("\n" + "=" * 70)
    print("  Résultats : brent_analysis_results.json")
    print("  Données : data/external/brent_monthly.csv")
    print("=" * 70)

    return RESULTS


if __name__ == "__main__":
    main()
