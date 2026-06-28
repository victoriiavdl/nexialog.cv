"""
Nexialog VR Prediction - Interactive Web Dashboard
===================================================
Usage:
    poetry run python app.py
    Open http://localhost:5000
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import urllib.request
import urllib.error
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

warnings.filterwarnings("ignore")

import vr_pipeline as vp

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
STATE = {"charts": {}, "meta": {}, "pipeline": None, "valid_options": {}}
CACHE_DIR = ".web_cache"

# Bump when the shape of the cached payload changes so old caches are
# invalidated instead of silently served back.
CACHE_SCHEMA_VERSION = 30  # risk_portfolio = snapshot v2.0 (prediction_portfolio.csv, 20.27 M€)


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            if np.isnan(o) or np.isinf(o):
                return None
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        return super().default(o)


def _clean(obj):
    """Recursively replace NaN / Inf with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# Data loading & processing
# ---------------------------------------------------------------------------

def _logratio_target_stats(df_test: pd.DataFrame) -> dict:
    y = df_test[vp.TARGET].to_numpy()
    return {
        "mean": round(float(y.mean()), 4),
        "std": round(float(y.std()), 4),
        "min": round(float(y.min()), 4),
        "max": round(float(y.max()), 4),
        "n": int(len(y)),
    }


def init_data():
    """Load data, train models, pre-compute all chart data."""
    global STATE

    cache_charts = os.path.join(CACHE_DIR, "charts.json")
    cache_model = os.path.join(CACHE_DIR, "pipeline.pkl")

    if os.path.exists(cache_charts) and os.path.exists(cache_model):
        with open(cache_charts) as f:
            cached = json.load(f)
        if cached.get("schema_version") == CACHE_SCHEMA_VERSION:
            print("[cache] Chargement depuis le cache...")
            STATE["charts"] = cached["charts"]
            STATE["meta"] = cached["meta"]
            STATE["valid_options"] = cached["valid_options"]
            with open(cache_model, "rb") as f:
                STATE["pipeline"] = pickle.load(f)
            print("[OK] Dashboard pret : http://localhost:5000")
            return
        print(
            f"[cache] Schema obsolete "
            f"(cached={cached.get('schema_version')}, attendu={CACHE_SCHEMA_VERSION})"
            " — recalcul complet."
        )

    print("=" * 60, flush=True)
    print("  NEXIALOG VR DASHBOARD - Initialisation", flush=True)
    print("=" * 60, flush=True)

    device = "GPU" if vp.gpu_available() else "CPU"
    print(f"\n[1/6] Chargement des donnees ({device})...", flush=True)
    um = vp.load_used_market()
    po = vp.load_portfolio()

    print("[2/6] Preparation used_market...", flush=True)
    df = vp.prepare_used_market(um)

    print(
        f"[3/6] Comparaison des modeles (split post-COVID, "
        f"{len(vp.DEFAULT_MODELS)} modeles + naif)...",
        flush=True,
    )
    # Charge les HP tunés persistés et sélectionne le chosen_model par
    # stabilité (min cv_mape_std_tuned). Ces HP servent à la fois pour
    # le split canonique (pour que feature_importance / pred_vs_obs /
    # résidus correspondent au modèle réellement retenu) et pour le
    # ré-entraînement final (fit_final_pipeline).
    tuning = vp.load_tuned_params()
    try:
        chosen_name, chosen_params = vp.pick_chosen_model(tuning)
        chosen_override = (chosen_name, chosen_params)
        print(
            f"       [tuned] chosen={chosen_name} "
            f"(std={tuning[chosen_name].get('cv_mape_std_tuned')}, "
            f"mape_cv={tuning[chosen_name].get('cv_mape_mean_tuned')})",
            flush=True,
        )
    except RuntimeError as exc:
        print(f"       [warn] {exc} — fallback sur MAPE-on-split", flush=True)
        chosen_name, chosen_params = None, {}
        chosen_override = None

    comparison = vp.compare_models_on_used_market(
        um, chosen_override=chosen_override
    )

    mdl_only_eur = comparison.metrics[comparison.metrics["model"] != "naive"]
    mdl_only_lr = comparison.metrics_logratio[
        comparison.metrics_logratio["model"] != "naive"
    ]
    chosen_eur_row = mdl_only_eur[mdl_only_eur["model"] == comparison.best_name]
    best_eur = chosen_eur_row.iloc[0] if not chosen_eur_row.empty else mdl_only_eur.iloc[0]
    best_lr_row = mdl_only_lr[mdl_only_lr["model"] == comparison.best_name]
    best_lr = best_lr_row.iloc[0] if not best_lr_row.empty else mdl_only_lr.iloc[0]
    naive = comparison.naive

    print(f"[4/6] Modele retenu (stabilite) : {comparison.best_name}", flush=True)
    print(
        f"       euros     -> MAPE = {best_eur['MAPE (%)']}%, "
        f"R2 = {best_eur['R²']}",
        flush=True,
    )
    print(
        f"       log_ratio -> MAPE = {best_lr['MAPE (log_ratio) %']}%, "
        f"R2 = {best_lr['R² (log_ratio)']}",
        flush=True,
    )
    print(
        f"       naive (k_MSE={naive['k_mse']}) -> "
        f"euros MAPE={naive['eur']['MAPE (%)']}% R2={naive['eur']['R²']}",
        flush=True,
    )

    print("[5/6] Re-entrainement final sur fenetre post-COVID...", flush=True)
    pipeline = vp.fit_final_pipeline(
        um, comparison.best_name, params=chosen_params or None
    )
    STATE["pipeline"] = pipeline

    print(f"[6/6] Prediction portfolio ({len(po)} vehicules)...", flush=True)
    preds = vp.predict_portfolio(pipeline, po)

    print("      Stress test temporel 49/51 post-COVID...", flush=True)
    try:
        stress = vp.stress_test_temporal_4951_split(
            um, tuned_params=tuning
        )
    except Exception as exc:
        print(f"      [warn] stress test : {exc}", flush=True)
        stress = None

    lr_target_stats = _logratio_target_stats(comparison.df_test)

    print("      Generation des visualisations...", flush=True)
    STATE["charts"] = _clean(
        _generate_all_charts(df, comparison, preds, stress=stress)
    )
    STATE["meta"] = _clean(
        _generate_meta(df, comparison, preds, lr_target_stats)
    )
    STATE["valid_options"] = _extract_options(df, preds)

    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_charts, "w") as f:
        json.dump(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "charts": STATE["charts"],
                "meta": STATE["meta"],
                "valid_options": STATE["valid_options"],
            },
            f,
            cls=_Enc,
        )
    with open(cache_model, "wb") as f:
        pickle.dump(pipeline, f)

    print("\n" + "=" * 60, flush=True)
    print("  Dashboard pret : http://localhost:5000", flush=True)
    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Meta information
# ---------------------------------------------------------------------------

def _walk_forward_summary(chosen_name: str) -> dict | None:
    """Agrège les métriques walk-forward post-COVID (3 folds × 6 mois) pour
    le modèle retenu. Source : rolling_origin_postcovid_results.json.

    Utilisé pour afficher les métriques canoniques (mean sur folds) plutôt
    que les métriques single-split long-horizon — plus représentatives de
    la performance production du modèle tuné.
    """
    path = "rolling_origin_postcovid_results.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            ropc = json.load(f)
    except Exception as e:
        print(f"  [warn] walk-forward summary load: {e}")
        return None

    per_model = ropc.get("summary", {}).get("per_model", {})
    pm = per_model.get(chosen_name)
    if not pm:
        return None

    maes, mape_lrs, mae_lrs = [], [], []
    for fold in ropc.get("folds", []):
        if fold.get("skipped"):
            continue
        m = fold.get("models", {}).get(chosen_name, {})
        if m.get("mae_eur") is not None:
            maes.append(float(m["mae_eur"]))
        if m.get("mape_logratio") is not None:
            mape_lrs.append(float(m["mape_logratio"]))
        if m.get("mae_logratio") is not None:
            mae_lrs.append(float(m["mae_logratio"]))

    n_folds = sum(1 for f in ropc.get("folds", []) if not f.get("skipped"))
    return {
        "model": chosen_name,
        "n_folds": n_folds,
        "horizon_months": ropc.get("horizon_months"),
        "cutoffs": ropc.get("cutoffs", []),
        "mape_eur_mean": pm.get("mape_eur_mean"),
        "mape_eur_std": pm.get("mape_eur_std"),
        "mape_eur_min": pm.get("mape_eur_min"),
        "mape_eur_max": pm.get("mape_eur_max"),
        "r2_eur_mean": pm.get("r2_eur_mean"),
        "r2_logratio_mean": pm.get("r2_logratio_mean"),
        "mae_eur_mean": round(float(np.mean(maes)), 0) if maes else None,
        "mape_logratio_mean": round(float(np.mean(mape_lrs)), 2) if mape_lrs else None,
        "mae_logratio_mean": round(float(np.mean(mae_lrs)), 4) if mae_lrs else None,
    }


def _generate_meta(df, comparison, preds, lr_target_stats):
    mdl_metrics = comparison.metrics[comparison.metrics["model"] != "naive"]
    # Métriques single-split (fallback + debug)
    best_row = mdl_metrics.loc[mdl_metrics["model"] == comparison.best_name]
    best = best_row.iloc[0] if not best_row.empty else mdl_metrics.sort_values("MAPE (%)").iloc[0]
    lr_metrics = comparison.metrics_logratio
    best_lr_row = lr_metrics.loc[lr_metrics["model"] == comparison.best_name]
    best_lr = (
        best_lr_row.iloc[0]
        if not best_lr_row.empty
        else lr_metrics[lr_metrics["model"] != "naive"].iloc[0]
    )

    # Walk-forward post-COVID (3 folds × 6 mois) — MÉTRIQUES CANONIQUES du
    # modèle retenu (XGBoost tuné). Plus représentatives que le single-split
    # long-horizon pour quantifier la performance production.
    wf = _walk_forward_summary(comparison.best_name)

    if wf is not None:
        best_mape = round(float(wf["mape_eur_mean"]), 2)
        best_r2 = round(float(wf["r2_eur_mean"]), 4)
        best_mae = round(float(wf["mae_eur_mean"]), 0) if wf["mae_eur_mean"] is not None else round(float(best["MAE (€)"]), 0)
        best_mape_lr = round(float(wf["mape_logratio_mean"]), 2) if wf["mape_logratio_mean"] is not None else round(float(best_lr["MAPE (log_ratio) %"]), 2)
        best_mae_lr = round(float(wf["mae_logratio_mean"]), 4) if wf["mae_logratio_mean"] is not None else round(float(best_lr["MAE (log_ratio)"]), 4)
        best_r2_lr = round(float(wf["r2_logratio_mean"]), 4)
        metrics_source = "walk_forward_postcovid"
    else:
        best_mape = round(float(best["MAPE (%)"]), 2)
        best_r2 = round(float(best["R²"]), 4)
        best_mae = round(float(best["MAE (€)"]), 0)
        best_mape_lr = round(float(best_lr["MAPE (log_ratio) %"]), 2)
        best_mae_lr = round(float(best_lr["MAE (log_ratio)"]), 4)
        best_r2_lr = round(float(best_lr["R² (log_ratio)"]), 4)
        metrics_source = "single_split_postcovid"

    naive_payload = {
        "k_mse": comparison.naive["k_mse"],
        "log_k": comparison.naive["log_k"],
        "formula": comparison.naive["formula"],
        "eur": comparison.naive["eur"],
        "log_ratio": comparison.naive["log_ratio"],
    }
    return {
        "n_transactions": int(len(df)),
        "n_portfolio": int(len(preds)),
        "n_brands": int(df["brand"].nunique()),
        "n_models_vehicle": int(df["model"].nunique()) if "model" in df.columns else 0,
        "best_model": comparison.best_name,
        "best_mape": best_mape,
        "best_mae": best_mae,
        "best_r2": best_r2,
        "best_mape_logratio": best_mape_lr,
        "best_mae_logratio": best_mae_lr,
        "best_rmse_logratio": round(float(best_lr["RMSE (log_ratio)"]), 4),
        "best_r2_logratio": best_r2_lr,
        "metrics_source": metrics_source,
        "walk_forward": wf,
        "best_mape_single_split": round(float(best["MAPE (%)"]), 2),
        "best_r2_single_split": round(float(best["R²"]), 4),
        "metrics": comparison.metrics.to_dict("records"),
        "metrics_logratio": comparison.metrics_logratio.to_dict("records"),
        "logratio_target_stats": lr_target_stats,
        "naive_baseline": naive_payload,
        "comparison_meta": comparison.meta,
        "avg_prediction": round(float(preds["prediction"].mean()), 0),
        "median_prediction": round(float(preds["prediction"].median()), 0),
        "avg_decote": round(float(preds["decote_pct"].mean()), 1),
        "price_range": [
            round(float(preds["prediction"].min()), 0),
            round(float(preds["prediction"].max()), 0),
        ],
        "age_range": [
            int(df["age_months"].min()),
            int(df["age_months"].max()),
        ],
    }


def _extract_options(df, preds):
    """Values for the simulator dropdowns."""
    # Models by brand
    models_by_brand = {}
    if "brand" in df.columns and "model" in df.columns:
        for brand in sorted(df["brand"].dropna().unique()):
            models = sorted(df[df["brand"] == brand]["model"].dropna().unique().tolist())
            models_by_brand[brand] = models

    return {
        "brands": sorted(df["brand"].dropna().unique().tolist()),
        "fuel_types": sorted(df["fuel_type"].dropna().unique().tolist()),
        "range_types": sorted(df["range_type"].dropna().unique().tolist()),
        "models_by_brand": models_by_brand,
        "model_families": sorted(
            preds["model_family"].dropna().unique().tolist()
            if "model_family" in preds.columns
            else []
        ),
    }


# ---------------------------------------------------------------------------
# Chart data generators
# ---------------------------------------------------------------------------

def _hist(values, n_bins=50):
    """Return binned histogram data."""
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    h, edges = np.histogram(vals, bins=n_bins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {
        "x": centers,
        "y": h.tolist(),
        "mean": round(float(np.mean(vals)), 2),
        "median": round(float(np.median(vals)), 2),
        "std": round(float(np.std(vals)), 2),
        "min": round(float(np.min(vals)), 2),
        "max": round(float(np.max(vals)), 2),
        "n": int(len(vals)),
    }


def _generate_all_charts(df, comparison, preds, stress=None):
    charts = {}

    # ==== 1. EDA: Numeric distributions ====
    charts["dist_age"] = _hist(df["age_months"])
    charts["dist_mileage"] = _hist(df["mileage_km"])
    charts["dist_price"] = _hist(df["prix_vente"])
    charts["dist_catalogue"] = _hist(df["prix_catalogue"])
    charts["dist_log_ratio"] = _hist(df["log_ratio"], n_bins=60)

    # ==== 2. EDA: Categorical distributions ====
    for col in ["brand", "fuel_type", "range_type"]:
        if col in df.columns:
            vc = df[col].value_counts().head(15)
            charts[f"cat_{col}"] = {
                "labels": vc.index.tolist(),
                "values": vc.values.tolist(),
            }

    # model distribution (top 15)
    if "model" in df.columns:
        vc_model = df["model"].value_counts().head(15)
        charts["cat_model"] = {
            "labels": vc_model.index.tolist(),
            "values": vc_model.values.tolist(),
        }

    # ==== 3. Correlation matrix ====
    num_cols = [
        "age_months", "mileage_km", "prix_vente", "prix_catalogue",
        "log_age", "log_mileage", "log_catalogue", "log_ratio",
        "log_cum_inflation_core", "hicp_headline_yoy_sale",
        "hicp_energy_yoy_sale",
    ]
    avail = [c for c in num_cols if c in df.columns]
    corr = df[avail].corr()
    # Short labels for display
    label_map = {
        "age_months": "Age", "mileage_km": "Km", "prix_vente": "Prix vente",
        "prix_catalogue": "Prix cat.", "log_age": "log(age)",
        "log_mileage": "log(km)", "log_catalogue": "log(cat.)",
        "log_ratio": "log_ratio", "log_cum_inflation_core": "Infl. cum.",
        "hicp_headline_yoy_sale": "HICP headline",
        "hicp_energy_yoy_sale": "HICP energie",
    }
    charts["correlation"] = {
        "labels": [label_map.get(c, c) for c in avail],
        "values": [[round(float(v), 3) for v in row] for row in corr.values],
    }

    # ==== 4. Depreciation by age ====
    # Bin age into 6-month intervals for smoother curve
    df_age = df.copy()
    df_age["age_bin"] = (df_age["age_months"] / 6).round() * 6
    dep_data = []
    for a in sorted(df_age["age_bin"].unique()):
        sub = df_age[df_age["age_bin"] == a]["prix_vente"]
        if len(sub) >= 30:
            dep_data.append({
                "age": int(a),
                "median": float(sub.median()),
                "q25": float(sub.quantile(0.25)),
                "q75": float(sub.quantile(0.75)),
                "n": int(len(sub)),
            })
    charts["depreciation_age"] = dep_data

    # Depreciation: median price by age (binned)
    df_dep = df.copy()
    df_dep["age_bin"] = (df_dep["age_months"] / 12).round().astype(int)
    dep_by_year = df_dep.groupby("age_bin")["prix_vente"].agg(
        ["median", "mean", "count"]
    ).reset_index()
    dep_by_year = dep_by_year[dep_by_year["count"] >= 50]
    charts["depreciation_by_year"] = {
        "x": dep_by_year["age_bin"].tolist(),
        "median": dep_by_year["median"].round(0).tolist(),
        "mean": dep_by_year["mean"].round(0).tolist(),
    }

    # ==== 5. Depreciation by fuel type ====
    fuel_dep = []
    for ft in df["fuel_type"].unique():
        sub = df[df["fuel_type"] == ft].copy()
        sub["age_year"] = (sub["age_months"] / 12).round().astype(int)
        grp = sub.groupby("age_year")["prix_vente"].median().reset_index()
        grp = grp[grp["age_year"].between(1, 8)]
        if len(grp) >= 3:
            fuel_dep.append({
                "fuel": str(ft),
                "x": grp["age_year"].tolist(),
                "y": grp["prix_vente"].round(0).tolist(),
            })
    charts["depreciation_fuel"] = fuel_dep

    # ==== 6. Price by brand (boxplot data) ====
    brand_stats = []
    for brand in df["brand"].unique():
        sub = df[df["brand"] == brand]["prix_vente"]
        if len(sub) >= 20:
            brand_stats.append({
                "brand": str(brand),
                "median": float(sub.median()),
                "q25": float(sub.quantile(0.25)),
                "q75": float(sub.quantile(0.75)),
                "min": float(sub.quantile(0.05)),
                "max": float(sub.quantile(0.95)),
                "n": int(len(sub)),
            })
    brand_stats.sort(key=lambda x: -x["median"])
    charts["price_by_brand"] = brand_stats

    # ==== 7. HICP time series ====
    try:
        hicp = vp.load_hicp()
        # Filter to 2018-2026 range (used_market + portfolio period)
        hicp_period = hicp.loc["2018-01":"2025-03"]
        charts["hicp_headline"] = {
            "dates": [str(p) for p in hicp_period.index],
            "values": hicp_period["hicp_headline_yoy"].tolist(),
        }
        charts["hicp_core"] = {
            "dates": [str(p) for p in hicp_period.index],
            "values": hicp_period["hicp_core_index"].tolist(),
        }
        charts["hicp_energy"] = {
            "dates": [str(p) for p in hicp_period.index],
            "values": hicp_period["hicp_energy_yoy"].tolist(),
        }
    except Exception:
        pass

    # ==== 8. HICP features vs log_ratio ====
    for feat in ["log_cum_inflation_core", "hicp_headline_yoy_sale", "hicp_energy_yoy_sale"]:
        if feat in df.columns:
            sub = df[[feat, "log_ratio"]].dropna()
            # Bin into 20 quantiles
            try:
                sub["bin"] = pd.qcut(sub[feat], q=20, duplicates="drop")
                agg = sub.groupby("bin", observed=True).agg(
                    x=(feat, "mean"),
                    y=("log_ratio", "mean"),
                    n=("log_ratio", "count"),
                ).reset_index(drop=True)
                corr_val = float(sub[feat].corr(sub["log_ratio"]))
                spearman_val = float(sub[feat].corr(sub["log_ratio"], method="spearman"))
                charts[f"hicp_vs_logratio_{feat}"] = {
                    "x": agg["x"].round(4).tolist(),
                    "y": agg["y"].round(4).tolist(),
                    "n": agg["n"].tolist(),
                    "corr": round(corr_val, 3),
                    "spearman": round(spearman_val, 3),
                }
            except Exception:
                pass

    # ==== 9. Model comparison (split temporel post-COVID, canonique) ====
    # Les artefacts proviennent directement du ModelComparison du pipeline :
    # métriques euros + log_ratio par modèle + baseline naïf, meta sur le
    # split (train_range, test_range, n_train, n_test, best_model).
    metrics = comparison.metrics
    lr_by_model = comparison.metrics_logratio.set_index("model")

    def _lr(m, col, nd):
        if m in lr_by_model.index:
            return round(float(lr_by_model.loc[m, col]), nd)
        return None

    charts["model_comparison"] = {
        "models": metrics["model"].tolist(),
        "mae": [round(float(v), 0) for v in metrics["MAE (€)"]],
        "mape": [round(float(v), 2) for v in metrics["MAPE (%)"]],
        "r2": [round(float(v), 4) for v in metrics["R²"]],
        "r2_logratio": [_lr(m, "R² (log_ratio)", 4) for m in metrics["model"]],
        "mape_logratio": [_lr(m, "MAPE (log_ratio) %", 2) for m in metrics["model"]],
        "mae_logratio": [_lr(m, "MAE (log_ratio)", 4) for m in metrics["model"]],
        "rmse_logratio": [_lr(m, "RMSE (log_ratio)", 4) for m in metrics["model"]],
    }
    charts["model_comparison_meta"] = comparison.meta

    # ==== 10-11. Feature importance + pred_vs_obs + residuals ====
    # Tous produits par le ModelComparison (sur le best model, test post-COVID).
    if comparison.feature_importance:
        charts["feature_importance"] = comparison.feature_importance
    charts["pred_vs_obs"] = comparison.pred_vs_obs
    charts["residuals"] = comparison.residuals

    # ==== 12. Portfolio predictions ====
    charts["portfolio_predictions"] = _hist(preds["prediction"], n_bins=40)
    charts["portfolio_decote"] = _hist(preds["decote_pct"], n_bins=40)

    # Portfolio by brand
    if "brand" in preds.columns:
        grp = preds.groupby("brand")["prediction"].agg(["mean", "median", "count"])
        grp = grp.sort_values("median", ascending=False).reset_index()
        charts["portfolio_by_brand"] = {
            "brands": grp["brand"].tolist(),
            "median": grp["median"].round(0).tolist(),
            "mean": grp["mean"].round(0).tolist(),
            "count": grp["count"].tolist(),
        }

    # Portfolio by fuel type
    if "fuel_type" in preds.columns:
        grp = preds.groupby("fuel_type").agg(
            median_pred=("prediction", "median"),
            median_decote=("decote_pct", "median"),
            count=("prediction", "count"),
        ).sort_values("median_decote").reset_index()
        charts["portfolio_by_fuel"] = {
            "fuels": grp["fuel_type"].tolist(),
            "median_pred": grp["median_pred"].round(0).tolist(),
            "median_decote": grp["median_decote"].round(1).tolist(),
            "count": grp["count"].tolist(),
        }

    # Portfolio prediction vs age
    if "age_months" in preds.columns:
        preds_copy = preds.copy()
        preds_copy["age_year"] = (preds_copy["age_months"] / 12).round(1)
        scatter_data = preds_copy[["age_year", "prediction", "brand"]].dropna()
        if len(scatter_data) > 2000:
            scatter_data = scatter_data.sample(2000, random_state=42)
        charts["portfolio_pred_vs_age"] = {
            "x": scatter_data["age_year"].tolist(),
            "y": scatter_data["prediction"].round(0).tolist(),
            "brand": scatter_data["brand"].tolist(),
        }

    # ==== 13. Depreciation curves by family (for portfolio) ====
    if "model_family" in preds.columns and "decote_pct" in preds.columns:
        family_stats = []
        for fam in sorted(preds["model_family"].unique()):
            sub = preds[preds["model_family"] == fam]
            family_stats.append({
                "family": str(fam),
                "median_decote": round(float(sub["decote_pct"].median()), 1),
                "median_pred": round(float(sub["prediction"].median()), 0),
                "count": int(len(sub)),
                "models": ", ".join(
                    sub["model"].value_counts().head(3).index.tolist()
                ) if "model" in sub.columns else "",
            })
        charts["family_stats"] = family_stats

    # ==== 14. Log ratio vs HICP energy x fuel_type (interaction) ====
    if "hicp_energy_yoy_sale" in df.columns and "fuel_type" in df.columns:
        df_int = df.copy()
        df_int["energy_regime"] = pd.cut(
            df_int["hicp_energy_yoy_sale"],
            bins=[-100, 0, 10, 100],
            labels=["Negative (<0%)", "Moderate (0-10%)", "Forte (>10%)"],
        )
        interaction = df_int.groupby(
            ["fuel_type", "energy_regime"], observed=True
        )["log_ratio"].mean().reset_index()
        charts["energy_fuel_interaction"] = {
            "data": [
                {
                    "fuel": str(row["fuel_type"]),
                    "regime": str(row["energy_regime"]),
                    "log_ratio": round(float(row["log_ratio"]), 4),
                }
                for _, row in interaction.iterrows()
            ]
        }

    # ==== 15. Cluster evaluation (elbow + silhouette) ====
    try:
        cluster_eval = vp.evaluate_cluster_range(
            df, column="model", k_values=range(2, 11)
        )
        m = cluster_eval["metrics"]
        charts["cluster_eval"] = {
            "k": m["k"].tolist(),
            "inertia": m["inertia"].round(2).tolist(),
            "silhouette": m["silhouette"].round(4).tolist(),
        }
    except Exception as e:
        print(f"  [warn] Cluster eval: {e}")

    # ==== 14b. Validation externe AutoScout24 ====
    try:
        scrape_path = "scrape_validation_results.json"
        if os.path.exists(scrape_path):
            with open(scrape_path) as f:
                scrape_data = json.load(f)
            if scrape_data:
                medians_delta = [r["delta_pct"] for r in scrape_data]
                charts["scrape_validation"] = {
                    "results": scrape_data,
                    "median_delta_pct": round(float(np.median(medians_delta)), 2),
                    "mean_delta_pct": round(float(np.mean(medians_delta)), 2),
                    "n_models": len(scrape_data),
                    "direction": "conservateur" if np.median(medians_delta) < 0 else "optimiste",
                }
    except Exception as e:
        print(f"  [warn] Scrape data: {e}")

    # (ex block 14c/14d supprimés : le split temporel canonique est désormais
    # intégré au ModelComparison ci-dessus, et rolling_origin baseline a été
    # remplacé par rolling_origin_postcovid qui couvre tous les modèles.)

    # ==== 14e. COVID regime study ====
    # Charge covid_regime_results.json : 3 variantes comparées sur le split
    # temporel (full / post-covid / covid_flag) pour quantifier l'impact du
    # shortage de semi-conducteurs sur la généralisation.
    try:
        covid_path = "covid_regime_results.json"
        if os.path.exists(covid_path):
            with open(covid_path) as f:
                covid_data = json.load(f)
            charts["covid_regime"] = covid_data
    except Exception as e:
        print(f"  [warn] COVID regime data: {e}")

    # ==== 14f. Conformal prediction intervals ====
    # Charge conformal_results.json : IC80/90/95 par split conformal sur le
    # split temporel, avec coverage empirique et width. Brique de calibration
    # d'incertitude (risk portfolio → VaR/CVaR sur reprises).
    try:
        conformal_path = "conformal_results.json"
        if os.path.exists(conformal_path):
            with open(conformal_path) as f:
                conformal_data = json.load(f)
            charts["conformal"] = conformal_data
    except Exception as e:
        print(f"  [warn] Conformal data: {e}")

    # ==== 14h. Rolling-origin POST-COVID (walk-forward) ====
    # Charge rolling_origin_postcovid_results.json : 3 folds (2024-01 / 2024-07
    # / 2025-01) avec train >= 2022-01 à chaque fois + delta vs baseline full
    # train pour valider que le gain -4.5pt MAPE tient hors de la coupure
    # unique 2024-01.
    try:
        ro_pc_path = "rolling_origin_postcovid_results.json"
        if os.path.exists(ro_pc_path):
            with open(ro_pc_path) as f:
                ro_pc_data = json.load(f)
            charts["rolling_origin_postcovid"] = ro_pc_data
    except Exception as e:
        print(f"  [warn] Rolling-origin post-COVID: {e}")

    # ==== 14h-quater. Stress test 49/51 (random split post-COVID) ====
    # On tire aléatoirement 49% train / 51% test sur la période post-COVID
    # avec les HP tunés. Vérifie que le ranking des modèles reste robuste
    # quand l'information temporelle n'avantage plus le train.
    if stress is not None:
        charts["stress_test_4951"] = stress

    # ==== 14h-bis. Tuning HP vs Baseline (post-COVID) ====
    # Charge tuning_vs_baseline_results.json : Optuna TPE sur CatBoost/XGBoost
    # (TSSplit(3) sur le train du cutoff 2024-01) + relance du walk-forward
    # avec HP gelés + diff métrique-par-métrique vs baseline.
    try:
        tvb_path = "tuning_vs_baseline_results.json"
        if os.path.exists(tvb_path):
            with open(tvb_path) as f:
                tvb_data = json.load(f)
            charts["tuning_vs_baseline"] = tvb_data
    except Exception as e:
        print(f"  [warn] Tuning vs Baseline: {e}")

    # ==== 14g. AS24 re-validation post-COVID ====
    # Charge as24_postcovid_results.json : on rejoue la comparaison portfolio
    # vs AutoScout24 avec le CatBoost post-COVID (variante gagnante du block
    # 14e) et on mesure la réduction du biais B2B->C2C et la nouvelle
    # constante de recalibration.
    try:
        as24_pc_path = "as24_postcovid_results.json"
        if os.path.exists(as24_pc_path):
            with open(as24_pc_path) as f:
                as24_pc_data = json.load(f)
            charts["as24_postcovid"] = as24_pc_data
    except Exception as e:
        print(f"  [warn] AS24 post-COVID data: {e}")

    # ==== 14i. Pipeline structuré — lecture log_ratio du vr_pipeline ====
    # Charge les artefacts CSV/JSON régénérés par `scripts/regenerate_houssem_artifacts.py`
    # à partir du VRAI vr_pipeline (Ridge, RandomForest, XGBoost, CatBoost),
    # même split temporel post-COVID que la section 06 mais métriques sur
    # l'échelle log_ratio + feature importance du modèle retenu + évaluation
    # par segment marque / carburant.
    try:
        pipeline_dir = "artifacts/houssem"
        if os.path.isdir(pipeline_dir):
            pipeline_data = {}

            # -- Tableau métriques par modèle (4 candidats sur log_ratio scale)
            mc_path = f"{pipeline_dir}/model_comparison.csv"
            if os.path.exists(mc_path):
                mc = pd.read_csv(mc_path)
                pipeline_data["model_comparison"] = {
                    "models":  mc["model"].tolist(),
                    "mae":     [round(float(v), 4) for v in mc["MAE"]],
                    "rmse":    [round(float(v), 4) for v in mc["RMSE"]],
                    "r2":      [round(float(v), 4) for v in mc["R2"]],
                    "mape":    [round(float(v), 2) for v in mc["MAPE"]],
                    "medae":   [round(float(v), 4) for v in mc["MedAE"]],
                }
                # Champion = plus petit MAE
                idx_best = mc["MAE"].idxmin()
                pipeline_data["best_model"] = str(mc.loc[idx_best, "model"])
                pipeline_data["best_mae"]   = round(float(mc.loc[idx_best, "MAE"]), 4)
                pipeline_data["best_r2"]    = round(float(mc.loc[idx_best, "R2"]), 4)
                pipeline_data["best_mape"]  = round(float(mc.loc[idx_best, "MAPE"]), 2)

            # -- SHAP global importance (top 10 features)
            shap_path = f"{pipeline_dir}/shap_importance.csv"
            if os.path.exists(shap_path):
                shap_df = pd.read_csv(shap_path)
                shap_df = shap_df[shap_df["mean_abs_shap"] > 0]
                shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).head(10)
                pipeline_data["shap_top"] = {
                    "features": shap_df["feature"].tolist(),
                    "values":   [round(float(v), 4) for v in shap_df["mean_abs_shap"]],
                }

            # -- Évaluation par segment (marque)
            seg_brand_path = f"{pipeline_dir}/segment_evaluation_brand.csv"
            if os.path.exists(seg_brand_path):
                sb = pd.read_csv(seg_brand_path)
                pipeline_data["segment_brand"] = [
                    {
                        "segment": str(r["segment"]),
                        "mae":     round(float(r["MAE"]), 4),
                        "r2":      round(float(r["R2"]), 4),
                        "mape":    round(float(r["MAPE"]), 2),
                        "n":       int(r["n"]),
                    }
                    for _, r in sb.iterrows()
                ]

            # -- Évaluation par segment (carburant)
            seg_fuel_path = f"{pipeline_dir}/segment_evaluation_fuel.csv"
            if os.path.exists(seg_fuel_path):
                sf = pd.read_csv(seg_fuel_path)
                pipeline_data["segment_fuel"] = [
                    {
                        "segment": str(r["segment"]),
                        "mae":     round(float(r["MAE"]), 4),
                        "r2":      round(float(r["R2"]), 4),
                        "mape":    round(float(r["MAPE"]), 2),
                        "n":       int(r["n"]),
                    }
                    for _, r in sf.iterrows()
                ]

            # -- Data audit (volumétrie used_market vs portfolio)
            audit_path = f"{pipeline_dir}/data_audit.json"
            if os.path.exists(audit_path):
                with open(audit_path) as f:
                    audit = json.load(f)
                pipeline_data["data_audit"] = {
                    "n_used_market": int(audit.get("used_market_rows", 0)),
                    "n_portfolio":   int(audit.get("portfolio_rows", 0)),
                }

            # -- Liste des features (numeric + categorical) pour la section FE
            feat_path = f"{pipeline_dir}/feature_lists.json"
            if os.path.exists(feat_path):
                with open(feat_path) as f:
                    feat = json.load(f)
                pipeline_data["features"] = {
                    "numeric":     feat.get("numeric_features", []),
                    "categorical": feat.get("categorical_features", []),
                }

            charts["houssem_pipeline"] = pipeline_data
    except Exception as e:
        print(f"  [warn] structured pipeline: {e}")

    # ==== 15a. Risk Portfolio Dashboard ====
    # Source des prédictions : snapshot figé `prediction_portfolio.csv`
    # (committé sur victoria-2.0, total 20,27 M€). C'est la référence
    # présentée en démo ; on évite que le risque dérive si la pipeline
    # est ré-entraînée avec des seeds/params différents.
    risk_preds = preds.copy()
    v2_csv = "prediction_portfolio.csv"
    if os.path.exists(v2_csv) and {"id"}.issubset(risk_preds.columns):
        v2 = pd.read_csv(v2_csv)
        v2 = v2.rename(columns={"prediction": "_v2_pred"})
        risk_preds = risk_preds.merge(v2[["id", "_v2_pred"]], on="id", how="left")
        if risk_preds["_v2_pred"].notna().all():
            risk_preds["prediction"] = risk_preds["_v2_pred"]
            risk_preds["decote_pct"] = (1 - risk_preds["prediction"] / risk_preds["prix_catalogue"]) * 100
            print(f"  [risk] utilise prediction_portfolio.csv (snapshot v2.0, sum={risk_preds['prediction'].sum()/1e6:.2f} M EUR)", flush=True)
        risk_preds = risk_preds.drop(columns=["_v2_pred"])

    if {"prediction", "prix_catalogue", "brand", "fuel_type"}.issubset(risk_preds.columns):
        total_exposure = float(risk_preds["prediction"].sum())
        total_catalogue = float(risk_preds["prix_catalogue"].sum())
        avg_decote = float(risk_preds["decote_pct"].mean()) if "decote_pct" in risk_preds.columns else 0

        # Scénarios de stress
        scenarios = []
        for name, shock_pct in [
            ("Central (base)", 0),
            ("Modéré (-5%)", -5),
            ("Adverse (-10%)", -10),
            ("Sévère (-15%)", -15),
        ]:
            vr_stressed = risk_preds["prediction"] * (1 + shock_pct / 100)
            total_stressed = float(vr_stressed.sum())
            loss = total_exposure - total_stressed
            scenarios.append({
                "name": name,
                "shock_pct": shock_pct,
                "total_vr": round(total_stressed, 0),
                "loss": round(loss, 0),
                "loss_pct": round(100 * loss / total_exposure, 2) if total_exposure > 0 else 0,
            })

        # Breakdown par marque
        brand_risk = risk_preds.groupby("brand").agg(
            n=("prediction", "count"),
            total_vr=("prediction", "sum"),
            median_vr=("prediction", "median"),
            median_decote=("decote_pct", "median") if "decote_pct" in risk_preds.columns else ("prediction", "count"),
        ).reset_index().sort_values("total_vr", ascending=False)

        # Breakdown par carburant
        fuel_risk = risk_preds.groupby("fuel_type").agg(
            n=("prediction", "count"),
            total_vr=("prediction", "sum"),
            median_vr=("prediction", "median"),
            median_decote=("decote_pct", "median") if "decote_pct" in risk_preds.columns else ("prediction", "count"),
        ).reset_index().sort_values("median_decote", ascending=False)

        # Top 10 véhicules à plus forte décote + liste complète catégorisée par niveau
        if "decote_pct" in risk_preds.columns:
            all_cols = ["brand", "model", "fuel_type", "age_months", "prix_catalogue", "prediction", "decote_pct"]
            all_cols = [c for c in all_cols if c in risk_preds.columns]
            all_risk = risk_preds[all_cols].copy()

            def _risk_level(d):
                if d > 60:
                    return "high"
                if d > 50:
                    return "medium"
                return "low"

            all_risk["risk_level"] = all_risk["decote_pct"].apply(_risk_level)
            if "age_months" in all_risk.columns:
                all_risk["age_years"] = (all_risk["age_months"] / 12).round(1)
                all_risk = all_risk.drop(columns=["age_months"])
            all_risk["prix_catalogue"] = all_risk["prix_catalogue"].round(0)
            all_risk["prediction"] = all_risk["prediction"].round(0)
            all_risk["decote_pct"] = all_risk["decote_pct"].round(1)
            all_risk = all_risk.sort_values("decote_pct", ascending=False)
            top_risk_data = all_risk.head(10).drop(columns=["risk_level"]).to_dict("records")
            all_risk_data = all_risk.to_dict("records")
        else:
            top_risk_data = []
            all_risk_data = []

        # Nombre de véhicules à risque (décote > 60%)
        if "decote_pct" in risk_preds.columns:
            n_high_risk = int((risk_preds["decote_pct"] > 60).sum())
            n_medium_risk = int(((risk_preds["decote_pct"] > 50) & (risk_preds["decote_pct"] <= 60)).sum())
            n_low_risk = int((risk_preds["decote_pct"] <= 50).sum())
        else:
            n_high_risk = n_medium_risk = n_low_risk = 0

        charts["risk_portfolio"] = {
            "total_exposure": round(total_exposure, 0),
            "total_catalogue": round(total_catalogue, 0),
            "avg_decote": round(avg_decote, 1),
            "n_vehicles": int(len(risk_preds)),
            "n_high_risk": n_high_risk,
            "n_medium_risk": n_medium_risk,
            "n_low_risk": n_low_risk,
            "scenarios": scenarios,
            "brand_risk": [
                {
                    "brand": row["brand"],
                    "n": int(row["n"]),
                    "total_vr": round(float(row["total_vr"]), 0),
                    "median_vr": round(float(row["median_vr"]), 0),
                    "median_decote": round(float(row["median_decote"]), 1),
                }
                for _, row in brand_risk.iterrows()
            ],
            "fuel_risk": [
                {
                    "fuel_type": row["fuel_type"],
                    "n": int(row["n"]),
                    "total_vr": round(float(row["total_vr"]), 0),
                    "median_vr": round(float(row["median_vr"]), 0),
                    "median_decote": round(float(row["median_decote"]), 1),
                }
                for _, row in fuel_risk.iterrows()
            ],
            "top_risk": top_risk_data,
            "all_risk": all_risk_data,
        }

    # ==== 15b. Innovation : Brent Crude comparison ====
    try:
        brent_path = "data/external/brent_monthly.csv"
        if os.path.exists(brent_path):
            brent_df = pd.read_csv(brent_path)
            brent_df["month"] = brent_df["month"].astype(str)
            # Garder la période 2013-2026
            charts["brent_timeseries"] = {
                "dates": brent_df["month"].tolist(),
                "price": brent_df["brent_price"].round(2).tolist(),
                "yoy": brent_df["brent_yoy"].round(2).fillna(0).tolist(),
            }

        # Résultats de l'analyse si disponibles
        results_path = "brent_analysis_results.json"
        if os.path.exists(results_path):
            with open(results_path) as f:
                brent_results = json.load(f)
            charts["brent_results"] = brent_results
    except Exception as e:
        print(f"  [warn] Brent data: {e}")

    # ==== 16. Sale year evolution ====
    if "date de vente" in df.columns:
        df_year = df.copy()
        df_year["sale_year"] = pd.to_datetime(df_year["date de vente"]).dt.year
        year_stats = df_year.groupby("sale_year")["prix_vente"].agg(
            ["median", "mean", "count"]
        ).reset_index()
        year_stats = year_stats[year_stats["count"] >= 100]
        charts["price_by_year"] = {
            "years": year_stats["sale_year"].tolist(),
            "median": year_stats["median"].round(0).tolist(),
            "mean": year_stats["mean"].round(0).tolist(),
        }

    return charts


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

NGROK_DEFAULT_URL = "https://expert-armory-concerned.ngrok-free.dev"


def _detect_ngrok_url() -> str | None:
    """Résout l'URL publique ngrok utilisée pour le QR code partagé (jury).

    Priorité :
      1. Variable d'env NGROK_URL (fixée dans docker-compose.yml pour le
         subdomain réservé — stable entre runs).
      2. API locale ngrok http://<host>:4040/api/tunnels (run natif hors
         Docker uniquement : l'API exige auth depuis host.docker.internal).
      3. Fallback sur le subdomain réservé présentation (identique à
         docker-compose.yml) — garantit que le QR natif pointe vers la
         bonne URL même sans env ni tunnel local.
    """
    env_url = os.environ.get("NGROK_URL", "").strip()
    if env_url:
        return env_url

    import urllib.request, urllib.error
    for host in ["localhost", "127.0.0.1"]:
        try:
            with urllib.request.urlopen(
                f"http://{host}:4040/api/tunnels", timeout=0.4
            ) as r:
                data = json.loads(r.read())
                tunnels = data.get("tunnels", [])
                https = next(
                    (t["public_url"] for t in tunnels
                     if t.get("proto") == "https" and t.get("public_url")),
                    None,
                )
                if https:
                    return https
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return NGROK_DEFAULT_URL


@app.route("/")
def index():
    ngrok_url = _detect_ngrok_url()
    resp = app.make_response(render_template("index.html", ngrok_url=ngrok_url))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/data")
def api_data():
    resp = app.response_class(
        response=json.dumps(
            {
                "meta": STATE["meta"],
                "charts": STATE["charts"],
                "options": STATE["valid_options"],
            },
            cls=_Enc,
        ),
        mimetype="application/json",
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run a prediction for user-specified vehicle characteristics."""
    data = request.json
    pipeline = STATE["pipeline"]
    if pipeline is None:
        return jsonify({"error": "Pipeline non chargé"}), 500

    try:
        brand = data["brand"]
        fuel_type = data["fuel_type"]
        range_type = data.get("range_type", "PC")
        production_year = int(data["production_year"])
        contract_end_year = int(data["contract_end_year"])
        initial_mileage = float(data.get("initial_mileage", 0))
        contract_mileage = float(data["contract_mileage"])
        prix_catalogue = float(data["prix_catalogue"])

        row = {
            "brand": brand,
            "fuel_type": fuel_type,
            "range_type": range_type,
            "production_year": production_year,
            "contract_end_year": contract_end_year,
            "initial_mileage": initial_mileage,
            "contract_mileage": contract_mileage,
            "prix catalogue d'origine": prix_catalogue,
            "model": data.get("model", "UNKNOWN"),
            "current_contract_planned_end_date": f"{contract_end_year}-06-01",
            "id": 0,
        }
        df_sim = pd.DataFrame([row])
        result = vp.predict_portfolio(pipeline, df_sim)

        prediction = float(result["prediction"].iloc[0])
        decote = float(result["decote_pct"].iloc[0])
        age_months = int(
            (contract_end_year - production_year) * 12
        )
        total_km = initial_mileage + contract_mileage

        # Bande d'incertitude : MAPE walk-forward post-COVID (3 folds × 6 mois)
        # du modèle retenu (XGBoost tuné). Source canonique unique partagée
        # avec le hero / benchmark — cohérent avec ce qui est affiché ailleurs.
        meta = STATE.get("meta", {})
        mape_ref_pct = meta.get("best_mape")
        mae_ref_eur = meta.get("best_mae")
        band_pct = mape_ref_pct if mape_ref_pct is not None else None
        band_eur = None
        if band_pct is not None:
            band_eur = round(prediction * band_pct / 100.0, 0)

        payload = {
            "prediction": round(prediction, 2),
            "decote_pct": round(decote, 1),
            "prix_catalogue": prix_catalogue,
            "age_months": age_months,
            "age_years": round(age_months / 12, 1),
            "total_km": round(total_km, 0),
            "ratio_pct": round((prediction / prix_catalogue) * 100, 1),
            "model_name": meta.get("best_model"),
            "metrics_source": meta.get("metrics_source"),
            "uncertainty": {
                "mape_pct": band_pct,
                "mae_eur": mae_ref_eur,
                "band_eur": band_eur,
                "prediction_low": round(prediction - band_eur, 0) if band_eur is not None else None,
                "prediction_high": round(prediction + band_eur, 0) if band_eur is not None else None,
                "source": meta.get("metrics_source"),
                "n_folds": (meta.get("walk_forward") or {}).get("n_folds"),
                "horizon_months": (meta.get("walk_forward") or {}).get("horizon_months"),
            },
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------------------------------------------------------
# RAG Assistant — proxy Ollama (llama3.1:8b) with dashboard context
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


def _build_rag_context() -> str:
    """Compact project brief injected as system prompt for every query."""
    meta = STATE.get("meta", {}) or {}
    wf = meta.get("walk_forward") or {}
    cutoffs = ", ".join(wf.get("cutoffs", [])) if wf.get("cutoffs") else "—"
    best_model = meta.get("best_model") or "—"
    best_mape = meta.get("best_mape")
    best_r2 = meta.get("best_r2")
    best_mae = meta.get("best_mae")
    single_mape = meta.get("best_mape_single_split")
    single_r2 = meta.get("best_r2_single_split")
    source = meta.get("metrics_source") or "—"
    n_portfolio = meta.get("n_portfolio")
    n_models = meta.get("n_models_compared")

    lines = [
        "Tu es l'assistant méthodologique du dashboard Nexialog VR (Challenge Nexialog 2026, [Client]).",
        "Tu réponds en FRANÇAIS, en 3-6 phrases max, ton analytique et honnête.",
        "Tu peux citer des chiffres SEULEMENT si présents dans le contexte ci-dessous — sinon dis que ce n'est pas dans le contexte.",
        "Tu formates les nombres clés en **gras** (markdown). Code inline entre backticks. Pas de section headers.",
        "",
        "CONTEXTE DU PROJET :",
        "- Objectif : prédire la valeur résiduelle (prix de revente en euros, fin de contrat de leasing) de N véhicules du portefeuille vendus sur le marché cible.",
        "- Features : marque, carburant, gamme, âge mois, kilométrage total, prix catalogue d'origine + macro HICP (inflation core/headline/energy YoY + log_cum_inflation_core).",
        "- Cible modélisée : log_ratio = log(prix_vente / prix_catalogue). Back-transform V_t = V_0 · exp(log_ratio_hat) pour les métriques en EUR.",
        "- Dataset : données de transactions du marché de l'occasion.",
        "",
        "MODÈLE RETENU :",
        f"- {best_model} (critère de sélection : stabilité intra-CV = std MAPE minimale sur TimeSeriesSplit(3)).",
        f"- Métriques production (source: {source}): MAPE = **{best_mape}%**, R² = **{best_r2}**, MAE = **{best_mae} €**.",
        f"- Walk-forward post-COVID : {wf.get('n_folds', '—')} folds × {wf.get('horizon_months', '—')} mois, cutoffs {cutoffs}.",
        f"- Rappel single-split (référence historique) : MAPE {single_mape}%, R² {single_r2}.",
        "",
        "PROTOCOLE :",
        "- Train restreint post-COVID (≥ 2022-01) pour éviter la distorsion shortage 2020-2022.",
        "- Hyperparamètres Optuna TPE (15 trials), tune-once-apply-everywhere sur les 3 cutoffs, MedianPruner.",
        "- Pas de leakage : chaque fold refait son clustering modèle-famille et les comp_stats sur son train-only.",
        "- Stress test temporel 49/51 : split chronologique au quantile 49%, vérifie robustesse sous rupture temporelle.",
        "- Baseline naïve stratifiée (brand × age, ratio médian) comme référence métier honnête.",
        "",
        "PORTEFEUILLE [CLIENT] :",
        f"- {n_portfolio if n_portfolio else 'N'} véhicules prédits, {n_models if n_models else 4} modèles comparés.",
        "- Décote moyenne observée sur le portefeuille. Stress tests calibrés BCE/EBA (-5/-10/-15%).",
        "",
        "VALIDATION EXTERNE :",
        "- AutoScout24 scraping : écart B2B vs B2C observé (attendu, borne supérieure en vente directe).",
        "- KBA (tension marché, fichiers scrapés depuis la plateforme de référence) : β non significatif au-dessus du signal HICP — modèle bien spécifié.",
        "",
        "Si la question sort du périmètre (ex: météo, code générique), redirige poliment vers le sujet du dashboard.",
    ]
    return "\n".join(lines)


def _ollama_stream(prompt: str, system: str):
    """Generator yielding NDJSON lines: {\"chunk\":str} or {\"done\":true,...}."""
    req_body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {
            "temperature": 0.4,
            "num_predict": 512,
            "top_p": 0.9,
        },
    }).encode("utf-8")
    url = f"{OLLAMA_HOST}/api/generate"
    req = urllib.request.Request(url, data=req_body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("response"):
                    yield json.dumps({"chunk": obj["response"]}, ensure_ascii=False) + "\n"
                if obj.get("done"):
                    dur_ms = int((obj.get("total_duration") or 0) / 1e6)
                    yield json.dumps({
                        "done": True,
                        "eval_count": obj.get("eval_count"),
                        "duration_ms": dur_ms,
                    }, ensure_ascii=False) + "\n"
                    return
    except urllib.error.URLError as e:
        yield json.dumps({"error": f"Ollama injoignable à {OLLAMA_HOST} — {e.reason if hasattr(e,'reason') else e}"}) + "\n"
    except Exception as e:
        yield json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n"


@app.route("/api/rag", methods=["POST"])
def api_rag():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    if len(query) > 2000:
        return jsonify({"error": "Query too long (max 2000 chars)"}), 400
    system = _build_rag_context()

    def gen():
        for line in _ollama_stream(query, system):
            yield line

    resp = Response(stream_with_context(gen()), mimetype="application/x-ndjson")
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_data()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
