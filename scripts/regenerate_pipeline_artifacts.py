"""Regenerate artifacts/pipeline/ CSVs with the REAL victoria-2.0 pipeline
models (Ridge / Random Forest / XGBoost / CatBoost) on the temporal post-COVID
split. Replaces the obsolete pipeline artifacts (ElasticNet / CatBoost_main with
monotonic constraints + propensity reweighting) which are no longer used.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vr_pipeline import (  # noqa: E402
    TARGET,
    compare_models_on_used_market,
    load_portfolio,
    load_used_market,
)

OUT_DIR = Path("artifacts/pipeline")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _metrics_logratio(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    denom = np.where(y_true == 0, 1e-9, y_true)
    return {
        "MAE": round(float(np.mean(np.abs(residuals))), 4),
        "RMSE": round(float(np.sqrt(np.mean(residuals**2))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAPE": round(float(np.mean(np.abs(residuals / denom)) * 100), 2),
        "MedAE": round(float(median_absolute_error(y_true, y_pred)), 4),
    }


def main() -> None:
    print("[1/5] Loading used_market + portfolio ...")
    um = load_used_market()
    pf = load_portfolio()

    print("[2/5] compare_models_on_used_market (4 models, post-COVID split) ...")
    mc = compare_models_on_used_market(
        um,
        models_to_run=("ridge", "xgboost", "catboost", "random_forest"),
    )

    df_test = mc.df_test
    models = mc.models
    n_features_ext = mc.num_features_ext

    display_name = {
        "ridge": "Ridge",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "random_forest": "RandomForest",
    }

    print("[3/5] Writing final_model_comparison.csv ...")
    rows = []
    for internal, disp in display_name.items():
        mdl = models[internal]
        y_true = df_test[TARGET].to_numpy()
        y_pred = mdl.predict(df_test)
        m = _metrics_logratio(y_true, y_pred)
        rows.append({"model": disp, **m})
    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUT_DIR / "final_model_comparison.csv", index=False)
    final_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    # Baseline alias = 3 first (without retained)
    final_df.head(3).to_csv(OUT_DIR / "baseline_comparison.csv", index=False)

    # Chosen = lowest MAE
    chosen_idx = final_df["MAE"].idxmin()
    chosen_display = final_df.loc[chosen_idx, "model"]
    chosen_internal = {v: k for k, v in display_name.items()}[chosen_display]
    print(f"     -> chosen = {chosen_display} (internal {chosen_internal})")

    print("[4/5] Writing shap_importance.csv (feature importance of chosen model) ...")
    # Reuse _extract_feature_importance but write TOP-N with importance column
    fi = mc.feature_importance
    if fi is None:
        raise RuntimeError("feature_importance missing for chosen model")
    feat_rows = [
        {"feature": f, "mean_abs_shap": v}
        for f, v in zip(fi["features"], fi["importance"])
    ]
    pd.DataFrame(feat_rows).to_csv(OUT_DIR / "shap_importance.csv", index=False)

    print("[5/5] Segment evaluation (brand + fuel_type) on chosen model ...")
    chosen_model = models[chosen_internal]
    y_true_all = df_test[TARGET].to_numpy()
    y_pred_all = chosen_model.predict(df_test)

    def _seg_metrics(df_seg: pd.DataFrame, y_true_seg: np.ndarray, y_pred_seg: np.ndarray) -> dict:
        residuals = y_true_seg - y_pred_seg
        denom = np.where(y_true_seg == 0, 1e-9, y_true_seg)
        return {
            "MAE":   round(float(np.mean(np.abs(residuals))), 4),
            "RMSE":  round(float(np.sqrt(np.mean(residuals**2))), 4),
            "R2":    round(float(r2_score(y_true_seg, y_pred_seg)), 4) if len(y_true_seg) > 1 else float("nan"),
            "MAPE":  round(float(np.mean(np.abs(residuals / denom)) * 100), 2),
            "MedAE": round(float(median_absolute_error(y_true_seg, y_pred_seg)), 4),
            "n":     int(len(y_true_seg)),
        }

    for seg_col, filename in [("brand", "segment_evaluation_brand.csv"),
                               ("fuel_type", "segment_evaluation_fuel.csv")]:
        if seg_col not in df_test.columns:
            print(f"     [warn] '{seg_col}' not in df_test; skipping {filename}")
            continue
        segs = []
        df_test_reset = df_test.reset_index(drop=True)
        for seg_val, idx in df_test_reset.groupby(seg_col).indices.items():
            y_t = y_true_all[idx]
            y_p = y_pred_all[idx]
            m = _seg_metrics(df_test_reset.iloc[idx], y_t, y_p)
            segs.append({"segment": str(seg_val), **m})
        seg_df = pd.DataFrame(segs).sort_values("MAPE").reset_index(drop=True)
        seg_df.to_csv(OUT_DIR / filename, index=False)
        print(f"     wrote {filename} ({len(seg_df)} segments)")

    # Data audit
    audit = {
        "used_market_rows": int(len(um)),
        "portfolio_rows":   int(len(pf)),
        "chosen_model":     chosen_display,
        "pipeline":         "vr_pipeline post-COVID (ridge/xgboost/catboost/random_forest)",
    }
    (OUT_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Feature lists: recompute from ext features
    feature_lists = {
        "numeric_features":     sorted(set(n_features_ext)),
        "categorical_features": ["brand", "model", "fuel_type", "gearbox",
                                  "body_type", "range_type", "power_band",
                                  "model_family"],
    }
    (OUT_DIR / "feature_lists.json").write_text(json.dumps(feature_lists, indent=2), encoding="utf-8")

    print("\n[OK] Regeneration complete — artifacts/pipeline/ now reflects the real pipeline.")


if __name__ == "__main__":
    main()
