"""
Validation AutoScout24 du modèle post-COVID (cutoff train >= 2022-01).

Contexte : la variante `train_post_covid` (retire 2018 -> 2021) gagne
-4.47pt de MAPE vs baseline_full sur le split temporel (9.72% vs 14.19%)
et +0.35 pt de R^2_log_ratio (0.84 vs 0.49). On valide ici que ce gain
in-sample se traduit aussi sur la cible business : la distribution de prix
B2C scrapée sur AutoScout24 pour 10 modèles portfolio.

Protocole :
  1. Train = used_market 2022-01-01 <= date_vente < 2024-01-01 (GPU CatBoost).
  2. Feature engineering (clustering modèle, comp_stats, derived_ratio) fit
     sur train only pour éviter le data leak.
  3. Prédiction sur les 1951 véhicules du portfolio Mobilize.
  4. Comparaison des médianes prédites vs médianes scrapées AS24 (10 modèles).
  5. Facteur de recalibration B2B -> C2C = médiane(as24/pred).
  6. Comparaison avec le baseline_full AS24 (temporal_split_results.json).

Output -> as24_postcovid_results.json (consommé par app.py pour la card
  "AS24 re-validation (post-COVID)" du dashboard).
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
SCRAPE_PATH = "scrape_validation_results.json"
BASELINE_PATH = "temporal_split_results.json"
OUTPUT_PATH = "as24_postcovid_results.json"
USE_GPU = True


def compare_portfolio_to_as24(
    portfolio_preds: pd.DataFrame, scraped: list, baseline_temporal: dict | None = None
) -> dict:
    """Compare post-COVID portfolio predictions aux prix AS24.

    `baseline_temporal` = bloc as24_validation de temporal_split_results.json
    (baseline_full CatBoost entraîné 2018+<2024-01). On l'utilise pour
    l'apples-to-apples : deux CatBoost temporels, seule diff = fenêtre train.
    """
    base_by_model = {}
    if baseline_temporal and "results" in baseline_temporal:
        base_by_model = {r["model"].upper(): r for r in baseline_temporal["results"]}

    rows = []
    for entry in scraped:
        model_key = entry["model"].upper()
        sub = portfolio_preds[portfolio_preds["model"] == model_key]
        if len(sub) < 3:
            continue
        pred_median = round(float(sub["prediction"].median()), 0)
        as24 = entry["as24_median"]
        delta_eur = pred_median - as24
        delta_pct = 100 * delta_eur / as24
        base = base_by_model.get(model_key, {})
        rows.append({
            "brand": entry["brand"],
            "model": entry["model"],
            "n_portfolio": int(len(sub)),
            "pred_median_postcovid": pred_median,
            "pred_median_baseline_temporal": base.get("pred_median_temporal"),
            "pred_median_baseline_random": entry["pred_median"],
            "as24_median": as24,
            "delta_pct_postcovid": round(delta_pct, 2),
            "delta_pct_baseline_temporal": base.get("delta_pct_temporal"),
            "delta_pct_baseline_random": entry["delta_pct"],
        })
    if not rows:
        return {"error": "aucune correspondance portfolio <-> scrape"}
    deltas_pc = [r["delta_pct_postcovid"] for r in rows]
    deltas_bt = [r["delta_pct_baseline_temporal"] for r in rows if r["delta_pct_baseline_temporal"] is not None]
    deltas_br = [r["delta_pct_baseline_random"] for r in rows]
    out = {
        "results": rows,
        "n_models": len(rows),
        "median_delta_pct_postcovid": round(float(np.median(deltas_pc)), 2),
        "median_delta_pct_baseline_random": round(float(np.median(deltas_br)), 2),
        "mean_delta_pct_postcovid": round(float(np.mean(deltas_pc)), 2),
        "mean_delta_pct_baseline_random": round(float(np.mean(deltas_br)), 2),
    }
    if deltas_bt:
        out["median_delta_pct_baseline_temporal"] = round(float(np.median(deltas_bt)), 2)
        out["mean_delta_pct_baseline_temporal"] = round(float(np.mean(deltas_bt)), 2)
    return out


def compute_recalibration(as24_block: dict) -> dict:
    rows = as24_block.get("results", [])
    if not rows:
        return {"error": "pas de données AS24 pour recalibrer"}
    ratios = [r["as24_median"] / r["pred_median_postcovid"] for r in rows]
    factor = float(np.median(ratios))
    recalibrated_deltas = []
    for r in rows:
        recal_pred = r["pred_median_postcovid"] * factor
        delta = 100 * (recal_pred - r["as24_median"]) / r["as24_median"]
        recalibrated_deltas.append(round(delta, 2))
    return {
        "factor": round(factor, 4),
        "method": "median(as24_median / pred_median_postcovid) sur 10 modèles AS24",
        "ratios": [round(x, 4) for x in ratios],
        "delta_pct_recalibrated": recalibrated_deltas,
        "median_delta_pct_recalibrated": round(float(np.median(recalibrated_deltas)), 2),
        "mean_delta_pct_recalibrated": round(float(np.mean(recalibrated_deltas)), 2),
        "max_abs_delta_pct_recalibrated": round(
            float(np.max(np.abs(recalibrated_deltas))), 2
        ),
    }


def run_as24_postcovid():
    print("=" * 72)
    device = "GPU" if USE_GPU else "CPU"
    print(f"  AS24 validation post-COVID | train >= {POST_COVID_CUTOFF.date()} "
          f"< {CUTOFF.date()} | CatBoost {device}")
    print("=" * 72)

    # ========= 1. Data & split =========
    um = vp.load_used_market()
    df = vp.prepare_used_market(um)
    df_train = df[
        (df["date de vente"] >= POST_COVID_CUTOFF)
        & (df["date de vente"] < CUTOFF)
    ].copy()
    df_test = df[df["date de vente"] >= CUTOFF].copy()
    print(f"  Train post-COVID : {len(df_train):>6,} "
          f"({df_train['date de vente'].min().date()} -> "
          f"{df_train['date de vente'].max().date()})")
    print(f"  Test (sanity)    : {len(df_test):>6,}")
    print()

    # ========= 2. Feature engineering (fit sur train only) =========
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

    # ========= 3. Fit CatBoost post-COVID =========
    print("  [fit] CatBoost post-COVID...", flush=True)
    t0 = time.perf_counter()
    model = vp.train_catboost(
        df_train, num_features=num_features_ext,
        task_type="GPU" if USE_GPU else "CPU",
    )
    fit_sec = round(time.perf_counter() - t0, 2)
    print(f"        fait en {fit_sec}s")

    # Sanity : re-check MAPE/R^2 sur le test >= 2024-01
    y_test_eur = df_test["prix_vente"].to_numpy()
    y_pred_eur = vp.predict_eur(model, df_test)
    mape = float(np.mean(np.abs((y_test_eur - y_pred_eur) / y_test_eur)) * 100)
    r2_eur = float(r2_score(y_test_eur, y_pred_eur))
    y_lr = df_test[vp.TARGET].to_numpy()
    y_lr_pred = model.predict(df_test)
    r2_lr = float(r2_score(y_lr, y_lr_pred))
    print(f"        sanity test >= 2024-01 : MAPE={mape:.2f}%  "
          f"R2={r2_eur:.4f}  R2_lr={r2_lr:.4f}")
    print()

    # ========= 4. Portfolio prediction =========
    print("  [portfolio] prediction 1951 véhicules Mobilize...")
    po = vp.load_portfolio()
    po_prep = vp.prepare_portfolio(po)
    po_prep = vp.apply_clustering(po_prep, clustering, new_col="model_family")
    po_prep = vp.add_derived_ratio_features(po_prep)
    po_prep = vp.merge_comparable_features(po_prep, comp_stats)
    po_prep["prediction"] = vp.predict_eur(model, po_prep)
    print(f"        prediction median = {po_prep['prediction'].median():,.0f}EUR  "
          f"mean = {po_prep['prediction'].mean():,.0f}EUR  "
          f"n = {len(po_prep)}")
    print()

    # ========= 5. AS24 comparison =========
    print("  [AS24] comparaison avec 10 modèles scrapés...")
    with open(SCRAPE_PATH) as f:
        scraped = json.load(f)
    baseline_temporal = None
    try:
        with open(BASELINE_PATH) as f:
            baseline_temporal = json.load(f).get("as24_validation")
    except FileNotFoundError:
        pass
    as24_block = compare_portfolio_to_as24(po_prep, scraped, baseline_temporal)
    base_temp_med = as24_block.get("median_delta_pct_baseline_temporal")
    print(f"        median delta post-COVID     = "
          f"{as24_block['median_delta_pct_postcovid']}%")
    if base_temp_med is not None:
        print(f"        median delta baseline_full  = {base_temp_med}%  "
              "(catboost 2018+ < 2024-01, même protocole)")
    print(f"        median delta random_split   = "
          f"{as24_block['median_delta_pct_baseline_random']}%  "
          "(réf. historique)")
    print()
    print(f"  {'brand':<10} {'model':<10} {'n':>4} {'pred_pc':>10} "
          f"{'pred_bt':>10} {'as24':>10} {'d_pc':>8} {'d_bt':>8}")
    print("  " + "-" * 78)
    for r in as24_block["results"]:
        pred_bt = r.get("pred_median_baseline_temporal") or 0
        d_bt = r.get("delta_pct_baseline_temporal")
        d_bt_str = f"{d_bt:>7.2f}%" if d_bt is not None else "     n/a"
        print(f"  {r['brand']:<10} {r['model']:<10} {r['n_portfolio']:>4} "
              f"{r['pred_median_postcovid']:>10,.0f} "
              f"{pred_bt:>10,.0f} "
              f"{r['as24_median']:>10,.0f} "
              f"{r['delta_pct_postcovid']:>7.2f}% "
              f"{d_bt_str}")
    print()

    # ========= 6. Recalibration B2B -> C2C =========
    print("  [recalibration] facteur global...")
    recal_block = compute_recalibration(as24_block)
    if "factor" in recal_block:
        print(f"        facteur = x{recal_block['factor']}  "
              f"median delta post-recal = "
              f"{recal_block['median_delta_pct_recalibrated']}%  "
              f"max |delta| = {recal_block['max_abs_delta_pct_recalibrated']}%")
    print()

    # ========= 7. Lecture baseline pour comparaison directe =========
    baseline_median_abs = None
    baseline_factor = None
    baseline_max_abs = None
    try:
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        base_med = baseline.get("as24_validation", {}).get("median_delta_pct_temporal")
        if base_med is not None:
            baseline_median_abs = abs(base_med)
        baseline_factor = baseline.get("recalibration", {}).get("factor")
        baseline_max_abs = baseline.get("recalibration", {}).get(
            "max_abs_delta_pct_recalibrated"
        )
    except FileNotFoundError:
        pass

    pc_median_abs = abs(as24_block["median_delta_pct_postcovid"])
    pc_max_abs = recal_block.get("max_abs_delta_pct_recalibrated")
    pc_factor = recal_block.get("factor")

    # Verdict : post-COVID améliore-t-il l'alignement à AS24 ?
    comparisons = []
    if baseline_median_abs is not None:
        delta_median_abs = round(pc_median_abs - baseline_median_abs, 2)
        comparisons.append(
            f"|median delta| : {pc_median_abs:.2f}% vs baseline "
            f"{baseline_median_abs:.2f}% (Δ={delta_median_abs:+.2f}pt)"
        )
    if baseline_factor is not None and pc_factor is not None:
        delta_factor = round(pc_factor - baseline_factor, 4)
        comparisons.append(
            f"facteur B2B->C2C : x{pc_factor} vs baseline x{baseline_factor} "
            f"(Δ={delta_factor:+.4f})"
        )
    if baseline_max_abs is not None and pc_max_abs is not None:
        comparisons.append(
            f"max |delta| post-recal : {pc_max_abs}% vs {baseline_max_abs}% "
            f"(baseline)"
        )

    if baseline_median_abs is not None:
        if pc_median_abs < baseline_median_abs - 0.5:
            verdict = (
                f"Post-COVID améliore l'alignement AS24 : |median delta| passe "
                f"de {baseline_median_abs:.2f}% a {pc_median_abs:.2f}% "
                f"(-{baseline_median_abs - pc_median_abs:.2f}pt). Le facteur de "
                f"recalibration x{pc_factor} reste proche, mais le gain "
                f"résiduel montre que le modèle capture mieux la dynamique C2C."
            )
        elif pc_median_abs > baseline_median_abs + 0.5:
            verdict = (
                f"Post-COVID dégrade l'alignement AS24 "
                f"({pc_median_abs:.2f}% vs {baseline_median_abs:.2f}%). "
                "Le gain in-sample ne se transfère pas à la cible C2C."
            )
        else:
            verdict = (
                f"Alignement AS24 équivalent au baseline_full "
                f"({pc_median_abs:.2f}% vs {baseline_median_abs:.2f}%). "
                "Le facteur de recalibration B2B -> C2C reste nécessaire, "
                "ce qui est attendu (les deux modèles apprennent une "
                "distribution B2B et non C2C)."
            )
    else:
        verdict = (
            f"|median delta| = {pc_median_abs:.2f}%. Recalibration "
            f"x{pc_factor} ramène max |delta| à {pc_max_abs}%."
        )

    print(f"  [verdict] {verdict}")
    for c in comparisons:
        print(f"            - {c}")

    result: dict[str, Any] = {
        "cutoff_test": str(CUTOFF.date()),
        "post_covid_cutoff": str(POST_COVID_CUTOFF.date()),
        "task_type": "GPU" if USE_GPU else "CPU",
        "train_seconds": fit_sec,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_portfolio": int(len(po_prep)),
        "sanity_test_metrics": {
            "mape_eur": round(mape, 2),
            "r2_eur": round(r2_eur, 4),
            "r2_logratio": round(r2_lr, 4),
        },
        "portfolio_prediction_stats": {
            "median_eur": round(float(po_prep["prediction"].median()), 0),
            "mean_eur": round(float(po_prep["prediction"].mean()), 0),
            "q25_eur": round(float(po_prep["prediction"].quantile(0.25)), 0),
            "q75_eur": round(float(po_prep["prediction"].quantile(0.75)), 0),
        },
        "as24_validation": as24_block,
        "recalibration": recal_block,
        "baseline_full_comparison": {
            "baseline_median_abs_delta_pct": baseline_median_abs,
            "baseline_recal_factor": baseline_factor,
            "baseline_max_abs_delta_recalibrated": baseline_max_abs,
            "postcovid_median_abs_delta_pct": round(pc_median_abs, 2),
            "postcovid_recal_factor": pc_factor,
            "postcovid_max_abs_delta_recalibrated": pc_max_abs,
        },
        "verdict": verdict,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  -> {OUTPUT_PATH}")
    print("=" * 72)
    return result


if __name__ == "__main__":
    run_as24_postcovid()
