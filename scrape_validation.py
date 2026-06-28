"""
Validation externe : comparaison des prédictions du modèle avec les prix
réels observés sur AutoScout24 (marché de l'occasion allemand).

Méthodologie :
1. Pour chaque modèle du portfolio, on calcule l'âge et km médians à la
   fin du contrat prévu.
2. On scrape AutoScout24 avec ces filtres (véhicules comparables).
3. On compare la médiane des prix demandés AutoScout24 avec la médiane
   prédite par notre modèle.

Les prix AutoScout24 sont des PRIX DEMANDÉS (annonces) — typiquement 10-15%
au-dessus des prix de TRANSACTION réels. Cet écart systématique doit être
pris en compte dans l'interprétation.
"""

from __future__ import annotations

import json
import re
import time
import warnings

import numpy as np
import pandas as pd
import requests

import vr_pipeline as vp

warnings.filterwarnings("ignore")


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}

MODELS_TO_SCRAPE = [
    ("RENAULT", "CAPTUR", "renault", "captur"),
    ("RENAULT", "CLIO", "renault", "clio"),
    ("RENAULT", "MEGANE", "renault", "megane"),
    ("RENAULT", "SCENIC", "renault", "scenic"),
    ("RENAULT", "TWINGO", "renault", "twingo"),
    ("RENAULT", "KANGOO", "renault", "kangoo"),
    ("DACIA", "DUSTER", "dacia", "duster"),
    ("DACIA", "SANDERO", "dacia", "sandero"),
    ("DACIA", "SPRING", "dacia", "spring"),
    ("NISSAN", "QASHQAI", "nissan", "qashqai"),
    ("NISSAN", "JUKE", "nissan", "juke"),
    ("NISSAN", "MICRA", "nissan", "micra"),
]


def scrape_autoscout24(brand_url: str, model_url: str,
                      fregfrom: int = None, fregto: int = None,
                      kmto: int = None) -> dict:
    """Scrape AutoScout24 avec filtres année de production et kilométrage."""
    url = f"https://www.autoscout24.de/lst/{brand_url}/{model_url}"
    params = []
    if fregfrom:
        params.append(f"fregfrom={fregfrom}")
    if fregto:
        params.append(f"fregto={fregto}")
    if kmto:
        params.append(f"kmto={kmto}")
    if params:
        url += "?" + "&".join(params)

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}", "url": url}

        price_matches = re.findall(
            r'data-testid="[^"]*price[^"]*"[^>]*>€\s*([\d.]+)', r.text
        )

        prices = []
        for p in price_matches:
            p_clean = p.replace(".", "").replace(",", ".")
            try:
                v = float(p_clean)
                if 500 < v < 150000:
                    prices.append(v)
            except ValueError:
                continue

        if len(prices) < 3:
            return {"error": f"Only {len(prices)} prices found", "url": url}

        return {
            "url": url,
            "n_listings": len(prices),
            "median": round(float(np.median(prices)), 0),
            "mean": round(float(np.mean(prices)), 0),
            "q25": round(float(np.percentile(prices, 25)), 0),
            "q75": round(float(np.percentile(prices, 75)), 0),
        }
    except Exception as e:
        return {"error": str(e), "url": url}


def get_portfolio_stats_per_model(pipeline, po):
    """Pour chaque modèle du portfolio, calcule âge/km médians et VR médiane."""
    preds = vp.predict_portfolio(pipeline, po)
    stats = {}
    for model in preds["model"].unique():
        sub = preds[preds["model"] == model]
        if len(sub) < 3:
            continue
        age_months = sub["age_months"].median()
        mileage_km = sub["mileage_km"].median()
        fregfrom = 2025 - int(np.ceil(age_months / 12))
        fregto = 2025 - int(np.floor(age_months / 12)) + 1
        kmto = int(mileage_km * 1.2 / 10000) * 10000  # arrondi 10k
        stats[model] = {
            "n_portfolio": int(len(sub)),
            "age_months_median": float(age_months),
            "mileage_km_median": float(mileage_km),
            "pred_median": round(float(sub["prediction"].median()), 0),
            "pred_q25": round(float(sub["prediction"].quantile(0.25)), 0),
            "pred_q75": round(float(sub["prediction"].quantile(0.75)), 0),
            "fregfrom": fregfrom,
            "fregto": fregto,
            "kmto": kmto,
        }
    return stats


def main():
    print("=" * 70)
    print("  Validation externe via AutoScout24")
    print("  (véhicules filtrés sur âge/km comparables au portfolio)")
    print("=" * 70)

    um = vp.load_used_market()
    po = vp.load_portfolio()
    pipeline = vp.fit_final_pipeline(um, "catboost", n_model_families=6)

    print("\n[pipeline] Stats portfolio par modèle...")
    portfolio_stats = get_portfolio_stats_per_model(pipeline, po)
    print(f"    {len(portfolio_stats)} modèles trouvés")

    # Scraping filtré
    print("\n[scrape] AutoScout24 avec filtres...")
    results = []
    for brand, model, brand_url, model_url in MODELS_TO_SCRAPE:
        if model not in portfolio_stats:
            print(f"    {brand} {model} : absent du portfolio, skip")
            continue
        pstats = portfolio_stats[model]
        scrape = scrape_autoscout24(
            brand_url, model_url,
            fregfrom=pstats["fregfrom"],
            fregto=pstats["fregto"],
            kmto=pstats["kmto"],
        )
        if "error" in scrape:
            print(f"    {brand:8s} {model:12s} | ERREUR : {scrape['error']}")
            continue
        delta = pstats["pred_median"] - scrape["median"]
        delta_pct = 100 * delta / scrape["median"]
        print(
            f"    {brand:8s} {model:12s} | "
            f"Portfolio âge {pstats['age_months_median']/12:.1f}a / "
            f"{pstats['mileage_km_median']/1000:.0f}k km | "
            f"Prédit {pstats['pred_median']:>6.0f} € | "
            f"AS24 {scrape['median']:>6.0f} € (n={scrape['n_listings']:2d}) | "
            f"Δ {delta_pct:+6.1f}%"
        )
        results.append({
            "brand": brand,
            "model": model,
            "n_portfolio": pstats["n_portfolio"],
            "age_years": round(pstats["age_months_median"] / 12, 1),
            "mileage_km": int(pstats["mileage_km_median"]),
            "pred_median": pstats["pred_median"],
            "pred_q25": pstats["pred_q25"],
            "pred_q75": pstats["pred_q75"],
            "as24_median": scrape["median"],
            "as24_q25": scrape["q25"],
            "as24_q75": scrape["q75"],
            "as24_n": scrape["n_listings"],
            "delta_eur": round(delta, 0),
            "delta_pct": round(delta_pct, 2),
        })
        time.sleep(1.5)

    # Synthèse
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print(f"  Synthèse (n={len(df)} modèles)")
        print("=" * 70)
        print(f"  Écart médian (prédit vs AutoScout24) : {df['delta_pct'].median():.2f}%")
        print(f"  Écart moyen : {df['delta_pct'].mean():.2f}%")
        print(f"  Modèles avec écart < ±15% : {((df['delta_pct'].abs() < 15).sum())}/{len(df)}")

    with open("scrape_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats : scrape_validation_results.json")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
