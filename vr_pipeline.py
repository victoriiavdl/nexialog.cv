"""
Pipeline de prédiction de valeur résiduelle (VR) automobile
===========================================================

Fonctions entraînées sur `used_market` puis appliquées au `portfolio`.


Étapes  d'utilisation :

    >>> import vr_pipeline as vp
    >>> um = vp.load_used_market()
    >>> po = vp.load_portfolio()
    >>> comparison = vp.compare_models_on_used_market(um, n_model_families=6)
    >>> comparison.metrics               # tableau MAE/MAPE/R2 par modèle
    >>> pipeline = vp.fit_final_pipeline(um, comparison.best_name)
    >>> preds = vp.predict_portfolio(pipeline, po)
    >>> preds[["id", "prediction"]].to_csv("prediction_portfolio.csv", index=False)


"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:  # pragma: no cover - dépendance optionnelle locale
    CatBoostRegressor = None
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor


CAT_FEATURES = ["brand", "fuel_type", "range_type", "model_family"]
NUM_FEATURES = [
    "log_age",
    "log_mileage",
    "log_catalogue",
    "log_cum_inflation_core",
    "hicp_headline_yoy_sale",
    "hicp_energy_yoy_sale",
]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "log_ratio"

# ---------------------------------------------------------------------------
# Protocole canonique : split temporel post-COVID
# ---------------------------------------------------------------------------
# Train : [2022-01-01, 2024-01-01)  -> ~24 mois post-shortage semi-conducteurs
# Test  : [2024-01-01, fin du dataset] -> fenêtre la plus récente
# Ce choix est validé par le rolling-origin post-COVID (stabilité des folds).
POST_COVID_TRAIN_START = pd.Timestamp("2022-01-01")
POST_COVID_CUTOFF = pd.Timestamp("2024-01-01")
N_MODEL_FAMILIES_DEFAULT = 6
RANDOM_SEED = 42


def _gpu_available() -> bool:
    """Best-effort detection d'un GPU NVIDIA exploitable par CatBoost/XGBoost."""
    import shutil, subprocess
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.STDOUT, timeout=3
        ).decode()
        return "GPU" in out
    except Exception:
        return False


_GPU_CACHED: bool | None = None


def gpu_available() -> bool:
    """Mémoïse la détection GPU (appelée plusieurs fois par trainer)."""
    global _GPU_CACHED
    if _GPU_CACHED is None:
        _GPU_CACHED = _gpu_available()
    return _GPU_CACHED


# ===========================================================================
# 1. Chargement et homogénéisation des var entre les 2 tables
# ===========================================================================

_UM_RENAME = {
    "MODEL": "model",
    "FUEL TYPE": "fuel_type",
    "RANGE TYPE": "range_type",
    "productionYear": "production_year",
}


def _normalize_used_market_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Accepte les noms bruts Excel ou les noms déjà harmonisés."""
    return df.rename(columns=_UM_RENAME)


def load_used_market(path: str = "data/used_market.xlsb") -> pd.DataFrame:
    """Charge used_market et aligne les noms de colonnes sur portfolio."""
    df = pd.read_excel(path, engine="pyxlsb")
    df["date de vente"] = pd.to_datetime(
        df["date de vente"], origin="1899-12-30", unit="D"
    )
    return _normalize_used_market_schema(df)


def load_portfolio(path: str = "data/portfolio.xlsx") -> pd.DataFrame:
    return pd.read_excel(path)


# ===========================================================================
# 1bis. Macro externe : HICP Allemagne (BCE / Destatis)
# ===========================================================================
#
# Trois séries mensuelles fournies dans data/external :
#   - hicp_all_items_yoy.csv : inflation headline, YoY (%)
#   - hicp_core_yoy.csv      : en réalité l'indice HICP core
#                              (total ex. énergie & alim. non transformée)
#   - hicp_energy_yoy.csv    : inflation énergie, YoY (%)
#
# On reconstruit un DataFrame mensuel homogène indexé par `month` (PeriodIndex,
# freq="M"), avec forward/backward fill pour couvrir :
#   - les production_year de used_market dès l'an 2000
#   - les dates de fin de contrat du portfolio jusqu'en 2030+
# Les dates futures au-delà de l'historique HICP observé (portfolio) sont
# couvertes par forward-fill (hypothèse : l'inflation la plus récemment
# observée est la meilleure estimation pour les mois à venir).

def load_hicp(path_dir: str = "data/external") -> pd.DataFrame:
    """
    Charge les 3 séries HICP Allemagne et retourne un DataFrame mensuel
    indexé par `month` (PeriodIndex freq="M"), aux colonnes :
      - hicp_headline_yoy  : inflation headline YoY (%)
      - hicp_core_index    : indice HICP core (base ~2015 = 100)
      - hicp_core_yoy      : inflation core YoY (%) dérivée de l'indice
      - hicp_energy_yoy    : inflation énergie YoY (%)

    L'index est étendu de 1995-01 à 2035-12 par ffill/bfill pour couvrir
    toutes les dates de vente et de production rencontrées dans used_market
    et le portfolio (y compris les fins de contrat projetées au-delà de
    l'historique observé).
    """
    import os

    def _load_one(fname: str) -> pd.DataFrame:
        d = pd.read_csv(os.path.join(path_dir, fname))
        d["month"] = pd.to_datetime(d["DATE"]).dt.to_period("M")
        hicp_col = [c for c in d.columns if "HICP" in c][0]
        return d[["month", hicp_col]].rename(columns={hicp_col: "value"})

    headline = _load_one("hicp_all_items_yoy.csv").rename(
        columns={"value": "hicp_headline_yoy"}
    )
    core = _load_one("hicp_core_yoy.csv").rename(
        columns={"value": "hicp_core_index"}
    )
    energy = _load_one("hicp_energy_yoy.csv").rename(
        columns={"value": "hicp_energy_yoy"}
    )

    out = (
        headline.set_index("month")
        .join(core.set_index("month"), how="outer")
        .join(energy.set_index("month"), how="outer")
        .sort_index()
    )
    out["hicp_core_yoy"] = out["hicp_core_index"].pct_change(12) * 100

    full_range = pd.period_range("1995-01", "2035-12", freq="M")
    out = out.reindex(full_range).ffill().bfill()
    out.index.name = "month"
    return out


def _attach_hicp_features(
    df: pd.DataFrame,
    sale_date_col: str,
    production_year_col: str = "production_year",
    hicp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Ajoute les features macro HICP au DataFrame :
      - log_cum_inflation_core : ln(core_index[sale] / core_index[production])
        → part du log_ratio imputable à l'inflation cumulée (effet nominal)
      - hicp_headline_yoy_sale : inflation headline YoY au mois de revente
        → régime macro contemporain (pression sur le marché de l'occasion)
      - hicp_energy_yoy_sale   : inflation énergie YoY au mois de revente
        → impact différencié sur les valorisations diesel / essence / EV

    Le mois de production est fixé au 1er juillet de `production_year`
    (ancre mi-année, médiane des mises en circulation).
    """
    if hicp is None:
        hicp = load_hicp()

    df = df.copy()
    sale_month = pd.to_datetime(df[sale_date_col]).dt.to_period("M")
    prod_month = pd.PeriodIndex(
        pd.to_datetime(
            df[production_year_col].astype(int).astype(str) + "-07-01"
        ),
        freq="M",
    )

    sale_df = hicp.reindex(sale_month.values).reset_index(drop=True)
    prod_df = hicp.reindex(prod_month).reset_index(drop=True)

    sale_df.index = df.index
    prod_df.index = df.index

    df["log_cum_inflation_core"] = np.log(
        sale_df["hicp_core_index"] / prod_df["hicp_core_index"]
    )
    df["hicp_headline_yoy_sale"] = sale_df["hicp_headline_yoy"].astype(float)
    df["hicp_energy_yoy_sale"] = sale_df["hicp_energy_yoy"].astype(float)

    for col in [
        "log_cum_inflation_core",
        "hicp_headline_yoy_sale",
        "hicp_energy_yoy_sale",
    ]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ===========================================================================
# 2. Préparation des features
# ===========================================================================


""" pourquoi la borne haute de 108 mois (9ans) pour l'âge ? Borne haute : 108 mois (9 ans). Elle est calibrée sur l'âge maximal atteint en fin de contrat dans portfolio.xlsx :

contract_duration max ≈ 61 mois (~5 ans),
initial_car_age max = 3 ans,
soit un âge maximal en fin de contrat d'environ 8 ans.
Une marge d'un an a été ajoutée (→ 9 ans) pour que le modèle dispose d'observations au-delà de la borne 
Borne basse : 12 mois (1 an). Les véhicules de moins d'un an présentent une dynamique de dépréciation 
atypique (forte décote initiale dite drive-off, effet "quasi-neuf") qui n'est pas représentative """

def prepare_used_market(
    df: pd.DataFrame,
    age_min_months: int = 12,
    age_max_months: int = 108,
    hicp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Filtre used_market sur le périmètre leasing (1-9 ans), construit les
    features communes avec portfolio et attache les features macro HICP.

    Target construite : log_ratio = ln(prix_vente / prix_catalogue).
    """
    df = _normalize_used_market_schema(df).copy()
    # robust date parsing: la colonne peut être soit déjà en datetime
    # (via load_used_market), soit en serial Excel (via pd.read_excel direct)
    if "date de vente" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["date de vente"]
    ):
        df["date de vente"] = pd.to_datetime(
            df["date de vente"], origin="1899-12-30", unit="D", errors="coerce"
        )
    df = df[(df["age"] >= age_min_months) & (df["age"] <= age_max_months)]
    df = df[df["prix de vente"] > 0]
    df = df[df["prix catalogue d'origine"] > 0]
    df = df[df["mileage"] > 0]

    df["age_months"] = df["age"].astype(float)
    df["mileage_km"] = df["mileage"].astype(float)
    df["prix_catalogue"] = df["prix catalogue d'origine"].astype(float)
    df["prix_vente"] = df["prix de vente"].astype(float)

    df["log_age"] = np.log(df["age_months"])
    df["log_mileage"] = np.log(df["mileage_km"])
    df["log_catalogue"] = np.log(df["prix_catalogue"])
    df[TARGET] = np.log(df["prix_vente"] / df["prix_catalogue"])

    df = df.reset_index(drop=True)
    df = _attach_hicp_features(
        df,
        sale_date_col="date de vente",
        production_year_col="production_year",
        hicp=hicp,
    )
    return df


def prepare_portfolio(
    df: pd.DataFrame,
    hicp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Reconstruit pour chaque véhicule du portfolio son état à la fin du
    contrat de leasing (moment de la revente), avec les mêmes unités et
    les mêmes noms de colonnes que used_market, et attache les features
    macro HICP (avec forward-fill pour les fins de contrat futures).
    """
    df = df.copy()

    df["age_months"] = (
        (df["contract_end_year"] - df["production_year"]) * 12
    ).clip(lower=1).astype(float)
    df["mileage_km"] = (
        df["initial_mileage"].fillna(0) + df["contract_mileage"].fillna(0)
    ).clip(lower=1).astype(float)
    df["prix_catalogue"] = df["prix catalogue d'origine"].astype(float)

    df["log_age"] = np.log(df["age_months"])
    df["log_mileage"] = np.log(df["mileage_km"])
    df["log_catalogue"] = np.log(df["prix_catalogue"])

    df = df.reset_index(drop=True)
    df = _attach_hicp_features(
        df,
        sale_date_col="current_contract_planned_end_date",
        production_year_col="production_year",
        hicp=hicp,
    )
    return df


# ===========================================================================
# 2bis. Features dérivées + comparable-stats (fit/transform, sans leakage)
# ===========================================================================
#
# Deux layers de features additionnelles, séparées pour garantir l'absence
# de leakage en cas de split temporel ou out-of-sample :
#
#   (a) derived ratio features — row-level pur, aucun fit nécessaire, donc
#       sûr à appliquer indifféremment sur train/test/portfolio.
#   (b) comparable-stats — résumé segment (brand × model × fuel_type) des
#       prix, mileage, age, ratio de revente. Doit impérativement être
#       FITTÉ sur df_train uniquement puis mergé sur df_test et portfolio.
#
# Usage type dans une évaluation temporelle :
#
#     df_train = add_derived_ratio_features(df_train)
#     df_test  = add_derived_ratio_features(df_test)
#     stats    = build_comparable_stats(df_train)           # fit on TRAIN
#     df_train = merge_comparable_features(df_train, stats)
#     df_test  = merge_comparable_features(df_test,  stats)

COMP_FEATURES = [
    "comp_median_price",
    "comp_count",
    "comp_price_iqr",
    "comp_median_mileage",
    "comp_median_age",
    "comp_median_ratio",
    "log_comp_median_price",
]

DERIVED_RATIO_FEATURES = [
    "annual_mileage",
    "mileage_per_month",
]


def add_derived_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées row-level. Aucun fit, aucun leakage possible."""
    df = df.copy()
    age_years = (df["age_months"] / 12).clip(lower=0.1)
    df["annual_mileage"] = df["mileage_km"] / age_years
    df["mileage_per_month"] = df["mileage_km"] / df["age_months"].clip(lower=1)
    return df


def build_comparable_stats(
    df_train: pd.DataFrame,
    group_cols: Iterable[str] = ("brand", "model", "fuel_type"),
) -> pd.DataFrame:
    """
    Statistiques segment (médiane/dispersion prix, km, âge, ratio de revente)
    calculées sur le train set. À merger ensuite sur test et portfolio via
    `merge_comparable_features` pour donner au modèle un prix-ancre local.
    """
    group_cols = list(group_cols)
    valid = df_train[
        (df_train["prix_vente"] > 0) & (df_train["prix_catalogue"] > 0)
    ].copy()

    agg = valid.groupby(group_cols, observed=True).agg(
        comp_median_price=("prix_vente", "median"),
        comp_count=("prix_vente", "count"),
        comp_price_q25=("prix_vente", lambda x: x.quantile(0.25)),
        comp_price_q75=("prix_vente", lambda x: x.quantile(0.75)),
        comp_median_mileage=("mileage_km", "median"),
        comp_median_age=("age_months", "median"),
    ).reset_index()
    agg["comp_price_iqr"] = agg["comp_price_q75"] - agg["comp_price_q25"]
    agg = agg.drop(columns=["comp_price_q25", "comp_price_q75"])

    ratio = valid.copy()
    ratio["_r"] = ratio["prix_vente"] / ratio["prix_catalogue"]
    ratio_stats = (
        ratio.groupby(group_cols, observed=True)["_r"].median().reset_index()
    )
    ratio_stats.columns = group_cols + ["comp_median_ratio"]
    agg = agg.merge(ratio_stats, on=group_cols, how="left")

    agg["log_comp_median_price"] = np.log(
        agg["comp_median_price"].clip(lower=1.0)
    )
    return agg


def merge_comparable_features(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    group_cols: Iterable[str] = ("brand", "model", "fuel_type"),
    fallback: dict | None = None,
) -> pd.DataFrame:
    """
    Merge les comp-stats sur `df` et remplit les segments absents avec la
    médiane globale (calculée à partir de `stats` lui-même pour rester
    déterministe et sans leakage).
    """
    group_cols = list(group_cols)
    df = df.merge(stats, on=group_cols, how="left")
    for col in COMP_FEATURES:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            if fallback and col in fallback:
                fill_val = fallback[col]
            else:
                fill_val = float(stats[col].median()) if col in stats.columns else 0.0
            df[col] = df[col].fillna(fill_val)
    return df


# ===========================================================================
# 2bis2. Recency weighting — combattre le concept drift au training
# ===========================================================================
#
# Motivation : quand on évalue sur split temporel (test 2024-2025), les
# transactions anciennes (2018-2020) tirent le modèle vers des régimes de
# marché obsolètes. Un poids décroissant exponentiel en fonction de l'âge
# de la vente par rapport à la date de référence ré-équilibre le training
# vers les époques proches du test.
#
# Formule : w_i = 0.5 ** (age_i / half_life), où age_i = ref_date - sale_i
# en jours. Le half_life contrôle la mémoire effective : 18 mois = moitié
# du poids à 18 mois, 25% à 36 mois, ~1.5% à 72 mois.
#
# Les poids sont normalisés pour que leur somme = n (comparable au cas
# sans poids, CatBoost ne change pas son régularisation relative).

def compute_recency_weights(
    df: pd.DataFrame,
    date_col: str = "date de vente",
    ref_date: pd.Timestamp | None = None,
    half_life_days: float = 548.0,  # ~18 mois
    normalize: bool = True,
) -> np.ndarray:
    """Poids exponentiels décroissants vers l'ancien.

    ref_date = max(date_col) par défaut (poids 1.0 pour la transaction la
    plus récente). half_life_days fixe la vitesse de décroissance."""
    dates = pd.to_datetime(df[date_col])
    if ref_date is None:
        ref_date = dates.max()
    age_days = (ref_date - dates).dt.total_seconds() / 86400.0
    age_days = age_days.clip(lower=0).to_numpy()
    w = 0.5 ** (age_days / float(half_life_days))
    if normalize:
        w = w * len(w) / w.sum()
    return w


# ===========================================================================
# 2ter. Domain adaptation — propensity-score importance weighting
# ===========================================================================
#
# Le problème : used_market (source) et portfolio (target) ne sont pas tirés
# de la même distribution. Used_market couvre un spectre large de véhicules
# tous âges/km, tandis que portfolio est concentré sur Renault/Dacia/Nissan
# récents en conditions de leasing spécifiques.
#
# Solution classique (Bickel & Scheffer, Sugiyama et al.) :
#   1. Entraîner un classifieur source vs target sur des features partagées.
#   2. Récupérer le score P(target | x) pour chaque obs source.
#   3. Convertir en poids w = p / (1 - p), clipé au q99 pour stabilité.
#   4. Entraîner le modèle principal avec ces poids : il voit davantage les
#      observations source qui ressemblent aux cars du portfolio.
#
# Coût : dégrade potentiellement la perf sur used_market test (on optimise
# pour une autre distribution). Gain : rapproche les prédictions portfolio
# de la vraie distribution cible (AS24 comme proxy). Protocol d'évaluation
# doit donc être DOUBLE : source-distrib + target-distrib.

def fit_propensity_weights(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    num_features: Iterable[str],
    cat_features: Iterable[str],
    clip_quantile: float = 0.99,
    normalize: bool = True,
    cv: int = 3,
) -> tuple[np.ndarray, dict]:
    """
    Calcule les poids d'importance pour les observations source via un
    classifieur logistique source (0) vs target (1).

    Retourne (weights_source, info_dict). info_dict contient l'AUC du
    classifieur (mesure de séparabilité source/target) et un résumé des
    poids (min/median/max/mean).

    Les features doivent exister dans les deux DataFrames. Les NaN sont
    remplis par la médiane (num) ou "UNKNOWN" (cat) avant fit.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.metrics import roc_auc_score

    num_features = [c for c in num_features if c in df_source.columns and c in df_target.columns]
    cat_features = [c for c in cat_features if c in df_source.columns and c in df_target.columns]
    if not num_features and not cat_features:
        raise ValueError("Aucune feature partagée entre source et target")

    cols = num_features + cat_features
    src = df_source[cols].copy()
    src["_label"] = 0
    tgt = df_target[cols].copy()
    tgt["_label"] = 1
    pooled = pd.concat([src, tgt], axis=0, ignore_index=True)

    for c in num_features:
        pooled[c] = pd.to_numeric(pooled[c], errors="coerce")
        pooled[c] = pooled[c].fillna(pooled[c].median())
    for c in cat_features:
        pooled[c] = pooled[c].astype(str).fillna("UNKNOWN")

    y = pooled.pop("_label").to_numpy()
    X = pooled

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             cat_features),
        ],
        remainder="drop",
    )
    pipe = SkPipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, C=1.0))])
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))

    source_mask = y == 0
    ps = np.clip(proba[source_mask], 1e-4, 1 - 1e-4)
    w = ps / (1 - ps)
    cap = float(np.quantile(w, clip_quantile))
    w = np.clip(w, 0.0, cap)
    if normalize:
        w = w * len(w) / w.sum()

    info = {
        "auc": round(auc, 4),
        "n_source": int(source_mask.sum()),
        "n_target": int((~source_mask).sum()),
        "weight_min": round(float(w.min()), 4),
        "weight_median": round(float(np.median(w)), 4),
        "weight_max": round(float(w.max()), 4),
        "weight_mean": round(float(w.mean()), 4),
        "clip_cap": round(cap, 4),
        "features_num": list(num_features),
        "features_cat": list(cat_features),
    }
    return w, info


# ===========================================================================
# 3. Clustering haute cardinalité -> familles de dépréciation (slide 14-16)
# ===========================================================================

@dataclass
class CategoricalClustering:
    """Mapping modalité -> famille, avec fallback pour modalités inconnues."""

    column: str
    mapping: dict
    default_family: str

    def transform(self, series: pd.Series) -> pd.Series:
        return (
            series.astype(str)
            .map(self.mapping)
            .fillna(self.default_family)
            .astype(str)
        )


def cluster_high_cardinality_categorical(
    df: pd.DataFrame,
    column: str,
    n_clusters: int = 6,
    min_obs_per_modality: int = 50,
    random_state: int = 42,
) -> CategoricalClustering:
    """
    Pour chaque modalité, ajuste une régression
        ln(V_t/V_0) = alpha + k1*log_age + k2*log_mileage
    et récupère le vecteur (alpha, k1, k2). KMeans sur ces vecteurs
    -> familles de dépréciation homogènes (slide 16 Nexialog).

    Les modalités avec trop peu d'observations sont agrégées dans la famille
    "rare" pour éviter des régressions instables.
    """
    needed = {column, "log_age", "log_mileage", TARGET}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour le clustering : {missing}")

    rows = []
    rare: list[str] = []

    for modality, grp in df.groupby(column, observed=True):
        if len(grp) < min_obs_per_modality:
            rare.append(str(modality))
            continue
        X = grp[["log_age", "log_mileage"]].to_numpy()
        y = grp[TARGET].to_numpy()
        reg = LinearRegression().fit(X, y)
        rows.append(
            {
                "modality": str(modality),
                "alpha": float(reg.intercept_),
                "k1": float(reg.coef_[0]),
                "k2": float(reg.coef_[1]),
                "n_obs": len(grp),
            }
        )

    params = pd.DataFrame(rows)
    if params.empty:
        mapping = {m: "family_rare" for m in rare}
        return CategoricalClustering(
            column=column, mapping=mapping, default_family="family_rare"
        )

    feats = StandardScaler().fit_transform(params[["alpha", "k1", "k2"]])
    k = min(n_clusters, len(params))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    params["family"] = [f"family_{c}" for c in kmeans.fit_predict(feats)]

    mapping = dict(zip(params["modality"], params["family"]))
    for m in rare:
        mapping[m] = "family_rare"

    return CategoricalClustering(
        column=column, mapping=mapping, default_family="family_rare"
    )


def build_modality_regression_params(
    df: pd.DataFrame,
    column: str,
    min_obs_per_modality: int = 50,
) -> pd.DataFrame:
    """
    Résume chaque modalité par les paramètres de sa régression locale :
    log_ratio ~ log_age + log_mileage.

    Renvoie un DataFrame avec les colonnes :
    [column, alpha, k1, k2, n_obs]
    """
    needed = {column, "log_age", "log_mileage", TARGET}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour l'analyse : {missing}")

    rows = []
    for modality, grp in df.groupby(column, observed=True):
        if len(grp) < min_obs_per_modality:
            continue
        X = grp[["log_age", "log_mileage"]].to_numpy()
        y = grp[TARGET].to_numpy()
        reg = LinearRegression().fit(X, y)
        rows.append(
            {
                column: str(modality),
                "alpha": float(reg.intercept_),
                "k1": float(reg.coef_[0]),
                "k2": float(reg.coef_[1]),
                "n_obs": len(grp),
            }
        )

    return pd.DataFrame(rows)


def evaluate_cluster_range(
    df: pd.DataFrame,
    column: str = "model",
    k_values: Iterable[int] = range(2, 11),
    min_obs_per_modality: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Évalue plusieurs nombres de clusters sur les vecteurs de paramètres
    (alpha, k1, k2) avec inertie et silhouette.
    """
    params = build_modality_regression_params(
        df=df,
        column=column,
        min_obs_per_modality=min_obs_per_modality,
    )
    if params.empty:
        raise ValueError("Aucune modalité suffisamment fréquente à clusteriser.")

    feats = params[["alpha", "k1", "k2"]].copy()
    X_scaled = StandardScaler().fit_transform(feats)

    rows = []
    max_k = len(params) - 1
    for k in k_values:
        if k < 2 or k > max_k:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        rows.append(
            {
                "k": int(k),
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(X_scaled, labels)),
            }
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("Aucune valeur de k valide pour calculer les métriques.")

    return {
        "params": params,
        "metrics": metrics.sort_values("k").reset_index(drop=True),
    }


def apply_clustering(
    df: pd.DataFrame, clustering: CategoricalClustering, new_col: str
) -> pd.DataFrame:
    df = df.copy()
    df[new_col] = clustering.transform(df[clustering.column])
    return df


# ===========================================================================
# 4. Entraînement des trois modèles
# ===========================================================================
#
# Chaque trainer reçoit un DataFrame déjà préparé (avec model_family déjà
# calculé) et renvoie un objet doté d'une méthode `.predict(df_features)`
# qui rend log_ratio. 

class _CatBoostAdapter:
    """Petit wrapper pour rendre CatBoost compatible avec une API
    `.predict(df)` qui prend un DataFrame de features brutes (cats en str)."""

    def __init__(self, model: CatBoostRegressor, cat_features, num_features):
        self.model = model
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)

    @property
    def feature_names(self):
        return self.cat_features + self.num_features

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].copy()
        for c in self.cat_features:
            X[c] = X[c].astype(str)
        return self.model.predict(X)


def _select_X_y(df: pd.DataFrame, cat_features, num_features, target):
    X = df[list(cat_features) + list(num_features)].copy()
    for c in cat_features:
        X[c] = X[c].astype(str)
    y = df[target].to_numpy()
    return X, y


def train_ridge(
    df_train: pd.DataFrame,
    cat_features: Iterable[str] = CAT_FEATURES,
    num_features: Iterable[str] = NUM_FEATURES,
    target: str = TARGET,
    alpha: float = 1.0,
) -> Pipeline:
    """Régression Ridge log-linéaire (modèle paramétrique de référence,
    slide 9 Nexialog). Cats encodées en one-hot, nums standardisées."""
    cat_features = list(cat_features)
    num_features = list(num_features)
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", StandardScaler(), num_features),
        ]
    )
    pipe = Pipeline([("prep", pre), ("reg", Ridge(alpha=alpha))])
    X, y = _select_X_y(df_train, cat_features, num_features, target)
    pipe.fit(X, y)
    return pipe


def train_xgboost(
    df_train: pd.DataFrame,
    cat_features: Iterable[str] = CAT_FEATURES,
    num_features: Iterable[str] = NUM_FEATURES,
    target: str = TARGET,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.08,
    random_state: int = 42,
    device: str | None = None,
) -> Pipeline:
    """XGBoost gradient boosting (gestion des non-linéarités et des
    interactions). Cats encodées en ordinal. `device="cuda"` active le GPU
    (auto-détecté si `device=None`)."""
    cat_features = list(cat_features)
    num_features = list(num_features)
    if device is None:
        device = "cuda" if gpu_available() else "cpu"
    pre = ColumnTransformer(
        [
            ("cat",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             cat_features),
            ("num", "passthrough", num_features),
        ]
    )
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        random_state=random_state,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", pre), ("reg", model)])
    X, y = _select_X_y(df_train, cat_features, num_features, target)
    pipe.fit(X, y)
    return pipe


def train_random_forest(
    df_train: pd.DataFrame,
    cat_features: Iterable[str] = CAT_FEATURES,
    num_features: Iterable[str] = NUM_FEATURES,
    target: str = TARGET,
    n_estimators: int = 300,
    max_depth: int | None = 18,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Random Forest sklearn (bagging d'arbres, variance non-paramétrique).
    Cats encodées en ordinal. Pas de GPU (sklearn CPU-only, mais n_jobs=-1
    parallélise sur tous les cœurs)."""
    cat_features = list(cat_features)
    num_features = list(num_features)
    pre = ColumnTransformer(
        [
            ("cat",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             cat_features),
            ("num", "passthrough", num_features),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", pre), ("reg", model)])
    X, y = _select_X_y(df_train, cat_features, num_features, target)
    pipe.fit(X, y)
    return pipe


def train_catboost(
    df_train: pd.DataFrame,
    cat_features: Iterable[str] = CAT_FEATURES,
    num_features: Iterable[str] = NUM_FEATURES,
    target: str = TARGET,
    iterations: int = 1200,
    learning_rate: float = 0.05,
    depth: int = 8,
    loss_function: str = "RMSE",
    random_seed: int = 42,
    sample_weight: np.ndarray | None = None,
    l2_leaf_reg: float | None = None,
    task_type: str | None = None,
    devices: str = "0",
) -> _CatBoostAdapter:
    """CatBoost (référence Nexialog, slide 13/18). `task_type=None` -> auto
    (GPU si dispo sinon CPU)."""
    if CatBoostRegressor is None:
        raise ImportError(
            "catboost n'est pas installé dans cet environnement."
        )
    cat_features = list(cat_features)
    num_features = list(num_features)
    if task_type is None:
        task_type = "GPU" if gpu_available() else "CPU"
    X, y = _select_X_y(df_train, cat_features, num_features, target)
    kwargs_cb = dict(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=loss_function,
        cat_features=cat_features,
        random_seed=random_seed,
        verbose=0,
        task_type=task_type,
    )
    if task_type == "GPU":
        kwargs_cb["devices"] = devices
    if l2_leaf_reg is not None:
        kwargs_cb["l2_leaf_reg"] = float(l2_leaf_reg)
    model = CatBoostRegressor(**kwargs_cb)
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        if len(w) != len(X):
            raise ValueError(
                f"sample_weight a {len(w)} éléments, attendu {len(X)}"
            )
        model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y)
    return _CatBoostAdapter(model, cat_features, num_features)


_TRAINERS = {
    "ridge": train_ridge,
    "xgboost": train_xgboost,
    "catboost": train_catboost,
    "random_forest": train_random_forest,
}

DEFAULT_MODELS = ("ridge", "xgboost", "catboost", "random_forest")


# ===========================================================================
# 4bis. Optimisation d'hyperparamètres (random search + CV stratifiée)
# ===========================================================================
#
# Ridge reste volontairement NON tuné : il joue le rôle de baseline
# paramétrique et nous sert de plancher de performance. Seuls les deux
# challengers (XGBoost et CatBoost) font l'objet d'une recherche
# d'hyperparamètres. La recherche est un random search sur une grille
# compacte, évaluée par K-fold CV stratifiée sur le décile de `prix_vente`,
# en optimisant directement la MAPE en euros (même métrique que le split
# test final). Budget réduit (~quelques minutes) pour rester compatible
# avec les délais du challenge.

XGB_SEARCH_SPACE = {
    "n_estimators": [300, 500, 700, 1000],
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
}

CATBOOST_SEARCH_SPACE = {
    "iterations": [800, 1200, 1500],
    "depth": [6, 7, 8],
    "learning_rate": [0.03, 0.05, 0.08],
}


def _cv_mape_eur(
    trainer,
    df: pd.DataFrame,
    params: dict,
    n_splits: int = 3,
    random_state: int = 42,
    num_features: Iterable[str] | None = None,
) -> float:
    """
    MAPE moyenne (%) en euros via KFold stratifiée sur le décile de
    `prix_vente`. `df` doit déjà contenir `model_family`.
    """
    bins = pd.qcut(df["prix_vente"], q=10, labels=False, duplicates="drop")
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    scores = []
    fit_kwargs = dict(params)
    if num_features is not None:
        fit_kwargs["num_features"] = list(num_features)
    for tr_idx, va_idx in skf.split(df, bins):
        df_tr = df.iloc[tr_idx]
        df_va = df.iloc[va_idx]
        model = trainer(df_tr, **fit_kwargs)
        scores.append(evaluate_model(model, df_va)["MAPE (%)"])
    return float(np.mean(scores))


def _sample_params(space: dict, rng: np.random.Generator) -> dict:
    out = {}
    for k, values in space.items():
        choice = rng.choice(values)
        out[k] = choice.item() if hasattr(choice, "item") else choice
    return out


def tune_hyperparams(
    trainer,
    df_train: pd.DataFrame,
    search_space: dict,
    n_trials: int = 10,
    n_splits: int = 3,
    random_state: int = 42,
    verbose: bool = False,
    num_features: Iterable[str] | None = None,
) -> dict:
    """
    Random search avec CV stratifiée sur le décile de prix. Retourne un
    dict `{"params": best_params, "cv_mape": best_score, "history": [...]}`.
    Les doublons d'échantillonnage sont écartés.
    """
    rng = np.random.default_rng(random_state)
    seen: set[tuple] = set()
    best_params: dict | None = None
    best_score = float("inf")
    history: list[dict] = []

    trials_done = 0
    attempts = 0
    max_attempts = n_trials * 10
    while trials_done < n_trials and attempts < max_attempts:
        attempts += 1
        params = _sample_params(search_space, rng)
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        trials_done += 1
        score = _cv_mape_eur(
            trainer, df_train, params,
            n_splits=n_splits, random_state=random_state,
            num_features=num_features,
        )
        history.append({**params, "cv_mape": score})
        if verbose:
            print(f"  trial {trials_done:02d}: {params} -> CV MAPE {score:.2f}%")
        if score < best_score:
            best_score = score
            best_params = params

    return {
        "params": best_params or {},
        "cv_mape": best_score,
        "history": pd.DataFrame(history).sort_values("cv_mape").reset_index(drop=True),
    }


def _cv_mape_eur_timeseries(
    trainer,
    df_train: pd.DataFrame,
    params: dict,
    n_splits: int = 3,
    num_features: Iterable[str] | None = None,
) -> float:
    """MAPE moyenne (%) en euros via TimeSeriesSplit (validation chronologique).

    `df_train` DOIT être trié par date. Chaque fold utilise les premiers
    indices pour le train et les suivants pour la validation, respectant
    l'ordre temporel — contrairement à KFold stratifié qui mélange.
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_index = np.arange(len(df_train))
    fit_kwargs = dict(params)
    if num_features is not None:
        fit_kwargs["num_features"] = list(num_features)
    scores = []
    for tr_idx, va_idx in tscv.split(X_index):
        df_tr = df_train.iloc[tr_idx]
        df_va = df_train.iloc[va_idx]
        model = trainer(df_tr, **fit_kwargs)
        scores.append(evaluate_model(model, df_va)["MAPE (%)"])
    return float(np.mean(scores))


def cv_mape_fold_values(
    trainer,
    df_train: pd.DataFrame,
    params: dict,
    n_splits: int = 3,
    num_features: Iterable[str] | None = None,
) -> list[float]:
    """Retourne la liste des MAPE par pli TimeSeriesSplit (pas juste la moyenne).

    Utile pour calculer moyenne ET écart-type sur les splits internes — mesure
    de stabilité intra-CV du modèle pour un jeu d'hyperparamètres donné.
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_index = np.arange(len(df_train))
    fit_kwargs = dict(params)
    if num_features is not None:
        fit_kwargs["num_features"] = list(num_features)
    scores: list[float] = []
    for tr_idx, va_idx in tscv.split(X_index):
        df_tr = df_train.iloc[tr_idx]
        df_va = df_train.iloc[va_idx]
        model = trainer(df_tr, **fit_kwargs)
        scores.append(float(evaluate_model(model, df_va)["MAPE (%)"]))
    return scores


# ===========================================================================
# 5. Évaluation : reconstruit V_t en euros et calcule MAE/MAPE/R²
# ===========================================================================

def predict_eur(model, df: pd.DataFrame, floor_eur: float = 500.0) -> np.ndarray:
    """V_t = V_0 * exp(log_ratio_hat), borné par `floor_eur`."""
    log_ratio_hat = model.predict(df)
    v0 = df["prix_catalogue"].to_numpy()
    return np.clip(v0 * np.exp(log_ratio_hat), floor_eur, None)


def evaluate_model(model, df_test: pd.DataFrame, model_name: str = "") -> dict:
    """Métriques sur le test set en EUROS (back-transformées depuis log_ratio)."""
    y_true = df_test["prix_vente"].to_numpy()
    y_pred = predict_eur(model, df_test)
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2 = r2_score(y_true, y_pred)
    return {
        "model": model_name,
        "MAE (€)": round(float(mae), 0),
        "MAPE (%)": round(mape, 2),
        "R²": round(float(r2), 4),
    }


# ===========================================================================
# 6. Orchestrateurs : comparaison + entraînement final
# ===========================================================================

@dataclass
class ModelComparison:
    metrics: pd.DataFrame
    metrics_logratio: pd.DataFrame
    models: dict
    clustering: CategoricalClustering
    comp_stats: pd.DataFrame
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    num_features_ext: list[str]
    naive: dict
    feature_importance: dict | None
    pred_vs_obs: dict
    residuals: dict
    meta: dict
    tuned_params: dict = field(default_factory=dict)
    tuning_history: dict = field(default_factory=dict)
    best_name: str = field(init=False)

    def __post_init__(self):
        meta_best = (self.meta or {}).get("best_model")
        if meta_best:
            self.best_name = meta_best
        else:
            model_only = self.metrics[self.metrics["model"] != "naive"]
            self.best_name = model_only.sort_values("MAPE (%)").iloc[0]["model"]

    def __getitem__(self, key: str):
        """Compatibilité avec les anciens notebooks."""
        return getattr(self, key)


def _log_ratio_metrics_row(model, df_test: pd.DataFrame, name: str) -> dict:
    y_true = df_test[TARGET].to_numpy()
    y_pred = model.predict(df_test)
    residuals = y_true - y_pred
    denom = np.where(y_true == 0, 1e-9, y_true)
    return {
        "model": name,
        "R² (log_ratio)": round(float(r2_score(y_true, y_pred)), 4),
        "MAE (log_ratio)": round(float(np.mean(np.abs(residuals))), 4),
        "RMSE (log_ratio)": round(float(np.sqrt(np.mean(residuals ** 2))), 4),
        "MAPE (log_ratio) %": round(
            float(np.mean(np.abs(residuals / denom)) * 100), 2
        ),
    }


def _naive_baseline(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Baseline V_t = k_MSE · V_0. OLS sans constante sur le train."""
    v0_tr = df_train["prix_catalogue"].to_numpy()
    y_tr = df_train["prix_vente"].to_numpy()
    k_mse = float(np.sum(v0_tr * y_tr) / np.sum(v0_tr ** 2))
    log_k = float(np.log(k_mse))

    v0_te = df_test["prix_catalogue"].to_numpy()
    y_te = df_test["prix_vente"].to_numpy()
    y_pred_eur = k_mse * v0_te
    eur = {
        "model": "naive",
        "MAE (€)": round(float(mean_absolute_error(y_te, y_pred_eur)), 0),
        "MAPE (%)": round(float(np.mean(np.abs((y_te - y_pred_eur) / y_te)) * 100), 2),
        "R²": round(float(r2_score(y_te, y_pred_eur)), 4),
    }

    y_lr_true = df_test[TARGET].to_numpy()
    y_lr_pred = np.full_like(y_lr_true, log_k)
    residuals = y_lr_true - y_lr_pred
    lr = {
        "model": "naive",
        "R² (log_ratio)": round(float(r2_score(y_lr_true, y_lr_pred)), 4),
        "MAE (log_ratio)": round(float(np.mean(np.abs(residuals))), 4),
        "RMSE (log_ratio)": round(float(np.sqrt(np.mean(residuals ** 2))), 4),
        "MAPE (log_ratio) %": round(
            float(np.mean(np.abs(residuals / np.where(y_lr_true == 0, 1e-9, y_lr_true))) * 100), 2
        ),
    }
    return {
        "k_mse": round(k_mse, 4),
        "log_k": round(log_k, 4),
        "formula": "V_t = k_MSE · V_0  (OLS sans constante sur le train post-COVID)",
        "eur": eur,
        "log_ratio": lr,
    }


def _extract_feature_importance(
    model, model_name: str, num_features_ext: list[str], top_k: int = 12
) -> dict | None:
    """Top-k features pondérées par importance (ou |coef| pour Ridge)."""
    try:
        if model_name == "catboost":
            m = model.model
            fi = np.asarray(m.get_feature_importance())
            feat_names = model.feature_names
        elif model_name == "xgboost":
            reg = model.named_steps["reg"]
            fi = np.asarray(reg.feature_importances_)
            feat_names = list(CAT_FEATURES) + list(num_features_ext)
        elif model_name == "random_forest":
            reg = model.named_steps["reg"]
            fi = np.asarray(reg.feature_importances_)
            feat_names = list(CAT_FEATURES) + list(num_features_ext)
        elif model_name == "ridge":
            reg = model.named_steps["reg"]
            fi = np.abs(np.asarray(reg.coef_))
            feat_names = [f"feat_{i}" for i in range(len(fi))]
        else:
            return None
        if len(fi) == 0 or len(fi) != len(feat_names):
            return None
        order = np.argsort(fi)[::-1][:top_k]
        return {
            "features": [str(feat_names[i]) for i in order],
            "importance": [round(float(fi[i]), 4) for i in order],
        }
    except Exception as exc:
        print(f"  [warn] feature importance {model_name}: {exc}")
        return None


def _pred_vs_obs_sample(
    model, df_test: pd.DataFrame,
    max_points: int = 5000, seed: int = RANDOM_SEED,
) -> tuple[dict, dict]:
    y_true = df_test["prix_vente"].to_numpy()
    y_pred = predict_eur(model, df_test)
    residuals = y_pred - y_true

    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n > max_points:
        idx = rng.choice(n, max_points, replace=False)
        y_t, y_p = y_true[idx], y_pred[idx]
    else:
        y_t, y_p = y_true, y_pred
    scatter = {"true": [float(v) for v in y_t], "pred": [float(v) for v in y_p]}

    clipped = np.clip(
        residuals, np.percentile(residuals, 0.5), np.percentile(residuals, 99.5)
    )
    hist, edges = np.histogram(clipped, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    hist_block = {
        "x": [round(float(v), 2) for v in centers],
        "y": [int(v) for v in hist],
        "mean": round(float(np.mean(residuals)), 2),
        "median": round(float(np.median(residuals)), 2),
        "std": round(float(np.std(residuals)), 2),
        "min": round(float(np.min(residuals)), 2),
        "max": round(float(np.max(residuals)), 2),
        "n": int(len(residuals)),
    }
    return scatter, hist_block


def _prepare_post_covid_split(
    used_market: pd.DataFrame,
    n_model_families: int = N_MODEL_FAMILIES_DEFAULT,
    train_start: pd.Timestamp = POST_COVID_TRAIN_START,
    cutoff: pd.Timestamp = POST_COVID_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame, CategoricalClustering, pd.DataFrame, list[str]]:
    """Retourne (df_train, df_test, clustering, comp_stats, num_features_ext)
    avec tout le feature engineering (clustering + ratios dérivés +
    comp_stats) fitté EXCLUSIVEMENT sur le train pour éviter la fuite."""
    df = prepare_used_market(used_market)
    df_train = df[(df["date de vente"] >= train_start) & (df["date de vente"] < cutoff)].copy()
    df_test = df[df["date de vente"] >= cutoff].copy()

    if df_train.empty or df_test.empty:
        raise ValueError(
            f"Split post-COVID vide : train={len(df_train)}, test={len(df_test)}"
        )

    clustering = cluster_high_cardinality_categorical(
        df_train, column="model", n_clusters=n_model_families
    )
    df_train = apply_clustering(df_train, clustering, new_col="model_family")
    df_test = apply_clustering(df_test, clustering, new_col="model_family")

    df_train = add_derived_ratio_features(df_train)
    df_test = add_derived_ratio_features(df_test)

    comp_stats = build_comparable_stats(
        df_train, group_cols=("brand", "model", "fuel_type")
    )
    df_train = merge_comparable_features(df_train, comp_stats)
    df_test = merge_comparable_features(df_test, comp_stats)

    num_features_ext = list(NUM_FEATURES) + DERIVED_RATIO_FEATURES + COMP_FEATURES
    return df_train, df_test, clustering, comp_stats, num_features_ext


TUNING_PARAMS_PATH = "tuning_params_postcovid.json"


def load_tuned_params(path: str = TUNING_PARAMS_PATH) -> dict:
    """Lit tuning_params_postcovid.json produit par optuna_tuning.py.

    Retourne un dict {nom_modele: {best_params, cv_mape_mean_tuned,
    cv_mape_std_tuned, ...}} — silencieusement vide si le fichier est absent.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_chosen_model(
    tuning: dict,
    candidates: Iterable[str] = ("catboost", "xgboost"),
) -> tuple[str, dict]:
    """Sélectionne le modèle retenu par critère de STABILITÉ.

    Départage sur `cv_mape_std_tuned` (min = plus stable sur les 3 folds
    TSSplit) parmi les challengers tunés. À égalité de std, on prend le
    MAPE moyen le plus faible. Retourne (nom, best_params).
    """
    pool = []
    for name in candidates:
        entry = tuning.get(name)
        if not entry or "best_params" not in entry:
            continue
        std = entry.get("cv_mape_std_tuned", float("inf"))
        mape = entry.get("cv_mape_mean_tuned", entry.get("best_cv_mape", float("inf")))
        pool.append((std, mape, name, entry["best_params"]))
    if not pool:
        raise RuntimeError(
            "Aucun modèle tuné exploitable — lance optuna_tuning.py d'abord."
        )
    pool.sort(key=lambda t: (t[0], t[1]))
    std, mape, name, params = pool[0]
    return name, dict(params)


def compare_models_on_used_market(
    used_market: pd.DataFrame,
    n_model_families: int = N_MODEL_FAMILIES_DEFAULT,
    models_to_run: Iterable[str] = DEFAULT_MODELS,
    train_start: pd.Timestamp = POST_COVID_TRAIN_START,
    cutoff: pd.Timestamp = POST_COVID_CUTOFF,
    random_state: int = RANDOM_SEED,
    tune: bool = False,
    xgb_trials: int = 10,
    catboost_trials: int = 6,
    cv_splits: int = 3,
    verbose_tuning: bool = False,
    chosen_override: tuple[str, dict] | None = None,
) -> ModelComparison:
    """
    Comparaison des modèles sur le SPLIT TEMPOREL POST-COVID (canonique) :
      - train : [2022-01-01, 2024-01-01)
      - test  : [2024-01-01, dernière transaction disponible]

    Feature engineering (clustering + derived ratios + comp_stats) fitté
    sur le train uniquement pour éviter la fuite temporelle. Les num_features
    étendues (NUM + DERIVED + COMP) sont passées à chaque trainer.

    Les 4 modèles par défaut (Ridge / XGBoost / CatBoost / RandomForest)
    sont entraînés, plus le baseline naïf V_t = k_MSE · V_0.
    Tous les artefacts nécessaires au dashboard sont produits ici :
    feature_importance du best, pred_vs_obs (5000 pts), histogramme des
    résidus, métriques log_ratio par modèle.
    """
    df_train, df_test, clustering, comp_stats, num_features_ext = (
        _prepare_post_covid_split(
            used_market,
            n_model_families=n_model_families,
            train_start=train_start,
            cutoff=cutoff,
        )
    )

    tuned_params: dict[str, dict] = {}
    tuning_history: dict[str, pd.DataFrame] = {}
    if tune:
        tuning_plan = [
            ("xgboost", train_xgboost, XGB_SEARCH_SPACE, xgb_trials),
            ("catboost", train_catboost, CATBOOST_SEARCH_SPACE, catboost_trials),
        ]
        for name, trainer, space, trials in tuning_plan:
            if name not in models_to_run:
                continue
            print(
                f"[tuning] {name} — {trials} trials × {cv_splits}-fold "
                f"CV stratifiée sur le décile de prix"
            )
            res = tune_hyperparams(
                trainer,
                df_train,
                space,
                n_trials=trials,
                n_splits=cv_splits,
                random_state=random_state,
                verbose=verbose_tuning,
                num_features=num_features_ext,
            )
            tuned_params[name] = res["params"]
            tuning_history[name] = res["history"]
            print(
                f"[tuning] {name} best: {res['params']} "
                f"(CV MAPE {res['cv_mape']:.2f}%)"
            )

    # Override : si un modèle "chosen" + ses HP sont fournis, on force son
    # entraînement avec ces HP (au lieu des valeurs par défaut) pour que les
    # artefacts (feature_importance, pred_vs_obs, résidus) correspondent au
    # modèle réellement retenu sur le dashboard.
    override_name: str | None = None
    override_params: dict = {}
    if chosen_override is not None:
        override_name, override_params = chosen_override
        override_params = dict(override_params)

    models: dict[str, object] = {}
    rows_eur: list[dict] = []
    rows_lr: list[dict] = []
    for name in models_to_run:
        if name not in _TRAINERS:
            raise ValueError(f"Modèle inconnu : {name}")
        if name == override_name:
            params = dict(override_params)
        else:
            params = dict(tuned_params.get(name, {}))
        models[name] = _TRAINERS[name](
            df_train, num_features=num_features_ext, **params
        )
        rows_eur.append(evaluate_model(models[name], df_test, model_name=name))
        rows_lr.append(_log_ratio_metrics_row(models[name], df_test, name))

    naive = _naive_baseline(df_train, df_test)
    rows_eur.append(naive["eur"])
    rows_lr.append(naive["log_ratio"])

    metrics_eur = (
        pd.DataFrame(rows_eur).sort_values("MAPE (%)").reset_index(drop=True)
    )
    metrics_lr = (
        pd.DataFrame(rows_lr)
        .sort_values("MAPE (log_ratio) %")
        .reset_index(drop=True)
    )

    # Critère de sélection du "chosen" :
    #  - si override fourni, on le respecte (sélection par stabilité en amont)
    #  - sinon, fallback historique : MAPE euros minimum sur le split
    model_only = metrics_eur[metrics_eur["model"] != "naive"]
    mape_best = model_only.sort_values("MAPE (%)").iloc[0]["model"]
    chosen_name = override_name if override_name in models else mape_best

    feat_imp = _extract_feature_importance(
        models[chosen_name], chosen_name, num_features_ext
    )
    scatter, residuals_block = _pred_vs_obs_sample(models[chosen_name], df_test)

    meta = {
        "protocol": "post_covid",
        "train_start": str(pd.Timestamp(train_start).date()),
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "train_range": [
            str(df_train["date de vente"].min().date()),
            str(df_train["date de vente"].max().date()),
        ],
        "test_range": [
            str(df_test["date de vente"].min().date()),
            str(df_test["date de vente"].max().date()),
        ],
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_model_families": int(n_model_families),
        "best_model": chosen_name,
        "mape_winner": mape_best,
        "chosen_source": "stability_override" if override_name else "mape_on_split",
        "chosen_params": override_params if override_name else {},
        "gpu": gpu_available(),
        "random_seed": int(random_state),
    }

    return ModelComparison(
        metrics=metrics_eur,
        metrics_logratio=metrics_lr,
        models=models,
        clustering=clustering,
        comp_stats=comp_stats,
        df_train=df_train,
        df_test=df_test,
        num_features_ext=num_features_ext,
        naive=naive,
        feature_importance=feat_imp,
        pred_vs_obs=scatter,
        residuals=residuals_block,
        meta=meta,
        tuned_params=tuned_params,
        tuning_history=tuning_history,
    )


def stress_test_temporal_4951_split(
    used_market: pd.DataFrame,
    n_model_families: int = N_MODEL_FAMILIES_DEFAULT,
    models_to_run: Iterable[str] = DEFAULT_MODELS,
    test_frac: float = 0.51,
    random_state: int = RANDOM_SEED,
    tuned_params: dict | None = None,
) -> dict:
    """Stress-test : split TEMPOREL 49/51 strict sur la fenêtre post-COVID
    (≥ 2022-01). On trie les transactions par date, on prend les 49%
    premiers comme train et les 51% derniers comme test (cutoff = quantile
    0.49 de `date de vente`). Chaque modèle est évalué avec ses HP retenus
    (si fournis). Le but : vérifier que la hiérarchie des modèles reste
    robuste quand le TRAIN est PLUS COURT QUE LE TEST (contrainte
    volontairement dure, vs. le split canonique qui a >80% du post-COVID
    en train) et qu'on étend l'horizon de généralisation sur une période
    post-COVID plus longue."""
    df = prepare_used_market(used_market)
    df = df[df["date de vente"] >= POST_COVID_TRAIN_START].copy()
    if df.empty:
        raise ValueError("Dataset post-COVID vide pour le stress test 49/51.")

    df = df.sort_values("date de vente").reset_index(drop=True)
    cutoff_idx = int(round(len(df) * (1 - test_frac)))
    if cutoff_idx <= 0 or cutoff_idx >= len(df):
        raise ValueError(
            f"test_frac={test_frac} impossible sur {len(df)} lignes."
        )
    cutoff_date = df.iloc[cutoff_idx]["date de vente"]
    df_train = df.iloc[:cutoff_idx].copy()
    df_test = df.iloc[cutoff_idx:].copy()

    clustering = cluster_high_cardinality_categorical(
        df_train, column="model", n_clusters=n_model_families
    )
    df_train = apply_clustering(df_train, clustering, new_col="model_family")
    df_test = apply_clustering(df_test, clustering, new_col="model_family")
    df_train = add_derived_ratio_features(df_train)
    df_test = add_derived_ratio_features(df_test)
    comp_stats = build_comparable_stats(
        df_train, group_cols=("brand", "model", "fuel_type")
    )
    df_train = merge_comparable_features(df_train, comp_stats)
    df_test = merge_comparable_features(df_test, comp_stats)

    num_features_ext = list(NUM_FEATURES) + DERIVED_RATIO_FEATURES + COMP_FEATURES
    tuned_params = tuned_params or {}

    rows_eur: list[dict] = []
    rows_lr: list[dict] = []
    for name in models_to_run:
        if name not in _TRAINERS:
            continue
        params = {}
        entry = tuned_params.get(name)
        if isinstance(entry, dict) and "best_params" in entry:
            params = dict(entry["best_params"])
        model = _TRAINERS[name](
            df_train, num_features=num_features_ext, **params
        )
        rows_eur.append(evaluate_model(model, df_test, model_name=name))
        rows_lr.append(_log_ratio_metrics_row(model, df_test, name))

    naive = _naive_baseline(df_train, df_test)
    rows_eur.append(naive["eur"])
    rows_lr.append(naive["log_ratio"])

    metrics_eur = (
        pd.DataFrame(rows_eur).sort_values("MAPE (%)").reset_index(drop=True)
    )
    metrics_lr = (
        pd.DataFrame(rows_lr)
        .sort_values("MAPE (log_ratio) %")
        .reset_index(drop=True)
    )

    return {
        "split": "temporal_49_51_postcovid",
        "train_frac": round(1 - test_frac, 2),
        "test_frac": round(test_frac, 2),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "post_covid_train_start": str(POST_COVID_TRAIN_START.date()),
        "cutoff_date": str(pd.Timestamp(cutoff_date).date()),
        "train_range": [
            str(df_train["date de vente"].min().date()),
            str(df_train["date de vente"].max().date()),
        ],
        "test_range": [
            str(df_test["date de vente"].min().date()),
            str(df_test["date de vente"].max().date()),
        ],
        "random_state": int(random_state),
        "metrics_eur": metrics_eur.to_dict("records"),
        "metrics_logratio": metrics_lr.to_dict("records"),
        "tuned_models": [
            name for name, entry in tuned_params.items()
            if isinstance(entry, dict) and "best_params" in entry
        ],
    }


def compare_cluster_choices(
    used_market: pd.DataFrame,
    k_values: Iterable[int] = (3, 4, 5, 6, 7, 8),
    models_to_run: Iterable[str] = DEFAULT_MODELS,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Compare plusieurs nombres de familles via la performance finale
    du pipeline sur used_market (split post-COVID).
    """
    rows = []
    for k in k_values:
        comparison = compare_models_on_used_market(
            used_market=used_market,
            n_model_families=int(k),
            random_state=random_state,
            models_to_run=models_to_run,
        )
        model_only = comparison.metrics[comparison.metrics["model"] != "naive"]
        best = model_only.sort_values("MAPE (%)").iloc[0].to_dict()
        rows.append(
            {
                "k": int(k),
                "best_model": comparison.best_name,
                "MAE (€)": best["MAE (€)"],
                "MAPE (%)": best["MAPE (%)"],
                "R²": best["R²"],
            }
        )

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


# ===========================================================================
# 7. Re-entraînement final sur la totalité de used_market
# ===========================================================================

@dataclass
class VRPipeline:
    model: object
    model_name: str
    clustering: CategoricalClustering
    comp_stats: pd.DataFrame
    num_features_ext: list[str]

    @property
    def model_clustering(self) -> CategoricalClustering:
        """Alias conservé pour les notebooks existants."""
        return self.clustering


def fit_final_pipeline(
    used_market: pd.DataFrame,
    model_name: str,
    n_model_families: int = N_MODEL_FAMILIES_DEFAULT,
    train_start: pd.Timestamp = POST_COVID_TRAIN_START,
    params: dict | None = None,
) -> VRPipeline:
    """
    Ré-entraîne le meilleur modèle sur la fenêtre post-COVID étendue
    jusqu'à la dernière transaction disponible (train_start -> max(date)),
    pour maximiser la quantité d'information récente injectée tout en
    restant homogène avec le régime du portfolio. Le clustering et les
    comp_stats sont refittés sur cette même fenêtre.

    `params` : hyperparamètres tunés à passer au trainer. Si None, le
    trainer utilise ses valeurs par défaut.
    """
    if model_name not in _TRAINERS:
        raise ValueError(f"Modèle inconnu : {model_name}")

    df = prepare_used_market(used_market)
    df = df[df["date de vente"] >= train_start].copy()
    if df.empty:
        raise ValueError(f"Aucune transaction après {train_start.date()}")

    clustering = cluster_high_cardinality_categorical(
        df, column="model", n_clusters=n_model_families
    )
    df = apply_clustering(df, clustering, new_col="model_family")
    df = add_derived_ratio_features(df)
    comp_stats = build_comparable_stats(df, group_cols=("brand", "model", "fuel_type"))
    df = merge_comparable_features(df, comp_stats)

    num_features_ext = list(NUM_FEATURES) + DERIVED_RATIO_FEATURES + COMP_FEATURES
    model = _TRAINERS[model_name](
        df, num_features=num_features_ext, **(params or {})
    )
    return VRPipeline(
        model=model,
        model_name=model_name,
        clustering=clustering,
        comp_stats=comp_stats,
        num_features_ext=num_features_ext,
    )


def tune_and_fit_pipeline(
    used_market: pd.DataFrame,
    n_model_families: int = N_MODEL_FAMILIES_DEFAULT,
    models_to_run: Iterable[str] = DEFAULT_MODELS,
    xgb_trials: int = 10,
    catboost_trials: int = 6,
    cv_splits: int = 3,
    random_state: int = RANDOM_SEED,
    verbose_tuning: bool = False,
) -> tuple["VRPipeline", ModelComparison]:
    """
    Flux complet : sélection du meilleur modèle avec tuning des challengers
    (XGBoost / CatBoost), puis ré-entraînement du gagnant sur la fenêtre
    post-COVID avec ses meilleurs hyperparamètres.
    """
    comparison = compare_models_on_used_market(
        used_market=used_market,
        n_model_families=n_model_families,
        models_to_run=models_to_run,
        tune=True,
        xgb_trials=xgb_trials,
        catboost_trials=catboost_trials,
        cv_splits=cv_splits,
        random_state=random_state,
        verbose_tuning=verbose_tuning,
    )
    best_params = comparison.tuned_params.get(comparison.best_name, {})
    print(
        f"[final] ré-entraînement {comparison.best_name} sur fenêtre "
        f"post-COVID avec params={best_params or '{defaults}'}"
    )
    pipeline = fit_final_pipeline(
        used_market=used_market,
        model_name=comparison.best_name,
        n_model_families=n_model_families,
        params=best_params,
    )
    return pipeline, comparison


# ===========================================================================
# 8. Application au portfolio (aucune métrique : pas de target)
# ===========================================================================

def predict_portfolio(
    pipeline: VRPipeline, portfolio: pd.DataFrame
) -> pd.DataFrame:
    """
    Applique le pipeline final à portfolio :
      1. reconstruit les features à la date de fin de contrat
      2. mappe chaque modèle vers sa famille de dépréciation
      3. attache derived ratios + comp_stats (fittés au training)
      4. prédit log_ratio puis reconstruit V_t en euros

    Renvoie un DataFrame avec id, features clés, model_family,
    `prediction` (VR en euros) et `decote_pct`.
    """
    df = prepare_portfolio(portfolio)
    df = apply_clustering(df, pipeline.clustering, new_col="model_family")
    df = add_derived_ratio_features(df)
    if pipeline.comp_stats is not None:
        df = merge_comparable_features(df, pipeline.comp_stats)
    df["prediction"] = predict_eur(pipeline.model, df)
    df["decote_pct"] = (1 - df["prediction"] / df["prix_catalogue"]) * 100
    return df
