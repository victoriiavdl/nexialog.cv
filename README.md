# Challenge Nexialog — Valorisation de portefeuille automobile

Dashboard Flask + Plotly pour la prédiction de valeur résiduelle automobile (portefeuille client), avec assistant IA (Ollama / llama3.2) entièrement containerisé.

> Projet réalisé dans le cadre du Master 2 [MoSEF](https://mosefparis1.com/) (Université Paris 1 Panthéon-Sorbonne), en partenariat avec [Nexialog Consulting](https://www.nexialog.com/) et un partenaire industriel.

---

## Problématique

Développer un outil de **stress test** pour estimer la valorisation d'un portefeuille de véhicules en leasing, en intégrant des scénarios macroéconomiques (prix du pétrole, inflation, évolution du marché de l'occasion).

---

## Méthodologie

| Étape | Description |
|---|---|
| **Feature engineering** | Construction de variables à partir de données marché (AS24, indices HICP, cours du Brent) et caractéristiques véhicules |
| **Benchmark de modèles** | Comparaison CatBoost, XGBoost, Random Forest — tuning Optuna, validation croisée |
| **Validation externe** | Scraping de prix AS24, prédiction conforme, analyse des résidus par segment |
| **Analyses complémentaires** | Impact du Brent sur les décotes, régimes COVID/post-COVID, rolling origin |

---

## Stack

| Composant | Technologies |
|---|---|
| **Backend** | Python · Flask |
| **ML** | CatBoost · XGBoost · Scikit-learn · Optuna |
| **Visualisation** | Plotly · Jinja2 templates |
| **Assistant IA** | Ollama (llama3.2) |
| **Infra** | Docker Compose |

---

## Lancement

```bash
git clone https://github.com/victoriiavdl/projet_nexialog_mobilize.git
cd projet_nexialog_mobilize
docker compose up --build
```

Ouvrir : [http://localhost:5000](http://localhost:5000)

> **Premier lancement** : compter 5-10 minutes — Docker build l'image Flask et télécharge le modèle Ollama (~2 Go). Les lancements suivants prennent ~1 minute (tout est en cache).

### Arrêter

```bash
docker compose down
```

### Sans Docker (dev)

```bash
poetry install
poetry run python app.py
```

Installer [Ollama](https://ollama.com/download) séparément pour l'assistant IA :

```bash
ollama pull llama3.2:latest
ollama serve
```

---

## Confidentialité

Les données du partenaire industriel sont confidentielles et **ne sont pas incluses dans ce dépôt**. Seul le code applicatif (dashboard, pipeline de prédiction, templates) est partagé. Les fichiers de données (CSV, notebooks, résultats JSON, slides) ont été retirés du suivi git.

> Les données restent dans l'historique git. Pour un nettoyage complet, il faudrait recréer le repo sans historique ou utiliser `git filter-repo`.

---

## Équipe

Projet réalisé par **Victoria Vidal** et son équipe — promotion MoSEF 2025-2026.

---
