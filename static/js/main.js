/* ============================================================
   NEXIALOG VR DASHBOARD — JavaScript
   Chart rendering (Plotly), navigation, simulation, i18n
   Branding: Blue #3b82f6 / Teal #4BC4BD / Nexialog-inspired
   ============================================================ */

// ---- Plotly dark theme (Nexialog palette) ----
const PLOTLY_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 12 },
    margin: { t: 30, r: 20, b: 50, l: 60 },
    xaxis: {
        gridcolor: 'rgba(148,163,184,0.08)',
        zerolinecolor: 'rgba(148,163,184,0.15)',
    },
    yaxis: {
        gridcolor: 'rgba(148,163,184,0.08)',
        zerolinecolor: 'rgba(148,163,184,0.15)',
    },
    hoverlabel: { bgcolor: '#1e293b', bordercolor: '#3b82f6', font: { size: 13 } },
    colorway: ['#3b82f6', '#4BC4BD', '#f59e0b', '#8b5cf6', '#ec4899', '#002c58', '#06b6d4', '#f97316'],
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false,
};

function mergeLayout(custom) {
    const layout = JSON.parse(JSON.stringify(PLOTLY_LAYOUT));
    for (const [key, val] of Object.entries(custom)) {
        if (typeof val === 'object' && !Array.isArray(val) && layout[key]) {
            layout[key] = { ...layout[key], ...val };
        } else {
            layout[key] = val;
        }
    }
    return layout;
}

function fmt(n) {
    if (n == null) return '--';
    return n.toLocaleString('fr-FR');
}

const MODEL_DISPLAY_NAMES = {
    naive: 'Naïf (k·V₀)',
    ridge: 'Ridge',
    xgboost: 'XGBoost',
    catboost: 'CatBoost',
    random_forest: 'Random Forest',
};
// Palette partagée par les renders post-COVID (stress test, walk-forward, etc.)
const MODEL_COLOR_MAP = {
    naive: '#64748b',
    ridge: '#94a3b8',
    xgboost: '#f59e0b',
    catboost: '#10b981',
    random_forest: '#8b5cf6',
};
function modelDisplayName(m) {
    if (m == null) return '';
    return MODEL_DISPLAY_NAMES[m] || (m.charAt(0).toUpperCase() + m.slice(1));
}

// ---- Data fetching ----
let DATA = null;

async function fetchData() {
    const resp = await fetch('/api/data?v=' + Date.now(), { cache: 'no-store' });
    DATA = await resp.json();
    return DATA;
}

// ---- Initialization ----
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await fetchData();
        populateHero();
        setupExecutiveSummary();
        renderAllCharts();
        setupTabs();
        setupNavigation();
        setupSimulator();
        setupScrollAnimations();
        setupLanguageSwitcher();
        setupMobileNav();
        setupReadingProgress();
        setupBackToTop();
        setupCollapsibles();
        setupRagDrawer();
        setupTour();
        setupNavPreview();
        setupQrShare();
    } catch (e) {
        console.error('Erreur initialisation:', e);
    } finally {
        document.getElementById('loading-screen').classList.add('hidden');
    }
});

// ---- Helpers ----
function buildModelsTable() {
    const mbb = DATA.options.models_by_brand;
    if (!mbb || Object.keys(mbb).length === 0) return '';
    let html = '<table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:0.78rem">';
    for (const [brand, models] of Object.entries(mbb)) {
        html += `<tr><td style="color:#60a5fa;font-weight:600;padding:4px 8px 4px 0;vertical-align:top;white-space:nowrap">${brand}</td>`;
        html += `<td style="padding:4px 0;color:#94a3b8">${models.join(', ')}</td></tr>`;
    }
    html += '</table>';
    return html;
}

// ---- Hero stats ----
function populateHero() {
    const m = DATA.meta;
    const setText = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };
    setText('stat-transactions', fmt(m.n_transactions));
    setText('stat-portfolio', fmt(m.n_portfolio));
    setText('stat-mape', m.best_mape + '%');
    setText('stat-r2', (m.best_r2 * 100).toFixed(2) + '%');
    setText('dyn-n-portfolio-risk', fmt(m.n_portfolio));
    setText('dyn-n-portfolio-tip2', fmt(m.n_portfolio));
    // Honest-scale metrics (log_ratio) — only filled if the secondary tier
    // is rendered. The simplified hero exposes them via the executive summary.
    if (m.best_mape_logratio != null) setText('stat-mape-logratio', m.best_mape_logratio + '%');
    if (m.best_r2_logratio != null) setText('stat-r2-logratio', (m.best_r2_logratio * 100).toFixed(2) + '%');

    // Data summary
    const summary = document.getElementById('data-summary');
    if (summary) {
        summary.innerHTML = `
            <div class="summary-item">
                <div class="summary-value">${fmt(m.n_transactions)}</div>
                <div class="summary-label">Transactions filtrées</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">${m.n_brands}</div>
                <div class="summary-label">Marques <span class="tip">i<span class="tip-text"><strong>Marques analysées</strong>Marques du groupe commercialisées sur le marché cible.</span></span></div>
            </div>
            <div class="summary-item">
                <div class="summary-value">${m.n_models_vehicle}</div>
                <div class="summary-label">Modèles <span class="tip tip-wide">i<span class="tip-text"><strong>Modèles par marque</strong>${buildModelsTable()}</span></span></div>
            </div>
            <div class="summary-item">
                <div class="summary-value">${m.age_range[0]}-${m.age_range[1]} mois</div>
                <div class="summary-label">Périmètre d'âge <span class="tip">i<span class="tip-text"><strong>Pourquoi 12-108 mois ?</strong>On ne retient que les véhicules de 1 à 9 ans. En dessous : véhicules quasi-neufs (forte décote initiale atypique). Au-dessus : risque de véhicules de collection, dépréciation non représentative du leasing.</span></span></div>
            </div>
        `;
    }
}

// ---- Executive summary (hero) — fill the 5 atomic spans inside the 3 bullets ----
function setupExecutiveSummary() {
    if (!DATA || !DATA.meta) return;
    const m = DATA.meta;

    const setText = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    // Detect ensemble (top 2 within 0.5 pp on EUR MAPE, mirrors metrics-table logic).
    const sorted = [...(m.metrics || [])]
        .filter(r => r.model !== 'naive')
        .sort((a, b) => a['MAPE (%)'] - b['MAPE (%)']);
    const isEnsemble = sorted.length >= 2
        && (sorted[1]['MAPE (%)'] - sorted[0]['MAPE (%)']) < 0.5;
    const modelLabel = isEnsemble
        ? `${modelDisplayName(sorted[0].model)} + ${modelDisplayName(sorted[1].model)}`
        : modelDisplayName(m.best_model);

    setText('exec-n-transactions', fmt(m.n_transactions));
    setText('exec-best-model', modelLabel);
    if (m.best_mape != null) setText('exec-mape-eur', m.best_mape + ' %');
    if (m.best_mape_logratio != null) setText('exec-mape-lr', m.best_mape_logratio + ' %');

    // Exposition portefeuille : moyenne prédite × n_portfolio, arrondi en M€.
    if (m.avg_prediction != null && m.n_portfolio != null) {
        const totalEur = m.avg_prediction * m.n_portfolio;
        const valM = (totalEur / 1e6);
        const fmtVal = valM >= 10 ? Math.round(valM) : valM.toFixed(1);
        setText('exec-exposition', `~${fmtVal} M€`);
    }
}

// ---- Reading progress bar ----
function setupReadingProgress() {
    const el = document.getElementById('reading-progress');
    if (!el) return;
    let ticking = false;
    const update = () => {
        const max = document.documentElement.scrollHeight - window.innerHeight;
        const r = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
        el.style.setProperty('--progress', r.toFixed(3));
        ticking = false;
    };
    const onScroll = () => {
        if (!ticking) { requestAnimationFrame(update); ticking = true; }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    update();
}

// ---- Collapse cards (progressive disclosure for deep theory) ----
// Classes qui ouvrent en modale centrée plutôt qu'en expansion in-place.
const MODAL_COLLAPSE_CLASSES = ['eda-bubble', 'conclusion-collapse'];
const MODAL_TRIGGER_SELECTOR = MODAL_COLLAPSE_CLASSES
    .map(c => `.${c} > .collapse-card__trigger[aria-expanded="true"]`)
    .join(', ');

function _isModalCollapse(card) {
    return !!card && MODAL_COLLAPSE_CLASSES.some(c => card.classList.contains(c));
}

function _ensureModalBackdrop() {
    let bd = document.getElementById('modal-backdrop');
    if (!bd) {
        bd = document.createElement('div');
        bd.id = 'modal-backdrop';
        bd.className = 'modal-backdrop';
        bd.addEventListener('click', () => {
            const openTrig = document.querySelector(MODAL_TRIGGER_SELECTOR);
            if (openTrig) _closeCollapse(openTrig);
        });
        document.body.appendChild(bd);
    }
    return bd;
}

// Flèches de navigation ◀ ▶ fixées en bas-droite quand une modale .eda-bubble
// est ouverte. Next sur la dernière bulle d'une section → passe à la section
// suivante (et vice versa pour prev sur la première).
function _ensureModalNav() {
    let nav = document.getElementById('modal-nav');
    if (!nav) {
        nav = document.createElement('div');
        nav.id = 'modal-nav';
        nav.className = 'modal-nav-group';
        nav.innerHTML = `
            <button type="button" class="modal-nav-btn modal-nav-btn--prev" aria-label="Précédent">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>
            </button>
            <button type="button" class="modal-nav-btn modal-nav-btn--next" aria-label="Suivant">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>
            </button>
        `;
        nav.querySelector('.modal-nav-btn--prev').addEventListener('click', () => _modalNavStep(-1));
        nav.querySelector('.modal-nav-btn--next').addEventListener('click', () => _modalNavStep(1));
        document.body.appendChild(nav);
    }
    return nav;
}
function _switchModal(currentTrig, newTrig) {
    currentTrig.setAttribute('aria-expanded', 'false');
    const curBody = document.getElementById(currentTrig.getAttribute('aria-controls'));
    if (curBody) curBody.setAttribute('hidden', '');
    _openCollapse(newTrig);
}
// Liste ordonnée des "stops" modaux d'une section : d'abord les bulles .eda-bubble
// puis, s'il existe, l'encadré conclusion de niveau section (.conclusion-collapse
// qui n'est pas imbriqué dans une autre modale).
function _sectionStops(section) {
    if (!section) return [];
    const stops = [];
    section.querySelectorAll('.eda-bubbles > .eda-bubble > .collapse-card__trigger').forEach(t => {
        if (section.contains(t)) stops.push(t);
    });
    section.querySelectorAll('.conclusion-collapse').forEach(c => {
        if (!section.contains(c)) return;
        if (c.parentElement && c.parentElement.closest('.collapse-card__body')) return;
        const t = c.querySelector(':scope > .collapse-card__trigger');
        if (t) stops.push(t);
    });
    return stops;
}
function _modalNavStep(direction) {
    const currentTrig = document.querySelector(MODAL_TRIGGER_SELECTOR);
    if (!currentTrig) return;
    const section = currentTrig.closest('section.section');
    const stops = _sectionStops(section);
    const idx = stops.indexOf(currentTrig);
    if (idx === -1) return;
    const newIdx = idx + direction;

    if (newIdx >= 0 && newIdx < stops.length) {
        _switchModal(currentTrig, stops[newIdx]);
    } else {
        // Bord de section : on ferme la modale et on va vers la section voisine (hors hero)
        _closeCollapse(currentTrig);
        if (!section) return;
        const sections = _navSections();
        const sIdx = sections.indexOf(section);
        if (sIdx === -1) return;
        const targetIdx = sIdx + direction;
        if (targetIdx >= 0 && targetIdx < sections.length) {
            setTimeout(() => {
                sections[targetIdx].scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 120);
        }
    }
}
function _navSections() {
    // Liste des sections « contenu » (on exclut le hero/landing qui n'a pas de bulles).
    return Array.from(document.querySelectorAll('section.section'))
        .filter(s => !s.classList.contains('hero-section'));
}
function _computeNavLabel(stops, idx, direction, trigger) {
    const newIdx = idx + direction;
    if (newIdx >= 0 && newIdx < stops.length) {
        const t = stops[newIdx].querySelector('.collapse-card__title > span');
        return t ? t.textContent.trim() : '';
    }
    // Bord de section : cible la section voisine (hors hero).
    const section = trigger.closest('section.section');
    if (!section) return '';
    const sections = _navSections();
    const sIdx = sections.indexOf(section);
    if (sIdx === -1) return '';
    const targetIdx = sIdx + direction;
    if (targetIdx < 0 || targetIdx >= sections.length) return '';
    const h2 = sections[targetIdx].querySelector('.section-header h2');
    const title = h2 ? h2.textContent.trim() : (sections[targetIdx].id || '');
    return `→ Section : ${title}`;
}
function _updateModalNav(trigger) {
    const nav = _ensureModalNav();
    const section = trigger && trigger.closest('section.section');
    const stops = _sectionStops(section);
    const idx = stops.indexOf(trigger);
    if (idx === -1) {
        nav.classList.remove('active');
        return;
    }
    const prevBtn = nav.querySelector('.modal-nav-btn--prev');
    const nextBtn = nav.querySelector('.modal-nav-btn--next');
    const prevLabel = _computeNavLabel(stops, idx, -1, trigger);
    const nextLabel = _computeNavLabel(stops, idx, 1, trigger);
    if (prevLabel) {
        prevBtn.setAttribute('data-tooltip', prevLabel);
        prevBtn.removeAttribute('disabled');
    } else {
        prevBtn.removeAttribute('data-tooltip');
        prevBtn.setAttribute('disabled', '');
    }
    if (nextLabel) {
        nextBtn.setAttribute('data-tooltip', nextLabel);
        nextBtn.removeAttribute('disabled');
    } else {
        nextBtn.removeAttribute('data-tooltip');
        nextBtn.setAttribute('disabled', '');
    }
    nav.classList.add('active');
}
function _hideModalNav() {
    const nav = document.getElementById('modal-nav');
    if (nav) nav.classList.remove('active');
}

function _openCollapse(trigger, { scrollIntoView = false } = {}) {
    const bodyId = trigger.getAttribute('aria-controls');
    const body = bodyId ? document.getElementById(bodyId) : null;
    trigger.setAttribute('aria-expanded', 'true');
    if (!body) return;
    body.removeAttribute('hidden');

    const card = trigger.closest('.collapse-card');
    const isModal = _isModalCollapse(card);
    if (isModal) {
        _ensureModalBackdrop().classList.add('active');
        document.body.classList.add('modal-open');
        _updateModalNav(trigger);
    }

    setTimeout(() => {
        body.querySelectorAll('.chart-container, [id^="chart-"]').forEach(c => {
            try { Plotly.Plots.resize(c); } catch (e) {}
        });
        // En mode modale on NE scroll PAS vers le trigger (la modale est au centre).
        if (scrollIntoView && !isModal) {
            trigger.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 60);
}
function _closeCollapse(trigger) {
    const bodyId = trigger.getAttribute('aria-controls');
    const body = bodyId ? document.getElementById(bodyId) : null;
    trigger.setAttribute('aria-expanded', 'false');
    if (body) body.setAttribute('hidden', '');

    const card = trigger.closest('.collapse-card');
    if (_isModalCollapse(card)) {
        const bd = document.getElementById('modal-backdrop');
        if (bd) bd.classList.remove('active');
        document.body.classList.remove('modal-open');
        _hideModalNav();
    }
}
function setupCollapsibles() {
    document.querySelectorAll('.collapse-card__trigger').forEach(trigger => {
        trigger.addEventListener('click', () => {
            const expanded = trigger.getAttribute('aria-expanded') === 'true';
            if (expanded) _closeCollapse(trigger);
            else _openCollapse(trigger);
        });
    });

    // Auto-open via URL hash (e.g. #coll-hicp-brent-body shared link).
    const hash = window.location.hash;
    if (hash && hash.length > 1) {
        const id = hash.slice(1);
        const trigger = document.querySelector(`.collapse-card__trigger[aria-controls="${id}"]`);
        if (trigger) _openCollapse(trigger, { scrollIntoView: true });
    }

    // ESC closes the focused collapse (when focus is inside the body or on the trigger).
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        // Priorité : si une modale est ouverte, ESC la ferme peu importe le focus.
        const modalTrig = document.querySelector(MODAL_TRIGGER_SELECTOR);
        if (modalTrig) {
            _closeCollapse(modalTrig);
            modalTrig.focus();
            return;
        }
        const active = document.activeElement;
        if (!active) return;
        // Either active is a trigger, or active is inside a body.
        const trigger = active.closest('.collapse-card__trigger')
            || active.closest('.collapse-card__body')?.previousElementSibling;
        if (trigger && trigger.classList.contains('collapse-card__trigger')
            && trigger.getAttribute('aria-expanded') === 'true') {
            _closeCollapse(trigger);
            trigger.focus();
        }
    });
}

// ============================================================
// MODE PRÉSENTATION JURY — guided tour (6 steps)
// ============================================================

const TOUR_STEPS = [
    {
        targetId: 'hero',
        openCollapseId: null,
        title: 'Accueil — le résultat en 1 phrase',
        text: 'Modèle ML <strong>{{CHOSEN_MODEL}}</strong> prédisant la valeur résiduelle de <strong>véhicules du portefeuille</strong> avec <strong>~7&nbsp;% de MAPE</strong> (EUR) sur un <strong>split temporel post-COVID</strong> — et un MAPE <em>log_ratio</em> ~11&nbsp;% audit-proof. Sélection par <strong>critère de stabilité</strong> (std du MAPE intra-CV), pas par la moyenne.',
    },
    {
        targetId: 'context',
        openCollapseId: null,
        title: '01 · Contexte métier',
        text: 'La VR (valeur résiduelle) pèse <strong>40-60&nbsp;%</strong> du prix catalogue d\'un véhicule en leasing. Une erreur de quelques points se traduit en <strong>millions d\'euros</strong> à l\'échelle du portefeuille. Enjeu : provisionner au plus juste, ni trop, ni trop peu.',
    },
    {
        targetId: 'eda',
        openCollapseId: null,
        title: '02 · Analyse exploratoire',
        text: 'Dataset used-car : <strong>transactions</strong> réelles du marché de l'occasion. Distributions, catégorielles, matrice de corrélation. La <strong>décote est convexe</strong> avec l\'âge — conforme à la littérature automobile et aux slides Nexialog.',
    },
    {
        targetId: 'hicp',
        openCollapseId: null,
        title: '03 · Macro (HICP)',
        text: 'Innovation macroéconomique : trois features <strong>HICP</strong> (inflation headline, core, énergie) intégrées via Eurostat. Brent crude oil et tension marché KBA testés puis écartés — non significatifs au-dessus du HICP. <strong>Modèle bien spécifié</strong> sur la dimension macro, sans over-engineering.',
    },
    {
        targetId: 'features',
        openCollapseId: null,
        title: '04 · Feature Engineering',
        text: 'Transformations <code>log</code> sur les variables numériques clés (age, mileage, catalogue) pour linéariser la dépréciation. <strong>Clustering KMeans</strong> des 62 modèles de véhicules en <strong>familles homogènes de dépréciation</strong> — évite que les arbres mémorisent chaque modèle commercial individuellement.',
    },
    {
        targetId: 'depreciation',
        openCollapseId: null,
        title: '05 · Dépréciation',
        text: 'Courbe de dépréciation empirique : <strong>convexe</strong> avec l\'âge (forte décote en début de vie, ralentissement ensuite). Analyse par marque, carburant et année de vente. L\'<strong>effet post-COVID</strong> (2021-2022) est visible : hausse des prix de l\'occasion due à la pénurie de semi-conducteurs.',
    },
    {
        targetId: 'results',
        openCollapseId: null,
        title: '06 · Évaluation des Modèles',
        text: '4 candidats (Ridge, RandomForest, XGBoost, CatBoost) + baseline naïve k·V₀ évalués en <strong>walk-forward post-COVID</strong> : 3 cutoffs successifs (2024-01, 2024-07, 2025-01) × horizon 6 mois. Tuning Optuna TPE <em>tune-once-apply-everywhere</em>. <strong>XGBoost/CatBoost dominent</strong>, ranking stable entre folds.',
    },
    {
        targetId: 'houssem',
        openCollapseId: null,
        title: '07 · Pipeline Structuré (validation indépendante)',
        text: 'Seconde méthodologie orthogonale : split aléatoire stratifié, target <code>log(price/msrp)</code>, monotonic constraints (âge↓, km↓), <strong>SHAP global</strong>. Champion identique : CatBoost (R² 0.74 sur log_ratio). Même verdict sur 2 protocoles différents → <strong>robustesse confirmée</strong>, pas d\'artefact de fenêtre.',
    },
    {
        targetId: 'portfolio',
        openCollapseId: null,
        title: '08 · Portfolio',
        text: 'Application du modèle retenu aux <strong>véhicules</strong> du portefeuille de leasing. Distribution des prix prédits, décote moyenne, répartition par marque, carburant, familles de dépréciation. Cohérence vérifiée avec la courbe de décote théorique.',
    },
    {
        targetId: 'stress',
        openCollapseId: null,
        title: '09 · Stress test — split temporel 49/51',
        text: 'Stress test de robustesse : <strong>train compressé (49&nbsp;%)</strong> / <strong>test étendu (51&nbsp;%)</strong> dans l\'ordre chronologique, pour mesurer la tenue du modèle sur un horizon de généralisation plus long. Ranking conservé = robustesse à la compression du train, confirmation du choix post-COVID.',
    },
    {
        targetId: 'validation',
        openCollapseId: null,
        title: '10 · Validation externe — AutoScout24',
        text: '10 modèles confrontés à <strong>AutoScout24</strong> par scraping temps réel. Écart médian <strong>X&nbsp;%</strong> — pas une erreur, mais la signature structurelle <strong>B2B (reprise lessor) vs B2C (retail)</strong>. Modèle conservateur = exactement ce qu\'il faut pour [Client] : on ne surestime pas les reprises.',
    },
    {
        targetId: 'risk',
        openCollapseId: null,
        title: '11 · Analyse de risque',
        text: 'Exposition totale <strong>~20 M€</strong>. Stress tests BCE/EBA -5 / -10 / -15 %. <strong>Intervalles conformels 80/90/95&nbsp;%</strong> calibrés → VaR/CVaR de portefeuille. Top 10 véhicules à risque actionnables pour revente anticipée. On passe d\'un modèle qui prédit à un <strong>outil de pilotage CFO</strong>.',
    },
    {
        targetId: 'simulator',
        openCollapseId: null,
        title: 'Simulateur — preuve opérationnelle',
        text: 'Démo unitaire : on entre les caractéristiques d\'un véhicule, le modèle prédit sa VR <strong>en temps réel</strong> avec une <strong>bande d\'incertitude</strong> (±MAPE walk-forward). Le modèle quitte le notebook et devient un outil actionnable.',
    },
    {
        targetId: 'simulator',
        openCollapseId: null,
        title: 'Pour aller plus loin — assistant & partage',
        text: 'Un <strong>assistant méthodologique</strong> (LLM local, bouton bas-droite) répond en direct aux questions sur le pipeline, les choix de modélisation et les métriques. Le <strong>QR code en haut-droite</strong> permet de partager le dashboard instantanément pour le consulter sur un autre écran ou le transmettre.',
    },
];

let _tourIndex = 0;
let _tourActive = false;
let _tourPrevSection = null;
let _tourKeyHandler = null;

function _tourRenderDots() {
    const el = document.getElementById('tour-step-dots');
    if (!el) return;
    const html = TOUR_STEPS.map((_, i) => {
        let cls = 'tour-bubble__step-dot';
        if (i < _tourIndex) cls += ' is-done';
        else if (i === _tourIndex) cls += ' is-active';
        return `<span class="${cls}"></span>`;
    }).join('');
    el.innerHTML = html;
}

function _tourGoTo(index) {
    if (index < 0 || index >= TOUR_STEPS.length) return;
    _tourIndex = index;
    const step = TOUR_STEPS[_tourIndex];

    // Open the collapse if step requires it.
    if (step.openCollapseId) {
        const trigger = document.querySelector(`.collapse-card__trigger[aria-controls="${step.openCollapseId}"]`);
        if (trigger && trigger.getAttribute('aria-expanded') !== 'true') {
            _openCollapse(trigger);
        }
    }

    // Highlight section + scroll.
    const target = document.getElementById(step.targetId);
    if (target) {
        if (_tourPrevSection && _tourPrevSection !== target) {
            _tourPrevSection.classList.remove('tour-active');
        }
        target.classList.add('tour-active');
        _tourPrevSection = target;
        // Scroll with offset so the section header is visible above the bubble.
        setTimeout(() => {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 80);
    }

    // Update bubble content.
    const bubble = document.getElementById('tour-bubble');
    const titleEl = document.getElementById('tour-bubble-title');
    const textEl = document.getElementById('tour-bubble-text');
    const counterEl = document.getElementById('tour-step-counter');
    const prevBtn = document.getElementById('tour-prev');
    const nextBtn = document.getElementById('tour-next');
    if (titleEl) titleEl.textContent = step.title;
    if (textEl) {
        // Remplace le token {{CHOSEN_MODEL}} par le nom du modèle retenu
        // (calculé dynamiquement par le tuning_vs_baseline — stockage global
        // dans window.__chosenModel, fallback 'XGBoost tuné Optuna' si absent).
        const chosen = (window.__chosenModel
            ? modelDisplayName(window.__chosenModel) + ' tuné Optuna'
            : 'XGBoost tuné Optuna');
        textEl.innerHTML = step.text.replace(/\{\{CHOSEN_MODEL\}\}/g, chosen);
    }
    if (counterEl) counterEl.textContent = `${_tourIndex + 1} / ${TOUR_STEPS.length}`;
    if (prevBtn) prevBtn.disabled = _tourIndex === 0;
    if (nextBtn) {
        nextBtn.innerHTML = _tourIndex === TOUR_STEPS.length - 1
            ? 'Terminer &check;'
            : 'Suivant &rarr;';
    }
    _tourRenderDots();

    if (bubble) {
        bubble.removeAttribute('hidden');
        requestAnimationFrame(() => bubble.classList.add('is-visible'));
    }
}

function _tourStart() {
    if (_tourActive) return;
    _tourActive = true;
    document.body.classList.add('tour-running');
    _tourGoTo(0);
    if (!_tourKeyHandler) {
        _tourKeyHandler = (e) => {
            if (!_tourActive) return;
            if (e.key === 'Escape') { e.preventDefault(); _tourExit(); }
            else if (e.key === ' ' || e.key === 'ArrowRight') {
                if (document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
                e.preventDefault();
                _tourNext();
            }
            else if (e.key === 'ArrowLeft') {
                if (document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
                e.preventDefault();
                _tourGoTo(_tourIndex - 1);
            }
        };
        document.addEventListener('keydown', _tourKeyHandler);
    }
}

function _tourNext() {
    if (_tourIndex >= TOUR_STEPS.length - 1) {
        _tourExit();
    } else {
        _tourGoTo(_tourIndex + 1);
    }
}

function _tourExit() {
    _tourActive = false;
    document.body.classList.remove('tour-running');
    if (_tourPrevSection) {
        _tourPrevSection.classList.remove('tour-active');
        _tourPrevSection = null;
    }
    const bubble = document.getElementById('tour-bubble');
    if (bubble) {
        bubble.classList.remove('is-visible');
        setTimeout(() => bubble.setAttribute('hidden', ''), 250);
    }
}

function setupTour() {
    const startBtn = document.getElementById('tour-start');
    const prevBtn = document.getElementById('tour-prev');
    const nextBtn = document.getElementById('tour-next');
    const exitBtn = document.getElementById('tour-exit');
    if (!startBtn) return;
    startBtn.addEventListener('click', _tourStart);
    if (prevBtn) prevBtn.addEventListener('click', () => _tourGoTo(_tourIndex - 1));
    if (nextBtn) nextBtn.addEventListener('click', _tourNext);
    if (exitBtn) exitBtn.addEventListener('click', _tourExit);
}

// ============================================================
// RAG ASSISTANT — drawer shell + canned responses
// ============================================================

// Réponses pré-baked basées sur les VRAIES données du dashboard.
// Schéma identique à la future réponse de /api/rag (voir app.py).
const RAG_CANNED = {
    'catboost-choice': {
        question: "Pourquoi XGBoost et pas un autre modèle ?",
        answer: "<strong>XGBoost</strong> est retenu pour le livrable final (MAPE 6.08 % EUR / 9.2 % log_ratio sur le test hold-out 20 %). CatBoost suit de très près (écart MAPE &lt; 0.5 pp). Ridge sert de baseline interprétable mais accuse <strong>~10 pp d'écart</strong> de MAPE.",
        methodology: "XGBoost et CatBoost sont quasi ex-aequo (écart MAPE &lt; 0.5 pp), mais XGBoost est retenu sur critère de stabilité (std CV MAPE min) et pour la vitesse d'inférence. Il gère les variables catégorielles via one-hot encoding léger (brand, fuel_type, range_type, model_family). Hyperparamètres optimisés par Optuna (15 trials TPE) + CV 3-fold stratifiée sur le décile de prix — conformément au protocole Nexialog (slides 13-18) et appuyé par Ghibellini et al. (IEEE Access, 2024). Le modèle final est ré-entraîné sur 100 % des données.",
        sources: [
            { type: 'data', label: 'log_ratio_metrics.csv' },
            { type: 'data', label: 'feature_importance' },
            { type: 'code', label: 'vr_pipeline.py' },
            { type: 'doc', label: 'Ghibellini et al. (2024)' },
        ],
        confidence: 'high',
        confidence_basis: "Validé empiriquement sur Transactions, test hold-out 20 % stratifié sur le décile de prix.",
    },
    'as24-gap': {
        question: "Comment interpréter l'écart vs AutoScout24 ?",
        answer: "Cet écart est <strong>attendu et souhaitable</strong> : il reflète le différentiel structurel B2B ([Client], enchères pros) vs B2C (AutoScout24, particuliers). Un modèle qui prédirait à parité serait <strong>dangereusement optimiste</strong> pour un usage financier.",
        methodology: "AutoScout24 affiche des prix <em>demandés</em> par les vendeurs (particuliers + concessions), avec une marge de négociation typique de 10-15 % et un biais d'échantillonnage vers les véhicules les mieux équipés. Notre modèle est entraîné sur prix de <em>transactions réelles</em> du marché VO cible, qui incluent les ventes B2B, reprises et ventes institutionnelles — typiquement 15 à 25 % sous les annonces B2C. [Client] revend en fin de contrat principalement via canaux B2B : notre modèle prédit donc les prix réellement obtenus, pas les prix optimistes affichés. L'écart X % (modèle post-COVID retenu, vs X % pour la baseline full-train) peut servir de <strong>borne supérieure</strong> pour estimer le prix maximum théorique en vente directe B2C, avec un facteur de recalibration ×N.",
        sources: [
            { type: 'data', label: 'scrape_validation_results.json' },
            { type: 'code', label: 'scrape_validation.py' },
            { type: 'data', label: '10 modèles confrontés (Renault×6, Dacia×3, Nissan×1)' },
        ],
        confidence: 'high',
        confidence_basis: "Confirmé sur 10 modèles distincts, écart cohérent avec la littérature B2B vs B2C.",
    },
    'kba-test': {
        question: "Le test de robustesse KBA est-il significatif ?",
        answer: "<strong>Non</strong> — avec HICP dans le baseline, β = -0.0053 (p ≈ 0.30). La tension marché KBA <strong>n'apporte rien au-dessus du signal HICP</strong> déjà capté par le modèle.",
        methodology: "Le test de robustesse a été conduit en 4 configurations pour trianguler le résultat. Avec HICP + tension KBA externe : β non significatif (p = 0.30). Avec HICP + tension interne (proxy) : p = 0.47, idem. Sans HICP, β devient significatif mais avec signe <strong>négatif</strong> — révélateur d'une multicolinéarité : la tension joue alors comme un proxy imparfait de l'inflation. Méthodo : régression post-hoc sur les résidus mensuels du CatBoost baseline (pondérés par volume de ventes), pour éviter l'endogénéité. Conclusion : le modèle CatBoost actuel est <strong>bien spécifié sur la dimension macro</strong> — les features HICP absorbent déjà le signal qu'apporterait une tension marché externe.",
        sources: [
            { type: 'data', label: 'market_robustness_results.json' },
            { type: 'code', label: 'used_market_analysis.py' },
            { type: 'data', label: 'fichiers KBA scrapés (FZ 9 + FZ 10, 2018-2025)' },
        ],
        confidence: 'high',
        confidence_basis: "4 configurations triangulent le résultat, p-values robustes sur 58 à 95 mois d'observations.",
    },
    'portfolio-vr': {
        question: "Quelle est la VR moyenne du portefeuille ?",
        answer: "<strong>Décote moyenne ~49.5 %</strong> sur les véhicules du portefeuille — soit une <strong>exposition totale ~20 M€</strong> de valeur résiduelle prédite à la fin des contrats.",
        methodology: "Prédictions générées par l'ensemble CatBoost + XGBoost appliqué à l'état prévisible de chaque véhicule à la fin de son contrat (age_months recalculé, mileage_at_end = initial_mileage + contract_mileage, transformations log cohérentes avec le training). La distribution centrée sur ~50 % de décote est typique pour des contrats 3-5 ans sur le segment Renault/Dacia/Nissan. Stress tests calibrés BCE/EBA appliqués (-5 / -10 / -15 %) pour donner une fourchette de pertes sous scénarios adverses. Le top 10 des décotes les plus fortes identifie les contrats actionnables (revente anticipée, renégociation).",
        sources: [
            { type: 'data', label: 'prediction_portfolio.csv' },
            { type: 'code', label: 'vr_pipeline.py' },
            { type: 'data', label: 'contrats × 8 colonnes' },
        ],
        confidence: 'medium',
        confidence_basis: "Confiance élevée sur la valeur centrale ; modérée sous chocs > 15 % (extrapolation hors historique observé).",
    },
};

function _ragSourceIcon(type) {
    if (type === 'data') return '⌗';
    if (type === 'code') return '<>';
    if (type === 'doc') return '¶';
    return '·';
}

function renderRagResponse(payload, opts = {}) {
    const conv = document.getElementById('rag-conversation');
    if (!conv) return;
    if (opts.userQuery) {
        const userArt = document.createElement('article');
        userArt.className = 'rag-msg rag-msg--user';
        userArt.innerHTML = `<div class="rag-msg__bubble"></div>`;
        userArt.querySelector('.rag-msg__bubble').textContent = opts.userQuery;
        conv.appendChild(userArt);
    }
    const art = document.createElement('article');
    art.className = 'rag-msg rag-msg--assistant';
    const sourcesHTML = (payload.sources || []).map(s =>
        `<span class="rag-source-chip"><span class="rag-source-chip__icon">${_ragSourceIcon(s.type)}</span>${s.label}</span>`
    ).join('');
    const confLevel = payload.confidence || 'low';
    const confLabel = { high: 'élevée', medium: 'modérée', low: 'faible' }[confLevel] || 'faible';
    art.innerHTML = `
        <div class="rag-msg__answer">${payload.answer}</div>
        ${payload.methodology ? `<details class="rag-msg__methodology"><summary>Justification méthodologique</summary><p>${payload.methodology}</p></details>` : ''}
        ${sourcesHTML ? `<div class="rag-msg__sources"><span class="rag-msg__sources-label">Sources</span>${sourcesHTML}</div>` : ''}
        <div class="rag-msg__confidence rag-msg__confidence--${confLevel}">
            <span class="rag-confidence-dot"></span>
            Confiance : <strong>${confLabel}</strong>${payload.confidence_basis ? ` · ${payload.confidence_basis}` : ''}
        </div>
    `;
    conv.appendChild(art);
    // Hide welcome state after first response.
    const welcome = document.getElementById('rag-welcome');
    if (welcome) welcome.style.display = 'none';
    // Scroll to bottom.
    requestAnimationFrame(() => {
        const body = document.querySelector('.rag-drawer__body');
        if (body) body.scrollTop = body.scrollHeight;
    });
}

// ============================================================
// RAG Assistant (Ollama llama3.1:8b via /api/rag streaming)
// ============================================================
function _ragEscape(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function _ragFormatStream(raw) {
    // Minimal markdown-ish formatting: **bold**, `code`, line breaks
    let s = _ragEscape(raw);
    s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    return s;
}

function appendUserMessage(text) {
    const conv = document.getElementById('rag-conversation');
    if (!conv) return;
    const art = document.createElement('article');
    art.className = 'rag-msg rag-msg--user';
    const bubble = document.createElement('div');
    bubble.className = 'rag-msg__bubble';
    bubble.textContent = text;
    art.appendChild(bubble);
    conv.appendChild(art);
    _ragScrollBottom();
}

function appendAssistantPlaceholder() {
    const conv = document.getElementById('rag-conversation');
    if (!conv) return null;
    const art = document.createElement('article');
    art.className = 'rag-msg rag-msg--assistant';
    const ans = document.createElement('div');
    ans.className = 'rag-msg__answer';
    const typing = document.createElement('span');
    typing.className = 'rag-msg__typing';
    typing.innerHTML = '<span></span><span></span><span></span>';
    ans.appendChild(typing);
    art.appendChild(ans);
    conv.appendChild(art);
    _ragScrollBottom();
    return ans;
}

function _ragScrollBottom() {
    requestAnimationFrame(() => {
        const body = document.querySelector('.rag-drawer__body');
        if (body) body.scrollTop = body.scrollHeight;
    });
}

async function askOllama(query, answerEl) {
    const resp = await fetch('/api/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
    });
    if (!resp.ok) {
        const err = await resp.text().catch(() => '');
        throw new Error(`HTTP ${resp.status} ${err}`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let full = '';
    let firstChunk = true;
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let lineBreak;
        while ((lineBreak = buf.indexOf('\n')) >= 0) {
            const line = buf.slice(0, lineBreak).trim();
            buf = buf.slice(lineBreak + 1);
            if (!line) continue;
            let payload;
            try { payload = JSON.parse(line); } catch { continue; }
            if (payload.error) throw new Error(payload.error);
            if (payload.chunk) {
                if (firstChunk) { answerEl.innerHTML = ''; firstChunk = false; }
                full += payload.chunk;
                answerEl.innerHTML = _ragFormatStream(full);
                _ragScrollBottom();
            }
            if (payload.done) {
                const meta = document.createElement('div');
                meta.className = 'rag-msg__meta';
                const dot = document.createElement('span');
                dot.className = 'rag-msg__meta-dot';
                meta.appendChild(dot);
                const label = document.createElement('span');
                const tok = payload.eval_count != null ? ` · ${payload.eval_count} tok` : '';
                const dur = payload.duration_ms != null ? ` · ${(payload.duration_ms/1000).toFixed(1)}s` : '';
                label.textContent = `llama3.1:8b${tok}${dur}`;
                meta.appendChild(label);
                answerEl.parentElement.appendChild(meta);
            }
        }
    }
    return full;
}

function setupRagDrawer() {
    const drawer = document.getElementById('rag-drawer');
    const trigger = document.getElementById('rag-trigger');
    const closeBtn = document.getElementById('rag-close');
    const backdrop = document.getElementById('rag-backdrop');
    const form = document.getElementById('rag-form');
    const input = document.getElementById('rag-input');
    const sendBtn = form ? form.querySelector('.rag-send') : null;
    if (!drawer || !trigger) return;

    let lastFocus = null;

    const open = () => {
        lastFocus = document.activeElement;
        drawer.removeAttribute('hidden');
        document.body.classList.add('rag-open');
        requestAnimationFrame(() => {
            drawer.classList.add('is-open');
            if (input) setTimeout(() => input.focus(), 350);
        });
    };
    const close = () => {
        drawer.classList.remove('is-open');
        document.body.classList.remove('rag-open');
        setTimeout(() => {
            drawer.setAttribute('hidden', '');
            if (lastFocus && typeof lastFocus.focus === 'function') lastFocus.focus();
        }, 300);
    };

    trigger.addEventListener('click', open);
    if (closeBtn) closeBtn.addEventListener('click', close);
    if (backdrop) backdrop.addEventListener('click', close);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && drawer.classList.contains('is-open')) close();
    });

    const submitQuery = async (q) => {
        if (!q || !q.trim()) return;
        const query = q.trim();
        const welcome = document.getElementById('rag-welcome');
        if (welcome) welcome.style.display = 'none';
        appendUserMessage(query);
        const answerEl = appendAssistantPlaceholder();
        if (sendBtn) sendBtn.disabled = true;
        try {
            await askOllama(query, answerEl);
        } catch (err) {
            answerEl.innerHTML = `<strong>Erreur.</strong> ${_ragEscape(err.message || 'Requête impossible.')}<br><em>Vérifie qu'Ollama tourne (<code>ollama serve</code>) et que <code>llama3.1:8b</code> est installé.</em>`;
        } finally {
            if (sendBtn) sendBtn.disabled = false;
            if (input) input.focus();
        }
    };

    document.querySelectorAll('.rag-suggestion-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.getAttribute('data-prompt') || chip.textContent;
            submitQuery(prompt);
        });
    });

    if (form && input) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const q = input.value;
            input.value = '';
            input.style.height = 'auto';
            submitQuery(q);
        });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                form.requestSubmit();
            }
        });
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(120, input.scrollHeight) + 'px';
        });
    }
}

// ---- Back-to-top button ----
function setupBackToTop() {
    const btn = document.getElementById('back-to-top');
    if (!btn) return;
    let ticking = false;
    const update = () => {
        btn.classList.toggle('visible', window.scrollY > 400);
        ticking = false;
    };
    const onScroll = () => {
        if (!ticking) { requestAnimationFrame(update); ticking = true; }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    btn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    update();
}

// Helper pour peupler les span dyn-* (utilisé par les renders post-COVID)
function setText(id, value) {
    const el = document.getElementById(id);
    if (el != null && value != null) el.textContent = value;
}

// ---- Chart rendering ----
function renderAllCharts() {
    const c = DATA.charts;

    // EDA distributions
    renderHistogram('chart-dist-age', c.dist_age, 'Âge (mois)', 'Nombre de véhicules');
    renderHistogram('chart-dist-mileage', c.dist_mileage, 'Kilométrage (km)', 'Nombre de véhicules');
    renderHistogram('chart-dist-price', c.dist_price, 'Prix de vente (EUR)', 'Nombre de véhicules');
    renderHistogram('chart-dist-catalogue', c.dist_catalogue, 'Prix catalogue (EUR)', 'Nombre de véhicules');
    renderHistogram('chart-dist-logratio', c.dist_log_ratio, 'log_ratio', 'Nombre de véhicules');

    // Categorical
    renderBarChart('chart-cat-brand', c.cat_brand, 'Marque', 'Nombre de transactions');
    renderPieChart('chart-cat-fuel', c.cat_fuel_type);
    renderPieChart('chart-cat-range', c.cat_range_type);
    renderModelBarChart('chart-cat-model', c.cat_model);

    // Correlation
    renderHeatmap('chart-correlation', c.correlation);

    // HICP
    renderTimeSeries('chart-hicp-headline', c.hicp_headline, 'Inflation YoY (%)', '#ef4444');
    renderTimeSeries('chart-hicp-core', c.hicp_core, 'Indice HICP Core', '#4BC4BD');
    renderTimeSeries('chart-hicp-energy', c.hicp_energy, 'Inflation Energie YoY (%)', '#f59e0b');

    // HICP vs log_ratio
    renderScatterBins('chart-hicp-vs-core', c.hicp_vs_logratio_log_cum_inflation_core, 'log_cum_inflation_core', 'corr-core');
    renderScatterBins('chart-hicp-vs-headline', c.hicp_vs_logratio_hicp_headline_yoy_sale, 'hicp_headline_yoy_sale', 'corr-headline');
    renderScatterBins('chart-hicp-vs-energy', c.hicp_vs_logratio_hicp_energy_yoy_sale, 'hicp_energy_yoy_sale', 'corr-energy');

    // Energy x Fuel interaction
    renderEnergyFuel('chart-energy-fuel', c.energy_fuel_interaction);

    // Clustering
    if (c.cluster_eval) {
        renderClusterElbow('chart-cluster-elbow', c.cluster_eval);
        renderClusterSilhouette('chart-cluster-silhouette', c.cluster_eval);
    }

    // Depreciation
    renderDepreciationAge('chart-depreciation-age', c.depreciation_age);
    renderPriceByYear('chart-price-year', c.price_by_year);
    renderDepreciationFuel('chart-depreciation-fuel', c.depreciation_fuel);
    renderPriceBrand('chart-price-brand', c.price_by_brand);

    // Model comparison
    renderModelComparison('chart-model-comparison', c.model_comparison);
    renderMetricsTable();

    // Feature importance
    renderFeatureImportance('chart-feature-importance', c.feature_importance);

    // Pred vs Obs
    renderPredVsObs('chart-pred-vs-obs', c.pred_vs_obs);
    renderHistogram('chart-residuals', c.residuals, 'Résidus (EUR)', 'Fréquence');

    // Validation externe AutoScout24
    renderScrapeValidation(c.scrape_validation);

    // Innovation : test de robustesse tension marché KBA
    renderKbaTension('chart-kba-tension', c.kba_timeseries, {
        hicp_energy: c.hicp_energy,
        hicp_headline: c.hicp_headline,
    });
    renderKbaRobustnessTable(c.market_robustness);
    // Si ni les données KBA ni la table ne sont disponibles, on masque la table
    // et la lecture (on garde le graphique fallback HICP).
    const kbaTable = document.getElementById('kba-robustness-wrap');
    if (kbaTable && !c.market_robustness) {
        const tableEl = kbaTable.querySelector('.table-responsive');
        if (tableEl) tableEl.style.display = 'none';
        const pNodes = kbaTable.querySelectorAll(':scope > p');
        pNodes.forEach(p => {
            const t = p.textContent || '';
            if (t.includes('Résultats du test') || t.includes('Lecture :')) p.style.display = 'none';
        });
    }

    // Analyses temporelles post-COVID (ajouts collègue) — chaque render isolé pour
    // qu'une erreur sur un graph n'empêche pas la suite (notamment setupCollapsibles).
    const safeRender = (name, fn, data) => {
        try { fn(data); }
        catch (e) { console.error(`[renderAllCharts] ${name} threw:`, e); }
    };
    safeRender('renderCovidRegime',            renderCovidRegime,            c.covid_regime);
    safeRender('renderConformal',              renderConformal,              c.conformal);
    safeRender('renderRollingOriginPostCovid', renderRollingOriginPostCovid, c.rolling_origin_postcovid);
    safeRender('renderTuningVsBaseline',       renderTuningVsBaseline,       c.tuning_vs_baseline);
    safeRender('renderStressTest4951',         renderStressTest4951,         c.stress_test_4951);
    safeRender('renderAs24PostCovid',          renderAs24PostCovid,          c.as24_postcovid);

    // Section 08 — Pipeline Structuré (Structured Pipeline)
    safeRender('renderHoussemPipeline',        renderHoussemPipeline,        c.houssem_pipeline);

    // Fill section 07 placeholders (« -- » bandeau et bullets « Lecture finale »)
    safeRender('fillSection07Placeholders',    fillSection07Placeholders,    c);

    // Risk Portfolio
    renderRiskPortfolio(c.risk_portfolio);

    // Portfolio
    renderHistogram('chart-portfolio-pred', c.portfolio_predictions, 'Prix prédit (EUR)', 'Nombre de véhicules');
    renderHistogram('chart-portfolio-decote', c.portfolio_decote, 'Décote (%)', 'Nombre de véhicules');
    renderPortfolioAge('chart-portfolio-age', c.portfolio_pred_vs_age);
    renderPortfolioBrand('chart-portfolio-brand', c.portfolio_by_brand);
    renderPortfolioFuel('chart-portfolio-fuel', c.portfolio_by_fuel);
    renderFamilyStats(c.family_stats);
}

// ---- Histogram ----
function renderHistogram(id, data, xlabel, ylabel) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        x: data.x,
        y: data.y,
        type: 'bar',
        marker: {
            color: 'rgba(59,130,246,0.55)',
            line: { color: 'rgba(59,130,246,0.85)', width: 1 },
        },
        hovertemplate: `${xlabel}: %{x:.0f}<br>${ylabel}: %{y:,}<extra></extra>`,
    };
    const layout = mergeLayout({
        xaxis: { title: xlabel },
        yaxis: { title: ylabel },
        bargap: 0.05,
    });
    // Add mean/median annotations
    const annotations = [];
    if (data.median != null) {
        annotations.push({
            x: data.median, y: Math.max(...data.y) * 0.9,
            text: `Med: ${fmt(Math.round(data.median))}`,
            showarrow: true, arrowhead: 2, arrowcolor: '#4BC4BD',
            font: { color: '#4BC4BD', size: 11 },
            bgcolor: 'rgba(12,18,37,0.9)', bordercolor: '#4BC4BD',
        });
    }
    layout.annotations = annotations;
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

// ---- Bar chart ----
function renderBarChart(id, data, xlabel, ylabel) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        x: data.values,
        y: data.labels,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: data.values.map((_, i) => {
                const colors = ['#3b82f6', '#60a5fa', '#93c5fd'];
                return colors[i % colors.length];
            }),
        },
        hovertemplate: '%{y}: %{x:,}<extra></extra>',
    };
    const layout = mergeLayout({
        xaxis: { title: ylabel },
        yaxis: { autorange: 'reversed' },
        margin: { l: 120 },
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

// ---- Model bar chart (with brand legend) ----
function renderModelBarChart(id, data) {
    if (!data || !document.getElementById(id)) return;
    const mbb = DATA.options.models_by_brand || {};
    const modelBrand = {};
    for (const [brand, models] of Object.entries(mbb)) {
        for (const m of models) modelBrand[m] = brand;
    }
    const colorMap = { RENAULT: '#3b82f6', DACIA: '#4BC4BD', NISSAN: '#f59e0b' };
    const brands = data.labels.map(m => modelBrand[m] || '');
    // Main trace with original order
    const trace = {
        x: data.values,
        y: data.labels,
        type: 'bar',
        orientation: 'h',
        marker: { color: brands.map(b => colorMap[b] || '#3b82f6') },
        customdata: brands,
        hovertemplate: '%{y} (%{customdata})<br>%{x:,} transactions<extra></extra>',
        showlegend: false,
    };
    // Invisible traces for legend only
    const legendTraces = ['RENAULT', 'DACIA', 'NISSAN'].map(b => ({
        x: [null], y: [null], type: 'bar', orientation: 'h',
        name: b, marker: { color: colorMap[b] },
        showlegend: true,
    }));
    const layout = mergeLayout({
        xaxis: { title: 'Nombre de transactions' },
        yaxis: { autorange: 'reversed' },
        margin: { l: 130, b: 80 },
        legend: { orientation: 'h', y: -0.32, x: 0.5, xanchor: 'center', font: { size: 11 } },
    });
    Plotly.newPlot(id, [trace, ...legendTraces], layout, PLOTLY_CONFIG);
}

// ---- Pie chart ----
function renderPieChart(id, data) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        labels: data.labels,
        values: data.values,
        type: 'pie',
        hole: 0.45,
        textinfo: 'label+percent',
        textposition: 'inside',
        insidetextorientation: 'horizontal',
        marker: {
            colors: ['#3b82f6', '#4BC4BD', '#f59e0b', '#002c58', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'],
        },
        hovertemplate: '%{label}: %{value:,} (%{percent})<extra></extra>',
    };
    const layout = mergeLayout({
        margin: { t: 20, b: 20, l: 20, r: 20 },
        showlegend: false,
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

// ---- Heatmap ----
function renderHeatmap(id, data) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        z: data.values,
        x: data.labels,
        y: data.labels,
        type: 'heatmap',
        colorscale: [
            [0, '#1e3a5f'], [0.25, '#2563eb'], [0.5, '#f8fafc'],
            [0.75, '#f59e0b'], [1, '#ef4444'],
        ],
        zmin: -1, zmax: 1,
        hovertemplate: '%{x} vs %{y}: %{z:.3f}<extra></extra>',
    };
    const layout = mergeLayout({
        margin: { t: 20, b: 80, l: 100, r: 20 },
        xaxis: { tickangle: -45, side: 'bottom' },
        yaxis: { autorange: 'reversed' },
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

// ---- Time series ----
function renderTimeSeries(id, data, ylabel, color) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        x: data.dates,
        y: data.values,
        type: 'scatter',
        mode: 'lines',
        line: { color: color, width: 2 },
        fill: 'tozeroy',
        fillcolor: color.replace(')', ',0.1)').replace('rgb', 'rgba'),
        hovertemplate: '%{x}: %{y:.1f}<extra></extra>',
    };
    const layout = mergeLayout({
        yaxis: { title: ylabel },
        margin: { t: 10, b: 40, l: 50, r: 10 },
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

// ---- Scatter bins (HICP vs log_ratio) ----
function renderScatterBins(id, data, xlabel, corrId) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        x: data.x,
        y: data.y,
        mode: 'markers',
        type: 'scatter',
        marker: {
            size: data.n.map(n => Math.min(Math.sqrt(n) / 5, 12)),
            color: '#3b82f6',
            opacity: 0.7,
        },
        hovertemplate: `${xlabel}: %{x:.3f}<br>log_ratio moyen: %{y:.3f}<extra></extra>`,
    };
    // Trendline
    const trace2 = {
        x: data.x,
        y: linearFit(data.x, data.y),
        mode: 'lines',
        type: 'scatter',
        name: 'Tendance linéaire',
        line: { color: '#ef4444', width: 2, dash: 'dash' },
        hoverlabel: { bgcolor: '#7f1d1d', bordercolor: '#ef4444', font: { color: '#fca5a5' } },
        hovertemplate: 'Droite de régression linéaire<br>(tendance moyenne)<extra></extra>',
    };
    const layout = mergeLayout({
        xaxis: { title: xlabel },
        yaxis: { title: 'log_ratio moyen' },
        showlegend: false,
        margin: { t: 10, b: 50, l: 60, r: 10 },
    });
    Plotly.newPlot(id, [trace, trace2], layout, PLOTLY_CONFIG);

    const el = document.getElementById(corrId);
    if (el) {
        const sp = (data.spearman !== undefined) ? ` &nbsp;|&nbsp; Spearman : ρ = ${data.spearman}` : '';
        el.innerHTML = `Pearson : r = ${data.corr}${sp}`;
    }
}

function linearFit(x, y) {
    const n = x.length;
    const sx = x.reduce((a, b) => a + b, 0);
    const sy = y.reduce((a, b) => a + b, 0);
    const sxy = x.reduce((a, xi, i) => a + xi * y[i], 0);
    const sx2 = x.reduce((a, xi) => a + xi * xi, 0);
    const slope = (n * sxy - sx * sy) / (n * sx2 - sx * sx);
    const intercept = (sy - slope * sx) / n;
    return x.map(xi => slope * xi + intercept);
}

// ---- Energy x Fuel interaction ----
function renderEnergyFuel(id, data) {
    if (!data || !data.data || !document.getElementById(id)) return;
    const regimes = [...new Set(data.data.map(d => d.regime))];
    const fuels = [...new Set(data.data.map(d => d.fuel))];

    const traces = regimes.map((regime, i) => {
        const filtered = data.data.filter(d => d.regime === regime);
        return {
            x: filtered.map(d => d.fuel),
            y: filtered.map(d => d.log_ratio),
            name: regime,
            type: 'bar',
            marker: { opacity: 0.85 },
        };
    });
    const layout = mergeLayout({
        barmode: 'group',
        xaxis: { title: 'Type de carburant' },
        yaxis: { title: 'log_ratio moyen' },
        legend: { orientation: 'h', y: -0.3, x: 0.5, xanchor: 'center' },
        margin: { t: 30, r: 20, b: 80, l: 60 },
    });
    Plotly.newPlot(id, traces, layout, PLOTLY_CONFIG);
}

// ---- Cluster charts ----
function renderClusterElbow(id, data) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        x: data.k,
        y: data.inertia,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#3b82f6', width: 2 },
        marker: { size: 8 },
    };
    const layout = mergeLayout({
        xaxis: { title: 'Nombre de clusters (k)', dtick: 1 },
        yaxis: { title: 'Inertie' },
        margin: { t: 10, b: 50, l: 60, r: 10 },
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

function renderClusterSilhouette(id, data) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        x: data.k,
        y: data.silhouette,
        type: 'bar',
        marker: {
            color: data.silhouette.map(s => s === Math.max(...data.silhouette) ? '#4BC4BD' : '#3b82f6'),
        },
    };
    const layout = mergeLayout({
        xaxis: { title: 'Nombre de clusters (k)', dtick: 1 },
        yaxis: { title: 'Score de silhouette' },
        margin: { t: 10, b: 50, l: 60, r: 10 },
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

// ---- Depreciation by age ----
function renderDepreciationAge(id, data) {
    if (!data || !document.getElementById(id)) return;
    const ages = data.map(d => d.age);
    const medians = data.map(d => d.median);
    const q25 = data.map(d => d.q25);
    const q75 = data.map(d => d.q75);

    const band = {
        x: [...ages, ...ages.slice().reverse()],
        y: [...q75, ...q25.slice().reverse()],
        type: 'scatter',
        fill: 'toself',
        fillcolor: 'rgba(59,130,246,0.08)',
        line: { color: 'transparent' },
        hoverinfo: 'skip',
        name: 'IQR (Q25-Q75)',
    };
    const line = {
        x: ages,
        y: medians,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#3b82f6', width: 3 },
        marker: { size: 4 },
        name: 'Prix médian',
        hovertemplate: 'Âge: %{x} mois<br>Prix médian: %{y:,.0f} EUR<extra></extra>',
    };
    const layout = mergeLayout({
        xaxis: { title: 'Âge du véhicule (mois)' },
        yaxis: { title: 'Prix de vente (EUR)' },
        showlegend: true,
        legend: { orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center' },
        margin: { t: 30, r: 20, b: 70, l: 60 },
    });
    Plotly.newPlot(id, [band, line], layout, PLOTLY_CONFIG);
}

// ---- Price by year ----
function renderPriceByYear(id, data) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        x: data.years,
        y: data.median,
        type: 'bar',
        marker: { color: '#3b82f6', opacity: 0.8 },
        hovertemplate: 'Année %{x}<br>Prix médian: %{y:,.0f} EUR<extra></extra>',
    };
    const layout = mergeLayout({
        xaxis: { title: 'Année de vente', dtick: 1 },
        yaxis: { title: 'Prix médian (EUR)' },
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

// ---- Depreciation by fuel ----
function renderDepreciationFuel(id, data) {
    if (!data || !document.getElementById(id)) return;
    const traces = data.map((fuel, i) => ({
        x: fuel.x,
        y: fuel.y,
        type: 'scatter',
        mode: 'lines+markers',
        name: fuel.fuel,
        line: { width: 2 },
        marker: { size: 6 },
        hovertemplate: `${fuel.fuel}<br>Age: %{x} ans<br>Prix: %{y:,.0f} EUR<extra></extra>`,
    }));
    const layout = mergeLayout({
        xaxis: { title: 'Âge (années)', dtick: 1 },
        yaxis: { title: 'Prix médian (EUR)' },
        legend: { orientation: 'h', y: -0.35, x: 0.5, xanchor: 'center', font: { size: 10 } },
        margin: { t: 30, r: 20, b: 90, l: 60 },
    });
    Plotly.newPlot(id, traces, layout, PLOTLY_CONFIG);
}

// ---- Price by brand (horizontal bar) ----
function renderPriceBrand(id, data) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        y: data.map(d => d.brand),
        x: data.map(d => d.median),
        type: 'bar',
        orientation: 'h',
        marker: { color: '#3b82f6', opacity: 0.8 },
        error_x: {
            type: 'data',
            symmetric: false,
            array: data.map(d => d.max - d.median),
            arrayminus: data.map(d => d.median - d.min),
            color: 'rgba(148,163,184,0.4)',
            thickness: 1,
        },
        hovertemplate: '%{y}<br>Median: %{x:,.0f} EUR<br>Q25-Q75: %{customdata[0]:,.0f}-%{customdata[1]:,.0f}<extra></extra>',
        customdata: data.map(d => [d.q25, d.q75]),
    };
    const layout = mergeLayout({
        xaxis: { title: 'Prix de revente (EUR)' },
        margin: { l: 120, t: 10, b: 50, r: 20 },
        yaxis: { autorange: 'reversed' },
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

// ---- Model comparison ----
function renderModelComparison(id, data) {
    if (!data || !document.getElementById(id)) return;
    const colorMap = { naive: '#64748b', ridge: '#94a3b8', xgboost: '#f59e0b', catboost: '#10b981' };
    const metrics = ['MAPE (%)', 'MAE (EUR)', 'R²'];
    // One trace per model
    const traces = data.models.map(m => {
        const label = modelDisplayName(m);
        return {
            x: metrics,
            y: [data.mape[data.models.indexOf(m)], data.mae[data.models.indexOf(m)] / 100, data.r2[data.models.indexOf(m)]],
            name: label,
            type: 'bar',
            marker: { color: colorMap[m], opacity: 0.85 },
            hovertemplate: label + '<br>%{x}: %{customdata}<extra></extra>',
            customdata: [
                data.mape[data.models.indexOf(m)] + '%',
                fmt(data.mae[data.models.indexOf(m)]) + ' EUR',
                data.r2[data.models.indexOf(m)].toFixed(4),
            ],
        };
    });
    // Use separate charts for clarity
    const el = document.getElementById(id);
    el.innerHTML = '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;height:100%;width:100%"><div id="' + id + '-mape" style="min-width:0"></div><div id="' + id + '-mae" style="min-width:0"></div><div id="' + id + '-r2" style="min-width:0"></div></div>';

    const commonLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 12 },
        xaxis: { gridcolor: 'rgba(148,163,184,0.08)' },
        yaxis: { gridcolor: 'rgba(148,163,184,0.08)' },
        showlegend: false,
        margin: { t: 35, b: 30, l: 50, r: 10 },
    };

    const modelLabels = data.models.map(modelDisplayName);
    const colors = data.models.map(m => colorMap[m] || '#64748b');

    const hasLogRatio = Array.isArray(data.mape_logratio)
        && data.mape_logratio.some(v => v != null);

    // Legend placed BELOW the chart so it never collides with the title.
    const legendLayout = {
        showlegend: true,
        legend: {
            orientation: 'h',
            x: 0.5, xanchor: 'center',
            y: -0.22, yanchor: 'top',
            font: { size: 10, color: '#cbd5e1' },
            bgcolor: 'rgba(0,0,0,0)',
        },
    };

    // Police uniforme pour TOUS les data-labels (même taille, lisible, non encombrant).
    const labelFont = { size: 11, family: 'JetBrains Mono' };

    // MAPE — grouped bars: euros vs log_ratio (both expressed in %)
    const mapeTraces = [{
        name: 'EUR (optimiste)',
        x: modelLabels, y: data.mape, type: 'bar',
        marker: { color: colors, opacity: 0.55, line: { width: 0 } },
        text: data.mape.map(v => v + '%'), textposition: 'outside',
        textfont: { ...labelFont, color: '#cbd5e1' },
        cliponaxis: false,
        hovertemplate: '%{x} — EUR: %{y}%<extra></extra>',
    }];
    if (hasLogRatio) {
        mapeTraces.push({
            name: 'log_ratio (honnête)',
            x: modelLabels, y: data.mape_logratio, type: 'bar',
            marker: { color: colors, opacity: 1, line: { color: '#f1f5f9', width: 1.5 } },
            text: data.mape_logratio.map(v => v == null ? '' : v + '%'),
            textposition: 'outside',
            textfont: { ...labelFont, color: '#f1f5f9' },
            cliponaxis: false,
            hovertemplate: '%{x} — log_ratio: %{y}%<extra></extra>',
        });
    }

    // Alignment strategy : tous les sous-graphes partagent
    //   - le même yaxis.domain  → plot-area de même hauteur
    //   - le même margin       → axe X à la même position pixel
    //   - des yaxis.range explicites avec 15% de headroom en haut
    //     (évite le clipping de "2 616 €" sur MAE)
    //   - cliponaxis:false sur les traces  → le texte outside ne déforme
    //     plus la plage auto-calculée par Plotly (source d'écarts entre plots)
    // Pour le R² (qui a une barre négative à -0.05), on aligne le "pied"
    // du plot-area MAPE/MAE sur le zéro du R² via un domaine raccourci.
    const r2Values = [...data.r2, ...(data.r2_logratio || [])].filter(v => v != null);
    const r2Min = Math.min(0, ...r2Values);
    const hasNeg = r2Min < 0;
    // Fraction du plot-area occupée par la partie négative dans R². On ajoute
    // 0.05 de headroom en haut côté R² pour respirer.
    const negPad = hasNeg ? 0.08 : 0;        // ex. -0.08..0 en R²
    const topPad = 0.05;                     // headroom visuel au-dessus
    const r2Range = [hasNeg ? r2Min - negPad : 0, 1 + topPad];
    // Même fraction négative reproduite comme marge « vide » en bas de MAPE/MAE,
    // ce qui pousse leurs labels de modèles exactement au même niveau que R².
    const negFrac = hasNeg ? Math.abs(r2Range[0]) / (r2Range[1] - r2Range[0]) : 0;
    const sharedDomain = [negFrac, 1];
    const sharedMargin = { t: 40, b: hasLogRatio ? 110 : 90, l: 55, r: 30 };
    const sharedXAxis = { ...commonLayout.xaxis, tickangle: -40, tickfont: { size: 10 }, automargin: false, fixedrange: true };

    const mapeMax = Math.max(...data.mape, ...(data.mape_logratio || []).filter(v => v != null));
    const mapeRange = [0, mapeMax * 1.15];   // 15% headroom au-dessus pour les labels

    Plotly.newPlot(id + '-mape', mapeTraces, {
        ...commonLayout,
        ...(hasLogRatio ? legendLayout : {}),
        barmode: 'group',
        title: { text: 'MAPE (%)', font: { size: 13, color: '#f1f5f9' }, x: 0.5, xanchor: 'center', y: 0.97, yanchor: 'top' },
        xaxis: { ...sharedXAxis, domain: [0, 1] },
        yaxis: { ...commonLayout.yaxis, title: '', domain: sharedDomain, range: mapeRange, automargin: false, fixedrange: true },
        margin: sharedMargin,
    }, PLOTLY_CONFIG);

    // MAE — euros only (log_ratio MAE is in log units, not on the same scale).
    const maeMax = Math.max(...data.mae);
    const maeRange = [0, maeMax * 1.15];     // ~15% headroom pour "2 616 €"
    Plotly.newPlot(id + '-mae', [{
        name: 'EUR (optimiste)',
        x: modelLabels, y: data.mae, type: 'bar',
        marker: { color: colors, opacity: 0.85, line: { width: 0 } },
        text: data.mae.map(v => fmt(v) + ' €'), textposition: 'outside',
        textfont: { ...labelFont, color: '#f1f5f9' },
        cliponaxis: false,
        hovertemplate: '%{x}: %{y:,.0f} EUR<extra></extra>',
    }], {
        ...commonLayout,
        ...(hasLogRatio ? legendLayout : {}),
        title: { text: 'MAE (EUR)', font: { size: 13, color: '#f1f5f9' }, x: 0.5, xanchor: 'center', y: 0.97, yanchor: 'top' },
        xaxis: { ...sharedXAxis, domain: [0, 1] },
        yaxis: { ...commonLayout.yaxis, title: '', domain: sharedDomain, range: maeRange, automargin: false, fixedrange: true },
        margin: sharedMargin,
    }, PLOTLY_CONFIG);

    // R² — grouped bars: euros vs log_ratio
    const r2Traces = [{
        name: 'EUR (optimiste)',
        x: modelLabels, y: data.r2, type: 'bar',
        marker: { color: colors, opacity: 0.55, line: { width: 0 } },
        text: data.r2.map(v => v.toFixed(3)), textposition: 'outside',
        textfont: { ...labelFont, color: '#cbd5e1' },
        cliponaxis: false,
        hovertemplate: '%{x} — EUR: R² = %{y:.4f}<extra></extra>',
    }];
    if (hasLogRatio && Array.isArray(data.r2_logratio)) {
        r2Traces.push({
            name: 'log_ratio (honnête)',
            x: modelLabels, y: data.r2_logratio, type: 'bar',
            marker: { color: colors, opacity: 1, line: { color: '#f1f5f9', width: 1.5 } },
            text: data.r2_logratio.map(v => v == null ? '' : v.toFixed(3)),
            textposition: 'outside',
            textfont: { ...labelFont, color: '#f1f5f9' },
            cliponaxis: false,
            hovertemplate: '%{x} — log_ratio: R² = %{y:.4f}<extra></extra>',
        });
    }
    // R² utilise le domaine VERTICAL COMPLET [0, 1] : sa "zone négative"
    // (en bas du plot-area) aligne son axe-X au même pixel que MAPE/MAE,
    // qui eux ont leur pied relevé via sharedDomain=[negFrac, 1].
    Plotly.newPlot(id + '-r2', r2Traces, {
        ...commonLayout,
        ...(hasLogRatio ? legendLayout : {}),
        barmode: 'group',
        title: { text: 'R²', font: { size: 13, color: '#f1f5f9' }, x: 0.5, xanchor: 'center', y: 0.97, yanchor: 'top' },
        xaxis: { ...sharedXAxis, domain: [0, 1] },
        yaxis: { ...commonLayout.yaxis, title: '', domain: [0, 1], range: r2Range, automargin: false, fixedrange: true, zeroline: true, zerolinecolor: 'rgba(148,163,184,0.3)' },
        margin: sharedMargin,
    }, PLOTLY_CONFIG);
}

function renderMetricsTable() {
    const el = document.getElementById('metrics-table');
    if (!el || !DATA.meta.metrics) return;
    const metrics = DATA.meta.metrics;
    const best = DATA.meta.best_model;

    // Index log_ratio metrics by model name for quick lookup.
    const lrByModel = {};
    for (const r of (DATA.meta.metrics_logratio || [])) {
        lrByModel[r.model] = r;
    }
    const hasLr = Object.keys(lrByModel).length > 0;

    // Check if top 2 models are within 0.5pp MAPE → ensemble
    const sorted = [...metrics].sort((a, b) => a['MAPE (%)'] - b['MAPE (%)']);
    const isEnsemble = sorted.length >= 2 && (sorted[1]['MAPE (%)'] - sorted[0]['MAPE (%)']) < 0.5;
    const ensembleModels = isEnsemble ? [sorted[0].model, sorted[1].model] : [best];

    const colspanIfEnsemble = hasLr ? 5 : 3;
    let html = '<table><thead><tr>'
        + '<th>Modèle</th>'
        + '<th>MAE (EUR)</th>'
        + '<th>MAPE (EUR)</th>'
        + '<th>R² (EUR)</th>';
    if (hasLr) {
        html += '<th>MAPE (log_ratio)</th><th>R² (log_ratio)</th>';
    }
    html += '</tr></thead><tbody>';
    for (const m of metrics) {
        const inEnsemble = ensembleModels.includes(m.model);
        const cls = inEnsemble ? ' class="best-row"' : '';
        const label = modelDisplayName(m.model);
        const lr = lrByModel[m.model] || {};
        const bestValClass = inEnsemble ? ' class="best-value"' : '';
        html += `<tr${cls}>`;
        html += `<td>${inEnsemble ? '<strong>' : ''}${label}${inEnsemble ? ' ✓' : ''}${inEnsemble ? '</strong>' : ''}</td>`;
        html += `<td${bestValClass}>${fmt(m['MAE (€)'])}</td>`;
        html += `<td${bestValClass}>${m['MAPE (%)']}%</td>`;
        html += `<td${bestValClass}>${m['R²']}</td>`;
        if (hasLr) {
            const mapeLr = lr['MAPE (log_ratio) %'];
            const r2Lr = lr['R² (log_ratio)'];
            html += `<td${bestValClass}>${mapeLr != null ? mapeLr + '%' : '—'}</td>`;
            html += `<td${bestValClass}>${r2Lr != null ? r2Lr : '—'}</td>`;
        }
        html += '</tr>';
    }
    if (isEnsemble) {
        html += `<tr class="best-row" style="border-top:2px solid var(--teal)"><td><strong>Ensemble retenu</strong></td><td colspan="${colspanIfEnsemble}" style="text-align:center;color:var(--teal)">Moyenne des prédictions ${ensembleModels.map(modelDisplayName).join(' + ')}</td></tr>`;
    }
    html += '</tbody></table>';
    if (hasLr) {
        html += '<div class="metrics-lecture">'
            + '<div class="metrics-lecture__head">'
            + '<span class="metrics-lecture__tag">Lecture</span>'
            + '<span class="metrics-lecture__title">Comment lire les deux échelles de la table</span>'
            + '</div>'
            + '<div class="metrics-lecture__grid">'
            + '<div class="metrics-lecture__col">'
            + '<span class="metrics-lecture__col-tag metrics-lecture__col-tag--eur">Colonnes EUR</span>'
            + '<p>Calculées sur le prix reconstruit <code>V_t = V_0 · exp(log_ratio_hat)</code>. Bénéficient de la variance apportée par <code>prix_catalogue</code> (feature ET facteur de back-transform).</p>'
            + '</div>'
            + '<div class="metrics-lecture__col">'
            + '<span class="metrics-lecture__col-tag metrics-lecture__col-tag--lr">Colonnes log_ratio</span>'
            + '<p>Skill du modèle sur la cible brute <code>log(V_t / V_0)</code> — l\'échelle honnête à défendre en soutenance.</p>'
            + '</div>'
            + '</div>'
            + '<p class="metrics-lecture__summary">L\'écart EUR ↔ log_ratio reflète l\'inflation mécanique apportée par V₀.</p>'
            + '</div>';
        const nb = DATA.meta.naive_baseline;
        if (nb) {
            html += '<div class="naive-baseline-card">'
                + '<div class="naive-baseline-card__head">'
                + '<span class="naive-baseline-card__tag">Baseline naïf — k·V₀</span>'
                + '<span class="naive-baseline-card__title">Régression OLS sans constante sur le train</span>'
                + '</div>'
                + '<p class="naive-baseline-card__def">Minimise la MSE sous la contrainte <code>V_t = k · V_0</code>. Solution fermée : <code>k_MSE = Σ(V₀·y) / Σ(V₀²) = ' + nb.k_mse + '</code>.</p>'
                + '<div class="naive-baseline-card__stats">'
                + '<div class="naive-baseline-card__stat">'
                + '<span class="naive-baseline-card__stat-label">Sur les EUR</span>'
                + '<span class="naive-baseline-card__stat-value">R² = ' + nb.eur['R²'] + '</span>'
                + '<span class="naive-baseline-card__stat-sub">MAPE ' + nb.eur['MAPE (%)'] + '% — contribution mécanique de V₀</span>'
                + '</div>'
                + '<div class="naive-baseline-card__stat">'
                + '<span class="naive-baseline-card__stat-label">Sur le log_ratio</span>'
                + '<span class="naive-baseline-card__stat-value">R² = ' + nb.log_ratio['R² (log_ratio)'] + '</span>'
                + '<span class="naive-baseline-card__stat-sub">constante <code>log(k_MSE) = ' + nb.log_k + '</code> — quasi-zéro, une constante est à peine pire que la moyenne</span>'
                + '</div>'
                + '</div>'
                + '<p class="naive-baseline-card__takeaway">L\'écart entre ce baseline et CatBoost sur le log_ratio est donc <strong>la vraie skill apprise</strong>.</p>'
                + '</div>';
        }
    }
    el.innerHTML = html;
}

// ---- Feature importance ----
function renderFeatureImportance(id, data) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        y: data.features.slice().reverse(),
        x: data.importance.slice().reverse(),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: data.importance.slice().reverse().map((v, i, arr) =>
                i === arr.length - 1 ? '#4BC4BD' : '#3b82f6'
            ),
            opacity: 0.85,
        },
        hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
    };
    const layout = mergeLayout({
        xaxis: { title: 'Importance' },
        margin: { l: 180, t: 10, b: 50, r: 20 },
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

// ---- Pred vs Obs ----
function renderPredVsObs(id, data) {
    if (!data || !document.getElementById(id)) return;
    const maxVal = Math.max(...data.true, ...data.pred);
    const trace = {
        x: data.true,
        y: data.pred,
        type: 'scatter',
        mode: 'markers',
        marker: { color: '#3b82f6', size: 3, opacity: 0.3 },
        hovertemplate: 'Observé: %{x:,.0f} EUR<br>Prédit: %{y:,.0f} EUR<extra></extra>',
    };
    // Generate many points along the diagonal for hover
    const nPts = 50;
    const diagX = Array.from({length: nPts}, (_, i) => (i / (nPts - 1)) * maxVal);
    const diagY = diagX.slice();
    const line = {
        x: diagX,
        y: diagY,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#ef4444', width: 2, dash: 'dash' },
        marker: { size: 12, color: 'rgba(0,0,0,0)' },
        name: 'Diagonale parfaite',
        hoverlabel: { bgcolor: '#7f1d1d', bordercolor: '#ef4444', font: { color: '#fca5a5' } },
        hovertemplate: 'Diagonale de prédiction parfaite<br>(prédit = observé)<extra></extra>',
    };
    const layout = mergeLayout({
        xaxis: { title: 'Prix observé (EUR)' },
        yaxis: { title: 'Prix predit (EUR)' },
        showlegend: false,
    });
    Plotly.newPlot(id, [trace, line], layout, PLOTLY_CONFIG);
}

// ---- Portfolio charts ----
function renderPortfolioAge(id, data) {
    if (!data || !document.getElementById(id)) return;
    const brands = [...new Set(data.brand)];
    const traces = brands.map(b => {
        const idx = data.brand.map((br, i) => br === b ? i : -1).filter(i => i >= 0);
        return {
            x: idx.map(i => data.x[i]),
            y: idx.map(i => data.y[i]),
            type: 'scatter',
            mode: 'markers',
            name: b,
            marker: { size: 5, opacity: 0.6 },
            hovertemplate: `${b}<br>Age: %{x:.1f} ans<br>VR: %{y:,.0f} EUR<extra></extra>`,
        };
    });
    const layout = mergeLayout({
        xaxis: { title: 'Âge à la fin du contrat (années)' },
        yaxis: { title: 'Valeur résiduelle prédite (EUR)' },
        legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
    });
    Plotly.newPlot(id, traces, layout, PLOTLY_CONFIG);
}

function renderPortfolioBrand(id, data) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        x: data.median,
        y: data.brands,
        type: 'bar',
        orientation: 'h',
        marker: { color: '#3b82f6', opacity: 0.85 },
        text: data.count.map(c => `n=${c}`),
        textposition: 'auto',
        textfont: { size: 11, color: '#f1f5f9' },
        hovertemplate: '%{y}<br>VR médiane: %{x:,.0f} EUR<extra></extra>',
    };
    const layout = mergeLayout({
        xaxis: { title: 'VR médiane prédite (EUR)' },
        margin: { l: 100, t: 10, b: 50, r: 20 },
        yaxis: { autorange: 'reversed' },
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

function renderPortfolioFuel(id, data) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        x: data.fuels,
        y: data.median_decote,
        type: 'bar',
        marker: {
            color: data.median_decote.map(d => d > 50 ? '#3b82f6' : d > 40 ? '#f59e0b' : '#4BC4BD'),
            opacity: 0.85,
        },
        text: data.median_decote.map(d => d + '%'),
        textposition: 'outside',
        textfont: { color: '#f1f5f9', size: 12 },
        hovertemplate: '%{x}<br>Décote médiane: %{y:.1f}%<br>n=%{customdata}<extra></extra>',
        customdata: data.count,
    };
    const layout = mergeLayout({
        xaxis: { title: 'Type de carburant', tickangle: -30 },
        yaxis: { title: 'Décote médiane (%)' },
        margin: { t: 30, b: 80 },
    });
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

function renderFamilyStats(data) {
    const el = document.getElementById('family-stats-table');
    if (!el || !data) return;

    let html = '<table><thead><tr><th>Famille</th><th>Véhicules</th><th>Décote médiane (%)</th><th>VR médiane (EUR)</th><th>Modèles principaux</th></tr></thead><tbody>';
    for (const f of data) {
        html += `<tr>`;
        html += `<td><strong>${f.family}</strong></td>`;
        html += `<td>${f.count}</td>`;
        html += `<td>${f.median_decote}%</td>`;
        html += `<td>${fmt(f.median_pred)}</td>`;
        html += `<td style="font-size:0.82rem;color:#64748b">${f.models}</td>`;
        html += `</tr>`;
    }
    html += '</tbody></table>';
    el.innerHTML = html;
}

// ---- Tabs ----
function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            btn.closest('.section').querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.closest('.section').querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('tab-' + tab).classList.add('active');
            // Resize charts (Plotly needs this after visibility change)
            setTimeout(() => {
                document.getElementById('tab-' + tab).querySelectorAll('.chart-container').forEach(c => {
                    Plotly.Plots.resize(c);
                });
            }, 50);
        });
    });
}

// ---- Sidebar preview popover (hover sur chaque nav-step) ----
const NAV_PREVIEW = {
    hero:         { title: 'Accueil',                title_en: 'Home',
                    text: "Page d'accueil : titre du projet, executive summary 3 bullets, KPIs primaires & audit-proof, présentation guidée 3 min.",
                    text_en: "Landing page: project title, 3-bullet executive summary, primary & audit-proof KPIs, 3-min guided tour." },
    context:      { title: 'Contexte Métier',        title_en: 'Business Context',
                    text: "Pourquoi la VR est cruciale pour [Client] : définition VR + enjeux financiers (sur/sous-estimation).",
                    text_en: "Why RV matters for [Client]: definition + financial stakes (over/under-estimation)." },
    eda:          { title: 'Analyse Exploratoire',   title_en: 'Exploratory Analysis',
                    text: "Transactions analysées sur 3 onglets : variables numériques, catégorielles, matrice de corrélation.",
                    text_en: "Transactions across 3 tabs: numeric variables, categorical, correlation matrix." },
    hicp:         { title: 'Variables Macro (HICP)', title_en: 'Macro Variables (HICP)',
                    text: "Innovation macroéconomique : intégration HICP + tests de robustesse Brent crude oil et tension marché KBA.",
                    text_en: "Macro innovation: HICP integration + Brent crude and KBA market tension robustness tests." },
    features:     { title: 'Feature Engineering',    title_en: 'Feature Engineering',
                    text: "Transformations log + clustering KMeans des 62 modèles en familles homogènes de dépréciation.",
                    text_en: "Log transformations + KMeans clustering of 62 models into homogeneous depreciation families." },
    depreciation: { title: 'Dépréciation',           title_en: 'Depreciation',
                    text: "4 graphiques : courbe d'âge convexe, prix par marque, par carburant, et par année de vente.",
                    text_en: "4 charts: convex age curve, price by brand, by fuel, by sale year." },
    results:      { title: 'Évaluation des Modèles', title_en: 'Model Evaluation',
                    text: "4 modèles testés (Ridge, RandomForest, XGBoost, CatBoost) + baseline naïve k·V₀. Walk-forward 3 cutoffs, tuning Optuna, conformal 80/90/95, prédit vs observé, résidus.",
                    text_en: "4 models tested (Ridge, RandomForest, XGBoost, CatBoost) + k·V₀ naive baseline. Walk-forward 3 cutoffs, Optuna tuning, conformal 80/90/95, predicted vs observed, residuals." },
    houssem:      { title: 'Pipeline Structuré',     title_en: 'Structured Pipeline',
                    text: "Validation indépendante : split aléatoire stratifié, target log_ratio, propensity reweighting, monotonic constraints, SHAP global. Verdict comparé à la section 06.",
                    text_en: "Independent validation: stratified random split, log_ratio target, propensity reweighting, monotonic constraints, global SHAP. Verdict cross-checked vs section 06." },
    portfolio:    { title: 'Prédictions Portfolio',  title_en: 'Portfolio Predictions',
                    text: "Véhicules prédits : distribution prix, décote, familles de dépréciation, validation cohérence.",
                    text_en: "Vehicles predicted: price distribution, depreciation, families, coherence validation." },
    stress:       { title: 'Stress Test 49/51',      title_en: 'Stress Test 49/51',
                    text: "Split temporel train 49% / test 51% chronologique post-COVID. Mesure la tenue du modèle sur un horizon de généralisation étendu — ranking conservé = robustesse.",
                    text_en: "Chronological post-COVID split train 49% / test 51%. Measures model stability on an extended generalization horizon — ranking preserved = robustness." },
    validation:   { title: 'Validation Externe',     title_en: 'External Validation',
                    text: "Scraping AutoScout24 temps réel. Modèles confrontés, écart médian X % (signature B2B vs B2C).",
                    text_en: "Real-time AutoScout24 scraping. Models confronted, X% median gap (B2B vs B2C signature)." },
    risk:         { title: 'Analyse de Risque',      title_en: 'Risk Analysis',
                    text: "Exposition portefeuille ~20 M€. Stress tests BCE/EBA + top 10 véhicules à risque actionnable.",
                    text_en: "Portfolio exposure ~€20M. ECB/EBA stress tests + top 10 actionable at-risk vehicles." },
    simulator:    { title: 'Simulateur',             title_en: 'Simulator',
                    text: "Outil interactif : entrez les caractéristiques d'un véhicule, le modèle prédit la VR en temps réel.",
                    text_en: "Interactive tool: enter vehicle characteristics, the model predicts RV in real time." },
};

function setupNavPreview() {
    const popover = document.getElementById('nav-preview');
    if (!popover) return;
    const titleEl = popover.querySelector('.nav-preview__title');
    const textEl = popover.querySelector('.nav-preview__text');
    let hideT = null;

    document.querySelectorAll('.nav-step').forEach(step => {
        step.addEventListener('mouseenter', () => {
            const id = step.getAttribute('data-section');
            const data = NAV_PREVIEW[id];
            if (!data) return;
            clearTimeout(hideT);
            const isEn = (typeof currentLang !== 'undefined' && currentLang === 'en');
            titleEl.textContent = isEn && data.title_en ? data.title_en : data.title;
            textEl.textContent = isEn && data.text_en ? data.text_en : data.text;
            // Position : aligné sur le top du step, à droite de la sidebar avec gap.
            const rect = step.getBoundingClientRect();
            popover.style.top = `${Math.max(8, rect.top - 4)}px`;
            popover.style.left = `${rect.right + 14}px`;
            popover.classList.add('is-visible');
        });
        step.addEventListener('mouseleave', () => {
            hideT = setTimeout(() => popover.classList.remove('is-visible'), 80);
        });
    });
}

// ---- Navigation scroll spy + auto-open du collapse au clic sidebar ----
function setupNavigation() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-links a');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                navLinks.forEach(link => link.classList.remove('active'));
                const activeLink = document.querySelector(`.nav-links a[data-section="${entry.target.id}"]`);
                if (activeLink) activeLink.classList.add('active');

                // Ferme toute modale .eda-bubble/.conclusion-collapse dont la
                // section d'origine n'est plus celle affichée — sinon la bulle
                // position:fixed reste centrée par-dessus la nouvelle section.
                const openTrig = document.querySelector(MODAL_TRIGGER_SELECTOR);
                if (openTrig) {
                    const openSection = openTrig.closest('section.section');
                    if (openSection && openSection !== entry.target) {
                        _closeCollapse(openTrig);
                    }
                }
            }
        });
    }, { rootMargin: '-30% 0px -70% 0px' });

    sections.forEach(section => observer.observe(section));

    // Auto-open du collapse section-explore quand on clique un step de la sidebar.
    // L'utilisateur arrive sur la section avec son contenu déjà déployé — pas de void.
    document.querySelectorAll('.nav-step').forEach(link => {
        link.addEventListener('click', () => {
            // Ferme toute modale .eda-bubble / .conclusion-collapse ouverte avant
            // de naviguer — sinon la bulle fixed reste centrée sur la nouvelle page.
            const openTrig = document.querySelector(MODAL_TRIGGER_SELECTOR);
            if (openTrig) _closeCollapse(openTrig);

            const sectionId = link.getAttribute('data-section');
            if (!sectionId) return;
            const trigger = document.querySelector(
                `.collapse-card.section-explore .collapse-card__trigger[aria-controls="explore-${sectionId}-body"]`
            );
            if (trigger && trigger.getAttribute('aria-expanded') !== 'true') {
                _openCollapse(trigger);
            }
        });
    });
}

// ---- Scroll animations ----
function setupScrollAnimations() {
    const cards = document.querySelectorAll('.card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // Resize any Plotly charts inside
                entry.target.querySelectorAll('.chart-container').forEach(c => {
                    try { Plotly.Plots.resize(c); } catch(e) {}
                });
            }
        });
    }, { threshold: 0.1 });

    cards.forEach(card => {
        card.classList.add('fade-in');
        observer.observe(card);
    });
}

// ---- Simulator ----
// Presets rapides : sélectionne la 1re option correspondante si disponible,
// sinon retombe sur la 1re option du select (évite les valeurs invalides).
const SIM_PRESETS = {
    clio:   { brand: /renault/i, fuel: /petrol|essence/i, range: /PC|passenger/i,
              prix: 22000, prod: 2022, end: 2027, initKm: 0, contractKm: 60000 },
    duster: { brand: /dacia/i,   fuel: /diesel/i,          range: /PC|passenger/i,
              prix: 24000, prod: 2021, end: 2026, initKm: 0, contractKm: 90000 },
    ev:     { brand: /renault/i, fuel: /electric/i,        range: /PC|passenger/i,
              prix: 35000, prod: 2023, end: 2028, initKm: 0, contractKm: 50000 },
};
function _setSelectMatch(id, regex) {
    const sel = document.getElementById(id);
    if (!sel) return;
    const opt = Array.from(sel.options).find(o => o.value && regex.test(o.value));
    if (opt) sel.value = opt.value;
}
function _applyPreset(name) {
    const p = SIM_PRESETS[name];
    if (!p) return;
    _setSelectMatch('sim-brand', p.brand);
    _setSelectMatch('sim-fuel',  p.fuel);
    _setSelectMatch('sim-range', p.range);
    document.getElementById('sim-catalogue').value   = p.prix;
    document.getElementById('sim-prod-year').value   = p.prod;
    document.getElementById('sim-end-year').value    = p.end;
    document.getElementById('sim-init-km').value     = p.initKm;
    document.getElementById('sim-contract-km').value = p.contractKm;
    // Petit highlight visuel pour signaler le pré-remplissage
    document.querySelectorAll('.sim-preset-chip').forEach(c => c.classList.remove('is-active'));
    const chip = document.querySelector(`.sim-preset-chip[data-preset="${name}"]`);
    if (chip) chip.classList.add('is-active');
}
function setupSimulator() {
    const opts = DATA.options;

    // Populate dropdowns
    populateSelect('sim-brand', opts.brands);
    populateSelect('sim-fuel', opts.fuel_types);
    populateSelect('sim-range', opts.range_types);

    // Preset chips
    document.querySelectorAll('.sim-preset-chip').forEach(chip => {
        chip.addEventListener('click', () => _applyPreset(chip.dataset.preset));
    });

    // Form submit
    document.getElementById('sim-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = e.target.querySelector('.btn-simulate');
        const btnLabel = btn.querySelector('span');
        const originalLabel = btnLabel ? btnLabel.textContent : btn.textContent;
        if (btnLabel) btnLabel.textContent = 'Calcul en cours...';
        else btn.textContent = 'Calcul en cours...';
        btn.disabled = true;

        const payload = {
            brand: document.getElementById('sim-brand').value,
            fuel_type: document.getElementById('sim-fuel').value,
            range_type: document.getElementById('sim-range').value,
            prix_catalogue: parseFloat(document.getElementById('sim-catalogue').value),
            production_year: parseInt(document.getElementById('sim-prod-year').value),
            contract_end_year: parseInt(document.getElementById('sim-end-year').value),
            initial_mileage: parseFloat(document.getElementById('sim-init-km').value),
            contract_mileage: parseFloat(document.getElementById('sim-contract-km').value),
        };

        try {
            const resp = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const result = await resp.json();

            if (result.error) {
                alert('Erreur: ' + result.error);
                return;
            }

            // Show results
            document.getElementById('sim-placeholder').style.display = 'none';
            document.getElementById('sim-results').classList.remove('hidden');

            document.getElementById('sim-price').textContent = fmt(Math.round(result.prediction));
            document.getElementById('sim-decote').textContent = result.decote_pct + '%';
            document.getElementById('sim-ratio').textContent = result.ratio_pct + '%';
            document.getElementById('sim-age').textContent = result.age_years + ' ans';
            document.getElementById('sim-km').textContent = fmt(Math.round(result.total_km)) + ' km';

            // Bande d'incertitude (backend calcule ±MAPE walk-forward autour
            // de la prédiction point). Affichée en dessous du prix principal.
            const unc = result.uncertainty || {};
            const uncEl = document.getElementById('sim-uncertainty');
            if (uncEl && unc.prediction_low != null && unc.prediction_high != null) {
                document.getElementById('sim-uncertainty-low').textContent = fmt(Math.round(unc.prediction_low));
                document.getElementById('sim-uncertainty-high').textContent = fmt(Math.round(unc.prediction_high));
                if (unc.mape_pct != null) {
                    document.getElementById('sim-uncertainty-pct').textContent = unc.mape_pct.toFixed(2);
                }
                uncEl.hidden = false;
            } else if (uncEl) {
                uncEl.hidden = true;
            }

            // Gauge chart
            renderGauge(result);
        } catch (err) {
            alert('Erreur de connexion: ' + err.message);
        } finally {
            if (btnLabel) btnLabel.textContent = originalLabel;
            else btn.textContent = originalLabel;
            btn.disabled = false;
        }
    });
}

function populateSelect(id, options) {
    const sel = document.getElementById(id);
    options.forEach(opt => {
        const o = document.createElement('option');
        o.value = opt;
        o.textContent = opt;
        sel.appendChild(o);
    });
}

function renderGauge(result) {
    const trace = {
        type: 'indicator',
        mode: 'gauge+number',
        value: result.ratio_pct,
        number: { suffix: '%', font: { size: 28, color: '#f1f5f9', family: 'JetBrains Mono' } },
        title: { text: 'Conservation de valeur', font: { size: 13, color: '#94a3b8' } },
        gauge: {
            axis: { range: [0, 100], tickfont: { color: '#64748b', size: 10 } },
            bar: { color: '#3b82f6' },
            bgcolor: 'rgba(15,23,42,0.6)',
            borderwidth: 0,
            steps: [
                { range: [0, 30], color: 'rgba(239,68,68,0.15)' },
                { range: [30, 60], color: 'rgba(245,158,11,0.15)' },
                { range: [60, 100], color: 'rgba(16,185,129,0.15)' },
            ],
        },
    };
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 40, b: 10, l: 30, r: 30 },
    };
    Plotly.newPlot('sim-gauge', [trace], layout, PLOTLY_CONFIG);
}

// Tooltips use pure CSS (position: absolute) — no JS needed

// ============================================================
// VALIDATION EXTERNE AUTOSCOUT24
// ============================================================
function renderScrapeValidation(data) {
    if (!data || !data.results) return;

    // Chiffres clés
    const md = document.getElementById('scrape-median-delta');
    if (md) md.textContent = (data.median_delta_pct > 0 ? '+' : '') + data.median_delta_pct + '%';
    const nm = document.getElementById('scrape-n-models');
    if (nm) nm.textContent = data.n_models;
    const dir = document.getElementById('scrape-direction');
    if (dir) dir.textContent = data.direction === 'conservateur' ? 'Conservateur' : 'Optimiste';

    // Bar chart comparatif
    const chartEl = document.getElementById('chart-scrape-comparison');
    if (chartEl) {
        const results = data.results;
        const labels = results.map(r => r.brand + ' ' + r.model);
        const traceBoard = {
            x: labels,
            y: results.map(r => r.pred_median),
            name: 'Prédit (modèle)',
            type: 'bar',
            marker: { color: '#3b82f6' },
            hovertemplate: '%{x}<br>Prédit : %{y:,.0f} €<extra></extra>',
        };
        const traceAS24 = {
            x: labels,
            y: results.map(r => r.as24_median),
            name: 'AutoScout24 (annonces)',
            type: 'bar',
            marker: { color: '#f59e0b' },
            hovertemplate: '%{x}<br>AS24 : %{y:,.0f} €<extra></extra>',
        };
        const layout = mergeLayout({
            barmode: 'group',
            yaxis: { title: 'Prix (EUR)' },
            xaxis: { tickangle: -30 },
            legend: { orientation: 'h', y: -0.55, x: 0.5, xanchor: 'center', yanchor: 'top' },
            margin: { t: 20, b: 115 },
            height: 440,
        });
        Plotly.newPlot(chartEl.id, [traceBoard, traceAS24], layout, PLOTLY_CONFIG);
    }

    // Tableau
    const tableEl = document.getElementById('scrape-table');
    if (tableEl) {
        let html = '<table><thead><tr><th>Marque</th><th>Modèle</th><th>Âge (ans)</th><th>Km</th><th>Portfolio n</th><th>Prédit (€)</th><th>AS24 (€)</th><th>Annonces AS24</th><th>Écart</th></tr></thead><tbody>';
        for (const r of data.results) {
            const deltaColor = Math.abs(r.delta_pct) < 15 ? 'var(--teal)' : r.delta_pct < -30 ? '#ef4444' : '#f59e0b';
            html += `<tr>`;
            html += `<td><strong>${r.brand}</strong></td>`;
            html += `<td>${r.model}</td>`;
            html += `<td>${r.age_years}</td>`;
            html += `<td>${fmt(r.mileage_km)}</td>`;
            html += `<td>${r.n_portfolio}</td>`;
            html += `<td>${fmt(r.pred_median)}</td>`;
            html += `<td>${fmt(r.as24_median)}</td>`;
            html += `<td>${r.as24_n}</td>`;
            html += `<td style="color:${deltaColor};font-weight:700">${r.delta_pct > 0 ? '+' : ''}${r.delta_pct}%</td>`;
            html += '</tr>';
        }
        html += '</tbody></table>';
        tableEl.innerHTML = html;
    }
}

// ============================================================
// ANALYSES POST-COVID (régime, conformal, walk-forward, tuning, stress 49/51, AS24 post-covid)
// ============================================================
function renderCovidRegime(data) {
    const card = document.getElementById('temporal-covid-card');
    if (!data || !data.variants || !card) {
        if (card) card.style.display = 'none';
        return;
    }
    const vs = data.variants;
    const labels = vs.map(v => v.label);
    const mapes = vs.map(v => v.mape_eur);
    const r2lrs = vs.map(v => v.r2_logratio);
    const colors = vs.map(v => v.name === data.best_variant ? '#4BC4BD' : '#64748b');
    const baseIdx = vs.findIndex(v => v.name === 'baseline_full');

    const subtitle = document.getElementById('temporal-covid-subtitle');
    if (subtitle) {
        subtitle.textContent =
            `cutoff=${data.cutoff} · ${data.covid_pct_in_train}% du train en zone COVID `
            + `(${data.covid_rows_in_train.toLocaleString()} lignes) · best=${data.best_variant}`;
    }

    const traceMape = {
        x: labels, y: mapes,
        name: 'MAPE EUR (%)',
        type: 'bar',
        marker: { color: colors },
        text: mapes.map(m => m.toFixed(2) + '%'),
        textposition: 'outside',
        yaxis: 'y',
        hovertemplate: '%{x}<br>MAPE=%{y:.2f}%<extra></extra>',
    };
    const traceR2 = {
        x: labels, y: r2lrs,
        name: 'R² log_ratio',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#3b82f6', width: 2, dash: 'dot' },
        marker: { color: '#3b82f6', size: 12 },
        yaxis: 'y2',
        hovertemplate: '%{x}<br>R²_lr=%{y:.4f}<extra></extra>',
    };
    const layout = mergeLayout({
        yaxis: {
            title: 'MAPE EUR (%)',
            range: [Math.min(...mapes) - 0.5, Math.max(...mapes) + 1.0],
        },
        yaxis2: {
            title: 'R² log_ratio',
            overlaying: 'y',
            side: 'right',
            range: [Math.min(...r2lrs) - 0.05, Math.max(...r2lrs) + 0.05],
            gridcolor: 'rgba(203,213,225,0.1)',
        },
        xaxis: { tickangle: -15 },
        legend: { orientation: 'h', y: -0.32, x: 0.5, xanchor: 'center', yanchor: 'top' },
        margin: { t: 30, b: 110, r: 60 },
    });
    Plotly.newPlot('chart-temporal-covid', [traceMape, traceR2], layout, PLOTLY_CONFIG);

    const verdict = document.getElementById('temporal-covid-verdict');
    if (verdict) {
        const parts = vs.slice(1).map(v => {
            const sign = v.delta_mape_pts > 0 ? '+' : '';
            return `${v.name}: ΔMAPE=${sign}${v.delta_mape_pts}pt, ΔR²_lr=${v.delta_r2_logratio > 0 ? '+' : ''}${v.delta_r2_logratio} (keep=${v.keep_vs_baseline})`;
        });
        verdict.textContent = `${data.verdict} Détail : ${parts.join(' · ')}.`;
    }
}

function renderConformal(data) {
    const card = document.getElementById('temporal-conformal-card');
    if (!data || !data.alphas || !card) {
        if (card) card.style.display = 'none';
        return;
    }
    const alphas = data.alphas;
    const labels = alphas.map(a => `IC${a.target_coverage_pct}`);
    const targets = alphas.map(a => a.target_coverage_pct);
    const empirical = alphas.map(a => a.empirical_coverage_pct);
    const widths = alphas.map(a => a.width_mean_pct);

    const subtitle = document.getElementById('temporal-conformal-subtitle');
    if (subtitle) {
        const pt = data.point_estimate || {};
        subtitle.textContent =
            `cutoff=${data.cutoff} · n_fit=${data.n_fit.toLocaleString()} · `
            + `n_cal=${data.n_cal.toLocaleString()} (${(100 * data.cal_fraction).toFixed(0)}%) · `
            + `n_test=${data.n_test.toLocaleString()} · point MAPE=${pt.mape_eur}% R²_lr=${pt.r2_logratio}`;
    }

    const traceTarget = {
        x: labels, y: targets,
        name: 'Cible',
        type: 'bar',
        marker: { color: 'rgba(100,116,139,0.4)' },
        text: targets.map(t => t + '%'),
        textposition: 'outside',
        yaxis: 'y',
        hovertemplate: 'Cible=%{y}%<extra></extra>',
    };
    const gapColors = alphas.map(a => {
        const g = Math.abs(a.coverage_gap_pts);
        if (g <= 2) return '#4BC4BD';
        if (g <= 5) return '#f59e0b';
        return '#ef4444';
    });
    const traceEmp = {
        x: labels, y: empirical,
        name: 'Empirique',
        type: 'bar',
        marker: { color: gapColors },
        text: empirical.map((e, i) => e.toFixed(1) + '% (' + (alphas[i].coverage_gap_pts > 0 ? '+' : '') + alphas[i].coverage_gap_pts.toFixed(1) + 'pt)'),
        textposition: 'outside',
        yaxis: 'y',
        hovertemplate: 'Empirique=%{y:.2f}%<extra></extra>',
    };
    const traceWidth = {
        x: labels, y: widths,
        name: 'Largeur moyenne (% V₀)',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#a78bfa', width: 2 },
        marker: { color: '#a78bfa', size: 12 },
        yaxis: 'y2',
        hovertemplate: '%{x}<br>width=%{y:.2f}%V₀<extra></extra>',
    };
    const layout = mergeLayout({
        yaxis: {
            title: 'Coverage (%)',
            range: [Math.min(...targets, ...empirical) - 5, 100],
        },
        yaxis2: {
            title: 'Largeur moyenne (% V₀)',
            overlaying: 'y',
            side: 'right',
            range: [0, Math.max(...widths) * 1.3],
            gridcolor: 'rgba(203,213,225,0.1)',
        },
        barmode: 'group',
        legend: { orientation: 'h', y: -0.22, x: 0.5, xanchor: 'center', yanchor: 'top' },
        margin: { t: 30, b: 70, r: 60 },
    });
    Plotly.newPlot('chart-temporal-conformal', [traceTarget, traceEmp, traceWidth], layout, PLOTLY_CONFIG);

    const verdict = document.getElementById('temporal-conformal-verdict');
    if (verdict) {
        const parts = alphas.map(a =>
            `IC${a.target_coverage_pct}: couverture=${a.empirical_coverage_pct}% (gap ${a.coverage_gap_pts > 0 ? '+' : ''}${a.coverage_gap_pts}pt, largeur ${a.width_mean_pct}%V₀)`
        );
        verdict.textContent = `${data.verdict} ${parts.join(' · ')}.`;
    }
}

function renderRollingOriginPostCovid(data) {
    const card = document.getElementById('temporal-ropc-card');
    if (!data || !data.folds || !card) {
        if (card) card.style.display = 'none';
        return;
    }
    const folds = data.folds.filter(f => !f.skipped);
    if (folds.length === 0) { card.style.display = 'none'; return; }

    const summary = data.summary || {};
    const perModel = summary.per_model || {};
    const bestGlobal = summary.best_model_global;
    const modelsList = data.models || Object.keys(folds[0].models || {});

    const subtitle = document.getElementById('temporal-ropc-subtitle');
    if (subtitle) {
        const bestSum = perModel[bestGlobal] || {};
        const lift = summary.lift_pts_mean;
        subtitle.textContent =
            `train >= ${data.post_covid_start} · horizon ${data.horizon_months} mois · `
            + `${folds.length} folds · ${modelsList.length} modèles `
            + `· best global: ${modelDisplayName(bestGlobal)} MAPE mean=${bestSum.mape_eur_mean}% `
            + `(std ${bestSum.mape_eur_std}pt)`
            + (lift != null ? ` · lift vs naïf_group = ${lift > 0 ? '+' : ''}${lift}pt` : '');
    }

    const labels = folds.map(f => f.cutoff);
    const naiveGroup = folds.map(f => f.naive && f.naive.naive_group ? f.naive.naive_group.mape_eur : null);

    const traces = modelsList.map(name => ({
        x: labels,
        y: folds.map(f => f.models[name] ? f.models[name].mape_eur : null),
        name: modelDisplayName(name),
        type: 'bar',
        marker: { color: MODEL_COLOR_MAP[name] || '#64748b' },
        text: folds.map(f => f.models[name] ? f.models[name].mape_eur.toFixed(2) + '%' : ''),
        textposition: 'outside',
        cliponaxis: false,
        textfont: { size: 10, family: 'JetBrains Mono', color: '#f1f5f9' },
        hovertemplate: '%{x} — ' + modelDisplayName(name) + ': %{y:.2f}%<extra></extra>',
    }));

    const traceNaive = {
        x: labels, y: naiveGroup,
        name: 'Naïf (brand × age)',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#ef4444', dash: 'dash', width: 1.5 },
        marker: { color: '#ef4444', size: 10 },
        hovertemplate: 'cutoff=%{x} — naïf_group : %{y:.2f}%<extra></extra>',
    };

    const layout = mergeLayout({
        yaxis: { title: 'MAPE (%)', rangemode: 'tozero' },
        xaxis: { title: 'Cutoff' },
        barmode: 'group',
        legend: { orientation: 'h', y: -0.22, x: 0.5, xanchor: 'center', yanchor: 'top' },
        margin: { t: 30, b: 80, r: 30 },
    });
    Plotly.newPlot('chart-temporal-ropc', [...traces, traceNaive], layout, PLOTLY_CONFIG);

    // Rendu table per-model summary (mean / std / range / R²)
    const tableEl = document.getElementById('temporal-ropc-summary-table');
    if (tableEl) {
        let html = '<table><thead><tr>'
            + '<th>Modèle</th><th>MAPE mean</th><th>MAPE std</th>'
            + '<th>MAPE min</th><th>MAPE max</th>'
            + '<th>R² EUR mean</th><th>R² log_ratio mean</th>'
            + '</tr></thead><tbody>';
        for (const name of modelsList) {
            const s = perModel[name] || {};
            const cls = name === bestGlobal ? ' class="best-row"' : '';
            html += `<tr${cls}>`
                + `<td><strong>${modelDisplayName(name)}</strong></td>`
                + `<td>${s.mape_eur_mean != null ? s.mape_eur_mean + '%' : '--'}</td>`
                + `<td>${s.mape_eur_std != null ? s.mape_eur_std + 'pt' : '--'}</td>`
                + `<td>${s.mape_eur_min != null ? s.mape_eur_min + '%' : '--'}</td>`
                + `<td>${s.mape_eur_max != null ? s.mape_eur_max + '%' : '--'}</td>`
                + `<td>${s.r2_eur_mean != null ? (s.r2_eur_mean * 100).toFixed(2) + '%' : '--'}</td>`
                + `<td>${s.r2_logratio_mean != null ? (s.r2_logratio_mean * 100).toFixed(2) + '%' : '--'}</td>`
                + '</tr>';
        }
        html += '</tbody></table>';
        tableEl.innerHTML = html;
    }

    const verdict = document.getElementById('temporal-ropc-verdict');
    if (verdict) {
        const parts = folds.map(f => `${f.cutoff}: best=${modelDisplayName(f.best_model)} ${f.best_mape_eur.toFixed(2)}%`);
        verdict.textContent = `${data.verdict} · ${parts.join(' · ')}.`;
    }
}

function renderTuningVsBaseline(data) {
    const card = document.getElementById('temporal-tuning-card');
    if (!data || !data.baseline || !data.tuned || !data.diff || !card) {
        if (card) card.style.display = 'none';
        return;
    }
    const diff = data.diff;
    const baseline = data.baseline;
    const tuned = data.tuned;
    const modelsList = baseline.models || [];

    // Subtitle
    const subtitle = document.getElementById('temporal-tuning-subtitle');
    if (subtitle) {
        subtitle.textContent =
            `${data.inner_cv || 'TSSplit(3)'} · tuning train cutoff=${data.tuning_train_cutoff} · `
            + `${data.n_trials} trials × {CatBoost, XGBoost} · ${data.device || 'GPU'} · `
            + `Ridge/RF gardés en baseline paramétrique`;
    }

    // Plot : MAPE mean par modèle, barres groupées baseline vs tuned
    const traceBaseline = {
        x: modelsList.map(m => modelDisplayName(m)),
        y: modelsList.map(m => diff.per_model[m].mape_eur_baseline),
        name: 'Baseline',
        type: 'bar',
        marker: { color: '#64748b' },
        text: modelsList.map(m => diff.per_model[m].mape_eur_baseline.toFixed(2) + '%'),
        textposition: 'outside',
        textfont: { size: 10, family: 'JetBrains Mono', color: '#cbd5e1' },
        cliponaxis: false,
        hovertemplate: '%{x} — baseline MAPE mean: %{y:.2f}%<extra></extra>',
    };
    const traceTuned = {
        x: modelsList.map(m => modelDisplayName(m)),
        y: modelsList.map(m => diff.per_model[m].mape_eur_tuned),
        name: 'Tuné (Optuna)',
        type: 'bar',
        marker: { color: '#3b82f6' },
        text: modelsList.map(m => {
            const d = diff.per_model[m].mape_eur_delta;
            return (d > 0 ? '+' : '') + d.toFixed(2) + 'pt';
        }),
        textposition: 'outside',
        textfont: { size: 10, family: 'JetBrains Mono', color: '#93c5fd' },
        cliponaxis: false,
        hovertemplate: '%{x} — tuné MAPE mean: %{y:.2f}%<extra></extra>',
    };

    const layout = mergeLayout({
        yaxis: { title: 'MAPE mean (%) — walk-forward 3 folds', rangemode: 'tozero' },
        xaxis: { title: { text: 'Modèle', standoff: 15 } },
        barmode: 'group',
        legend: { orientation: 'h', y: -0.32, x: 0.5, xanchor: 'center', yanchor: 'top' },
        margin: { t: 30, b: 110, r: 30 },
    });
    Plotly.newPlot('chart-temporal-tuning', [traceBaseline, traceTuned], layout, PLOTLY_CONFIG);

    // Table summary (mean par modèle, baseline vs tuné, Δ)
    const tableEl = document.getElementById('temporal-tuning-summary-table');
    if (tableEl) {
        const fmtDelta = (v, unit) => {
            if (v == null) return '--';
            const sign = v > 0 ? '+' : '';
            const cls = v < -0.01 ? 'delta-pos' : v > 0.01 ? 'delta-neg' : '';
            return `<span class="${cls}">${sign}${v.toFixed(unit === 'pt' ? 2 : 4)}${unit === 'pt' ? 'pt' : ''}</span>`;
        };
        let html = '<table><thead><tr>'
            + '<th>Modèle</th>'
            + '<th>MAPE baseline</th><th>MAPE tuné</th><th>Δ MAPE</th>'
            + '<th>R²(EUR) baseline</th><th>R²(EUR) tuné</th><th>Δ R²(EUR)</th>'
            + '<th>R²(logr) baseline</th><th>R²(logr) tuné</th><th>Δ R²(logr)</th>'
            + '</tr></thead><tbody>';
        for (const name of modelsList) {
            const d = diff.per_model[name];
            const isTuned = (name === 'catboost' || name === 'xgboost');
            const cls = name === diff.best_model_tuned ? ' class="best-row"' : '';
            html += `<tr${cls}>`
                + `<td><strong>${modelDisplayName(name)}</strong>${isTuned ? '' : ' <em style="color:#94a3b8">(baseline only)</em>'}</td>`
                + `<td>${d.mape_eur_baseline.toFixed(2)}%</td>`
                + `<td>${d.mape_eur_tuned.toFixed(2)}%</td>`
                + `<td>${fmtDelta(d.mape_eur_delta, 'pt')}</td>`
                + `<td>${(d.r2_eur_baseline * 100).toFixed(2)}%</td>`
                + `<td>${(d.r2_eur_tuned * 100).toFixed(2)}%</td>`
                + `<td>${fmtDelta(d.r2_eur_delta, '')}</td>`
                + `<td>${(d.r2_logratio_baseline * 100).toFixed(2)}%</td>`
                + `<td>${(d.r2_logratio_tuned * 100).toFixed(2)}%</td>`
                + `<td>${fmtDelta(d.r2_logratio_delta, '')}</td>`
                + '</tr>';
        }
        html += '</tbody></table>';
        tableEl.innerHTML = html;
    }

    // Table HP tunés (CatBoost + XGBoost)
    // Ajoute CV MAPE std (inner TSSplit(3)) — critère stabilité pour arbitrer
    // entre CB et XGB quand l'écart de MAPE mean tombe dans le bruit.
    const hpEl = document.getElementById('temporal-tuning-hp-table');
    if (hpEl && data.tuning) {
        const hp = data.tuning;
        const tunedNames = ['catboost', 'xgboost'].filter(n => hp[n]);
        const stdLookup = {};
        tunedNames.forEach(n => {
            const v = hp[n].cv_mape_std_tuned;
            if (v != null) stdLookup[n] = v;
        });
        const minStd = Object.values(stdLookup).length
            ? Math.min(...Object.values(stdLookup))
            : null;

        let html = '<table><thead><tr>'
            + '<th>Modèle</th><th>Best HP</th>'
            + '<th>Baseline CV MAPE (mean ± std)</th>'
            + '<th>Best CV MAPE (mean ± std)</th>'
            + '<th>Δ CV MAPE mean</th><th>Temps tuning</th>'
            + '</tr></thead><tbody>';
        for (const name of tunedNames) {
            const t = hp[name];
            const hpStr = Object.entries(t.best_params).map(([k, v]) => {
                const vStr = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : v;
                return `${k}=${vStr}`;
            }).join(', ');
            const baseMean = t.cv_mape_mean_baseline != null ? t.cv_mape_mean_baseline : t.baseline_cv_mape;
            const baseStd  = t.cv_mape_std_baseline;
            const tunedMean = t.cv_mape_mean_tuned != null ? t.cv_mape_mean_tuned : t.best_cv_mape;
            const tunedStd  = t.cv_mape_std_tuned;
            const deltaCv = tunedMean - baseMean;
            const sign = deltaCv > 0 ? '+' : '';
            const cls = deltaCv < -0.01 ? 'delta-pos' : deltaCv > 0.01 ? 'delta-neg' : '';
            // Surligne la ligne du modèle le plus stable (std min)
            const isMinStd = minStd != null && tunedStd != null && Math.abs(tunedStd - minStd) < 1e-6;
            const rowCls = isMinStd ? ' class="best-row"' : '';
            const stdCell = (m, s) =>
                (m != null ? m.toFixed(2) + '%' : '--')
                + (s != null ? ` <span style="color:#94a3b8">± ${s.toFixed(2)}pt</span>` : '');
            html += `<tr${rowCls}>`
                + `<td><strong>${modelDisplayName(name)}</strong>${isMinStd ? ' <span style="color:#4ade80;font-size:0.75em">● std min</span>' : ''}</td>`
                + `<td style="font-family: 'JetBrains Mono', monospace; font-size: 0.85em">${hpStr}</td>`
                + `<td>${stdCell(baseMean, baseStd)}</td>`
                + `<td>${stdCell(tunedMean, tunedStd)}</td>`
                + `<td><span class="${cls}">${sign}${deltaCv.toFixed(2)}pt</span></td>`
                + `<td>${t.elapsed_seconds.toFixed(1)}s</td>`
                + '</tr>';
        }
        html += '</tbody></table>';
        hpEl.innerHTML = html;
    }

    const verdictEl = document.getElementById('temporal-tuning-verdict');
    if (verdictEl && data.verdict) {
        verdictEl.textContent = data.verdict;
    }

    // Spans dynamiques
    const setText = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };
    const fmtSignedPt = (v) => (v > 0 ? '+' : '') + v.toFixed(2) + 'pt';
    const fmtSigned = (v) => (v > 0 ? '+' : '') + v.toFixed(4);

    setText('dyn-tuning-best-model', modelDisplayName(diff.best_model_tuned));
    setText('dyn-tuning-mape-tuned', diff.best_mape_tuned.toFixed(2));
    setText('dyn-tuning-mape-baseline', diff.best_mape_baseline.toFixed(2));
    setText('dyn-tuning-mape-delta', fmtSignedPt(diff.best_mape_delta));
    setText('dyn-tuning-n-trials', data.n_trials);

    const cb = diff.per_model.catboost;
    setText('dyn-tuning-cb-mape-delta', fmtSignedPt(cb.mape_eur_delta));
    setText('dyn-tuning-cb-r2eur-delta', fmtSigned(cb.r2_eur_delta));
    setText('dyn-tuning-cb-r2lr-delta', fmtSigned(cb.r2_logratio_delta));

    const xgb = diff.per_model.xgboost;
    setText('dyn-tuning-xgb-mape-delta', fmtSignedPt(xgb.mape_eur_delta));
    setText('dyn-tuning-xgb-r2eur-delta', fmtSigned(xgb.r2_eur_delta));
    setText('dyn-tuning-xgb-r2lr-delta', fmtSigned(xgb.r2_logratio_delta));

    // Décision textuelle basée sur la STABILITÉ (CV MAPE std + WF MAPE std).
    // Critère : parmi les modèles tunés, on retient celui qui a le plus petit
    // écart-type — un modèle stable généralise mieux en production qu'un
    // modèle marginalement plus précis mais volatile entre folds.
    const tuningHp = data.tuning || {};
    const stabilityPool = ['catboost', 'xgboost']
        .filter(n => tuningHp[n] && tuningHp[n].cv_mape_std_tuned != null)
        .map(n => ({
            name: n,
            cvStd: tuningHp[n].cv_mape_std_tuned,
            wfStd: (diff.per_model[n] && diff.per_model[n].mape_eur_std_tuned) || null,
            mapeTuned: diff.per_model[n].mape_eur_tuned,
        }));

    let stabilityWinner = null;
    if (stabilityPool.length >= 2) {
        stabilityPool.sort((a, b) => a.cvStd - b.cvStd);
        stabilityWinner = stabilityPool[0];
    }

    let decision;
    if (stabilityWinner) {
        const other = stabilityPool[1];
        const cvGap = (other.cvStd - stabilityWinner.cvStd).toFixed(2);
        const mapeGap = (other.mapeTuned - stabilityWinner.mapeTuned).toFixed(2);
        decision =
            `Critère retenu = stabilité intra-CV (std du MAPE sur TSSplit(3)). `
            + `${modelDisplayName(stabilityWinner.name)} tuné présente `
            + `std=${stabilityWinner.cvStd.toFixed(2)}pt vs `
            + `${modelDisplayName(other.name)} std=${other.cvStd.toFixed(2)}pt `
            + `(écart +${cvGap}pt). `
            + `Même si l'écart de MAPE mean est faible (${mapeGap}pt), `
            + `on retient ${modelDisplayName(stabilityWinner.name)} — un modèle stable `
            + `se rejoue mieux en production qu'un modèle volatile entre plis temporels.`;
    } else {
        const d = diff.best_mape_delta;
        if (d < -0.25) {
            decision = `Gain significatif : on retient les HP tunés (${fmtSignedPt(d)} sur MAPE mean).`;
        } else if (d > 0.25) {
            decision = `Tuning dégrade la performance (${fmtSignedPt(d)}) — on garde les HP baseline.`;
        } else {
            decision = `Gain non significatif (|Δ| < 0.25pt). HP baseline conservés.`;
        }
    }
    setText('dyn-tuning-decision', decision);

    // Écrit le nom du modèle retenu (stabilité) pour le reste du dashboard.
    if (stabilityWinner) {
        const chosenName = modelDisplayName(stabilityWinner.name);
        setText('dyn-chosen-model', chosenName);
        setText('dyn-chosen-model-2', chosenName);
        setText('dyn-chosen-model-3', chosenName);
        setText('dyn-chosen-model-4', chosenName);
        setText('dyn-chosen-model-hero', chosenName);
        setText('dyn-chosen-model-as24', chosenName);
        setText('dyn-chosen-model-serp', chosenName);
        setText('dyn-chosen-model-cvstd', stabilityWinner.cvStd.toFixed(2));
        setText('dyn-chosen-model-wfstd',
            stabilityWinner.wfStd != null ? stabilityWinner.wfStd.toFixed(2) : '--');
        setText('dyn-chosen-model-mape', stabilityWinner.mapeTuned.toFixed(2));
        window.__chosenModel = stabilityWinner.name;

        // ---- Override stat cards + "Lecture finale" avec le modèle retenu ----
        // Les stat cards (MAPE/R² post-COVID) et les spans dyn-postcovid-*
        // doivent refléter le MODÈLE RETENU sur critère de stabilité,
        // avec les métriques CANONIQUES walk-forward post-COVID (3 folds × 6 mois)
        // exposées par meta.walk_forward (fallback single-split du
        // tuning_vs_baseline si indisponible).
        const wf = (DATA.meta && DATA.meta.walk_forward) || null;
        const chosenPer = diff.per_model[stabilityWinner.name] || {};
        const chosenMape = wf && wf.mape_eur_mean != null
            ? wf.mape_eur_mean
            : chosenPer.mape_eur_tuned;
        const chosenR2 = wf && wf.r2_eur_mean != null
            ? wf.r2_eur_mean
            : chosenPer.r2_eur_tuned;
        const chosenR2Lr = wf && wf.r2_logratio_mean != null
            ? wf.r2_logratio_mean
            : chosenPer.r2_logratio_tuned;
        if (chosenMape != null) {
            setText('temporal-catboost-mape', chosenMape.toFixed(2) + '%');
            setText('dyn-postcovid-mape', chosenMape.toFixed(2));
            setText('dyn-postcovid-mape-2', chosenMape.toFixed(2));
        }
        if (chosenR2 != null) {
            setText('temporal-catboost-r2', (chosenR2 * 100).toFixed(2) + '%');
            setText('dyn-postcovid-r2', chosenR2.toFixed(3));
        }
        if (chosenR2Lr != null) {
            setText('dyn-postcovid-r2-lr', chosenR2Lr.toFixed(3));
        }

        // Facteur de recalibration B2B → C2C (issu de l'étude AS24 post-COVID)
        const as24pc = (DATA.charts && DATA.charts.as24_postcovid) || {};
        const recalFactor = as24pc.recalibration && as24pc.recalibration.factor;
        if (recalFactor != null) {
            setText('temporal-recal-factor', '×' + Number(recalFactor).toFixed(3));
        }

        // Replicate train/test range spans in section 7.3 + 7.4.
        const cmeta2 = (DATA.meta && DATA.meta.comparison_meta) || {};
        const nTrain2 = cmeta2.n_train != null ? cmeta2.n_train.toLocaleString('fr-FR') : '--';
        const nTest2 = cmeta2.n_test != null ? cmeta2.n_test.toLocaleString('fr-FR') : '--';
        const trainRange2 = cmeta2.train_range
            ? `${cmeta2.train_range[0]} → ${cmeta2.train_range[1]}` : '--';
        const testRange2 = cmeta2.test_range
            ? `${cmeta2.test_range[0]} → ${cmeta2.test_range[1]}` : '--';
        setText('dyn-train-range-2', trainRange2);
        setText('dyn-n-train-2', nTrain2);
        setText('dyn-test-range-2', testRange2);
        setText('dyn-n-test-2', nTest2);
        setText('dyn-test-range-3', testRange2);
        setText('dyn-test-range-4', testRange2);
    }
}


// ============================================================
// STRESS TEST — RANDOM SPLIT 49/51 POST-COVID
// Rompt l'ordre temporel pour vérifier que le ranking des modèles
// tient sans l'avantage structurel du train récent.
// ============================================================
function renderStressTest4951(data) {
    const host = document.getElementById('chart-stress-4951');
    if (!data || !data.metrics_eur || !host) {
        const section = document.getElementById('stress');
        if (section && !data) section.style.display = 'none';
        return;
    }

    setText('dyn-stress-n-train', (data.n_train || 0).toLocaleString('fr-FR'));
    setText('dyn-stress-n-test', (data.n_test || 0).toLocaleString('fr-FR'));
    const tr = data.train_range || [];
    const te = data.test_range || [];
    setText('dyn-stress-train-range', tr.length === 2 ? `${tr[0]} → ${tr[1]}` : '--');
    setText('dyn-stress-test-range', te.length === 2 ? `${te[0]} → ${te[1]}` : '--');
    setText('dyn-stress-cutoff', data.cutoff_date || '--');

    // Ref MAPE from canonical split (used to frame the comparison)
    try {
        const canon = (DATA.charts.model_comparison || {});
        const mapes = canon.mape || [];
        const refMape = Math.min(...mapes.filter(v => v != null && v > 0));
        if (isFinite(refMape)) setText('dyn-stress-ref-mape', refMape.toFixed(2));
    } catch (_) { /* noop */ }

    const lrIndex = {};
    (data.metrics_logratio || []).forEach(r => {
        lrIndex[r.model] = r["MAPE (log_ratio) %"];
    });

    const metrics = data.metrics_eur.slice().sort((a, b) => a["MAPE (%)"] - b["MAPE (%)"]);
    const labels = metrics.map(r => modelDisplayName(r.model));
    const colors = metrics.map(r => MODEL_COLOR_MAP[r.model] || '#64748b');
    const mapes = metrics.map(r => Number(r["MAPE (%)"]));
    // Accès défensif au R² : selon la sérialisation JSON, la clé peut être
    // "R²" (U+00B2) ou "R\u00b2". On essaie les deux puis on scanne les clés
    // restantes si rien n'est trouvé — évite que la map ne contienne des
    // undefined qui feraient sauter des barres du chart.
    const pickR2 = (r) => {
        if (r["R²"] != null) return Number(r["R²"]);
        for (const k of Object.keys(r)) {
            if (k.startsWith('R') && k.length <= 3) {
                const v = r[k];
                if (v != null && !isNaN(Number(v))) return Number(v);
            }
        }
        return null;
    };
    const r2s = metrics.map(pickR2);
    if (r2s.some(v => v == null)) {
        console.warn('[stress-4951] R² manquant pour certaines lignes', metrics, r2s);
    }

    const labelFont = { size: 11, family: 'JetBrains Mono' };

    host.style.height = 'auto';
    host.innerHTML = '<div style="display:flex;flex-direction:column;gap:1.5rem">'
        + '<div id="chart-stress-4951-mape" style="height:340px;width:100%"></div>'
        + '<div id="chart-stress-4951-r2" style="height:340px;width:100%"></div></div>';

    const mapeTrace = {
        x: labels, y: mapes, type: 'bar',
        marker: { color: colors, opacity: 0.9, line: { width: 0 } },
        text: mapes.map(v => v.toFixed(2) + '%'), textposition: 'outside',
        textfont: { ...labelFont, color: '#f1f5f9' },
        cliponaxis: false,
        hovertemplate: '%{x}: MAPE %{y:.2f}%<extra></extra>',
    };
    Plotly.newPlot('chart-stress-4951-mape', [mapeTrace], mergeLayout({
        title: { text: 'MAPE (%) — random 49/51', font: { size: 13, color: '#cbd5e1' } },
        yaxis: { title: 'MAPE (%)', rangemode: 'tozero' },
        xaxis: { tickangle: -25, automargin: true, tickfont: { size: 11, color: '#cbd5e1' } },
        margin: { t: 40, b: 90, l: 55, r: 20 },
        showlegend: false,
    }), PLOTLY_CONFIG);

    const r2Trace = {
        x: labels.slice(), y: r2s.slice(), type: 'bar',
        marker: { color: colors.slice(), opacity: 0.9, line: { width: 0 } },
        text: r2s.map(v => (v == null ? '' : v.toFixed(3))), textposition: 'outside',
        textfont: { ...labelFont, color: '#f1f5f9' },
        cliponaxis: false,
        hovertemplate: '%{x}: R² %{y:.4f}<extra></extra>',
    };
    Plotly.newPlot('chart-stress-4951-r2', [r2Trace], mergeLayout({
        title: { text: 'R² — random 49/51', font: { size: 13, color: '#cbd5e1' } },
        yaxis: { title: 'R²', rangemode: 'tozero' },
        xaxis: { tickangle: -25, automargin: true, tickfont: { size: 11, color: '#cbd5e1' } },
        margin: { t: 40, b: 90, l: 55, r: 20 },
        showlegend: false,
    }), PLOTLY_CONFIG);

    const tbody = document.querySelector('#stress-4951-table tbody');
    if (tbody) {
        tbody.innerHTML = '';
        metrics.forEach(r => {
            const tr = document.createElement('tr');
            if (r.model === 'naive') tr.style.opacity = '0.7';
            const mapeLr = lrIndex[r.model];
            tr.innerHTML = [
                `<td>${modelDisplayName(r.model)}</td>`,
                `<td>${fmt(r["MAE (€)"])} €</td>`,
                `<td>${r["MAPE (%)"].toFixed(2)}%</td>`,
                `<td>${r["R²"].toFixed(4)}</td>`,
                `<td>${mapeLr != null ? mapeLr.toFixed(2) + '%' : '—'}</td>`,
            ].join('');
            tbody.appendChild(tr);
        });
    }

    const verdict = document.getElementById('stress-verdict');
    if (verdict) {
        const nonNaive = metrics.filter(r => r.model !== 'naive');
        const winner = nonNaive[0];
        const canonBest = (DATA.meta || {}).best_model;
        const sameWinner = winner && canonBest && winner.model === canonBest;
        const tunedList = (data.tuned_models || []).map(modelDisplayName).join(' + ') || '—';
        const naiveRow = metrics.find(r => r.model === 'naive');
        verdict.textContent =
            `Gagnant du stress temporel 49/51: ${winner ? modelDisplayName(winner.model) : '—'} `
            + `(MAPE ${winner ? winner["MAPE (%)"].toFixed(2) : '--'}%). `
            + (sameWinner
                ? `Même modèle que le split canonique (${modelDisplayName(canonBest)}) — ranking stable quand on compresse le train à 49% et étend le test à 51% sur un horizon chronologique plus long.`
                : `Ranking différent du split canonique (${canonBest ? modelDisplayName(canonBest) : '—'}) — à interpréter : dépendance au volume de train récent.`)
            + ` HP tunés utilisés : ${tunedList}. `
            + (naiveRow ? `Baseline naïf MAPE ${naiveRow["MAPE (%)"].toFixed(2)}%.` : '');
    }
}


function renderAs24PostCovid(data) {
    const card = document.getElementById('temporal-as24pc-card');
    if (!data || !data.as24_validation || !data.as24_validation.results || !card) {
        if (card) card.style.display = 'none';
        return;
    }
    const rows = data.as24_validation.results;
    const recal = data.recalibration || {};
    const cmp = data.baseline_full_comparison || {};
    const sanity = data.sanity_test_metrics || {};

    const subtitle = document.getElementById('temporal-as24pc-subtitle');
    if (subtitle) {
        subtitle.textContent =
            `train post-COVID=${data.n_train.toLocaleString()} (>= ${data.post_covid_cutoff}) · `
            + `portfolio=${data.n_portfolio.toLocaleString()} véh · `
            + `sanity test MAPE=${sanity.mape_eur}% R²=${sanity.r2_eur} R²_lr=${sanity.r2_logratio} · `
            + `facteur B2B→C2C x${recal.factor} (baseline x${cmp.baseline_recal_factor})`;
    }

    const labels = rows.map(r => `${r.brand} ${r.model}`);
    const deltaPc = rows.map(r => r.delta_pct_postcovid);
    const deltaBs = rows.map(r => (r.delta_pct_baseline_temporal != null ? r.delta_pct_baseline_temporal : r.delta_pct_baseline_random));
    const colorPc = deltaPc.map(d => (Math.abs(d) < 10 ? '#4BC4BD' : (Math.abs(d) < 20 ? '#f59e0b' : '#ef4444')));

    const traceBaseline = {
        x: labels, y: deltaBs,
        name: 'Baseline full train (temporel)',
        type: 'bar',
        marker: { color: 'rgba(148,163,184,0.55)' },
        text: deltaBs.map(d => d.toFixed(1) + '%'),
        textposition: 'outside',
        hovertemplate: 'Baseline: %{y:.2f}%<extra></extra>',
    };
    const tracePostCovid = {
        x: labels, y: deltaPc,
        name: 'Post-COVID',
        type: 'bar',
        marker: { color: colorPc },
        text: deltaPc.map(d => d.toFixed(1) + '%'),
        textposition: 'outside',
        hovertemplate: 'Post-COVID: %{y:.2f}%<extra></extra>',
    };
    const zeroLine = {
        x: labels, y: labels.map(() => 0),
        name: 'AS24 (0%)',
        type: 'scatter',
        mode: 'lines',
        line: { color: 'rgba(100,116,139,0.6)', dash: 'dash', width: 1 },
        hoverinfo: 'skip',
        showlegend: false,
    };
    const layout = mergeLayout({
        yaxis: {
            title: 'Écart médian vs AS24 (%)',
            zeroline: true,
            zerolinecolor: 'rgba(100,116,139,0.5)',
        },
        xaxis: { tickangle: -25 },
        barmode: 'group',
        legend: { orientation: 'h', y: -0.32, x: 0.5, xanchor: 'center', yanchor: 'top' },
        margin: { t: 30, b: 110, r: 30 },
    });
    Plotly.newPlot('chart-temporal-as24pc', [traceBaseline, tracePostCovid, zeroLine], layout, PLOTLY_CONFIG);

    const verdict = document.getElementById('temporal-as24pc-verdict');
    if (verdict) {
        const pcMed = data.as24_validation.median_delta_pct_postcovid;
        const bsMed = data.as24_validation.median_delta_pct_baseline_temporal != null
            ? data.as24_validation.median_delta_pct_baseline_temporal
            : data.as24_validation.median_delta_pct_baseline_random;
        const diff = (Math.abs(pcMed) - Math.abs(bsMed)).toFixed(2);
        const dirWord = Math.abs(pcMed) < Math.abs(bsMed) ? 'réduit' : 'augmente';
        verdict.textContent =
            `${data.verdict} Median |delta| ${dirWord} de ${Math.abs(bsMed).toFixed(2)}% à ${Math.abs(pcMed).toFixed(2)}% (Δ ${diff}pt). `
            + `Facteur de recalibration x${recal.factor} ramène la médiane du biais à ${recal.median_delta_pct_recalibrated}% (max |delta| = ${recal.max_abs_delta_pct_recalibrated}%).`;
    }
}

// ============================================================
// RISK PORTFOLIO DASHBOARD
// ============================================================
function renderRiskPortfolio(data) {
    if (!data) return;

    // Chiffres clés
    const expo = document.getElementById('risk-total-exposure');
    if (expo) expo.textContent = (data.total_exposure / 1e6).toFixed(2) + ' M€';
    const expoApprox = document.getElementById('dyn-exposure-approx');
    if (expoApprox && data.total_exposure != null) {
        expoApprox.textContent = Math.round(data.total_exposure / 1e6).toString();
    }
    const avg = document.getElementById('risk-avg-decote');
    if (avg) avg.textContent = data.avg_decote + '%';
    const cat = document.getElementById('risk-catalogue');
    if (cat) cat.textContent = (data.total_catalogue / 1e6).toFixed(2) + ' M€';

    // Scénarios de stress - bar chart
    if (data.scenarios && document.getElementById('chart-risk-scenarios')) {
        const sc = data.scenarios;
        const colors = ['#4BC4BD', '#3b82f6', '#f59e0b', '#ef4444'];
        const trace = {
            x: sc.map(s => s.name),
            y: sc.map(s => s.total_vr / 1e6),
            type: 'bar',
            marker: { color: colors.slice(0, sc.length), opacity: 0.85 },
            text: sc.map(s => (s.total_vr / 1e6).toFixed(2) + ' M€'),
            textposition: 'outside',
            cliponaxis: false,
            textfont: { color: '#f1f5f9', size: 12, family: 'JetBrains Mono' },
            hovertemplate: '%{x}<br>VR totale : %{y:.2f} M€<br>Perte : %{customdata:.2f} M€<extra></extra>',
            customdata: sc.map(s => s.loss / 1e6),
        };
        const maxVr = Math.max(...sc.map(s => s.total_vr / 1e6));
        const layout = mergeLayout({
            yaxis: { title: 'VR totale (M€)', range: [0, maxVr * 1.18] },
            margin: { t: 50 },
        });
        Plotly.newPlot('chart-risk-scenarios', [trace], layout, PLOTLY_CONFIG);

        // Table des scénarios
        const tbl = document.getElementById('risk-scenarios-table');
        if (tbl) {
            let html = '<table><thead><tr><th>Scénario</th><th>Choc</th><th>VR totale</th><th>Perte</th><th>Perte %</th></tr></thead><tbody>';
            for (const s of sc) {
                const cls = s.shock_pct === 0 ? ' class="best-row"' : '';
                html += `<tr${cls}>`;
                html += `<td><strong>${s.name}</strong></td>`;
                html += `<td>${s.shock_pct > 0 ? '+' : ''}${s.shock_pct}%</td>`;
                html += `<td>${(s.total_vr / 1e6).toFixed(2)} M€</td>`;
                html += `<td style="color:${s.loss > 0 ? '#ef4444' : 'var(--teal)'};font-weight:600">${s.loss > 0 ? '-' : ''}${(Math.abs(s.loss) / 1e6).toFixed(2)} M€</td>`;
                html += `<td style="color:${s.loss_pct > 0 ? '#ef4444' : 'var(--teal)'};font-weight:600">${s.loss_pct > 0 ? '-' : ''}${Math.abs(s.loss_pct)}%</td>`;
                html += '</tr>';
            }
            html += '</tbody></table>';
            tbl.innerHTML = html;
        }
    }

    // Distribution du risque - donut (cliquable pour filtrer la table à droite)
    const LEVEL_META = {
        low:    { label: 'Faible (≤ 50%)',    color: '#4BC4BD' },
        medium: { label: 'Modéré (50-60%)',   color: '#f59e0b' },
        high:   { label: 'Élevé (> 60%)',     color: '#ef4444' },
    };

    const renderRiskTable = (level /* null | 'low' | 'medium' | 'high' */) => {
        const topEl = document.getElementById('risk-top-table');
        const titleEl = document.getElementById('risk-top-title');
        const chipsEl = document.getElementById('risk-filter-chips');
        if (!topEl) return;

        let rows, title;
        if (level && data.all_risk) {
            rows = data.all_risk.filter(v => v.risk_level === level);
            const m = LEVEL_META[level];
            title = `Véhicules — risque ${m.label.toLowerCase()} (${rows.length})`;
        } else {
            rows = data.top_risk || [];
            title = `Top 10 véhicules à risque`;
        }

        if (titleEl) titleEl.textContent = title;

        if (chipsEl) {
            const chip = (key, text) => {
                const active = (key === level) || (!level && key === 'top');
                const color = key === 'top' ? '#94a3b8' : LEVEL_META[key].color;
                return `<button type="button" data-level="${key}" class="risk-chip${active ? ' active' : ''}" style="--chip-color:${color}">${text}</button>`;
            };
            chipsEl.innerHTML = [
                chip('top', 'Top 10 décote'),
                chip('low', 'Faible'),
                chip('medium', 'Modéré'),
                chip('high', 'Élevé'),
            ].join('');
            chipsEl.querySelectorAll('button').forEach(b => {
                b.onclick = () => {
                    const k = b.getAttribute('data-level');
                    renderRiskTable(k === 'top' ? null : k);
                };
            });
        }

        let html = '<table><thead><tr><th>Marque</th><th>Modèle</th><th>Carburant</th><th>Âge</th><th>V₀ (€)</th><th>VR (€)</th><th>Décote</th></tr></thead><tbody>';
        for (const v of rows) {
            const decoteColor = v.decote_pct > 60 ? '#ef4444' : v.decote_pct > 50 ? '#f59e0b' : '#4BC4BD';
            html += '<tr>';
            html += `<td><strong>${v.brand}</strong></td>`;
            html += `<td>${v.model || '—'}</td>`;
            html += `<td style="font-size:0.75rem">${v.fuel_type}</td>`;
            html += `<td>${v.age_years != null ? v.age_years + ' ans' : '—'}</td>`;
            html += `<td>${fmt(v.prix_catalogue)}</td>`;
            html += `<td>${fmt(v.prediction)}</td>`;
            html += `<td style="color:${decoteColor};font-weight:700">${v.decote_pct}%</td>`;
            html += '</tr>';
        }
        html += '</tbody></table>';
        topEl.innerHTML = html;
    };

    if (document.getElementById('chart-risk-distribution')) {
        const trace = {
            labels: [LEVEL_META.low.label, LEVEL_META.medium.label, LEVEL_META.high.label],
            values: [data.n_low_risk, data.n_medium_risk, data.n_high_risk],
            customdata: ['low', 'medium', 'high'],
            type: 'pie',
            hole: 0.55,
            textinfo: 'label+percent',
            textposition: 'inside',
            marker: { colors: [LEVEL_META.low.color, LEVEL_META.medium.color, LEVEL_META.high.color] },
            hovertemplate: '%{label}<br>%{value} véhicules (%{percent})<br><i>cliquer pour filtrer</i><extra></extra>',
        };
        const layout = mergeLayout({
            margin: { t: 20, b: 20, l: 20, r: 20 },
            showlegend: false,
            annotations: [{
                text: `<b>${data.n_vehicles}</b><br><span style="font-size:0.75rem;color:#94a3b8">véhicules</span>`,
                x: 0.5, y: 0.5, showarrow: false,
                font: { size: 20, color: '#f1f5f9', family: 'JetBrains Mono' },
            }],
        });
        Plotly.newPlot('chart-risk-distribution', [trace], layout, PLOTLY_CONFIG).then(gd => {
            gd.on('plotly_click', (ev) => {
                if (!ev || !ev.points || !ev.points.length) return;
                const pt = ev.points[0];
                const levels = ['low', 'medium', 'high'];
                let lvl = levels[pt.pointNumber];
                if (!lvl) {
                    let cd = pt.customdata;
                    if (Array.isArray(cd)) cd = cd[0];
                    lvl = cd;
                }
                if (lvl) renderRiskTable(lvl);
            });
        });
    }

    renderRiskTable(null);

    // Exposition par marque
    if (data.brand_risk && document.getElementById('chart-risk-brand')) {
        const colorMap = { RENAULT: '#3b82f6', DACIA: '#4BC4BD', NISSAN: '#f59e0b' };
        const trace = {
            x: data.brand_risk.map(b => b.brand),
            y: data.brand_risk.map(b => b.total_vr / 1e6),
            type: 'bar',
            marker: { color: data.brand_risk.map(b => colorMap[b.brand] || '#3b82f6'), opacity: 0.85 },
            text: data.brand_risk.map(b => (b.total_vr / 1e6).toFixed(2) + ' M€'),
            textposition: 'outside',
            cliponaxis: false,
            textfont: { color: '#f1f5f9', size: 12 },
            hovertemplate: '%{x}<br>Exposition : %{y:.2f} M€<br>n = %{customdata}<extra></extra>',
            customdata: data.brand_risk.map(b => b.n),
        };
        const maxBrand = Math.max(...data.brand_risk.map(b => b.total_vr / 1e6));
        const layout = mergeLayout({
            yaxis: { title: 'Exposition VR (M€)', range: [0, maxBrand * 1.18] },
            margin: { t: 50 },
        });
        Plotly.newPlot('chart-risk-brand', [trace], layout, PLOTLY_CONFIG);
    }

    // Décote par carburant
    if (data.fuel_risk && document.getElementById('chart-risk-fuel')) {
        const trace = {
            x: data.fuel_risk.map(f => f.fuel_type),
            y: data.fuel_risk.map(f => f.median_decote),
            type: 'bar',
            marker: {
                color: data.fuel_risk.map(f => f.median_decote > 60 ? '#ef4444' : f.median_decote > 50 ? '#f59e0b' : '#4BC4BD'),
                opacity: 0.85,
            },
            text: data.fuel_risk.map(f => f.median_decote + '%'),
            textposition: 'outside',
            cliponaxis: false,
            textfont: { color: '#f1f5f9', size: 12 },
            hovertemplate: '%{x}<br>Décote médiane : %{y}%<br>n = %{customdata}<extra></extra>',
            customdata: data.fuel_risk.map(f => f.n),
        };
        const maxFuel = Math.max(...data.fuel_risk.map(f => f.median_decote));
        const layout = mergeLayout({
            xaxis: { tickangle: -30 },
            yaxis: { title: 'Décote médiane (%)', range: [0, maxFuel * 1.18] },
            margin: { t: 50, b: 80 },
        });
        Plotly.newPlot('chart-risk-fuel', [trace], layout, PLOTLY_CONFIG);
    }
}

// ============================================================
// LEGACY Brent functions (kept for compatibility, unused)
// ============================================================
function renderBrentTimeSeries(id, data, field, ylabel, color) {
    if (!data || !document.getElementById(id)) return;
    const trace = {
        x: data.dates,
        y: data[field],
        type: 'scatter',
        mode: 'lines',
        line: { color: color, width: 2 },
        fill: 'tozeroy',
        fillcolor: color + '20',
        hovertemplate: '%{x}<br>' + ylabel + ': %{y:.2f}<extra></extra>',
    };
    const layout = mergeLayout({
        yaxis: { title: ylabel },
        margin: { t: 10, b: 40, l: 60, r: 10 },
    });
    if (field === 'yoy') {
        layout.shapes = [{ type: 'line', y0: 0, y1: 0, x0: data.dates[0], x1: data.dates[data.dates.length-1],
                          line: { color: 'rgba(148,163,184,0.4)', width: 1, dash: 'dash' } }];
    }
    Plotly.newPlot(id, [trace], layout, PLOTLY_CONFIG);
}

function fillBrentCorrelation(results) {
    if (!results || !results.correlation) return;
    const c = results.correlation;
    document.getElementById('brent-corr-pearson').textContent = c.pearson;
    document.getElementById('brent-corr-spearman').textContent = c.spearman;
    document.getElementById('brent-best-lag').textContent = c.best_lag_months + ' mois';
    document.getElementById('brent-best-corr').textContent = c.best_lag_corr.toFixed(3);
    document.getElementById('brent-n-obs').textContent = c.n_obs;
}

function renderBrentComparison(id, results) {
    if (!results || !results.used_market_impact || !document.getElementById(id)) return;
    const rows = results.used_market_impact;
    const models = rows.map(r => modelDisplayName(r.model));

    const el = document.getElementById(id);
    el.innerHTML = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;height:100%">'
        + '<div id="' + id + '-mape"></div>'
        + '<div id="' + id + '-mae"></div>'
        + '<div id="' + id + '-r2"></div></div>';

    const commonLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 12 },
        xaxis: { gridcolor: 'rgba(148,163,184,0.08)' },
        yaxis: { gridcolor: 'rgba(148,163,184,0.08)' },
        margin: { t: 35, b: 30, l: 50, r: 10 },
        legend: { orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' },
        barmode: 'group',
    };

    // MAPE
    Plotly.newPlot(id + '-mape', [
        { x: models, y: rows.map(r => r.baseline_mape), name: 'Baseline', type: 'bar', marker: { color: '#94a3b8' } },
        { x: models, y: rows.map(r => r.with_brent_mape), name: '+ Brent', type: 'bar', marker: { color: '#3b82f6' } },
    ], { ...commonLayout, title: { text: 'MAPE (%)', font: { size: 13, color: '#f1f5f9' } } }, PLOTLY_CONFIG);

    // MAE
    Plotly.newPlot(id + '-mae', [
        { x: models, y: rows.map(r => r.baseline_mae), name: 'Baseline', type: 'bar', marker: { color: '#94a3b8' } },
        { x: models, y: rows.map(r => r.with_brent_mae), name: '+ Brent', type: 'bar', marker: { color: '#3b82f6' } },
    ], { ...commonLayout, title: { text: 'MAE (€)', font: { size: 13, color: '#f1f5f9' } } }, PLOTLY_CONFIG);

    // R²
    Plotly.newPlot(id + '-r2', [
        { x: models, y: rows.map(r => r.baseline_r2), name: 'Baseline', type: 'bar', marker: { color: '#94a3b8' } },
        { x: models, y: rows.map(r => r.with_brent_r2), name: '+ Brent', type: 'bar', marker: { color: '#3b82f6' } },
    ], { ...commonLayout, title: { text: 'R²', font: { size: 13, color: '#f1f5f9' } }, yaxis: { ...commonLayout.yaxis, range: [0.8, 1] } }, PLOTLY_CONFIG);
}

function renderBrentMetricsTable(results) {
    const el = document.getElementById('brent-metrics-table');
    if (!el || !results || !results.used_market_impact) return;
    const rows = results.used_market_impact;
    let html = '<table><thead><tr><th>Modèle</th><th>MAPE baseline</th><th>MAPE + Brent</th><th>Δ MAPE</th><th>MAE baseline</th><th>MAE + Brent</th><th>Δ MAE</th></tr></thead><tbody>';
    for (const r of rows) {
        const improve = r.delta_mape_pp < 0;
        const deltaCls = improve ? 'style="color:var(--teal);font-weight:700"' : 'style="color:var(--text-secondary)"';
        html += `<tr>`;
        html += `<td><strong>${modelDisplayName(r.model)}</strong></td>`;
        html += `<td>${r.baseline_mape}%</td>`;
        html += `<td>${r.with_brent_mape}%</td>`;
        html += `<td ${deltaCls}>${r.delta_mape_pp > 0 ? '+' : ''}${r.delta_mape_pp} pp</td>`;
        html += `<td>${fmt(r.baseline_mae)} €</td>`;
        html += `<td>${fmt(r.with_brent_mae)} €</td>`;
        html += `<td ${deltaCls}>${r.delta_mae_eur > 0 ? '+' : ''}${r.delta_mae_eur} €</td>`;
        html += '</tr>';
    }
    if (results.portfolio_impact) {
        const p = results.portfolio_impact;
        html += `<tr style="border-top:2px solid var(--teal)"><td colspan="7" style="text-align:center;color:var(--teal);padding-top:1rem"><strong>Impact Portfolio :</strong> VR médiane ${fmt(p.baseline_median)} € → ${fmt(p.enriched_median)} € (${p.delta_median_pct > 0 ? '+' : ''}${p.delta_median_pct}%)</td></tr>`;
    }
    html += '</tbody></table>';
    el.innerHTML = html;
}

// ============================================================
// KBA TENSION — courbe de contexte marché VO cible
// ============================================================
function renderKbaTension(id, data, fallback) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.display = '';
    // Fallback : si les données KBA ne sont pas exposées par l'API (fichiers
    // absents du repo), on rend les HICP Allemagne qui capturent les mêmes
    // régimes de tension (COVID 2021-2022, choc énergie 2022-2023).
    if (!data || !data.dates) {
        const he = fallback && fallback.hicp_energy;
        const hh = fallback && fallback.hicp_headline;
        if (!he || !he.months) { el.style.display = 'none'; return; }
        const energyTrace = {
            x: he.months, y: he.yoy,
            type: 'scatter', mode: 'lines',
            name: 'HICP énergie YoY (%)',
            line: { color: '#f59e0b', width: 2.5 },
            fill: 'tozeroy', fillcolor: 'rgba(245,158,11,0.12)',
            hovertemplate: '%{x}<br>Énergie YoY : %{y:+.1f}%<extra></extra>',
        };
        const headlineTrace = hh && hh.months ? {
            x: hh.months, y: hh.yoy,
            type: 'scatter', mode: 'lines',
            name: 'HICP headline YoY (%)',
            line: { color: '#4BC4BD', width: 2 },
            hovertemplate: '%{x}<br>Headline YoY : %{y:+.1f}%<extra></extra>',
        } : null;
        const traces = headlineTrace ? [energyTrace, headlineTrace] : [energyTrace];
        const layout = mergeLayout({
            title: { text: 'Contexte macro du marché cible (HICP 2018-2025)', font: { size: 13, color: '#cbd5e1' }, x: 0, xanchor: 'left' },
            yaxis: { title: 'Variation annuelle (%)', gridcolor: 'rgba(148,163,184,0.08)', zeroline: true, zerolinecolor: 'rgba(148,163,184,0.25)' },
            margin: { t: 40, b: 40, l: 55, r: 20 },
            legend: { orientation: 'h', y: -0.22, x: 0.5, xanchor: 'center' },
            hovermode: 'x unified',
        });
        Plotly.newPlot(id, traces, layout, PLOTLY_CONFIG);
        return;
    }

    const neuTrace = {
        x: data.dates,
        y: data.neuzulassungen,
        type: 'scatter', mode: 'lines',
        name: 'Neuzulassungen (VN)',
        line: { color: '#94a3b8', width: 2 },
        yaxis: 'y',
        hovertemplate: '%{x}<br>VN : %{y:,.0f}<extra></extra>',
    };
    const besTrace = {
        x: data.dates,
        y: data.besitzumschreibungen,
        type: 'scatter', mode: 'lines',
        name: 'Besitzumschreibungen (VO)',
        line: { color: '#3b82f6', width: 2 },
        yaxis: 'y',
        hovertemplate: '%{x}<br>VO : %{y:,.0f}<extra></extra>',
    };
    const tensionTrace = {
        x: data.dates,
        y: data.tension_centered,
        type: 'scatter', mode: 'lines',
        name: 'Tension (VO/VN, log centré)',
        line: { color: '#4BC4BD', width: 3 },
        yaxis: 'y2',
        hovertemplate: '%{x}<br>Tension : %{y:+.3f}<extra></extra>',
        fill: 'tozeroy',
        fillcolor: 'rgba(75,196,189,0.12)',
    };

    const layout = mergeLayout({
        yaxis: {
            title: 'Volumes mensuels Pkw',
            side: 'left',
            gridcolor: 'rgba(148,163,184,0.08)',
        },
        yaxis2: {
            title: 'Tension (log)',
            overlaying: 'y', side: 'right',
            gridcolor: 'rgba(75,196,189,0.05)',
            zeroline: true,
            zerolinecolor: 'rgba(75,196,189,0.3)',
        },
        margin: { t: 20, b: 40, l: 65, r: 60 },
        legend: { orientation: 'h', y: -0.18, x: 0.5, xanchor: 'center' },
        hovermode: 'x unified',
    });
    Plotly.newPlot(id, [neuTrace, besTrace, tensionTrace], layout, PLOTLY_CONFIG);
}

function renderKbaRobustnessTable(results) {
    const el = document.getElementById('kba-robustness-table');
    if (!el || !results || !results.robustness_test) return;
    const r = results.robustness_test;
    const labels = {
        with_hicp__kba:      { name: 'AVEC HICP · tension KBA',     period: '2021-2025 (58m)' },
        with_hicp__internal: { name: 'AVEC HICP · tension interne', period: '2018-2025 (95m)' },
        no_hicp__kba:        { name: 'SANS HICP · tension KBA',     period: '2021-2025 (58m)' },
        no_hicp__internal:   { name: 'SANS HICP · tension interne', period: '2018-2025 (95m)' },
    };
    const stars = (p) => p < 0.001 ? '***' : p < 0.01 ? '**' : p < 0.05 ? '*' : 'ns';
    const colorize = (p) => p < 0.05 ? 'color:var(--teal);font-weight:700' : 'color:var(--text-secondary)';
    let html = '';
    for (const key of ['with_hicp__kba', 'with_hicp__internal', 'no_hicp__kba', 'no_hicp__internal']) {
        const row = r[key]; if (!row) continue;
        const sig = stars(row.p);
        html += `<tr>`;
        html += `<td><strong>${labels[key].name}</strong></td>`;
        html += `<td>${labels[key].period}</td>`;
        html += `<td>${row.beta >= 0 ? '+' : ''}${row.beta.toFixed(4)}</td>`;
        html += `<td>${row.se.toFixed(4)}</td>`;
        html += `<td>${row.t >= 0 ? '+' : ''}${row.t.toFixed(2)}</td>`;
        html += `<td>${row.p < 0.001 ? '< 0.001' : row.p.toFixed(3)}</td>`;
        html += `<td>${row.r2.toFixed(3)}</td>`;
        html += `<td style="${colorize(row.p)}">${sig}</td>`;
        html += `</tr>`;
    }
    el.innerHTML = html;
}

// ============================================================
// LANGUAGE SWITCHER (FR / EN)
// ============================================================
const TRANSLATIONS = {
    fr: {
        loading_title: 'Chargement du Dashboard',
        loading_sub: 'Préparation des visualisations...',
        nav_home: 'Accueil', nav_context: 'Contexte', nav_eda: 'Analyse Exploratoire',
        nav_hicp: 'Macro (HICP)', nav_features: 'Feature Engineering',
        nav_depreciation: 'Dépréciation',
        nav_results: 'Évaluation des Modèles', nav_portfolio: 'Portfolio', nav_simulator: 'Simulateur',
        hero_title: 'Prédiction de la<br><span class="gradient-text">Valeur Résiduelle</span><br>Automobile',
        hero_badge: "Challenge Nexialog — [Client]",
        hero_subtitle: "Modèle ML pour le leasing automobile.",
        exec_what_built: "Ce que l'on a construit",
        exec_what_built_text: 'Modèle ML de valeur résiduelle entraîné sur <strong id="exec-n-transactions">transactions</strong> du marché de l\'occasion.',
        exec_key_result: "Résultat principal",
        exec_key_result_text: 'Ensemble <strong id="exec-best-model">CatBoost + XGBoost</strong> &mdash; <span class="exec-accent" id="exec-mape-eur">6.08&nbsp;%</span> MAPE EUR (audit-proof : <span id="exec-mape-lr">9.2&nbsp;%</span> log_ratio) sur test hold-out 20&nbsp;%.',
        exec_why_matters: "Pourquoi ça compte",
        exec_why_matters_text: 'Exposition portefeuille <strong id="exec-exposition">~20&nbsp;M€</strong> pilotée avec stress tests BCE/EBA et validation externe AutoScout24.',
        hero_cta: 'Explorer les résultats',
        stat_transactions: 'Transactions analysées', stat_portfolio: 'Véhicules prédits',
        ctx_title: 'Contexte Métier',
        ctx_desc: "Comprendre l'enjeu stratégique de la valeur résiduelle dans le leasing automobile",
        ctx_vr_title: "Qu'est-ce que la Valeur Résiduelle ?",
        ctx_stakes_title: 'Enjeu pour [Client]',
        ctx_mission: 'Notre mission',
        ctx_methodo: 'Méthodologie',
        eda_title: 'Analyse Exploratoire',
        eda_desc: "Exploration des transactions du marché de l'occasion",
        eda_overview: "Vue d'ensemble des données",
        tab_numeric: 'Variables Numériques', tab_categorical: 'Variables Catégorielles', tab_correlations: 'Corrélations',
        hicp_title: 'Variables Macroéconomiques (HICP)',
        hicp_desc: "Intégration de l'inflation allemande pour séparer dépréciation réelle et effet nominal",
        feat_title: 'Feature Engineering',
        feat_desc: 'Construction des variables et clustering des modèles en familles de dépréciation',
        dep_title: 'Valeurs Résiduelles & Dépréciation',
        dep_desc: "Analyse de la dépréciation du prix dans le temps sur le marché de l'occasion",
        res_title: 'Évaluation des Modèles',
        res_desc: 'Comparaison des candidats, performance en production (split temporel post-COVID) et diagnostics résiduels',
        port_title: 'Prédictions sur le Portfolio',
        port_desc: 'Application du modèle retenu aux véhicules du portefeuille de leasing',
        sim_title: 'Simulateur de Valeur Résiduelle',
        sim_desc: "Estimez la valeur de revente d'un véhicule en fonction de ses caractéristiques",
        sim_form_title: 'Caractéristiques du véhicule',
        sim_brand: 'Marque', sim_fuel: 'Type de carburant', sim_range: 'Type de gamme',
        sim_price: 'Prix catalogue (€)', sim_prod: 'Année de production',
        sim_end: 'Année fin de contrat', sim_initkm: 'Km initiaux', sim_contractkm: 'Km contractuels',
        sim_btn: 'Estimer la valeur résiduelle',
        sim_placeholder: 'Remplissez le formulaire et cliquez sur « Estimer » pour obtenir une prédiction',
        sim_result_title: 'Estimation de la Valeur Résiduelle',
        sim_decote: 'Décote', sim_ratio: 'Ratio VR/Catalogue', sim_age: 'Âge à la revente', sim_km: 'Km total estimé',
        // --- v2 quick wins ---
        hero_tier_audit: 'Métriques audit-proof',
        trust_innovation: '<span class="hero-trust-chip__label">Innovation testée</span> <span class="hero-trust-chip__body">Brent crude &amp; tension marché KBA</span>',
        trust_validation: '<span class="hero-trust-chip__label">Validation externe</span> <span class="hero-trust-chip__body">AutoScout24, 10 modèles confrontés</span>',
        trust_risk: '<span class="hero-trust-chip__label">Risque calibré</span> <span class="hero-trust-chip__body">stress tests BCE/EBA + top 10 actionnable</span>',
        // --- Phase 2 : progressive disclosure ---
        intuition_label: 'Intuition',
        hicp_intuition: "L'inflation contamine la dépréciation nominale : un véhicule peut « valoir plus en euros » sans valoir plus en réalité. On isole donc l'effet inflation pour mesurer la vraie dépréciation économique.",
        feat_intuition: "Réduire 62 modèles à une dizaine de familles évite que les arbres ne mémorisent chaque modèle individuellement — ils apprennent les comportements de dépréciation, pas les noms commerciaux.",
        val_intuition: "Un bon score interne ne suffit pas — confronter le modèle à des annonces publiques scrapées en temps réel est la seule preuve qu'il généralise hors training et reste cohérent avec le marché actuel.",
        coll_hicp_limit: 'Voir le détail des limites macroéconomiques',
        coll_hicp_brent: "Voir l'expérimentation Brent (innovation)",
        coll_hicp_kba: 'Voir le test de robustesse KBA (innovation)',
        coll_val_method: 'Voir le protocole de scraping détaillé',
        coll_val_interp: "Voir l'interprétation B2B vs B2C",
        coll_risk_top10: 'Voir les 10 véhicules à risque',
        // --- Phase 4 : RAG drawer ---
        rag_trigger_label: 'Assistant méthodologique',
        rag_title: 'Assistant méthodologique',
        rag_subtitle: 'RAG sur données & code du projet',
        rag_welcome: "Posez une question sur les choix méthodologiques, les résultats, ou la robustesse du modèle. Les réponses s'appuient sur les données du dashboard, le code source du projet et la documentation académique.",
        rag_suggestions_label: 'Questions fréquentes',
        rag_input_placeholder: 'Posez votre question…',
        rag_footnote: 'Endpoint <code>POST /api/rag</code> — branchement à l\'API Claude (Anthropic) en cours d\'intégration.',
        skip_link: 'Aller au contenu',
        back_to_top: 'Retour en haut',
        exec_label: 'Résultat principal',
        exec_gain_label: 'gain MAPE vs baseline naïve k·V₀',
        exec_portfolio_label: 'véhicules prédits sur le portefeuille',
        exec_decote_label: 'décote moyenne portefeuille',
        ctx_insight: 'La VR pèse <strong>40&ndash;60&nbsp;%</strong> du prix catalogue : une erreur de quelques points de pourcentage se traduit en millions d\'euros à l\'échelle du portefeuille.',
        eda_insight: 'Distribution stable sur <strong>transactions</strong> filtrées 12&ndash;108&nbsp;mois ; <code>log_ratio</code> médian &asymp;&nbsp;-1 confirme une décote typique de&nbsp;~63&nbsp;%.',
        hicp_insight: 'Trois features HICP ramènent la MAPE de <strong>~13&nbsp;%</strong> à <strong>~7&nbsp;%</strong> ; Brent et tension marché KBA testés n\'apportent rien de plus — le signal macro est déjà capté.',
        feat_insight: '<strong>62 modèles</strong> réduits à une dizaine de familles homogènes de dépréciation via KMeans sur les vecteurs (&alpha;, k&#8321;, k&#8322;).',
        dep_insight: 'Courbe d\'âge <strong>décroissante convexe</strong> confirmée empiriquement : le modèle exponentiel de Nexialog est validé sur données réelles.',
        mod_insight: '<strong>CatBoost</strong> et XGBoost quasi ex-aequo, <strong>~10&nbsp;pp devant Ridge</strong> ; un ensemble pondéré XGB + CatBoost est retenu pour le livrable final.',
        res_insight: 'Résidus <strong>centrés sur 0</strong>, majorité dans &plusmn;&nbsp;1&nbsp;000&nbsp;€ ; aucun biais systématique sur les 20&nbsp;% jamais vus pendant l\'entraînement.',
        port_insight: '<strong>Véhicules</strong> prédits, cohérence économique vérifiée : dépréciation monotone par âge et hiérarchie par marque maintenue.',
        val_insight: 'Écart médian <strong>X&nbsp;%</strong> vs AutoScout24 : cohérent avec le différentiel B2B / B2C — signature d\'un modèle conservateur, <em>pas</em> d\'une erreur.',
        risk_insight: 'Exposition totale <strong>~20&nbsp;M€</strong>, stress tests BCE/EBA -5&nbsp;% à -15&nbsp;%, top&nbsp;10 des véhicules à risque <strong>directement actionnable</strong> pour revente anticipée.',
        sim_insight: 'Prédiction <strong>CatBoost temps réel</strong> sur un véhicule personnalisé — démonstration opérationnelle bout-en-bout du modèle.',
    },
    en: {
        loading_title: 'Loading Dashboard',
        loading_sub: 'Preparing visualizations...',
        nav_home: 'Home', nav_context: 'Context', nav_eda: 'Exploratory Analysis',
        nav_hicp: 'Macro (HICP)', nav_features: 'Feature Engineering',
        nav_depreciation: 'Depreciation',
        nav_results: 'Model Evaluation', nav_portfolio: 'Portfolio', nav_simulator: 'Simulator',
        hero_title: 'Automotive<br><span class="gradient-text">Residual Value</span><br>Prediction Model',
        hero_badge: "Nexialog Challenge — [Client]",
        hero_subtitle: 'ML model for automotive leasing.',
        exec_what_built: 'What we built',
        exec_what_built_text: 'ML residual value model trained on <strong id="exec-n-transactions">transactions</strong> from the used-car market.',
        exec_key_result: 'Key result',
        exec_key_result_text: 'Ensemble <strong id="exec-best-model">CatBoost + XGBoost</strong> &mdash; <span class="exec-accent" id="exec-mape-eur">6.08%</span> MAPE EUR (audit-proof: <span id="exec-mape-lr">9.2%</span> log_ratio) on 20% hold-out test.',
        exec_why_matters: 'Why it matters',
        exec_why_matters_text: 'Portfolio exposure <strong id="exec-exposition">~&euro;20M</strong> managed with ECB/EBA stress tests and external AutoScout24 validation.',
        hero_cta: 'Explore results',
        stat_transactions: 'Transactions analyzed', stat_portfolio: 'Vehicles predicted',
        ctx_title: 'Business Context',
        ctx_desc: 'Understanding the strategic importance of residual value in automotive leasing',
        ctx_vr_title: 'What is Residual Value?',
        ctx_stakes_title: 'Stakes for [Client]',
        ctx_mission: 'Our mission',
        ctx_methodo: 'Methodology',
        eda_title: 'Exploratory Analysis',
        eda_desc: 'Exploration of transactions from the used car market',
        eda_overview: 'Data overview',
        tab_numeric: 'Numeric Variables', tab_categorical: 'Categorical Variables', tab_correlations: 'Correlations',
        hicp_title: 'Macroeconomic Variables (HICP)',
        hicp_desc: 'Integrating German inflation to separate real depreciation from nominal effects',
        feat_title: 'Feature Engineering',
        feat_desc: 'Variable construction and model clustering into depreciation families',
        dep_title: 'Residual Values & Depreciation',
        dep_desc: 'Analysis of price depreciation over time in the used car market',
        res_title: 'Model Evaluation',
        res_desc: 'Candidate comparison, production performance (post-COVID temporal split) and residual diagnostics',
        port_title: 'Portfolio Predictions',
        port_desc: 'Applying the selected model to the vehicles in the leasing portfolio',
        sim_title: 'Residual Value Simulator',
        sim_desc: 'Estimate the resale value of a vehicle based on its characteristics',
        sim_form_title: 'Vehicle characteristics',
        sim_brand: 'Brand', sim_fuel: 'Fuel type', sim_range: 'Range type',
        sim_price: 'Catalogue price (€)', sim_prod: 'Production year',
        sim_end: 'Contract end year', sim_initkm: 'Initial km', sim_contractkm: 'Contract km',
        sim_btn: 'Estimate residual value',
        sim_placeholder: 'Fill in the form and click "Estimate" to get a prediction',
        sim_result_title: 'Residual Value Estimation',
        sim_decote: 'Depreciation', sim_ratio: 'RV/Catalogue ratio', sim_age: 'Age at resale', sim_km: 'Total km estimated',
        // --- v2 quick wins ---
        hero_tier_audit: 'Audit-proof metrics',
        trust_innovation: '<span class="hero-trust-chip__label">Innovation tested</span> <span class="hero-trust-chip__body">Brent crude &amp; KBA market tension</span>',
        trust_validation: '<span class="hero-trust-chip__label">External validation</span> <span class="hero-trust-chip__body">AutoScout24, 10 models confronted</span>',
        trust_risk: '<span class="hero-trust-chip__label">Calibrated risk</span> <span class="hero-trust-chip__body">ECB/EBA stress tests + actionable top 10</span>',
        // --- Phase 2 : progressive disclosure ---
        intuition_label: 'Intuition',
        hicp_intuition: "Inflation contaminates nominal depreciation: a vehicle can be «worth more in euros» without being worth more in reality. We isolate the inflation effect to measure true economic depreciation.",
        feat_intuition: "Reducing 62 models to about ten families prevents trees from memorising each model individually — they learn depreciation behaviours, not commercial names.",
        val_intuition: "A good internal score is not enough — confronting the model with publicly scraped real-time listings is the only proof it generalises outside training and remains consistent with today's market.",
        coll_hicp_limit: 'See macro-economic caveats in detail',
        coll_hicp_brent: 'See the Brent experiment (innovation)',
        coll_hicp_kba: 'See the KBA robustness test (innovation)',
        coll_val_method: 'See the detailed scraping protocol',
        coll_val_interp: 'See the B2B vs B2C interpretation',
        coll_risk_top10: 'See the 10 at-risk vehicles',
        // --- Phase 4 : RAG drawer ---
        rag_trigger_label: 'Methodological assistant',
        rag_title: 'Methodological assistant',
        rag_subtitle: 'RAG on project data & code',
        rag_welcome: "Ask a question about methodological choices, results, or model robustness. Answers are based on the dashboard data, the project source code and academic documentation.",
        rag_suggestions_label: 'Frequent questions',
        rag_input_placeholder: 'Ask your question…',
        rag_footnote: 'Endpoint <code>POST /api/rag</code> — Claude API (Anthropic) integration in progress.',
        skip_link: 'Skip to content',
        back_to_top: 'Back to top',
        exec_label: 'Key result',
        exec_gain_label: 'MAPE gain vs naïve baseline k·V₀',
        exec_portfolio_label: 'vehicles predicted on the portfolio',
        exec_decote_label: 'average portfolio depreciation',
        ctx_insight: 'RV represents <strong>40&ndash;60%</strong> of the catalogue price: a few percentage points of error translate to millions of euros at the scale of the portfolio.',
        eda_insight: 'Stable distribution over <strong>transactions</strong> filtered to 12&ndash;108 months; median <code>log_ratio</code> &asymp; -1 confirms a typical depreciation of ~63%.',
        hicp_insight: 'Three HICP features bring MAPE down from <strong>~13%</strong> to <strong>~7%</strong>; Brent and KBA market tension tested add nothing more — the macro signal is already captured.',
        feat_insight: '<strong>62 models</strong> reduced to around ten homogeneous depreciation families via KMeans on (&alpha;, k&#8321;, k&#8322;) vectors.',
        dep_insight: 'Monotonically decreasing <strong>convex age curve</strong> confirmed empirically: Nexialog\'s exponential model is validated on real data.',
        mod_insight: '<strong>CatBoost</strong> and XGBoost nearly tied, <strong>~10 pp ahead of Ridge</strong>; a weighted XGB + CatBoost ensemble is retained for the final deliverable.',
        res_insight: 'Residuals <strong>centred on 0</strong>, majority within &plusmn;&nbsp;&euro;1,000; no systematic bias on the 20% never seen during training.',
        port_insight: '<strong>Vehicles</strong> predicted; economic coherence verified: monotonic depreciation by age and brand hierarchy preserved.',
        val_insight: 'Median gap of <strong>X%</strong> vs AutoScout24: consistent with the B2B / B2C differential — signature of a conservative model, <em>not</em> an error.',
        risk_insight: 'Total exposure <strong>~&euro;20M</strong>, ECB/EBA stress tests -5% to -15%, top&nbsp;10 at-risk vehicles <strong>directly actionable</strong> for early resale.',
        sim_insight: 'Real-time <strong>CatBoost</strong> prediction on a personalised vehicle — end-to-end operational demonstration of the model.',
    },
};

let currentLang = 'fr';

function setupMobileNav() {
    const toggle = document.getElementById('mobile-nav-toggle');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('mobile-nav-overlay');
    if (!toggle || !sidebar || !overlay) return;
    const close = () => {
        sidebar.classList.remove('open');
        overlay.classList.remove('visible');
        toggle.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
        document.body.style.overflow = '';
    };
    const open = () => {
        sidebar.classList.add('open');
        overlay.classList.add('visible');
        toggle.classList.add('open');
        toggle.setAttribute('aria-expanded', 'true');
        document.body.style.overflow = 'hidden';
    };
    toggle.addEventListener('click', () => {
        if (sidebar.classList.contains('open')) close(); else open();
    });
    overlay.addEventListener('click', close);
    sidebar.querySelectorAll('a').forEach(a => a.addEventListener('click', close));
    window.addEventListener('resize', () => { if (window.innerWidth > 900) close(); });
}

function setupLanguageSwitcher() {
    const btn = document.getElementById('lang-switcher');
    const label = document.getElementById('lang-label');
    if (!btn || !label) return;
    btn.addEventListener('click', () => {
        currentLang = currentLang === 'fr' ? 'en' : 'fr';
        label.textContent = currentLang.toUpperCase();
        applyTranslations();
    });
}

function applyTranslations() {
    const t = TRANSLATIONS[currentLang];
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (t[key] !== undefined) el.textContent = t[key];
    });
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
        const key = el.getAttribute('data-i18n-html');
        if (t[key] !== undefined) el.innerHTML = t[key];
    });
    // Attribute translations (e.g. aria-label="back_to_top")
    const btnTop = document.getElementById('back-to-top');
    if (btnTop && t.back_to_top) btnTop.setAttribute('aria-label', t.back_to_top);
    // Executive summary headline contains dynamic values — rebuild.
    if (typeof DATA !== 'undefined' && DATA && DATA.meta) {
        setupExecutiveSummary();
    }
}


// ============================================================
// SECTION 08 — PIPELINE STRUCTURÉ (STRUCTURED PIPELINE)
// Validation indépendante : split aléatoire stratifié + target log_ratio.
// 5 sous-renders : metrics chart + table, SHAP barres, segments brand/fuel,
// + remplissage des spans dyn-houssem-*.
// ============================================================
function renderHoussemPipeline(data) {
    if (!data) return;

    // -- Volumétrie + champion (spans dyn-houssem-*)
    if (data.data_audit) {
        setText('dyn-houssem-n-um', data.data_audit.n_used_market.toLocaleString('fr-FR'));
        setText('dyn-houssem-n-pf', data.data_audit.n_portfolio.toLocaleString('fr-FR'));
    }
    if (data.best_model) {
        setText('dyn-houssem-best-model', data.best_model);
    }

    // -- Chart metrics : 4 modèles × {MAE, R², MAPE} dans 3 chart-cards séparées
    const mc = data.model_comparison;
    if (mc && document.getElementById('chart-houssem-mae')) {
        const labels = mc.models;
        const colorByName = {
            ElasticNet:        '#94a3b8',
            RandomForest:      '#8b5cf6',
            CatBoost_baseline: '#4ade80',
            CatBoost_main:     '#10b981',
        };
        const colors = labels.map(n => colorByName[n] || '#64748b');

        const labelFont = { size: 11, family: 'JetBrains Mono', color: '#f1f5f9' };
        const baseLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 11 },
            xaxis: { gridcolor: 'rgba(148,163,184,0.08)', tickangle: -25, automargin: true },
            yaxis: { gridcolor: 'rgba(148,163,184,0.08)', automargin: true },
            showlegend: false,
            margin: { t: 20, b: 50, l: 50, r: 15 },
            height: 280,
        };
        const plotConfig = { displayModeBar: false, responsive: true };

        // MAE
        Plotly.newPlot('chart-houssem-mae', [{
            x: labels, y: mc.mae, type: 'bar',
            marker: { color: colors, opacity: 0.9 },
            text: mc.mae.map(v => v.toFixed(3)),
            textposition: 'outside', textfont: labelFont, cliponaxis: false,
            hovertemplate: '%{x} — MAE log_ratio: %{y:.4f}<extra></extra>',
        }], baseLayout, plotConfig);

        // R²
        const r2Min = Math.min(0, ...mc.r2);
        Plotly.newPlot('chart-houssem-r2', [{
            x: labels, y: mc.r2, type: 'bar',
            marker: { color: colors, opacity: 0.9 },
            text: mc.r2.map(v => v.toFixed(3)),
            textposition: 'outside', textfont: labelFont, cliponaxis: false,
            hovertemplate: '%{x} — R² log_ratio: %{y:.4f}<extra></extra>',
        }], { ...baseLayout, yaxis: { ...baseLayout.yaxis, range: [Math.min(r2Min, 0) - 0.05, 1] } },
            plotConfig);

        // MAPE
        Plotly.newPlot('chart-houssem-mape', [{
            x: labels, y: mc.mape, type: 'bar',
            marker: { color: colors, opacity: 0.9 },
            text: mc.mape.map(v => v.toFixed(1) + '%'),
            textposition: 'outside', textfont: labelFont, cliponaxis: false,
            hovertemplate: '%{x} — MAPE log_ratio: %{y:.2f}%<extra></extra>',
        }], baseLayout, plotConfig);

        // Table métriques en dessous
        const tableEl = document.getElementById('houssem-metrics-table');
        if (tableEl) {
            const rows = labels.map((m, i) => {
                const isBest = m === data.best_model;
                const cls = isBest ? ' style="font-weight:600;color:#10b981"' : '';
                return `<tr${cls}>
                    <td>${m}${isBest ? ' ★' : ''}</td>
                    <td>${mc.mae[i].toFixed(4)}</td>
                    <td>${mc.rmse[i].toFixed(4)}</td>
                    <td>${mc.r2[i].toFixed(4)}</td>
                    <td>${mc.mape[i].toFixed(2)}%</td>
                    <td>${mc.medae[i].toFixed(4)}</td>
                </tr>`;
            }).join('');
            tableEl.innerHTML = `
                <table style="width:100%;border-collapse:collapse">
                    <thead>
                        <tr style="border-bottom:1px solid rgba(148,163,184,0.2);color:#cbd5e1;text-align:left">
                            <th style="padding:0.5rem">Modèle</th>
                            <th style="padding:0.5rem">MAE</th>
                            <th style="padding:0.5rem">RMSE</th>
                            <th style="padding:0.5rem">R²</th>
                            <th style="padding:0.5rem">MAPE</th>
                            <th style="padding:0.5rem">MedAE</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>`;
        }
    }

    // -- SHAP top features (barres horizontales)
    const shap = data.shap_top;
    const elShap = document.getElementById('chart-houssem-shap');
    if (shap && elShap) {
        Plotly.newPlot('chart-houssem-shap', [{
            x: shap.values.slice().reverse(),
            y: shap.features.slice().reverse(),
            type: 'bar', orientation: 'h',
            marker: { color: '#10b981', opacity: 0.85 },
            text: shap.values.slice().reverse().map(v => v.toFixed(3)),
            textposition: 'outside',
            textfont: { size: 11, family: 'JetBrains Mono', color: '#f1f5f9' },
            cliponaxis: false,
            hovertemplate: '%{y} — mean |SHAP|: %{x:.4f}<extra></extra>',
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 12 },
            xaxis: { gridcolor: 'rgba(148,163,184,0.08)', title: { text: 'mean |SHAP value|' } },
            yaxis: { gridcolor: 'rgba(148,163,184,0.08)', automargin: true },
            margin: { t: 20, b: 50, l: 140, r: 60 },
            height: Math.max(280, shap.features.length * 32),
        }, { displayModeBar: false, responsive: true });
    }

    // -- Segment evaluation (brand)
    const sb = data.segment_brand;
    const elSb = document.getElementById('chart-houssem-segment-brand');
    if (sb && elSb) {
        const labels = sb.map(r => r.segment);
        const mapes = sb.map(r => r.mape);
        Plotly.newPlot(elSb.id, [{
            x: labels, y: mapes, type: 'bar',
            marker: { color: ['#3b82f6', '#10b981', '#f59e0b'], opacity: 0.85 },
            text: mapes.map(v => v.toFixed(1) + '%'),
            textposition: 'outside',
            textfont: { size: 11, family: 'JetBrains Mono', color: '#f1f5f9' },
            cliponaxis: false,
            hovertemplate: '%{x} — MAPE: %{y:.2f}% (n=%{customdata})<extra></extra>',
            customdata: sb.map(r => r.n.toLocaleString('fr-FR')),
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 11 },
            xaxis: { gridcolor: 'rgba(148,163,184,0.08)' },
            yaxis: { gridcolor: 'rgba(148,163,184,0.08)', title: { text: 'MAPE (log_ratio, %)' } },
            margin: { t: 20, b: 50, l: 60, r: 15 },
        }, { displayModeBar: false, responsive: true });

        const tEl = document.getElementById('houssem-segment-brand-table');
        if (tEl) {
            tEl.innerHTML = `
                <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
                    <thead><tr style="border-bottom:1px solid rgba(148,163,184,0.2);color:#cbd5e1;text-align:left">
                        <th style="padding:0.4rem">Marque</th><th style="padding:0.4rem">N</th>
                        <th style="padding:0.4rem">MAE</th><th style="padding:0.4rem">R²</th><th style="padding:0.4rem">MAPE</th>
                    </tr></thead>
                    <tbody>${sb.map(r => `<tr>
                        <td style="padding:0.4rem">${r.segment}</td>
                        <td style="padding:0.4rem">${r.n.toLocaleString('fr-FR')}</td>
                        <td style="padding:0.4rem">${r.mae.toFixed(4)}</td>
                        <td style="padding:0.4rem">${r.r2.toFixed(3)}</td>
                        <td style="padding:0.4rem">${r.mape.toFixed(2)}%</td>
                    </tr>`).join('')}</tbody>
                </table>`;
        }
    }

    // -- Segment evaluation (fuel)
    const sf = data.segment_fuel;
    const elSf = document.getElementById('chart-houssem-segment-fuel');
    if (sf && elSf) {
        const labels = sf.map(r => r.segment);
        const mapes = sf.map(r => r.mape);
        const fuelColors = {
            DIESEL: '#3b82f6', PETROL: '#f59e0b', HYBRID: '#10b981',
            ELECTRIC: '#06b6d4', PHEV: '#8b5cf6', LPG: '#94a3b8',
        };
        Plotly.newPlot(elSf.id, [{
            x: labels, y: mapes, type: 'bar',
            marker: { color: labels.map(l => fuelColors[l] || '#64748b'), opacity: 0.85 },
            text: mapes.map(v => v.toFixed(1) + '%'),
            textposition: 'outside',
            textfont: { size: 11, family: 'JetBrains Mono', color: '#f1f5f9' },
            cliponaxis: false,
            hovertemplate: '%{x} — MAPE: %{y:.2f}% (n=%{customdata})<extra></extra>',
            customdata: sf.map(r => r.n.toLocaleString('fr-FR')),
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 11 },
            xaxis: { gridcolor: 'rgba(148,163,184,0.08)' },
            yaxis: { gridcolor: 'rgba(148,163,184,0.08)', title: { text: 'MAPE (log_ratio, %)' } },
            margin: { t: 20, b: 50, l: 60, r: 15 },
        }, { displayModeBar: false, responsive: true });

        const tEl = document.getElementById('houssem-segment-fuel-table');
        if (tEl) {
            tEl.innerHTML = `
                <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
                    <thead><tr style="border-bottom:1px solid rgba(148,163,184,0.2);color:#cbd5e1;text-align:left">
                        <th style="padding:0.4rem">Carburant</th><th style="padding:0.4rem">N</th>
                        <th style="padding:0.4rem">MAE</th><th style="padding:0.4rem">R²</th><th style="padding:0.4rem">MAPE</th>
                    </tr></thead>
                    <tbody>${sf.map(r => `<tr>
                        <td style="padding:0.4rem">${r.segment}</td>
                        <td style="padding:0.4rem">${r.n.toLocaleString('fr-FR')}</td>
                        <td style="padding:0.4rem">${r.mae.toFixed(4)}</td>
                        <td style="padding:0.4rem">${r.r2.toFixed(3)}</td>
                        <td style="padding:0.4rem">${r.mape.toFixed(2)}%</td>
                    </tr>`).join('')}</tbody>
                </table>`;
        }
    }
}


// ============================================================
// SECTION 07 — Placeholders « -- » non remplis par les renders.
// Branche dyn-train-range, dyn-n-train, dyn-test-range, dyn-n-test,
// dyn-n-models-bench (bandeau du haut de 07.1) et les spans
// dyn-ropc-* (bullets « Verdict walk-forward ») + dyn-top-features.
// ============================================================
function fillSection07Placeholders(charts) {
    if (!charts) return;

    // 1) Bandeau split temporel post-COVID — depuis comparison_meta
    const meta = charts.model_comparison_meta || {};
    const mc   = charts.model_comparison || {};
    const fmtN = v => (v != null ? Number(v).toLocaleString('fr-FR') : '--');
    const fmtRange = r => (Array.isArray(r) ? `${r[0]} → ${r[1]}` : '--');

    setText('dyn-n-models-bench', mc.models ? mc.models.length : '--');
    setText('dyn-train-range',     fmtRange(meta.train_range));
    setText('dyn-test-range',      fmtRange(meta.test_range));
    setText('dyn-n-train',         fmtN(meta.n_train));
    setText('dyn-n-test',          fmtN(meta.n_test));
    // Variantes -tip dans les tooltips de la même section
    setText('dyn-train-range-tip', fmtRange(meta.train_range));
    setText('dyn-test-range-tip',  fmtRange(meta.test_range));
    setText('dyn-n-train-tip',     fmtN(meta.n_train));
    setText('dyn-n-test-tip',      fmtN(meta.n_test));

    // 2) Bullets « Verdict walk-forward » — depuis rolling_origin_postcovid.summary
    const ropc = charts.rolling_origin_postcovid;
    if (ropc && ropc.summary) {
        const s = ropc.summary;
        const best = s.best_model_global;
        const per = (s.per_model && best) ? s.per_model[best] || {} : {};
        setText('dyn-ropc-n-models',  (ropc.models || []).length || '4');
        setText('dyn-ropc-best-model', modelDisplayName(best));
        setText('dyn-ropc-mape-mean',  per.mape_eur_mean != null ? per.mape_eur_mean.toFixed(2) : '--');
        setText('dyn-ropc-mape-std',   per.mape_eur_std  != null ? per.mape_eur_std.toFixed(2)  : '--');
        setText('dyn-ropc-mape-min',   per.mape_eur_min  != null ? per.mape_eur_min.toFixed(2)  : '--');
        setText('dyn-ropc-mape-max',   per.mape_eur_max  != null ? per.mape_eur_max.toFixed(2)  : '--');
        if (per.mape_eur_std != null) {
            const stable = per.mape_eur_std < 1.0 ? 'Classement stable entre cutoffs.'
                          : per.mape_eur_std < 2.0 ? 'Classement modérément stable.'
                          : 'Classement plus volatile entre cutoffs.';
            setText('dyn-ropc-stability', stable);
        }
        if (s.lift_pts_mean != null) {
            setText('dyn-ropc-lift', (s.lift_pts_mean > 0 ? '+' : '') + s.lift_pts_mean.toFixed(2));
        }
    }

    // 3) Top features (bullet « Lecture finale ») — depuis feature_importance
    const fi = charts.feature_importance;
    if (fi && Array.isArray(fi.features) && fi.features.length > 0) {
        const top3 = fi.features.slice(0, 3).map(f => `<code>${f}</code>`).join(', ');
        const el = document.getElementById('dyn-top-features');
        if (el) el.innerHTML = top3;
    }

    // 4) Best R² / MAE pour les chart-notes (Prédit vs Observé, Résidus)
    if (mc.models && mc.r2 && mc.mae) {
        // Index du best model = celui choisi par le pipeline (best_name côté backend)
        // Heuristique : index du R² maximal côté EUR
        let idxBest = 0;
        let r2Max = -Infinity;
        for (let i = 0; i < mc.r2.length; i++) {
            if (mc.r2[i] != null && mc.r2[i] > r2Max) { r2Max = mc.r2[i]; idxBest = i; }
        }
        setText('dyn-best-r2',    mc.r2[idxBest] != null ? mc.r2[idxBest].toFixed(3) : '--');
        setText('dyn-best-mae',   mc.mae[idxBest] != null ? Number(mc.mae[idxBest]).toLocaleString('fr-FR') : '--');
        if (mc.r2_logratio && mc.r2_logratio[idxBest] != null) {
            setText('dyn-best-r2-lr', mc.r2_logratio[idxBest].toFixed(3));
        }
    }
}


// ============================================================
// QR CODE SHARE — bouton flottant qui ouvre un modal avec un QR
// pointant par défaut sur window.location.href (override possible
// via un champ texte, ex. URL ngrok). Utilise la lib `qrious`
// (jsdelivr) — canvas direct, API simple : new QRious({...}).
// ============================================================
function setupQrShare() {
    const trigger = document.getElementById('qr-trigger');
    const modal = document.getElementById('qr-modal');
    const backdrop = document.getElementById('qr-modal-backdrop');
    const closeBtn = document.getElementById('qr-modal-close');
    const urlInput = document.getElementById('qr-modal-url');
    const qrContainer = document.getElementById('qr-modal-qr');
    const copyBtn = document.getElementById('qr-modal-copy');
    const copyLabel = document.getElementById('qr-modal-copy-label');

    if (!trigger || !modal || !urlInput || !qrContainer) return;

    // qrious expose QRious globalement. Si le CDN n'a pas chargé, on
    // désactive proprement le bouton plutôt que de crasher.
    if (typeof QRious === 'undefined') {
        console.warn('[qr-share] QRious lib absente — bouton désactivé.');
        trigger.disabled = true;
        trigger.style.opacity = '0.4';
        trigger.title = 'QR code indisponible (lib non chargée)';
        return;
    }

    // Un canvas unique réutilisé via qr.value = newText (QRious re-rend
    // automatiquement quand on assigne une nouvelle valeur).
    qrContainer.innerHTML = '';
    const canvas = document.createElement('canvas');
    qrContainer.appendChild(canvas);
    const qr = new QRious({
        element: canvas,
        value: 'about:blank',
        size: 220,
        background: '#ffffff',
        foreground: '#0f172a',
        level: 'M',
    });
    const renderQr = (text) => {
        qr.value = text || 'about:blank';
    };

    // URL par défaut : priorité à l'URL ngrok détectée côté Flask (via
    // <meta name="ngrok-url">) si on est sur localhost/127.0.0.1 — sinon
    // window.location.href.
    const getDefaultUrl = () => {
        const ngrokMeta = document.querySelector('meta[name="ngrok-url"]');
        const ngrokUrl = ngrokMeta ? ngrokMeta.getAttribute('content') : null;
        const host = window.location.hostname;
        const isLocal = host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0';
        if (ngrokUrl && isLocal) return ngrokUrl;
        return window.location.href;
    };

    const linkEl = document.getElementById('qr-modal-link');
    const openModal = () => {
        const url = getDefaultUrl();
        urlInput.value = url;
        if (linkEl) linkEl.href = url;
        renderQr(url);
        modal.hidden = false;
    };
    const closeModal = () => {
        modal.hidden = true;
    };

    trigger.addEventListener('click', openModal);
    closeBtn.addEventListener('click', closeModal);
    backdrop.addEventListener('click', closeModal);
    document.addEventListener('keydown', (e) => {
        if (!modal.hidden && e.key === 'Escape') closeModal();
    });

    // Copie dans le presse-papier avec feedback visuel.
    copyBtn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(urlInput.value);
            copyBtn.classList.add('is-success');
            copyLabel.textContent = 'Copié !';
            setTimeout(() => {
                copyBtn.classList.remove('is-success');
                copyLabel.textContent = 'Copier';
            }, 1600);
        } catch (e) {
            // Fallback : sélectionne le champ pour que l'utilisateur copie
            // manuellement (navigateurs sans API clipboard / contexte non HTTPS).
            urlInput.select();
            copyLabel.textContent = 'Cmd+C';
        }
    });
}
