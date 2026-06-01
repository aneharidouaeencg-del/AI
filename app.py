"""
Outil de Recommandation d'Incoterms - Gala Transit Transport
Compatible Google Colab + Streamlit
Un seul fichier autonome - Aucun caractere special interdit
"""

import streamlit as st
import re
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# CONFIGURATION PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Incoterm Advisor | Gala Transit",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS PERSONNALISE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Fond general */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid #21262d;
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 2rem;
}
.app-title {
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-subtitle {
    font-size: 0.9rem;
    color: #7d8590;
    margin-top: 0.25rem;
    font-family: 'IBM Plex Mono', monospace;
}
.brand-badge {
    display: inline-block;
    background: #1f6feb;
    color: white;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 2rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Onglets */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #161b22;
    border-radius: 0.5rem;
    padding: 0.25rem;
    border: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7d8590;
    border-radius: 0.35rem;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    font-size: 0.9rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e6edf3 !important;
}

/* Cards */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 0.75rem;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}

/* Table de saisie */
.cost-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}
.cost-table th {
    background: #21262d;
    color: #7d8590;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.6rem 1rem;
    text-align: left;
}
.cost-table th:not(:first-child) {
    text-align: center;
}
.cost-table td {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #21262d;
    color: #c9d1d9;
}
.cost-table tr:hover td {
    background: #1c2128;
}

/* Resultat */
.result-card {
    background: linear-gradient(135deg, #0d1117, #161b22);
    border: 1px solid #1f6feb;
    border-radius: 0.75rem;
    padding: 2rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
}
.incoterm-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1;
    letter-spacing: -0.02em;
}
.incoterm-name {
    font-size: 1rem;
    color: #7d8590;
    margin-top: 0.25rem;
}
.confidence-bar {
    background: #21262d;
    border-radius: 1rem;
    height: 8px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 1rem;
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
    transition: width 0.6s ease;
}

/* Tableau de repartition */
.repartition-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 1rem;
}
.repartition-table th {
    background: #21262d;
    color: #7d8590;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.5rem 0.75rem;
}
.repartition-table td {
    padding: 0.45rem 0.75rem;
    border-bottom: 1px solid #21262d;
    color: #c9d1d9;
}
.pill-vendeur {
    background: #1a3a5c;
    color: #58a6ff;
    border-radius: 0.25rem;
    padding: 0.1rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}
.pill-acheteur {
    background: #1a3a2c;
    color: #3fb950;
    border-radius: 0.25rem;
    padding: 0.1rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}

/* Alerte */
.alert-warning {
    background: #2d2007;
    border: 1px solid #9e6a03;
    border-left: 4px solid #d29922;
    border-radius: 0.5rem;
    padding: 1rem 1.25rem;
    color: #e3b341;
    font-size: 0.875rem;
    margin-top: 1rem;
}
.alert-info {
    background: #0d2340;
    border: 1px solid #1158b0;
    border-left: 4px solid #1f6feb;
    border-radius: 0.5rem;
    padding: 1rem 1.25rem;
    color: #79c0ff;
    font-size: 0.875rem;
    margin-top: 1rem;
}
.alert-success {
    background: #0d2a1f;
    border: 1px solid #1a6430;
    border-left: 4px solid #3fb950;
    border-radius: 0.5rem;
    padding: 1rem 1.25rem;
    color: #56d364;
    font-size: 0.875rem;
    margin-top: 1rem;
}

/* Bouton principal */
.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.65rem 2rem;
    font-weight: 600;
    font-size: 0.9rem;
    font-family: 'IBM Plex Sans', sans-serif;
    cursor: pointer;
    transition: background 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: #388bfd;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #21262d;
    border-color: #30363d;
    color: #e6edf3;
    border-radius: 0.5rem;
}

/* Divider */
hr {
    border-color: #21262d;
    margin: 1.5rem 0;
}

/* Upload zone */
.uploadedFile {
    background: #161b22 !important;
    border-color: #21262d !important;
    color: #e6edf3 !important;
}

/* Radio buttons */
.stRadio > div {
    gap: 0.5rem;
}
.stRadio label {
    color: #c9d1d9;
}

/* Metric */
.metric-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: center;
}
.metric-label {
    font-size: 0.7rem;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DONNEES : POSTES DE COUT
# ─────────────────────────────────────────────
POSTES = [
    ("emballage",       "Emballage et marquage"),
    ("chargement",      "Chargement initial (usine/entrepot)"),
    ("dedouanement_ex", "Dedouanement export"),
    ("transport_dep",   "Transport local depart"),
    ("mise_a_bord",     "Mise a bord (FOB)"),
    ("fret",            "Fret maritime / transport principal"),
    ("assurance",       "Assurance"),
    ("dechargement",    "Dechargement"),
    ("transport_arr",   "Transport final / livraison"),
    ("dedouanement_im", "Dedouanement import"),
    ("droits_taxes",    "Droits et taxes import"),
]

# ─────────────────────────────────────────────
# LOGIQUE DE RECOMMANDATION
# ─────────────────────────────────────────────
@dataclass
class IncoResult:
    code: str
    name: str
    confidence: int
    repartition: dict  # poste -> "vendeur" | "acheteur"
    alerte: str
    alerte_type: str   # "warning" | "info" | "success"
    point_livraison: str

INCOTERM_META = {
    "EXW": {
        "name": "Ex Works",
        "point": "Entrepot du vendeur",
        "alerte": ("ATTENTION : Avec EXW, l'acheteur supporte tous les risques et frais "
                   "des le depart de l'entrepot, y compris le chargement. Ce terme est "
                   "deconseille pour l'export car le vendeur ne depose pas les formalites douanieres."),
        "alerte_type": "warning",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "acheteur",
            "dedouanement_ex": "acheteur",
            "transport_dep": "acheteur",
            "mise_a_bord": "acheteur",
            "fret": "acheteur",
            "assurance": "acheteur",
            "dechargement": "acheteur",
            "transport_arr": "acheteur",
            "dedouanement_im": "acheteur",
            "droits_taxes": "acheteur",
        }
    },
    "FCA": {
        "name": "Free Carrier",
        "point": "Lieu designe (terminal / entrepot depart)",
        "alerte": ("INFO : FCA est l'Incoterm polyvalent recommande par la CCI pour remplacer "
                   "FOB dans le cadre du transport conteneurise. Le risque se transfere a "
                   "la remise au transporteur designe par l'acheteur."),
        "alerte_type": "info",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "vendeur",
            "dedouanement_ex": "vendeur",
            "transport_dep": "vendeur",
            "mise_a_bord": "acheteur",
            "fret": "acheteur",
            "assurance": "acheteur",
            "dechargement": "acheteur",
            "transport_arr": "acheteur",
            "dedouanement_im": "acheteur",
            "droits_taxes": "acheteur",
        }
    },
    "FAS": {
        "name": "Free Alongside Ship",
        "point": "Quai du port d'embarquement (le long du navire)",
        "alerte": ("INFO : FAS est reserve au transport maritime et fluvial de marchandises "
                   "en vrac ou non conteneurisees. Le risque se transfere quand la marchandise "
                   "est placee le long du navire."),
        "alerte_type": "info",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "vendeur",
            "dedouanement_ex": "vendeur",
            "transport_dep": "vendeur",
            "mise_a_bord": "acheteur",
            "fret": "acheteur",
            "assurance": "acheteur",
            "dechargement": "acheteur",
            "transport_arr": "acheteur",
            "dedouanement_im": "acheteur",
            "droits_taxes": "acheteur",
        }
    },
    "FOB": {
        "name": "Free On Board",
        "point": "A bord du navire, port d'embarquement",
        "alerte": ("INFO : FOB est le terme le plus utilise dans le commerce international. "
                   "Il est strictement reserve au transport maritime et fluvial. "
                   "Pour le transport conteneurise moderne, FCA est generalement prefere."),
        "alerte_type": "info",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "vendeur",
            "dedouanement_ex": "vendeur",
            "transport_dep": "vendeur",
            "mise_a_bord": "vendeur",
            "fret": "acheteur",
            "assurance": "acheteur",
            "dechargement": "acheteur",
            "transport_arr": "acheteur",
            "dedouanement_im": "acheteur",
            "droits_taxes": "acheteur",
        }
    },
    "CFR": {
        "name": "Cost and Freight",
        "point": "Port de destination convenu",
        "alerte": ("ATTENTION : Avec CFR, le vendeur paie le fret mais le risque se transfere "
                   "a l'acheteur des la mise a bord. L'acheteur doit imperativement souscrire "
                   "sa propre assurance marchandise."),
        "alerte_type": "warning",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "vendeur",
            "dedouanement_ex": "vendeur",
            "transport_dep": "vendeur",
            "mise_a_bord": "vendeur",
            "fret": "vendeur",
            "assurance": "acheteur",
            "dechargement": "acheteur",
            "transport_arr": "acheteur",
            "dedouanement_im": "acheteur",
            "droits_taxes": "acheteur",
        }
    },
    "CIF": {
        "name": "Cost, Insurance and Freight",
        "point": "Port de destination convenu",
        "alerte": ("INFO : CIF inclut une assurance minimale (Clause C - couverture limitee). "
                   "Pour une protection optimale, l'acheteur devrait exiger une couverture "
                   "etendue (Clause A) ou souscrire sa propre assurance complementaire."),
        "alerte_type": "info",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "vendeur",
            "dedouanement_ex": "vendeur",
            "transport_dep": "vendeur",
            "mise_a_bord": "vendeur",
            "fret": "vendeur",
            "assurance": "vendeur",
            "dechargement": "acheteur",
            "transport_arr": "acheteur",
            "dedouanement_im": "acheteur",
            "droits_taxes": "acheteur",
        }
    },
    "CPT": {
        "name": "Carriage Paid To",
        "point": "Lieu de destination convenu",
        "alerte": ("INFO : CPT est l'equivalent multimodal de CFR. Le risque se transfere "
                   "au premier transporteur, mais le vendeur paie jusqu'a destination. "
                   "Recommande pour le transport aerien et multimodal conteneurise."),
        "alerte_type": "info",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "vendeur",
            "dedouanement_ex": "vendeur",
            "transport_dep": "vendeur",
            "mise_a_bord": "vendeur",
            "fret": "vendeur",
            "assurance": "acheteur",
            "dechargement": "acheteur",
            "transport_arr": "acheteur",
            "dedouanement_im": "acheteur",
            "droits_taxes": "acheteur",
        }
    },
    "CIP": {
        "name": "Carriage and Insurance Paid To",
        "point": "Lieu de destination convenu",
        "alerte": ("SUCCES : CIP est le terme le plus protecteur pour l'acheteur dans la "
                   "categorie C. Le vendeur doit fournir une assurance tous risques (Clause A), "
                   "ce qui est superieur au CIF. Ideal pour les marchandises a forte valeur."),
        "alerte_type": "success",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "vendeur",
            "dedouanement_ex": "vendeur",
            "transport_dep": "vendeur",
            "mise_a_bord": "vendeur",
            "fret": "vendeur",
            "assurance": "vendeur",
            "dechargement": "acheteur",
            "transport_arr": "acheteur",
            "dedouanement_im": "acheteur",
            "droits_taxes": "acheteur",
        }
    },
    "DAP": {
        "name": "Delivered at Place",
        "point": "Lieu de destination convenu (non decharge)",
        "alerte": ("INFO : Avec DAP, le vendeur livre jusqu'au lieu convenu mais ne decharge "
                   "pas. L'acheteur prend en charge le dechargement et les formalites douanieres "
                   "d'importation. Flexibilite maximale sur le lieu de livraison."),
        "alerte_type": "info",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "vendeur",
            "dedouanement_ex": "vendeur",
            "transport_dep": "vendeur",
            "mise_a_bord": "vendeur",
            "fret": "vendeur",
            "assurance": "vendeur",
            "dechargement": "acheteur",
            "transport_arr": "vendeur",
            "dedouanement_im": "acheteur",
            "droits_taxes": "acheteur",
        }
    },
    "DPU": {
        "name": "Delivered at Place Unloaded",
        "point": "Terminal ou lieu de destination convenu (decharge)",
        "alerte": ("INFO : DPU est le seul Incoterm ou le vendeur est responsable du "
                   "dechargement a destination. Le vendeur doit s'assurer d'avoir les moyens "
                   "techniques de dechargement sur le lieu convenu."),
        "alerte_type": "info",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "vendeur",
            "dedouanement_ex": "vendeur",
            "transport_dep": "vendeur",
            "mise_a_bord": "vendeur",
            "fret": "vendeur",
            "assurance": "vendeur",
            "dechargement": "vendeur",
            "transport_arr": "vendeur",
            "dedouanement_im": "acheteur",
            "droits_taxes": "acheteur",
        }
    },
    "DDP": {
        "name": "Delivered Duty Paid",
        "point": "Lieu de destination convenu (droits acquittes)",
        "alerte": ("ATTENTION : DDP impose au vendeur la responsabilite maximale, y compris "
                   "le dedouanement import et le paiement des droits et taxes dans le pays "
                   "de destination. Le vendeur doit imperativement avoir le statut d'importateur "
                   "ou mandater un commissionnaire agree dans le pays acheteur."),
        "alerte_type": "warning",
        "repartition": {
            "emballage": "vendeur",
            "chargement": "vendeur",
            "dedouanement_ex": "vendeur",
            "transport_dep": "vendeur",
            "mise_a_bord": "vendeur",
            "fret": "vendeur",
            "assurance": "vendeur",
            "dechargement": "vendeur",
            "transport_arr": "vendeur",
            "dedouanement_im": "vendeur",
            "droits_taxes": "vendeur",
        }
    },
}

def score_incoterm(selections: dict, mode_transport: str, type_marchandise: str) -> list:
    """
    Calcule un score de correspondance pour chaque Incoterm.
    selections: dict {poste_id: "vendeur" | "acheteur"}
    Retourne une liste triee de (incoterm_code, score, confidence)
    """
    scores = {}
    for code, meta in INCOTERM_META.items():
        ref = meta["repartition"]
        total = len(POSTES)
        matches = sum(1 for pid, _ in POSTES if selections.get(pid) == ref.get(pid))
        scores[code] = matches

    # Ajustements selon mode de transport
    MARITIME_ONLY = ["FAS", "FOB", "CFR", "CIF"]
    ALL_MODES     = ["EXW", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP"]

    if mode_transport == "Aerien":
        for code in MARITIME_ONLY:
            scores[code] = max(0, scores[code] - 3)
    elif mode_transport == "Routier":
        for code in MARITIME_ONLY:
            scores[code] = max(0, scores[code] - 2)

    # Ajustements selon type de marchandise
    if type_marchandise == "Vrac":
        scores["FAS"] = scores.get("FAS", 0) + 1
        scores["FOB"] = scores.get("FOB", 0) + 1
    if type_marchandise in ("Fragile", "Perissable"):
        scores["CIP"] = scores.get("CIP", 0) + 1
        scores["CIF"] = scores.get("CIF", 0) + 1
    if type_marchandise == "Conteneurise":
        scores["FCA"] = scores.get("FCA", 0) + 1
        scores["CPT"] = scores.get("CPT", 0) + 1

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_score = sorted_scores[0][1]
    total_possible = len(POSTES) + 2  # max avec bonus

    # Confiance : part du meilleur score sur le max theorique
    confidence = min(100, int((best_score / max(total_possible, 1)) * 100))
    # Bonus si dominance claire
    if len(sorted_scores) >= 2 and sorted_scores[0][1] > sorted_scores[1][1] + 2:
        confidence = min(100, confidence + 10)

    return sorted_scores, confidence


def build_result(code: str, confidence: int) -> IncoResult:
    meta = INCOTERM_META[code]
    return IncoResult(
        code=code,
        name=meta["name"],
        confidence=confidence,
        repartition=meta["repartition"],
        alerte=meta["alerte"],
        alerte_type=meta["alerte_type"],
        point_livraison=meta["point"],
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="brand-badge">GALA TRANSIT TRANSPORT</div>
    <div class="app-title">Incoterm Advisor</div>
    <div class="app-subtitle">Outil de recommandation des regles Incoterms 2020 — CCI Paris</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ─────────────────────────────────────────────
tab_calc, tab_ocr = st.tabs(["  Calculateur interactif", "  Import de facture (OCR)"])


# ═══════════════════════════════════════════════
# ONGLET 1 : CALCULATEUR INTERACTIF
# ═══════════════════════════════════════════════
with tab_calc:

    col_params, col_result = st.columns([1.2, 1], gap="large")

    with col_params:
        # Parametres generaux
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Parametres de l\'expedition</div>', unsafe_allow_html=True)

        col_m, col_t = st.columns(2)
        with col_m:
            mode_transport = st.selectbox(
                "Mode de transport",
                ["Maritime", "Aerien", "Routier"],
                key="mode_transport"
            )
        with col_t:
            type_marchandise = st.selectbox(
                "Type de marchandise",
                ["Standard", "Conteneurise", "Vrac", "Fragile", "Perissable"],
                key="type_marchandise"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Tableau de saisie
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Repartition des couts et responsabilites</div>', unsafe_allow_html=True)
        st.caption("Pour chaque poste, selectionnez la partie qui en a la charge.")

        selections = {}
        for pid, label in POSTES:
            col_label, col_sel = st.columns([2, 1])
            with col_label:
                st.markdown(f"<div style='padding:0.4rem 0; color:#c9d1d9; font-size:0.875rem;'>{label}</div>", unsafe_allow_html=True)
            with col_sel:
                choice = st.radio(
                    label,
                    ["Vendeur", "Acheteur"],
                    key=f"sel_{pid}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                selections[pid] = choice.lower()

        st.markdown('</div>', unsafe_allow_html=True)

        # Bouton
        btn = st.button("Analyser et recommander l'Incoterm", key="btn_calc")

    with col_result:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if btn:
            sorted_scores, confidence = score_incoterm(selections, mode_transport, type_marchandise)
            best_code = sorted_scores[0][0]
            result = build_result(best_code, confidence)

            # Badge Incoterm
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:0.72rem; color:#7d8590; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">
                    Incoterm recommande
                </div>
                <div class="incoterm-badge">{result.code}</div>
                <div class="incoterm-name">{result.name}</div>
                <hr style="border-color:#21262d; margin:1rem 0;">
                <div style="font-size:0.72rem; color:#7d8590; margin-bottom:0.4rem;">
                    Niveau de confiance
                </div>
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div class="confidence-bar" style="flex:1;">
                        <div class="confidence-fill" style="width:{result.confidence}%;"></div>
                    </div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:1.2rem; font-weight:700; color:#58a6ff; min-width:3rem;">
                        {result.confidence}%
                    </div>
                </div>
                <div style="font-size:0.8rem; color:#7d8590; margin-top:1rem;">
                    Point de transfert de risque<br>
                    <span style="color:#c9d1d9;">{result.point_livraison}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Alternatives
            if len(sorted_scores) >= 3:
                alt1_code, alt1_score = sorted_scores[1]
                alt2_code, alt2_score = sorted_scores[2]
                st.markdown(f"""
                <div style="display:flex; gap:0.75rem; margin-top:0.75rem;">
                    <div class="metric-box" style="flex:1;">
                        <div class="metric-label">Alternative 1</div>
                        <div class="metric-value" style="font-size:1.3rem;">{alt1_code}</div>
                        <div style="font-size:0.7rem; color:#7d8590;">{INCOTERM_META[alt1_code]['name']}</div>
                    </div>
                    <div class="metric-box" style="flex:1;">
                        <div class="metric-label">Alternative 2</div>
                        <div class="metric-value" style="font-size:1.3rem;">{alt2_code}</div>
                        <div style="font-size:0.7rem; color:#7d8590;">{INCOTERM_META[alt2_code]['name']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Alerte
            icon_map = {"warning": "!", "info": "i", "success": "v"}
            icon = icon_map.get(result.alerte_type, "i")
            st.markdown(f"""
            <div class="alert-{result.alerte_type}">
                <strong>[{icon}]</strong> {result.alerte}
            </div>
            """, unsafe_allow_html=True)

            # Tableau de repartition
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">Tableau de repartition — {result.code}</div>', unsafe_allow_html=True)

            rows_html = ""
            for pid, label in POSTES:
                qui = result.repartition.get(pid, "acheteur")
                pill = (f'<span class="pill-vendeur">VENDEUR</span>' if qui == "vendeur"
                        else f'<span class="pill-acheteur">ACHETEUR</span>')
                rows_html += f"<tr><td>{label}</td><td style='text-align:center;'>{pill}</td></tr>"

            st.markdown(f"""
            <table class="repartition-table">
                <thead>
                    <tr>
                        <th>Poste de cout</th>
                        <th style="text-align:center;">Responsable</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Etat initial
            st.markdown("""
            <div style="background:#161b22; border:1px dashed #30363d; border-radius:0.75rem;
                        padding:3rem 2rem; text-align:center; margin-top:1rem;">
                <div style="font-size:2.5rem; margin-bottom:1rem;">🚢</div>
                <div style="color:#7d8590; font-size:0.9rem; line-height:1.6;">
                    Remplissez le formulaire a gauche<br>et cliquez sur <strong style="color:#c9d1d9;">Analyser</strong>
                    pour obtenir<br>votre recommandation Incoterm.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Rappel des 11 Incoterms
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Les 11 Incoterms 2020 (CCI)</div>', unsafe_allow_html=True)

            groups = [
                ("Groupe E — Depart", ["EXW"]),
                ("Groupe F — Transport principal non paye par le vendeur", ["FCA", "FAS", "FOB"]),
                ("Groupe C — Transport principal paye par le vendeur", ["CFR", "CIF", "CPT", "CIP"]),
                ("Groupe D — Livraison a destination", ["DAP", "DPU", "DDP"]),
            ]
            for group_name, codes in groups:
                st.markdown(f"<div style='font-size:0.72rem; color:#7d8590; margin-top:0.75rem; margin-bottom:0.3rem;'>{group_name}</div>", unsafe_allow_html=True)
                pills = " ".join([
                    f'<span style="background:#21262d; color:#58a6ff; font-family:IBM Plex Mono,monospace; '
                    f'font-weight:600; font-size:0.8rem; padding:0.2rem 0.7rem; border-radius:0.25rem; '
                    f'margin-right:0.3rem;">{c}</span>'
                    for c in codes
                ])
                st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# ONGLET 2 : IMPORT DE FACTURE OCR
# ═══════════════════════════════════════════════
with tab_ocr:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Import de facture commerciale</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#7d8590; font-size:0.875rem; margin-bottom:1rem; line-height:1.6;">
        Importez une facture PDF ou image. L'outil extrait automatiquement les informations
        pertinentes (Incoterm mentionne, conditions de vente, lieu de livraison) pour
        valider ou suggerer l'Incoterm adapte.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Deposez votre fichier ici",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Formats acceptes : PDF, PNG, JPG, JPEG"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Detection du type de fichier
        file_type = uploaded_file.type
        raw_text = ""
        ocr_ok = False

        with st.spinner("Extraction du texte en cours..."):
            try:
                if file_type == "application/pdf":
                    try:
                        import pdfplumber
                        import io
                        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                            for page in pdf.pages:
                                t = page.extract_text()
                                if t:
                                    raw_text += t + "\n"
                        ocr_ok = True
                    except ImportError:
                        raw_text = ""
                        st.warning("pdfplumber non installe. Installez-le avec : pip install pdfplumber")
                else:
                    try:
                        from PIL import Image
                        import pytesseract
                        import io
                        image = Image.open(io.BytesIO(uploaded_file.read()))
                        raw_text = pytesseract.image_to_string(image, lang="fra+eng")
                        ocr_ok = True
                    except ImportError:
                        raw_text = ""
                        st.warning("Pytesseract ou Pillow non installe. Installez avec : pip install pytesseract Pillow")

            except Exception as e:
                st.error(f"Erreur lors de l'extraction : {e}")

        if raw_text.strip():
            # Affichage du texte extrait
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Texte extrait (OCR)</div>', unsafe_allow_html=True)
            st.text_area("", value=raw_text[:3000], height=200, disabled=True, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

            # Detection de l'Incoterm dans le texte
            incoterms_detectes = []
            for code in INCOTERM_META.keys():
                pattern = rf'\b{code}\b'
                if re.search(pattern, raw_text.upper()):
                    incoterms_detectes.append(code)

            if incoterms_detectes:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Incoterms detectes dans la facture</div>', unsafe_allow_html=True)

                for code in incoterms_detectes:
                    meta = INCOTERM_META[code]
                    st.markdown(f"""
                    <div style="background:#1c2128; border:1px solid #30363d; border-radius:0.5rem;
                                padding:1rem; margin-bottom:0.75rem; display:flex; align-items:center; gap:1.5rem;">
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:2rem; font-weight:700;
                                    color:#58a6ff; min-width:5rem;">{code}</div>
                        <div>
                            <div style="color:#e6edf3; font-weight:600;">{meta['name']}</div>
                            <div style="color:#7d8590; font-size:0.8rem; margin-top:0.2rem;">
                                Point de livraison : {meta['point']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Tableau de repartition
                    with st.expander(f"Voir la repartition des responsabilites pour {code}"):
                        rows_html = ""
                        for pid, label in POSTES:
                            qui = meta["repartition"].get(pid, "acheteur")
                            pill = (f'<span class="pill-vendeur">VENDEUR</span>' if qui == "vendeur"
                                    else f'<span class="pill-acheteur">ACHETEUR</span>')
                            rows_html += f"<tr><td>{label}</td><td style='text-align:center;'>{pill}</td></tr>"
                        st.markdown(f"""
                        <table class="repartition-table">
                            <thead><tr><th>Poste de cout</th><th style="text-align:center;">Responsable</th></tr></thead>
                            <tbody>{rows_html}</tbody>
                        </table>
                        """, unsafe_allow_html=True)

                        alerte_type = meta["alerte_type"]
                        icon = {"warning": "!", "info": "i", "success": "v"}.get(alerte_type, "i")
                        st.markdown(f"""
                        <div class="alert-{alerte_type}" style="margin-top:0.75rem;">
                            <strong>[{icon}]</strong> {meta['alerte']}
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="alert-warning">
                    <strong>[!]</strong> Aucun Incoterm standard n'a ete detecte dans ce document.
                    Verifiez que la facture mentionne bien les conditions de vente (ex. : "FOB Shanghai",
                    "CIF Casablanca", etc.) ou utilisez le calculateur interactif.
                </div>
                """, unsafe_allow_html=True)

        elif uploaded_file is not None and not ocr_ok:
            st.markdown("""
            <div class="alert-info">
                <strong>[i]</strong> Les bibliotheques OCR ne sont pas disponibles dans cet environnement.
                Installez pdfplumber (pour PDF) et pytesseract + Pillow (pour images) :
                <br><br>
                <code>pip install pdfplumber pytesseract Pillow</code>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Guide d'installation
        st.markdown("""
        <div class="alert-info">
            <strong>[i]</strong> Pour activer l'extraction OCR, installez les dependances suivantes :
            <br><br>
            <code>pip install pdfplumber pytesseract Pillow</code>
            <br><br>
            Pour les images, Tesseract-OCR doit etre installe sur le systeme.
            Sur Ubuntu/Colab : <code>apt-get install -y tesseract-ocr tesseract-ocr-fra</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-success" style="margin-top:1rem;">
            <strong>[v]</strong> Sans OCR, vous pouvez coller directement le texte de votre facture
            dans le champ ci-dessous pour l'analyse.
        </div>
        """, unsafe_allow_html=True)

        # Mode texte libre
        manual_text = st.text_area(
            "Ou collez ici le texte de votre facture",
            height=200,
            placeholder="Ex: Invoice No. 2024-001\nIncoterm: CIF Casablanca\nShipper: ABC Company...",
            key="manual_text"
        )

        if manual_text.strip():
            incoterms_trouves = []
            for code in INCOTERM_META.keys():
                if re.search(rf'\b{code}\b', manual_text.upper()):
                    incoterms_trouves.append(code)

            if incoterms_trouves:
                st.success(f"Incoterms detectes : {', '.join(incoterms_trouves)}")
                for code in incoterms_trouves:
                    meta = INCOTERM_META[code]
                    st.markdown(f"""
                    <div style="background:#1c2128; border:1px solid #30363d; border-radius:0.5rem;
                                padding:1rem; margin-bottom:0.75rem;">
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.8rem;
                                    font-weight:700; color:#58a6ff;">{code} — {meta['name']}</div>
                        <div style="color:#7d8590; font-size:0.8rem; margin-top:0.25rem;">
                            Point de livraison : {meta['point']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Aucun Incoterm detecte dans le texte saisi.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #21262d; padding-top:1rem; text-align:center;
            color:#484f58; font-size:0.75rem; font-family:'IBM Plex Mono',monospace;">
    Gala Transit Transport — Incoterm Advisor v1.0 — Regles Incoterms 2020 (CCI Paris)
    <br>Outil interne a usage professionnel
</div>
""", unsafe_allow_html=True)
