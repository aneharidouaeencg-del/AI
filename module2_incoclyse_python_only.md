# 🚢 Gala Transit — Outil IA : Amélioration du Module 2
## Idée inspirée d'Incoclyse pour la recommandation d'Incoterm

---

## 💡 Idée centrale

Améliorer le **Module 2 (Recommandation d'Incoterm)** en s'inspirant de la plateforme **Incoclyse** (https://incoclyse.com/) — un logiciel Incoterms® IA pour PME exportatrices qui permet à l'utilisateur de :

- Télécharger une facture (extraction automatique des infos via OCR **100% Python**)
- Saisir manuellement les informations de l'expédition
- Obtenir l'Incoterm recommandé avec un **niveau de confiance**
- Lire une **explication claire** du choix
- Consulter un **tableau de répartition des coûts** vendeur / acheteur

> ✅ **Choix technique : 100% Python open source — aucune API externe requise**  
> L'extraction OCR est réalisée localement avec **Pytesseract** + **pdf2image** + **Pillow**.  
> Pas de clé API, pas de coût, pas de dépendance à un service tiers.

---

## 🔄 Ce qui change par rapport à la version actuelle

| Fonctionnalité | Version actuelle | Nouvelle version inspirée Incoclyse |
|---|---|---|
| Mode de saisie | Manuel uniquement | **Manuel + Import de facture (OCR Python)** |
| Résultat | Incoterm recommandé + justification | Incoterm + **niveau de confiance %** + justification |
| Répartition coûts | Absente | **Tableau vendeur / acheteur** par composante |
| Alertes | Warning texte simple | **Alertes contextuelles automatiques** (FOB + LC, DDP sans capacité import) |
| Alternatives | Liste simple | **Top 4 alternatives** avec score relatif |
| OCR | Absent | **Pytesseract + pdf2image — 100% local, gratuit** |

---

## 🏗️ Architecture de la solution améliorée

### Mode 1 — Saisie manuelle
L'agent Gala Transit remplit un formulaire avec :
- Type d'opération (Export / Import)
- Mode de transport (Maritime / Aérien / Routier / Ferroviaire)
- Type de marchandise (Conteneurisé / Vrac / Fragile / Périssable / Standard)
- Profil du client (Client habituel / Nouveau client)
- Niveau d'assurance souhaité (Tous risques Clause A / Minimale Clause C)
- Valeur de la marchandise (USD)
- Checkboxes : vendeur gère export, vendeur gère import destination, crédit documentaire

### Mode 2 — Import de facture (OCR 100% Python)
L'agent glisse-dépose ou sélectionne une facture (PDF, PNG, JPG).

**Stack technique locale :**
- `Pytesseract` → moteur OCR open source (wrapper Python de Tesseract)
- `pdf2image` → conversion PDF → image avant OCR
- `Pillow` → prétraitement de l'image (contraste, resize) pour améliorer l'OCR
- `re` (regex) → extraction structurée des champs depuis le texte brut OCR

Les champs extraits pré-remplissent automatiquement le formulaire pour validation.

---

## 🛠️ Installation de Tesseract (prérequis système)

Tesseract doit être installé sur la machine **une seule fois** avant de lancer l'app :

```bash
# Windows
winget install UB-Mannheim.TesseractOCR
# ou télécharger l'installeur : https://github.com/UB-Mannheim/tesseract/wiki

# macOS
brew install tesseract

# Linux / Ubuntu
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-fra  # Pack langue française (optionnel)
```

---

## 📦 Dépendances Python à ajouter à requirements.txt

```
# Remplace anthropic, python-dotenv (plus nécessaires)
pytesseract>=0.3.10    # Wrapper Python pour Tesseract OCR
pdf2image>=1.16.0      # Conversion PDF → image PIL
Pillow>=10.0.0         # Manipulation et prétraitement des images
```

**requirements.txt complet mis à jour :**
```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
pytesseract>=0.3.10
pdf2image>=1.16.0
Pillow>=10.0.0
```

---

## 🤖 Implémentation technique — OCR 100% Python

### Fichier : `models/ocr_invoice.py`

```python
"""
=============================================================================
OCR EXTRACTION — Factures commerciales
Gala Transit Transport | Master PSCM — ENCG Settat
=============================================================================
Stack : Pytesseract + pdf2image + Pillow + regex
100% Python open source — aucune API externe requise
=============================================================================
"""

import re
import json
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_bytes
import io


# ── Chemin Tesseract (Windows uniquement — commenter sur Mac/Linux) ───────────
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── Mots-clés pour chaque Incoterm ───────────────────────────────────────────
INCOTERM_KEYWORDS = {
    "EXW": ["ex works", "exw", "ex-works", "franco usine"],
    "FCA": ["free carrier", "fca", "franco transporteur"],
    "FAS": ["free alongside ship", "fas", "franco le long"],
    "FOB": ["free on board", "fob", "franco à bord", "franco bord"],
    "CFR": ["cost and freight", "cfr", "c&f", "coût et fret"],
    "CIF": ["cost insurance freight", "cif", "coût assurance fret"],
    "CPT": ["carriage paid to", "cpt", "port payé jusqu"],
    "CIP": ["carriage insurance paid", "cip", "port payé assurance"],
    "DAP": ["delivered at place", "dap", "rendu au lieu"],
    "DPU": ["delivered at place unloaded", "dpu", "rendu déchargé"],
    "DDP": ["delivered duty paid", "ddp", "rendu droits acquittés"],
}

# ── Mots-clés pour les modes de transport ────────────────────────────────────
TRANSPORT_KEYWORDS = {
    "Maritime": ["maritime", "sea", "ocean", "vessel", "navire", "ship",
                 "fcl", "lcl", "container", "conteneur", "port", "b/l",
                 "bill of lading", "connaissement"],
    "Aérien":   ["air", "aérien", "aerian", "airway", "awb", "airfreight",
                 "cargo air", "fret aérien", "aéroport", "airport"],
    "Routier":  ["road", "truck", "routier", "camion", "cmr", "lorry",
                 "terrestre", "ground", "route"],
    "Ferroviaire": ["rail", "train", "ferroviaire", "railway", "cim", "fer"],
}


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Améliore la qualité de l'image avant OCR :
    - Conversion en niveaux de gris
    - Augmentation du contraste
    - Légère mise au point
    - Redimensionnement si trop petit
    """
    # Niveaux de gris
    image = image.convert("L")

    # Augmenter le contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Légère mise au point
    image = image.filter(ImageFilter.SHARPEN)

    # Redimensionner si trop petite (min 1000px de large)
    if image.width < 1000:
        ratio = 1000 / image.width
        new_size = (1000, int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    return image


def extract_text_from_file(file_bytes: bytes, file_type: str) -> str:
    """
    Extrait le texte brut d'un fichier image ou PDF via Pytesseract.

    Args:
        file_bytes : contenu du fichier en bytes
        file_type  : "pdf", "png", "jpg" ou "jpeg"

    Returns:
        Texte brut extrait par OCR
    """
    text = ""

    if file_type.lower() == "pdf":
        # Conversion PDF → liste d'images PIL (une par page)
        pages = convert_from_bytes(file_bytes, dpi=200)
        for page in pages[:3]:  # Max 3 premières pages
            processed = preprocess_image(page)
            # OCR avec config optimisée pour factures (texte structuré)
            page_text = pytesseract.image_to_string(
                processed,
                config="--psm 6 --oem 3",  # psm 6 = bloc de texte uniforme
                lang="fra+eng"              # Français + anglais
            )
            text += page_text + "\n"
    else:
        # Image PNG/JPG
        image = Image.open(io.BytesIO(file_bytes))
        processed = preprocess_image(image)
        text = pytesseract.image_to_string(
            processed,
            config="--psm 6 --oem 3",
            lang="fra+eng"
        )

    return text.lower()  # Tout en minuscules pour les comparaisons


def extract_incoterm(text: str) -> str | None:
    """Détecte l'Incoterm mentionné dans le texte de la facture."""
    for code, keywords in INCOTERM_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return code
    return None


def extract_transport_mode(text: str) -> str | None:
    """Détecte le mode de transport mentionné dans le texte."""
    for mode, keywords in TRANSPORT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return mode
    return None


def extract_value(text: str) -> float | None:
    """
    Extrait la valeur monétaire la plus probable de la facture.
    Cherche des patterns comme : USD 45,000 / EUR 38.500 / 50000 USD
    """
    # Patterns : montants avec devises connues
    patterns = [
        r"(?:usd|eur|mad|€|\$)\s*([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)",
        r"([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)\s*(?:usd|eur|mad|€|\$)",
        r"total[^\d]*([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)",
        r"montant[^\d]*([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)",
        r"amount[^\d]*([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)",
    ]
    candidates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            # Nettoyer : supprimer espaces et virgules comme séparateurs de milliers
            clean = re.sub(r"[\s,]", "", m).replace(",", ".")
            try:
                val = float(clean)
                if 100 <= val <= 10_000_000:  # Plage réaliste pour une facture
                    candidates.append(val)
            except ValueError:
                continue

    if candidates:
        # Retourner la valeur la plus élevée (probablement le total)
        return max(candidates)
    return None


def extract_country(text: str, field: str) -> str | None:
    """
    Extrait le pays d'origine ou de destination.
    Cherche des patterns comme : "from: Morocco" / "destination: France"
    """
    origin_patterns = [
        r"(?:from|origine|origin|expéditeur|shipper)[:\s]+([a-zA-Z\s]{3,30})",
        r"(?:country of origin|pays d.origine)[:\s]+([a-zA-Z\s]{3,30})",
    ]
    dest_patterns = [
        r"(?:to|destination|destinataire|consignee)[:\s]+([a-zA-Z\s]{3,30})",
        r"(?:country of destination|pays de destination)[:\s]+([a-zA-Z\s]{3,30})",
        r"(?:deliver(?:y|ed) to)[:\s]+([a-zA-Z\s]{3,30})",
    ]

    patterns = origin_patterns if field == "origin" else dest_patterns
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            country = match.group(1).strip().title()
            # Filtrer les résultats trop courts ou qui ressemblent à des mots courants
            if len(country) >= 4 and country.lower() not in [
                "the", "and", "for", "with", "this", "from", "that"
            ]:
                return country
    return None


def extract_company_name(text: str, field: str) -> str | None:
    """
    Extrait le nom de l'exportateur ou de l'acheteur.
    Cherche les patterns courants dans les factures commerciales.
    """
    if field == "seller":
        patterns = [
            r"(?:sold by|vendeur|seller|shipper|expéditeur)[:\s]+([A-Za-z\s&.-]{3,50})",
            r"(?:from|de la société)[:\s]+([A-Za-z\s&.-]{3,50})",
        ]
    else:
        patterns = [
            r"(?:sold to|acheteur|buyer|consignee|destinataire)[:\s]+([A-Za-z\s&.-]{3,50})",
            r"(?:bill to|facturer à|client)[:\s]+([A-Za-z\s&.-]{3,50})",
        ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip().title()
            if len(name) >= 3:
                return name
    return None


def extract_invoice_data(file_bytes: bytes, file_type: str) -> dict:
    """
    Fonction principale d'extraction OCR.
    Retourne un dictionnaire avec tous les champs extraits.

    Args:
        file_bytes : contenu du fichier uploadé
        file_type  : extension du fichier (pdf, png, jpg, jpeg)

    Returns:
        dict avec les clés : incoterm, mode_transport, valeur_usd,
                             pays_origine, pays_destination,
                             exportateur, acheteur, texte_brut
    """
    try:
        raw_text = extract_text_from_file(file_bytes, file_type)
    except Exception as e:
        return {"erreur": f"Lecture du fichier impossible : {str(e)}"}

    if not raw_text.strip():
        return {"erreur": "Aucun texte détecté — vérifiez la qualité de l'image."}

    return {
        "incoterm":        extract_incoterm(raw_text),
        "mode_transport":  extract_transport_mode(raw_text),
        "valeur_usd":      extract_value(raw_text),
        "pays_origine":    extract_country(raw_text, "origin"),
        "pays_destination": extract_country(raw_text, "destination"),
        "exportateur":     extract_company_name(raw_text, "seller"),
        "acheteur":        extract_company_name(raw_text, "buyer"),
        "texte_brut":      raw_text[:500],  # Aperçu pour débogage
    }
```

---

## 📊 Moteur de scoring — Niveau de confiance

```python
def get_confidence(best_score: int, all_scores: list[int]) -> int:
    """
    Calcule le niveau de confiance en % basé sur l'écart entre
    le meilleur score et les alternatives.
    Plage : 60% (recommandation peu différenciée) → 96% (choix évident)
    Plafond à 96% : en logistique, la certitude absolue n'existe jamais.
    """
    valid_scores = [s for s in all_scores if s > 0]
    if not valid_scores:
        return 60

    max_s = max(valid_scores)
    min_s = min(valid_scores)
    gap   = max_s - min_s

    if gap == 0:
        return 60

    confidence = round(60 + ((best_score - min_s) / gap) * 35)
    return min(confidence, 96)
```

---

## 🧮 Tableau de répartition des coûts (11 Incoterms complets)

```python
INCOTERM_COSTS = {
    "EXW": {
        "vendeur": ["Emballage"],
        "acheteur": ["Chargement usine", "Transport intérieur",
                     "Douane export", "Fret principal", "Assurance",
                     "Douane import", "Livraison finale"]
    },
    "FCA": {
        "vendeur": ["Emballage", "Transport intérieur", "Douane export"],
        "acheteur": ["Fret principal", "Assurance",
                     "Douane import", "Livraison finale"]
    },
    "FAS": {
        "vendeur": ["Emballage", "Transport jusqu'au quai", "Douane export"],
        "acheteur": ["Chargement à bord", "Fret maritime",
                     "Assurance", "Douane import", "Livraison finale"]
    },
    "FOB": {
        "vendeur": ["Emballage", "Transport jusqu'au port",
                    "Chargement à bord", "Douane export"],
        "acheteur": ["Fret maritime", "Assurance",
                     "Douane import", "Livraison finale"]
    },
    "CFR": {
        "vendeur": ["Emballage", "Transport jusqu'au port",
                    "Chargement", "Fret maritime", "Douane export"],
        "acheteur": ["Assurance", "Déchargement",
                     "Douane import", "Livraison finale"]
    },
    "CIF": {
        "vendeur": ["Emballage", "Transport jusqu'au port",
                    "Chargement", "Fret maritime",
                    "Assurance (Clause C)", "Douane export"],
        "acheteur": ["Déchargement", "Douane import", "Livraison finale"]
    },
    "CPT": {
        "vendeur": ["Emballage", "Transport intérieur",
                    "Fret principal", "Douane export"],
        "acheteur": ["Assurance", "Douane import", "Livraison finale"]
    },
    "CIP": {
        "vendeur": ["Emballage", "Transport intérieur",
                    "Fret principal", "Assurance (Clause A)", "Douane export"],
        "acheteur": ["Douane import", "Livraison finale"]
    },
    "DAP": {
        "vendeur": ["Emballage", "Transport complet",
                    "Assurance (recommandée)", "Douane export"],
        "acheteur": ["Déchargement", "Douane import"]
    },
    "DPU": {
        "vendeur": ["Emballage", "Transport complet",
                    "Déchargement", "Assurance (recommandée)", "Douane export"],
        "acheteur": ["Douane import"]
    },
    "DDP": {
        "vendeur": ["Emballage", "Transport complet", "Assurance",
                    "Douane export", "Douane import", "Taxes & TVA"],
        "acheteur": ["Déchargement"]
    },
}


def display_cost_table(incoterm: str) -> None:
    """Affiche le tableau de répartition des coûts dans Streamlit."""
    import pandas as pd
    import streamlit as st

    costs = INCOTERM_COSTS.get(incoterm, {})
    if not costs:
        st.info("Tableau de coûts non disponible pour cet Incoterm.")
        return

    rows = []
    for item in costs.get("vendeur", []):
        rows.append({"Composante de coût": item, "À charge de": "🟢 Vendeur"})
    for item in costs.get("acheteur", []):
        rows.append({"Composante de coût": item, "À charge de": "🔵 Acheteur"})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
```

---

## ⚠️ Alertes contextuelles automatiques

```python
def check_alerts(incoterm: str, params: dict) -> str | None:
    """
    Retourne un message d'alerte si la combinaison est risquée.
    Retourne None si aucune alerte à signaler.
    """
    if incoterm == "DDP" and not params.get("seller_import"):
        return ("⚠️ Le DDP exige que le vendeur dispose d'un statut d'importateur "
                "reconnu dans le pays de destination. Vérifiez la faisabilité "
                "juridique avant de valider cet Incoterm.")

    if incoterm == "FOB" and params.get("letter_of_credit"):
        return ("⚠️ Le FOB peut poser des difficultés avec un crédit documentaire. "
                "Envisagez FCA (disposition connaissement à bord, Incoterms® 2020).")

    if incoterm == "EXW" and params.get("op_type") == "Export":
        return ("⚠️ Sous EXW, l'acheteur étranger doit gérer les formalités "
                "d'export dans votre pays — cela peut être problématique. "
                "Préférez FCA qui offre plus de sécurité.")

    if incoterm in ["CIF", "CFR"] and params.get("transport") != "Maritime":
        return ("⚠️ CIF et CFR sont réservés au transport maritime et fluvial. "
                "Pour ce mode de transport, utilisez CIP ou CPT.")

    if incoterm == "CIF" and params.get("insurance") == "Tous risques (Clause A)":
        return ("ℹ️ Le CIF impose seulement une assurance Clause C (couverture minimale). "
                "Pour une couverture tous risques, préférez CIP (Clause A obligatoire "
                "depuis Incoterms® 2020).")

    if incoterm == "DDP" and params.get("op_type") == "Import":
        return ("ℹ️ En DDP, le vendeur paie la TVA à l'import dans votre pays, "
                "mais ne peut généralement pas la récupérer. "
                "Assurez-vous que ce coût est répercuté dans le prix de vente.")

    return None
```

---

## 🖥️ Interface Streamlit — Module 2 complet (100% Python)

```python
"""
=============================================================================
MODULE 2 — INTERFACE STREAMLIT COMPLÈTE
Gala Transit Transport | Master PSCM — ENCG Settat
=============================================================================
OCR : Pytesseract + pdf2image + Pillow (100% Python, aucune API externe)
=============================================================================
"""

import streamlit as st
import pandas as pd
import json
from models.ocr_invoice import extract_invoice_data
from models.module2_incoterm import (
    recommend_incoterm_rules,
    get_confidence,
    check_alerts,
    display_cost_table,
)


def show_module2():
    st.markdown("## 📋 Module 2 — Recommandation d'Incoterm")
    st.markdown("""
    <div style="background:#D6E4F0; border-left:4px solid #2E6DA4;
         padding:0.8rem 1rem; border-radius:4px; margin-bottom:1rem; font-size:0.88rem;">
    🏛️ Ce module applique un moteur de règles multicritères basé sur les
    <b>Incoterms® 2020</b> de la Chambre de Commerce Internationale,
    adapté au contexte opérationnel de <b>Gala Transit Transport</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Mode de saisie ────────────────────────────────────────────────────────
    mode = st.radio(
        "Mode de saisie",
        ["✏️ Saisie manuelle", "📄 Importer une facture (OCR)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    extracted_data = {}

    # ── Mode OCR ──────────────────────────────────────────────────────────────
    if "Importer" in mode:
        st.markdown("### 📄 Import de facture")
        uploaded = st.file_uploader(
            "Sélectionnez ou glissez-déposez votre facture",
            type=["pdf", "png", "jpg", "jpeg"],
            help="La facture est analysée localement — aucune donnée n'est envoyée à un service externe."
        )

        if uploaded:
            file_bytes = uploaded.read()
            file_type  = uploaded.name.split(".")[-1]

            with st.spinner("🔍 Extraction OCR en cours..."):
                extracted_data = extract_invoice_data(file_bytes, file_type)

            if "erreur" in extracted_data:
                st.error(f"❌ {extracted_data['erreur']}")
                st.info("💡 Conseil : assurez-vous que la facture est lisible et non chiffrée.")
                extracted_data = {}
            else:
                st.success("✅ Extraction réussie — vérifiez et complétez les champs si nécessaire")

                with st.expander("📋 Champs extraits automatiquement", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Exportateur :** {extracted_data.get('exportateur') or '—'}")
                        st.write(f"**Incoterm sur facture :** `{extracted_data.get('incoterm') or 'Non détecté'}`")
                        st.write(f"**Mode transport :** {extracted_data.get('mode_transport') or '—'}")
                    with col2:
                        st.write(f"**Acheteur :** {extracted_data.get('acheteur') or '—'}")
                        val = extracted_data.get('valeur_usd')
                        st.write(f"**Valeur :** {f'{val:,.0f} USD' if val else '—'}")
                        st.write(f"**Pays destination :** {extracted_data.get('pays_destination') or '—'}")

                with st.expander("🔍 Texte brut OCR (débogage)", expanded=False):
                    st.code(extracted_data.get("texte_brut", "—"), language=None)

        st.markdown("---")
        st.markdown("**Complétez ou corrigez les informations ci-dessous :**")

    # ── Formulaire de saisie ──────────────────────────────────────────────────
    # Pré-remplissage avec les données OCR si disponibles
    transport_options = ["Maritime", "Aérien", "Routier", "Ferroviaire"]
    default_transport = extracted_data.get("mode_transport", "Maritime")
    if default_transport not in transport_options:
        default_transport = "Maritime"

    default_value = int(extracted_data.get("valeur_usd") or 50000)

    st.markdown("### ✏️ Paramètres de la transaction")
    with st.form("incoterm_form"):
        col1, col2 = st.columns(2)
        with col1:
            op_type    = st.selectbox("Type d'opération *",
                                      ["Export", "Import"])
            transport  = st.selectbox("Mode de transport *",
                                      transport_options,
                                      index=transport_options.index(default_transport))
            cargo      = st.selectbox("Type de marchandise *",
                                      ["Conteneurisé", "Vrac",
                                       "Fragile / Valeur élevée",
                                       "Périssable", "Standard"])
        with col2:
            buyer_type = st.selectbox("Profil du client *",
                                      ["Client habituel", "Nouveau client"])
            insurance  = st.selectbox("Niveau d'assurance *",
                                      ["Tous risques (Clause A)",
                                       "Minimale (Clause C)"])
            value      = st.number_input("Valeur marchandise (USD) *",
                                         min_value=0,
                                         value=default_value,
                                         step=1000)

        st.markdown("**Capacités douanières**")
        col3, col4, col5 = st.columns(3)
        with col3:
            seller_exp = st.checkbox("✅ Vendeur gère export", value=True)
        with col4:
            seller_imp = st.checkbox("✅ Vendeur gère import destination")
        with col5:
            loc = st.checkbox("💳 Paiement par crédit documentaire")

        submitted = st.form_submit_button(
            "🎯 Obtenir la recommandation",
            type="primary",
            use_container_width=True
        )

    # ── Résultats ─────────────────────────────────────────────────────────────
    if submitted:
        params = {
            "transport_mode":             transport,
            "cargo_type":                 cargo,
            "seller_export_capability":   seller_exp,
            "seller_import_capability":   seller_imp,
            "buyer_type":                 buyer_type,
            "insurance_needed":           insurance,
            "letter_of_credit":           loc,
            "operation_type":             op_type,
        }

        recs = recommend_incoterm_rules(**params)
        if not recs:
            st.error("Aucune recommandation possible — vérifiez les paramètres.")
            return

        best       = recs[0]
        all_scores = [r["score"] for r in recs]
        conf       = get_confidence(best["score"], all_scores)
        alert      = check_alerts(best["incoterm"], {
            "seller_import": seller_imp,
            "letter_of_credit": loc,
            "op_type": op_type,
            "transport": transport,
            "insurance": insurance,
        })

        st.markdown("---")
        st.markdown("### 📊 Résultats")

        # Carte principale
        col_main, col_conf = st.columns([3, 1])
        with col_main:
            st.markdown(f"""
            <div style="background:#E8F5EE; border:2px solid #1E6B3C;
                 border-radius:10px; padding:1.2rem;">
                <div style="font-size:0.75rem; color:#1E6B3C; font-weight:600;
                     margin-bottom:4px; text-transform:uppercase; letter-spacing:0.5px;">
                    ✅ Recommandation principale
                </div>
                <div style="font-size:2rem; font-weight:700; color:#1B3A6B;">
                    {best['incoterm']}
                    <span style="font-size:1rem; font-weight:400; color:#555;">
                        — {best['name']}
                    </span>
                </div>
                <div style="font-size:0.88rem; color:#333; margin-top:0.5rem;">
                    {best['description']}
                </div>
                <div style="font-size:0.85rem; color:#1E6B3C; margin-top:0.4rem;">
                    📌 {best['when_to_use']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_conf:
            st.metric(
                label="Niveau de confiance",
                value=f"{conf}%",
                help="Calculé à partir de l'écart entre le score du meilleur Incoterm et ses alternatives."
            )
            # Couleur indicative
            if conf >= 80:
                st.success("Confiance élevée")
            elif conf >= 65:
                st.warning("Confiance modérée")
            else:
                st.info("Plusieurs options proches")

        # Alerte contextuelle
        if alert:
            st.warning(alert)

        st.info(best['warning'])

        # Alerte Incoterm facture ≠ Incoterm recommandé
        if extracted_data.get("incoterm"):
            inv_inco = str(extracted_data["incoterm"]).upper().strip()[:3]
            if inv_inco != best["incoterm"]:
                st.warning(
                    f"⚠️ L'Incoterm sur votre facture (**{extracted_data['incoterm']}**) "
                    f"diffère de la recommandation (**{best['incoterm']}**). "
                    f"Vérifiez si l'Incoterm actuel est optimal pour cette opération."
                )

        # Tableau de répartition des coûts
        st.markdown("### 🧮 Répartition des coûts vendeur / acheteur")
        display_cost_table(best["incoterm"])

        # Top 4 alternatives
        st.markdown("### 🔄 Alternatives envisageables")
        if len(recs) > 1:
            alt_cols = st.columns(min(4, len(recs) - 1))
            for col, r in zip(alt_cols, recs[1:5]):
                c = get_confidence(r["score"], all_scores)
                with col:
                    st.markdown(f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0;
                         border-radius:8px; padding:0.75rem; text-align:center;">
                        <div style="font-size:1.3rem; font-weight:700;
                             color:#1B3A6B;">{r['incoterm']}</div>
                        <div style="font-size:0.75rem; color:#555;">{r['name']}</div>
                        <div style="font-size:0.85rem; color:#888;
                             margin-top:4px;">Confiance : {c}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Aucune alternative significative pour ce profil d'opération.")

        # Tableau de référence complet
        with st.expander("📖 Référence rapide — Les 11 Incoterms® 2020"):
            from models.module2_incoterm import INCOTERMS_INFO
            df_ref = pd.DataFrame([{
                "Code": code,
                "Nom complet": info["name"],
                "Transport": "Maritime" if info["transport"] == "maritime" else "Tous modes",
                "Risque vendeur": info["seller_risk"].capitalize(),
            } for code, info in INCOTERMS_INFO.items()])
            st.dataframe(df_ref, use_container_width=True, hide_index=True)
```

---

## 🔧 Test de l'extraction OCR (script autonome)

```python
# test_ocr.py — à lancer depuis le terminal pour tester l'OCR

from models.ocr_invoice import extract_invoice_data

# Test avec un fichier image
with open("data/facture_test.jpg", "rb") as f:
    result = extract_invoice_data(f.read(), "jpg")

print("=== Résultats extraction OCR ===")
for key, value in result.items():
    if key != "texte_brut":
        print(f"{key:20s} : {value}")

print("\n=== Texte brut (aperçu) ===")
print(result.get("texte_brut", "—"))
```

---

## 📋 Ce qu'il faut mentionner dans le rapport (Chapitre 6)

### Section à ajouter — 6.4.4 Amélioration du Module 2 : import de facture et niveau de confiance

> *"Dans une démarche d'amélioration continue de l'expérience utilisateur, le Module 2 a été enrichi de deux fonctionnalités inspirées de la plateforme Incoclyse (incoclyse.com), outil de référence dans le domaine de la recommandation d'Incoterms par intelligence artificielle.*
>
> *La première fonctionnalité est l'import de factures commerciales avec extraction automatique des informations par OCR. L'agent transit peut soumettre directement une facture au format PDF, PNG ou JPG. L'extraction est réalisée entièrement en local grâce à la bibliothèque Pytesseract — moteur OCR open source développé par Google — combinée à pdf2image pour la conversion des fichiers PDF et à Pillow pour le prétraitement des images (amélioration du contraste, mise au point, redimensionnement). Une série d'expressions régulières (regex) identifie ensuite les champs pertinents dans le texte brut extrait : Incoterm mentionné, mode de transport, pays de destination, valeur de la marchandise, noms des parties. Ce choix technique 100% Python présente l'avantage d'être gratuit, de ne nécessiter aucune clé API, et de garantir que les données confidentielles des factures ne quittent jamais le système de l'entreprise.*
>
> *La seconde fonctionnalité est l'affichage d'un niveau de confiance associé à chaque recommandation, calculé à partir de l'écart entre le score du meilleur Incoterm et ceux de ses alternatives. Ce score, exprimé en pourcentage et plafonné à 96% — la certitude absolue n'existant pas en logistique internationale —, permet à l'agent d'évaluer d'un coup d'œil la robustesse de la recommandation.*
>
> *À ces deux fonctionnalités s'ajoutent un tableau de répartition des coûts vendeur/acheteur généré pour chacun des 11 Incoterms, et un système d'alertes contextuelles automatiques signalant les combinaisons à risque — comme l'utilisation de FOB avec un crédit documentaire, ou le recours au DDP sans statut d'importateur à destination."*

---

## 📊 Comparaison Incoclyse vs Notre outil (version mise à jour)

| Fonctionnalité | Incoclyse | Gala Transit Tool |
|---|---|---|
| Import de facture OCR | ✅ | ✅ **Pytesseract — 100% local et gratuit** |
| Saisie manuelle | ✅ | ✅ |
| Niveau de confiance | ✅ | ✅ |
| Répartition coûts vendeur/acheteur | ✅ | ✅ (11 Incoterms complets) |
| Contextualisé Maroc / ADII | ❌ | ✅ (FOB export, CIF import ADII) |
| Intégré au Module 1 (coût fret) | ❌ | ✅ (simulation complète) |
| Alertes contextuelles | Partiel | ✅ (6 règles d'alerte métier) |
| Alertes CBAM 2026 | ✅ | ❌ (future évolution) |
| Dépendance API externe | ✅ (requise) | ✅ **Aucune — 100% Python** |
| Open source / Personnalisable | ❌ | ✅ (Python complet) |
| Gratuit | Freemium | ✅ (développé en interne) |
| Données facture envoyées en ligne | ✅ | ✅ **Non — traitement 100% local** |

---

## 🗂️ Fichiers modifiés / créés

```
gala_transit_tool/
├── models/
│   ├── ocr_invoice.py          ← NOUVEAU : extraction OCR 100% Python
│   ├── module2_incoterm.py     ← MODIFIÉ : + get_confidence(), check_alerts(),
│   │                                        display_cost_table(), INCOTERM_COSTS complet
│   └── saved/
│       └── incoterm_model.pkl  ← Inchangé
│
├── app.py                      ← MODIFIÉ : show_module2() enrichi avec OCR
├── requirements.txt            ← MODIFIÉ : + pytesseract, pdf2image, Pillow
│                                           - anthropic, python-dotenv (supprimés)
└── test_ocr.py                 ← NOUVEAU : script de test OCR autonome
```

---

*Document mis à jour le 17 mai 2026 — Remplacement Claude Vision API par Pytesseract (100% Python)*
*Rapport de stage — Master PSCM M1 — ENCG Settat — Gala Transit Transport*
