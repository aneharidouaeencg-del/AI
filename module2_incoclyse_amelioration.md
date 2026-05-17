# 🚢 Gala Transit — Outil IA : Amélioration du Module 2
## Idée inspirée d'Incoclyse pour la recommandation d'Incoterm

---

## 💡 Idée centrale

Améliorer le **Module 2 (Recommandation d'Incoterm)** en s'inspirant de la plateforme **Incoclyse** (https://incoclyse.com/) — un logiciel Incoterms® IA pour PME exportatrices qui permet à l'utilisateur de :

- Télécharger une facture (extraction automatique des infos via OCR)
- Saisir manuellement les informations de l'expédition
- Obtenir l'Incoterm recommandé avec un **niveau de confiance**
- Lire une **explication claire** du choix
- Consulter un **tableau de répartition des coûts** vendeur / acheteur

---

## 🔄 Ce qui change par rapport à la version actuelle

| Fonctionnalité | Version actuelle | Nouvelle version inspirée Incoclyse |
|---|---|---|
| Mode de saisie | Manuel uniquement | **Manuel + Import de facture (OCR)** |
| Résultat | Incoterm recommandé + justification | Incoterm + **niveau de confiance %** + justification |
| Répartition coûts | Absente | **Tableau vendeur / acheteur** par composante |
| Alertes | Warning texte simple | **Alertes contextuelles automatiques** (FOB + LC, DDP sans capacité import) |
| Alternatives | Liste simple | **Top 4 alternatives** avec score relatif |

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

### Mode 2 — Import de facture (OCR + Vision IA)
L'agent glisse-dépose ou sélectionne une facture (PDF, PNG, JPG).

L'IA extrait automatiquement :
- Nom de l'exportateur et de l'acheteur
- Incoterm mentionné sur la facture (pour comparaison)
- Valeur de la marchandise
- Mode de transport
- Pays de destination

Les champs extraits pré-remplissent automatiquement le formulaire pour validation.

---

## 🤖 Implémentation technique — OCR avec Claude Vision API

```python
import anthropic
import base64
import json
import streamlit as st

# Upload de la facture dans Streamlit
uploaded_file = st.file_uploader(
    "Importer une facture",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    # Conversion en base64
    file_bytes = uploaded_file.read()
    base64_image = base64.b64encode(file_bytes).decode("utf-8")

    # Appel à l'API Claude Vision pour extraction
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": """Analyse cette facture commerciale et extrais les informations suivantes.
                    Réponds UNIQUEMENT en JSON, sans texte avant ou après :
                    {
                      "incoterm": "Incoterm mentionné sur la facture (ex: FOB, CIF, DAP...)",
                      "mode_transport": "Maritime | Aérien | Routier | Ferroviaire",
                      "pays_origine": "pays d'origine",
                      "pays_destination": "pays de destination",
                      "valeur_usd": "valeur en USD (nombre uniquement)",
                      "exportateur": "nom de l'exportateur",
                      "acheteur": "nom de l'acheteur"
                    }
                    Si une information est absente, mets null."""
                }
            ]
        }]
    )

    # Parsing du résultat
    try:
        extracted = json.loads(response.content[0].text)
        st.success("✅ Informations extraites automatiquement")

        # Affichage des champs extraits pour validation
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Exportateur :** {extracted.get('exportateur', 'Non détecté')}")
            st.write(f"**Incoterm facture :** {extracted.get('incoterm', 'Non détecté')}")
            st.write(f"**Mode transport :** {extracted.get('mode_transport', 'Non détecté')}")
        with col2:
            st.write(f"**Acheteur :** {extracted.get('acheteur', 'Non détecté')}")
            st.write(f"**Valeur :** {extracted.get('valeur_usd', 'Non détecté')} USD")
            st.write(f"**Destination :** {extracted.get('pays_destination', 'Non détecté')}")

    except json.JSONDecodeError:
        st.error("Extraction impossible — vérifiez la qualité de l'image")
```

---

## 📊 Moteur de scoring — Niveau de confiance

Le niveau de confiance est calculé à partir du score différentiel entre le meilleur Incoterm et les alternatives :

```python
def get_confidence(best_score: int, all_scores: list[int]) -> int:
    """
    Calcule le niveau de confiance en % basé sur l'écart entre
    le meilleur score et les alternatives.
    Plage : 60% (recommandation peu différenciée) → 96% (choix évident)
    """
    valid_scores = [s for s in all_scores if s > 0]
    if not valid_scores:
        return 60

    max_s = max(valid_scores)
    min_s = min(valid_scores)
    gap = max_s - min_s

    if gap == 0:
        return 60

    confidence = round(60 + ((best_score - min_s) / gap) * 35)
    return min(confidence, 96)  # Plafond à 96% — jamais 100% en logistique
```

---

## 🧮 Tableau de répartition des coûts

Pour chaque Incoterm recommandé, un tableau affiche qui supporte chaque composante de coût :

```python
INCOTERM_COSTS = {
    "FOB": {
        "vendeur": ["Emballage", "Transport intérieur", "Chargement à bord", "Douane export"],
        "acheteur": ["Fret maritime", "Assurance", "Douane import", "Livraison finale"]
    },
    "CIF": {
        "vendeur": ["Emballage", "Transport", "Chargement", "Fret maritime", "Assurance (Cl.C)", "Douane export"],
        "acheteur": ["Déchargement", "Douane import", "Livraison finale"]
    },
    "FCA": {
        "vendeur": ["Emballage", "Transport intérieur", "Douane export"],
        "acheteur": ["Fret principal", "Assurance", "Douane import", "Livraison finale"]
    },
    "DAP": {
        "vendeur": ["Emballage", "Transport complet", "Assurance (recommandée)", "Douane export"],
        "acheteur": ["Déchargement", "Douane import"]
    },
    "DDP": {
        "vendeur": ["Emballage", "Transport complet", "Assurance", "Douane export", "Douane import", "Taxes & TVA"],
        "acheteur": ["Déchargement"]
    },
    # ... (11 Incoterms au total)
}

def display_cost_table(incoterm: str) -> None:
    """Affiche le tableau de répartition des coûts dans Streamlit"""
    costs = INCOTERM_COSTS.get(incoterm, {})
    if not costs:
        return

    import pandas as pd
    rows = []
    for item in costs.get("vendeur", []):
        rows.append({"Composante": item, "À charge de": "🟢 Vendeur"})
    for item in costs.get("acheteur", []):
        rows.append({"Composante": item, "À charge de": "🔵 Acheteur"})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
```

---

## ⚠️ Alertes contextuelles automatiques

```python
def check_alerts(incoterm: str, params: dict) -> str | None:
    """
    Retourne un message d'alerte si la combinaison est risquée.
    Retourne None si pas d'alerte.
    """
    if incoterm == "DDP" and not params.get("seller_import"):
        return ("⚠️ Attention : le DDP exige que le vendeur dispose d'un statut "
                "d'importateur reconnu dans le pays de destination. "
                "Vérifiez la faisabilité juridique avant de valider.")

    if incoterm == "FOB" and params.get("letter_of_credit"):
        return ("⚠️ Le FOB peut poser des difficultés avec un crédit documentaire. "
                "Envisagez FCA (disposition connaissement à bord, Incoterms® 2020).")

    if incoterm == "EXW" and params.get("op_type") == "Export":
        return ("⚠️ Sous EXW, l'acheteur étranger doit gérer les formalités d'export "
                "dans votre pays — cela peut être problématique. Préférez FCA.")

    if incoterm in ["CIF", "CFR"] and params.get("transport") != "Maritime":
        return ("⚠️ CIF/CFR sont réservés au transport maritime. "
                "Pour ce mode de transport, utilisez CIP ou CPT.")

    return None
```

---

## 🖥️ Interface Streamlit — Structure complète du Module 2 amélioré

```python
import streamlit as st
import json
import anthropic
import base64
from module2_incoterm import recommend_incoterm_rules, INCOTERM_COSTS, get_confidence, check_alerts

def show_module2():
    st.markdown("## 📋 Module 2 — Recommandation d'Incoterm")

    # ── Sélection du mode de saisie ──────────────────────────────────────────
    mode = st.radio(
        "Mode de saisie",
        ["✏️ Saisie manuelle", "📄 Importer une facture (OCR)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    extracted_data = {}

    # ── Mode OCR ──────────────────────────────────────────────────────────────
    if "Importer" in mode:
        uploaded = st.file_uploader("Facture", type=["pdf","png","jpg","jpeg"])
        if uploaded:
            extracted_data = extract_from_invoice(uploaded)
            if extracted_data:
                st.success("✅ Informations extraites — vérifiez et complétez si nécessaire")
                with st.expander("📋 Champs extraits", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Exportateur :** {extracted_data.get('exportateur','—')}")
                        st.write(f"**Incoterm facture :** `{extracted_data.get('incoterm','—')}`")
                        st.write(f"**Mode transport :** {extracted_data.get('mode_transport','—')}")
                    with col2:
                        st.write(f"**Acheteur :** {extracted_data.get('acheteur','—')}")
                        st.write(f"**Valeur :** {extracted_data.get('valeur_usd','—')} USD")
                        st.write(f"**Destination :** {extracted_data.get('pays_destination','—')}")

    # ── Formulaire de saisie ──────────────────────────────────────────────────
    st.markdown("---")
    with st.form("incoterm_form"):
        col1, col2 = st.columns(2)
        with col1:
            op_type    = st.selectbox("Type d'opération", ["Export", "Import"])
            transport  = st.selectbox("Mode de transport",
                           ["Maritime","Aérien","Routier","Ferroviaire"],
                           index=["Maritime","Aérien","Routier","Ferroviaire"].index(
                               extracted_data.get("mode_transport","Maritime")
                               if extracted_data.get("mode_transport") in
                               ["Maritime","Aérien","Routier","Ferroviaire"] else "Maritime"))
            cargo      = st.selectbox("Type de marchandise",
                           ["Conteneurisé","Vrac","Fragile / Valeur élevée","Périssable","Standard"])
        with col2:
            buyer_type = st.selectbox("Profil du client", ["Client habituel","Nouveau client"])
            insurance  = st.selectbox("Niveau d'assurance",
                           ["Tous risques (Clause A)","Minimale (Clause C)"])
            value      = st.number_input("Valeur marchandise (USD)",
                           min_value=0, value=int(extracted_data.get("valeur_usd",50000) or 50000))

        col3, col4, col5 = st.columns(3)
        with col3: seller_exp = st.checkbox("Vendeur gère export", value=True)
        with col4: seller_imp = st.checkbox("Vendeur gère import destination")
        with col5: loc        = st.checkbox("Crédit documentaire")

        submitted = st.form_submit_button(
            "🎯 Obtenir la recommandation", type="primary", use_container_width=True)

    # ── Résultats ─────────────────────────────────────────────────────────────
    if submitted:
        params = {
            "op_type": op_type, "transport": transport, "cargo": cargo,
            "buyer_type": buyer_type, "insurance": insurance, "value": value,
            "seller_export": seller_exp, "seller_import": seller_imp,
            "letter_of_credit": loc
        }
        recs = recommend_incoterm_rules(**params)
        if not recs:
            st.error("Aucune recommandation possible — vérifiez les paramètres.")
            return

        best    = recs[0]
        all_scores = [r["score"] for r in recs]
        conf    = get_confidence(best["score"], all_scores)
        alert   = check_alerts(best["incoterm"], params)

        # Carte principale
        col_main, col_conf = st.columns([3, 1])
        with col_main:
            st.markdown(f"""
            <div style="background:#E8F5EE; border:2px solid #1E6B3C; border-radius:10px; padding:1.2rem;">
                <div style="font-size:0.75rem; color:#1E6B3C; font-weight:600; margin-bottom:4px;">
                    ✅ RECOMMANDATION PRINCIPALE
                </div>
                <div style="font-size:2rem; font-weight:700; color:#1B3A6B;">
                    {best['incoterm']} <span style="font-size:1rem; color:#555;">— {best['name']}</span>
                </div>
                <p style="font-size:0.85rem; color:#333; margin:0.5rem 0 0;">{best['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_conf:
            st.metric("Niveau de confiance", f"{conf}%")

        if alert:
            st.warning(alert)

        st.markdown(f"**📌** {best['when_to_use']}")
        st.info(best['warning'])

        # Tableau de répartition des coûts
        st.markdown("### 🧮 Répartition des coûts vendeur / acheteur")
        display_cost_table(best["incoterm"])

        # Alternatives
        st.markdown("### 🔄 Alternatives envisageables")
        alt_cols = st.columns(min(4, len(recs)-1))
        for i, (col, r) in enumerate(zip(alt_cols, recs[1:5])):
            c = get_confidence(r["score"], all_scores)
            with col:
                st.markdown(f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0;
                     border-radius:8px; padding:0.75rem; text-align:center;">
                    <div style="font-size:1.2rem; font-weight:700;">{r['incoterm']}</div>
                    <div style="font-size:0.75rem; color:#555;">{r['name']}</div>
                    <div style="font-size:0.9rem; color:#555; margin-top:4px;">{c}%</div>
                </div>
                """, unsafe_allow_html=True)

        # Alerte si Incoterm facture différent du recommandé
        if extracted_data.get("incoterm"):
            inv_inco = extracted_data["incoterm"].upper().strip()[:3]
            if inv_inco != best["incoterm"]:
                st.warning(
                    f"⚠️ L'Incoterm sur votre facture ({extracted_data['incoterm']}) "
                    f"diffère de la recommandation ({best['incoterm']}). "
                    f"Pensez à vérifier si cet Incoterm est optimal pour cette opération."
                )
```

---

## 📋 Ce qu'il faut mentionner dans le rapport (Chapitre 6)

### Section à ajouter — 6.4.4 Amélioration du Module 2 : import de facture et niveau de confiance

> *"Dans une démarche d'amélioration continue de l'expérience utilisateur, le Module 2 a été enrichi de deux fonctionnalités inspirées de la plateforme Incoclyse (incoclyse.com), outil de référence dans le domaine de la recommandation d'Incoterms par intelligence artificielle.*
>
> *La première fonctionnalité est l'import de factures commerciales avec extraction automatique des informations par OCR. L'agent transit peut soumettre directement une facture au format PDF, PNG ou JPG ; l'outil mobilise l'API Claude Vision d'Anthropic pour identifier et extraire les champs pertinents — Incoterm mentionné, mode de transport, pays de destination, valeur de la marchandise — qui pré-remplissent automatiquement le formulaire de saisie. Cette approche réduit significativement le temps de traitement et élimine les risques d'erreurs de retranscription manuelle.*
>
> *La seconde fonctionnalité est l'affichage d'un niveau de confiance associé à chaque recommandation, calculé à partir de l'écart entre le score du meilleur Incoterm et ceux de ses alternatives. Ce score, exprimé en pourcentage, permet à l'agent d'évaluer d'un coup d'œil la robustesse de la recommandation : un score élevé indique que l'Incoterm recommandé se démarque nettement des alternatives, tandis qu'un score plus modéré signale qu'un examen approfondi des options secondaires est conseillé.*
>
> *À ces deux fonctionnalités s'ajoute un tableau de répartition des coûts vendeur/acheteur, généré automatiquement pour chaque Incoterm recommandé, permettant au client de visualiser immédiatement les implications financières de chaque option."*

---

## 🗄️ Dépendances supplémentaires à ajouter à requirements.txt

```
anthropic>=0.28.0      # API Claude Vision pour OCR des factures
pillow>=10.0.0         # Manipulation d'images
pdf2image>=1.16.0      # Conversion PDF → image pour l'OCR
python-dotenv>=1.0.0   # Gestion de la clé API Anthropic
```

---

## 🔑 Configuration de la clé API

```python
# .env (ne pas committer ce fichier !)
ANTHROPIC_API_KEY=sk-ant-...

# Dans app.py
from dotenv import load_dotenv
import os
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

---

## 📊 Comparaison Incoclyse vs Notre outil

| Fonctionnalité | Incoclyse | Gala Transit Tool |
|---|---|---|
| Import de facture OCR | ✅ | ✅ (Claude Vision API) |
| Saisie manuelle | ✅ | ✅ |
| Niveau de confiance | ✅ | ✅ |
| Répartition coûts vendeur/acheteur | ✅ | ✅ |
| Contextualisé Maroc / ADII | ❌ | ✅ (FOB export, CIF import ADII) |
| Intégré au Module 1 (coût fret) | ❌ | ✅ (simulation complète) |
| Alertes CBAM 2026 | ✅ | ❌ (future évolution) |
| Open source / Personnalisable | ❌ | ✅ (Python complet) |
| Gratuit | Freemium | ✅ (développé en interne) |

---

*Document créé le 16 mai 2026 — Amélioration Module 2 inspirée d'Incoclyse*
*Rapport de stage — Master PSCM M1 — ENCG Settat — Gala Transit Transport*
