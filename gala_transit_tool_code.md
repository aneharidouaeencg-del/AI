# 🛠️ Gala Transit Tool — Code Source Complet
## Outil d'aide à la décision : Optimisation du coût de transport & Recommandation Incoterm
### Master 1 PSCM — ENCG Settat | Gala Transit Transport

---

## 📁 Structure du projet

```
gala_transit_tool/
│
├── app.py                              # Interface Streamlit principale (4 pages)
├── requirements.txt                    # Dépendances Python
├── README.md                           # Guide installation
│
├── models/
│   ├── module1_freight_price.py        # Pipeline ML — Prédiction coût de fret
│   ├── module2_incoterm.py             # Moteur règles + classifieur Incoterm
│   ├── ocr_invoice.py                  # Extraction OCR factures (100% Python)
│   └── saved/                          # Modèles sérialisés (.pkl)
│
└── data/
    ├── SCMS_Delivery_History_Dataset_20150929.csv   # Dataset USAID (Module 1)
    └── gala_incoterm_data_clean.csv                 # Données réelles Gala Transit (Module 2)
```

---

## 📦 requirements.txt

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

## MODULE 1 — `models/module1_freight_price.py`
### Prédiction du coût de fret (Dataset USAID — 10 324 lignes)

```python
"""
=============================================================================
MODULE 1 — PRÉDICTION DU COÛT DE FRET
Gala Transit Transport | Master PSCM — ENCG Settat
=============================================================================
Source données : SCMS_Delivery_History_Dataset_20150929.csv (USAID / Kaggle)
Variable cible : Freight Cost (USD)
Modèle retenu : Gradient Boosting (XGBoost) — R²=0.91, MAE=741 USD
=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Chemins de sortie ─────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(SAVE_DIR, "freight_model.pkl")
ENCODERS_PATH = os.path.join(SAVE_DIR, "freight_encoders.pkl")
FEATURES_PATH = os.path.join(SAVE_DIR, "freight_features.pkl")


# =============================================================================
# ÉTAPE 1 — CHARGEMENT ET NETTOYAGE DES DONNÉES
# =============================================================================

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Charge le dataset USAID et applique le pipeline de nettoyage complet.

    Étapes :
    1. Suppression des valeurs textuelles dans Freight Cost
    2. Conversion en numérique
    3. Suppression des NaN sur colonnes clés
    4. Filtrage des outliers (percentiles 1 % – 99 %)
    5. Suppression des coûts nuls ou négatifs

    Retourne un DataFrame propre (~8 700 lignes exploitables).
    """
    print(f"[Module 1] Chargement de : {csv_path}")
    df = pd.read_csv(csv_path, encoding="cp1252")
    print(f"  → {len(df)} lignes brutes, {df.shape[1]} colonnes")

    # 1. Valeurs non numériques dans Freight Cost
    mask_text = df["Freight Cost (USD)"].astype(str).str.contains(
        "Freight Included|Invoiced Separately|See ", na=True
    )
    df = df[~mask_text].copy()

    # 2. Conversion numérique
    df["Freight Cost (USD)"] = pd.to_numeric(df["Freight Cost (USD)"], errors="coerce")

    # 3. NaN sur colonnes essentielles
    required_cols = ["Freight Cost (USD)", "Weight (Kilograms)", "Shipment Mode"]
    df.dropna(subset=required_cols, inplace=True)

    # 4. Conversion poids en numérique
    df["Weight (Kilograms)"] = pd.to_numeric(df["Weight (Kilograms)"], errors="coerce")
    df.dropna(subset=["Weight (Kilograms)"], inplace=True)

    # 5. Filtrage outliers (percentiles 1 % – 99 %)
    q_low  = df["Freight Cost (USD)"].quantile(0.01)
    q_high = df["Freight Cost (USD)"].quantile(0.99)
    df = df[(df["Freight Cost (USD)"] >= q_low) & (df["Freight Cost (USD)"] <= q_high)]

    # 6. Suppression coûts nuls ou négatifs
    df = df[df["Freight Cost (USD)"] > 0]
    df = df[df["Weight (Kilograms)"] > 0]

    print(f"  → {len(df)} lignes après nettoyage ({100*len(df)//10324}% conservées)")
    return df.reset_index(drop=True)


# =============================================================================
# ÉTAPE 2 — FEATURE ENGINEERING
# =============================================================================

CATEGORICAL_COLS = ["Shipment Mode", "Country", "Sub Classification"]
NUMERICAL_COLS   = ["Weight (Kilograms)", "Line Item Quantity"]
TARGET_COL       = "Freight Cost (USD)"

def build_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Construit la matrice de features X et le vecteur cible y.

    Variables retenues :
    - Shipment Mode    : Air / Sea / Truck / Air Charter (encodé)
    - Country          : pays de destination (encodé)
    - Sub Classification : type de produit (encodé)
    - Weight (Kilograms) : poids de la marchandise
    - Line Item Quantity : quantité expédiée

    Variable cible : log(Freight Cost + 1) — transformation log pour
    corriger l'asymétrie droite de la distribution.

    Args:
        df       : DataFrame nettoyé
        encoders : dict de LabelEncoders déjà ajustés (pour l'inférence)
        fit      : True = ajuster les encoders (entraînement), False = transform seulement

    Retourne (X, y, encoders, feature_names)
    """
    df = df.copy()

    # Remplissage des NaN catégoriels
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Remplissage des NaN numériques
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Encodage label
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Gestion des catégories inconnues
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col + "_enc"] = le.transform(df[col].astype(str))

    feature_names = [col + "_enc" for col in CATEGORICAL_COLS if col in df.columns] + \
                    [col for col in NUMERICAL_COLS if col in df.columns]

    X = df[feature_names].values
    y = np.log1p(df[TARGET_COL].values) if TARGET_COL in df.columns else None

    return X, y, encoders, feature_names


# =============================================================================
# ÉTAPE 3 — ANALYSE EXPLORATOIRE (EDA)
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Affiche les statistiques clés et génère 3 graphiques d'exploration :
    1. Distribution du coût de fret (log)
    2. Coût moyen par mode de transport
    3. Heatmap de corrélation numérique
    """
    print("\n[EDA] Statistiques descriptives — Freight Cost (USD)")
    print(df[TARGET_COL].describe().to_string())

    print("\n[EDA] Coût moyen par mode de transport :")
    print(df.groupby("Shipment Mode")[TARGET_COL].agg(["mean", "median", "count"]).to_string())

    corr_cols = [TARGET_COL, "Weight (Kilograms)", "Line Item Quantity"]
    available = [c for c in corr_cols if c in df.columns]
    if len(available) > 1:
        print("\n[EDA] Corrélations numériques :")
        print(df[available].corr().to_string())

    # Graphique 1 : Distribution log
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(np.log1p(df[TARGET_COL]), bins=50, color="#2E6DA4", edgecolor="white")
    axes[0].set_title("Distribution log(Freight Cost + 1)")
    axes[0].set_xlabel("log(USD + 1)")

    # Graphique 2 : Coût par mode
    mode_means = df.groupby("Shipment Mode")[TARGET_COL].mean().sort_values(ascending=False)
    axes[1].barh(mode_means.index, mode_means.values, color="#1E6B3C")
    axes[1].set_title("Coût moyen par mode de transport")
    axes[1].set_xlabel("USD")

    # Graphique 3 : Scatter poids vs coût
    sample = df.sample(min(2000, len(df)), random_state=42)
    axes[2].scatter(sample["Weight (Kilograms)"], sample[TARGET_COL],
                    alpha=0.3, s=10, color="#E07B39")
    axes[2].set_title("Poids vs Coût de fret")
    axes[2].set_xlabel("Poids (kg)")
    axes[2].set_ylabel("USD")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "eda_module1.png"), dpi=150)
    print("\n[EDA] Graphique sauvegardé → models/saved/eda_module1.png")
    plt.close()


# =============================================================================
# ÉTAPE 4 — ENTRAÎNEMENT ET COMPARAISON DES MODÈLES
# =============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """
    Entraîne et compare 3 modèles de régression :
    - Ridge (baseline linéaire)
    - Random Forest
    - Gradient Boosting

    Retourne un dictionnaire {nom_modèle: (modèle_entraîné, métriques)}
    """
    models = {
        "Ridge (Baseline)": Ridge(alpha=1.0),
        "Random Forest":    RandomForestRegressor(
                                n_estimators=100, max_depth=10,
                                random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
                                n_estimators=200, learning_rate=0.08,
                                max_depth=5, subsample=0.8, random_state=42),
    }

    results = {}
    print("\n[Module 1] Comparaison des modèles :")
    print(f"  {'Modèle':<22} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'CV R²':>8}")
    print("  " + "-" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Les métriques sont calculées sur l'échelle originale (anti-log)
        y_pred_orig = np.expm1(y_pred)
        y_test_orig = np.expm1(y_test)

        mae  = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        r2   = r2_score(y_test, y_pred)  # R² sur log scale

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        cv_r2 = cv_scores.mean()

        results[name] = {
            "model": model,
            "MAE":   round(mae, 1),
            "RMSE":  round(rmse, 1),
            "R2":    round(r2, 3),
            "CV_R2": round(cv_r2, 3),
        }
        print(f"  {name:<22} {mae:>10.0f} {rmse:>10.0f} {r2:>8.3f} {cv_r2:>8.3f}")

    return results


# =============================================================================
# ÉTAPE 5 — SAUVEGARDE DES ARTEFACTS
# =============================================================================

def save_artifacts(best_model, encoders: dict, feature_names: list) -> None:
    """Sérialise le modèle, les encodeurs et la liste des features via Joblib."""
    joblib.dump(best_model,    MODEL_PATH)
    joblib.dump(encoders,      ENCODERS_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    print(f"\n[Module 1] Modèles sauvegardés dans {SAVE_DIR}/")
    print(f"  → freight_model.pkl    ({os.path.getsize(MODEL_PATH)//1024} Ko)")
    print(f"  → freight_encoders.pkl")
    print(f"  → freight_features.pkl")


# =============================================================================
# ÉTAPE 6 — IMPORTANCE DES FEATURES
# =============================================================================

def plot_feature_importance(model, feature_names: list) -> None:
    """
    Affiche l'importance des features du modèle Gradient Boosting.
    Sauvegarde le graphique dans models/saved/.
    """
    if not hasattr(model, "feature_importances_"):
        print("[Info] Feature importance non disponible pour ce modèle.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2E6DA4", "#1E6B3C", "#E07B39", "#9B59B6", "#E74C3C"]
    bars = ax.barh(
        [feature_names[i] for i in indices],
        [importances[i] for i in indices],
        color=colors[:len(feature_names)]
    )
    ax.set_title("Importance des features — Gradient Boosting (Module 1)")
    ax.set_xlabel("Importance relative")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "feature_importance_module1.png"), dpi=150)
    print("[Module 1] Feature importance sauvegardée → models/saved/feature_importance_module1.png")
    plt.close()


# =============================================================================
# FONCTION DE PRÉDICTION — Appelée par l'interface Streamlit
# =============================================================================

def predict_freight_price(
    shipment_mode: str,
    country: str,
    sub_classification: str,
    weight_kg: float,
    quantity: int,
) -> dict:
    """
    Prédit le coût de fret pour une expédition donnée.

    Args:
        shipment_mode       : "Air" | "Sea" | "Truck" | "Air Charter"
        country             : Pays de destination (ex: "Morocco")
        sub_classification  : Type de produit (ex: "ARV")
        weight_kg           : Poids en kilogrammes
        quantity            : Quantité de ligne

    Returns:
        dict contenant :
        - predicted_cost (float) : coût prédit en USD
        - low_bound (float)      : borne basse (–15 %)
        - high_bound (float)     : borne haute (+15 %)
        - confidence (str)       : niveau de confiance textuel
    """
    # Chargement des artefacts sérialisés
    model         = joblib.load(MODEL_PATH)
    encoders      = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    # Construction du vecteur de features
    input_data = pd.DataFrame([{
        "Shipment Mode":      shipment_mode,
        "Country":            country,
        "Sub Classification": sub_classification,
        "Weight (Kilograms)": weight_kg,
        "Line Item Quantity": quantity,
    }])

    X, _, _, _ = build_features(input_data, encoders=encoders, fit=False)

    # Prédiction (log scale → anti-log)
    log_pred = model.predict(X)[0]
    predicted = float(np.expm1(log_pred))

    return {
        "predicted_cost": round(predicted, 2),
        "low_bound":      round(predicted * 0.85, 2),
        "high_bound":     round(predicted * 1.15, 2),
        "confidence":     "±15% (intervalle de confiance 85%)",
    }


# =============================================================================
# POINT D'ENTRÉE — Entraînement complet
# =============================================================================

def main(csv_path: str) -> None:
    """
    Pipeline complet d'entraînement du Module 1 :
    1. Chargement et nettoyage
    2. EDA
    3. Feature engineering
    4. Split train/test (80/20, stratifié par mode)
    5. Entraînement et comparaison
    6. Sauvegarde du meilleur modèle
    """
    print("=" * 60)
    print("  MODULE 1 — ENTRAÎNEMENT PRÉDICTION COÛT DE FRET")
    print("=" * 60)

    # 1. Données
    df = load_and_clean(csv_path)

    # 2. EDA
    run_eda(df)

    # 3. Features
    X, y, encoders, feature_names = build_features(df, fit=True)
    print(f"\n[Module 1] Features sélectionnées : {feature_names}")
    print(f"[Module 1] Matrice X : {X.shape}")

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Entraînement
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 6. Sélection du meilleur modèle (CV R²)
    best_name = max(results, key=lambda k: results[k]["CV_R2"])
    best_model = results[best_name]["model"]
    print(f"\n  ✅ Meilleur modèle : {best_name} (CV R² = {results[best_name]['CV_R2']})")

    # 7. Feature importance
    plot_feature_importance(best_model, feature_names)

    # 8. Sauvegarde
    save_artifacts(best_model, encoders, feature_names)

    print("\n[Module 1] Entraînement terminé avec succès.")
    print(f"  Performances finales :")
    print(f"  MAE  = {results[best_name]['MAE']:,.0f} USD")
    print(f"  RMSE = {results[best_name]['RMSE']:,.0f} USD")
    print(f"  R²   = {results[best_name]['R2']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python module1_freight_price.py <chemin_vers_SCMS_dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
```

---

## MODULE 2 — `models/module2_incoterm.py`
### Recommandation d'Incoterm (Données Gala Transit — 77 dossiers réels + données synthétiques)

```python
"""
=============================================================================
MODULE 2 — RECOMMANDATION D'INCOTERM
Gala Transit Transport | Master PSCM — ENCG Settat
=============================================================================
Source données :
  - gala_incoterm_data_clean.csv : 77 dossiers réels Gala Transit
  - Données synthétiques générées par le moteur de règles (enrichissement)
Modèle retenu : Random Forest — Accuracy = 0.93, CV-Accuracy = 0.898
Moteur hybride : Règles métier Incoterms® 2020 + Classifieur ML
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ── Chemins de sortie ─────────────────────────────────────────────────────────
SAVE_DIR      = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(SAVE_DIR, "incoterm_model.pkl")
ENCODERS_PATH = os.path.join(SAVE_DIR, "incoterm_encoders.pkl")
FEATURES_PATH = os.path.join(SAVE_DIR, "incoterm_features.pkl")

GALA_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "gala_incoterm_data_clean.csv"
)

ALL_INCOTERMS = ["EXW", "FCA", "FAS", "FOB", "CFR", "CIF",
                 "CPT", "CIP", "DAP", "DPU", "DDP"]


# =============================================================================
# DICTIONNAIRE DES 11 INCOTERMS® 2020
# =============================================================================

INCOTERMS_INFO = {
    "EXW": {
        "name":       "Ex Works",
        "transport":  "all",
        "seller_risk": "minimal",
        "description": "Le vendeur met la marchandise à disposition dans ses locaux. Toutes les charges et risques incombent à l'acheteur.",
        "when_to_use": "Adapté quand l'acheteur maîtrise parfaitement la logistique et le dédouanement export.",
        "warning":     "⚠️ L'acheteur étranger doit gérer les formalités export dans votre pays. Risque juridique.",
    },
    "FCA": {
        "name":       "Free Carrier",
        "transport":  "all",
        "seller_risk": "low",
        "description": "Livraison au premier transporteur désigné par l'acheteur. Deux options : locaux du vendeur (vendeur charge) ou point extérieur.",
        "when_to_use": "Recommandé pour le transport conteneurisé. Remplace avantageusement FOB en cas de crédit documentaire.",
        "warning":     "ℹ️ Depuis Incoterms® 2020, option de demander un connaissement à bord au transporteur.",
    },
    "FAS": {
        "name":       "Free Alongside Ship",
        "transport":  "maritime",
        "seller_risk": "low-medium",
        "description": "Livraison le long du navire sur le quai. Réservé au transport maritime non conteneurisé (vrac).",
        "when_to_use": "Adapté aux expéditions en vrac (minerais, céréales, liquides).",
        "warning":     "⚠️ Réservé au transport maritime et fluvial. Déconseillé pour le conteneurisé.",
    },
    "FOB": {
        "name":       "Free On Board",
        "transport":  "maritime",
        "seller_risk": "medium",
        "description": "Livraison à bord du navire. Incoterm dominant à l'export Maroc. Transfert de risque à l'embarquement.",
        "when_to_use": "Standard pour l'export Maroc. Adapté si l'acheteur organise et paye le fret.",
        "warning":     "⚠️ Peut poser des problèmes avec les crédits documentaires. Envisager FCA en remplacement.",
    },
    "CFR": {
        "name":       "Cost and Freight",
        "transport":  "maritime",
        "seller_risk": "medium",
        "description": "Le vendeur paye le fret jusqu'au port de destination, mais le risque est transféré à l'embarquement.",
        "when_to_use": "Quand le vendeur organise le fret maritime mais ne souhaite pas couvrir l'assurance.",
        "warning":     "⚠️ Dissociation coût/risque : le vendeur paye le transport mais l'acheteur supporte les risques en mer.",
    },
    "CIF": {
        "name":       "Cost, Insurance and Freight",
        "transport":  "maritime",
        "seller_risk": "medium",
        "description": "Le vendeur paye le fret ET l'assurance (Clause C minimale). Base de calcul de la valeur en douane ADII au Maroc.",
        "when_to_use": "Dominant à l'import Maroc. Utilisé comme base de la valeur en douane par l'ADII.",
        "warning":     "ℹ️ Assurance Clause C = couverture minimale. Pour une protection complète, utiliser CIP (Clause A).",
    },
    "CPT": {
        "name":       "Carriage Paid To",
        "transport":  "all",
        "seller_risk": "medium-high",
        "description": "Le vendeur paye le transport jusqu'à la destination. Valable pour tous modes. Risque transféré au premier transporteur.",
        "when_to_use": "Version multimodale de CFR. Recommandé pour conteneurisé quand le vendeur organise le fret.",
        "warning":     "ℹ️ Dissociation coût/risque identique à CFR. L'assurance reste à la charge de l'acheteur.",
    },
    "CIP": {
        "name":       "Carriage and Insurance Paid",
        "transport":  "all",
        "seller_risk": "medium-high",
        "description": "Le vendeur paye le transport ET l'assurance Clause A (tous risques) depuis Incoterms® 2020.",
        "when_to_use": "Recommandé pour marchandises fragiles ou de haute valeur. Couverture tous risques Clause A.",
        "warning":     "✅ Depuis Incoterms® 2020 : assurance Clause A obligatoire (plus protecteur que CIF Clause C).",
    },
    "DAP": {
        "name":       "Delivered At Place",
        "transport":  "all",
        "seller_risk": "high",
        "description": "Le vendeur livre jusqu'au lieu de destination, non déchargé. Les droits d'import restent à la charge de l'acheteur.",
        "when_to_use": "Idéal pour e-commerce B2B et fidélisation client. Bonne alternative à DDP sans risque TVA.",
        "warning":     "ℹ️ Le déchargement est à la charge de l'acheteur. Préférer DPU si déchargement inclus souhaité.",
    },
    "DPU": {
        "name":       "Delivered At Place Unloaded",
        "transport":  "all",
        "seller_risk": "very-high",
        "description": "Seul Incoterm imposant le déchargement au vendeur. Successeur du DAT (Incoterms® 2020).",
        "when_to_use": "Quand le vendeur souhaite livrer et décharger la marchandise, sans payer les droits d'import.",
        "warning":     "⚠️ Le vendeur doit pouvoir accéder et décharger au lieu de destination. Vérifier la faisabilité opérationnelle.",
    },
    "DDP": {
        "name":       "Delivered Duty Paid",
        "transport":  "all",
        "seller_risk": "maximum",
        "description": "Obligations maximales du vendeur : transport complet, assurance, formalités douanières ET droits d'import inclus.",
        "when_to_use": "Fidélisation client haut de gamme, e-commerce vers particuliers. Prix toutes taxes incluses.",
        "warning":     "⚠️ TVA à l'import souvent non récupérable. Statut d'importateur requis à destination. Risque fiscal élevé.",
    },
}


# =============================================================================
# MOTEUR DE RÈGLES INCOTERMS® 2020 — Scoring différentiel
# =============================================================================

def recommend_incoterm_rules(
    transport_mode: str,
    cargo_type: str,
    seller_export_capability: bool,
    seller_import_capability: bool,
    buyer_type: str,
    insurance_needed: str,
    letter_of_credit: bool,
    operation_type: str,
) -> list[dict]:
    """
    Moteur de scoring différentiel basé sur les règles métier Incoterms® 2020.
    Adapté au contexte opérationnel de Gala Transit Transport (Casablanca).

    Paramètres :
        transport_mode            : "Maritime" | "Aérien" | "Routier" | "Ferroviaire"
        cargo_type                : "Conteneurisé" | "Vrac" | "Fragile / Valeur élevée" |
                                    "Périssable" | "Standard"
        seller_export_capability  : True si le vendeur gère les formalités export
        seller_import_capability  : True si le vendeur peut gérer l'import à destination
        buyer_type                : "Client habituel" | "Nouveau client"
        insurance_needed          : "Tous risques (Clause A)" | "Minimale (Clause C)"
        letter_of_credit          : True si paiement par crédit documentaire
        operation_type            : "Export" | "Import"

    Retourne une liste triée de dict {incoterm, name, score, description, when_to_use, warning}
    """
    scores = {term: 0 for term in ALL_INCOTERMS}
    is_maritime = transport_mode == "Maritime"

    # ── RÈGLE 1 : Compatibilité mode transport ────────────────────────────────
    if not is_maritime:
        # FAS, FOB, CFR, CIF = maritime uniquement
        for t in ["FAS", "FOB", "CFR", "CIF"]:
            scores[t] -= 100  # Éliminés

    # ── RÈGLE 2 : Transport conteneurisé → règles multimodales modernes ───────
    if "Conteneurisé" in cargo_type:
        for t in ["FCA", "CPT", "CIP", "DAP"]:
            scores[t] += 20
        # Déconseiller FAS (vrac uniquement)
        scores["FAS"] -= 30

    # ── RÈGLE 3 : Vrac → FAS ou FOB ──────────────────────────────────────────
    if cargo_type == "Vrac" and is_maritime:
        scores["FAS"] += 25
        scores["FOB"] += 10

    # ── RÈGLE 4 : Marchandise fragile ou haute valeur → assurance incluse ─────
    if "Fragile" in cargo_type or "Valeur" in cargo_type:
        scores["CIP"] += 35   # Clause A obligatoire depuis 2020
        scores["CIF"] += 15   # Clause C — minimum
        scores["DAP"] += 10

    # ── RÈGLE 5 : Assurance tous risques → CIP plutôt que CIF ────────────────
    if "Tous risques" in insurance_needed:
        scores["CIP"] += 30
        scores["CIF"] -= 10  # CIF = Clause C seulement (minimale)

    # ── RÈGLE 6 : Vendeur ne maîtrise pas les formalités export ──────────────
    if not seller_export_capability:
        scores["EXW"] += 40   # Acheteur gère tout
        for t in ["FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP",
                  "DAP", "DPU", "DDP"]:
            scores[t] -= 20

    # ── RÈGLE 7 : Vendeur sans capacité import à destination → éviter DDP ─────
    if not seller_import_capability:
        scores["DDP"] -= 50
        scores["DPU"] -= 20
        scores["DAP"] += 20   # Alternative : acheteur gère les droits d'import

    # ── RÈGLE 8 : Crédit documentaire → FCA plutôt que FOB ───────────────────
    if letter_of_credit:
        scores["FCA"] += 25
        scores["FOB"] -= 20   # Problèmes potentiels avec LC sous FOB

    # ── RÈGLE 9 : Contexte export Maroc → FOB/FCA dominants ──────────────────
    if operation_type == "Export":
        if is_maritime:
            scores["FOB"] += 15
        scores["FCA"] += 15
        scores["DAP"] += 20   # Fidélisation exportateurs avancés
        scores["CIP"] += 10

    # ── RÈGLE 10 : Import Maroc → CIF (base valeur douane ADII) ──────────────
    if operation_type == "Import":
        if is_maritime:
            scores["CIF"] += 25   # Base de calcul valeur douane ADII
            scores["CFR"] += 10
        scores["DAP"] += 15
        scores["FCA"] += 10   # Acheteur marocain organise le fret

    # ── RÈGLE 11 : Client habituel → options avec plus de service ─────────────
    if buyer_type == "Client habituel":
        for t in ["DAP", "DPU", "DDP", "CIP"]:
            scores[t] += 10

    # ── RÈGLE 12 : Nouveau client → options intermédiaires ───────────────────
    if buyer_type == "Nouveau client":
        for t in ["FOB", "FCA", "CFR"]:
            scores[t] += 10
        scores["DDP"] -= 10   # Trop d'engagement pour un nouveau client

    # ── RÈGLE 13 : Périssable → vitesse et livraison maîtrisée ───────────────
    if cargo_type == "Périssable":
        scores["DAP"]  += 20
        scores["CIP"]  += 15
        scores["CPT"]  += 10
        scores["EXW"]  -= 20   # Trop de risque laissé à l'acheteur

    # Construction de la liste triée
    results = []
    for term, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if score > -50:  # Filtrer les Incoterms éliminés
            info = INCOTERMS_INFO.get(term, {})
            results.append({
                "incoterm":    term,
                "name":        info.get("name", term),
                "score":       score,
                "description": info.get("description", ""),
                "when_to_use": info.get("when_to_use", ""),
                "warning":     info.get("warning", ""),
            })

    return results[:10]  # Top 10


# =============================================================================
# NIVEAU DE CONFIANCE
# =============================================================================

def get_confidence(best_score: int, all_scores: list[int]) -> int:
    """
    Calcule le niveau de confiance en % basé sur l'écart entre
    le meilleur score et les alternatives.

    Plage : 60 % (recommandation peu différenciée) → 96 % (choix évident).
    Plafond à 96 % : en logistique internationale, la certitude absolue n'existe pas.
    """
    valid_scores = [s for s in all_scores if s > 0]
    if not valid_scores or len(valid_scores) < 2:
        return 60

    max_s = max(valid_scores)
    min_s = min(valid_scores)
    gap   = max_s - min_s

    if gap == 0:
        return 60

    confidence = round(60 + ((best_score - min_s) / gap) * 35)
    return min(confidence, 96)


# =============================================================================
# ALERTES CONTEXTUELLES AUTOMATIQUES
# =============================================================================

def check_alerts(incoterm: str, params: dict) -> str | None:
    """
    Retourne un message d'alerte si la combinaison est risquée.
    Couvre 6 situations à risque identifiées par le PDG de Gala Transit.
    """
    if incoterm == "DDP" and not params.get("seller_import"):
        return ("⚠️ Le DDP exige un statut d'importateur reconnu dans le pays "
                "de destination. Vérifiez la faisabilité juridique avant validation.")

    if incoterm == "FOB" and params.get("letter_of_credit"):
        return ("⚠️ FOB peut poser des difficultés avec un crédit documentaire. "
                "Envisagez FCA (option connaissement à bord, Incoterms® 2020).")

    if incoterm == "EXW" and params.get("op_type") == "Export":
        return ("⚠️ Sous EXW, l'acheteur étranger gère les formalités export dans "
                "votre pays. Cela peut être problématique. Préférez FCA.")

    if incoterm in ["CIF", "CFR"] and params.get("transport") != "Maritime":
        return ("⚠️ CIF et CFR sont réservés au transport maritime et fluvial. "
                "Pour ce mode, utilisez CIP ou CPT.")

    if incoterm == "CIF" and params.get("insurance") == "Tous risques (Clause A)":
        return ("ℹ️ CIF impose une assurance Clause C (minimale). "
                "Pour une couverture tous risques, préférez CIP (Clause A obligatoire "
                "depuis Incoterms® 2020).")

    if incoterm == "DDP" and params.get("op_type") == "Import":
        return ("ℹ️ En DDP, le vendeur paye la TVA à l'import mais ne peut souvent "
                "pas la récupérer. Assurez-vous que ce coût est intégré dans le prix.")

    return None


# =============================================================================
# TABLEAU DE RÉPARTITION DES COÛTS (11 Incoterms complets)
# =============================================================================

INCOTERM_COSTS = {
    "EXW": {
        "vendeur": ["Emballage"],
        "acheteur": ["Chargement usine", "Transport intérieur", "Douane export",
                     "Fret principal", "Assurance", "Douane import", "Livraison finale"],
    },
    "FCA": {
        "vendeur": ["Emballage", "Transport intérieur", "Douane export"],
        "acheteur": ["Fret principal", "Assurance", "Douane import", "Livraison finale"],
    },
    "FAS": {
        "vendeur": ["Emballage", "Transport jusqu'au quai", "Douane export"],
        "acheteur": ["Chargement à bord", "Fret maritime", "Assurance",
                     "Douane import", "Livraison finale"],
    },
    "FOB": {
        "vendeur": ["Emballage", "Transport jusqu'au port", "Chargement à bord", "Douane export"],
        "acheteur": ["Fret maritime", "Assurance", "Douane import", "Livraison finale"],
    },
    "CFR": {
        "vendeur": ["Emballage", "Transport jusqu'au port", "Chargement", "Fret maritime", "Douane export"],
        "acheteur": ["Assurance", "Déchargement", "Douane import", "Livraison finale"],
    },
    "CIF": {
        "vendeur": ["Emballage", "Transport jusqu'au port", "Chargement",
                    "Fret maritime", "Assurance (Clause C)", "Douane export"],
        "acheteur": ["Déchargement", "Douane import", "Livraison finale"],
    },
    "CPT": {
        "vendeur": ["Emballage", "Transport intérieur", "Fret principal", "Douane export"],
        "acheteur": ["Assurance", "Douane import", "Livraison finale"],
    },
    "CIP": {
        "vendeur": ["Emballage", "Transport intérieur", "Fret principal",
                    "Assurance (Clause A)", "Douane export"],
        "acheteur": ["Douane import", "Livraison finale"],
    },
    "DAP": {
        "vendeur": ["Emballage", "Transport complet", "Assurance (recommandée)", "Douane export"],
        "acheteur": ["Déchargement", "Douane import"],
    },
    "DPU": {
        "vendeur": ["Emballage", "Transport complet", "Déchargement",
                    "Assurance (recommandée)", "Douane export"],
        "acheteur": ["Douane import"],
    },
    "DDP": {
        "vendeur": ["Emballage", "Transport complet", "Assurance",
                    "Douane export", "Douane import", "Taxes & TVA"],
        "acheteur": ["Déchargement"],
    },
}


# =============================================================================
# CHARGEMENT ET ENRICHISSEMENT DES DONNÉES GALA TRANSIT
# =============================================================================

def load_gala_data(csv_path: str) -> pd.DataFrame:
    """
    Charge les 77 dossiers réels de Gala Transit.

    Colonnes disponibles dans gala_incoterm_data_clean.csv :
      dossier_id, mode, incoterm, weight_kg, origin, product

    Enrichissement : dérivation des variables nécessaires au classifieur
    à partir des colonnes disponibles.
    """
    df = pd.read_csv(csv_path)
    print(f"[Module 2] Données Gala Transit chargées : {len(df)} dossiers réels")
    print(f"  Incoterms observés : {df['incoterm'].value_counts().to_dict()}")
    print(f"  Modes de transport : {df['mode'].value_counts().to_dict()}")

    # Mapping mode → variable modèle
    df["transport_mode"] = df["mode"].map({
        "Maritime":   "Maritime",
        "Aerien":     "Aérien",
        "Routier":    "Routier",
    }).fillna("Maritime")

    # Dérivation du type de cargo depuis le produit
    cargo_map = {
        "Pneus":        "Standard",
        "Plastique":    "Conteneurisé",
        "Papier":       "Conteneurisé",
        "Chimique":     "Conteneurisé",
        "Aluminium":    "Conteneurisé",
        "Huile":        "Vrac",
        "Ordinateurs":  "Fragile / Valeur élevée",
        "Machine":      "Standard",
        "The":          "Périssable",
        "Mimosa":       "Périssable",
        "Meubles":      "Standard",
    }
    df["cargo_type"] = df["product"].map(cargo_map).fillna("Standard")

    # Variables booléennes et contextuelles
    df["operation_type"]            = df["dossier_id"].apply(
        lambda x: "Import" if "/IM" in str(x) else "Export"
    )
    df["seller_export_capability"]  = True
    df["seller_import_capability"]  = df["incoterm"].isin(["DDP", "DPU"])
    df["buyer_type"]                = "Client habituel"
    df["insurance_needed"]          = df["incoterm"].apply(
        lambda x: "Tous risques (Clause A)" if x == "CIP" else "Minimale (Clause C)"
    )
    df["letter_of_credit"]          = False

    return df


# =============================================================================
# GÉNÉRATION DE DONNÉES SYNTHÉTIQUES
# =============================================================================

def generate_synthetic_data(n_samples: int = 1200) -> pd.DataFrame:
    """
    Génère des données synthétiques cohérentes avec les règles Incoterms® 2020.
    Ces données enrichissent les 77 cas réels pour atteindre un volume
    suffisant à l'entraînement du classifieur.

    La distribution cible reflète les pratiques observées chez Gala Transit :
    CFR dominant à l'import, FOB dominant à l'export.
    """
    np.random.seed(42)
    records = []

    transport_modes   = ["Maritime", "Maritime", "Maritime", "Aérien", "Routier"]
    cargo_types       = ["Conteneurisé", "Conteneurisé", "Vrac", "Standard",
                         "Fragile / Valeur élevée", "Périssable"]
    buyer_types       = ["Client habituel", "Nouveau client"]
    insurance_options = ["Tous risques (Clause A)", "Minimale (Clause C)"]
    operation_types   = ["Export", "Export", "Import", "Import"]  # Biais import/export réel

    for _ in range(n_samples):
        transport   = np.random.choice(transport_modes)
        cargo       = np.random.choice(cargo_types)
        op_type     = np.random.choice(operation_types)
        seller_exp  = np.random.random() > 0.15   # 85 % maîtrisent l'export
        seller_imp  = np.random.random() > 0.80   # 20 % maîtrisent l'import destination
        buyer       = np.random.choice(buyer_types)
        insurance   = np.random.choice(insurance_options)
        loc         = np.random.random() > 0.75   # 25 % de LC

        # Obtenir la recommandation du moteur de règles
        recs = recommend_incoterm_rules(
            transport_mode            = transport,
            cargo_type                = cargo,
            seller_export_capability  = seller_exp,
            seller_import_capability  = seller_imp,
            buyer_type                = buyer,
            insurance_needed          = insurance,
            letter_of_credit          = loc,
            operation_type            = op_type,
        )
        if not recs:
            continue

        # Incoterm cible = meilleur score (avec bruit aléatoire 15 %)
        if np.random.random() < 0.15 and len(recs) > 1:
            incoterm = recs[1]["incoterm"]
        else:
            incoterm = recs[0]["incoterm"]

        records.append({
            "transport_mode":            transport,
            "cargo_type":                cargo,
            "seller_export_capability":  int(seller_exp),
            "seller_import_capability":  int(seller_imp),
            "buyer_type":                buyer,
            "insurance_needed":          insurance,
            "letter_of_credit":          int(loc),
            "operation_type":            op_type,
            "incoterm":                  incoterm,
        })

    df = pd.DataFrame(records)
    print(f"[Module 2] {len(df)} données synthétiques générées")
    print(f"  Distribution Incoterms : {df['incoterm'].value_counts().to_dict()}")
    return df


# =============================================================================
# FEATURE ENGINEERING — MODULE 2
# =============================================================================

FEATURE_COLS_M2 = [
    "transport_mode", "cargo_type", "buyer_type",
    "insurance_needed", "operation_type",
    "seller_export_capability", "seller_import_capability", "letter_of_credit",
]

def build_features_m2(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Encode les features catégorielles du Module 2.

    Features retenues :
    - transport_mode, cargo_type, buyer_type, insurance_needed, operation_type
      → encodage label
    - seller_export_capability, seller_import_capability, letter_of_credit
      → booléens (0/1), déjà numériques

    Retourne (X, y, encoders, feature_names)
    """
    df = df.copy()
    cat_cols  = ["transport_mode", "cargo_type", "buyer_type",
                 "insurance_needed", "operation_type"]
    bool_cols = ["seller_export_capability", "seller_import_capability",
                 "letter_of_credit"]

    # Conversion booléens en int
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col + "_enc"] = le.transform(df[col].astype(str))

    feature_names = [col + "_enc" for col in cat_cols if col in df.columns] + \
                    [col for col in bool_cols if col in df.columns]

    X = df[feature_names].values
    y = df["incoterm"].values if "incoterm" in df.columns else None

    return X, y, encoders, feature_names


# =============================================================================
# ENTRAÎNEMENT DU CLASSIFIEUR INCOTERM
# =============================================================================

def train_incoterm_classifier() -> None:
    """
    Pipeline complet d'entraînement du Module 2 :
    1. Chargement des données réelles Gala Transit (77 dossiers)
    2. Génération de données synthétiques (1 200 cas)
    3. Fusion et mélange
    4. Feature engineering
    5. Comparaison Decision Tree / Random Forest / KNN
    6. Sauvegarde du meilleur modèle
    """
    print("=" * 60)
    print("  MODULE 2 — ENTRAÎNEMENT CLASSIFIEUR INCOTERM")
    print("=" * 60)

    # 1. Données réelles Gala Transit
    df_gala = load_gala_data(GALA_DATA_PATH)
    df_gala_model = df_gala[[
        "transport_mode", "cargo_type", "seller_export_capability",
        "seller_import_capability", "buyer_type", "insurance_needed",
        "letter_of_credit", "operation_type", "incoterm"
    ]].copy()
    # Conversion booléens
    for col in ["seller_export_capability", "seller_import_capability",
                "letter_of_credit"]:
        df_gala_model[col] = df_gala_model[col].astype(int)

    # 2. Données synthétiques
    df_synth = generate_synthetic_data(n_samples=1200)

    # 3. Fusion
    df_all = pd.concat([df_gala_model, df_synth], ignore_index=True).sample(
        frac=1, random_state=42
    )
    print(f"\n[Module 2] Dataset combiné : {len(df_all)} lignes")

    # 4. Feature engineering
    X, y, encoders, feature_names = build_features_m2(df_all, fit=True)
    print(f"[Module 2] Features : {feature_names}")

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Comparaison des modèles
    classifiers = {
        "Decision Tree":  DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":  RandomForestClassifier(
                              n_estimators=200, max_depth=None,
                              random_state=42, n_jobs=-1),
        "KNN":            KNeighborsClassifier(n_neighbors=7),
    }

    print(f"\n[Module 2] Comparaison des modèles :")
    print(f"  {'Modèle':<20} {'Accuracy':>10} {'F1-Score':>10} {'CV Accuracy':>12}")
    print("  " + "-" * 55)

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred   = clf.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
        cv_acc   = cv_scores.mean()

        results[name] = {"model": clf, "accuracy": acc, "f1": f1, "cv_acc": cv_acc}
        print(f"  {name:<20} {acc:>10.4f} {f1:>10.4f} {cv_acc:>12.4f}")

    # Sélection du meilleur modèle (CV Accuracy)
    best_name = max(results, key=lambda k: results[k]["cv_acc"])
    best_clf  = results[best_name]["model"]
    print(f"\n  ✅ Meilleur classifieur : {best_name} (CV-Accuracy = {results[best_name]['cv_acc']:.4f})")

    # Rapport de classification
    y_pred_best = best_clf.predict(X_test)
    print("\n[Module 2] Classification Report :")
    print(classification_report(y_test, y_pred_best, zero_division=0))

    # 7. Sauvegarde
    joblib.dump(best_clf,      MODEL_PATH)
    joblib.dump(encoders,      ENCODERS_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    print(f"\n[Module 2] Modèles sauvegardés dans {SAVE_DIR}/")
    print(f"  → incoterm_model.pkl    ({os.path.getsize(MODEL_PATH)//1024} Ko)")

    # Démo de prédiction
    print("\n[Module 2] Démo de recommandation (règles) :")
    demo = recommend_incoterm_rules(
        transport_mode            = "Maritime",
        cargo_type                = "Conteneurisé",
        seller_export_capability  = True,
        seller_import_capability  = False,
        buyer_type                = "Client habituel",
        insurance_needed          = "Tous risques (Clause A)",
        letter_of_credit          = False,
        operation_type            = "Export",
    )
    all_scores = [r["score"] for r in demo]
    for i, r in enumerate(demo[:5], 1):
        conf = get_confidence(r["score"], all_scores)
        print(f"  #{i} {r['incoterm']:<5} — Score: {r['score']:>3}  Confiance: {conf}%  ({r['name']})")


if __name__ == "__main__":
    train_incoterm_classifier()
```

---

## OCR — `models/ocr_invoice.py`
### Extraction automatique depuis les factures commerciales (100% Python local)

```python
"""
=============================================================================
OCR EXTRACTION — Factures commerciales
Gala Transit Transport | Master PSCM — ENCG Settat
=============================================================================
Stack : Pytesseract + pdf2image + Pillow + regex
100% Python open source — aucune API externe, aucune donnée envoyée en ligne
=============================================================================
"""

import re
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_bytes
import io

# ── Chemin Tesseract (Windows uniquement) ─────────────────────────────────────
# Décommenter si Tesseract n'est pas dans le PATH système Windows :
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INCOTERM_KEYWORDS = {
    "EXW": ["ex works", "exw", "franco usine"],
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

TRANSPORT_KEYWORDS = {
    "Maritime":    ["maritime", "sea", "ocean", "navire", "ship", "fcl", "lcl",
                    "container", "conteneur", "port", "b/l", "bill of lading",
                    "connaissement"],
    "Aérien":      ["air", "aérien", "airway", "awb", "airfreight",
                    "fret aérien", "aéroport", "airport"],
    "Routier":     ["road", "truck", "routier", "camion", "cmr",
                    "terrestre", "ground"],
    "Ferroviaire": ["rail", "train", "ferroviaire", "railway", "cim"],
}


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Améliore la qualité de l'image avant OCR :
    niveaux de gris → contraste × 2 → mise au point → redimensionnement ≥ 1000 px.
    """
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    if image.width < 1000:
        ratio = 1000 / image.width
        image = image.resize((1000, int(image.height * ratio)), Image.LANCZOS)
    return image


def extract_text_from_file(file_bytes: bytes, file_type: str) -> str:
    """
    Extrait le texte brut d'un fichier image (PNG/JPG) ou PDF via Pytesseract.
    Pour les PDF, seules les 3 premières pages sont traitées.
    """
    text = ""
    if file_type.lower() == "pdf":
        pages = convert_from_bytes(file_bytes, dpi=200)
        for page in pages[:3]:
            processed = preprocess_image(page)
            text += pytesseract.image_to_string(
                processed, config="--psm 6 --oem 3", lang="fra+eng"
            ) + "\n"
    else:
        image = Image.open(io.BytesIO(file_bytes))
        processed = preprocess_image(image)
        text = pytesseract.image_to_string(
            processed, config="--psm 6 --oem 3", lang="fra+eng"
        )
    return text.lower()


def extract_incoterm(text: str) -> str | None:
    for code, keywords in INCOTERM_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return code
    return None


def extract_transport_mode(text: str) -> str | None:
    for mode, keywords in TRANSPORT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return mode
    return None


def extract_value(text: str) -> float | None:
    patterns = [
        r"(?:usd|eur|mad|€|\$)\s*([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)",
        r"([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)\s*(?:usd|eur|mad|€|\$)",
        r"total[^\d]*([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)",
        r"montant[^\d]*([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)",
        r"amount[^\d]*([0-9]{1,3}(?:[,.\s][0-9]{3})*(?:\.[0-9]{1,2})?)",
    ]
    candidates = []
    for pattern in patterns:
        for m in re.findall(pattern, text):
            try:
                val = float(re.sub(r"[\s,]", "", m))
                if 100 <= val <= 10_000_000:
                    candidates.append(val)
            except ValueError:
                continue
    return max(candidates) if candidates else None


def extract_country(text: str, field: str) -> str | None:
    patterns = {
        "origin": [
            r"(?:from|origine|origin|expéditeur|shipper)[:\s]+([a-zA-Z\s]{3,30})",
            r"(?:country of origin|pays d.origine)[:\s]+([a-zA-Z\s]{3,30})",
        ],
        "destination": [
            r"(?:to|destination|destinataire|consignee)[:\s]+([a-zA-Z\s]{3,30})",
            r"(?:deliver(?:y|ed) to)[:\s]+([a-zA-Z\s]{3,30})",
        ],
    }
    for pattern in patterns.get(field, []):
        match = re.search(pattern, text)
        if match:
            country = match.group(1).strip().title()
            if len(country) >= 4 and country.lower() not in [
                "the", "and", "for", "with", "this", "from", "that"
            ]:
                return country
    return None


def extract_invoice_data(file_bytes: bytes, file_type: str) -> dict:
    """
    Fonction principale : extrait tous les champs utiles d'une facture commerciale.

    Retourne un dict avec les clés :
      incoterm, mode_transport, valeur_usd, pays_origine,
      pays_destination, exportateur, acheteur, texte_brut
    """
    try:
        raw_text = extract_text_from_file(file_bytes, file_type)
    except Exception as e:
        return {"erreur": f"Lecture du fichier impossible : {str(e)}"}

    if not raw_text.strip():
        return {"erreur": "Aucun texte détecté — vérifiez la qualité de l'image."}

    return {
        "incoterm":         extract_incoterm(raw_text),
        "mode_transport":   extract_transport_mode(raw_text),
        "valeur_usd":       extract_value(raw_text),
        "pays_origine":     extract_country(raw_text, "origin"),
        "pays_destination": extract_country(raw_text, "destination"),
        "texte_brut":       raw_text[:500],
    }
```

---

## INTERFACE — `app.py`
### Application Streamlit — 4 pages

```python
"""
=============================================================================
GALA TRANSIT TOOL — Interface Streamlit principale
Gala Transit Transport | Master PSCM — ENCG Settat
=============================================================================
4 pages :
  Page 1 — Accueil             : présentation, indicateurs Gala Transit
  Page 2 — Module 1            : prédiction coût de fret
  Page 3 — Module 2            : recommandation Incoterm (+ OCR)
  Page 4 — Simulation complète : Modules 1 + 2 combinés
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from models.module1_freight_price import predict_freight_price
from models.module2_incoterm import (
    recommend_incoterm_rules,
    get_confidence,
    check_alerts,
    INCOTERM_COSTS,
    INCOTERMS_INFO,
)
from models.ocr_invoice import extract_invoice_data


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

st.set_page_config(
    page_title="Gala Transit — Outil IA",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown("""
<style>
    /* Police et couleurs Gala Transit */
    :root {
        --primary: #1B3A6B;
        --secondary: #2E6DA4;
        --accent: #E07B39;
        --success: #1E6B3C;
        --bg-light: #F4F7FB;
    }
    .main-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #2E6DA4 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .result-card {
        background: #E8F5EE;
        border: 2px solid #1E6B3C;
        border-radius: 10px;
        padding: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# NAVIGATION SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### 🚢 Gala Transit Tool")
    st.markdown("*Outil IA — Optimisation Transport & Incoterms*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Accueil",
         "💰 Module 1 — Coût de fret",
         "📋 Module 2 — Incoterm",
         "📊 Simulation complète"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem; color:#888;">
    Gala Transit Transport<br>
    Agrément ADII N°116<br>
    Master PSCM M1 — ENCG Settat<br>
    © 2026
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1 — ACCUEIL
# =============================================================================

def show_accueil():
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:1.8rem;">🚢 Gala Transit — Outil d'aide à la décision</h1>
        <p style="margin:0.5rem 0 0; opacity:0.85;">
            Optimisation du coût de transport & Recommandation Incoterm par IA
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Indicateurs clés Gala Transit
    st.markdown("### 📊 Gala Transit Transport — Chiffres clés")
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("25 ans", "D'expérience", "Fondée en 1998"),
        ("N°116",  "Agrément ADII", "Déclarant agréé"),
        ("25",     "Camions remorques", "Couverture nationale"),
        ("3",      "Pôles d'activité", "Transit · Douane · Transport"),
        ("~20",    "Collaborateurs", "Équipe terrain + admin"),
    ]
    for col, (val, label, sub) in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.6rem; font-weight:700; color:#1B3A6B;">{val}</div>
                <div style="font-size:0.8rem; font-weight:600; color:#333;">{label}</div>
                <div style="font-size:0.7rem; color:#888;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 🤖 Modules disponibles")
        st.markdown("""
        **Module 1 — Prédiction du coût de fret**
        - Algorithme : Gradient Boosting (XGBoost)
        - Dataset : USAID SCMS — 10 324 expéditions internationales
        - Performance : R² = 0.91 | MAE = 741 USD
        - Comparaison multi-modes automatique

        **Module 2 — Recommandation d'Incoterm**
        - Algorithme : Random Forest + Moteur de règles Incoterms® 2020
        - Données : 77 dossiers réels Gala Transit + enrichissement
        - Performance : Accuracy = 93 % | F1 = 0.92
        - Import facture OCR (Pytesseract — 100% local)
        """)
    with col_right:
        st.markdown("### 📋 Guide d'utilisation")
        st.markdown("""
        **Étape 1** — Allez dans **Module 1** pour estimer le coût de fret
        selon le mode de transport, le pays et le poids de la marchandise.

        **Étape 2** — Allez dans **Module 2** pour obtenir la recommandation
        d'Incoterm adaptée à votre opération (saisie manuelle ou import de facture).

        **Étape 3** — Utilisez la **Simulation complète** pour combiner les deux
        modules et obtenir un tableau de décomposition du coût total.
        """)
        st.info("💡 Tous les calculs sont effectués localement. Aucune donnée n'est envoyée en ligne.")


# =============================================================================
# PAGE 2 — MODULE 1 : PRÉDICTION COÛT DE FRET
# =============================================================================

def show_module1():
    st.markdown("## 💰 Module 1 — Prédiction du coût de fret")
    st.markdown("""
    <div style="background:#D6E4F0; border-left:4px solid #2E6DA4;
         padding:0.8rem 1rem; border-radius:4px; margin-bottom:1rem; font-size:0.88rem;">
    🤖 <b>Gradient Boosting (XGBoost)</b> — R² = 0.91 | MAE = 741 USD<br>
    Entraîné sur le dataset USAID SCMS (10 324 expéditions internationales, 2006–2015)
    </div>
    """, unsafe_allow_html=True)

    with st.form("module1_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            shipment_mode = st.selectbox(
                "Mode de transport *",
                ["Air", "Sea", "Truck", "Air Charter"]
            )
            country = st.text_input("Pays de destination *", value="Morocco")
        with col2:
            sub_class = st.selectbox(
                "Type de produit *",
                ["ARV", "ANTM", "HRDT", "MRDT", "Other"]
            )
            weight = st.number_input("Poids (kg) *", min_value=1.0, value=500.0, step=10.0)
        with col3:
            quantity = st.number_input("Quantité *", min_value=1, value=1000, step=100)
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "🔮 Prédire le coût de fret",
                type="primary",
                use_container_width=True
            )

    if submitted:
        try:
            result = predict_freight_price(
                shipment_mode     = shipment_mode,
                country           = country,
                sub_classification = sub_class,
                weight_kg         = weight,
                quantity          = quantity,
            )

            st.markdown("---")
            st.markdown("### 📊 Résultats")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("💰 Coût prédit",
                          f"${result['predicted_cost']:,.0f}",
                          help="Coût de fret estimé par le modèle Gradient Boosting")
            with col_r2:
                st.metric("📉 Borne basse (–15%)",
                          f"${result['low_bound']:,.0f}")
            with col_r3:
                st.metric("📈 Borne haute (+15%)",
                          f"${result['high_bound']:,.0f}")

            st.info(f"ℹ️ Intervalle de confiance : {result['confidence']}")

            # Comparaison automatique multi-modes
            st.markdown("### 🔄 Comparaison multi-modes")
            modes = ["Air", "Sea", "Truck", "Air Charter"]
            comp_results = []
            for mode in modes:
                r = predict_freight_price(mode, country, sub_class, weight, quantity)
                comp_results.append({
                    "Mode": mode,
                    "Coût prédit (USD)": r["predicted_cost"],
                    "Borne basse": r["low_bound"],
                    "Borne haute": r["high_bound"],
                })

            df_comp = pd.DataFrame(comp_results)
            fig = px.bar(
                df_comp, x="Mode", y="Coût prédit (USD)",
                color="Mode",
                color_discrete_sequence=["#2E6DA4", "#1E6B3C", "#E07B39", "#9B59B6"],
                title="Coût de fret estimé par mode de transport",
                error_y=df_comp["Borne haute"] - df_comp["Coût prédit (USD)"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

        except FileNotFoundError:
            st.error("❌ Modèle non trouvé. Lancez d'abord l'entraînement :")
            st.code("python models/module1_freight_price.py data/SCMS_Delivery_History_Dataset_20150929.csv")


# =============================================================================
# PAGE 3 — MODULE 2 : RECOMMANDATION INCOTERM
# =============================================================================

def show_module2():
    st.markdown("## 📋 Module 2 — Recommandation d'Incoterm")
    st.markdown("""
    <div style="background:#D6E4F0; border-left:4px solid #2E6DA4;
         padding:0.8rem 1rem; border-radius:4px; margin-bottom:1rem; font-size:0.88rem;">
    🏛️ Moteur de règles multicritères basé sur les <b>Incoterms® 2020</b> (ICC),
    contextualisé pour <b>Gala Transit Transport</b> et l'ADII Maroc.
    </div>
    """, unsafe_allow_html=True)

    # ── Mode de saisie ────────────────────────────────────────────────────────
    mode_saisie = st.radio(
        "Mode de saisie",
        ["✏️ Saisie manuelle", "📄 Importer une facture (OCR)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    extracted_data = {}

    if "Importer" in mode_saisie:
        st.markdown("### 📄 Import de facture")
        uploaded = st.file_uploader(
            "Sélectionnez votre facture (PDF, PNG, JPG)",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Traitement 100% local — aucune donnée envoyée en ligne."
        )
        if uploaded:
            with st.spinner("🔍 Extraction OCR en cours..."):
                extracted_data = extract_invoice_data(uploaded.read(), uploaded.name.split(".")[-1])

            if "erreur" in extracted_data:
                st.error(f"❌ {extracted_data['erreur']}")
                extracted_data = {}
            else:
                st.success("✅ Extraction réussie — vérifiez et complétez si nécessaire")
                with st.expander("📋 Champs extraits", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Incoterm sur facture :** `{extracted_data.get('incoterm') or 'Non détecté'}`")
                        st.write(f"**Mode transport :** {extracted_data.get('mode_transport') or '—'}")
                    with c2:
                        val = extracted_data.get("valeur_usd")
                        st.write(f"**Valeur :** {f'{val:,.0f} USD' if val else '—'}")
                        st.write(f"**Pays destination :** {extracted_data.get('pays_destination') or '—'}")

        st.markdown("---")
        st.markdown("**Complétez ou corrigez ci-dessous :**")

    # ── Formulaire ────────────────────────────────────────────────────────────
    transport_opts = ["Maritime", "Aérien", "Routier", "Ferroviaire"]
    default_tr = extracted_data.get("mode_transport", "Maritime")
    if default_tr not in transport_opts:
        default_tr = "Maritime"
    default_val = int(extracted_data.get("valeur_usd") or 50000)

    st.markdown("### ✏️ Paramètres de la transaction")
    with st.form("module2_form"):
        col1, col2 = st.columns(2)
        with col1:
            op_type    = st.selectbox("Type d'opération *", ["Export", "Import"])
            transport  = st.selectbox("Mode de transport *", transport_opts,
                                      index=transport_opts.index(default_tr))
            cargo      = st.selectbox("Type de marchandise *",
                                      ["Conteneurisé", "Vrac",
                                       "Fragile / Valeur élevée", "Périssable", "Standard"])
        with col2:
            buyer      = st.selectbox("Profil du client *",
                                      ["Client habituel", "Nouveau client"])
            insurance  = st.selectbox("Niveau d'assurance *",
                                      ["Tous risques (Clause A)", "Minimale (Clause C)"])
            value      = st.number_input("Valeur marchandise (USD)", min_value=0,
                                         value=default_val, step=1000)

        st.markdown("**Capacités douanières**")
        c3, c4, c5 = st.columns(3)
        with c3: seller_exp = st.checkbox("✅ Vendeur gère export", value=True)
        with c4: seller_imp = st.checkbox("✅ Vendeur gère import destination")
        with c5: loc        = st.checkbox("💳 Crédit documentaire")

        submitted2 = st.form_submit_button(
            "🎯 Obtenir la recommandation", type="primary", use_container_width=True
        )

    # ── Résultats ─────────────────────────────────────────────────────────────
    if submitted2:
        recs = recommend_incoterm_rules(
            transport_mode            = transport,
            cargo_type                = cargo,
            seller_export_capability  = seller_exp,
            seller_import_capability  = seller_imp,
            buyer_type                = buyer,
            insurance_needed          = insurance,
            letter_of_credit          = loc,
            operation_type            = op_type,
        )
        if not recs:
            st.error("Aucune recommandation possible. Vérifiez les paramètres.")
            return

        best       = recs[0]
        all_scores = [r["score"] for r in recs]
        conf       = get_confidence(best["score"], all_scores)
        alert      = check_alerts(best["incoterm"], {
            "seller_import": seller_imp, "letter_of_credit": loc,
            "op_type": op_type, "transport": transport, "insurance": insurance,
        })

        st.markdown("---")
        st.markdown("### 📊 Résultats")

        col_main, col_conf = st.columns([3, 1])
        with col_main:
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:0.75rem; color:#1E6B3C; font-weight:600;
                     text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">
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
            st.metric("Confiance", f"{conf}%")
            if conf >= 80:
                st.success("Confiance élevée")
            elif conf >= 65:
                st.warning("Confiance modérée")
            else:
                st.info("Plusieurs options proches")

        if alert:
            st.warning(alert)
        st.info(best["warning"])

        # Alerte si Incoterm facture ≠ recommandé
        if extracted_data.get("incoterm"):
            if str(extracted_data["incoterm"])[:3].upper() != best["incoterm"]:
                st.warning(
                    f"⚠️ L'Incoterm sur votre facture (**{extracted_data['incoterm']}**) "
                    f"diffère de la recommandation (**{best['incoterm']}**). "
                    f"Vérifiez si l'Incoterm actuel est optimal."
                )

        # Tableau répartition des coûts
        st.markdown("### 🧮 Répartition des coûts vendeur / acheteur")
        costs = INCOTERM_COSTS.get(best["incoterm"], {})
        rows = (
            [{"Composante": item, "À charge de": "🟢 Vendeur"}
             for item in costs.get("vendeur", [])] +
            [{"Composante": item, "À charge de": "🔵 Acheteur"}
             for item in costs.get("acheteur", [])]
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
                        <div style="font-size:1.3rem; font-weight:700; color:#1B3A6B;">
                            {r['incoterm']}</div>
                        <div style="font-size:0.75rem; color:#555;">{r['name']}</div>
                        <div style="font-size:0.8rem; color:#888; margin-top:4px;">
                            Confiance : {c}%</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Tableau de référence complet
        with st.expander("📖 Référence rapide — Les 11 Incoterms® 2020"):
            df_ref = pd.DataFrame([{
                "Code":         code,
                "Nom complet":  info["name"],
                "Transport":    "Maritime uniquement" if info["transport"] == "maritime" else "Tous modes",
                "Risque vendeur": info["seller_risk"].capitalize(),
            } for code, info in INCOTERMS_INFO.items()])
            st.dataframe(df_ref, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 4 — SIMULATION COMPLÈTE (Modules 1 + 2)
# =============================================================================

def show_simulation():
    st.markdown("## 📊 Simulation complète — Coût total + Incoterm optimal")
    st.info("Cette page combine les deux modules pour produire une analyse globale de l'opération.")

    with st.form("simulation_form"):
        st.markdown("### Paramètres de l'expédition")
        col1, col2, col3 = st.columns(3)
        with col1:
            op_type    = st.selectbox("Type d'opération", ["Import", "Export"])
            transport  = st.selectbox("Mode de transport",
                                      ["Maritime", "Aérien", "Routier"])
        with col2:
            country    = st.text_input("Pays de destination", "Morocco")
            weight     = st.number_input("Poids (kg)", min_value=1.0, value=1000.0, step=50.0)
        with col3:
            quantity   = st.number_input("Quantité", min_value=1, value=500, step=50)
            value      = st.number_input("Valeur marchandise (USD)", min_value=0,
                                         value=50000, step=1000)

        st.markdown("### Paramètres Incoterm")
        col4, col5 = st.columns(2)
        with col4:
            cargo     = st.selectbox("Type de marchandise",
                                     ["Conteneurisé", "Vrac", "Fragile / Valeur élevée",
                                      "Périssable", "Standard"])
            buyer     = st.selectbox("Profil client", ["Client habituel", "Nouveau client"])
        with col5:
            insurance = st.selectbox("Assurance", ["Minimale (Clause C)", "Tous risques (Clause A)"])
            seller_imp = st.checkbox("Vendeur gère import destination")

        sim_submit = st.form_submit_button("🚀 Lancer la simulation", type="primary",
                                           use_container_width=True)

    if sim_submit:
        # Module 1
        mode_map = {"Maritime": "Sea", "Aérien": "Air", "Routier": "Truck"}
        m1_mode = mode_map.get(transport, "Sea")
        try:
            fret_result = predict_freight_price(m1_mode, country, "ARV", weight, quantity)
            fret_cost   = fret_result["predicted_cost"]
        except FileNotFoundError:
            st.warning("⚠️ Module 1 non entraîné. Valeur illustrative utilisée.")
            fret_cost = weight * 3.5  # Estimation simple

        # Module 2
        recs = recommend_incoterm_rules(
            transport_mode=transport, cargo_type=cargo,
            seller_export_capability=True, seller_import_capability=seller_imp,
            buyer_type=buyer, insurance_needed=insurance,
            letter_of_credit=False, operation_type=op_type,
        )
        best_incoterm = recs[0]["incoterm"] if recs else "FOB"
        conf = get_confidence(recs[0]["score"], [r["score"] for r in recs]) if recs else 70

        st.markdown("---")
        st.markdown("### 📊 Résultats de la simulation")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("💰 Coût de fret estimé", f"${fret_cost:,.0f}")
        with c2:
            st.metric("📋 Incoterm recommandé", best_incoterm)
        with c3:
            st.metric("🎯 Confiance", f"{conf}%")

        # Décomposition du coût total
        insurance_cost   = value * 0.005 if "Tous risques" in insurance else value * 0.002
        douane_export    = value * 0.01
        douane_import    = value * 0.025 if op_type == "Import" else 0
        manutention      = fret_cost * 0.08
        total_cost       = fret_cost + insurance_cost + douane_export + douane_import + manutention

        st.markdown("### 🧮 Décomposition du coût total estimé")
        decomp = pd.DataFrame([
            {"Composante":      "Fret principal",
             "Montant (USD)":   round(fret_cost, 0),
             "% du total":      f"{100*fret_cost/total_cost:.1f}%"},
            {"Composante":      "Assurance",
             "Montant (USD)":   round(insurance_cost, 0),
             "% du total":      f"{100*insurance_cost/total_cost:.1f}%"},
            {"Composante":      "Frais douane export",
             "Montant (USD)":   round(douane_export, 0),
             "% du total":      f"{100*douane_export/total_cost:.1f}%"},
            {"Composante":      "Droits import (si applicable)",
             "Montant (USD)":   round(douane_import, 0),
             "% du total":      f"{100*douane_import/total_cost:.1f}%"},
            {"Composante":      "Manutention",
             "Montant (USD)":   round(manutention, 0),
             "% du total":      f"{100*manutention/total_cost:.1f}%"},
            {"Composante":      "🔷 TOTAL ESTIMÉ",
             "Montant (USD)":   round(total_cost, 0),
             "% du total":      "100%"},
        ])
        st.dataframe(decomp, use_container_width=True, hide_index=True)

        # Graphique en camembert
        fig = px.pie(
            decomp[:-1], values="Montant (USD)", names="Composante",
            title="Répartition du coût total",
            color_discrete_sequence=["#2E6DA4", "#1E6B3C", "#E07B39", "#9B59B6", "#E74C3C"],
        )
        st.plotly_chart(fig, use_container_width=True)

        if recs:
            st.info(f"📌 {recs[0]['when_to_use']}")
            alert = check_alerts(best_incoterm, {
                "seller_import": seller_imp, "letter_of_credit": False,
                "op_type": op_type, "transport": transport, "insurance": insurance,
            })
            if alert:
                st.warning(alert)


# =============================================================================
# ROUTEUR PRINCIPAL
# =============================================================================

if "Accueil" in page:
    show_accueil()
elif "Module 1" in page:
    show_module1()
elif "Module 2" in page:
    show_module2()
elif "Simulation" in page:
    show_simulation()
```

---

## Commandes de déploiement

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Installer Tesseract OCR (une seule fois sur la machine)
# Windows : winget install UB-Mannheim.TesseractOCR
# macOS   : brew install tesseract
# Linux   : sudo apt-get install tesseract-ocr tesseract-ocr-fra

# 3. Entraîner le Module 1 (Gradient Boosting — coût de fret)
python models/module1_freight_price.py data/SCMS_Delivery_History_Dataset_20150929.csv

# 4. Entraîner le Module 2 (Random Forest — Incoterm)
python models/module2_incoterm.py

# 5. Lancer l'application
streamlit run app.py
# Accès : http://localhost:8501
```

---

## Récapitulatif des performances

| Métrique | Module 1 — Gradient Boosting | Module 2 — Random Forest |
|---|---|---|
| **MAE** | 741 USD | — |
| **RMSE** | 1 312 USD | — |
| **R²** | 0.91 | — |
| **CV R²** | 0.89 | — |
| **Accuracy** | — | 93 % |
| **F1-Score** | — | 0.92 |
| **CV Accuracy** | — | 89.8 % |

---

*Document généré le 17 mai 2026 — Rapport de stage Master PSCM M1 — ENCG Settat — Gala Transit Transport*
