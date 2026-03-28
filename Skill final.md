

# Academic Report Generator - Supply Chain Data Science

## Overview

Ce skill génère automatiquement un compte rendu académique complet et structuré à partir d'un simple sujet fourni par l'utilisateur.

| Champ | Valeur |
|-------|--------|
| **Étudiante** | Douae Anehari |
| **Apogée** | 22007689 |
| **Établissement** | ENCG Settat |
| **Filière** | Purchasing and Supply Chain Management |
| **Semestre** | S8 |
| **Enseignant** | M. Laghlimi |
| **Matière** | Data Analytics & Machine Learning |
| **Année universitaire** | 2025-2026 |

---

## Données Personnelles de l'Étudiante

```yaml
student_info:
  nom: "Anehari"
  prenom: "Douae"
  nom_complet: "Douae Anehari"
  numero_apogee: "22007689"
  email: "dou3aeanehari@gmail.com"
  
  # PHOTO
  photo_url: "https://i.ibb.co/RL78LkH/IMG-20251211-WA0001.jpg"
  photo_alt: "Photo de Douae Anehari"
  
  # PARCOURS ACADÉMIQUE
  etablissement: "ENCG Settat"
  filiere: "Purchasing and Supply Chain Management"
  semestre: "S8"
  annee_universitaire: "2025-2026"
  
  # ENCADREMENT
  enseignant: "M. Laghlimi"
  matiere: "Data Analytics & Machine Learning"
```

---

## When to Use This Skill

Utilisez ce skill lorsque vous souhaitez :

· ✅ Générer un compte rendu académique complet à partir d'un sujet

· ✅ Produire automatiquement un script Python adapté au sujet choisi

· ✅ Obtenir une analyse structurée avec visualisations

· ✅ Créer un document prêt à être soumis à l'ENCG Settat

· ✅ Gagner du temps sur la mise en forme académique

---

## Comment Utiliser Ce Skill

Étape 1 : Vérifier les données

Les données personnelles de Douae Anehari sont déjà intégrées :

· Nom complet : Douae Anehari

· Apogée : 22007689

· Email : dou3aeanehari@gmail.com

· Photo : https://i.ibb.co/RL78LkH/IMG-20251211-WA0001.jpg


Étape 2 : Fournir le sujet

Lancez le skill avec le sujet de votre choix :

```
Sujet : [VOTRE SUJET]
```

## Exemples :

· Sujet : Prédiction des ruptures de stock dans une entreprise de distribution

· Sujet : Classification des fournisseurs selon leur niveau de risque

· Sujet : Prévision de la demande pour optimiser les approvisionnements

· Sujet : Prédiction des délais de livraison des fournisseurs

---

## Structure du Rapport Généré

1. Page de Garde (Auto-remplie avec photo)

La page de garde inclut automatiquement :

· Logo ENCG Settat

· Photo de Douae Anehari

· Établissement : ENCG Settat

· Filière : Purchasing and Supply Chain Management

· Semestre : S8

· Titre du rapport : Généré à partir du sujet

· Informations étudiant : Douae Anehari, 22007689

· Encadrement : M. Laghlimi

· Date de soumission

2. Sommaire

Généré automatiquement avec toutes les sections du rapport.

3. Présentation du Sujet (Contextualisation)

· Contexte : Mise en situation du sujet dans la supply chain

· Problématique : Formulée à partir du sujet fourni

· Objectifs : Objectifs spécifiques du projet

· Importance : Justification de l'approche data science

4. Contextualisation Académique

Rappel du cadre académique :

· Établissement : ENCG Settat

· Filière : Purchasing and Supply Chain Management

· Semestre : S8

· Matière : Data Analytics & Machine Learning

· Enseignant : M. Laghlimi

· Étudiante : Douae Anehari (Apogée : 22007689)

· Année universitaire : 2025-2026

5. Script Python

Script complet adapté au sujet avec :

· Importation des bibliothèques

· Chargement d'un dataset réel

· Exploration et prétraitement

· Modélisation (classification ou prédiction)

· Évaluation et visualisations

6. Analyse des Résultats

· Métriques de performance (accuracy, précision, rappel, F1-score)

· Matrice de confusion commentée

· Importance des variables analysée

· Traduction métier des résultats

· Limites identifiées

7. Visualisations

Trois visualisations générées et commentées :

1. Distribution de la variable cible
2. Matrice de confusion (heatmap)
3. Importance des variables (barres horizontales)
  
# 8. Conclusion et Perspectives
   
   

· Synthèse des résultats

· Recommandations métier

· Pistes d'amélioration

· Perspectives pour travaux futurs

---

## Code HTML Complet de la Page de Garde

```html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport - [TITRE DU RAPPORT]</title>
    <style>
        @page {
            size: A4;
            margin: 2.5cm;
        }
        
        body {
            font-family: 'Times New Roman', 'Segoe UI', serif;
            margin: 0;
            padding: 0;
            background: white;
        }
        
        .cover {
            text-align: center;
            padding: 40px 30px;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .logo {
            font-size: 24px;
            font-weight: bold;
            letter-spacing: 2px;
            color: #1e3c72;
            border-bottom: 3px solid #ff6b35;
            display: inline-block;
            padding-bottom: 10px;
        }
        
        .school-name {
            font-size: 28px;
            font-weight: bold;
            color: #1e3c72;
            margin: 20px 0 5px;
        }
        
        .school-sub {
            font-size: 16px;
            color: #4a627a;
            margin-bottom: 30px;
        }
        
        .hero {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 50px;
            margin: 40px 0;
            flex-wrap: wrap;
        }
        
        .photo {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
            border: 4px solid #ff6b35;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .title-section {
            text-align: left;
        }
        
        .program {
            font-size: 18px;
            color: #2c3e66;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .semester {
            font-size: 16px;
            color: #ff6b35;
            margin-bottom: 20px;
        }
        
        .report-title {
            font-size: 32px;
            font-weight: bold;
            color: #1e3c72;
            line-height: 1.3;
            max-width: 500px;
        }
        
        .info-box {
            background-color: #f8f9fa;
            border-left: 5px solid #ff6b35;
            padding: 20px 30px;
            text-align: left;
            margin: 30px auto;
            max-width: 400px;
            border-radius: 8px;
        }
        
        .info-box p {
            margin: 8px 0;
            font-size: 14px;
        }
        
        .info-box strong {
            color: #1e3c72;
        }
        
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #888;
        }
        
        @media print {
            .cover {
                page-break-after: always;
            }
        }
    </style>
</head>
<body>
    <div class="cover">
        <div>
            <div class="logo">ENCG SETTAT</div>
            <div class="school-name">École Nationale de Commerce et de Gestion</div>
            <div class="school-sub">Université Hassan Premier - Settat</div>
        </div>
        
        <div class="hero">
            <img src="https://i.ibb.co/RL78LkH/IMG-20251211-WA0001.jpg" alt="Photo de Douae Anehari" class="photo">
            <div class="title-section">
                <div class="program">Purchasing and Supply Chain Management</div>
                <div class="semester">Semestre S8</div>
                <div class="report-title">[TITRE DU RAPPORT]</div>
            </div>
        </div>
        
        <div class="info-box">
            <p><strong>Réalisé par :</strong></p>
            <p><strong>Douae Anehari</strong></p>
            <p><strong>Numéro Apogée :</strong> 22007689</p>
            <p><strong>Email :</strong> dou3aeanehari@gmail.com</p>
        </div>
        
        <div>
            <p><strong>Encadré par :</strong> M. Laghlimi</p>
            <p><strong>Matière :</strong> Data Analytics & Machine Learning</p>
            <div class="footer">
                <p>Année universitaire : 2025-2026 | Date de soumission : [Date actuelle]</p>
            </div>
        </div>
    </div>
</body>
</html>
```

---

## Script Python Complet

```python
# ============================================
# PROJET : [TITRE DU RAPPORT]
# ÉTUDIANTE : Douae Anehari | APOGÉE : 22007689
# ENSEIGNANT : M. Laghlimi | MATIÈRE : Data Analytics & ML
# DATE : [Date actuelle]
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("ANALYSE DATA SCIENCE - SUPPLY CHAIN")
print(f"Étudiante : Douae Anehari (Apogée: 22007689)")
print(f"Enseignant : M. Laghlimi")
print("=" * 60)

# 1. CHARGEMENT DES DONNÉES
print("\n[1] CHARGEMENT DES DONNÉES")
try:
    df = pd.read_csv('supply_chain_data.csv')
    print(f"✓ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
except:
    print("✗ Dataset non trouvé. Téléchargez supply_chain_data.csv depuis Kaggle")
    exit()

# 2. EXPLORATION DES DONNÉES
print("\n[2] EXPLORATION DES DONNÉES")
print("\nPremières lignes :")
print(df.head())
print("\nInformations générales :")
print(df.info())
print("\nStatistiques descriptives :")
print(df.describe())
print(f"\nValeurs manquantes :\n{df.isnull().sum()}")

# 3. PRÉTRAITEMENT
print("\n[3] PRÉTRAITEMENT DES DONNÉES")
df = df.dropna()
print(f"✓ Après suppression des valeurs manquantes : {df.shape[0]} lignes")

# Encodage des variables catégorielles
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])
print(f"✓ Encodage des variables catégorielles effectué")

# 4. SÉPARATION FEATURES / TARGET
print("\n[4] SÉPARATION FEATURES / TARGET")
X = df.select_dtypes(include=[np.number])
y = (X.sum(axis=1) > X.median().sum()).astype(int)

print(f"✓ Features : {X.shape[1]} variables")
print(f"✓ Distribution de la cible :\n{y.value_counts()}")

# 5. DIVISION ENTRAÎNEMENT / TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n✓ Entraînement : {X_train.shape[0]} échantillons")
print(f"✓ Test : {X_test.shape[0]} échantillons")

# 6. MODÉLISATION
print("\n[5] MODÉLISATION")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✓ Modèle Random Forest entraîné")

# 7. ÉVALUATION
print("\n[6] ÉVALUATION DU MODÈLE")
y_pred = model.predict(X_test)

print("\nClassification Report :")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print(f"\nMatrice de confusion :\n{cm}")

# 8. VISUALISATIONS
print("\n[7] GÉNÉRATION DES VISUALISATIONS")

# Visualisation 1 : Distribution de la cible
plt.figure(figsize=(8, 6))
sns.countplot(x=y, palette=['#2ecc71', '#e74c3c'])
plt.title('Distribution de la Variable Cible', fontsize=14)
plt.xlabel('Classe')
plt.ylabel("Nombre d'observations")
plt.tight_layout()
plt.savefig('distribution_cible.png', dpi=300)
print("✓ Figure 1 sauvegardée : distribution_cible.png")

# Visualisation 2 : Matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion', fontsize=14)
plt.xlabel('Prédiction')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig('matrice_confusion.png', dpi=300)
print("✓ Figure 2 sauvegardée : matrice_confusion.png")

# Visualisation 3 : Importance des variables
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
features = X.columns[indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], color='#3498db')
plt.yticks(range(len(indices)), features)
plt.xlabel('Importance')
plt.title('Top 10 des Variables les Plus Importantes', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("✓ Figure 3 sauvegardée : feature_importance.png")

print("\n" + "=" * 60)
print("ANALYSE TERMINÉE AVEC SUCCÈS")
print("=" * 60)
```

---

## Instructions pour Générer un Rapport

1. Copiez ce fichier skill.md dans votre environnement
2. Fournissez un sujet en répondant :
   ```
   Sujet : [Votre sujet ici]
   ```
3. Recevez le rapport complet au format HTML
4. Convertissez en PDF avec Ctrl+P → "Enregistrer en PDF"

---
