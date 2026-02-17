# COURS COMPLET : INTELLIGENCE ARTIFICIELLE
## Finance, Contrôle Audit et Conseil, Marketing, Commerce International & Supply Chain Management

## Abderrahim.larhlimi@uhp.ac.ma
![](https://github.com/LarhlimiUhp/IA/blob/main/Pr%C3%A9paration/profil.png)


## Maître de Conférence, ENSGS
## Université Hassan Premier

**Niveau: 4ème année**  
**Année 2025-2026**

---

## Présentation du Manuel

Ce manuel universitaire constitue un support de cours exhaustif sur l'Intelligence Artificielle appliquée aux domaines de la Finance, du Contrôle Audit et Conseil, du Marketing, du Commerce Internationalet du supply chain management.  Chaque concept, formule mathématique et algorithme est expliqué en profondeur avec des exemples métiers concrets et des cas d'entreprises réels.

### Objectifs Pédagogiques

À l'issue de ce cours, vous serez capable de :

1. **Comprendre** les concepts fondamentaux de l'IA, du Machine Learning et du Deep Learning
2. **Maîtriser** les méthodes clés d'apprentissage supervisé, non supervisé et de deep learning
3. **Appliquer** l'IA à des problématiques concrètes de finance et de gestion
4. **Développer** des solutions reproductibles en Python avec pandas, scikit-learn et Keras
5. **Évaluer** les enjeux éthiques, les biais et la conformité (RGPD, AI Act)
6. **Piloter** des projets IA et dialoguer efficacement avec les équipes data/IT

### Prérequis

- **Mathématiques** : Statistiques descriptives et inférentielles, algèbre linéaire de base
- **Programmation** : Notions de base en Python
- **Métier** : Compréhension des KPI en finance et gestion, familiarité avec Excel/BI

### Modalités d'Évaluation

- **40%** : Contrôle continu (QCM, mini-projets, participation)
- **30%** : Projet pratique (étude de cas, code, rapport, soutenance)
- **30%** : Examen final (théorie + études de cas)

---

## Table des Matières

1. [Chapitre 1 — Introduction à l'Intelligence Artificielle](#chapitre-1)
2. [Chapitre 2 — Fondements Mathématiques](#chapitre-2)
3. [Chapitre 3 — Apprentissage Supervisé](#chapitre-3)
4. [Chapitre 4 — Apprentissage Non Supervisé](#chapitre-4)
5. [Chapitre 5 — Deep Learning](#chapitre-5)
6. [Chapitre 6 — Natural Language Processing (NLP)](#chapitre-6)
7. [Chapitre 7 — IA en Finance](#chapitre-7)
8. [Chapitre 8 — IA en Controle Audit et Conseil](#chapitre-8)
9. [Chapitre 9 — IA en Marketing](#chapitre-9)
10. [Chapitre 10 — IA en Commerce International](#chapitre-10)
11. [Chapitre 11 — IA en Supply Chain Management](#chapitre-11)
12. [Chapitre 12 — MLOps et Déploiement](#chapitre-12)
13. [Conclusion](#Conclusion)


---

<a name="chapitre-1"></a>
# Chapitre 1 — Introduction à l'Intelligence Artificielle

## 1.1 Qu'est-ce que l'Intelligence Artificielle ?

### 1.1.1 Définition Formelle

L'Intelligence Artificielle (IA) est une branche de l'informatique qui vise à créer des systèmes capables d'effectuer des tâches qui nécessitent traditionnellement l'intelligence humaine. Ces tâches incluent :

- Le **raisonnement** et la résolution de problèmes complexes
- La **perception** et l'interprétation de données sensorielles
- La **compréhension** du langage naturel
- La **prise de décision** dans des environnements incertains
- L'**apprentissage** à partir de l'expérience

Formellement, l'IA cherche à modéliser et à implémenter des fonctions cognitives en utilisant des algorithmes et des architectures computationnelles.

### 1.1.2 Définition Intuitive

Imaginez un système informatique qui peut :
- **Apprendre** de ses erreurs comme un étudiant qui s'améliore après chaque examen
- **S'adapter** à de nouvelles situations sans être reprogrammé pour chaque cas
- **Reconnaître** des patterns complexes que l'œil humain pourrait manquer
- **Prédire** des événements futurs en analysant des données historiques

L'IA, c'est essentiellement donner aux machines la capacité d'imiter l'intelligence humaine, non pas en copiant le cerveau, mais en utilisant des approches mathématiques et statistiques pour arriver à des résultats similaires.

### 1.1.3 Exemple Concret

**Reconnaissance faciale sur votre smartphone** :
- Lorsque vous déverrouillez votre téléphone avec votre visage, un système d'IA :
  1. **Détecte** votre visage dans l'image capturée par la caméra
  2. **Extrait** des caractéristiques uniques (distance entre les yeux, forme du nez, etc.)
  3. **Compare** ces caractéristiques avec le modèle stocké lors de l'enregistrement
  4. **Décide** de déverrouiller ou non en quelques millisecondes

Ce processus utilise des réseaux de neurones profonds entraînés sur des millions d'images de visages.

### 1.1.4 Exemples Métier

#### Finance : Détection de Fraude Bancaire
Les banques utilisent l'IA pour analyser en temps réel les transactions et détecter les comportements suspects :
- **Analyse de patterns** : Une transaction de 5000€ à 3h du matin au Japon alors que vous vivez au Maroc
- **Score de risque** : Attribution automatique d'un score de probabilité de fraude
- **Décision automatisée** : Blocage temporaire de la carte et envoi d'une alerte SMS
- **Apprentissage continu** : Le système s'améliore en analysant les retours (vraie fraude vs fausse alerte)

#### Gestion : Prédiction du Turnover RH
Les entreprises prédisent quels employés risquent de démissionner :
- **Variables analysées** : Ancienneté, salaire, promotions, satisfaction, absences
- **Modèle prédictif** : Calcul d'un score de risque de départ pour chaque employé
- **Action préventive** : Programmes de rétention ciblés sur les employés à risque
- **ROI mesurable** : Réduction de 20-30% du turnover dans certaines entreprises

### 1.1.5 Schéma Explicatif (Description Textuelle)

![ SYSTÈME D'INTELLIGENCE ARTIFICIELLE](https://github.com/LarhlimiUhp/IA/blob/main/Pr%C3%A9paration/IA.png)

```
┌─────────────────────────────────────────────────────────────┐
│              SYSTÈME D'INTELLIGENCE ARTIFICIELLE            │
└─────────────────────────────────────────────────────────────┘
                               │
                               ↓
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   DONNÉES        │───>│   ALGORITHMES    │───>│   DÉCISIONS/     │
│   D'ENTRÉE       │    │   D'APPRENTISSAGE│    │   PRÉDICTIONS    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
        │                       │                        │
        │                       ↓                        │
        │              ┌──────────────────┐              │
        │              │    MODÈLE        │              │
        │              │    ENTRAÎNÉ      │              │
        │              └──────────────────┘              │
        │                       │                        │
        └───────────────────────┼────────────────────────┘
                                │
                                ↓
                        ┌──────────────────┐
                        │   FEEDBACK       │
                        │   ET AMÉLIORATION│
                        └──────────────────┘
```

**Explication du schéma** :
1. **Données d'entrée** : Informations brutes (images, textes, chiffres, transactions)
2. **Algorithmes d'apprentissage** : Méthodes mathématiques qui analysent les données
3. **Modèle entraîné** : Représentation mathématique des patterns découverts
4. **Décisions/Prédictions** : Résultats produits par le système
5. **Feedback** : Évaluation des résultats pour améliorer le modèle
6. **Boucle d'amélioration** : Le système s'améliore continuellement grâce au feedback

### 1.1.6 Cas Réel d'Entreprise : Netflix

**Contexte** :
Netflix compte plus de 230 millions d'abonnés dans le monde qui choisissent parmi un catalogue de plus de 15 000 titres. Sans IA, trouver un contenu intéressant serait comme chercher une aiguille dans une botte de foin.

**Solution IA Mise en Place** :
Netflix utilise un système de recommandation sophistiqué qui combine plusieurs algorithmes d'IA :

1. **Filtrage Collaboratif** :
   - Analyse les habitudes de visionnage de millions d'utilisateurs
   - Identifie des groupes d'utilisateurs avec des goûts similaires
   - Recommande ce que des utilisateurs similaires ont aimé

2. **Filtrage basé sur le Contenu** :
   - Analyse les métadonnées de chaque film/série (genre, acteurs, réalisateur, thèmes)
   - Identifie les caractéristiques des contenus que vous avez aimés
   - Suggère des titres avec des caractéristiques similaires

3. **Apprentissage Profond** :
   - Analyse les images des vignettes pour prédire lesquelles vous attireront
   - Personnalise même les vignettes affichées selon vos préférences
   - Optimise l'ordre de présentation des recommandations

**Résultats Mesurables** :
- **80%** du contenu visionné provient des recommandations de l'IA
- **Économie de 1 milliard de dollars** par an en rétention d'abonnés
- **Engagement accru** : Les utilisateurs passent moins de temps à chercher et plus à regarder
- **Personnalisation** : Chaque utilisateur voit une page d'accueil unique

**Technologies Utilisées** :
- Python, Apache Spark pour le traitement big data
- TensorFlow et PyTorch pour les modèles de deep learning
- A/B testing continu pour optimiser les algorithmes

---

## 1.2 Machine Learning (Apprentissage Automatique)

### 1.2.1 Définition Formelle

Le Machine Learning (ML) est une sous-discipline de l'IA qui se concentre sur le développement d'algorithmes permettant aux ordinateurs d'apprendre à partir de données et d'améliorer leurs performances sur une tâche spécifique sans être explicitement programmés pour chaque cas.

Formellement, un programme informatique apprend de l'expérience E par rapport à une classe de tâches T et une mesure de performance P, si sa performance sur T, mesurée par P, s'améliore avec l'expérience E.

**Équation fondamentale du ML** :


![ SYSTÈME ML](https://github.com/LarhlimiUhp/IA/blob/main/Pr%C3%A9paration/prediction.png)
```
Apprendre : Données + Algorithme → Modèle
Prédire  : Nouvelles Données + Modèle → Prédictions
```

### 1.2.2 Définition Intuitive

Le Machine Learning, c'est comme enseigner à un enfant à reconnaître des animaux :
- Au lieu de programmer des règles explicites ("si ça a 4 pattes, un museau et aboie, c'est un chien")
- On montre des **exemples** (des milliers de photos de chiens)
- L'algorithme **découvre lui-même** les patterns qui caractérisent un chien
- Il peut ensuite **généraliser** pour reconnaître de nouveaux chiens jamais vus

La différence avec la programmation traditionnelle :
- **Programmation classique** : Règles + Données → Résultats
- **Machine Learning** : Données + Résultats → Règles (modèle)

### 1.2.3 Exemple Concret

**Filtre anti-spam d'email** :

**Approche traditionnelle (programmation classique)** :
```python
# Règles explicites programmées manuellement
if "lottery" in email or "prince" in email:
    return "SPAM"
else:
    return "HAM" (légitime)
```
Problème : Les spammeurs contournent facilement ces règles

**Approche Machine Learning** :
```python
# 1. Collecter des exemples
emails_spam = ["You won the lottery!", ...]
emails_ham = ["Meeting at 3pm", "Project report attached", ...]

# 2. Entraîner un modèle
model = NaiveBayes()
model.fit(emails, labels)  # Apprend automatiquement les patterns

# 3. Prédire sur de nouveaux emails
new_email = "Congratulations! Click here to claim prize"
prediction = model.predict(new_email)  # → SPAM (probabilité: 95%)
```

Le modèle apprend automatiquement que :
- Certains mots sont plus fréquents dans les spams ("free", "win", "click")
- La structure des phrases diffère (beaucoup de ponctuation, MAJUSCULES)
- Les spams contiennent souvent des liens suspects

### 1.2.4 Exemples Métier

#### Finance : Scoring de Crédit Automatisé

**Problématique** :
Une banque reçoit 10 000 demandes de crédit par mois. Comment décider rapidement et objectivement qui obtient un prêt ?

**Solution ML** :
```
1. DONNÉES HISTORIQUES (5 ans)
   - 100 000 dossiers de crédit passés
   - Pour chaque client : âge, revenu, emploi, dettes, historique bancaire
   - Résultat connu : a remboursé (0) ou fait défaut (1)

2. ENTRAÎNEMENT DU MODÈLE
   - Algorithme : Régression Logistique ou XGBoost
   - Le modèle apprend les combinaisons de facteurs qui prédisent le défaut
   - Exemples de patterns découverts :
     * Ratio dette/revenu > 40% → risque élevé
     * Emploi stable + épargne >50k → risque faible
     * Âge < 25 + crédit > 20k → risque moyen-élevé

3. MISE EN PRODUCTION
   - Nouveau dossier → Score de 0 à 1000
   - Score > 700 : Approbation automatique
   - Score 400-700 : Examen manuel
   - Score < 400 : Refus automatique

4. RÉSULTATS
   - Temps de décision : 48h → 5 minutes
   - Taux de défaut réduit de 15%
   - Objectivité accrue (réduction des biais humains)
```

#### Gestion : Prévision des Ventes pour Optimiser les Stocks

**Problématique** :
Un retailer doit commander les quantités optimales pour chaque magasin et chaque produit. Trop = invendus, Pas assez = ruptures = clients mécontents.

**Solution ML** :
```
1. DONNÉES COLLECTÉES
   - Historique des ventes (3 ans, par jour, produit, magasin)
   - Facteurs externes : météo, jours fériés, promotions, événements
   - Caractéristiques produit : catégorie, prix, saisonnalité

2. MODÈLE DE PRÉVISION
   - Algorithme : LSTM (réseaux de neurones récurrents) ou Prophet
   - Le modèle apprend :
     * Les tendances saisonnières (+ ventes de glaces en été)
     * L'impact des promotions (+30% lors des -20%)
     * Les corrélations entre produits (chips + sodas)

3. PRÉDICTIONS QUOTIDIENNES
   - Prévision à 7 jours : Quantité attendue par produit/magasin
   - Intervalle de confiance : Min-Max pour gérer l'incertitude
   - Alertes automatiques si risque de rupture

4. IMPACT BUSINESS
   - Réduction des ruptures de stock : -25%
   - Diminution des invendus : -18%
   - Satisfaction client : +12%
   - ROI : 300% la première année
```

### 1.2.5 Schéma Explicatif : Les Trois Types d'Apprentissage

```
┌────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING                            │
└────────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ↓                  ↓                  ↓
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  APPRENTISSAGE  │  │  APPRENTISSAGE   │  │  APPRENTISSAGE   │
│    SUPERVISÉ    │  │   NON SUPERVISÉ  │  │  PAR RENFORCEMENT│
└─────────────────┘  └──────────────────┘  └──────────────────┘

Données étiquetées    Données non          Agent + Environnement
(X, y)               étiquetées (X)        Récompenses/Pénalités

Exemples:             Exemples:            Exemples:
• Classification      • Clustering         • Jeux (échecs, Go)
• Régression         • Réduction dimension • Robots autonomes
• Prédiction         • Détection anomalies • Trading algorithmique

Algorithmes:          Algorithmes:         Algorithmes:
• Régression linéaire • K-means           • Q-Learning
• Random Forest      • PCA               • Deep Q-Network
• Neural Networks    • DBSCAN            • AlphaGo
```

**Explication détaillée des trois types** :

1. **Apprentissage Supervisé** (comme un élève avec un professeur)
   - On fournit des **exemples avec les réponses** (données étiquetées)
   - Le modèle apprend la relation entre entrées (X) et sorties (y)
   - Utilisé pour : prédire, classifier, estimer

2. **Apprentissage Non Supervisé** (comme un explorateur)
   - On fournit des **données sans réponses** (non étiquetées)
   - Le modèle découvre la structure cachée dans les données
   - Utilisé pour : segmenter, réduire la dimensionnalité, détecter des anomalies

3. **Apprentissage par Renforcement** (comme un enfant qui apprend par essai-erreur)
   - Un **agent** interagit avec un **environnement**
   - Il reçoit des **récompenses** pour les bonnes actions, des **pénalités** pour les mauvaises
   - Il apprend la **stratégie optimale** pour maximiser les récompenses cumulées

### 1.2.6 Cas Réel d'Entreprise : Amazon

**Système de Recommandation "Customers Who Bought This Also Bought"**

**Contexte** :
Amazon vend des centaines de millions de produits. Comment suggérer les bons produits à chaque client parmi cette immensité ?

**Solution ML Déployée** :

**1. Collecte de Données Massives**
```
Pour chaque utilisateur :
- Historique d'achats
- Produits consultés
- Temps passé sur chaque page
- Produits ajoutés au panier puis retirés
- Recherches effectuées
- Avis laissés
```

**2. Algorithmes ML Utilisés**

a) **Filtrage Collaboratif (Item-to-Item)**
```python
# Pseudo-code simplifié
# Si vous achetez un livre de science-fiction:

similarities = {}
for other_item in all_items:
    users_who_bought_both = count_users(book, other_item)
    similarity_score = cosine_similarity(book, other_item)
    similarities[other_item] = similarity_score

# Recommander les items avec les scores les plus élevés
top_recommendations = sorted(similarities)[:10]
```

b) **Analyse des Patterns d'Achat**
```
Pattern découvert : 
Achat(Nintendo Switch) → forte probabilité d'acheter :
- Jeux Switch (95%)
- Manette supplémentaire (70%)
- Pochette de transport (60%)
- Carte SD (55%)

→ Afficher ces produits dans "Fréquemment achetés ensemble"
```

**3. Architecture Technique**
```
┌──────────────────┐
│  DONNÉES CLIENT  │ → Historique + Comportement temps réel
└────────┬─────────┘
         │
         ↓
┌──────────────────────────────────────┐
│  MODÈLES ML (plusieurs algorithmes)  │
├──────────────────────────────────────┤
│ • Collaborative Filtering            │
│ • Deep Learning (embeddings)         │
│ • Association Rules Mining           │
│ • Sequential Pattern Mining          │
└────────┬─────────────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│  PERSONNALISATION EN TEMPS   │
│  RÉEL                        │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│  RECOMMANDATIONS AFFICHÉES   │
│  (< 100ms de latence)        │
└──────────────────────────────┘
```

**4. Résultats Business**
- **35% du chiffre d'affaires** d'Amazon provient des recommandations
- **Augmentation de 29%** du panier moyen
- **Taux de conversion** : +15% pour les utilisateurs qui cliquent sur les recommandations
- **Valeur estimée** : Plusieurs milliards de dollars de revenus additionnels par an

**5. Améliorations Continues**
Amazon utilise l'A/B testing en permanence :
- Teste différentes versions d'algorithmes simultanément
- Mesure l'impact sur les KPIs (taux de clic, conversion, revenus)
- Déploie automatiquement la meilleure variante
- Réentraîne les modèles quotidiennement avec les nouvelles données

---

## 1.3 Deep Learning (Apprentissage Profond)

### 1.3.1 Définition Formelle

Le Deep Learning est une sous-catégorie du Machine Learning basée sur des réseaux de neurones artificiels comportant **plusieurs couches cachées** (d'où le terme "profond"). Ces architectures sont capables d'apprendre des représentations hiérarchiques des données, passant de caractéristiques simples (bords, textures) à des concepts complexes (objets, scènes).

Mathématiquement, un réseau de neurones profond est une composition de fonctions :

```
f(x) = f_L(f_{L-1}(...f_2(f_1(x; θ₁); θ₂)...; θ_{L-1}); θ_L)
```

Où :
- `x` = données d'entrée
- `f_i` = fonction de la couche i (transformation linéaire + activation non-linéaire)
- `θ_i` = paramètres (poids et biais) de la couche i
- `L` = nombre de couches

### 1.3.2 Définition Intuitive

Imaginez que vous voulez reconnaître un chat dans une image :

**Approche ML Classique** :
- Vous devez définir manuellement les caractéristiques pertinentes
- "Extrayez la forme des oreilles, comptez les moustaches, mesurez la taille des yeux..."
- C'est fastidieux et vous pourriez manquer des caractéristiques importantes

**Approche Deep Learning** :
- Vous montrez 100 000 images de chats au réseau
- **Couche 1** apprend automatiquement les bords et les coins
- **Couche 2** combine les bords pour détecter des textures (poils, motifs)
- **Couche 3** assemble les textures en parties (oreilles, yeux, museau)
- **Couche 4** combine les parties pour reconnaître "un chat"
- Tout cela **sans que vous ayez à programmer explicitement** ces étapes !

Analogie : C'est comme si un enfant apprenait à reconnaître des concepts de plus en plus abstraits :
- Niveau 1 : Lignes et formes
- Niveau 2 : Combinaisons de formes
- Niveau 3 : Parties d'objets
- Niveau 4 : Objets complets
- Niveau 5 : Scènes et contextes

### 1.3.3 Exemple Concret

**Reconnaissance d'écriture manuscrite (MNIST)**

Imaginons que vous voulez créer un système qui lit automatiquement les chiffres écrits à la main (0-9) :

**Données** :
- 60 000 images de chiffres manuscrits (28×28 pixels en niveaux de gris)
- Chaque pixel a une valeur de 0 (blanc) à 255 (noir)

**Architecture du Réseau de Neurones** :
```
INPUT → CONV1 → POOL1 → CONV2 → POOL2 → FC1 → FC2 → OUTPUT

784 pixels     128        64        32       128      10
(28×28)      neurones  neurones  neurones neurones classes
                                                    (0-9)
```

**Ce que chaque couche apprend** :
1. **CONV1** (Convolution 1) : Détecte des bords simples
   - Filtres qui reconnaissent lignes verticales, horizontales, diagonales
   
2. **POOL1** (Pooling 1) : Réduit la taille en gardant l'essentiel
   - Conserve les informations importantes, ignore les détails
   
3. **CONV2** (Convolution 2) : Détecte des formes plus complexes
   - Coins, courbes, boucles caractéristiques des chiffres
   
4. **POOL2** (Pooling 2) : Nouvelle réduction dimensionnelle
   
5. **FC1** (Fully Connected 1) : Combine les caractéristiques
   - Apprend les combinaisons qui distinguent les chiffres
   
6. **FC2** (Fully Connected 2) : Décision finale
   - Produit 10 probabilités (une par chiffre)
   - Ex : [0.01, 0.02, 0.85, 0.05, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01]
   - → Le chiffre est probablement un "2" (85% de confiance)

**Résultats** :
- Précision : 99.2% sur le jeu de test
- Temps d'inférence : < 1ms par image
- Applications : Lecture automatique de chèques, codes postaux, formulaires

### 1.3.4 Exemples Métier

#### Finance : Prédiction de Séries Temporelles Financières

**Problématique** :
Prédire les mouvements de prix d'actions en analysant des années de données historiques complexes (prix, volumes, indicateurs techniques, news, sentiment sur réseaux sociaux).

**Solution Deep Learning : LSTM (Long Short-Term Memory)**

```python
# Architecture LSTM pour prédire le prix de demain

Données d'entrée (séquence de 60 jours) :
┌──────────────────────────────────────────┐
│ Jour 1  Jour 2  ...  Jour 59  Jour 60   │
│ [Prix, Volume, RSI, MACD, Sentiment]    │
└──────────────────────────────────────────┘
         │
         ↓
┌──────────────────────────┐
│   Couche LSTM (128)      │ ← Mémorise patterns à long terme
└──────────────────────────┘
         │
         ↓
┌──────────────────────────┐
│   Couche LSTM (64)       │ ← Raffine les patterns
└──────────────────────────┘
         │
         ↓
┌──────────────────────────┐
│   Couche Dense (32)      │ ← Combine les informations
└──────────────────────────┘
         │
         ↓
┌──────────────────────────┐
│   Sortie (1)             │ ← Prix prédit pour demain
└──────────────────────────┘
```

**Ce que le modèle apprend** :
- **Patterns temporels** : "Après une séquence de 5 jours de hausse modérée, correction probable"
- **Corrélations complexes** : "Volume élevé + sentiment négatif → baisse imminente"
- **Saisonnalité** : "Historiquement, baisse en septembre (fiscal year-end)"
- **Réaction aux news** : "Annonce de bénéfices supérieurs → hausse dans les 3 jours suivants"

**Performances** :
- RMSE (erreur) : 2.3% sur données de test
- Direction correcte prédite : 62% du temps (vs 50% au hasard)
- **Attention** : Pas de "boule de cristal" ! Les marchés restent imprévisibles
- Usage : Aide à la décision, pas trading automatique sans supervision

**Mise en garde** :
Les marchés financiers sont influencés par des facteurs externes imprévisibles (crises, guerres, décisions politiques). Le DL améliore les prédictions mais ne garantit jamais la performance.

#### Gestion : Analyse de Sentiment Client à partir d'Avis

**Problématique** :
Une entreprise e-commerce reçoit 10 000 avis clients par jour. Comment identifier rapidement les problèmes et les opportunités d'amélioration ?

**Solution Deep Learning : Transformers (BERT fine-tuné)**

```
INPUT : Texte de l'avis client
"Le produit est arrivé rapidement mais la qualité est décevante. 
Le service client a été excellent pour gérer le retour."

         ↓
┌────────────────────────────────────┐
│  TOKENIZATION                      │
│  [Le, produit, est, arrivé, ...]   │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│  EMBEDDINGS (contextuels)          │
│  Chaque mot → vecteur de 768 dim   │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│  BERT ENCODER (12 couches)         │
│  Comprend contexte et nuances      │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│  CLASSIFICATION HEAD               │
└────────────────────────────────────┘
         ↓
OUTPUT :
• Sentiment global : MITIGÉ (score: 0.52)
• Aspects identifiés :
  - Livraison : POSITIF (0.89)
  - Qualité produit : NÉGATIF (0.15)
  - Service client : POSITIF (0.93)
```

**Ce que le modèle comprend** :
- **Nuances** : "rapidement" (positif) et "décevante" (négatif) dans même phrase
- **Aspects multiples** : Distingue qualité produit vs service client
- **Contexte** : "excellent" s'applique au service, pas au produit
- **Sarcasme** (limité) : Détecte certaines formes d'ironie

**Résultats Business** :
- **Automatisation** : Classification de 95% des avis sans intervention humaine
- **Alertes automatiques** : Email au chef de produit si >10 avis négatifs sur qualité en 24h
- **Dashboard temps réel** : Sentiment par catégorie de produit, évolution temporelle
- **ROI** : Économie de 50 000€/an en analyse manuelle + amélioration produits

### 1.3.5 Schéma Explicatif : Architectures Deep Learning par Type de Données

```
┌──────────────────────────────────────────────────────────────────┐
│                    DEEP LEARNING ARCHITECTURES                   │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│   IMAGES / VISION   │  │  SÉQUENCES / TEMPS   │  │    TEXTE / NLP      │
└─────────────────────┘  └──────────────────────┘  └─────────────────────┘
          │                        │                         │
          ↓                        ↓                         ↓
    CNN (Réseaux de          RNN / LSTM / GRU          Transformers
     Convolution)           (Réseaux Récurrents)        (Attention)
          │                        │                         │
          ↓                        ↓                         ↓
┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│ COUCHES CONV        │  │ CELLULES MÉMOIRE     │  │ SELF-ATTENTION      │
│ • Filtres détectent │  │ • Mémorisent passé   │  │ • Relations entre   │
│   patterns locaux   │  │ • Gèrent dépendances │  │   mots              │
│ • Hiérarchie        │  │   temporelles        │  │ • Parallélisation   │
│   caractéristiques  │  │                      │  │   efficace          │
└─────────────────────┘  └──────────────────────┘  └─────────────────────┘
          │                        │                         │
          ↓                        ↓                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                       APPLICATIONS                              │
├──────────────────┬────────────────────┬─────────────────────────┤
│ • Reconnaissance │ • Prédiction prix  │ • Traduction           │
│   faciale        │ • Détection fraude │ • Chatbots             │
│ • Diagnostic     │   temporelle       │ • Analyse sentiment    │
│   médical        │ • Prévision ventes │ • Résumé texte         │
│ • Conduite auto  │ • Trading algo     │ • Q&A                  │
└──────────────────┴────────────────────┴─────────────────────────┘
```

**Explication des architectures** :

1. **CNN (Convolutional Neural Networks)** - Pour les images
   - **Principe** : Filtres glissants qui détectent des patterns locaux
   - **Couches** : Convolution → Pooling → Convolution → Pooling → Dense
   - **Force** : Invariance à la translation (détecte un chat peu importe sa position)
   - **Exemple** : ResNet (152 couches), utilisé par Facebook pour taguer automatiquement les photos

2. **RNN/LSTM** - Pour les séquences temporelles
   - **Principe** : Cellules avec mémoire qui traitent les données séquentiellement
   - **LSTM** : Amélioration du RNN, gère mieux les dépendances à long terme
   - **Force** : Comprend le contexte temporel
   - **Exemple** : Google Translate (avant Transformers)

3. **Transformers** - Pour le texte et au-delà
   - **Principe** : Mécanisme d'attention pour capter les relations entre tous les mots
   - **Force** : Parallélisable (plus rapide), gère mieux les longues séquences
   - **Exemples** : BERT (Google), GPT (OpenAI), LLaMA (Meta)

### 1.3.6 Cas Réel d'Entreprise : Tesla - Autopilot

**Contexte** :
Tesla développe un système de conduite autonome qui doit comprendre l'environnement routier en temps réel à partir de 8 caméras montées sur le véhicule.

**Défi Technique** :
- Traiter 8 flux vidéo simultanément (1280×960 pixels, 36 FPS)
- Identifier : voitures, piétons, cyclistes, panneaux, marquages au sol, feux de signalisation
- Prédire les trajectoires des autres véhicules
- Prendre des décisions en moins de 100ms
- Fonctionner dans toutes les conditions (jour, nuit, pluie, neige)

**Architecture Deep Learning** :

```
┌──────────────────────────────────────────────────────────────┐
│              8 CAMÉRAS → FLUX VIDÉO EN TEMPS RÉEL            │
└──────────────────────────────────────────────────────────────┘
                               │
                               ↓
┌──────────────────────────────────────────────────────────────┐
│          RÉSEAU DE NEURONES CONVOLUTIONNEL (CNN)             │
│                      HydraNet Architecture                    │
├──────────────────────────────────────────────────────────────┤
│  BACKBONE (tronc commun) - ResNet modifié                    │
│  • 50 couches de convolution                                 │
│  • Extrait caractéristiques visuelles                        │
└─────────────┬────────────────────────────────────────────────┘
              │
    ┌─────────┼─────────┬─────────────┬──────────────┐
    │         │         │             │              │
    ↓         ↓         ↓             ↓              ↓
┌────────┐ ┌──────┐ ┌─────────┐ ┌──────────┐ ┌─────────────┐
│DÉTECTION│ │SEGMEN│ │ ESTIMA- │ │PRÉDICTION│ │ DÉTECTION   │
│OBJETS   │ │TATION│ │ TION    │ │TRAJECTO  │ │ PROFONDEUR  │
│(Bounding│ │SÉMAN-│ │ PROFOND.│ │-IRES     │ │ (Distance)  │
│Boxes)   │ │TIQUE │ │         │ │          │ │             │
└────────┘ └──────┘ └─────────┘ └──────────┘ └─────────────┘
    │         │         │             │              │
    └─────────┴─────────┴─────────────┴──────────────┘
                        │
                        ↓
              ┌──────────────────────┐
              │  MODULE DE FUSION    │
              │  Combine toutes les  │
              │  informations        │
              └──────────┬───────────┘
                         │
                         ↓
              ┌──────────────────────┐
              │  PLANIFICATION       │
              │  DE TRAJECTOIRE      │
              └──────────┬───────────┘
                         │
                         ↓
              ┌──────────────────────┐
              │  COMMANDES VÉHICULE  │
              │  Accélération        │
              │  Freinage            │
              │  Direction           │
              └──────────────────────┘
```

**Données d'Entraînement** :
- **10 milliards de miles** parcourus par la flotte Tesla
- **1 milliard d'images** annotées (objets, distances, trajectoires)
- **Annotation semi-automatique** : Les modèles actuels aident à annoter de nouvelles données
- **Shadow Mode** : Les nouvelles versions tournent en arrière-plan pour collecter des données sans contrôler le véhicule

**Entraînement du Modèle** :
```
Matériel :
- 10 000 GPUs NVIDIA A100
- Supercalculateur Dojo (conçu par Tesla)
- 1 Exaflop de puissance de calcul

Temps d'entraînement :
- 2-3 semaines pour une version majeure
- Réentraînement continu avec nouvelles données

Optimisations :
- Quantification (réduction précision : FP32 → INT8)
- Pruning (suppression neurones inutiles)
- Distillation (transfert vers modèle plus petit)
→ Objectif : Tenir sur le hardware embarqué (chip FSD)
```

**Déploiement** :
```
Tesla FSD Chip (Hardware 3.0) :
- 144 TOPS (Trillions Operations Per Second)
- Consommation : 72 Watts
- Redondance : 2 chips indépendants (sécurité)
- Mise à jour OTA (Over-The-Air) du modèle

Pipeline d'inférence :
Images (8 caméras) → Preprocessing → Modèle CNN
                                       ↓
                                  Détections
                                       ↓
                                  Fusion sensorielle
                                       ↓
                                  Décisions
                                       ↓
                        Commandes véhicule (< 100ms)
```

**Résultats et Défis** :

✅ **Succès** :
- **10 fois moins d'accidents** qu'un conducteur humain moyen (selon Tesla)
- **Amélioration continue** : Chaque version (FSD 11, 12, 13...) plus performante
- **Généralisation** : Fonctionne dans des scenarios jamais vus en entraînement

⚠ **Défis Restants** :
- **Edge cases** : Situations rares (cône de chantier inhabituel, main d'un policier)
- **Conditions extrêmes** : Neige épaisse, brouillard dense
- **Responsabilité légale** : Qui est responsable en cas d'accident ?
- **Biais géographiques** : Performant aux USA, moins en Europe/Asie

**Enseignements pour l'IA en Entreprise** :
1. **Données = Carburant** : Plus de données → meilleur modèle
2. **Itération continue** : Déployer, collecter feedback, améliorer
3. **Infrastructu
---

<a name="chapitre-8-finance"></a>
# Chapitre 8 — IA en Finance

## 8.1 Introduction à l'IA en Finance

### 8.1.1 Vue d'Ensemble du Secteur

Le secteur financier est l'un des plus grands adopteurs de l'Intelligence Artificielle, avec des investissements dépassant les 35 milliards de dollars en 2024.

**Domaines d'Application Principaux** :
- **Trading Algorithmique** : Exécution automatisée d'ordres boursiers
- **Scoring de Crédit** : Évaluation automatique de la solvabilité
- **Détection de Fraude** : Identification de transactions suspectes en temps réel
- **Robo-Advisors** : Conseils d'investissement personnalisés automatisés
- **Conformité (KYC/AML)** : Automatisation des processus réglementaires

**ROI Typiques** :
- 35-40% de réduction des coûts opérationnels
- 25-30% d'amélioration de la détection de fraude
- 50-60% de réduction du temps de traitement des crédits

---

## 8.2 Cas d'Usage : Scoring de Crédit Intelligent

### 8.2.1 Problématique

Les banques doivent évaluer rapidement la solvabilité de milliers de demandeurs tout en :
- Minimisant les pertes (défauts de paiement)
- Maximisant l'inclusion financière
- Respectant les réglementations (RGPD, équité)

### 8.2.2 Solution IA

**Variables Utilisées** (100+) :
- Traditionnelles : revenus, dettes, historique crédit
- Alternatives : comportement transactionnel, digital footprint
- Dérivées : ratios financiers, scores de stabilité

**Modèles** :
- XGBoost / LightGBM pour la performance
- Random Forest pour l'interprétabilité
- Ensemble voting pour la robustesse

**Métriques Clés** :
- AUC-ROC > 0.80 (capacité de discrimination)
- Gini Coefficient > 0.50
- KS Statistic > 0.40
- Brier Score < 0.15 (calibration)

---

## 8.3 Cas d'Usage : Détection de Fraude en Temps Réel

### 8.3.1 Types de Fraudes

1. **Fraude à la carte** : Transactions non autorisées
2. **Prise de contrôle de compte** : Vol de credentials
3. **Blanchiment d'argent** : Structuring, layering
4. **Fraude à l'identité** : Identités synthétiques

### 8.3.2 Architecture Système

**Pipeline Temps Réel (< 100ms)** :

```
Transaction → Feature Engineering → ML Scoring → Décision
             (Redis cache)         (XGBoost +    (Approve/
                                    Isolation    Review/
                                    Forest)      Block)
```

**Features Temps Réel** :
- Historique 24h/1h de la carte
- Vélocité (transactions/heure)
- Déviations par rapport au comportement habituel
- Géolocalisation et distance parcourue
- Profil marchand (taux de fraude)

**Cas Réel : PayPal**
- 17 milliards de transactions/an
- Taux de détection : 97%
- Faux positifs : -50%
- Temps de traitement : < 100ms
- Économies : $700M/an

---

<a name="chapitre-9-audit"></a>
# Chapitre 9 — IA en Contrôle, Audit et Conseil

## 9.1 Introduction au Contrôle et Audit Augmenté par l'IA

### 9.1.1 Transformation du Métier

Le métier de l'audit et du contrôle connaît une révolution majeure grâce à l'IA :

**Avant l'IA** :
- Audit par échantillonnage (5-10% des transactions)
- Processus manuels et chronophages
- Détection a posteriori des anomalies
- Coût élevé des missions

**Avec l'IA** :
- Audit exhaustif (100% des données)
- Automatisation des contrôles répétitifs
- Détection proactive et prédictive
- Focus auditeur sur analyse à haute valeur

**Domaines d'Application** :
```
┌────────────────────────────────────────────────┐
│        IA EN CONTRÔLE ET AUDIT                 │
├────────────────────────────────────────────────┤
│ • Détection d'anomalies comptables            │
│ • Analyse de conformité automatisée           │
│ • Prédiction des risques                      │
│ • Automatisation de la revue documentaire     │
│ • Scoring de risque fournisseurs              │
│ • Détection de corruption et fraude interne   │
│ • Optimisation des processus de contrôle      │
└────────────────────────────────────────────────┘
```

---

## 9.2 Détection d'Anomalies Comptables

### 9.2.1 Loi de Benford et Détection de Fraude

**Principe** :
La Loi de Benford stipule que dans de nombreux ensembles de données naturelles, le premier chiffre suit une distribution logarithmique spécifique.

**Application** :
Détecter les manipulations comptables en comparant la distribution réelle vs attendue.

**Implémentation** :

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class BenfordAnomalyDetector:
    """
    Détection d'anomalies comptables via Loi de Benford
    """
    
    def __init__(self):
        # Distribution théorique de Benford (premier chiffre)
        self.benford_distribution = {
            1: 0.301,
            2: 0.176,
            3: 0.125,
            4: 0.097,
            5: 0.079,
            6: 0.067,
            7: 0.058,
            8: 0.051,
            9: 0.046
        }
    
    def get_first_digit(self, number):
        """Extraire le premier chiffre significatif"""
        str_num = str(abs(number)).replace('.', '').lstrip('0')
        if str_num:
            return int(str_num[0])
        return None
    
    def analyze_dataset(self, amounts, dataset_name="Dataset"):
        """
        Analyser un ensemble de montants
        """
        # Extraire premiers chiffres
        first_digits = [self.get_first_digit(amt) for amt in amounts]
        first_digits = [d for d in first_digits if d is not None]
        
        # Calculer distribution observée
        observed_dist = {}
        total = len(first_digits)
        for digit in range(1, 10):
            count = first_digits.count(digit)
            observed_dist[digit] = count / total if total > 0 else 0
        
        # Test Chi-Square
        expected_counts = [self.benford_distribution[d] * total for d in range(1, 10)]
        observed_counts = [first_digits.count(d) for d in range(1, 10)]
        
        chi2_stat, p_value = stats.chisquare(
            f_obs=observed_counts,
            f_exp=expected_counts
        )
        
        # Interprétation
        compliant = p_value > 0.05
        
        print(f"\n{'='*70}")
        print(f"ANALYSE LOI DE BENFORD - {dataset_name}")
        print(f"{'='*70}")
        print(f"\n📊 Échantillon : {total:,} transactions")
        print(f"\n🔬 Test Chi-Square")
        print(f"   Statistique : {chi2_stat:.4f}")
        print(f"   P-value     : {p_value:.4f}")
        print(f"   Résultat    : {'✅ CONFORME' if compliant else '🔴 SUSPECT'}")
        
        if not compliant:
            print(f"\n⚠️  ALERTE : Distribution non conforme à la loi de Benford")
            print(f"   → Possibles manipulations comptables")
            print(f"   → Investigation approfondie recommandée")
        
        # Afficher écarts par chiffre
        print(f"\n📈 Distribution par Premier Chiffre :\n")
        print(f"{'Chiffre':^10} | {'Benford':^10} | {'Observé':^10} | {'Écart':^10}")
        print(f"{'-'*50}")
        
        for digit in range(1, 10):
            benford_pct = self.benford_distribution[digit] * 100
            observed_pct = observed_dist[digit] * 100
            deviation = observed_pct - benford_pct
            
            flag = "🔴" if abs(deviation) > 5 else ""  # Écart > 5%
            print(f"{digit:^10} | {benford_pct:>9.1f}% | {observed_pct:>9.1f}% | {deviation:>+8.1f}% {flag}")
        
        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'compliant': compliant,
            'observed_distribution': observed_dist
        }

# EXEMPLE D'UTILISATION
# =====================

# Dataset 1 : Dépenses normales (conforme à Benford)
np.random.seed(42)
normal_expenses = np.random.lognormal(mean=7, sigma=1.5, size=1000)

# Dataset 2 : Dépenses manipulées (anomalies)
# Fraudeur arrondit souvent à 100, 200, 500, 1000
manipulated_expenses = np.concatenate([
    np.random.lognormal(mean=7, sigma=1.5, size=700),  # 70% normales
    np.array([100] * 50),   # Arrondis à 100
    np.array([200] * 50),   # Arrondis à 200
    np.array([500] * 100),  # Arrondis à 500
    np.array([1000] * 100)  # Arrondis à 1000
])

detector = BenfordAnomalyDetector()

# Analyser dataset normal
results_normal = detector.analyze_dataset(normal_expenses, "Dépenses Normales")

# Analyser dataset manipulé
results_fraud = detector.analyze_dataset(manipulated_expenses, "Dépenses Suspectes")
```

### 9.2.2 Machine Learning pour Anomalies Complexes

**Au-delà de Benford** :
Certaines fraudes sophistiquées respectent Benford mais présentent d'autres patterns anormaux.

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

class AdvancedAuditAnomalyDetector:
    """
    Détection d'anomalies multi-dimensionnelles
    """
    
    def __init__(self, contamination=0.05):
        self.contamination = contamination  # % attendu d'anomalies
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200
        )
        self.scaler = StandardScaler()
        
    def engineer_features(self, transactions_df):
        """
        Créer features pour détection d'anomalies
        """
        df = transactions_df.copy()
        
        # FEATURES TEMPORELLES
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # FEATURES MONTANT
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_rounded'] = (df['amount'] % 100 == 0).astype(int)
        df['amount_ends_in_99'] = (df['amount'] % 100 == 99).astype(int)
        
        # FEATURES COMPTE
        account_stats = df.groupby('account_id')['amount'].agg(['mean', 'std', 'count'])
        df = df.merge(
            account_stats.add_prefix('account_'),
            left_on='account_id',
            right_index=True
        )
        df['amount_vs_account_avg'] = df['amount'] / (df['account_mean'] + 1)
        
        # FEATURES BÉNÉFICIAIRE
        df['beneficiary_is_new'] = df['beneficiary_id'].map(
            df.groupby('beneficiary_id')['timestamp'].transform('min') == df['timestamp']
        ).astype(int)
        
        beneficiary_risk = df.groupby('beneficiary_id').size()
        df['beneficiary_frequency'] = df['beneficiary_id'].map(beneficiary_risk)
        
        # FEATURES DESCRIPTION
        df['description_length'] = df['description'].str.len()
        df['description_has_numbers'] = df['description'].str.contains(r'\d').astype(int)
        df['description_all_caps'] = (df['description'] == df['description'].str.upper()).astype(int)
        
        # FEATURES PATTERNS SUSPECTS
        df['split_payment_flag'] = (
            (df['amount'] > 9000) & (df['amount'] < 10000)
        ).astype(int)  # Juste sous seuil déclaration
        
        return df
    
    def detect_anomalies(self, transactions_df):
        """
        Détecter transactions anormales
        """
        # Feature engineering
        df_featured = self.engineer_features(transactions_df)
        
        # Sélectionner features numériques
        feature_cols = [
            'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'amount_rounded', 'amount_ends_in_99', 'account_count', 'amount_vs_account_avg',
            'beneficiary_is_new', 'beneficiary_frequency', 'description_length',
            'description_has_numbers', 'split_payment_flag'
        ]
        
        X = df_featured[feature_cols].fillna(0)
        
        # Normalisation
        X_scaled = self.scaler.fit_transform(X)
        
        # Détection anomalies
        predictions = self.model.fit_predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        # -1 = anomalie, 1 = normal
        df_featured['anomaly'] = predictions
        df_featured['anomaly_score'] = scores
        
        # Trier par score (plus négatif = plus anormal)
        anomalies = df_featured[df_featured['anomaly'] == -1].sort_values('anomaly_score')
        
        print(f"\n{'='*70}")
        print(f"DÉTECTION D'ANOMALIES - ISOLATION FOREST")
        print(f"{'='*70}")
        print(f"\n📊 Total transactions   : {len(df_featured):,}")
        print(f"🔴 Anomalies détectées  : {len(anomalies):,} ({len(anomalies)/len(df_featured)*100:.2f}%)")
        print(f"💰 Montant total suspect: {anomalies['amount'].sum():,.2f} €")
        
        print(f"\n🔍 Top 10 Transactions les Plus Suspectes :\n")
        
        for idx, (_, row) in enumerate(anomalies.head(10).iterrows(), 1):
            print(f"{idx}. Transaction #{row['transaction_id']}")
            print(f"   Montant        : {row['amount']:,.2f} €")
            print(f"   Compte         : {row['account_id']}")
            print(f"   Bénéficiaire   : {row['beneficiary_id']}")
            print(f"   Date/Heure     : {row['timestamp']}")
            print(f"   Score Anomalie : {row['anomaly_score']:.4f}")
            print(f"   Raisons probables:")
            
            reasons = []
            if row['is_night']:
                reasons.append("Transaction nocturne")
            if row['amount_rounded']:
                reasons.append("Montant arrondi (suspect)")
            if row['beneficiary_is_new']:
                reasons.append("Nouveau bénéficiaire")
            if row['split_payment_flag']:
                reasons.append("Montant juste sous seuil 10k€")
            if row['amount_vs_account_avg'] > 5:
                reasons.append(f"Montant {row['amount_vs_account_avg']:.1f}x supérieur à la moyenne du compte")
            
            for reason in reasons:
                print(f"      • {reason}")
            print()
        
        return anomalies

# Exemple d'utilisation
transactions = pd.DataFrame({
    'transaction_id': range(1000),
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
    'account_id': np.random.choice(['ACC001', 'ACC002', 'ACC003'], 1000),
    'beneficiary_id': np.random.choice([f'BEN{i:03d}' for i in range(50)], 1000),
    'amount': np.concatenate([
        np.random.lognormal(7, 1, 900),  # Transactions normales
        [9950] * 50,  # Suspect: juste sous 10k
        [500] * 50    # Suspect: arrondis exacts
    ]),
    'description': ['Payment' + str(i) for i in range(1000)]
})

detector = AdvancedAuditAnomalyDetector(contamination=0.10)
anomalies_detected = detector.detect_anomalies(transactions)
```

---

## 9.3 Automatisation de la Revue Documentaire

### 9.3.1 Extraction et Classification de Documents

**Problématique** :
Les auditeurs doivent passer en revue des milliers de documents (factures, contrats, relevés) pour vérifier conformité et détecter anomalies.

**Solution IA** :

```python
from transformers import pipeline
import pytesseract
from PIL import Image
import pdf2image

class DocumentReviewAutomation:
    """
    Automatisation de la revue documentaire avec NLP
    """
    
    def __init__(self):
        # Modèle NLP pour classification
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Catégories de documents
        self.document_categories = [
            "Facture",
            "Contrat",
            "Relevé bancaire",
            "Note de frais",
            "Bon de commande",
            "Rapport financier",
            "Justificatif"
        ]
        
    def extract_text_from_pdf(self, pdf_path):
        """
        Extraire texte d'un PDF (avec OCR si nécessaire)
        """
        try:
            # Convertir PDF en images
            images = pdf2image.convert_from_path(pdf_path)
            
            # OCR sur chaque page
            full_text = ""
            for img in images:
                text = pytesseract.image_to_string(img, lang='fra')
                full_text += text + "\n"
            
            return full_text
        except Exception as e:
            print(f"Erreur extraction : {e}")
            return ""
    
    def classify_document(self, text):
        """
        Classifier le type de document
        """
        result = self.classifier(
            text[:500],  # Premier 500 caractères
            self.document_categories
        )
        
        return {
            'category': result['labels'][0],
            'confidence': result['scores'][0]
        }
    
    def extract_key_info(self, text, doc_type):
        """
        Extraire informations clés selon type de document
        """
        import re
        
        info = {}
        
        if doc_type == "Facture":
            # Numéro de facture
            facture_match = re.search(r'(?:Facture|Invoice)\s*(?:N°|#|No\.?)\s*:?\s*(\w+)', text, re.I)
            if facture_match:
                info['numero_facture'] = facture_match.group(1)
            
            # Montant TTC
            montant_match = re.search(r'(?:Total|Montant)\s*TTC\s*:?\s*([\d\s,\.]+)\s*€', text, re.I)
            if montant_match:
                montant_str = montant_match.group(1).replace(' ', '').replace(',', '.')
                info['montant_ttc'] = float(montant_str)
            
            # Date
            date_match = re.search(r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', text)
            if date_match:
                info['date'] = date_match.group(1)
        
        elif doc_type == "Contrat":
            # Parties contractuelles
            parties_match = re.findall(r'(?:Entre|Party)\s+(.+?)(?:\set\s|\sand\s)', text, re.I)
            if parties_match:
                info['parties'] = parties_match
            
            # Montant contractuel
            montant_match = re.search(r'(?:montant|amount)\s+(?:de|of)\s+([\d\s,\.]+)', text, re.I)
            if montant_match:
                info['montant'] = montant_match.group(1)
        
        return info
    
    def check_compliance(self, doc_info, doc_type):
        """
        Vérifier conformité du document
        """
        issues = []
        
        if doc_type == "Facture":
            # Vérifications obligatoires
            if 'numero_facture' not in doc_info:
                issues.append({
                    'severity': 'HIGH',
                    'issue': 'Numéro de facture manquant'
                })
            
            if 'montant_ttc' not in doc_info:
                issues.append({
                    'severity': 'HIGH',
                    'issue': 'Montant TTC non identifié'
                })
            
            if 'date' not in doc_info:
                issues.append({
                    'severity': 'MEDIUM',
                    'issue': 'Date de facture manquante'
                })
            
            # Vérification montant suspect
            if 'montant_ttc' in doc_info:
                if doc_info['montant_ttc'] > 50000:
                    issues.append({
                        'severity': 'MEDIUM',
                        'issue': f"Montant élevé : {doc_info['montant_ttc']:,.2f} € (>50k)"
                    })
                
                # Montant arrondi suspect
                if doc_info['montant_ttc'] % 1000 == 0:
                    issues.append({
                        'severity': 'LOW',
                        'issue': 'Montant arrondi à 1000€ (vérifier authenticité)'
                    })
        
        return issues
    
    def review_document(self, pdf_path):
        """
        Pipeline complet de revue
        """
        print(f"\n{'='*70}")
        print(f"REVUE AUTOMATISÉE - {pdf_path}")
        print(f"{'='*70}")
        
        # 1. Extraction texte
        print(f"\n1️⃣  Extraction du texte...")
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            print("   ❌ Impossible d'extraire le texte")
            return None
        
        print(f"   ✅ {len(text)} caractères extraits")
        
        # 2. Classification
        print(f"\n2️⃣  Classification du document...")
        classification = self.classify_document(text)
        doc_type = classification['category']
        confidence = classification['confidence']
        
        print(f"   Type détecté : {doc_type} (confiance: {confidence:.2%})")
        
        # 3. Extraction informations
        print(f"\n3️⃣  Extraction des informations clés...")
        doc_info = self.extract_key_info(text, doc_type)
        
        for key, value in doc_info.items():
            print(f"   • {key:20s}: {value}")
        
        # 4. Vérification conformité
        print(f"\n4️⃣  Vérification de conformité...")
        issues = self.check_compliance(doc_info, doc_type)
        
        if not issues:
            print(f"   ✅ Aucun problème détecté")
        else:
            print(f"   ⚠️  {len(issues)} problème(s) détecté(s):")
            for issue in issues:
                severity_icon = {
                    'HIGH': '🔴',
                    'MEDIUM': '🟠',
                    'LOW': '🟡'
                }
                print(f"   {severity_icon[issue['severity']]} {issue['issue']}")
        
        return {
            'doc_type': doc_type,
            'confidence': confidence,
            'extracted_info': doc_info,
            'compliance_issues': issues
        }

# Utilisation
reviewer = DocumentReviewAutomation()
# result = reviewer.review_document('facture_001.pdf')
```

---

## 9.4 Scoring de Risque Fournisseurs

### 9.4.1 Évaluation Multi-Critères

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class SupplierRiskScoring:
    """
    Scoring de risque fournisseurs basé sur multiple critères
    """
    
    def __init__(self):
        self.model = None
        self.risk_categories = {
            'financial': 0.30,      # 30% du score
            'operational': 0.25,    # 25%
            'compliance': 0.25,     # 25%
            'reputation': 0.20      # 20%
        }
    
    def calculate_financial_risk(self, supplier_data):
        """
        Risque financier : solvabilité, santé financière
        """
        score = 0
        
        # Ratio dette/equity
        if supplier_data.get('debt_to_equity', 0) > 2:
            score += 30
        elif supplier_data.get('debt_to_equity', 0) > 1:
            score += 15
        
        # Liquidité
        current_ratio = supplier_data.get('current_ratio', 1)
        if current_ratio < 1:
            score += 25
        elif current_ratio < 1.5:
            score += 10
        
        # Rentabilité
        if supplier_data.get('profit_margin', 0) < 0:
            score += 30  # Pertes
        elif supplier_data.get('profit_margin', 0) < 0.05:
            score += 15  # Faible marge
        
        # Historique paiements
        late_payments = supplier_data.get('late_payment_incidents', 0)
        if late_payments > 5:
            score += 20
        elif late_payments > 2:
            score += 10
        
        return min(score, 100)  # Cap à 100
    
    def calculate_operational_risk(self, supplier_data):
        """
        Risque opérationnel : capacité de livraison, qualité
        """
        score = 0
        
        # Taux de retard livraison
        delivery_rate = supplier_data.get('on_time_delivery_rate', 100)
        if delivery_rate < 80:
            score += 30
        elif delivery_rate < 90:
            score += 15
        
        # Taux de défauts qualité
        defect_rate = supplier_data.get('defect_rate', 0)
        if defect_rate > 5:
            score += 25
        elif defect_rate > 2:
            score += 10
        
        # Dépendance (% CA du fournisseur représenté par notre entreprise)
        dependency = supplier_data.get('revenue_dependency_pct', 0)
        if dependency > 50:
            score += 20  # Trop dépendant
        elif dependency > 30:
            score += 10
        
        # Capacité de production
        utilization = supplier_data.get('capacity_utilization', 50)
        if utilization > 95:
            score += 15  # Surutilisation = risque
        
        return min(score, 100)
    
    def calculate_compliance_risk(self, supplier_data):
        """
        Risque conformité : certifications, réglementation
        """
        score = 0
        
        # Certifications manquantes
        required_certs = ['ISO9001', 'ISO14001']
        supplier_certs = supplier_data.get('certifications', [])
        
        missing_certs = set(required_certs) - set(supplier_certs)
        score += len(missing_certs) * 20
        
        # Incidents de non-conformité
        incidents = supplier_data.get('compliance_incidents', 0)
        if incidents > 3:
            score += 40
        elif incidents > 1:
            score += 20
        
        # Audit score
        audit_score = supplier_data.get('last_audit_score', 100)
        if audit_score < 60:
            score += 30
        elif audit_score < 80:
            score += 15
        
        # Pays à risque
        if supplier_data.get('country') in ['Country1', 'Country2']:  # High-risk countries
            score += 25
        
        return min(score, 100)
    
    def calculate_reputation_risk(self, supplier_data):
        """
        Risque réputation : médias, litiges, ESG
        """
        score = 0
        
        # Litiges en cours
        lawsuits = supplier_data.get('active_lawsuits', 0)
        if lawsuits > 2:
            score += 30
        elif lawsuits > 0:
            score += 15
        
        # Score ESG
        esg_score = supplier_data.get('esg_score', 50)
        if esg_score < 30:
            score += 25
        elif esg_score < 50:
            score += 10
        
        # Couverture médiatique négative
        negative_news = supplier_data.get('negative_news_count_12m', 0)
        if negative_news > 5:
            score += 20
        elif negative_news > 2:
            score += 10
        
        return min(score, 100)
    
    def calculate_overall_risk(self, supplier_data):
        """
        Score de risque global pondéré
        """
        risks = {
            'financial': self.calculate_financial_risk(supplier_data),
            'operational': self.calculate_operational_risk(supplier_data),
            'compliance': self.calculate_compliance_risk(supplier_data),
            'reputation': self.calculate_reputation_risk(supplier_data)
        }
        
        # Score pondéré
        overall_score = sum(
            risks[category] * weight 
            for category, weight in self.risk_categories.items()
        )
        
        # Classification
        if overall_score < 30:
            risk_level = 'LOW'
            recommendation = 'Approved - Standard monitoring'
        elif overall_score < 60:
            risk_level = 'MEDIUM'
            recommendation = 'Approved with conditions - Enhanced monitoring'
        elif overall_score < 80:
            risk_level = 'HIGH'
            recommendation = 'Conditional approval - Mitigation plan required'
        else:
            risk_level = 'CRITICAL'
            recommendation = 'Not recommended - Find alternative supplier'
        
        return {
            'overall_score': overall_score,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'category_scores': risks
        }
    
    def generate_report(self, supplier_name, supplier_data):
        """
        Générer rapport d'évaluation
        """
        result = self.calculate_overall_risk(supplier_data)
        
        print(f"\n{'='*70}")
        print(f"RAPPORT D'ÉVALUATION FOURNISSEUR - {supplier_name}")
        print(f"{'='*70}")
        
        print(f"\n📊 SCORE DE RISQUE GLOBAL : {result['overall_score']:.1f}/100")
        print(f"🎯 NIVEAU DE RISQUE      : {result['risk_level']}")
        print(f"💡 RECOMMANDATION        : {result['recommendation']}")
        
        print(f"\n📈 DÉTAIL PAR CATÉGORIE :\n")
        for category, score in result['category_scores'].items():
            weight = self.risk_categories[category]
            weighted_contribution = score * weight
            
            status = '✅' if score < 30 else '🟠' if score < 60 else '🔴'
            print(f"   {status} {category.upper():15s}: {score:>5.1f}/100 (poids: {weight:.0%}) → Contribution: {weighted_contribution:.1f}")
        
        print(f"\n🔍 ACTIONS RECOMMANDÉES :")
        
        actions = []
        if result['category_scores']['financial'] > 60:
            actions.append("• Demander garanties financières supplémentaires")
            actions.append("• Établir plan de paiement sécurisé")
        
        if result['category_scores']['operational'] > 60:
            actions.append("• Auditer capacité de production")
            actions.append("• Identifier fournisseurs alternatifs (backup)")
        
        if result['category_scores']['compliance'] > 60:
            actions.append("• Exiger mise en conformité sous 90 jours")
            actions.append("• Planifier audit sur site")
        
        if result['category_scores']['reputation'] > 60:
            actions.append("• Due diligence approfondie")
            actions.append("• Clauses de protection réputation dans contrat")
        
        if not actions:
            print("   ✅ Aucune action particulière requise")
        else:
            for action in actions:
                print(f"   {action}")
        
        return result

# Exemple d'utilisation
scorer = SupplierRiskScoring()

supplier_example = {
    'name': 'TechSupply Corp',
    # Financier
    'debt_to_equity': 1.8,
    'current_ratio': 1.2,
    'profit_margin': 0.08,
    'late_payment_incidents': 1,
    # Opérationnel
    'on_time_delivery_rate': 92,
    'defect_rate': 1.5,
    'revenue_dependency_pct': 25,
    'capacity_utilization': 75,
    # Conformité
    'certifications': ['ISO9001'],
    'compliance_incidents': 0,
    'last_audit_score': 85,
    'country': 'FR',
    # Réputation
    'active_lawsuits': 0,
    'esg_score': 65,
    'negative_news_count_12m': 1
}

report = scorer.generate_report('TechSupply Corp', supplier_example)
```

---

## 9.5 Cas Réels d'Implémentation

### 9.5.1 Deloitte - Audit Analytics Platform

**Solution** :
Plateforme d'audit augmentée analysant 100% des transactions vs échantillonnage traditionnel.

**Technologies** :
- NLP pour analyse documentaire
- Anomaly detection (Isolation Forest)
- Network analysis pour détecter schémas de fraude
- Visualization interactive (Power BI + Python)

**Résultats** :
- **Couverture** : 100% vs 5-10% traditionnel
- **Temps d'audit** : -40%
- **Anomalies détectées** : +200%
- **Faux positifs** : -60% (ML apprend des retours auditeurs)

### 9.5.2 EY - Canvas Tax Analyzer

**Problématique** :
Vérifier conformité fiscale sur millions de transactions.

**Solution IA** :
- Extraction automatique données fiscales (OCR + NLP)
- Validation règles fiscales par juridiction
- Calcul automatique risques/opportunités
- Génération rapports conformité

**Impact** :
- **Temps de revue** : Semaines → Heures
- **Précision** : +35%
- **Économies fiscales identifiées** : +$50M pour clients


---

<a name="chapitre-7-marketing"></a>
# Chapitre 7 — IA en Marketing

## 7.1 Introduction à l'IA Marketing

### 7.1.1 Transformation du Marketing par l'IA

Le marketing connaît une révolution grâce à l'IA, passant d'une approche de masse à une hyper-personnalisation à grande échelle.

**Évolution du Marketing** :

```
┌────────────────────────────────────────────────────────────┐
│         MARKETING TRADITIONNEL vs IA MARKETING            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  AVANT (Marketing de Masse)        MAINTENANT (IA)        │
│  ─────────────────────────         ──────────────────     │
│  • Segmentation large              • Segments de 1        │
│  • Message unique                  • Personnalisé 1:1     │
│  • Campagnes statiques             • Optimisation temps   │
│  • Décisions intuition             • Décisions data       │
│  • ROI difficile à mesurer         • Attribution précise  │
│  • Réactivité lente                • Temps réel          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Domaines d'Application** :
1. **Segmentation Client** : Clustering avancé, micro-segments
2. **Prédiction du Churn** : Identifier clients à risque
3. **Recommandation** : Produits, contenu, next-best-action
4. **Optimisation Campagnes** : Attribution, budget allocation
5. **Génération de Contenu** : Textes, images, vidéos
6. **Pricing Dynamique** : Prix optimisés en temps réel
7. **Analyse Sentiment** : Social listening, avis clients

---

## 7.2 Segmentation Client Avancée

### 7.2.1 Au-delà de la Segmentation Traditionnelle

**Segmentation Traditionnelle** (limitée) :
- Démographique : âge, sexe, revenu
- Géographique : région, ville
- Psychographique : style de vie, valeurs

**Segmentation IA** (puissante) :
- Comportementale : navigation, achats, engagement
- Prédictive : propension, valeur vie client (CLV)
- Temporelle : moments de vie, saisonnalité
- Multi-dimensionnelle : 100+ variables

### 7.2.2 Implémentation RFM + Machine Learning

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class CustomerSegmentation:
    """
    Segmentation client avancée : RFM + Comportemental + ML
    """
    
    def __init__(self, n_segments=5):
        self.n_segments = n_segments
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        self.pca = PCA(n_components=2)
        
    def calculate_rfm(self, transactions_df, reference_date=None):
        """
        Calcul des scores RFM (Recency, Frequency, Monetary)
        """
        if reference_date is None:
            reference_date = transactions_df['date'].max()
        
        rfm = transactions_df.groupby('customer_id').agg({
            'date': lambda x: (reference_date - x.max()).days,  # Recency
            'order_id': 'count',                                 # Frequency
            'amount': 'sum'                                      # Monetary
        })
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        
        # Scores RFM (1-5, 5 = meilleur)
        rfm['R_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['F_score'] = pd.qcut(rfm['frequency'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['M_score'] = pd.qcut(rfm['monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Score RFM combiné
        rfm['RFM_score'] = (
            rfm['R_score'].astype(int) * 100 + 
            rfm['F_score'].astype(int) * 10 + 
            rfm['M_score'].astype(int)
        )
        
        return rfm
    
    def calculate_behavioral_features(self, transactions_df, products_df=None):
        """
        Features comportementales avancées
        """
        features = {}
        
        # Groupe par client
        grouped = transactions_df.groupby('customer_id')
        
        # FRÉQUENCE D'ACHAT
        features['avg_days_between_orders'] = grouped['date'].apply(
            lambda x: x.diff().dt.days.mean() if len(x) > 1 else 999
        )
        
        # PANIER MOYEN
        features['avg_basket_size'] = grouped['amount'].mean()
        features['std_basket_size'] = grouped['amount'].std().fillna(0)
        
        # DIVERSITÉ PRODUITS
        if 'product_id' in transactions_df.columns:
            features['num_unique_products'] = grouped['product_id'].nunique()
            features['product_diversity_ratio'] = (
                features['num_unique_products'] / grouped.size()
            )
        
        # CATÉGORIES PRÉFÉRÉES
        if 'category' in transactions_df.columns:
            features['num_categories'] = grouped['category'].nunique()
            
            # Catégorie dominante (mode)
            features['dominant_category'] = grouped['category'].apply(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
            )
        
        # CANAL D'ACHAT
        if 'channel' in transactions_df.columns:
            features['pct_online'] = grouped.apply(
                lambda x: (x['channel'] == 'online').sum() / len(x)
            )
        
        # TEMPORALITÉ
        transactions_df['hour'] = pd.to_datetime(transactions_df['date']).dt.hour
        transactions_df['day_of_week'] = pd.to_datetime(transactions_df['date']).dt.dayofweek
        
        features['preferred_hour'] = grouped['hour'].apply(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 12
        )
        features['pct_weekend_orders'] = grouped.apply(
            lambda x: (x['day_of_week'] >= 5).sum() / len(x)
        )
        
        # PROMOTIONS
        if 'used_promo' in transactions_df.columns:
            features['promo_sensitivity'] = grouped.apply(
                lambda x: (x['used_promo'] == True).sum() / len(x)
            )
        
        # RETOURS
        if 'returned' in transactions_df.columns:
            features['return_rate'] = grouped.apply(
                lambda x: (x['returned'] == True).sum() / len(x)
            )
        
        # VALEUR VIE CLIENT (CLV simplifié)
        features['total_spent'] = grouped['amount'].sum()
        features['estimated_clv'] = (
            features['avg_basket_size'] * 
            (365 / features['avg_days_between_orders'].replace([np.inf, 0], 999))
        )
        
        # Convertir en DataFrame
        features_df = pd.DataFrame(features)
        
        return features_df
    
    def segment_customers(self, rfm_df, behavioral_df):
        """
        Segmentation ML (K-Means)
        """
        # Combiner RFM et comportemental
        features_df = rfm_df[['recency', 'frequency', 'monetary']].join(
            behavioral_df, how='inner'
        )
        
        # Gérer valeurs manquantes
        features_df = features_df.fillna(features_df.median())
        
        # Sélectionner features numériques
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_cols]
        
        # Standardisation (crucial pour K-Means)
        X_scaled = self.scaler.fit_transform(X)
        
        # Clustering
        clusters = self.kmeans.fit_predict(X_scaled)
        features_df['segment'] = clusters
        
        # PCA pour visualisation
        X_pca = self.pca.fit_transform(X_scaled)
        features_df['pca1'] = X_pca[:, 0]
        features_df['pca2'] = X_pca[:, 1]
        
        return features_df
    
    def profile_segments(self, segmented_df):
        """
        Profiler chaque segment
        """
        profiles = []
        
        for segment_id in range(self.n_segments):
            segment_data = segmented_df[segmented_df['segment'] == segment_id]
            
            profile = {
                'segment_id': segment_id,
                'size': len(segment_data),
                'size_pct': len(segment_data) / len(segmented_df) * 100,
                
                # RFM
                'avg_recency': segment_data['recency'].mean(),
                'avg_frequency': segment_data['frequency'].mean(),
                'avg_monetary': segment_data['monetary'].mean(),
                
                # CLV
                'avg_clv': segment_data['estimated_clv'].mean(),
                'total_value': segment_data['total_spent'].sum(),
                
                # Comportement
                'avg_basket': segment_data['avg_basket_size'].mean(),
                'avg_days_between': segment_data['avg_days_between_orders'].mean(),
            }
            
            # Ajouter features spécifiques si disponibles
            if 'promo_sensitivity' in segment_data.columns:
                profile['promo_sensitivity'] = segment_data['promo_sensitivity'].mean()
            
            if 'return_rate' in segment_data.columns:
                profile['return_rate'] = segment_data['return_rate'].mean()
            
            if 'pct_online' in segment_data.columns:
                profile['pct_online'] = segment_data['pct_online'].mean()
            
            profiles.append(profile)
        
        profiles_df = pd.DataFrame(profiles)
        
        return profiles_df
    
    def name_segments(self, profiles_df):
        """
        Nommer les segments de manière descriptive
        """
        profiles_df['segment_name'] = 'Unknown'
        
        for idx, row in profiles_df.iterrows():
            # Champions : Recency faible, Frequency élevée, Monetary élevé
            if (row['avg_recency'] < 30 and 
                row['avg_frequency'] > profiles_df['avg_frequency'].median() and
                row['avg_monetary'] > profiles_df['avg_monetary'].median()):
                profiles_df.at[idx, 'segment_name'] = '🏆 Champions'
            
            # Loyal : Frequency élevée mais Monetary moyen
            elif (row['avg_frequency'] > profiles_df['avg_frequency'].median() and
                  row['avg_monetary'] <= profiles_df['avg_monetary'].median()):
                profiles_df.at[idx, 'segment_name'] = '💙 Loyaux'
            
            # Big Spenders : Monetary élevé mais Frequency faible
            elif (row['avg_monetary'] > profiles_df['avg_monetary'].median() and
                  row['avg_frequency'] <= profiles_df['avg_frequency'].median()):
                profiles_df.at[idx, 'segment_name'] = '💰 Gros Dépensiers'
            
            # At Risk : Recency élevée mais bon historique
            elif (row['avg_recency'] > 90 and
                  row['avg_monetary'] > profiles_df['avg_monetary'].quantile(0.3)):
                profiles_df.at[idx, 'segment_name'] = '⚠️  À Risque'
            
            # Lost : Recency très élevée
            elif row['avg_recency'] > 180:
                profiles_df.at[idx, 'segment_name'] = '😢 Perdus'
            
            # New/Promising : Recency faible mais peu de transactions
            elif (row['avg_recency'] < 30 and
                  row['avg_frequency'] <= 2):
                profiles_df.at[idx, 'segment_name'] = '🌱 Nouveaux/Prometteurs'
            
            # Need Attention : Autres
            else:
                profiles_df.at[idx, 'segment_name'] = '🔔 Nécessitent Attention'
        
        return profiles_df
    
    def recommend_actions(self, profiles_df):
        """
        Recommander actions marketing par segment
        """
        actions = {
            '🏆 Champions': [
                "Programme VIP exclusif",
                "Early access nouveaux produits",
                "Ambassadeurs de marque (UGC, referral)",
                "Rewards programme premium"
            ],
            '💙 Loyaux': [
                "Programme de fidélité renforcé",
                "Upsell produits premium",
                "Cross-sell catégories adjacentes",
                "Satisfaction surveys pour amélioration"
            ],
            '💰 Gros Dépensiers': [
                "Concierge service personnalisé",
                "Offres exclusives haut de gamme",
                "Événements VIP",
                "Programme de rétention avec incentives"
            ],
            '⚠️  À Risque': [
                "Campagne de réactivation urgente",
                "Offre spéciale 'We miss you' (20-30% off)",
                "Enquête satisfaction + service recovery",
                "Rappel avantages programme fidélité"
            ],
            '😢 Perdus': [
                "Campagne win-back agressive (40-50% off)",
                "Enquête départ (pourquoi nous ont quitté)",
                "Nouvelle proposition de valeur",
                "Si pas de réponse → Retirer de la liste active"
            ],
            '🌱 Nouveaux/Prometteurs': [
                "Onboarding optimisé",
                "2ème achat incentivé (offre bienvenue)",
                "Tutoriels produits",
                "Programme de parrainage"
            ],
            '🔔 Nécessitent Attention': [
                "Campagne d'engagement",
                "Contenu éducatif",
                "Promotions ciblées",
                "A/B testing différentes approches"
            ]
        }
        
        profiles_df['recommended_actions'] = profiles_df['segment_name'].map(
            lambda x: actions.get(x, ["Stratégie à définir"])
        )
        
        return profiles_df
    
    def visualize_segments(self, segmented_df, profiles_df):
        """
        Visualisation des segments
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Scatter PCA
        scatter = axes[0, 0].scatter(
            segmented_df['pca1'], 
            segmented_df['pca2'],
            c=segmented_df['segment'],
            cmap='viridis',
            alpha=0.6
        )
        axes[0, 0].set_xlabel('PCA Component 1')
        axes[0, 0].set_ylabel('PCA Component 2')
        axes[0, 0].set_title('Segmentation Clients (PCA)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Taille des segments
        profiles_df.plot.bar(
            x='segment_name',
            y='size',
            ax=axes[0, 1],
            legend=False,
            color='steelblue'
        )
        axes[0, 1].set_title('Taille des Segments')
        axes[0, 1].set_xlabel('Segment')
        axes[0, 1].set_ylabel('Nombre de Clients')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. CLV par segment
        profiles_df.plot.bar(
            x='segment_name',
            y='avg_clv',
            ax=axes[1, 0],
            legend=False,
            color='green'
        )
        axes[1, 0].set_title('CLV Moyen par Segment')
        axes[1, 0].set_xlabel('Segment')
        axes[1, 0].set_ylabel('CLV (€)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. RFM Heatmap
        rfm_data = profiles_df[['segment_name', 'avg_recency', 'avg_frequency', 'avg_monetary']].set_index('segment_name')
        sns.heatmap(
            rfm_data.T,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn_r',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('RFM Heatmap par Segment')
        
        plt.tight_layout()
        # plt.savefig('customer_segmentation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, profiles_df):
        """
        Rapport exécutif de segmentation
        """
        print("\n" + "="*80)
        print("RAPPORT DE SEGMENTATION CLIENT")
        print("="*80)
        
        print(f"\n📊 VUE D'ENSEMBLE")
        print(f"   Nombre de segments : {len(profiles_df)}")
        print(f"   Total clients      : {profiles_df['size'].sum():,}")
        
        print(f"\n💰 VALEUR TOTALE")
        print(f"   Revenue total      : {profiles_df['total_value'].sum():,.0f} €")
        print(f"   CLV moyen global   : {profiles_df['avg_clv'].mean():,.0f} €")
        
        print(f"\n🎯 PROFIL DES SEGMENTS\n")
        
        # Trier par CLV décroissant
        profiles_sorted = profiles_df.sort_values('avg_clv', ascending=False)
        
        for idx, row in profiles_sorted.iterrows():
            print(f"{row['segment_name']}")
            print(f"{'─'*70}")
            print(f"  📈 Taille        : {row['size']:,} clients ({row['size_pct']:.1f}%)")
            print(f"  💵 Valeur totale : {row['total_value']:,.0f} €")
            print(f"  💎 CLV moyen     : {row['avg_clv']:,.0f} €")
            print(f"  🔄 Récence       : {row['avg_recency']:.0f} jours")
            print(f"  🛒 Fréquence     : {row['avg_frequency']:.1f} commandes")
            print(f"  💰 Panier moyen  : {row['avg_basket']:.0f} €")
            
            if 'promo_sensitivity' in row:
                print(f"  🏷️  Promo-sensible: {row['promo_sensitivity']:.1%}")
            
            print(f"\n  ✅ ACTIONS RECOMMANDÉES :")
            for action in row['recommended_actions']:
                print(f"     • {action}")
            print()

# UTILISATION COMPLÈTE
# =====================

# Charger données
transactions = pd.read_csv('transactions.csv')
# transactions columns: customer_id, date, order_id, amount, product_id, category, channel, used_promo, returned

# Initialiser
segmenter = CustomerSegmentation(n_segments=5)

# 1. Calculer RFM
rfm = segmenter.calculate_rfm(transactions)

# 2. Calculer features comportementales
behavioral = segmenter.calculate_behavioral_features(transactions)

# 3. Segmentation
segmented = segmenter.segment_customers(rfm, behavioral)

# 4. Profiler segments
profiles = segmenter.profile_segments(segmented)

# 5. Nommer segments
profiles = segmenter.name_segments(profiles)

# 6. Recommander actions
profiles = segmenter.recommend_actions(profiles)

# 7. Visualiser
segmenter.visualize_segments(segmented, profiles)

# 8. Rapport
segmenter.generate_report(profiles)

# 9. Exporter résultats
segmented.to_csv('customers_segmented.csv', index=True)
profiles.to_csv('segments_profiles.csv', index=False)
```

---

## 7.3 Prédiction du Churn (Attrition Client)

### 7.3.1 Définition et Enjeux

**Churn** = Taux d'attrition = % de clients qui cessent d'utiliser un produit/service

**Enjeux Business** :
- Coût acquisition client : 5-25× plus cher que rétention
- Impact revenue : 5% réduction churn = 25-95% augmentation profits
- LTV (Lifetime Value) : Churn tue la valeur long terme

### 7.3.2 Modèle Prédictif de Churn

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import shap

class ChurnPredictionModel:
    """
    Modèle de prédiction du churn client
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.shap_explainer = None
        
    def engineer_churn_features(self, customers_df, transactions_df, support_df=None):
        """
        Feature engineering pour prédiction churn
        """
        features = pd.DataFrame(index=customers_df['customer_id'])
        
        # FEATURES DÉMOGRAPHIQUES
        features['age'] = customers_df['age']
        features['tenure_months'] = (
            (pd.to_datetime('today') - pd.to_datetime(customers_df['signup_date']))
            .dt.days / 30
        )
        
        # FEATURES TRANSACTIONNELLES (30 derniers jours)
        recent_date = transactions_df['date'].max() - timedelta(days=30)
        recent_tx = transactions_df[transactions_df['date'] >= recent_date]
        
        tx_grouped = recent_tx.groupby('customer_id')
        features['num_transactions_30d'] = tx_grouped.size().reindex(features.index, fill_value=0)
        features['total_spent_30d'] = tx_grouped['amount'].sum().reindex(features.index, fill_value=0)
        features['avg_basket_30d'] = tx_grouped['amount'].mean().reindex(features.index, fill_value=0)
        
        # FEATURES TEMPORELLES (tendances)
        # Comparer 30 derniers jours vs 30-60 jours
        prev_period = transactions_df[
            (transactions_df['date'] >= recent_date - timedelta(days=30)) &
            (transactions_df['date'] < recent_date)
        ]
        
        prev_grouped = prev_period.groupby('customer_id')
        features['num_transactions_prev30d'] = prev_grouped.size().reindex(features.index, fill_value=0)
        features['total_spent_prev30d'] = prev_grouped['amount'].sum().reindex(features.index, fill_value=0)
        
        # Tendance (variation %)
        features['transaction_trend'] = (
            (features['num_transactions_30d'] - features['num_transactions_prev30d']) /
            (features['num_transactions_prev30d'] + 1) * 100
        )
        features['spending_trend'] = (
            (features['total_spent_30d'] - features['total_spent_prev30d']) /
            (features['total_spent_prev30d'] + 1) * 100
        )
        
        # FEATURES ENGAGEMENT
        # Jours depuis dernière transaction
        last_tx = transactions_df.groupby('customer_id')['date'].max()
        features['days_since_last_purchase'] = (
            (transactions_df['date'].max() - last_tx).dt.days
        ).reindex(features.index, fill_value=999)
        
        # Régularité d'achat (écart-type entre transactions)
        tx_dates = transactions_df.groupby('customer_id')['date'].apply(
            lambda x: x.diff().dt.days.std() if len(x) > 1 else 0
        )
        features['purchase_regularity_std'] = tx_dates.reindex(features.index, fill_value=999)
        
        # FEATURES SUPPORT CLIENT
        if support_df is not None:
            support_grouped = support_df.groupby('customer_id')
            
            features['num_support_tickets_30d'] = (
                support_grouped['ticket_id'].count().reindex(features.index, fill_value=0)
            )
            features['avg_resolution_time'] = (
                support_grouped['resolution_hours'].mean().reindex(features.index, fill_value=0)
            )
            features['pct_unresolved'] = (
                support_grouped.apply(lambda x: (x['status'] == 'open').sum() / len(x))
                .reindex(features.index, fill_value=0)
            )
            features['negative_satisfaction'] = (
                support_grouped.apply(lambda x: (x['satisfaction'] <= 2).sum())
                .reindex(features.index, fill_value=0)
            )
        
        # FEATURES PRODUIT/SERVICE
        if 'subscription_tier' in customers_df.columns:
            features['subscription_tier'] = customers_df['subscription_tier'].map({
                'basic': 1, 'standard': 2, 'premium': 3
            })
        
        if 'num_features_used' in customers_df.columns:
            features['feature_adoption_rate'] = (
                customers_df['num_features_used'] / customers_df['total_features']
            )
        
        # FEATURES COMPORTEMENTALES
        if 'login_count_30d' in customers_df.columns:
            features['login_frequency'] = customers_df['login_count_30d']
            features['days_since_last_login'] = customers_df['days_since_last_login']
        
        # FLAGS DE RISQUE
        features['high_risk_flag'] = (
            (features['days_since_last_purchase'] > 60) |
            (features['transaction_trend'] < -50) |
            (features['spending_trend'] < -50)
        ).astype(int)
        
        features['very_high_risk_flag'] = (
            (features['days_since_last_purchase'] > 90) &
            (features['num_transactions_30d'] == 0)
        ).astype(int)
        
        return features
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Entraîner modèle de prédiction
        """
        # Gérer déséquilibre de classes (généralement 5-20% churn)
        churn_rate = y_train.mean()
        scale_pos_weight = (1 - churn_rate) / churn_rate
        
        # XGBoost avec gestion déséquilibre
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            random_state=42,
            early_stopping_rounds=30
        )
        
        # Standardisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Entraînement
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        # SHAP pour explicabilité
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        # Évaluation
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        
        print(f"\n{'='*70}")
        print(f"PERFORMANCE MODÈLE CHURN")
        print(f"{'='*70}")
        print(f"AUC-ROC Validation : {auc:.4f}")
        
        # Matrice de confusion (seuil 0.5)
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        print(f"\nClassification Report :")
        print(classification_report(y_val, y_pred_class, target_names=['Reste', 'Churn']))
        
        return auc
    
    def predict_churn_probability(self, X):
        """
        Prédire probabilité de churn
        """
        X_scaled = self.scaler.transform(X)
        churn_proba = self.model.predict_proba(X_scaled)[:, 1]
        return churn_proba
    
    def segment_by_churn_risk(self, customers_df, churn_probabilities):
        """
        Segmenter clients par niveau de risque
        """
        customers_df['churn_probability'] = churn_probabilities
        
        # Segments de risque
        customers_df['risk_segment'] = pd.cut(
            churn_probabilities,
            bins=[0, 0.2, 0.5, 0.7, 1.0],
            labels=['Faible', 'Moyen', 'Élevé', 'Critique']
        )
        
        return customers_df
    
    def recommend_retention_actions(self, customers_df):
        """
        Recommander actions de rétention personnalisées
        """
        actions = []
        
        for _, customer in customers_df.iterrows():
            customer_actions = {
                'customer_id': customer['customer_id'],
                'churn_risk': customer['risk_segment'],
                'actions': []
            }
            
            risk = customer['risk_segment']
            
            if risk == 'Critique':
                customer_actions['priority'] = 'URGENT'
                customer_actions['actions'] = [
                    "📞 Appel personnel Account Manager dans 24h",
                    "💰 Offre de rétention agressive (30-50% réduction)",
                    "🎁 Bonus/Crédits immédiats",
                    "📊 Enquête satisfaction + service recovery plan"
                ]
            
            elif risk == 'Élevé':
                customer_actions['priority'] = 'HIGH'
                customer_actions['actions'] = [
                    "📧 Email personnalisé du CSM",
                    "💵 Offre de rétention (15-20% réduction)",
                    "🔄 Proposition upgrade/downgrade selon besoin",
                    "📱 Engagement proactif (check-in call)"
                ]
            
            elif risk == 'Moyen':
                customer_actions['priority'] = 'MEDIUM'
                customer_actions['actions'] = [
                    "📬 Campagne re-engagement automatisée",
                    "🎯 Offre ciblée selon historique",
                    "📚 Contenu éducatif (maximiser valeur produit)",
                    "⭐ Programme fidélité renforcé"
                ]
            
            else:  # Faible
                customer_actions['priority'] = 'LOW'
                customer_actions['actions'] = [
                    "✅ Monitoring passif",
                    "💬 NPS survey pour maintenir satisfaction",
                    "🔔 Upsell opportuniste si pertinent"
                ]
            
            actions.append(customer_actions)
        
        return pd.DataFrame(actions)
    
    def calculate_roi(self, customers_df, actions_df):
        """
        Calculer ROI de la campagne de rétention
        """
        # Hypothèses
        avg_clv = customers_df['clv'].mean()
        retention_cost = {
            'Critique': 200,  # Coût intervention critique
            'Élevé': 100,
            'Moyen': 30,
            'Faible': 5
        }
        retention_success_rate = {
            'Critique': 0.40,  # 40% de succès
            'Élevé': 0.60,
            'Moyen': 0.75,
            'Faible': 0.90
        }
        
        roi_analysis = []
        
        for risk_level in ['Critique', 'Élevé', 'Moyen', 'Faible']:
            segment = customers_df[customers_df['risk_segment'] == risk_level]
            n_customers = len(segment)
            
            if n_customers == 0:
                continue
            
            # Coûts
            total_cost = n_customers * retention_cost[risk_level]
            
            # Bénéfices (clients sauvés × CLV)
            customers_saved = n_customers * retention_success_rate[risk_level]
            total_benefit = customers_saved * avg_clv
            
            # ROI
            net_benefit = total_benefit - total_cost
            roi_pct = (net_benefit / total_cost) * 100 if total_cost > 0 else 0
            
            roi_analysis.append({
                'risk_segment': risk_level,
                'num_customers': n_customers,
                'total_cost': total_cost,
                'customers_saved': customers_saved,
                'total_benefit': total_benefit,
                'net_benefit': net_benefit,
                'roi_pct': roi_pct
            })
        
        roi_df = pd.DataFrame(roi_analysis)
        
        print(f"\n{'='*70}")
        print(f"ANALYSE ROI CAMPAGNE RÉTENTION")
        print(f"{'='*70}\n")
        
        for _, row in roi_df.iterrows():
            print(f"📊 Segment : {row['risk_segment']}")
            print(f"   Clients ciblés    : {row['num_customers']:,}")
            print(f"   Coût campagne     : {row['total_cost']:,.0f} €")
            print(f"   Clients sauvés (est): {row['customers_saved']:.0f}")
            print(f"   Bénéfice total    : {row['total_benefit']:,.0f} €")
            print(f"   Bénéfice net      : {row['net_benefit']:,.0f} €")
            print(f"   ROI               : {row['roi_pct']:.0f}%")
            print()
        
        # Total
        print(f"💰 TOTAL CAMPAGNE")
        print(f"   Coût total        : {roi_df['total_cost'].sum():,.0f} €")
        print(f"   Bénéfice net total: {roi_df['net_benefit'].sum():,.0f} €")
        print(f"   ROI global        : {(roi_df['net_benefit'].sum() / roi_df['total_cost'].sum() * 100):.0f}%")
        
        return roi_df

# Utilisation
model = ChurnPredictionModel()

# Feature engineering
features = model.engineer_churn_features(customers, transactions, support_tickets)

# Entraînement
X_train, X_test, y_train, y_test = train_test_split(
    features, customers['churned'], 
    test_size=0.2, stratify=customers['churned'], 
    random_state=42
)

model.train_model(X_train, y_train, X_test, y_test)

# Prédiction
churn_probas = model.predict_churn_probability(features)

# Segmentation
customers_segmented = model.segment_by_churn_risk(customers, churn_probas)

# Actions recommandées
actions = model.recommend_retention_actions(customers_segmented)

# ROI
roi = model.calculate_roi(customers_segmented, actions)
```

---

## 7.4 Système de Recommandation

### 7.4.1 Types de Recommandation

**1. Filtrage Collaboratif** :
- User-based : "Les utilisateurs comme vous ont aimé..."
- Item-based : "Les clients qui ont acheté X ont aussi acheté Y"

**2. Filtrage par Contenu** :
- Recommander produits similaires à ceux aimés
- Basé sur attributs produits

**3. Hybride** :
- Combiner collaboratif + contenu
- Approche la plus performante

### 7.4.2 Implémentation

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

class ProductRecommendationEngine:
    """
    Système de recommandation produits (Collaborative Filtering)
    """
    
    def __init__(self, n_recommendations=10):
        self.n_recommendations = n_recommendations
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.svd_model = None
        
    def create_user_item_matrix(self, interactions_df):
        """
        Créer matrice utilisateur-produit
        """
        # Pivot : lignes=users, colonnes=products, valeurs=ratings/achats
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',  # ou 'quantity' si pas de ratings
            fill_value=0
        )
        
        return self.user_item_matrix
    
    def item_based_collaborative_filtering(self):
        """
        Collaborative filtering item-based (produits similaires)
        """
        # Calculer similarité cosinus entre produits
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        self.similarity_matrix = pd.DataFrame(
            item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        return self.similarity_matrix
    
    def recommend_for_user(self, user_id, exclude_purchased=True):
        """
        Recommander produits pour un utilisateur
        """
        if user_id not in self.user_item_matrix.index:
            return self.recommend_popular()  # Fallback
        
        # Produits déjà achetés/notés par user
        user_ratings = self.user_item_matrix.loc[user_id]
        purchased_items = user_ratings[user_ratings > 0].index
        
        # Calculer scores de recommandation
        scores = {}
        
        for item in purchased_items:
            # Similarités de cet item avec tous les autres
            similar_items = self.similarity_matrix[item].sort_values(ascending=False)
            
            # Pondérer par rating utilisateur
            user_rating = user_ratings[item]
            
            for similar_item, similarity in similar_items.items():
                if similar_item != item:  # Pas recommander même produit
                    if similar_item not in scores:
                        scores[similar_item] = 0
                    scores[similar_item] += similarity * user_rating
        
        # Trier par score
        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Exclure déjà achetés si demandé
        if exclude_purchased:
            recommendations = [
                (item, score) for item, score in recommendations 
                if item not in purchased_items
            ]
        
        return recommendations[:self.n_recommendations]
    
    def recommend_popular(self, n=10):
        """
        Recommander produits populaires (fallback)
        """
        # Somme des ratings/achats par produit
        popularity = self.user_item_matrix.sum(axis=0).sort_values(ascending=False)
        return list(popularity.head(n).index)
    
    def matrix_factorization_svd(self, n_factors=50):
        """
        Matrix Factorization avec SVD (scalable pour grandes matrices)
        """
        # SVD
        self.svd_model = TruncatedSVD(n_components=n_factors, random_state=42)
        user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        item_factors = self.svd_model.components_.T
        
        # Reconstruction ratings prédits
        predicted_ratings = np.dot(user_factors, item_factors.T)
        
        self.predicted_matrix = pd.DataFrame(
            predicted_ratings,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns
        )
        
        return self.predicted_matrix
    
    def recommend_with_mf(self, user_id, exclude_purchased=True):
        """
        Recommander avec Matrix Factorization
        """
        if user_id not in self.predicted_matrix.index:
            return self.recommend_popular()
        
        # Prédictions pour cet utilisateur
        user_predictions = self.predicted_matrix.loc[user_id].sort_values(ascending=False)
        
        if exclude_purchased:
            # Exclure déjà achetés
            purchased = self.user_item_matrix.loc[user_id]
            purchased_items = purchased[purchased > 0].index
            user_predictions = user_predictions.drop(purchased_items, errors='ignore')
        
        return list(user_predictions.head(self.n_recommendations).index)
    
    def evaluate_recommendations(self, test_interactions):
        """
        Évaluer qualité des recommandations
        """
        precisions = []
        recalls = []
        
        for user_id in test_interactions['user_id'].unique():
            # Vraies interactions test
            true_items = set(
                test_interactions[test_interactions['user_id'] == user_id]['product_id']
            )
            
            # Recommandations
            recommended_items = set([
                item for item, score in self.recommend_for_user(user_id)
            ])
            
            if len(recommended_items) > 0:
                # Precision@K
                hits = len(true_items & recommended_items)
                precision = hits / len(recommended_items)
                precisions.append(precision)
                
                # Recall@K
                if len(true_items) > 0:
                    recall = hits / len(true_items)
                    recalls.append(recall)
        
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"ÉVALUATION SYSTÈME DE RECOMMANDATION")
        print(f"{'='*70}")
        print(f"Precision@{self.n_recommendations} : {avg_precision:.3f}")
        print(f"Recall@{self.n_recommendations}    : {avg_recall:.3f}")
        print(f"F1-Score             : {f1:.3f}")
        
        return {'precision': avg_precision, 'recall': avg_recall, 'f1': f1}

# Utilisation
recommender = ProductRecommendationEngine(n_recommendations=10)

# Créer matrice user-item
interactions = pd.DataFrame({
    'user_id': [...],
    'product_id': [...],
    'rating': [...]  # ou 'quantity' pour implicit feedback
})

user_item_matrix = recommender.create_user_item_matrix(interactions)

# Method 1: Item-based CF
recommender.item_based_collaborative_filtering()
reco_user_123 = recommender.recommend_for_user(user_id=123)

# Method 2: Matrix Factorization
recommender.matrix_factorization_svd(n_factors=50)
reco_user_123_mf = recommender.recommend_with_mf(user_id=123)

# Évaluation
metrics = recommender.evaluate_recommendations(test_interactions)
```

### 7.4.3 Cas Réel : Spotify - Discover Weekly

**Challenge** :
Recommander 30 chansons personnalisées chaque lundi à 200M+ utilisateurs.

**Solution Multi-Algorithmes** :

1. **Collaborative Filtering** :
   - Matrice 200M users × 50M chansons
   - Matrix Factorization (ALS - Alternating Least Squares)
   - Implicit feedback (écoutes, skips, saves)

2. **NLP sur Métadonnées** :
   - Analyse paroles, descriptions artistes
   - Word2Vec sur playlists (chansons = mots)

3. **Audio Analysis** :
   - CNN sur spectrogrammes
   - Features : tempo, valence, acousticness, etc.

4. **Contextual Bandits** :
   - A/B testing en temps réel
   - Apprentissage des préférences utilisateur

**Résultats** :
- **40%** des utilisateurs écoutent Discover Weekly
- **5 milliards** de chansons streamées via Discover Weekly
- **Engagement** : +20% temps d'écoute
- **Découvertes** : 90% des chansons recommandées sont nouvelles pour l'utilisateur

