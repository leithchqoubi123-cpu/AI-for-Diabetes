# IA Médicale — Détection du Diabète

Projet de machine learning appliqué à la santé.  
Objectif : entraîner un modèle capable d’estimer le risque de diabète à partir de paramètres cliniques et biologiques.

## 🔍 Données

- Jeu de données : Pima Indians Diabetes Dataset (UCI) *(ou préciser la source exacte que tu utilises)*.
- Variables : nombre de grossesses, glycémie, pression artérielle, épaisseur cutanée, insuline, IMC, âge, etc.
- Problème : classification binaire (diabétique / non diabétique) avec classes déséquilibrées.

## 🧠 Modèle et pipeline

- Prétraitement des données (remplacement des zéros aberrants, features dérivées : ratios, produits, etc.).
- Gestion du déséquilibre : **SMOTE**.
- Modèle principal : **XGBoostClassifier** intégré dans un pipeline `scikit-learn` avec `StandardScaler`.
- Optimisation des hyperparamètres : `RandomizedSearchCV` + `StratifiedKFold`.

## 📊 Évaluation

- Métriques : accuracy, précision, rappel, F1-score, **ROC AUC**.
- Visualisations :
  - Courbe ROC
  - Matrice de confusion
  - Courbe Precision–Recall

## 🩺 Interprétabilité

- Utilisation de **SHAP** pour expliquer l’importance des variables pour chaque prédiction.
- Génération de graphiques montrant l’impact des features sur le risque de diabète.

## 💻 Mini-app Gradio

- Interface développée avec **Gradio**.
- L’utilisateur renseigne les paramètres d’un patient (glycémie, IMC, âge, etc.).
- L’IA renvoie :
  - une probabilité de diabète,
  - une interprétation textuelle,
  - un graphique SHAP pour expliquer la décision.

## ▶️ Utilisation

1. Ouvrir le notebook `diabetes_ai.ipynb` dans Google Colab.
2. Exécuter la cellule d’installation des dépendances (`pip install ...`).
3. Lancer toutes les cellules pour :
   - charger et préparer les données,
   - entraîner le modèle,
   - évaluer les performances,
   - lancer l’interface Gradio.

## 👤 Auteur

Projet développé par **Leith**, étudiant en pharmacie, intéressé par l’IA appliquée à la santé.
