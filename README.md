**# 🔍 Détection de Fraude dans les Paiements en Ligne

Ce projet vise à détecter les fraudes dans les transactions financières en ligne à l'aide de modèles de Machine Learning. Grâce à une analyse approfondie des données et la mise en œuvre de plusieurs algorithmes, nous identifions les transactions potentiellement frauduleuses.

## 📁 Fichiers du projet

- `dataset.csv` : Jeu de données utilisé pour l'entraînement et les tests.
- `fraud_detection.py` ou `notebook.ipynb` : Script principal contenant le code d'analyse, de visualisation et de modélisation.
- `README.md` : Description du projet.

## 🎯 Objectifs

- Analyser les caractéristiques des transactions.
- Identifier les transactions suspectes ou frauduleuses.
- Comparer les performances de plusieurs modèles de classification :
  - Régression Logistique
  - Random Forest
  - XGBoost

## 🧪 Pipeline de Traitement

1. **Chargement et exploration des données**
2. **Visualisation des variables importantes** :
   - Type de transaction
   - Montant
   - Répartition des fraudes
3. **Prétraitement des données** :
   - Encodage des variables catégorielles (`type`)
   - Suppression des identifiants (`nameOrig`, `nameDest`)
4. **Division du dataset en train/test**
5. **Entraînement de plusieurs modèles**
6. **Évaluation des performances via le ROC-AUC Score**
7. **Visualisation avec une matrice de confusion**

## 📊 Visualisations réalisées

- Répartition des types de transactions
- Montants moyens par type
- Histogrammes des étapes (`step`)
- Heatmap des corrélations
- Matrice de confusion pour le meilleur modèle

## 🛠️ Technologies utilisées

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- XGBoost

## ✅ Résultats

Le modèle **XGBoostClassifier** a obtenu les meilleures performances en termes de **ROC-AUC score**, montrant une excellente capacité à détecter les fraudes tout en minimisant les faux positifs.

## 🚀 Lancer le projet

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/<votre-utilisateur>/fraud_detection_project.git
   cd fraud_detection_project
**
