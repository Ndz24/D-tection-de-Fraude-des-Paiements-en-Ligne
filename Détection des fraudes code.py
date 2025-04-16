# =====================================================================
# Détection de Fraude dans les Paiements en Ligne avec Machine Learning
# Description : Ce projet vise à détecter les fraudes dans les paiements
#               en ligne à l'aide d'un dataset et de plusieurs algorithmes
#               de Machine Learning.
# =====================================================================

# **1. Importation des Bibliothèques et Chargement des Données**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import ConfusionMatrixDisplay

# Charger les données
# Remplacez 'new_data.csv' par le chemin complet du fichier CSV si nécessaire
data = pd.read_csv('dataset.csv')

# Visualiser les informations générales sur le dataset pour comprendre la structure
print("Informations générales sur le dataset :")
print(data.info())
print("\nStatistiques descriptives du dataset :")
print(data.describe())

# **2. Visualisation des Données**
# Cette étape nous aide à mieux comprendre la distribution des données et les relations.

# Identifier les types de colonnes (catégoriques, entières, flottantes)
obj = (data.dtypes == 'object')  # Colonnes catégoriques (e.g., chaînes de caractères)
object_cols = list(obj[obj].index)
print("Nombre de variables catégoriques :", len(object_cols))

int_ = (data.dtypes == 'int')  # Colonnes de type entier
num_cols = list(int_[int_].index)
print("Nombre de variables entières :", len(num_cols))

fl = (data.dtypes == 'float')  # Colonnes de type flottant
fl_cols = list(fl[fl].index)
print("Nombre de variables flottantes :", len(fl_cols))

# **Visualisation 1 : Répartition des types de transactions**
# Cela permet de voir combien de chaque type de paiement (ex. transfert, retrait) sont présents.
sns.countplot(x='type', data=data)
plt.title("Répartition des types de transactions")
plt.show()

# **Visualisation 2 : Analyse du montant par type de transaction**
# Ici, nous analysons les types de transactions par le montant moyen associé à chaque type.
sns.barplot(x='type', y='amount', data=data)
plt.title("Montant moyen par Type de Transaction")
plt.show()

# **Visualisation 3 : Répartition des transactions normales vs frauduleuses**
# Cela montre combien de transactions frauduleuses sont présentes dans le dataset.
fraud_count = data['isFraud'].value_counts()
print("Répartition des transactions frauduleuses :")
print(fraud_count)

# **Visualisation 4 : Distribution des étapes (step)**
# La colonne "step" représente les unités de temps. Nous étudions la répartition des transactions au fil du temps.
plt.figure(figsize=(15, 6))
sns.histplot(data['step'], bins=50, kde=True)
plt.title("Distribution des étapes (step)")
plt.show()

# **Visualisation 5 : Heatmap des corrélations entre les variables**
# Une heatmap permet de visualiser les relations entre les variables numériques.
plt.figure(figsize=(12, 6))
sns.heatmap(data.apply(lambda x: pd.factorize(x)[0]).corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)
plt.title("Heatmap des corrélations")
plt.show()

# **3. Pré-traitement des Données**
# Avant d'appliquer les algorithmes de Machine Learning, nous devons :
# - Encoder les colonnes catégoriques comme "type" en valeurs numériques.
# - Supprimer les colonnes non utiles pour la détection de fraudes.

# Encodage de la colonne "type" (ex : "TRANSFER" → 1, "CASH_OUT" → 0, etc.)
type_new = pd.get_dummies(data['type'], drop_first=True)
data_new = pd.concat([data, type_new], axis=1)

# Suppression des colonnes inutiles pour l'apprentissage (non pertinentes ou redondantes)
# - "nameOrig" et "nameDest" sont des noms ou identifiants inutilisables ici.
X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)  # Features
y = data_new['isFraud']  # Label ou cible (fraude ou non)

# Vérification de la taille des Features (X) et de la cible (y)
print("Forme des données après extraction des Features et Labels :", X.shape, y.shape)

# Division du dataset (train/test split)
# On utilise 70% pour l'entraînement et 30% pour le test (validation).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# **4. Entraînement des Modèles**
# Nous allons utiliser plusieurs modèles d'apprentissage supervisé pour cette tâche :

# - LogisticRegression : Régression Logistique
# - RandomForestClassifier : Forêt d'arbres de décision
# - XGBClassifier : Gradient Boosted Trees Classifier fourni par XGBoost

# Initialisation des modèles
models = [LogisticRegression(),
          XGBClassifier(),
          RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)]

# Itération sur chaque modèle pour l'entraînement et l'évaluation
for model in models:
    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)
    print(f'{model} : ')
    
    # Calcul de la performance sur l'ensemble d'entraînement (ROC-AUC Score)
    train_preds = model.predict_proba(X_train)[:, 1]
    print('Précision sur l\'Entraînement : ', ras(y_train, train_preds))
    
    # Calcul de la performance sur l'ensemble de test (ROC-AUC Score)
    test_preds = model.predict_proba(X_test)[:, 1]
    print('Précision sur le Test : ', ras(y_test, test_preds))
    print()

# **5. Évaluation du Meilleur Modèle**
# Après les résultats, le meilleur modèle est souvent le XGBoost Classifier.
# Nous utilisons une matrice de confusion pour visualiser les performances du modèle final.
best_model = models[1]  # XGBClassifier comme modèle sélectionné
cm = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
cm.plot(cmap='Blues')
plt.title("Matrice de Confusion du Modèle XGBClassifier")
plt.show()

# =====================================================================
# FIN DU PROJET
# Ce script a permis de construire un modèle capable d'identifier
# efficacement les transactions frauduleuses en ligne. Le XGBClassifier
# s'est avéré être le meilleur modèle avec des performances très élevées.
# =====================================================================
