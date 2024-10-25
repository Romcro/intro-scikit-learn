import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Charger les données
df = pd.read_csv("age_vs_poids_vs_taille_vs_sexe.csv")

# Les variables prédictives
X = df[['sexe', 'age', 'taille']]

# La variable cible, le poids
y = df['poids']

# Étape 1 : Choisir un modèle de régression linéaire
reg = LinearRegression()

# Étape 2 : Entraîner le modèle
reg.fit(X, y)

# Prédictions sur l'ensemble d'entraînement
y_pred = reg.predict(X)

# Étape 3 : Évaluer le modèle
print("Score du modèle:", reg.score(X, y))
print("Coefficients a, b, c:", reg.coef_)

# Calcul des métriques d'évaluation
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mape = mean_absolute_percentage_error(y, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)

# Prédiction avec un exemple (sexe=0, age=150, taille=153)
poids = reg.predict(np.array([[0, 150, 153]]))
print("Poids prédit pour (sexe=0, age=150, taille=153):", poids)
print("Poids prédit pour (sexe=1, age=150, taille=153):", poids)
