# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 16:37:45 2025

@author: theok
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score




def parse_number(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
        if x.endswith("K"):
            return float(x[:-1]) * 1_000
        elif x.endswith("M"):
            return float(x[:-1]) * 1_000_000
        elif x.endswith("B"):
            return float(x[:-1]) * 1_000_000_000
        else:
            return float(x)
    return x

#Evaluer un modèle
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

# Charger les données
df = pd.read_csv('Trady_Flow.csv', sep=',') 
for col in ["Vol", "Prems", "OI"]:
    df[col] = df[col].apply(parse_number)
print(df.isna().sum())

# ---- Feature Engineering Avancé ----

# Convertir les dates
df["Time"] = pd.to_datetime(df["Time"])
df["Exp"] = pd.to_datetime(df["Exp"])

# 1️⃣ Temps avant expiration
df["days_to_exp"] = (df["Exp"] - df["Time"]).dt.days.clip(lower=0)
df["sqrt_days_to_exp"] = np.sqrt(df["days_to_exp"])  # pour linéariser les effets du temps
df["inv_days_to_exp"] = 1 / (df["days_to_exp"] + 1)  # pour pondérer les échéances courtes

# 2️⃣ Moneyness et log-moneyness
df["moneyness"] = df["Spot"] / df["Strike"]
df["log_moneyness"] = np.log(df["moneyness"].replace(0, np.nan))

# 3️⃣ Distance au strike
df["spot_minus_strike"] = df["Spot"] - df["Strike"]
df["spot_minus_strike_sq"] = df["spot_minus_strike"] ** 2

# 4️⃣ Encodage C/P
df["C/P"] = df["C/P"].map({"Call": 1, "Put": 0})

# 5️⃣ Interactions pertinentes
df["strike_x_cp"] = df["Strike"] * df["C/P"]               # effet directionnel des calls/puts

# 6️⃣ Ratio Premiums/Volume (coût moyen par contrat)
df["prem_per_share"] = df["Prems"] / (df["Vol"].replace(0, np.nan))

# 7️⃣ ITM : garder comme variable cible potentielle explicative
df["ITM"] = df["ITM"].astype(int)



print("Vérification des NaN après feature engineering :")
print(df.isna().sum())

# ---- Sélection finale des features ----
features = [
    "Strike", "Spot", "Diff(%)", "days_to_exp", "sqrt_days_to_exp", "inv_days_to_exp",
    "C/P", "moneyness", "log_moneyness",
    "spot_minus_strike", "spot_minus_strike_sq",
    "strike_x_cp"
]

print("Colonnes disponibles :", df.columns.tolist())
print("Colonnes demandées :", features)

X = df[features].fillna(0).values
y = (df["Prems"] / df["Vol"].replace(0, np.nan)).fillna(0).values.reshape(-1,1)
y_log = np.log(y + 1e-6)  # target pour l'entraînement
# Normalisation de y en espace log
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_log)

print("min:", np.min(y))
print("max:", np.max(y))
print("moyenne:", np.mean(y))
print("écart-type:", np.std(y))
print("min:", np.min(y_log))
print("max:", np.max(y_log))
print("moyenne:", np.mean(y_log))
print("écart-type:", np.std(y_log))

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Activation ReLU
def relu(x, a=0.01):
    return np.where(x > 0, x, a*x)

def relu_deriv(x, a=0.01):
    return np.where(x > 0, 1, a)

# Initialisation aléatoire des poids
np.random.seed(42)

n_input = X_train.shape[1]
n_hidden1 = 16
n_hidden2 = 8
n_hidden3 = 8
n_output = 1
lr = 0.31
epochs = 1000

W1 = np.random.randn(n_input, n_hidden1) * np.sqrt(2/n_input)
b1 = np.zeros((1, n_hidden1))
W2 = np.random.randn(n_hidden1, n_hidden2) * np.sqrt(2/n_hidden1)
b2 = np.zeros((1, n_hidden2))
W3 = np.random.randn(n_hidden2, n_hidden3) * np.sqrt(2/n_hidden2)
b3 = np.zeros((1, n_hidden3))
W4 = np.random.randn(n_hidden3, n_output) * np.sqrt(2/n_hidden3)
b4 = np.zeros((1, n_output))

# Forward pass
def forward(X):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = relu(Z3)
    Z4 = A3 @ W4 + b4
    
    y_pred = Z4  # linéaire pour régression
    cache = (X, Z1, A1, Z2, A2, Z3, A3, Z4)
    return y_pred, cache

def backward(y_true, y_pred, cache):
    X, Z1, A1, Z2, A2, Z3, A3, Z4 = cache
    m = y_true.shape[0]

    # dLoss/dy_pred (MSE)
    dZ4 = (y_pred - y_true) / m  # shape (m, 1)
    dW4 = A3.T @ dZ4
    db4 = np.sum(dZ4, axis=0, keepdims=True)
    
    dA3 = dZ4 @ W4.T
    dZ3 = dA3 * relu_deriv(Z3)
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_deriv(Z2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    grads = (dW1, db1, dW2, db2, dW3, db3, dW4, db4)
    return grads

def update_weights(grads, lr):
    global W1, b1, W2, b2, W3, b3, W4, b4
    dW1, db1, dW2, db2, dW3, db3, dW4, db4 = grads
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
    W4 -= lr * dW4
    b4 -= lr * db4

# Loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Train
for epoch in range(epochs):
    y_pred, cache = forward(X_train)
    loss = np.mean((y_train - y_pred)**2)
    grads = backward(y_train, y_pred, cache)
    update_weights(grads, lr)

    if epoch % 100 == 0 or epoch == epochs-1:
        print(f"Epoch {epoch}: Train MSE = {loss:.5f}")
        
# Evaluation
y_train_pred_scaled, _ = forward(X_train)
y_test_pred_scaled, _ = forward(X_test)

# Re-scaler pour RMSE réel
# repasser dans l'espace log
y_train_pred_log = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_pred_log = scaler_y.inverse_transform(y_test_pred_scaled)

# repasser dans l'espace réel
y_train_pred = np.exp(y_train_pred_log) - 1e-6
y_test_pred  = np.exp(y_test_pred_log) - 1e-6

y_train_true = np.exp(scaler_y.inverse_transform(y_train)) - 1e-6
y_test_true  = np.exp(scaler_y.inverse_transform(y_test)) - 1e-6

rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
r2_train = r2_score(y_train_true, y_train_pred)

rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
r2_test = r2_score(y_test_true, y_test_pred)

print("\nRésultats :")
print(f"Train RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
print(f"Test RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")
    