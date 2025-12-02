# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:45:46 2025

@author: theok
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import PartialDependenceDisplay
from py_vollib.black_scholes.implied_volatility import implied_volatility

def calculate_iv_vollib(row, risk_free_rate=0.045):
    """
    Calcule l'IV avec py_vollib (méthode Newton-Raphson)
    À partir du PRIX de l'option (pas du spread bid-ask)
    """
    try:
        spot = float(row["Spot"])
        strike = float(row["Strike"])
        time_to_exp = max(row["days_to_exp"] / 365.0, 0.001)  # En années
        
        # ⭐ CHANGEMENT CRITIQUE : Utiliser le prix de l'option
        price = float(row["option_price"])  # ← Au lieu de row["BidAsk"]
        
        flag = 'c' if row["C/P"] == 1 else 'p'  # 'c' pour call, 'p' pour put
        
        # Validations supplémentaires
        if spot <= 0 or strike <= 0 or price <= 0:
            return np.nan
        
        # Vérifier arbitrage (prix doit être >= valeur intrinsèque)
        if flag == 'c':
            intrinsic = max(spot - strike, 0)
        else:
            intrinsic = max(strike - spot, 0)
        
        if price < intrinsic * 0.95:  # Tolérance 5%
            return np.nan
        
        # Calcul de l'IV
        iv = implied_volatility(
            price=price,
            S=spot,
            K=strike,
            t=time_to_exp,
            r=risk_free_rate,
            flag=flag
        )
        
        # Filtrer valeurs aberrantes
        if iv < 0.01 or iv > 5.0:  # IV entre 1% et 500%
            return np.nan
        
        return iv
    
    except Exception as e:
        return np.nan

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

# 7️⃣ ITM : garder comme variable cible potentielle explicative
df["ITM"] = df["ITM"].astype(int)

#8
#df["IV"] = df.apply(calculate_iv_vollib, axis=1)

print("Vérification des NaN après feature engineering :")
print(df.isna().sum())

# ---- Sélection finale des features ----
X = df[
    [
        "Strike", "Spot", "Diff(%)", "days_to_exp", "sqrt_days_to_exp", "inv_days_to_exp",
        "C/P", "moneyness", "log_moneyness",
        "spot_minus_strike", "spot_minus_strike_sq",
        "strike_x_cp"
    ]
]


# Target    
y = df["Prems"] / (df["Vol"].replace(0, np.nan))

# Étape 1 : Split train + temp (train=70%, temp=30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Étape 2 : Split temp en validation (15%) et test (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Taille du train set :", X_train.shape[0])
print("Taille du validation set :", X_val.shape[0])
print("Taille du test set :", X_test.shape[0])

# Standardisation (fit uniquement sur train !)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Modèle
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    print(f"\n=== {name} Regression ===")
    rmse_train, r2_train = evaluate(y_train, y_train_pred)
    rmse_val, r2_val = evaluate(y_val, y_val_pred)
    rmse_test, r2_test = evaluate(y_test, y_test_pred)
    
    print(f"Train - RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")
    print(f"Validation - RMSE: {rmse_val:.4f}, R²: {r2_val:.4f}")
    print(f"Test - RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")
    
    # Optionnel : coefficients
    if name in ["Linear", "Ridge", "Lasso"]:
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_
        }).sort_values(by="Coefficient", key=abs, ascending=False)
        print("Top coefficients :")
        print(coef_df.head(5))
        
importances = pd.Series(models["RandomForest"].feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='barh', figsize=(8,5))
plt.title("Top 10 des features importantes (Random Forest)")
plt.show()

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=2
)

# === Optimisation du Random Forest ===
"""
print("\n🔍 Lancement de GridSearchCV pour RandomForest...")
grid_search.fit(X_train_scaled, y_train)

print("\n✅ Meilleurs hyperparamètres trouvés :")
print(grid_search.best_params_)
print(f"Score R² (moyen CV): {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
"""

# ----------------------------
# PDP & ICE (Partial Dependence + Individual Conditional Expectation)
# ----------------------------

# 1) Refit un RandomForest sur les données NON-scalées pour obtenir des axes lisibles
rf_unscaled = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)
rf_unscaled.fit(X_train, y_train)  # X_train non-scaled

# 2) Features à étudier (univariées + interactions)
uni_features = ["moneyness", "days_to_exp", "Spot", "Strike", "C/P"]
pair_features = [("moneyness", "days_to_exp")]

# 3) PDP univariées + ICE (sample pour ICE)
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Univariées (PDP moyen)
fig, axes = plt.subplots(1, len(uni_features), figsize=(5*len(uni_features), 4))
for i, feat in enumerate(uni_features):
    PartialDependenceDisplay.from_estimator(
        rf_unscaled,
        X_train,                      # DataFrame non-scaled
        [feat],
        feature_names=X.columns,
        ax=axes[i],
        grid_resolution=50,
        kind='average'                # PDP moyen
    )
    axes[i].set_title(f"PDP: {feat}")
plt.tight_layout()
plt.show()

# Séparer les données pour Call et Put
X_train_call = X_train[X_train["C/P"] == 1]  # Call
X_train_put = X_train[X_train["C/P"] == 0]   # Put

features_to_plot = ["Spot", "Strike"]

for feat in features_to_plot:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # PDP pour les Calls
    PartialDependenceDisplay.from_estimator(
        rf_unscaled,
        X_train_call,
        [feat],
        feature_names=X.columns,
        ax=axes[0],
        grid_resolution=50,
        kind='average'
    )
    axes[0].set_title(f"PDP Call: {feat}")
    
    # PDP pour les Puts
    PartialDependenceDisplay.from_estimator(
        rf_unscaled,
        X_train_put,
        [feat],
        feature_names=X.columns,
        ax=axes[1],
        grid_resolution=50,
        kind='average'
    )
    axes[1].set_title(f"PDP Put: {feat}")
    
    plt.tight_layout()
    plt.show()

# PDP + ICE (affiche à la fois la courbe moyenne et quelques lignes individuelles)
# Pour vitesse, échantillonne quelques observations si X_train est grand
X_ice = X_train.sample(min(500, X_train.shape[0]), random_state=42)
plt.figure(figsize=(6,4))
PartialDependenceDisplay.from_estimator(
    rf_unscaled,
    X_ice,
    ["moneyness"],
    feature_names=X.columns,
    kind='both',        # affiche PDP moyen + ICE individuelles
    grid_resolution=60
)
plt.title("PDP + ICE : moneyness (échantillon)")
plt.tight_layout()
plt.show()

# 4) PDP 2D (interaction moneyness × days_to_exp)
fig = plt.figure(figsize=(7,6))
PartialDependenceDisplay.from_estimator(
    rf_unscaled,
    X_train.sample(min(2000, X_train.shape[0]), random_state=1),  # sample pour vitesse si gros dataset
    [("moneyness", "days_to_exp")],
    feature_names=X.columns,
    kind='average',
    grid_resolution=40,
    ax=plt.gca()
)
plt.title("PDP 2D: moneyness × days_to_exp (moyenne)")
plt.tight_layout()
plt.show()

# 5) PDP par sous-groupe (Calls vs Puts) — utile pour voir différences structurelles
X_calls = X_train[X_train["C/P"] == 1]
X_puts = X_train[X_train["C/P"] == 0]

if len(X_calls) > 50:
    plt.figure(figsize=(6,4))
    PartialDependenceDisplay.from_estimator(
        rf_unscaled,
        X_calls.sample(min(500, len(X_calls)), random_state=2),
        ["moneyness"],
        feature_names=X.columns,
        kind='both',
        grid_resolution=50
    )
    plt.title("Calls: PDP + ICE (moneyness)")
    plt.tight_layout()
    plt.show()

if len(X_puts) > 50:
    plt.figure(figsize=(6,4))
    PartialDependenceDisplay.from_estimator(
        rf_unscaled,
        X_puts.sample(min(500, len(X_puts)), random_state=3),
        ["moneyness"],
        feature_names=X.columns,
        kind='both',
        grid_resolution=50
    )
    plt.title("Puts: PDP + ICE (moneyness)")
    plt.tight_layout()
    plt.show()
