import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1) Chargement & Nettoyage du dataset
# ============================================================

def parse_number(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
        if x.endswith("K"): return float(x[:-1]) * 1_000
        if x.endswith("M"): return float(x[:-1]) * 1_000_000
        if x.endswith("B"): return float(x[:-1]) * 1_000_000_000
        return float(x)
    return x

df = pd.read_csv("Trady_Flow.csv")
for col in ["Vol", "Prems", "OI"]:
    df[col] = df[col].apply(parse_number)

df["Time"] = pd.to_datetime(df["Time"])
df["Exp"] = pd.to_datetime(df["Exp"])

df["T"] = (df["Exp"] - df["Time"]).dt.total_seconds() / (365 * 24 * 3600)
df = df[df["T"] > 0]
df["option_price"] = df["Prems"] / df["Vol"] / 100

# ============================================================
# 2) Black-Scholes
# ============================================================

def bs_components(S, K, r, T, sigma):
    denom = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / denom
    return d1, d1 - denom

def bs_price(S, K, r, T, sigma, cp):
    if cp == "c":
        d1, d2 = bs_components(S, K, r, T, sigma)
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        d1, d2 = bs_components(S, K, r, T, sigma)
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# ============================================================
# 3) Vol implicite
# ============================================================

def implied_vol(option_price, S, K, T, r, cp):
    sigma = 0.2
    for _ in range(100):
        price = bs_price(S, K, r, T, sigma, cp)
        d1, _ = bs_components(S, K, r, T, sigma)
        vega = S * norm.pdf(d1) * math.sqrt(T)
        diff = price - option_price
        if abs(diff) < 1e-6:
            break
        sigma = max(0.001, min(sigma - diff / (vega + 1e-8), 5))
    return sigma

r = 0.045
df["cp"] = df["C/P"].str.lower().str[0].map({"c": "c", "p": "p"})
df["IV"] = df.apply(lambda row: implied_vol(
    row["option_price"], row["Spot"], row["Strike"], row["T"], r, row["cp"]), axis=1)
df["BS_price"] = df.apply(lambda row: bs_price(row["Spot"], row["Strike"], r, row["T"], row["IV"], row["cp"]), axis=1)

# ============================================================
# 4) Modèle binomial
# ============================================================

def binomial_price(cp, S0, K, r, T, sigma, N=200):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    ST = np.array([S0 * u**i * d**(N - i) for i in range(N+1)])
    payoff = np.maximum(ST - K, 0) if cp == "c" else np.maximum(K - ST, 0)

    for step in range(N):
        payoff = math.exp(-r*dt) * (p*payoff[1:] + (1-p)*payoff[:-1])

    return payoff[0]

df["Binomial_price"] = df.apply(
    lambda row: binomial_price(row["cp"], row["Spot"], row["Strike"], r, row["T"], row["IV"]), axis=1
)

# ============================================================
# 5) Modèle trinomial
# ============================================================

def trinomial_price(cp, S0, K, r, T, sigma, N=100):
    dt = T / N
    disc = math.exp(-r * dt)
    
    # Trinomial factors (Boyle)
    u = math.exp(sigma * math.sqrt(3 * dt))
    d = 1 / u
    m = 1.0
    
    drift = r - 0.5 * sigma**2
    pu = max(0, min(1, 1/6 + drift * math.sqrt(dt) / (2 * sigma * math.sqrt(3))))
    pd = max(0, min(1, 1/6 - drift * math.sqrt(dt) / (2 * sigma * math.sqrt(3))))
    pm = 1 - pu - pd
    
    if pu < 0 or pd < 0 or pm < 0:
        raise ValueError("Probabilités négatives – augmenter N")
    
    payoff = (lambda S: max(S - K, 0)) if cp == "c" else (lambda S: max(K - S, 0))
    
    # Terminal values
    values = {}
    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            S_T = S0 * (u**i) * (m**j) * (d**k)
            values[(i, j, k)] = payoff(S_T)
    
    # Backward induction
    for step in range(N - 1, -1, -1):
        new_values = {}
        for i in range(step + 1):
            for j in range(step + 1 - i):
                k = step - i - j
                new_values[(i, j, k)] = disc * (
                    pu * values[(i + 1, j, k)] +
                    pm * values[(i, j + 1, k)] +
                    pd * values[(i, j, k + 1)]
                )
        values = new_values
    
    return values[(0, 0, 0)]

df["Trinomial_price"] = df.apply(
    lambda row: trinomial_price(row["cp"], row["Spot"], row["Strike"], r, row["T"], row["IV"]),
    axis=1
)

# ============================================================
# 6) Réseau de neurones
# ============================================================

with open("nn_model.pkl", "rb") as f:
    nn = pickle.load(f)

W1, b1 = nn["W1"], nn["b1"]
W2, b2 = nn["W2"], nn["b2"]
W3, b3 = nn["W3"], nn["b3"]
W4, b4 = nn["W4"], nn["b4"]
scaler_X_nn = nn["scaler_X"]
scaler_y_nn = nn["scaler_y"]

def relu(x, a=0.01): return np.where(x > 0, x, a*x)

def forward_nn(X):
    A1 = relu(X @ W1 + b1)
    A2 = relu(A1 @ W2 + b2)
    A3 = relu(A2 @ W3 + b3)
    return A3 @ W4 + b4

# Features NN
df["days_to_exp"] = (df["Exp"] - df["Time"]).dt.days.clip(lower=0)
df["sqrt_days_to_exp"] = np.sqrt(df["days_to_exp"])
df["inv_days_to_exp"] = 1 / (df["days_to_exp"] + 1)
df["moneyness"] = df["Spot"] / df["Strike"]
df["log_moneyness"] = np.log(df["moneyness"].replace(0, np.nan))
df["spot_minus_strike"] = df["Spot"] - df["Strike"]
df["spot_minus_strike_sq"] = df["spot_minus_strike"] ** 2
df["strike_x_cp"] = df["Strike"] * df["C/P"].map({"Call": 1, "Put": 0})
df["log_return"] = df.groupby("Sym")["Spot"].transform(lambda x: np.log(x).diff())
df["C/P"] = df["C/P"].map({"Call": 1, "Put": 0})

nn_features = [
    "Strike", "Spot", "Diff(%)", "days_to_exp", "sqrt_days_to_exp", "inv_days_to_exp",
    "C/P", "moneyness", "log_moneyness",
    "spot_minus_strike", "spot_minus_strike_sq",
    "strike_x_cp", "log_return"
]

X_nn = df[nn_features].fillna(0).values
X_nn_scaled = scaler_X_nn.transform(X_nn)
df["NN_price"] = np.exp(scaler_y_nn.inverse_transform(forward_nn(X_nn_scaled))) - 1e-6

# ============================================================
# 7) Random Forest
# ============================================================

with open("rf_model.pkl", "rb") as f:
    rf = pickle.load(f)

rf_model = rf["rf_model"]
scaler_X_rf = rf["scaler_X"]

rf_features = [
    "Strike","Spot","Diff(%)","days_to_exp","sqrt_days_to_exp","inv_days_to_exp",
    "C/P","moneyness","log_moneyness","spot_minus_strike",
    "spot_minus_strike_sq","strike_x_cp"
]

X_rf = df[rf_features].fillna(0).values
df["RF_price"] = rf_model.predict(scaler_X_rf.transform(X_rf))

# ============================================================
# 8) Affichage final : comparaison des modèles
# ============================================================

result = df[[
    "option_price", "BS_price","Binomial_price","Trinomial_price",
    "NN_price","RF_price"
]]

print("\n===== COMPARAISON FINALE DES MODÈLES =====\n")
print(result.head(25))


models = ["BS_price","Binomial_price","Trinomial_price","NN_price","RF_price"]

error_table = {}

for m in models:
    mae = mean_absolute_error(df["option_price"], df[m])
    mse = mean_squared_error(df["option_price"], df[m])
    rmse = math.sqrt(mse)
    mape = ( (df[m] - df["option_price"]).abs() / df["option_price"] ).mean()
    r2 = r2_score(df["option_price"], df[m])
    
    error_table[m] = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Correlation": df[m].corr(df["option_price"])
    }

error_df = pd.DataFrame(error_table).T
print(error_df)

# ============================================================
# 9) Affichage final : graphiques
# ============================================================

plt.figure(figsize=(8,6))
plt.scatter(df["option_price"], df["BS_price"], alpha=0.5, label="BS")
plt.scatter(df["option_price"], df["Trinomial_price"], alpha=0.5, label="Trinomial")
plt.scatter(df["option_price"], df["NN_price"], alpha=0.5, label="NN")
plt.scatter(df["option_price"], df["RF_price"], alpha=0.5, label="Random Forest")

plt.plot(df["option_price"], df["option_price"], 'k--', label="Parfait")
plt.xlabel("Prix marché")
plt.ylabel("Prix modèle")
plt.legend()
plt.title("Comparaison modèles vs Marché")
plt.show()


error_data = [
    (df[m] - df["option_price"]).abs()
    for m in models
]
plt.figure(figsize=(8,6))
plt.boxplot(error_data, labels=models)
plt.title("Distribution des erreurs absolues")
plt.ylabel("Erreur |modèle - marché|")
plt.yscale("log")
plt.show()


corr = df[["option_price"] + models].corr()

plt.figure(figsize=(10,8))
import seaborn as sns
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Corrélation entre modèles et marché")
plt.show()
