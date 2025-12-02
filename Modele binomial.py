# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 16:06:48 2025

@author: theok
"""

"""
Calcule la valeur actuelle f d'une option européenne avec le modèle binomial (1 étape).

S0 : spot
K : strike
r : taux sans risque (continu)
T : maturité (ici 1 étape = T)
u,d : facteurs de montée, descente
"""

import math

def binomial_one_step (type_of_option, S0, K, r, u, d):
    p = ((1+r)-d)/(u-d)
    if type_of_option=="c":
        fu = max(S0*u-K, 0)
        fd = max(S0*d-K, 0)
    else: #put
        fu = max(K-S0*u,0)
        fd = max(K-S0*d,0)
    f = (p*fu + (1-p)*fd)/(1+r)
    return f

def binomial_n_step (type_of_option, n, S0, K, r, u, d):
    p = ((1+r)-d)/(u-d)
    somme = 0
    for i in range(n+1):
        if type_of_option == "c":
            payoff = max(S0 * (u**i) * (d**(n-i)) - K, 0)
        else:  # put
            payoff = max(K - S0 * (u**i) * (d**(n-i)), 0)
        somme += math.comb(n, i) * p**(i) * (1-p)**(n-i) * payoff
    f = somme/(1+r)**n
    return f

def binomial_fx_option(type_of_option, n, S0, K, rd, rf, u, d, T=1):
    """
    Option FX selon le modèle binomial CRR adapté Garman–Kohlhagen.
    rd : taux domestique
    rf : taux étranger (dividende continu)
    """
    dt = T / n
    
    # Probabilité neutre au risque FX
    p = (math.exp((rd - rf) * dt) - d) / (u - d)

    somme = 0
    
    for i in range(n+1):
        ST = S0 * (u**i) * (d**(n-i))
        if type_of_option == "c":
            payoff = max(ST - K, 0)
        else:
            payoff = max(K - ST, 0)
        prob = math.comb(n, i) * (p**i) * ((1-p)**(n-i))
        somme += prob * payoff
    
    # Actualisation domestique
    f = somme * math.exp(-rd * T)
    return f

print(binomial_one_step("c",100, 110, 0.04, 1.4, 0.75))
print(binomial_n_step("c", 1, 100, 110, 0.04, 1.4, 0.75))

# Exemple : call EUR/USD
print(binomial_fx_option("call", 3, 1.10, 1.12, 0.03, 0.01, 1.2, 0.8, T=1))
