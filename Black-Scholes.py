# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 17:32:43 2025

@author: theok
"""

"""
Calcule la valeur actuelle f d'une option européenne avec Black-Scholes.

S : spot
K : strike
r : taux sans risque (continu)
T : maturité (ici en années)
vol : Volatility (sigma)
"""

import math
from scipy.stats import norm 

def BlackScholes(type_of_option, S, K, r, T, vol):
    d1 = (math.log(S/K) + (r + 0.5 * vol**2)*T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    if type_of_option=="c":
        res = S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    else: 
        res = K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return res

def BlackScholesFX(type_of_option, S, K, rd, rf, T, vol):
    d1 = (math.log(S/K) + (rd - rf + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    
    if type_of_option == "c":  # call
        res = S * math.exp(-rf*T) * norm.cdf(d1) - K * math.exp(-rd*T) * norm.cdf(d2)
    elif type_of_option == "p":  # put
        res = K * math.exp(-rd*T) * norm.cdf(-d2) - S * math.exp(-rf*T) * norm.cdf(-d1)
    else:
        raise ValueError("type_of_option doit être 'c' (call) ou 'p' (put)")
    
    return res

    
print(BlackScholes("c", 42, 40, 0.1, 0.5, 0.2))
print(BlackScholes("p", 42, 40, 0.1, 0.5, 0.2))

# Exemple : call EUR/USD
print(BlackScholesFX("c", 1.10, 1.12, 0.03, 0.01, 1, 0.2))
