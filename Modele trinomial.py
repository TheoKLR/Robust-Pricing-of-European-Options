# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:04:37 2025

@author: theok
"""

import math
from collections import defaultdict

def trinomial_price_interval_one_step(S0, K, r, T, u, m, d):
    """
    Calcule l'intervalle des prix possibles pour un call européen
    dans un modèle trinomial (1 étape).
    
    S0 : spot
    K : strike
    r : taux sans risque (continu)
    T : maturité (ici 1 étape = T)
    u,m,d : facteurs de montée, milieu, descente
    """
    # actualisation
    disc = math.exp(-r*T)
    growth = math.exp(r*T)

    # équations : p_u + p_m + p_d = 1
    # et p_u*u + p_m*m + p_d*d = e^{rT}
    # ==> on exprime en fonction de p_d
    # On choisit p_d comme variable libre
    def get_probs(p_d):
        p_u = (growth - m - p_d*(d - m)) / (u - m)
        p_m = 1 - p_u - p_d
        return p_u, p_m, p_d

    # bornes admissibles pour p_d : toutes probas doivent être >= 0
    feasible_prices = []
    step = 0.001  # pas pour explorer p_d
    p_d = 0.0
    while p_d <= 1.0:
        p_u, p_m, p_d_val = get_probs(p_d)
        if p_u >= -1e-12 and p_m >= -1e-12 and p_d_val >= -1e-12:
            # payoff du call
            payoff_u = max(S0*u - K, 0)
            payoff_m = max(S0*m - K, 0)
            payoff_d = max(S0*d - K, 0)
            price = disc * (p_u*payoff_u + p_m*payoff_m + p_d_val*payoff_d)
            feasible_prices.append(price)
        p_d += step

    return min(feasible_prices), max(feasible_prices)



def trinomial_price_interval_multi_step(S0, K, r, T, N, u, m, d):
    """
    Intervalle de prix possibles pour un call européen
    dans un modèle trinomial à N étapes (backward induction).
    """
    dt = T / N
    disc = math.exp(-r*dt)
    growth = math.exp(r*dt)

    # --- Construire l'arbre complet ---
    # Chaque nœud = (step, i, j, k) avec i+j+k=step
    # i = nb de up, j = nb de mid, k = nb de down
    def get_price(i,j,k):
        return S0 * (u**i) * (m**j) * (d**k)

    # Feuilles à maturité
    node_values = {}
    for i in range(N+1):
        for j in range(N+1-i):
            k = N - i - j
            S_T = get_price(i,j,k)
            payoff = max(S_T - K, 0)
            node_values[(N,i,j,k)] = (payoff, payoff)  # intervalle = payoff fixe

    # --- Backward induction ---
    for step in range(N-1, -1, -1):  # de N-1 jusqu'à 0
        for i in range(step+1):
            for j in range(step+1-i):
                k = step - i - j
                state = (step,i,j,k)

                # enfants : on avance de 1 étape
                children = [
                    (step+1, i+1, j, k),   # up
                    (step+1, i, j+1, k),   # mid
                    (step+1, i, j, k+1)    # down
                ]

                child_intervals = [node_values[ch] for ch in children]

                # relaxation simple : intervalle parent = disc * [min des mins, max des maxs]
                min_val = disc * min(c[0] for c in child_intervals)
                max_val = disc * max(c[1] for c in child_intervals)

                node_values[state] = (min_val, max_val)

    return node_values[(0,0,0,0)]

# Exemple
S0 = 100
K = 100
r = 0.05
T = 1
u, m, d = 1.2, 1.0, 0.8  # montée, stable, baisse
N =1

pmin, pmax = trinomial_price_interval_one_step(S0, K, r, T, u, m, d)
print(f"Intervalle des prix possibles du call: [{pmin:.4f}, {pmax:.4f}]")

pmin, pmax = trinomial_price_interval_multi_step(S0, K, r, T, N, u, m, d)
print(f"Intervalle du call à N={N} étapes: [{pmin:.4f}, {pmax:.4f}]")
