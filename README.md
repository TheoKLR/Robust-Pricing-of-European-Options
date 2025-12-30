# Robust-Pricing-of-European-Options

Résumé
-----
Ce dépôt contient l'implémentation et l'étude comparative de plusieurs méthodes de pricing d'options européennes, incluant des approches classiques et des méthodes de pricing robuste basées sur le Machine Learning (Random Forest, Réseau de neurone). L'objectif principal est d'analyser, comparer et évaluer la robustesse des estimateurs sur des jeux de données synthétiques et réels, et de proposer des pistes d'amélioration.

Contexte et motivation
----------------------
Le pricing d'options est un problème central en finance quantitative. Outre les formules analytiques classiques (ex. Black–Scholes), l'intérêt croissant pour des méthodes basées sur les données (Random Forests, réseaux de neurones) permet de construire des estimateurs flexibles et potentiellement plus robustes face à des hypothèses de marché irréalistes ou à des données bruitées. Ce projet vise à :
- Implémenter plusieurs méthodes de pricing,
- Comparer leurs performances et leur robustesse,
- Discuter des limites et des axes d'amélioration.

Points clés et conclusion (extrait adapté)
------------------------------------------
Finalement, l’implémentation de nos différents modèles a été un succès. Il nous a été possible de comparer leurs résultats entre eux et de les analyser.  
Malgré des modèles satisfaisants, il reste possible de les améliorer. La marche d’amélioration la plus conséquente concerne les performances des méthodes de pricing robustes. En effet, notre Random Forest et le Réseau de neurones se basent sur des données pour apprendre. Or, si nous obtenions un dataset avec plus de données, une plus grande variété et une meilleure clarification des features, alors nos performances seraient améliorées. De plus, certaines architectures ou algorithmes de Machine Learning auraient pu être encore plus performants mais ils sont souvent lourds à implémenter et requièrent une puissance de calcul hors de notre portée.  
D’autre part, ce projet étudiant a été pour moi une occasion précieuse de découvrir un aspect de la finance tout en progressant en développement. Travailler sur les différents modèles de pricing m’a permis de comprendre leur fonctionnement, leur utilité et leur complexité — compétences utiles pour mon cursus d’ingénieur et mon master OSS (recherche d’information, amélioration continue, développement informatique, curiosité).

Fonctionnalités
--------------
- Génération et manipulation de jeux de données pour options européennes (synthétiques et import de jeux externes).
- Implémentation de méthodes classiques (p. ex. Black–Scholes) comme ligne de base.
- Implémentation de méthodes robustes basées sur le Machine Learning : Random Forest, Réseau de neurones (MLP).
- Pipelines d'entraînement, validation croisée, évaluation (MSE, MAE, erreurs relatives).
- Scripts et notebooks pour reproduire les expériences, visualiser les résultats et tracer les diagnostics (erreurs, biais, variance).
- Export / sauvegarde des modèles entraînés et des résultats.

Description des modèles et choix méthodologiques
-----------------------------------------------
- Baseline (Black–Scholes) :
  - Sert de référence. Évalue la performance des méthodes basées sur des hypothèses fermées.
- Random Forest (pricing robuste) :
  - Avantages : robustesse, peu de pré-traitement des features, interprétabilité relative (importance des features).
  - Inconvénients : limites sur l'extrapolation, nécessite des données représentatives.
- Réseau de neurones (MLP) :
  - Avantages : capacité à modéliser des relations non linéaires complexes.
  - Inconvénients : nécessite davantage de données, réglages d'hyperparamètres, et ressources de calcul.
- Évaluation :
  - Utilisation de jeux de test hors échantillon, validation croisée, courbes d'erreur, et tests de robustesse (bruit, changement de distribution).

Résultats et limites
--------------------
- Les implémentations ont permis de comparer efficacement les méthodes et d'analyser leur comportement.
- Limites observées :
  - Quantité et qualité des données : les performances des méthodes ML sont fortement liées à la richesse du dataset.
  - Features : une meilleure construction et clarification des features améliorerait l'apprentissage.
  - Ressources : architectures ML plus performantes existent, mais leur mise en œuvre nécessite plus de calcul.
- Axes d'amélioration :
  - Collecter ou simuler davantage de données et couvrant plus de scénarios de marché.
  - Enrichir et normaliser les features (ex. volatilité implicite, historisation des features).
  - Tester des architectures plus sophistiquées (CNNs pour time-series, LSTM, ensembles avancés) et optimiser l'hyperparamétrage.
  - Expérimenter des méthodes de régularisation et d'explicabilité (SHAP, LIME).
 
Remarques finales
-----------------
Ce projet m'a permis d'approfondir mes connaissances en finance quantitative et en développement logiciel. L'implémentation et l'analyse des différentes méthodes ont été formatrices et constituent une base solide pour des travaux futurs plus avancés (meilleure collecte de données, architectures ML plus complexes, métriques de robustesse supplémentaires).

Contact
-------
TheoKLR — pour toute question relative au projet, discuter d'améliorations ou proposer des données supplémentaires.

