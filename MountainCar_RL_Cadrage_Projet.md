# Projet AML — MountainCar & Reinforcement Learning

**Cadrage du projet : problématique, état de l'art, objectifs métier (BO) et objectifs data science (DSO)**

- Cours : Advanced Machine Learning (AML) — ESPRIT, 4DAS
- Sujet : Projet 3 — Reinforcement Learning : Q-Learning
- Article de référence : *Modelling a Reinforcement Learning Agent for Mountain Car Problem Using Q-Learning With Tabular Discretization* (MysuruCon, 2022) — DOI : 10.1109/mysurucon55714.2022.9972352
- Environnement : Gymnasium `MountainCar-v0` (et `MountainCarContinuous-v0` pour TD3)
- Méthodologie : Team Data Science Process (TDSP)

---

## 1. Contexte

L'apprentissage par renforcement (Reinforcement Learning, RL) est une branche du machine learning dans laquelle un **agent** apprend à prendre des décisions en interagissant avec un **environnement**. À chaque étape, l'agent observe un **état**, choisit une **action**, et reçoit une **récompense**. Son objectif est d'apprendre une **politique** (une règle de décision) qui maximise la somme des récompenses obtenues sur le long terme.

Contrairement à l'apprentissage supervisé, il n'y a pas de dataset étiqueté : l'agent apprend uniquement par essai-erreur, à partir des conséquences de ses actions.

### 1.1 Le problème MountainCar

Une voiture est placée au fond d'une vallée, entre deux collines. Son objectif est d'atteindre le drapeau situé au sommet de la colline de droite. Le moteur est **trop faible** pour monter directement : la voiture doit d'abord reculer sur la colline de gauche pour prendre de l'élan, puis repartir vers la droite. L'agent doit découvrir seul cette stratégie de balancement.

MountainCar a été introduit par Andrew Moore (1990) et popularisé par Sutton & Barto. C'est aujourd'hui un **benchmark classique** du RL, utilisé pour tester les méthodes de contrôle avec récompense rare.

### 1.2 Spécifications de l'environnement (MountainCar-v0)

| Élément | Description |
|---|---|
| Observation | 2 variables continues : position et vitesse |
| Position | [−1,2 ; 0,6] |
| Vitesse | [−0,07 ; 0,07] |
| Actions | 3 actions discrètes : 0 = pousser à gauche, 1 = ne rien faire, 2 = pousser à droite |
| Récompense | −1 à chaque étape |
| Fin d'épisode (terminated) | La voiture atteint la position ≥ 0,5 (le drapeau) |
| Troncature (truncated) | Limite de 200 étapes par épisode |
| État initial | Position aléatoire dans [−0,6 ; −0,4], vitesse nulle |

Remarque : Gymnasium génère les données par simulation. Il ne s'agit pas d'un dataset classique à télécharger.

### 1.3 Pourquoi ce problème est difficile

- **Récompense rare (sparse reward)** : l'agent reçoit −1 à chaque étape, sans signal positif intermédiaire. Un agent aléatoire n'atteint presque jamais le drapeau en 200 étapes, il obtient donc −200 à chaque épisode et n'a rien pour apprendre au début.
- **Action contre-intuitive** : pour atteindre le but à droite, il faut d'abord aller à gauche. Une stratégie « gloutonne » échoue.
- **Espace d'états continu** : le Q-Learning tabulaire exige un nombre fini d'états, il faut donc discrétiser la position et la vitesse, ce qui entraîne une perte d'information.
- **Exploration vs exploitation** : l'agent doit explorer suffisamment pour découvrir la stratégie d'élan, sans explorer indéfiniment.

---

## 2. Problématique

**Comment un agent peut-il apprendre, uniquement par essai-erreur et sans connaître la physique du système, une stratégie efficace et fiable pour résoudre un problème de contrôle à récompense rare — et dans quelle mesure les méthodes de Deep Reinforcement Learning apportent-elles un gain réel par rapport au Q-Learning tabulaire sur un problème de faible dimension comme MountainCar ?**

Cette problématique se décompose en quatre questions de recherche :

1. **Reproductibilité** : les résultats publiés dans l'article de 2022 sont-ils reproductibles à partir de son protocole ?
2. **Amélioration** : peut-on améliorer les performances et l'efficacité du Q-Learning tabulaire en optimisant la discrétisation, les hyperparamètres et la stratégie d'exploration ?
3. **Nouvelles approches** : que valent les méthodes de Deep RL (DQN, PPO, et TD3 pour les actions continues) sur ce problème ?
4. **Compromis** : quelle solution offre le meilleur équilibre entre performance, stabilité, temps d'apprentissage et ressources consommées ?

---

## 3. État de l'art

### 3.1 Fondements : le Q-Learning

Le **Q-Learning** a été proposé par Watkins (1989) et sa convergence démontrée par Watkins & Dayan (1992). Il s'agit d'une méthode **model-free** (l'agent n'apprend pas la dynamique de l'environnement) et **off-policy** (l'agent apprend la meilleure politique tout en se comportant de manière partiellement aléatoire).

L'agent maintient une table Q(s, a) qui estime la récompense cumulée attendue en prenant l'action *a* dans l'état *s*. Après chaque étape, la table est mise à jour selon l'équation de Bellman :

**Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s′, a′) − Q(s,a) ]**

| Paramètre | Rôle |
|---|---|
| α (learning rate) | Vitesse de correction des estimations |
| γ (discount factor) | Importance accordée aux récompenses futures |
| ε (epsilon-greedy) | Probabilité de choisir une action aléatoire (exploration) |

Le Q-Learning converge vers la politique optimale si tous les couples état-action sont visités suffisamment souvent et si le learning rate décroît de façon appropriée. Sa limite principale : il n'est applicable qu'à des espaces d'états **finis et de petite taille**.

Une méthode voisine, **SARSA** (Rummery & Niranjan, 1994), est on-policy : elle met à jour Q selon l'action réellement choisie, et non selon l'action optimale.

### 3.2 Gérer un espace d'états continu

Pour appliquer une méthode tabulaire à MountainCar, l'espace continu doit être transformé en espace discret. Plusieurs approches existent dans la littérature :

| Approche | Principe | Avantages | Limites |
|---|---|---|---|
| Discrétisation uniforme (grille) | Découper chaque variable en N intervalles égaux → N² états | Très simple, rapide | Perte d'information ; compromis difficile sur N |
| Tile coding (Sutton, 1996) | Plusieurs grilles décalées superposées | Meilleure généralisation | Plus complexe à implémenter |
| Fonctions de base radiales (RBF) | Approximation lisse de la fonction de valeur | Généralisation continue | Choix des centres et largeurs |
| Approximation par réseau de neurones | Le réseau remplace la table (Deep RL) | Passe à l'échelle | Instabilité, coût d'entraînement |

**Le compromis central de la discrétisation** : trop peu d'intervalles fusionnent des états qui demandent des actions différentes ; trop d'intervalles rendent la table immense et clairsemée, ce qui ralentit fortement l'apprentissage.

### 3.3 L'article de référence (MysuruCon, 2022)

L'article modélise un agent Q-Learning tabulaire pour MountainCar. Points clés :

- **Discrétisation** : la position est arrondie à 0,1 et la vitesse à 0,01, puis convertie en indices de la Q-table.
- **Structure de la Q-table** : l'article mentionne « 12 tables de 12 lignes et 3 colonnes ». Cette description est **ambiguë** ; l'interprétation retenue devra être explicitée et justifiée.
- **Stratégie** : epsilon-greedy avec décroissance d'epsilon, mise à jour de Bellman.

**Résultats publiés :**

| Exp. | α | Decay | ε | γ | Score | Temps |
|---|---|---|---|---|---|---|
| 1 | 0,008 | 100 | 0,01 | 0,55 | −200 | 2 h 05 |
| 2 | 0,008 | 100 | 0,1 | 0,99 | −166 | 3 h 03 |
| 3 | 0,05 | 100 | 0,01 | 0,99 | −164 | 1 h 59 |
| 4 | 0,008 | 100 | 0,01 | 0,99 | −108 | 8 h 46 |
| 5 | 0,008 | 10 | 0,01 | 0,99 | −109 | 1 h 42 |

**Enseignements :**
- Le meilleur score rapporté est **−108**. La configuration à −109 atteint presque la même performance en **5 fois moins de temps** : c'est le meilleur compromis.
- Avec **γ = 0,55** (exp. 1), l'agent échoue complètement (−200) : un horizon trop court empêche la récompense de se propager jusqu'aux premiers états, ce qui confirme l'importance d'un γ élevé dans un problème à récompense différée.
- Un ε plus élevé (exp. 2) dégrade la performance finale.

**Limites identifiées de l'article :**
- Résultats rapportés sur un seul entraînement par configuration : pas de moyenne ni d'écart-type sur plusieurs graines aléatoires.
- Description ambiguë de la structure de la Q-table.
- Temps d'entraînement très longs (jusqu'à 8 h 46) pour un problème de petite taille.
- Une seule méthode étudiée : pas de comparaison avec des approches de Deep RL (évoquées seulement comme perspectives).

Ces limites justifient directement les objectifs de ce projet (reproduction, amélioration, comparaison).

### 3.4 Deep Reinforcement Learning

**DQN — Deep Q-Network (Mnih et al., 2015)**
- Remplace la Q-table par un réseau de neurones : entrée = état (position, vitesse), sortie = une Q-value par action.
- Deux mécanismes de stabilisation : l'**experience replay** (rejouer des transitions passées stockées dans un buffer) et le **target network** (un réseau cible mis à jour moins souvent).
- Adapté aux actions **discrètes**, donc applicable directement à MountainCar-v0.
- Sur MountainCar, DQN peut avoir du mal au début à cause de la récompense rare, et demande un réglage soigné.

**PPO — Proximal Policy Optimization (Schulman et al., 2017)**
- Méthode **policy gradient** de type **Actor-Critic** : apprend directement une politique (acteur) et une estimation de valeur (critique).
- Limite l'ampleur de chaque mise à jour de la politique (« clipping ») pour stabiliser l'apprentissage.
- Applicable aux actions discrètes et continues ; implémentation de référence disponible dans Stable-Baselines3.

**TD3 — Twin Delayed DDPG (Fujimoto et al., 2018)**
- Conçu pour les **actions continues** ; amélioration de DDPG (Lillicrap et al., 2016).
- Trois mécanismes : deux critiques (pour réduire la surestimation des Q-values), mises à jour retardées de la politique, et lissage de la politique cible.
- **Ne s'applique pas à MountainCar-v0** (3 actions discrètes) : il faut utiliser **MountainCarContinuous-v0**, dont les récompenses **ne sont pas directement comparables** à celles de MountainCar-v0.

### 3.5 Exploration avec récompense rare

MountainCar est un cas d'école du problème d'exploration. Les solutions classiques :
- **Epsilon-greedy avec décroissance** : beaucoup d'exploration au début, puis de moins en moins.
- **Initialisation optimiste** : initialiser Q à 0 alors que toutes les récompenses réelles sont négatives rend les actions non essayées plus attractives, ce qui pousse naturellement l'agent à explorer.
- Des méthodes plus avancées (bonus d'exploration fondés sur la nouveauté des états) existent, mais dépassent le cadre de ce projet.

### 3.6 Reproductibilité et évaluation en RL

La littérature a montré que les résultats en RL sont **très sensibles aux graines aléatoires** et aux détails d'implémentation (Henderson et al., 2018). Rapporter le meilleur entraînement seul peut être trompeur. Les bonnes pratiques recommandées (Agarwal et al., 2021) :
- Répéter chaque expérience sur **plusieurs graines** (5 à 10).
- Rapporter **moyenne et écart-type**, pas seulement le meilleur essai.
- Évaluer l'agent **sans exploration** (ε = 0), sur des épisodes séparés de l'entraînement.
- Documenter le **budget d'entraînement** pour comparer équitablement les méthodes.

Référence historique : MountainCar-v0 était considéré comme « résolu » avec une récompense moyenne de −110 sur 100 épisodes consécutifs.

### 3.7 Outils

| Outil | Rôle |
|---|---|
| Gymnasium (successeur d'OpenAI Gym) | Environnements MountainCar-v0 et MountainCarContinuous-v0 |
| NumPy | Implémentation du Q-Learning tabulaire |
| PyTorch | Implémentation de DQN |
| Stable-Baselines3 | Implémentations de référence de PPO et TD3 |
| Matplotlib | Courbes d'apprentissage et visualisations |
| Google Colab | Exécution (CPU d'abord, GPU T4 facultative) |
| Streamlit | Démonstration de l'agent entraîné |

### 3.8 Synthèse de l'état de l'art

| Méthode | Type | Actions | Forces | Faiblesses |
|---|---|---|---|---|
| Q-Learning tabulaire | Value-based, model-free | Discrètes | Simple, rapide, convergence garantie (cas tabulaire) | Exige une discrétisation ; ne passe pas à l'échelle |
| DQN | Value-based, deep | Discrètes | Utilise l'état continu exact ; passe à l'échelle | Instable, sensible aux hyperparamètres, plus coûteux |
| PPO | Policy-based, Actor-Critic | Discrètes et continues | Stable, polyvalent | Coûteux en échantillons ; exploration difficile avec récompense rare |
| TD3 | Actor-Critic, deep | Continues | Très performant en contrôle continu | Ne s'applique pas à MountainCar-v0 |

**Positionnement du projet** : partir de la méthode la plus simple (Q-Learning tabulaire, article de 2022), la reproduire et l'améliorer avec un protocole rigoureux (plusieurs graines, moyenne ± écart-type), puis évaluer objectivement si les méthodes de Deep RL justifient leur complexité supplémentaire sur ce problème.

---

## 4. Business Objectives (BO)

| N° | Objectif métier | Description |
|---|---|---|
| **BO1** | Reproduction des résultats scientifiques | Vérifier la reproductibilité des résultats présentés dans l'étude de référence et évaluer la fiabilité de son approche expérimentale. |
| **BO2** | Amélioration des performances de l'agent | Améliorer la capacité de l'agent à atteindre son objectif tout en réduisant les ressources et le temps nécessaires à son apprentissage. |
| **BO3** | Exploration de nouvelles approches d'apprentissage | Étudier l'apport des approches de Deep Reinforcement Learning pour améliorer la prise de décision et l'efficacité de l'agent. |
| **BO4** | Évaluation comparative et sélection de la meilleure solution | Identifier la solution offrant le meilleur compromis entre performance, stabilité, temps d'apprentissage et consommation des ressources. |

---

## 5. Data Science Objectives (DSO)

Chaque DSO est directement associé à un BO.

| N° | BO associé | Objectif data science | Lien |
|---|---|---|---|
| **DSO1** | BO1 | Reproduire expérimentalement le modèle de référence basé sur le Q-Learning tabulaire, en reprenant les configurations et les paramètres présentés dans l'article, puis comparer les résultats obtenus aux résultats publiés. | Vérifier la reproductibilité de l'étude. |
| **DSO2** | BO2 | Optimiser la discrétisation des états, les hyperparamètres et les stratégies d'exploration du modèle Q-Learning afin d'améliorer les performances et l'efficacité de l'apprentissage. | Améliorer la solution de référence. |
| **DSO3** | BO3 | Développer et expérimenter des approches de Deep Reinforcement Learning basées sur les réseaux de neurones, notamment DQN et PPO, ainsi que TD3 dans un environnement à actions continues. | Étudier de nouvelles méthodes de prise de décision. |
| **DSO4** | BO4 | Évaluer et comparer les modèles développés à travers des indicateurs de performance, de convergence, de stabilité et de coût computationnel, en utilisant des protocoles expérimentaux reproductibles. | Identifier objectivement la solution la plus adaptée. |

### 5.1 Détail opérationnel de chaque DSO

**DSO1 — Reproduction**
- Implémenter la discrétisation de l'article (position arrondie à 0,1, vitesse à 0,01) et documenter l'interprétation retenue pour la structure de la Q-table.
- Relancer les 5 configurations publiées (α, decay, ε, γ).
- Répéter chaque configuration sur plusieurs graines contrôlées.
- Comparer aux scores publiés (meilleur score : −108) en signalant les différences de protocole.

**DSO2 — Amélioration**
- Discrétisation : comparer 10, 12, 20 et 30 intervalles par variable.
- Hyperparamètres : α ∈ {0,001 ; 0,008 ; 0,01 ; 0,05 ; 0,1}, γ ∈ {0,90 ; 0,95 ; 0,99}, epsilon initial, epsilon minimum, vitesse de décroissance.
- Entraînement : nombre d'épisodes, arrêt anticipé, sauvegarde des Q-tables.
- Modifier **un facteur à la fois**, puis tester la combinaison des meilleures options.

**DSO3 — Deep RL**
- DQN (priorité) : réseau PyTorch, 2 couches cachées de 128 neurones (ReLU), experience replay, target network, epsilon-greedy.
- PPO : Stable-Baselines3 sur MountainCar-v0, hyperparamètres et budget documentés.
- TD3 : Stable-Baselines3 sur MountainCarContinuous-v0, évalué séparément.

**DSO4 — Comparaison**
- Comparer Q-Learning original, Q-Learning amélioré, DQN et PPO sur MountainCar-v0, avec le même protocole.
- Évaluer TD3 séparément (récompenses non comparables).

---

## 6. Indicateurs et critères de succès

### 6.1 Métriques

| Métrique | Interprétation |
|---|---|
| Average Reward | Récompense moyenne par épisode ; moins négative = meilleure |
| Success Rate | Pourcentage d'épisodes où la voiture atteint le drapeau |
| Average Steps | Nombre moyen d'étapes pour atteindre le drapeau |
| Training Time | Durée de l'entraînement |
| Convergence | Nombre d'épisodes avant stabilisation des performances |
| Stability | Variabilité des résultats entre plusieurs graines (écart-type) |

### 6.2 Critères de succès par BO

| BO | Critère de succès |
|---|---|
| BO1 | Résultats du même ordre de grandeur que l'article (meilleur score publié : −108), écarts expliqués. |
| BO2 | Récompense moyenne ≥ −110 et taux de réussite ≥ 90 % en évaluation (ε = 0), avec un temps d'entraînement réduit par rapport à l'article. |
| BO3 | Agents DQN et PPO entraînés et évalués sur MountainCar-v0 ; agent TD3 entraîné sur MountainCarContinuous-v0. |
| BO4 | Tableau comparatif complet (moyenne ± écart-type) et conclusion argumentée sur le meilleur compromis. |

### 6.3 Protocole de comparaison

- 5 à 10 graines aléatoires par configuration.
- Évaluation sans exploration (ε = 0), sur 100 épisodes séparés de l'entraînement.
- Même environnement, même protocole et budget d'entraînement documenté pour toutes les méthodes.
- Moyenne et écart-type, pas seulement le meilleur essai.
- Temps d'entraînement et temps d'évaluation séparés.
- Pas de comparaison directe entre MountainCar-v0 et MountainCarContinuous-v0.

---

## 7. Méthodologie TDSP

| Phase TDSP | Application au projet | Livrable |
|---|---|---|
| 1. Business Understanding | Problématique, BO, DSO, critères de succès, risques | Fiche de cadrage |
| 2. Data Acquisition & Understanding | Exploration de l'environnement Gymnasium, agent aléatoire de référence | Analyse de l'environnement |
| 3. Modeling | Reproduction, amélioration du Q-Learning, DQN, PPO, TD3 | Agents entraînés et expériences |
| 4. Deployment | Sauvegarde des modèles, démonstration Streamlit, vidéos | Modèles et démonstration |
| 5. Customer Acceptance | Validation des objectifs, de la reproductibilité et des résultats | Rapport et présentation finale |

---

## 8. Risques et mesures

| Risque | Impact | Mesure |
|---|---|---|
| L'agent n'atteint jamais le drapeau au début | Aucun signal d'apprentissage | Exploration initiale élevée (ε = 1), initialisation optimiste de Q |
| Ambiguïté du protocole de l'article | Reproduction imparfaite | Documenter et justifier chaque choix d'implémentation |
| Forte variabilité entre graines | Conclusions fragiles | 5 à 10 graines, moyenne ± écart-type |
| Mauvaise gestion de terminated / truncated | Q-values biaisées | Ne pas traiter une troncature (200 étapes) comme un vrai état terminal |
| Charge de travail Deep RL (DQN + PPO + TD3) | Retard sur le planning | Prioriser DQN ; PPO et TD3 avec Stable-Baselines3 |
| Temps d'entraînement long | Moins d'expériences possibles | Pas de rendu graphique pendant l'entraînement ; Colab si nécessaire |

---

## 9. Références

1. Article de référence — *Modelling a Reinforcement Learning Agent for Mountain Car Problem Using Q-Learning With Tabular Discretization*, IEEE MysuruCon, 2022. DOI : 10.1109/mysurucon55714.2022.9972352
2. Moore, A. W. (1990). *Efficient Memory-Based Learning for Robot Control*. PhD thesis, University of Cambridge.
3. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, University of Cambridge.
4. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.
5. Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using connectionist systems*. Technical Report, Cambridge University.
6. Sutton, R. S. (1996). Generalization in reinforcement learning: Successful examples using sparse coarse coding. *NeurIPS*.
7. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
8. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
9. Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning (DDPG). *ICLR*.
10. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
11. Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods (TD3). *ICML*.
12. Henderson, P., et al. (2018). Deep Reinforcement Learning that Matters. *AAAI*.
13. Agarwal, R., et al. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. *NeurIPS*.
14. Brockman, G., et al. (2016). OpenAI Gym. arXiv:1606.01540.
15. Towers, M., et al. (2024). Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv:2407.17032.
16. Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *Journal of Machine Learning Research*, 22(268).
17. Documentation Gymnasium — MountainCar : https://gymnasium.farama.org/environments/classic_control/mountain_car/
