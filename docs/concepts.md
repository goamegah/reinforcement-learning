# Concepts théoriques de l'apprentissage par renforcement

Ce document présente les concepts théoriques fondamentaux de l'apprentissage par renforcement qui sont implémentés dans ce framework.

## Processus de décision markovien (MDP)

Un processus de décision markovien est défini par un tuple (S, A, P, R, γ) où :
- S est l'ensemble des états
- A est l'ensemble des actions
- P est la fonction de transition qui donne la probabilité P(s'|s,a) d'atteindre l'état s' en effectuant l'action a dans l'état s
- R est la fonction de récompense qui donne la récompense R(s,a,s') obtenue en passant de l'état s à l'état s' via l'action a
- γ est le facteur d'actualisation (discount factor) qui détermine l'importance des récompenses futures

Dans notre framework, tous les environnements implémentent l'interface `BaseEnvironment` qui fournit les méthodes nécessaires pour interagir avec un MDP.

## Fonctions de valeur

### Fonction de valeur d'état (V)

La fonction de valeur d'état V(s) représente la valeur espérée de l'état s en suivant une politique π :

V<sup>π</sup>(s) = E<sub>π</sub>[G<sub>t</sub> | S<sub>t</sub> = s]

où G<sub>t</sub> est le retour (somme des récompenses futures actualisées) à partir du temps t.

### Fonction de valeur d'action (Q)

La fonction de valeur d'action Q(s,a) représente la valeur espérée de l'exécution de l'action a dans l'état s, puis de suivre la politique π :

Q<sup>π</sup>(s,a) = E<sub>π</sub>[G<sub>t</sub> | S<sub>t</sub> = s, A<sub>t</sub> = a]

## Politiques

Une politique π définit le comportement de l'agent. Elle peut être :
- Déterministe : π(s) = a
- Stochastique : π(a|s) = P(A<sub>t</sub> = a | S<sub>t</sub> = s)

Dans notre framework, les politiques sont généralement représentées par un tableau numpy où l'indice correspond à l'état et la valeur correspond à l'action à effectuer.

## Équations de Bellman

Les équations de Bellman sont fondamentales en apprentissage par renforcement. Elles expriment la relation récursive entre la valeur d'un état et la valeur des états suivants.

### Équation de Bellman pour V<sup>π</sup>

V<sup>π</sup>(s) = Σ<sub>a</sub> π(a|s) Σ<sub>s',r</sub> p(s',r|s,a)[r + γV<sup>π</sup>(s')]

### Équation de Bellman pour Q<sup>π</sup>

Q<sup>π</sup>(s,a) = Σ<sub>s',r</sub> p(s',r|s,a)[r + γ Σ<sub>a'</sub> π(a'|s')Q<sup>π</sup>(s',a')]

### Équations de Bellman optimales

V<sup>*</sup>(s) = max<sub>a</sub> Σ<sub>s',r</sub> p(s',r|s,a)[r + γV<sup>*</sup>(s')]

Q<sup>*</sup>(s,a) = Σ<sub>s',r</sub> p(s',r|s,a)[r + γ max<sub>a'</sub> Q<sup>*</sup>(s',a')]

## Méthodes de résolution

### Programmation Dynamique (DP)

Les méthodes de programmation dynamique supposent une connaissance complète du MDP et utilisent les équations de Bellman pour calculer les fonctions de valeur.

#### Policy Iteration

1. **Évaluation de la politique** : Calcul de V<sup>π</sup> pour une politique fixe π
2. **Amélioration de la politique** : π'(s) = argmax<sub>a</sub> Σ<sub>s',r</sub> p(s',r|s,a)[r + γV<sup>π</sup>(s')]

#### Value Iteration

Combine l'évaluation et l'amélioration de la politique en une seule étape :
V(s) ← max<sub>a</sub> Σ<sub>s',r</sub> p(s',r|s,a)[r + γV(s')]

### Méthodes de Monte Carlo (MC)

Les méthodes de Monte Carlo apprennent à partir d'épisodes complets d'expérience, sans nécessiter de connaissance préalable du MDP.

#### First-Visit MC

Mise à jour de V(s) en utilisant la moyenne des retours observés après la première visite de l'état s dans chaque épisode.

#### Every-Visit MC

Mise à jour de V(s) en utilisant la moyenne des retours observés après chaque visite de l'état s dans chaque épisode.

### Méthodes de Différence Temporelle (TD)

Les méthodes TD combinent les idées de DP et MC en apprenant à partir d'expériences partielles et en mettant à jour les estimations en fonction des estimations suivantes.

#### TD(0)

V(s) ← V(s) + α[r + γV(s') - V(s)]

#### SARSA (On-policy TD Control)

Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]

#### Q-Learning (Off-policy TD Control)

Q(s,a) ← Q(s,a) + α[r + γ max<sub>a'</sub> Q(s',a') - Q(s,a)]

#### Expected SARSA

Q(s,a) ← Q(s,a) + α[r + γ Σ<sub>a'</sub> π(a'|s')Q(s',a') - Q(s,a)]

### Méthodes de Planification et Apprentissage

Ces méthodes combinent l'apprentissage par renforcement avec la planification en utilisant un modèle appris de l'environnement.

#### Dyna-Q

1. Mise à jour de Q(s,a) à partir de l'expérience réelle
2. Mise à jour du modèle de l'environnement
3. Planification : mise à jour de Q(s,a) à partir d'expériences simulées générées par le modèle

#### Dyna-Q+

Extension de Dyna-Q qui encourage l'exploration des états et actions non visités depuis longtemps en ajoutant un bonus d'exploration.

## Exploration vs Exploitation

L'équilibre entre exploration (découvrir de nouvelles informations) et exploitation (utiliser les informations connues) est un défi central en apprentissage par renforcement.

### ε-greedy

Avec une probabilité ε, choisir une action aléatoire (exploration), sinon choisir l'action avec la valeur Q la plus élevée (exploitation).

### Softmax

Choisir une action avec une probabilité proportionnelle à e<sup>Q(s,a)/τ</sup>, où τ est un paramètre de température qui contrôle le degré d'exploration.

## Références

Pour approfondir ces concepts, nous recommandons les ressources suivantes :

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Szepesvári, C. (2010). Algorithms for reinforcement learning. Synthesis lectures on artificial intelligence and machine learning, 4(1), 1-103.
- Silver, D. (2015). UCL Course on RL. [http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)