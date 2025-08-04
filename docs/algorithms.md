# Documentation des Algorithmes

Ce document décrit les différents algorithmes d'apprentissage par renforcement implémentés dans le framework.

## Programmation Dynamique (DP)

Les algorithmes de programmation dynamique supposent une connaissance complète du modèle de l'environnement (MDP).

### Policy Iteration

**Module**: `reinfolearn.algorithm.dp.policy_iteration`

L'algorithme Policy Iteration alterne entre deux phases :
1. **Évaluation de la politique** : calcul des valeurs d'état pour une politique fixe
2. **Amélioration de la politique** : mise à jour de la politique en fonction des valeurs calculées

**Paramètres**:
- `env` : L'environnement (doit implémenter l'interface BaseEnvironment)
- `gamma` : Facteur d'actualisation (discount factor)
- `theta` : Seuil de convergence pour l'évaluation de la politique
- `max_iterations` : Nombre maximum d'itérations

**Retourne**:
- `policy` : La politique optimale (tableau numpy)
- `V` : Les valeurs d'état finales (tableau numpy)
- `mean_value_history` : L'historique des valeurs moyennes à chaque itération

### Value Iteration

**Module**: `reinfolearn.algorithm.dp.value_iteration`

L'algorithme Value Iteration combine l'évaluation et l'amélioration de la politique en une seule étape, en mettant à jour directement les valeurs d'état.

**Paramètres**:
- `env` : L'environnement (doit implémenter l'interface BaseEnvironment)
- `gamma` : Facteur d'actualisation (discount factor)
- `theta` : Seuil de convergence
- `max_iterations` : Nombre maximum d'itérations

**Retourne**:
- `policy` : La politique optimale (tableau numpy)
- `V` : Les valeurs d'état finales (tableau numpy)
- `delta_history` : L'historique des variations maximales à chaque itération

## Méthodes de Monte Carlo (MC)

Les méthodes de Monte Carlo apprennent à partir d'épisodes complets d'expérience.

### Monte Carlo avec Exploring Starts

**Module**: `reinfolearn.algorithm.mc.mc_exploring_starts`

Cette méthode garantit l'exploration en commençant chaque épisode par un état-action aléatoire.

**Paramètres**:
- `env` : L'environnement
- `nb_episodes` : Nombre d'épisodes
- `gamma` : Facteur d'actualisation

**Retourne**:
- `policy` : La politique apprise
- `Q` : La table Q finale
- `episode_scores` : Les scores obtenus à chaque épisode

### Monte Carlo On-Policy (First-Visit)

**Module**: `reinfolearn.algorithm.mc.mc_on_policy`

Cette méthode utilise une politique ε-greedy pour assurer l'exploration, et met à jour les valeurs Q lors de la première visite d'une paire état-action dans un épisode.

**Paramètres**:
- `env` : L'environnement
- `nb_episodes` : Nombre d'épisodes
- `gamma` : Facteur d'actualisation
- `epsilon` : Paramètre d'exploration pour la politique ε-greedy

**Retourne**:
- `policy` : La politique apprise
- `Q` : La table Q finale
- `episode_scores` : Les scores obtenus à chaque épisode

### Monte Carlo Off-Policy

**Module**: `reinfolearn.algorithm.mc.mc_off_policy`

Cette méthode utilise deux politiques : une politique comportementale pour l'exploration et une politique cible pour l'apprentissage.

**Paramètres**:
- `env` : L'environnement
- `nb_episodes` : Nombre d'épisodes
- `gamma` : Facteur d'actualisation

**Retourne**:
- `policy` : La politique apprise
- `Q` : La table Q finale
- `episode_scores` : Les scores obtenus à chaque épisode

## Méthodes de Différence Temporelle (TD)

Les méthodes TD apprennent à partir d'expériences partielles et mettent à jour les estimations en fonction des estimations suivantes.

### Q-Learning

**Module**: `reinfolearn.algorithm.td.q_learning`

Q-Learning est une méthode off-policy qui apprend directement la fonction de valeur d'action optimale, indépendamment de la politique suivie.

**Paramètres**:
- `env` : L'environnement
- `nb_episodes` : Nombre d'épisodes
- `gamma` : Facteur d'actualisation
- `alpha` : Taux d'apprentissage
- `epsilon` : Paramètre d'exploration pour la politique ε-greedy

**Retourne**:
- `policy` : La politique apprise
- `Q` : La table Q finale
- `episode_scores` : Les scores obtenus à chaque épisode

### SARSA

**Module**: `reinfolearn.algorithm.td.sarsa`

SARSA (State-Action-Reward-State-Action) est une méthode on-policy qui apprend la fonction de valeur d'action pour la politique comportementale.

**Paramètres**:
- `env` : L'environnement
- `nb_episodes` : Nombre d'épisodes
- `gamma` : Facteur d'actualisation
- `alpha` : Taux d'apprentissage
- `epsilon` : Paramètre d'exploration pour la politique ε-greedy

**Retourne**:
- `policy` : La politique apprise
- `Q` : La table Q finale
- `episode_scores` : Les scores obtenus à chaque épisode

### Expected SARSA

**Module**: `reinfolearn.algorithm.td.expected_sarsa`

Expected SARSA est une variante de SARSA qui utilise l'espérance de la valeur d'action suivante plutôt que la valeur d'une action spécifique.

**Paramètres**:
- `env` : L'environnement
- `nb_episodes` : Nombre d'épisodes
- `gamma` : Facteur d'actualisation
- `alpha` : Taux d'apprentissage
- `epsilon` : Paramètre d'exploration pour la politique ε-greedy

**Retourne**:
- `policy` : La politique apprise
- `Q` : La table Q finale
- `episode_scores` : Les scores obtenus à chaque épisode

## Méthodes de Planification et Apprentissage (PN)

Ces méthodes combinent l'apprentissage par renforcement avec la planification en utilisant un modèle appris de l'environnement.

### Dyna-Q

**Module**: `reinfolearn.algorithm.pn.dyna_q`

Dyna-Q combine Q-Learning avec la planification en utilisant un modèle appris de l'environnement pour générer des expériences supplémentaires.

**Paramètres**:
- `env` : L'environnement
- `nb_episodes` : Nombre d'épisodes
- `gamma` : Facteur d'actualisation
- `alpha` : Taux d'apprentissage
- `epsilon` : Paramètre d'exploration pour la politique ε-greedy
- `planning_steps` : Nombre d'étapes de planification après chaque interaction réelle

**Retourne**:
- `policy` : La politique apprise
- `Q` : La table Q finale
- `episode_scores` : Les scores obtenus à chaque épisode

### Dyna-Q+

**Module**: `reinfolearn.algorithm.pn.dyna_q_plus`

Dyna-Q+ est une extension de Dyna-Q qui encourage l'exploration des états et actions non visités depuis longtemps.

**Paramètres**:
- `env` : L'environnement
- `nb_episodes` : Nombre d'épisodes
- `gamma` : Facteur d'actualisation
- `alpha` : Taux d'apprentissage
- `epsilon` : Paramètre d'exploration pour la politique ε-greedy
- `planning_steps` : Nombre d'étapes de planification après chaque interaction réelle
- `kappa` : Paramètre de bonus d'exploration

**Retourne**:
- `policy` : La politique apprise
- `Q` : La table Q finale
- `episode_scores` : Les scores obtenus à chaque épisode