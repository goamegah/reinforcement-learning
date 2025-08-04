# Documentation des Environnements

Ce document décrit les différents environnements disponibles dans le framework pour tester les algorithmes d'apprentissage par renforcement.

## Interface commune

Tous les environnements implémentent l'interface `BaseEnvironment` définie dans `reinfolearn.environment.base_environment`. Cette interface garantit que tous les environnements peuvent être utilisés avec tous les algorithmes du framework.

### Méthodes principales

#### Méthodes liées au MDP
- `num_states()` : Retourne le nombre total d'états dans l'environnement
- `num_actions()` : Retourne le nombre total d'actions possibles
- `num_rewards()` : Retourne le nombre total de récompenses distinctes
- `reward(index)` : Retourne la récompense associée à un index donné
- `p(s, a, s_p, r_index)` : Retourne la probabilité de transition P(s', r | s, a)

#### Méthodes pour Monte Carlo et TD
- `state_id()` : Retourne l'identifiant unique de l'état courant
- `reset()` : Réinitialise l'environnement
- `display()` : Affiche l'environnement à l'état courant
- `is_forbidden(action)` : Indique si une action est interdite dans l'état courant
- `is_game_over()` : Indique si l'épisode est terminé
- `available_actions()` : Retourne les actions valides à l'état courant
- `step(action)` : Exécute l'action spécifiée
- `score()` : Retourne le score global de l'épisode

## Environnements disponibles

### GridWorldMDP

**Module**: `reinfolearn.environment.grid_world`

Un monde en grille 2D où l'agent peut se déplacer dans quatre directions (haut, bas, gauche, droite). L'environnement contient des états terminaux avec des récompenses positives et négatives.

**Paramètres du constructeur**:
- `size` : Taille de la grille (par défaut: 5x5)

**Caractéristiques**:
- États: `size * size`
- Actions: 4 (haut, bas, gauche, droite)
- Récompenses: -3.0 (échec), 0.0 (déplacement), 1.0 (succès)
- États terminaux: coin supérieur droit (récompense négative) et coin inférieur droit (récompense positive)

### LineWorldMDP

**Module**: `reinfolearn.environment.line_world`

Un monde linéaire simple où l'agent peut se déplacer à gauche ou à droite. L'objectif est d'atteindre l'extrémité droite tout en évitant l'extrémité gauche.

**Paramètres du constructeur**:
- `size` : Longueur de la ligne (par défaut: 7)

**Caractéristiques**:
- États: `size`
- Actions: 2 (gauche, droite)
- Récompenses: -1.0 (échec), 0.0 (déplacement), 1.0 (succès)
- États terminaux: extrémité gauche (récompense négative) et extrémité droite (récompense positive)

### MontyHallLevel1

**Module**: `reinfolearn.environment.montyhall_level1`

Une implémentation du problème de Monty Hall, où l'agent doit choisir une porte, puis décider de rester sur son choix initial ou de changer après que l'hôte ait révélé une porte vide.

**Caractéristiques**:
- États: 7 (représentant les différentes étapes du jeu)
- Actions: 3 (choisir porte 0, 1 ou 2)
- Récompenses: 0.0 (pas de récompense), 1.0 (gagner la voiture)

### MontyHallLevel2

**Module**: `reinfolearn.environment.montyhall_level2`

Une version plus complexe du problème de Monty Hall, avec plus d'états pour représenter plus précisément les différentes étapes du jeu.

**Caractéristiques**:
- États: Plus nombreux que dans Level1
- Actions: 3 (choisir porte 0, 1 ou 2)
- Récompenses: 0.0 (pas de récompense), 1.0 (gagner la voiture)

### RockPaperScissors

**Module**: `reinfolearn.environment.rock_paper_scissors`

Une implémentation du jeu Pierre-Papier-Ciseaux, où l'agent joue contre un adversaire qui choisit aléatoirement.

**Caractéristiques**:
- États: 1 (un seul état initial)
- Actions: 3 (pierre, papier, ciseaux)
- Récompenses: -1.0 (perdre), 0.0 (match nul), 1.0 (gagner)

### RockPaperScissorsTwoRounds

**Module**: `reinfolearn.environment.rock_paper_scissors_two_rounds`

Une extension du jeu Pierre-Papier-Ciseaux sur deux tours, où l'agent doit gagner les deux tours pour obtenir la récompense maximale.

**Caractéristiques**:
- États: 10 (représentant les différentes combinaisons de résultats du premier tour)
- Actions: 3 (pierre, papier, ciseaux)
- Récompenses: Diverses valeurs selon les résultats des deux tours

## Environnements secrets

Le framework inclut également des environnements "secrets" utilisés pour les expérimentations. Ces environnements sont accessibles via des wrappers dans le module `experiments.secret_envs_wrapper`.

### SecretEnv0

Un environnement secret avec les caractéristiques suivantes:
- États: 16
- Actions: 4
- Récompenses: 3 valeurs distinctes

### SecretEnv1

Un environnement secret avec les caractéristiques suivantes:
- États: 64
- Actions: 4
- Récompenses: 3 valeurs distinctes

### SecretEnv2

Un environnement secret avec les caractéristiques suivantes:
- États: 100
- Actions: 4
- Récompenses: 3 valeurs distinctes

### SecretEnv3

Un environnement secret avec les caractéristiques suivantes:
- États: 81
- Actions: 4
- Récompenses: 3 valeurs distinctes

## Utilisation des environnements

Exemple d'utilisation d'un environnement:

```python
from reinfolearn.environment.grid_world import GridWorldMDP

# Créer l'environnement
env = GridWorldMDP(size=5)

# Réinitialiser l'environnement
env.reset()

# Afficher l'environnement
env.display()

# Obtenir les actions valides
valid_actions = env.available_actions()

# Exécuter une action
env.step(action=1)  # Aller vers le bas

# Vérifier si l'épisode est terminé
if env.is_game_over():
    print(f"Score final: {env.score()}")
```