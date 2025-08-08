# Reinforcement Learning Framework

Ce projet est un framework d'apprentissage par renforcement implémentant divers algorithmes classiques et permettant de les comparer sur différents environnements.

## Description

Ce framework fournit une implémentation modulaire et extensible de plusieurs algorithmes d'apprentissage par renforcement, ainsi que des environnements pour les tester. Il est conçu pour être utilisé à des fins éducatives et de recherche, permettant de comprendre et de comparer les performances des différents algorithmes.

### Algorithmes implémentés

Le framework inclut les algorithmes suivants :

#### Programmation Dynamique (DP)
- Policy Iteration
- Value Iteration

#### Méthodes de Monte Carlo (MC)
- Monte Carlo avec Exploring Starts
- Monte Carlo On-Policy (First-Visit)
- Monte Carlo Off-Policy

#### Méthodes de Différence Temporelle (TD)
- Q-Learning
- SARSA
- Expected SARSA

#### Méthodes de Planification et Apprentissage (PN)
- Dyna-Q
- Dyna-Q+

### Environnements disponibles

Le framework inclut plusieurs environnements pour tester les algorithmes :

- GridWorld : un monde en grille 2D avec des états terminaux
- LineWorld : un monde linéaire simple
- MontyHall : une implémentation du problème de Monty Hall
- Rock-Paper-Scissors : le jeu pierre-papier-ciseaux
- Environnements secrets : des environnements spécifiques pour les expérimentations

## Structure du projet

```
reinforcement-learning/
├── assets/                  # Ressources statiques (images, etc.)
├── experiments/             # Notebooks et scripts d'expérimentation
│   ├── libs/               # Bibliothèques pour les environnements secrets
│   └── outputs/            # Résultats des expérimentations
├── src/                     # Code source principal
│   └── reinfolearn/        # Package principal
│       ├── algorithm/      # Implémentation des algorithmes
│       │   ├── dp/        # Programmation dynamique
│       │   ├── mc/        # Méthodes Monte Carlo
│       │   ├── td/        # Méthodes TD
│       │   └── pn/        # Méthodes de planification
│       ├── display/       # Fonctions d'affichage des environnements
│       ├── environment/   # Implémentation des environnements
│       ├── evaluation/    # Outils d'évaluation des politiques
│       └── utils/         # Utilitaires (IO, métriques, visualisation)
└── tests/                   # Tests unitaires et d'intégration
```

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances (Linux)

```bash
git clone git@github.com:goamegah/reinforcement-learning-libs.git
cd reinforcement-learning-libs
python -m venv rl-env
source rl-env\bin\activate
pip install -e .
```

## Utilisation

### Exemples de base

Voici un exemple simple d'utilisation de l'algorithme Q-Learning sur l'environnement GridWorld :

```python
from reinfolearn.environment.grid_world import GridWorldMDP
from reinfolearn.algorithm.td.q_learning import q_learning
from reinfolearn.utils.plot_utils import plot_scores

# Créer l'environnement
env = GridWorldMDP(size=5)

# Exécuter l'algorithme Q-Learning
policy, Q, scores = q_learning(env, nb_episodes=1000, gamma=0.99, alpha=0.1, epsilon=0.1)

# Visualiser les résultats
plot_scores(scores, title="Q-Learning sur GridWorld")
```

### Exécution des expérimentations

Les notebooks d'expérimentation dans le dossier `experiments/` permettent de comparer les performances des différents algorithmes sur divers environnements.

```bash
jupyter notebook experiments/experimentation_secret_envs.ipynb
```

## Expérimentations

Le projet inclut plusieurs notebooks d'expérimentation qui comparent les performances des différents algorithmes sur divers environnements. Ces expérimentations permettent de comprendre les forces et les faiblesses de chaque algorithme dans différents contextes.

Les résultats des expérimentations sont sauvegardés dans le dossier `experiments/outputs/` et peuvent être analysés à l'aide des outils de visualisation fournis dans le module `reinfolearn.utils.plot_utils`.