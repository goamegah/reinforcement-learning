# Documentation du Framework d'Apprentissage par Renforcement

Bienvenue dans la documentation du framework d'apprentissage par renforcement. Ce framework fournit une implémentation modulaire et extensible de plusieurs algorithmes d'apprentissage par renforcement, ainsi que des environnements pour les tester.

## Table des matières

1. [Présentation générale](../README.md)
2. [Concepts théoriques](concepts.md)
3. [Algorithmes](algorithms.md)
4. [Environnements](environments.md)
5. [Utilitaires](utilities.md)
6. [Expérimentations](experiments.md)

## Présentation générale

Ce framework est conçu pour être utilisé à des fins éducatives et de recherche, permettant de comprendre et de comparer les performances des différents algorithmes d'apprentissage par renforcement.

Il implémente plusieurs familles d'algorithmes :
- Programmation Dynamique (DP)
- Méthodes de Monte Carlo (MC)
- Méthodes de Différence Temporelle (TD)
- Méthodes de Planification et Apprentissage (PN)

Et fournit plusieurs environnements pour les tester :
- GridWorld
- LineWorld
- MontyHall
- Rock-Paper-Scissors
- Environnements secrets

## Installation

Pour installer le framework, suivez les instructions dans le [README](../README.md#installation).

## Utilisation rapide

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

## Documentation détaillée

Pour plus de détails sur les différents composants du framework, consultez les sections suivantes :

- [Concepts théoriques](concepts.md) : Fondements mathématiques et théoriques de l'apprentissage par renforcement
- [Algorithmes](algorithms.md) : Description détaillée des algorithmes implémentés
- [Environnements](environments.md) : Description des environnements disponibles
- [Utilitaires](utilities.md) : Documentation des fonctions utilitaires
- [Expérimentations](experiments.md) : Description des expérimentations réalisées et des résultats obtenus