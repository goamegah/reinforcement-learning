# Documentation des Utilitaires

Ce document décrit les différents modules utilitaires disponibles dans le framework pour faciliter l'utilisation, l'évaluation et la visualisation des algorithmes d'apprentissage par renforcement.

## Utilitaires d'entrée/sortie

**Module**: `reinfolearn.utils.io_utils`

Ce module fournit des fonctions pour sauvegarder et charger différents types de données générées par les algorithmes d'apprentissage par renforcement.

### Fonctions principales

#### Gestion des scores

- `save_scores(scores, path)`: Sauvegarde un tableau de scores (liste ou np.array) au format .npy
- `load_scores(path)`: Charge un tableau de scores depuis un fichier .npy

#### Gestion des politiques

- `save_policy(policy, path)`: Sauvegarde une politique (tableau numpy) au format JSON
- `load_policy(path)`: Charge une politique depuis un fichier JSON

#### Gestion des tables Q

- `save_q_table(Q, path)`: Sauvegarde une Q-table (dictionnaire d'état -> actions) au format pickle
- `load_q_table(path)`: Charge une Q-table depuis un fichier pickle

#### Gestion des valeurs d'état

- `save_values(V, path)`: Sauvegarde un vecteur de valeurs V (np.array) au format .npy
- `load_values(path)`: Charge un vecteur V depuis un fichier .npy

## Utilitaires de métriques

**Module**: `reinfolearn.utils.metrics_utils`

Ce module fournit des fonctions pour évaluer les performances des politiques apprises.

### Fonctions principales

- `evaluate_policy_score(env, policy, nb_episodes=500, seed=None)`: Évalue une politique donnée sur un environnement en moyenne sur plusieurs épisodes
  - **Paramètres**:
    - `env`: L'environnement à utiliser
    - `policy`: La politique à évaluer (tableau numpy)
    - `nb_episodes`: Nombre d'épisodes à jouer
    - `seed`: Graine aléatoire pour reproductibilité
  - **Retourne**: Liste des scores obtenus

- `summarize_policy_scores(scores)`: Donne un résumé statistique des scores de la politique
  - **Paramètres**:
    - `scores`: Liste des scores obtenus
  - **Retourne**: Dictionnaire avec moyenne, écart-type, minimum et maximum

- `compute_success_rate(scores, threshold=1.0)`: Calcule le taux d'épisodes avec un score supérieur ou égal à un seuil
  - **Paramètres**:
    - `scores`: Liste des scores
    - `threshold`: Seuil considéré comme succès
  - **Retourne**: Taux de succès (float entre 0 et 1)

## Utilitaires de visualisation

**Module**: `reinfolearn.utils.plot_utils`

Ce module fournit des fonctions pour visualiser les résultats des algorithmes d'apprentissage par renforcement.

### Fonctions principales

- `plot_convergence(values, title="Convergence", ylabel="Value", save_path=None)`: Affiche une courbe de convergence
  - **Paramètres**:
    - `values`: Valeurs à tracer
    - `title`: Titre du graphique
    - `ylabel`: Étiquette de l'axe Y
    - `save_path`: Chemin pour sauvegarder la figure (optionnel)

- `plot_scores(scores, title="Score par épisode", save_path=None)`: Affiche les scores par épisode
  - **Paramètres**:
    - `scores`: Scores à tracer
    - `title`: Titre du graphique
    - `save_path`: Chemin pour sauvegarder la figure (optionnel)

- `plot_episode_scores(scores, window=100, title="Scores par épisode")`: Affiche la moyenne glissante des scores par épisode
  - **Paramètres**:
    - `scores`: Scores à tracer
    - `window`: Taille de la fenêtre pour la moyenne glissante
    - `title`: Titre du graphique

- `plot_delta_history(delta_history, title="Convergence (delta max)")`: Affiche la variation max entre deux itérations
  - **Paramètres**:
    - `delta_history`: Historique des variations
    - `title`: Titre du graphique

- `plot_vi_convergence(delta_history, mean_value_history)`: Affiche une courbe double avec delta max et moyenne des valeurs
  - **Paramètres**:
    - `delta_history`: Historique des variations
    - `mean_value_history`: Historique des valeurs moyennes

- `plot_scores_mavg(scores, window=100, title="Score moyen")`: Affiche la moyenne mobile des scores
  - **Paramètres**:
    - `scores`: Scores à tracer
    - `window`: Taille de la fenêtre pour la moyenne mobile
    - `title`: Titre du graphique

## Exemples d'utilisation

### Évaluation et sauvegarde d'une politique

```python
from reinfolearn.environment.grid_world import GridWorldMDP
from reinfolearn.algorithm.td.q_learning import q_learning
from reinfolearn.utils.metrics_utils import evaluate_policy_score, summarize_policy_scores
from reinfolearn.utils.io_utils import save_policy, save_q_table

# Créer l'environnement et exécuter Q-Learning
env = GridWorldMDP(size=5)
policy, Q, _ = q_learning(env, nb_episodes=1000, gamma=0.99, alpha=0.1, epsilon=0.1)

# Évaluer la politique
scores = evaluate_policy_score(env, policy, nb_episodes=500)
summary = summarize_policy_scores(scores)
print(f"Performance moyenne: {summary['mean']:.2f} ± {summary['std']:.2f}")

# Sauvegarder la politique et la table Q
save_policy(policy, "outputs/grid_world/q_learning/policy.json")
save_q_table(Q, "outputs/grid_world/q_learning/q_table.pkl")
```

### Visualisation des résultats

```python
from reinfolearn.utils.plot_utils import plot_scores, plot_scores_mavg
from reinfolearn.utils.io_utils import load_scores

# Charger les scores
scores = load_scores("outputs/grid_world/q_learning/episode_scores.npy")

# Afficher les scores bruts
plot_scores(scores, title="Q-Learning sur GridWorld")

# Afficher la moyenne mobile
plot_scores_mavg(scores, window=100, title="Score moyen - Q-Learning sur GridWorld")
```