from rlearn.environments.grid_world import GridWorldMDP
from rlearn.algorithms.mc.mc_exploring_starts import mc_exploring_starts, plot_scores

# Initialisation de l'environnement
env = GridWorldMDP()

# Entraînement avec MC Exploring Starts
policy, Q, scores = mc_exploring_starts(env, gamma=0.95, nb_episodes=5000, epsilon=0.1)

# Affichage des scores
plot_scores(scores, title="GridWorld - MC Exploring Starts")

# Exemple d'affichage de la politique
print("\nPolitique apprise (état -> action):")
for s in range(env.num_states()):
    print(f"État {s} → Action {policy[s]}")
