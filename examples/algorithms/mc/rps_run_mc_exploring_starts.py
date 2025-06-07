from rlearn.environments.rock_paper_scissors import RockPaperScissorsMDP
from rlearn.algorithms.mc.mc_exploring_starts import mc_exploring_starts, plot_scores

# Initialisation de l'environnement
env = RockPaperScissorsMDP()

# Entraînement
policy, Q, scores = mc_exploring_starts(env, gamma=1.0, nb_episodes=5000, epsilon=0.1)

# Affichage du graphique de scores
plot_scores(scores, title="RPS - MC Exploring Starts")

# Affichage politique
print("\nPolitique apprise (état -> action):")
for s in range(env.num_states()):
    print(f"État {s} → Action {policy[s]}")
