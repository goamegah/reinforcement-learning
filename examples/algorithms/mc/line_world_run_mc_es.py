# examples/algorithms/mc/line_world_run_mc_es.py
from rlearn.environments.line_world import LineWorldMDP
from rlearn.algorithms.mc.mc_exploring_starts import mc_exploring_starts, plot_scores

env = LineWorldMDP(size=5)
policy, Q, scores = mc_exploring_starts(env, nb_episodes=10_000)

plot_scores(scores)

# Affichage de la politique apprise
print("Politique apprise :")
for s in range(env.num_states()):
    print(f"État {s} → Action {policy[s]}")
