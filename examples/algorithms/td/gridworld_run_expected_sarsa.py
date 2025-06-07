from rlearn.environments.grid_world import GridWorldMDP
from rlearn.algorithms.td.expected_sarsa import expected_sarsa, plot_scores

env = GridWorldMDP()
policy, Q, scores = expected_sarsa(env, nb_episodes=3000, gamma=0.95, alpha=0.1, epsilon=0.1)
plot_scores(scores)

print("Politique Expected SARSA (état → action) :")
print(policy.reshape(env.size, env.size))
