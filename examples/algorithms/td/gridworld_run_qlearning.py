from rlearn.environments.grid_world import GridWorldMDP
from rlearn.algorithms.td.q_learning import q_learning, plot_scores

env = GridWorldMDP()
policy, Q, scores = q_learning(env, nb_episodes=3000, gamma=0.95, alpha=0.1, epsilon=0.1)
plot_scores(scores)

print("Politique Q-Learning (état → action) :")
print(policy.reshape(env.size, env.size))
