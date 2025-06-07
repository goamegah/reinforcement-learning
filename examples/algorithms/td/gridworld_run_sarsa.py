from rlearn.environments.grid_world import GridWorldMDP
from rlearn.algorithms.td.sarsa import sarsa, plot_scores
from rlearn.utils.io_utils import save_scores, save_policy, load_scores, load_policy
import os

env = GridWorldMDP()
policy, Q, scores = sarsa(env, nb_episodes=3000, gamma=0.95, alpha=0.1, epsilon=0.1)

# 📁 Création du dossier si nécessaire
os.makedirs("outputs/gridworld/sarsa", exist_ok=True)

# 💾 Sauvegarde
save_scores(scores, "outputs/gridworld/sarsa/scores.npy")
save_policy(policy, "outputs/gridworld/sarsa/policy.json")

# 📈 Affichage
plot_scores(scores)
print("Politique SARSA :")
print(policy.reshape(env.size, env.size))

# Affichage visuel état final
env.reset()
env.display()



# Chargement

# loaded_scores = load_scores("outputs/gridworld/sarsa/scores.npy")
# loaded_policy = load_policy("outputs/gridworld/sarsa/policy.json")

# print("Policy chargée :", loaded_policy)
# plot_scores(loaded_scores, title="SARSA - Policy rechargée")

