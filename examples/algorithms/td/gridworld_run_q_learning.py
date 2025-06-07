from rlearn.environments.grid_world import GridWorldMDP
from rlearn.algorithms.td.q_learning import q_learning, plot_scores
from rlearn.utils.io_utils import save_scores, save_policy, save_q_table
import os

env = GridWorldMDP()
policy, Q, scores = q_learning(env, nb_episodes=3000, gamma=0.95, alpha=0.1, epsilon=0.1)

# 📁 Création du dossier
out_dir = "outputs/gridworld/q_learning"
os.makedirs(out_dir, exist_ok=True)

# 💾 Sauvegarde
save_scores(scores, f"{out_dir}/scores.npy")
save_policy(policy, f"{out_dir}/policy.json")
save_q_table(Q, f"{out_dir}/q_table.pkl")

# 📈 Affichage
plot_scores(scores)
print("Politique Q-Learning :")
print(policy.reshape(env.height, env.width))



scores = load_scores("outputs/gridworld/q_learning/scores.npy")
policy = load_policy("outputs/gridworld/q_learning/policy.json")
Q = load_q_table("outputs/gridworld/q_learning/q_table.pkl")

print("Politique chargée :", policy)
plot_scores(scores, title="Q-Learning - Policy chargée")
