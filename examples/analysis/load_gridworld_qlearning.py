from rlearn.utils.io_utils import load_scores, load_policy, load_q_table
from rlearn.algorithms.td.q_learning import plot_scores

scores = load_scores("outputs/gridworld/q_learning/scores.npy")
policy = load_policy("outputs/gridworld/q_learning/policy.json")
Q = load_q_table("outputs/gridworld/q_learning/q_table.pkl")

print("Politique chargée :", policy)
plot_scores(scores, title="Q-Learning - Policy chargée")
