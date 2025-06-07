
# rlearn/algorithms/dp/policy_evaluation.py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def policy_evaluation(env, policy, gamma=0.99, theta=1e-6, max_iterations=1000, verbose=False):
    """
    Évalue une politique donnée pour un environnement MDP compatible.
    :param env: Environnement avec p(s,a,s',r), num_states, num_actions, etc.
    :param policy: tableau [s] -> a (politique à évaluer)
    :param gamma: facteur d’actualisation
    :param theta: seuil pour l’arrêt de la convergence
    :param max_iterations: nombre maximal d’itérations
    :param verbose: si True, affiche l’évolution des erreurs
    :return: V (valeurs), history (liste des erreurs à chaque itération)
    """
    V = np.zeros(env.num_states())
    delta_history = []

    for i in tqdm(range(max_iterations), desc="Policy Evaluation"):
        delta = 0
        V_new = V.copy()

        for s in range(env.num_states()):
            a = policy[s]
            v = 0
            for s_p in range(env.num_states()):
                for r_idx in range(env.num_rewards()):
                    p = env.p(s, a, s_p, r_idx)
                    r = env.reward(r_idx)
                    v += p * (r + gamma * V[s_p])
            delta = max(delta, abs(V[s] - v))
            V_new[s] = v

        V = V_new
        delta_history.append(delta)

        if verbose:
            print(f"Iter {i}: Δ = {delta}")

        if delta < theta:
            break

    return V, delta_history


def plot_convergence(delta_history, title="Policy Evaluation Convergence"):
    plt.figure(figsize=(8, 4))
    plt.plot(delta_history)
    plt.xlabel("Itération")
    plt.ylabel("Δ (écart max)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     from rlearn.environments.line_world import LineWorldMDP
#     from rlearn.environments.grid_world import GridWorldMDP

#     env = GridWorldMDP(size=5)
#     # env = LineWorldMDP(size=5)

#     policy = np.zeros(env.num_states(), dtype=int)  # politique fixe : aller à gauche

#     V, history = policy_evaluation(env, policy, gamma=0.99, theta=1e-6, verbose=True)

#     for s, v in enumerate(V):
#         print(f"État {s} → V = {v:.3f}")

#     plot_convergence(history)
