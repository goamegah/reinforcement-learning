# rlearn/algorithms/dp/value_iteration.py

import numpy as np
from tqdm import tqdm


def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000, verbose=False):
    """
    Itération sur la valeur (Value Iteration) pour résoudre un MDP.
    :param env: environnement MDP
    :param gamma: facteur d'actualisation
    :param theta: seuil de convergence
    :param max_iterations: nombre max d'itérations
    :param verbose: si True, affiche les détails
    :return: politique optimale, valeurs V, historique des scores
    """
    V = np.zeros(env.num_states())
    scores = []

    for i in tqdm(range(max_iterations), desc="Value Iteration"):
        delta = 0
        for s in range(env.num_states()):
            v = V[s]
            action_values = np.zeros(env.num_actions())

            for a in range(env.num_actions()):
                for s_p in range(env.num_states()):
                    for r_idx in range(env.num_rewards()):
                        p = env.p(s, a, s_p, r_idx)
                        r = env.reward(r_idx)
                        action_values[a] += p * (r + gamma * V[s_p])

            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))

        scores.append(np.mean(V))

        if verbose:
            print(f"Iter {i}: Δ = {delta:.6f}")

        if delta < theta:
            break

    # Déduire la politique optimale
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        action_values = np.zeros(env.num_actions())
        for a in range(env.num_actions()):
            for s_p in range(env.num_states()):
                for r_idx in range(env.num_rewards()):
                    p = env.p(s, a, s_p, r_idx)
                    r = env.reward(r_idx)
                    action_values[a] += p * (r + gamma * V[s_p])
        policy[s] = np.argmax(action_values)

    return policy, V, scores



# def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000, verbose=False):
#     """
#     Algorithme de Value Iteration pour les environnements MDP.

#     :param env: environnement compatible (avec p, reward, num_states, etc.)
#     :param gamma: facteur de discount
#     :param theta: seuil de convergence
#     :param max_iterations: maximum d’itérations
#     :param verbose: affichage intermédiaire
#     :return: V, policy, delta_history
#     """
#     V = np.zeros(env.num_states())
#     delta_history = []

#     for it in tqdm(range(max_iterations), desc="Value Iteration"):
#         delta = 0
#         V_new = V.copy()

#         for s in range(env.num_states()):
#             action_values = []
#             for a in range(env.num_actions()):
#                 value = 0
#                 for s_p in range(env.num_states()):
#                     for r_idx in range(env.num_rewards()):
#                         p = env.p(s, a, s_p, r_idx)
#                         r = env.reward(r_idx)
#                         value += p * (r + gamma * V[s_p])
#                 action_values.append(value)

#             best_value = max(action_values)
#             delta = max(delta, abs(V[s] - best_value))
#             V_new[s] = best_value

#         V = V_new
#         delta_history.append(delta)

#         if verbose:
#             print(f"Iter {it}: Δ = {delta:.6f}")

#         if delta < theta:
#             break

#     # Politique extraite à la fin
#     policy = np.zeros(env.num_states(), dtype=int)
#     for s in range(env.num_states()):
#         action_values = []
#         for a in range(env.num_actions()):
#             value = 0
#             for s_p in range(env.num_states()):
#                 for r_idx in range(env.num_rewards()):
#                     p = env.p(s, a, s_p, r_idx)
#                     r = env.reward(r_idx)
#                     value += p * (r + gamma * V[s_p])
#             action_values.append(value)
#         policy[s] = np.argmax(action_values)

#     return V, policy, delta_history


# def plot_value_iteration_convergence(delta_history, title="Value Iteration Convergence"):
#     plt.figure(figsize=(8, 4))
#     plt.plot(delta_history)
#     plt.xlabel("Itération")
#     plt.ylabel("Δ (écart max)")
#     plt.title(title)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     from rlearn.environments.line_world import LineWorldMDP
#     # from rlearn.environments.grid_world import GridWorldMDP

#     env = LineWorldMDP(size=5)
#     V, policy, history = value_iteration(env, gamma=0.99, theta=1e-6, verbose=True)

#     print("\n🎯 Politique optimale :")
#     for s in range(env.num_states()):
#         print(f"État {s} → action {policy[s]}")

#     print("\n📈 Valeurs d'état :")
#     for s, v in enumerate(V):
#         print(f"État {s} → V = {v:.3f}")

#     plot_value_iteration_convergence(history)
