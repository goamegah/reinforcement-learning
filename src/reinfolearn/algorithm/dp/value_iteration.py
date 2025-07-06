# rlearn/algorithms/dp/value_iteration.py

import numpy as np
from tqdm import tqdm

def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000, verbose=False):
    """
    Value Iteration - Résolution d'un MDP par itération sur la valeur.
    
    :param env: environnement MDP compatible (avec p, reward, num_states, num_actions)
    :param gamma: facteur d'actualisation (0 < gamma <= 1)
    :param theta: seuil de convergence
    :param max_iterations: nombre maximal d'itérations
    :param verbose: affichage détaillé
    :return: politique optimale, valeurs V, historique des deltas
    """
    V = np.zeros(env.num_states())
    delta_history = []

    for it in tqdm(range(max_iterations), desc="Value Iteration"):
        delta = 0
        for s in range(env.num_states()):
            v = V[s]

            # Calcul de max_a Q(s,a)
            action_values = np.zeros(env.num_actions())
            for a in range(env.num_actions()):
                for s_p in range(env.num_states()):
                    for r_idx in range(env.num_rewards()):
                        p = env.p(s, a, s_p, r_idx)
                        r = env.reward(r_idx)
                        action_values[a] += p * (r + gamma * V[s_p])

            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))

        delta_history.append(delta)

        if verbose:
            print(f"Iteration {it}: Δ = {delta:.6f}")

        if delta < theta:
            break

    # Politique optimale : argmax_a Q(s,a)
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

    return policy, V, delta_history


def plot_value_iteration_convergence(delta_history, title="Value Iteration Convergence"):
    """
    Affiche la convergence de Δ au cours des itérations.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(delta_history)
    plt.title(title)
    plt.xlabel("Itération")
    plt.ylabel("Δ (écart max)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



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
