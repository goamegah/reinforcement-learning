# rlearn/algorithms/dp/policy_iteration.py

import numpy as np
from tqdm import tqdm

def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    """
    Policy Evaluation: évalue la valeur V(s) pour une politique donnée.
    """
    V = np.zeros(env.num_states())
    delta_history = []

    while True:
        delta = 0
        for s in range(env.num_states()):
            v = V[s]
            a = policy[s]
            v_new = 0.0
            for s_p in range(env.num_states()):
                for r_idx in range(env.num_rewards()):
                    p = env.p(s, a, s_p, r_idx)
                    r = env.reward(r_idx)
                    v_new += p * (r + gamma * V[s_p])
            V[s] = v_new
            delta = max(delta, abs(v - v_new))
        delta_history.append(delta)
        if delta < theta:
            break
    return V, delta_history

def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000, verbose=False):
    """
    Implémentation pédagogique et conforme de Policy Iteration.
    """
    policy = np.zeros(env.num_states(), dtype=int)  # politique initiale aléatoire (0 partout)
    V = np.zeros(env.num_states())
    
    scores = []

    for i in tqdm(range(max_iterations), desc="Policy Iteration"):
        # === Step 1. Policy Evaluation ===
        V, delta_history = policy_evaluation(env, policy, gamma, theta)
        scores.append(np.mean(V))

        # === Step 2. Policy Improvement ===
        policy_stable = True
        for s in range(env.num_states()):
            old_action = policy[s]
            action_values = np.zeros(env.num_actions())

            for a in range(env.num_actions()):
                for s_p in range(env.num_states()):
                    for r_idx in range(env.num_rewards()):
                        p = env.p(s, a, s_p, r_idx)
                        r = env.reward(r_idx)
                        action_values[a] += p * (r + gamma * V[s_p])

            best_action = np.argmax(action_values)
            policy[s] = best_action

            if old_action != best_action:
                policy_stable = False

        if verbose:
            print(f"Iteration {i}: Mean V = {np.mean(V):.4f}, Policy Stable = {policy_stable}")

        if policy_stable:
            print(f"✅ Policy stable trouvée à l'itération {i}")
            break

    return policy, V, scores


def plot_policy_iteration_scores(scores, title="Policy Iteration - Convergence"):
    """
    Plot de l'évolution des valeurs moyennes V(s) au fil des itérations.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(scores)
    plt.title(title)
    plt.xlabel("Itération")
    plt.ylabel("Valeur moyenne V(s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()







# # rlearn/algorithms/dp/policy_iteration.py

# import numpy as np
# from tqdm import tqdm
# from reinfolearn.algorithm.dp.policy_evaluation import policy_evaluation


# def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000, verbose=False):
#     """
#     Politique itérative : amélioration successive d'une politique jusqu'à convergence.
#     :param env: environnement MDP compatible
#     :param gamma: facteur d'actualisation
#     :param theta: seuil de convergence pour policy_evaluation
#     :param max_iterations: nombre max d'itérations
#     :param verbose: si True, affiche l'évolution
#     :return: politique optimale, valeurs V(s), scores (valeurs moyennes à chaque itération)
#     """
#     policy = np.zeros(env.num_states(), dtype=int)
#     scores = []
#     V = np.zeros(env.num_states())

#     for i in tqdm(range(max_iterations), desc="Policy Iteration"):
#         V, _ = policy_evaluation(env, policy, gamma, theta)
#         scores.append(np.mean(V))  # score moyen (valeur des états)

#         stable = True
#         for s in range(env.num_states()):
#             old_action = policy[s]
#             action_values = np.zeros(env.num_actions())
#             for a in range(env.num_actions()):
#                 for s_p in range(env.num_states()):
#                     for r_idx in range(env.num_rewards()):
#                         p = env.p(s, a, s_p, r_idx)
#                         r = env.reward(r_idx)
#                         action_values[a] += p * (r + gamma * V[s_p])
#             best_action = np.argmax(action_values)
#             policy[s] = best_action
#             if best_action != old_action:
#                 stable = False

#         if verbose:
#             print(f"Iter {i} - Politique stable : {stable}")

#         if stable:
#             break

#     return policy, V, scores
