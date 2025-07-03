
# rlearn/algorithms/dp/policy_iteration.py


import numpy as np
from tqdm import tqdm
from reinfo.algorithms.dp.policy_evaluation import policy_evaluation


def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000, verbose=False):
    """
    Politique itérative : amélioration successive d'une politique jusqu'à convergence.
    :param env: environnement MDP compatible
    :param gamma: facteur d'actualisation
    :param theta: seuil de convergence pour policy_evaluation
    :param max_iterations: nombre max d'itérations
    :param verbose: si True, affiche l'évolution
    :return: politique optimale, valeurs V(s), scores (valeurs moyennes à chaque itération)
    """
    policy = np.zeros(env.num_states(), dtype=int)
    scores = []
    V = np.zeros(env.num_states())

    for i in tqdm(range(max_iterations), desc="Policy Iteration"):
        V, _ = policy_evaluation(env, policy, gamma, theta)
        scores.append(np.mean(V))  # score moyen (valeur des états)

        stable = True
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
            if best_action != old_action:
                stable = False

        if verbose:
            print(f"Iter {i} - Politique stable : {stable}")

        if stable:
            break

    return policy, V, scores
