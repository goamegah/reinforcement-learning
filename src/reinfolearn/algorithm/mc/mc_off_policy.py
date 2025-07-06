# rlearn/algorithms/mc/mc_off_policy.py

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

def mc_off_policy_control(env, nb_episodes=5000, gamma=0.99, epsilon=0.1):
    """
    Off-policy MC Control with Weighted Importance Sampling (Sutton & Barto Fig. 5.6)
    Compatible SecretEnv0-3 (reward immediate via p*r).
    """
    Q = defaultdict(dict)
    C = defaultdict(lambda: defaultdict(float))
    scores = []

    for ep in tqdm(range(nb_episodes), desc="Off-policy MC Control"):
        env.reset()
        episode = []

        # === Génération de l'épisode sous policy comportementale b (ε-soft) ===
        while not env.is_game_over():
            s = env.state_id()
            actions = env.available_actions()

            # ε-soft policy
            if np.random.rand() < epsilon:
                a = np.random.choice(actions)
            else:
                q_vals = [Q[s].get(a_, 0.0) for a_ in actions]
                max_q = max(q_vals)
                best_actions = [a_ for a_, q in zip(actions, q_vals) if q == max_q]
                a = np.random.choice(best_actions)

            env.step(a)
            episode.append((s, a))

        scores.append(env.score())

        # === Backward update with Importance Sampling ===
        G = 0.0
        W = 1.0

        for t in reversed(range(len(episode))):
            s, a = episode[t]

            # ✅ Reward immediate (MDP definition)
            r = 0.0
            for s_p in range(env.num_states()):
                for r_idx in range(env.num_rewards()):
                    p_trans = env.p(s, a, s_p, r_idx)
                    r_val = env.reward(r_idx)
                    r += p_trans * r_val

            G = gamma * G + r

            if a not in Q[s]:
                Q[s][a] = 0.0

            C[s][a] += W
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])

            # Policy target greedy
            pi_a = max(Q[s], key=Q[s].get)

            if a != pi_a:
                break

            # Importance sampling ratio
            prob_b = epsilon / len(actions)
            if a == pi_a:
                prob_b += (1 - epsilon)

            W *= 1.0 / prob_b

            if W == 0.0:
                break

    # === Politique finale ===
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in Q and Q[s]:
            policy[s] = max(Q[s], key=Q[s].get)
        else:
            policy[s] = 0

    return policy, Q, scores

def plot_scores(scores, window=100, title="Off-policy MC Control - Score moyen"):
    moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid") if len(scores) >= window else scores
    plt.figure(figsize=(8, 4))
    plt.plot(moving_avg)
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Score moyen")
    plt.grid(True)
    plt.tight_layout()
    plt.show()











# # rlearn/algorithms/mc/mc_off_policy.py

# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from tqdm import tqdm


# def mc_off_policy_control(env, nb_episodes=5000, gamma=0.99, epsilon=0.1, weighted=True):
#     """
#     Contrôle Monte Carlo Hors-Politique (Off-policy) avec importance sampling.
#     Apprend Q et en déduit une politique optimale.
#     :param env: Environnement compatible
#     :param nb_episodes: nombre d'épisodes à simuler
#     :param gamma: facteur de réduction (discount)
#     :param epsilon: exploration de la behavior policy (ε-greedy)
#     :param weighted: True = Weighted IS, False = Ordinary IS
#     :return: policy, Q, scores
#     """
#     Q = defaultdict(lambda: {})  # Q[s][a]
#     C = defaultdict(lambda: {})  # cumul des poids W
#     scores = []

#     for _ in tqdm(range(nb_episodes), desc="MC Off-Policy"):
#         episode = []
#         env.reset()

#         while not env.is_game_over():
#             s = env.state_id()
#             actions = env.available_actions()

#             # ε-greedy behavior policy
#             if s in Q and len(Q[s]) > 0:
#                 if np.random.rand() < epsilon:
#                     a = np.random.choice(actions)
#                 else:
#                     q_vals = np.array([Q[s].get(a_, 0.0) for a_ in actions])
#                     a = actions[np.argmax(q_vals)]
#             else:
#                 a = np.random.choice(actions)

#             episode.append((s, a))
#             env.step(a)

#         scores.append(env.score())

#         # @@ Calcul du retour G avec gamma
#         G = 0.0
#         W = 1.0
#         for t in reversed(range(len(episode))):
#             s, a = episode[t]
#             reward = 0.0
#             if t == len(episode) - 1:
#                 reward = env.score()
#             G = gamma * G + reward

#             if a not in Q[s]:
#                 Q[s][a] = 0.0
#                 C[s][a] = 0.0

#             C[s][a] += W
#             if weighted:
#                 Q[s][a] += (W / C[s][a]) * (G - Q[s][a])
#             else:
#                 Q[s][a] += W * (G - Q[s][a])

#             # stop si action ≠ greedy
#             greedy_a = max(Q[s], key=Q[s].get)
#             if a != greedy_a:
#                 break

#             prob_action = (1 - epsilon) + epsilon / len(actions)
#             W *= 1.0 / prob_action

#     # Politique finale greedy
#     policy = np.zeros(env.num_states(), dtype=int)
#     for s in range(env.num_states()):
#         if s in Q and Q[s]:
#             policy[s] = max(Q[s], key=Q[s].get)
#         else:
#             policy[s] = 0

#     return policy, Q, scores


# def plot_scores(scores, window=100, title="MC Off-Policy - Score moyen"):
#     if len(scores) >= window:
#         moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
#     else:
#         moving_avg = scores
#     plt.figure(figsize=(8, 4))
#     plt.plot(moving_avg)
#     plt.title(title)
#     plt.xlabel("Épisode")
#     plt.ylabel("Score moyen")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
