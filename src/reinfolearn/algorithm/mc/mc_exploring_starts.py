# -*- coding: utf-8 -*-
# rlearn/algorithms/mc/mc_exploring_starts.py

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def mc_exploring_starts(env, gamma=0.99, nb_episodes=5000):
    """
    Monte Carlo Exploring Starts (MC ES)
    Estimation de π ≈ π* en explorant aléatoirement les (s,a) initiaux.
    
    Conforme au pseudo-code Sutton & Barto (Fig. 5.3 Exploring Starts MC Control).
    """
    # === Initialisation ===
    Q = defaultdict(lambda: np.zeros(env.num_actions()))
    returns = defaultdict(list)
    scores = []

    for ep in tqdm(range(nb_episodes), desc="MC Exploring Starts"):
        env.reset()

        # === Exploring Start: choisir un (s0,a0) aléatoire ===
        state = env.state_id()
        valid_actions = env.available_actions()
        action = np.random.choice(valid_actions)
        
        # Générer l'épisode complet (s,a,r) en partant de (s0,a0)
        episode = []
        env.step(action)

        while not env.is_game_over():
            next_state = env.state_id()

            # Calcul de la reward immédiate selon la définition MDP
            reward = 0.0
            for r_idx in range(env.num_rewards()):
                reward += env.p(state, action, next_state, r_idx) * env.reward(r_idx)

            episode.append((state, action, reward))

            state = next_state
            valid_actions = env.available_actions()

            # Politique greedy sur Q
            q_vals = Q[state][valid_actions] if state in Q else np.zeros(len(valid_actions))
            max_q = np.max(q_vals)
            best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
            action = np.random.choice(best_actions)

            env.step(action)

        # Ajouter la dernière transition terminale avec score
        final_reward = env.score()
        episode.append((state, action, final_reward))

        # === Calcul backward des retours G ===
        G = 0.0
        visited = set()

        for (s, a, r) in reversed(episode):
            G = gamma * G + r

            if (s, a) not in visited:
                returns[(s, a)].append(G)
                Q[s][a] = np.mean(returns[(s, a)])
                visited.add((s, a))

        scores.append(env.score())

    # === Politique finale extraite de Q ===
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in Q:
            policy[s] = np.argmax(Q[s])
        else:
            policy[s] = 0  # action par défaut

    return policy, Q, scores

def plot_scores(scores, window=100, title="MC Exploring Starts - Score moyen"):
    """
    Affiche la moyenne glissante des scores.
    """
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
    else:
        moving_avg = scores

    plt.figure(figsize=(8, 4))
    plt.plot(moving_avg)
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Score moyen")
    plt.grid(True)
    plt.tight_layout()
    plt.show()






# # -*- coding: utf-8 -*-
# # rlearn/algorithms/mc/mc_exploring_starts.py

# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm
# import matplotlib.pyplot as plt


# def mc_exploring_starts(env, gamma=0.99, nb_episodes=5000, epsilon=0.1):
#     """
#     Monte Carlo Exploring Starts : apprentissage d'une politique optimale par estimation de Q(s,a)
#     :param env: environnement compatible
#     :param gamma: facteur d'actualisation
#     :param nb_episodes: nombre d'épisodes
#     :param epsilon: exploration aléatoire
#     :return: politique, Q, historique des scores
#     """
#     Q = defaultdict(dict)
#     returns = defaultdict(list)
#     scores = []

#     for ep in tqdm(range(nb_episodes), desc="MC Exploring Starts"):
#         env.reset()
#         state = env.state_id()

#         # @@ Initialisation (Exploring Starts) : choisir (s0, a0) aléatoirement
#         valid_actions = env.available_actions()
#         action = np.random.choice(valid_actions)
#         episode = [(state, action)]

#         env.step(action)

#         while not env.is_game_over():
#             state = env.state_id()
#             valid_actions = env.available_actions()

#             # Politique ε-greedy sur Q
#             if np.random.rand() < epsilon:
#                 action = np.random.choice(valid_actions)
#             else:
#                 q_vals = [Q[state].get(a, 0.0) for a in valid_actions]
#                 max_q = np.max(q_vals)
#                 best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
#                 action = np.random.choice(best_actions)

#             episode.append((state, action))
#             env.step(action)

#         # @@ Calcul du retour G avec gamma
#         G = 0.0
#         visited = set()
#         for t, (s, a) in enumerate(reversed(episode)):
#             G = gamma * G + env.score()
#             if (s, a) not in visited:
#                 returns[(s, a)].append(G)
#                 Q[s][a] = np.mean(returns[(s, a)])
#                 visited.add((s, a))

#         scores.append(env.score())

#     # Politique optimale extraite de Q
#     policy = np.zeros(env.num_states(), dtype=int)
#     for s in range(env.num_states()):
#         if s in Q and Q[s]:
#             policy[s] = max(Q[s], key=Q[s].get)
#         else:
#             policy[s] = 0

#     return policy, Q, scores


# def plot_scores(scores, window=100, title="MC Exploring Starts - Scores"):
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





# # import numpy as np
# # from collections import defaultdict
# # from tqdm import tqdm
# # import matplotlib.pyplot as plt


# # def mc_exploring_starts(env, gamma=0.99, nb_episodes=5000, epsilon=0.1):
# #     """
# #     Monte Carlo Exploring Starts : apprentissage d'une politique optimale par estimation de Q(s,a)
# #     :param env: environnement compatible
# #     :param gamma: facteur d’actualisation
# #     :param nb_episodes: nombre d’épisodes
# #     :param epsilon: exploration aléatoire
# #     :return: politique, Q, historique des scores
# #     """
# #     Q = defaultdict(lambda: np.zeros(env.num_actions()))
# #     returns = defaultdict(list)
# #     scores = []

# #     for ep in tqdm(range(nb_episodes), desc="MC Exploring Starts"):
# #         env.reset()
# #         state = env.state_id()

# #         # @@ Initialisation (Exploring Starts) : choisir (s0, a0) aléatoirement
# #         valid_actions = env.available_actions()
# #         action = np.random.choice(valid_actions)
# #         episode = [(state, action)]

# #         env.step(action)

# #         while not env.is_game_over():
# #             state = env.state_id()
# #             valid_actions = env.available_actions()

# #             # Politique ε-greedy sur Q
# #             if np.random.rand() < epsilon:
# #                 action = np.random.choice(valid_actions)
# #             else:
# #                 q_vals = Q[state]
# #                 action = np.random.choice(np.flatnonzero(q_vals == q_vals.max()))

# #             episode.append((state, action))
# #             env.step(action)

# #         # @@ Calcul du retour G avec gamma
# #         G = 0.0
# #         visited = set()
# #         for t, (s, a) in enumerate(reversed(episode)):
# #             G = gamma * G + env.score()  # env.score() est la récompense terminale pour l’instant
# #             if (s, a) not in visited:
# #                 returns[(s, a)].append(G)
# #                 Q[s][a] = np.mean(returns[(s, a)])
# #                 visited.add((s, a))

# #         scores.append(env.score())

# #     # Politique optimale extraite de Q
# #     policy = np.zeros(env.num_states(), dtype=int)
# #     for s in range(env.num_states()):
# #         policy[s] = np.argmax(Q[s]) if s in Q else 0

# #     return policy, Q, scores


# # def plot_scores(scores, window=100, title="MC Exploring Starts - Scores"):
# #     moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
# #     plt.figure(figsize=(8, 4))
# #     plt.plot(moving_avg)
# #     plt.title(title)
# #     plt.xlabel("Épisode")
# #     plt.ylabel("Score moyen")
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.show()
