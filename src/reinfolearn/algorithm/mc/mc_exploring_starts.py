# src/reinfolearn/algorithm/mc/mc_exploring_starts.py

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def mc_exploring_starts(env, gamma=0.99, nb_episodes=5000):
    """
    Monte Carlo Exploring Starts (MC-ES)
    Estimation de la politique optimale π* par retour complet en partant de (s0, a0) aléatoires.

    Conforme à Sutton & Barto - Fig 5.3.
    Fonctionne avec des environnements respectant l’interface SecretEnv.
    """
    q_table = defaultdict(lambda: np.zeros(env.num_actions()))
    returns = defaultdict(list)
    episode_scores = []

    for _ in tqdm(range(nb_episodes), desc="MC Exploring Starts"):
        env.reset()
        state = env.state_id()
        actions = env.available_actions()

        # Démarrage aléatoire (s0, a0)
        action = np.random.choice(actions)
        env.step(action)
        score_before = 0
        episode = [(state, action, 0.0)]  # Première transition, reward à compléter plus tard

        # Suite de l’épisode
        while not env.is_game_over():
            state = env.state_id()
            actions = env.available_actions()

            if state in q_table:
                q_vals = q_table[state][actions]
                best_q = np.max(q_vals)
                best_actions = [a for a, q in zip(actions, q_vals) if q == best_q]
                action = np.random.choice(best_actions)
            else:
                action = np.random.choice(actions)

            env.step(action)
            score_after = env.score()
            reward = score_after - score_before
            score_before = score_after

            episode.append((state, action, reward))

        episode_scores.append(env.score())

        # === Backward Update First-Visit ===
        G = 0.0
        visited = set()
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if (state, action) not in visited:
                returns[(state, action)].append(G)
                q_table[state][action] = np.mean(returns[(state, action)])
                visited.add((state, action))

    # === Politique finale extraite de Q (greedy) ===
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in q_table:
            policy[s] = int(np.argmax(q_table[s]))
        else:
            policy[s] = 0

    return policy, q_table, episode_scores


def plot_scores(scores, window=100, title="MC Exploring Starts - Score moyen par épisode"):
    """
    Affiche la moyenne glissante des scores.
    """
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
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


# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# def mc_exploring_starts(env, gamma=0.99, nb_episodes=5000):
#     """
#     Monte Carlo Exploring Starts (MC-ES)
#     Estimation de la politique optimale π* par retour complet en partant de (s0, a0) aléatoires.
    
#     Basé sur Sutton & Barto - Fig 5.3 - Exploring Starts MC Control.
#     Fonctionne avec tous les environnements respectant l’interface SecretEnv.
#     """
#     # Q(s,a) : table d’action-value
#     q_table = defaultdict(lambda: np.zeros(env.num_actions()))
#     returns = defaultdict(list)  # Liste des retours G pour chaque (s,a)
#     episode_scores = []

#     for ep in tqdm(range(nb_episodes), desc="MC Exploring Starts"):
#         env.reset()
#         state = env.state_id()
#         valid_actions = env.available_actions()

#         # === Start (s0, a0) aléatoire ===
#         action = np.random.choice(valid_actions)
#         episode = []

#         env.step(action)

#         # Génération de l’épisode complet
#         while not env.is_game_over():
#             next_state = env.state_id()
#             reward = sum(
#                 env.p(state, action, next_state, r_idx) * env.reward(r_idx)
#                 for r_idx in range(env.num_rewards())
#             )
#             episode.append((state, action, reward))

#             state = next_state
#             valid_actions = env.available_actions()

#             # Politique greedy sur Q
#             if state in q_table:
#                 q_values = q_table[state][valid_actions]
#                 best_q = np.max(q_values)
#                 best_actions = [a for a, q in zip(valid_actions, q_values) if q == best_q]
#                 action = np.random.choice(best_actions)
#             else:
#                 action = np.random.choice(valid_actions)

#             env.step(action)

#         # Ajouter la dernière transition
#         episode.append((state, action, env.score()))
#         episode_scores.append(env.score())

#         # === Backward update ===
#         G = 0.0
#         visited = set()

#         for state, action, reward in reversed(episode):
#             G = gamma * G + reward
#             if (state, action) not in visited:
#                 returns[(state, action)].append(G)
#                 q_table[state][action] = np.mean(returns[(state, action)])
#                 visited.add((state, action))

#     # === Politique finale extraite de Q ===
#     learned_policy = np.zeros(env.num_states(), dtype=int)
#     for state in range(env.num_states()):
#         if state in q_table:
#             learned_policy[state] = int(np.argmax(q_table[state]))
#         else:
#             learned_policy[state] = 0

#     return learned_policy, q_table, episode_scores


# def plot_scores(scores, window=100, title="MC Exploring Starts - Moyenne glissante des scores"):
#     """
#     Affiche la moyenne glissante des scores au fil des épisodes.
#     """
#     if len(scores) >= window:
#         moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
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
