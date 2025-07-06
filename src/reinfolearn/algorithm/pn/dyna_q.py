# # # reinfolearn/algorithm/pn/dyna_q.py


import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random

def dyna_q(env, nb_episodes=3000, alpha=0.1, gamma=0.99, epsilon=0.1, planning_steps=50):
    Q = defaultdict(lambda: np.zeros(env.num_actions()))
    model = dict()  # model[(s,a)] = (s', r)
    scores = []
    seen = set()

    for _ in tqdm(range(nb_episodes), desc="Dyna-Q"):
        env.reset()
        state = env.state_id()

        while not env.is_game_over():
            actions = env.available_actions()

            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                q_vals = np.array([Q[state][a] for a in actions])
                best_actions = actions[np.flatnonzero(q_vals == q_vals.max())]
                action = np.random.choice(best_actions)

            # Step and observe next_state and reward
            env.step(action)
            next_state = env.state_id()

            # 🔥 Calcul de la reward immédiate
            reward = 0.0
            for r_idx in range(env.num_rewards()):
                reward += env.p(state, action, next_state, r_idx) * env.reward(r_idx)

            # Q-learning update
            next_actions = env.available_actions()
            if len(next_actions) > 0:
                max_q_next = max([Q[next_state][a] for a in next_actions])
            else:
                max_q_next = 0.0

            Q[state][action] += alpha * (reward + gamma * max_q_next - Q[state][action])

            # Update model
            model[(state, action)] = (next_state, reward)
            seen.add((state, action))

            state = next_state

            # Planning (simulate n steps)
            for _ in range(planning_steps):
                s_sim, a_sim = random.choice(list(seen))
                s_p_sim, r_sim = model[(s_sim, a_sim)]

                # Simulate next action-value
                max_q_s_p = np.max(Q[s_p_sim]) if s_p_sim in Q else 0.0
                Q[s_sim][a_sim] += alpha * (r_sim + gamma * max_q_s_p - Q[s_sim][a_sim])

        scores.append(env.score())

    # Policy extraction (greedy)
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in Q:
            q_vals = np.array([Q[s][a] for a in range(env.num_actions())])
            policy[s] = np.argmax(q_vals)
        else:
            policy[s] = 0

    return policy, Q, scores




# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import random

# def dyna_q(env, nb_episodes=3000, alpha=0.1, gamma=0.99, epsilon=0.1, planning_steps=5):
#     """
#     Implémentation de l'algorithme Dyna-Q.
#     :param env: environnement compatible
#     :param nb_episodes: nombre d'épisodes
#     :param alpha: taux d'apprentissage
#     :param gamma: facteur de réduction
#     :param epsilon: exploration ε-greedy
#     :param planning_steps: nombre de transitions simulées par épisode
#     :return: policy, Q, scores
#     """
#     Q = defaultdict(lambda: np.zeros(env.num_actions()))
#     model = dict()  # model[s,a] = (s', r)
#     scores = []
#     seen = set()

#     for _ in tqdm(range(nb_episodes), desc="Dyna-Q"):
#         env.reset()
#         while not env.is_game_over():
#             s = env.state_id()
#             actions = env.available_actions()

#             # ε-greedy filtré sur actions valides
#             if np.random.rand() < epsilon:
#                 a = np.random.choice(actions)
#             else:
#                 q_vals = np.array([Q[s][act] for act in actions])
#                 best_actions = actions[np.flatnonzero(q_vals == q_vals.max())]
#                 a = np.random.choice(best_actions)

#             env.step(a)
#             s_p = env.state_id()
#             r = env.score() if env.is_game_over() else 0.0

#             # Mise à jour Q-table
#             Q[s][a] += alpha * (r + gamma * np.max(Q[s_p]) - Q[s][a])

#             # Mise à jour modèle
#             model[(s, a)] = (s_p, r)
#             seen.add((s, a))

#             # Planification
#             for _ in range(planning_steps):
#                 s_sim, a_sim = random.choice(list(seen))
#                 s_p_sim, r_sim = model[(s_sim, a_sim)]
#                 Q[s_sim][a_sim] += alpha * (r_sim + gamma * np.max(Q[s_p_sim]) - Q[s_sim][a_sim])

#         scores.append(env.score())

#     # Politique finale (greedy)
#     policy = np.zeros(env.num_states(), dtype=int)
#     for s in range(env.num_states()):
#         q_vals = np.array([Q[s][a] for a in range(env.num_actions())])
#         policy[s] = np.argmax(q_vals)

#     return policy, Q, scores


# def plot_scores(scores, window=100, title="Dyna-Q - Score moyen"):
#     plt.figure(figsize=(8, 4))
    
#     if len(scores) >= window:
#         moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
#         plt.plot(moving_avg, label=f"Moving Avg ({window})")
    
#     plt.plot(scores, alpha=0.3, label="Raw Scores")
    
#     plt.title(title)
#     plt.xlabel("Épisode")
#     plt.ylabel("Score")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

