import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def expected_sarsa(env, nb_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    Expected SARSA - version générique et compatible avec tout environnement.
    
    Mise à jour Q(s,a) en utilisant la valeur espérée des Q(s', a'),
    où les a' sont pondérées selon la politique ε-greedy.
    
    ✅ Compatible avec les environnements ne définissant pas p() ni reward(),
    grâce à l’utilisation de env.score() pour estimer les rewards.
    """
    q_table = defaultdict(lambda: np.zeros(env.num_actions()))
    episode_scores = []

    for ep in tqdm(range(nb_episodes), desc="Expected SARSA"):
        env.reset()
        state = env.state_id()
        score_before = env.score()

        while not env.is_game_over():
            valid_actions = env.available_actions()

            # === Choix ε-greedy ===
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                q_vals = np.array([q_table[state][a] for a in valid_actions])
                best_actions = [a for a, q in zip(valid_actions, q_vals) if q == q_vals.max()]
                action = np.random.choice(best_actions)

            env.step(action)
            next_state = env.state_id()
            score_after = env.score()

            # ✅ Reward implicite
            reward = score_after - score_before
            score_before = score_after

            next_valid_actions = env.available_actions()
            q_vals_next = np.array([q_table[next_state][a] for a in next_valid_actions])

            # === Politique ε-greedy sur s’ ===
            action_probs = np.ones(len(next_valid_actions)) * (epsilon / len(next_valid_actions))
            best_action_idx = np.argmax(q_vals_next)
            action_probs[best_action_idx] += (1.0 - epsilon)

            expected_q_value = np.dot(action_probs, q_vals_next)

            # === Mise à jour Expected SARSA ===
            td_target = reward + gamma * expected_q_value
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state

        episode_scores.append(env.score())

    # === Politique finale (greedy) ===
    learned_policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in q_table:
            q_vals = np.array([q_table[s][a] for a in range(env.num_actions())])
            learned_policy[s] = int(np.argmax(q_vals))
        else:
            learned_policy[s] = 0

    return learned_policy, q_table, episode_scores


def plot_scores(scores, window=100, title="Expected SARSA - Moyenne glissante des scores"):
    """
    Affiche la moyenne glissante des scores au fil des épisodes.
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


# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from collections import defaultdict

# def expected_sarsa(env, nb_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
#     """
#     Expected SARSA (Contrôle TD on-policy avec valeurs espérées).
    
#     Conforme au livre Sutton & Barto (Fig. 6.8) :
#     - Mise à jour de Q(s,a) en utilisant la valeur espérée des actions suivantes
#       selon la politique ε-greedy actuelle.
#     - Reward immédiate selon la définition MDP.
#     """
#     q_table = defaultdict(lambda: np.zeros(env.num_actions()))
#     episode_scores = []

#     for ep in tqdm(range(nb_episodes), desc="Expected SARSA"):
#         env.reset()
#         state = env.state_id()

#         while not env.is_game_over():
#             valid_actions = env.available_actions()

#             # === Choix ε-greedy ===
#             if np.random.rand() < epsilon:
#                 action = np.random.choice(valid_actions)
#             else:
#                 q_vals = np.array([q_table[state][a] for a in valid_actions])
#                 best_actions = [a for a, q in zip(valid_actions, q_vals) if q == q_vals.max()]
#                 action = np.random.choice(best_actions)

#             # Transition vers l’état suivant
#             env.step(action)
#             next_state = env.state_id()
#             next_valid_actions = env.available_actions()

#             # === Reward immédiate (définition MDP) ===
#             reward = sum(
#                 env.p(state, action, next_state, r_idx) * env.reward(r_idx)
#                 for r_idx in range(env.num_rewards())
#             )

#             # === Valeur espérée des Q(next_state, a') ===
#             q_vals_next = np.array([q_table[next_state][a] for a in next_valid_actions])
#             action_probs = np.ones(len(next_valid_actions)) * (epsilon / len(next_valid_actions))
#             best_action_idx = np.argmax(q_vals_next)
#             action_probs[best_action_idx] += (1.0 - epsilon)
#             expected_q_value = np.dot(action_probs, q_vals_next)

#             # === Mise à jour Expected SARSA ===
#             td_target = reward + gamma * expected_q_value
#             q_table[state][action] += alpha * (td_target - q_table[state][action])

#             # Avancer dans l’épisode
#             state = next_state

#         episode_scores.append(env.score())

#     # === Politique finale (greedy sur Q) ===
#     learned_policy = np.zeros(env.num_states(), dtype=int)
#     for s in range(env.num_states()):
#         if s in q_table:
#             q_vals = np.array([q_table[s][a] for a in range(env.num_actions())])
#             learned_policy[s] = int(np.argmax(q_vals))
#         else:
#             learned_policy[s] = 0

#     return learned_policy, q_table, episode_scores


# def plot_scores(scores, window=100, title="Expected SARSA - Moyenne glissante des scores"):
#     """
#     Affiche la moyenne glissante des scores au fil des épisodes.
#     """
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
