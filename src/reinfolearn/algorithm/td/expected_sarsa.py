import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def expected_sarsa(env, nb_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    Expected SARSA algorithm (Sutton & Barto Fig. 6.8)
    """
    Q = defaultdict(lambda: np.zeros(env.num_actions()))
    scores = []

    for _ in tqdm(range(nb_episodes), desc="Expected SARSA"):
        env.reset()
        state = env.state_id()

        while not env.is_game_over():
            actions = env.available_actions()

            # ε-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                q_vals = np.array([Q[state][a] for a in actions])
                best_actions = actions[np.flatnonzero(q_vals == q_vals.max())]
                action = np.random.choice(best_actions)

            # Step and observe next_state
            env.step(action)
            next_state = env.state_id()
            next_actions = env.available_actions()

            # 🔥 Reward calculation (expected immediate reward)
            reward = 0.0
            for r_idx in range(env.num_rewards()):
                reward += env.p(state, action, next_state, r_idx) * env.reward(r_idx)

            # Expected value over next actions under ε-greedy
            q_vals_next = np.array([Q[next_state][a] for a in next_actions])
            probs = np.ones(len(next_actions)) * (epsilon / len(next_actions))
            best_next_idx = np.argmax(q_vals_next)
            probs[best_next_idx] += (1.0 - epsilon)
            expected_value = np.dot(probs, q_vals_next)

            # TD update
            td_target = reward + gamma * expected_value
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state

        scores.append(env.score())

    # Greedy policy extraction
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in Q:
            q_vals = np.array([Q[s][a] for a in range(env.num_actions())])
            policy[s] = np.argmax(q_vals)
        else:
            policy[s] = 0

    return policy, Q, scores

def plot_scores(scores, window=100, title="Expected SARSA - Score moyen"):
    moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(moving_avg)
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Score moyen")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# # reinfolearn/algorithm/td/expected_sarsa.py

# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from collections import defaultdict

# def expected_sarsa(env, nb_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
#     """
#     Expected SARSA algorithm for reinforcement learning.
#     This algorithm learns the action-value function Q(s, a) and derives a policy from it.
#     :param env: Environment compatible with the expected SARSA algorithm
#     :param nb_episodes: Number of episodes to simulate
#     :param gamma: Discount factor
#     :param alpha: Learning rate
#     :param epsilon: Exploration rate for ε-greedy policy
#     :return: policy, Q, scores
#     """
#     Q = defaultdict(lambda: np.zeros(env.num_actions()))
#     scores = []

#     for _ in tqdm(range(nb_episodes), desc="Expected SARSA"):
#         env.reset()
#         state = env.state_id()

#         while not env.is_game_over():
#             actions = env.available_actions()

#             # Sélection ε-greedy parmi actions valides
#             if np.random.rand() < epsilon:
#                 action = np.random.choice(actions)
#             else:
#                 q_vals = np.array([Q[state][a] for a in actions])
#                 best_actions = actions[np.flatnonzero(q_vals == q_vals.max())]
#                 action = np.random.choice(best_actions)

#             env.step(action)
#             next_state = env.state_id()
#             next_actions = env.available_actions()

#             # Calcul de l'expected value parmi actions valides uniquement
#             probs = np.ones(len(next_actions)) * (epsilon / len(next_actions))
#             q_vals_next = np.array([Q[next_state][a] for a in next_actions])
#             best_action_idx = np.argmax(q_vals_next)
#             probs[best_action_idx] += (1.0 - epsilon)

#             expected_value = np.dot(probs, q_vals_next)

#             reward = env.score() if env.is_game_over() else 0
#             td_target = reward + gamma * expected_value
#             Q[state][action] += alpha * (td_target - Q[state][action])

#             state = next_state

#         scores.append(env.score())

#     policy = np.zeros(env.num_states(), dtype=int)
#     for s in range(env.num_states()):
#         if s in Q:
#             q_vals = np.array([Q[s][a] for a in range(env.num_actions())])
#             policy[s] = np.argmax(q_vals)
#         else:
#             policy[s] = 0

#     return policy, Q, scores


# def plot_scores(scores, window=100, title="Expected SARSA - Score moyen"):
#     moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
#     plt.figure(figsize=(8, 4))
#     plt.plot(moving_avg)
#     plt.title(title)
#     plt.xlabel("Épisode")
#     plt.ylabel("Score moyen")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
