import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


def sarsa(env, nb_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    Algorithme SARSA (on-policy TD control).
    :param env: environnement compatible
    :param nb_episodes: nombre d’épisodes
    :param gamma: facteur de discount
    :param alpha: taux d’apprentissage
    :param epsilon: exploration ε-greedy
    :return: policy, Q, scores
    """
    Q = defaultdict(lambda: np.zeros(env.num_actions()))
    scores = []

    for _ in tqdm(range(nb_episodes), desc="SARSA"):
        env.reset()
        s = env.state_id()
        actions = env.available_actions()

        # ε-greedy pour choisir a
        if np.random.rand() < epsilon:
            a = np.random.choice(actions)
        else:
            q_vals = Q[s][actions]
            a = actions[np.argmax(q_vals)]

        while not env.is_game_over():
            env.step(a)
            s_next = env.state_id()
            actions_next = env.available_actions()

            # ε-greedy pour choisir a'
            if np.random.rand() < epsilon:
                a_next = np.random.choice(actions_next)
            else:
                q_vals_next = Q[s_next][actions_next]
                a_next = actions_next[np.argmax(q_vals_next)]

            reward = 0
            for r_idx in range(env.num_rewards()):
                reward += env.p(s, a, s_next, r_idx) * env.reward(r_idx)

            # Mise à jour SARSA
            Q[s][a] += alpha * (reward + gamma * Q[s_next][a_next] - Q[s][a])

            s, a = s_next, a_next

        scores.append(env.score())

    # Politique finale (greedy)
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in Q:
            policy[s] = np.argmax(Q[s])
        else:
            policy[s] = 0

    return policy, Q, scores


def plot_scores(scores, window=100, title="SARSA - Score moyen"):
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
