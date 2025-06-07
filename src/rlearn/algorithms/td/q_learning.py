import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def q_learning(env, nb_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    Q-Learning (off-policy TD control).
    :param env: Environnement compatible
    :param nb_episodes: Nombre d'épisodes à exécuter
    :param gamma: Facteur de discount
    :param alpha: Taux d'apprentissage
    :param epsilon: Taux d'exploration (ε-greedy)
    :return: policy, Q, scores
    """
    # Initialisation de la table Q
    Q = defaultdict(lambda: np.zeros(env.num_actions()))
    scores = []

    for _ in tqdm(range(nb_episodes), desc="Q-Learning"):
        env.reset()
        state = env.state_id()

        while not env.is_game_over():
            actions = env.available_actions()
            action = np.random.choice(actions) if np.random.rand() < epsilon else np.argmax(Q[state])
            env.step(action)
            next_state = env.state_id()

            reward = env.score() if env.is_game_over() else 0
            td_target = reward + gamma * np.max(Q[next_state])
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state

        scores.append(env.score())

    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        policy[s] = np.argmax(Q[s]) if s in Q else 0

    return policy, Q, scores


def plot_scores(scores, window=100, title="Q-Learning - Score moyen"):
    moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(moving_avg)
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Score moyen")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
