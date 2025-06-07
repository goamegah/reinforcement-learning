import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


def dyna_q_plus(env, nb_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, planning_steps=10, kappa=1e-4):
    """
    Implémentation de Dyna-Q+ avec bonus d'exploration.
    :param env: environnement compatible
    :param nb_episodes: nombre d'épisodes
    :param alpha: taux d'apprentissage
    :param gamma: facteur de réduction
    :param epsilon: taux d'exploration ε-greedy
    :param planning_steps: nombre de simulations par étape
    :param kappa: bonus d'exploration (faible)
    :return: policy, Q-table, scores
    """
    Q = defaultdict(lambda: np.zeros(env.num_actions()))
    model = dict()
    last_visit = dict()
    scores = []

    time = 0

    for episode in tqdm(range(nb_episodes), desc="Dyna-Q+"):
        env.reset()
        state = env.state_id()
        done = False

        while not env.is_game_over():
            time += 1
            actions = env.available_actions()

            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                q_vals = Q[state]
                best_actions = np.flatnonzero(q_vals == q_vals.max())
                action = np.random.choice(best_actions)

            env.step(action)
            next_state = env.state_id()
            reward = env.score() if env.is_game_over() else 0.0

            # Apprentissage réel
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

            # Mise à jour du modèle
            model[(state, action)] = (next_state, reward)
            last_visit[(state, action)] = time

            # Simulation de planning
            for _ in range(planning_steps):
                if not model:
                    break
                s_a = random.choice(list(model.keys()))
                s, a = s_a
                s_p, r = model[s_a]

                tau = time - last_visit.get((s, a), 0)
                bonus = kappa * np.sqrt(tau)
                total_reward = r + bonus

                Q[s][a] += alpha * (total_reward + gamma * np.max(Q[s_p]) - Q[s][a])

            state = next_state

        scores.append(env.score())

    # Politique finale
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in Q:
            policy[s] = np.argmax(Q[s])
        else:
            policy[s] = 0

    return policy, Q, scores


def plot_scores(scores, window=100, title="Dyna-Q+ - Score moyen"):
    moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(moving_avg)
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Score moyen")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
