# rlearn/algorithms/mc/mc_on_policy.py

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


def mc_on_policy_first_visit(env, gamma=0.99, nb_episodes=5000, epsilon=0.1):
    """
    Monte Carlo On-Policy First-Visit avec epsilon-greedy et discount gamma
    :param env: environnement compatible
    :param gamma: facteur de discount
    :param nb_episodes: nombre d'épisodes à exécuter
    :param epsilon: exploration aléatoire (ε-greedy)
    :return: politique apprise, Q, historique des scores
    """
    Q = defaultdict(lambda: {})  # Q[s][a]
    returns = defaultdict(list)
    scores = []

    for ep in tqdm(range(nb_episodes), desc="MC On-Policy First Visit"):
        env.reset()
        episode = []  # (s, a, r)
        visited = set()

        while not env.is_game_over():
            s = env.state_id()
            actions = env.available_actions()

            # ε-greedy
            if s in Q and len(Q[s]) > 0:
                if np.random.rand() < epsilon:
                    a = np.random.choice(actions)
                else:
                    q_vals = np.array([Q[s].get(a_, 0.0) for a_ in actions])
                    a = actions[np.argmax(q_vals)]
            else:
                a = np.random.choice(actions)

            episode.append((s, a))
            env.step(a)

        # @@ Calcul du retour G avec discount
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a = episode[t]
            reward = 0.0
            if t == len(episode) - 1:
                reward = env.score()
            G = gamma * G + reward

            if (s, a) not in visited:
                if a not in Q[s]:
                    Q[s][a] = 0.0
                returns[(s, a)].append(G)
                Q[s][a] = np.mean(returns[(s, a)])
                visited.add((s, a))

        scores.append(env.score())

    # Politique finale : greedy
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in Q and Q[s]:
            policy[s] = max(Q[s], key=Q[s].get)
        else:
            policy[s] = 0

    return policy, Q, scores


def plot_scores(scores, window=100, title="MC On-Policy First-Visit - Score moyen"):
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
