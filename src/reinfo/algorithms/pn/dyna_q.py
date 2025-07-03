import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


def dyna_q(env, nb_episodes=3000, alpha=0.1, gamma=0.99, epsilon=0.1, planning_steps=50):
    """
    Implémentation de l'algorithme Dyna-Q.
    :param env: environnement compatible
    :param nb_episodes: nombre d'épisodes
    :param alpha: taux d'apprentissage
    :param gamma: facteur de réduction
    :param epsilon: exploration ε-greedy
    :param planning_steps: nombre de transitions simulées par épisode
    :return: policy, Q, scores
    """
    Q = defaultdict(lambda: np.zeros(env.num_actions()))
    model = dict()  # model[s][a] = (s', r)
    scores = []
    seen = set()

    for _ in tqdm(range(nb_episodes), desc="Dyna-Q"):
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            actions = env.available_actions()

            # ε-greedy
            if np.random.rand() < epsilon:
                a = np.random.choice(actions)
            else:
                q_vals = Q[s]
                a = np.random.choice(np.flatnonzero(q_vals == q_vals.max()))

            env.step(a)
            s_p = env.state_id()
            r = env.score() if env.is_game_over() else 0.0

            # Mise à jour Q-table
            Q[s][a] += alpha * (r + gamma * np.max(Q[s_p]) - Q[s][a])

            # Mise à jour du modèle
            model[(s, a)] = (s_p, r)
            seen.add((s, a))

            # Planification
            for _ in range(planning_steps):
                s_sim, a_sim = random_choice(seen)
                s_p_sim, r_sim = model[(s_sim, a_sim)]
                Q[s_sim][a_sim] += alpha * (r_sim + gamma * np.max(Q[s_p_sim]) - Q[s_sim][a_sim])

        scores.append(env.score())

    # Politique finale
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in Q:
            policy[s] = np.argmax(Q[s])
        else:
            policy[s] = 0

    return policy, Q, scores


def random_choice(pairs):
    pairs = list(pairs)
    return pairs[np.random.randint(len(pairs))]


def plot_scores(scores, window=100, title="Dyna-Q - Score moyen"):
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
