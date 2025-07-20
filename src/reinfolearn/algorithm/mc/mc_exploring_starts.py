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
    q_table = defaultdict(dict)  # q_table[state][action] = value
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

            if state in q_table and len(q_table[state]) > 0:
                q_vals = np.array([q_table[state].get(a, 0.0) for a in actions])
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
        if s in q_table and len(q_table[s]) > 0:
            best_action = max(q_table[s], key=q_table[s].get)
            policy[s] = best_action
        else:
            policy[s] = 0  # Action par défaut

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
