import numpy as np
import matplotlib.pyplot as plt


def evaluate_policy(env, policy, n_episodes=100, max_steps=100):
    """
    Évalue une politique sur un ensemble d'épisodes.
    :param env: Environnement MDP compatible.
    :param policy: Liste ou array [s] → a.
    :param n_episodes: Nombre d'épisodes à simuler.
    :param max_steps: Nombre max de pas par épisode.
    :return: Liste des scores obtenus.
    """
    scores = []

    for _ in range(n_episodes):
        env.reset()
        total_reward = 0.0
        steps = 0

        while not env.is_game_over() and steps < max_steps:
            s = env.state_id()
            a = policy[s]
            env.step(a)
            steps += 1
        total_reward = env.score()
        scores.append(total_reward)

    return scores


def plot_score_distribution(scores, title="Distribution des scores"):
    """
    Affiche un histogramme des scores obtenus.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=10, edgecolor='black')
    plt.xlabel("Score obtenu")
    plt.ylabel("Nombre d'épisodes")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_score_summary(scores):
    """
    Affiche les statistiques de base des scores.
    """
    print(f"* Moyenne : {np.mean(scores):.2f}")
    print(f"* Écart-type : {np.std(scores):.2f}")
    print(f"* Max : {np.max(scores)}")
    print(f"* Min : {np.min(scores)}")
