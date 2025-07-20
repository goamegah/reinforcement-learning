import numpy as np
from tqdm import tqdm


def evaluate_policy_score(env, policy, nb_episodes=500, seed=None):
    """
    Évalue une politique donnée sur un environnement en moyenne sur plusieurs épisodes.

    :param env: Environnement compatible
    :param policy: Tableau numpy de la politique (int pour chaque état)
    :param nb_episodes: Nombre d’épisodes à jouer
    :param seed: Graine aléatoire pour reproductibilité
    :return: liste des scores obtenus
    """
    scores = []
    rng = np.random.default_rng(seed)

    for _ in tqdm(range(nb_episodes), desc="Évaluation de la politique"):
        env.reset()
        while not env.is_game_over():
            state = env.state_id()
            action = policy[state]
            env.step(action)
        scores.append(env.score())

    return scores


def summarize_policy_scores(scores):
    """
    Donne un résumé statistique des scores de la politique.

    :param scores: liste des scores obtenus (float ou int)
    :return: dict avec moyenne, min, max, std
    """
    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
    }


def compute_success_rate(scores, threshold=1.0):
    """
    Calcule le taux d’épisodes avec un score supérieur ou égal à un seuil (ex: score > 0).

    :param scores: liste des scores
    :param threshold: seuil considéré comme succès
    :return: taux de succès (float entre 0 et 1)
    """
    return np.mean(np.array(scores) >= threshold)
