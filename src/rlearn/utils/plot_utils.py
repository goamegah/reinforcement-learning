import matplotlib.pyplot as plt
import numpy as np
import os


def plot_scores(scores, window=100, title="Performance", save=False, save_path=None):
    """
    Affiche ou enregistre un graphique des scores moyens.
    :param scores: liste des scores bruts
    :param window: taille de la fenêtre pour la moyenne mobile
    :param title: titre du graphique
    :param save: True = enregistrer au lieu d'afficher
    :param save_path: chemin du fichier PNG si save=True
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

    if save:
        if save_path is None:
            save_path = f"{title.lower().replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"[OK] Graphique enregistré : {save_path}")
    else:
        plt.show()
