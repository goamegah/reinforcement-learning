# reinfolearn/utils/plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# import plotly.graph_objects as go


def plot_convergence(values, title="Convergence", ylabel="Value", save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Itération")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Figure sauvegardée : {save_path}")
    plt.show()

def plot_scores(scores, title="Score par épisode", save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(scores)
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Score total")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f" Figure sauvegardée : {save_path}")
    plt.show()



def plot_episode_scores(scores, window=100, title="Scores par épisode"):
    """
    Affiche la moyenne glissante des scores par épisode.
    """
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
    else:
        moving_avg = scores

    plt.figure(figsize=(10, 4))
    plt.plot(moving_avg, label="Score moyen")
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_delta_history(delta_history, title="Convergence (delta max)"):
    """
    Affiche la variation max entre deux itérations (pour VI ou PE).
    """
    plt.figure(figsize=(10, 4))
    plt.plot(delta_history, label="Delta max")
    plt.title(title)
    plt.xlabel("Itération")
    plt.ylabel("Variation max (delta)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_vi_convergence(delta_history, mean_value_history):
    """
    Courbe double : delta max et moyenne des valeurs de V(s).
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(delta_history, label="Max Δ V", color="blue")
    ax1.set_xlabel("Itération")
    ax1.set_ylabel("Δ V", color="blue")
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(mean_value_history, label="Moyenne V", color="green", linestyle="--")
    ax2.set_ylabel("Valeur moyenne V", color="green")
    ax2.tick_params(axis='y', labelcolor='green')

    fig.suptitle("Convergence de Value Iteration")
    fig.tight_layout()
    plt.grid(True)
    plt.show()