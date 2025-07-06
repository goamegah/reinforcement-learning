# reinfolearn/utils/plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# import plotly.graph_objects as go


def plot_scores(scores, window=10, title="Performance", save=False, save_path=None):
    scores = np.array(scores)
    if len(scores) == 0:
        print("[WARN] Aucun score à afficher.")
        return

    if len(scores) >= window and window > 1:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
    else:
        moving_avg = scores

    plt.figure(figsize=(8, 4))
    plt.plot(moving_avg)
    plt.title(title + f" (window={window})")
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


def plot_scores_overlay(scores, window=10, title="Performance"):
    plt.figure(figsize=(10, 5))
    episodes = np.arange(len(scores))
    
    plt.plot(episodes, scores, alpha=0.3, label="Score brut")

    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
        plt.plot(episodes[:len(moving_avg)], moving_avg, color="orange", label=f"Moyenne mobile (window={window})")
    
    plt.title(title + " (Overlay)")
    plt.xlabel("Épisode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_scores_boxplot(scores, step=100, title="Distribution des scores"):
    scores = np.array(scores)
    num_steps = len(scores) // step
    data = [scores[i*step:(i+1)*step] for i in range(num_steps)]
    
    plt.figure(figsize=(12, 5))
    plt.boxplot(data, positions=np.arange(num_steps)*step)
    plt.title(title + f" (boxplot every {step} episodes)")
    plt.xlabel("Épisode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_scores_with_std(scores, window=100, title="Performance (mean ± std)"):
    df = pd.DataFrame(scores, columns=["score"])
    rolling = df["score"].rolling(window)
    mean = rolling.mean()
    std = rolling.std()

    plt.figure(figsize=(10, 5))
    plt.plot(mean, label="Moyenne mobile")
    plt.fill_between(mean.index, mean - std, mean + std, color="orange", alpha=0.3, label="±1 std")
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# def plot_scores_interactive(scores, window=100, title="Performance (Interactive)"):
#     episodes = np.arange(len(scores))
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=episodes, y=scores, mode='lines', name='Score brut', opacity=0.3))
    
#     if len(scores) >= window:
#         moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
#         fig.add_trace(go.Scatter(x=episodes[:len(moving_avg)], y=moving_avg, mode='lines', name=f"Moyenne mobile ({window})"))
    
#     fig.update_layout(title=title, xaxis_title="Épisode", yaxis_title="Score", template="plotly_white")
#     fig.show()

