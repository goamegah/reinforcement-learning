# reinfolearn/utils/plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# import plotly.graph_objects as go


def plot_unified(data, title="", xlabel="Épisodes", ylabel="Score", window=None, 
                 figsize=(10, 5), grid=True, save_path=None, legend_labels=None):
    """
    Unified plotting function that can handle both single data series and multiple algorithm comparisons.
    
    Parameters:
    -----------
    data : list or dict
        If list: a single data series to plot
        If dict: multiple data series with keys as algorithm names and values as data series
    title : str
        Title of the plot
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    window : int or None
        Window size for moving average. If None, no moving average is applied.
    figsize : tuple
        Figure size (width, height)
    grid : bool
        Whether to show grid
    save_path : str or None
        Path to save the figure. If None, figure is not saved.
    legend_labels : list or None
        Custom labels for legend. If None, keys from data dict are used.
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    
    # Handle single data series
    if isinstance(data, list) or isinstance(data, np.ndarray):
        if window is not None:
            # Apply moving average
            data_plot = np.convolve(data, np.ones(window)/window, mode="valid")
        else:
            data_plot = data
        plt.plot(data_plot)
    
    # Handle multiple data series (algorithm comparison)
    elif isinstance(data, dict):
        for i, (name, values) in enumerate(data.items()):
            if window is not None:
                # Apply moving average
                values_plot = np.convolve(values, np.ones(window)/window, mode="valid")
            else:
                values_plot = values
            
            label = legend_labels[i] if legend_labels is not None else name
            plt.plot(values_plot, label=label)
        
        plt.legend()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if grid:
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure sauvegardée : {save_path}")
    
    plt.show()


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
    
    
def plot_scores_mavg(scores, window=100, title="Score moyen"):
    moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(moving_avg)
    plt.title(title)
    plt.xlabel("Épisodes")
    plt.ylabel("Score moyen")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    