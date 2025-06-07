import matplotlib.pyplot as plt

def plot_score_distribution(scores, title="Distribution des scores"):
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=10, edgecolor='black')
    plt.xlabel("Score obtenu")
    plt.ylabel("Nombre d'épisodes")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
