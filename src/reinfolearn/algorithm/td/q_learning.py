import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def q_learning(env, nb_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    Q-Learning (Off-policy TD Control)
    Apprentissage d’une fonction Q* optimale en mettant à jour les valeurs Q(s,a)
    selon le maximum des actions suivantes. Conforme à Sutton & Barto Fig 6.5.
    
    L'agent suit une politique ε-greedy pour explorer l’environnement.
    """
    q_table = defaultdict(lambda: np.zeros(env.num_actions()))
    episode_scores = []

    for ep in tqdm(range(nb_episodes), desc="Q-Learning"):
        env.reset()
        state = env.state_id()

        while not env.is_game_over():
            valid_actions = env.available_actions()

            # === ε-greedy ===
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                q_values = np.array([q_table[state][a] for a in valid_actions])
                best_q = np.max(q_values)
                best_actions = [a for a, q in zip(valid_actions, q_values) if q == best_q]
                action = np.random.choice(best_actions)

            env.step(action)
            next_state = env.state_id()

            # === Reward immédiate (définition MDP) ===
            reward = sum(
                env.p(state, action, next_state, r_idx) * env.reward(r_idx)
                for r_idx in range(env.num_rewards())
            )

            # === Mise à jour Q(s,a) ===
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            # Passer à l'état suivant
            state = next_state

        # 🎯 Score global obtenu sur l’épisode
        episode_scores.append(env.score())

    # === Politique finale extraite de Q ===
    learned_policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in q_table and len(q_table[s]) > 0:
            learned_policy[s] = int(np.argmax(q_table[s]))
        else:
            learned_policy[s] = 0

    return learned_policy, q_table, episode_scores


def plot_scores(scores, window=100, title="Q-Learning - Moyenne glissante des scores"):
    """
    Affiche la moyenne glissante des scores au fil des épisodes.
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
    plt.show()
