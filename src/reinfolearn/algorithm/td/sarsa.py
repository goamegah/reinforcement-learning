import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def sarsa(env, nb_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    SARSA (State-Action-Reward-State-Action) - On-policy TD Control.
    
    L’agent met à jour Q(s,a) en suivant la politique ε-greedy qu’il utilise pour agir.
    Conforme à Sutton & Barto Fig. 6.5, avec définition MDP par reward immédiate.
    """
    q_table = defaultdict(lambda: np.zeros(env.num_actions()))
    episode_scores = []

    for ep in tqdm(range(nb_episodes), desc="SARSA"):
        env.reset()
        state = env.state_id()
        valid_actions = env.available_actions()

        # === Choix de l’action a via ε-greedy ===
        if np.random.rand() < epsilon:
            action = np.random.choice(valid_actions)
        else:
            q_vals = np.array([q_table[state][a] for a in valid_actions])
            best_actions = [a for a, q in zip(valid_actions, q_vals) if q == q_vals.max()]
            action = np.random.choice(best_actions)

        while not env.is_game_over():
            env.step(action)
            next_state = env.state_id()
            next_valid_actions = env.available_actions()

            # === Reward immédiate (définition MDP) ===
            reward = sum(
                env.p(state, action, next_state, r_idx) * env.reward(r_idx)
                for r_idx in range(env.num_rewards())
            )

            # === Choix de l’action suivante a’ via ε-greedy ===
            if np.random.rand() < epsilon:
                next_action = np.random.choice(next_valid_actions)
            else:
                q_vals_next = np.array([q_table[next_state][a] for a in next_valid_actions])
                best_next_actions = [a for a, q in zip(next_valid_actions, q_vals_next) if q == q_vals_next.max()]
                next_action = np.random.choice(best_next_actions)

            # === Mise à jour SARSA ===
            q_table[state][action] += alpha * (
                reward + gamma * q_table[next_state][next_action] - q_table[state][action]
            )

            # 🔁 Avancer dans l’épisode
            state, action = next_state, next_action

        # 🎯 Score global obtenu sur l’épisode
        episode_scores.append(env.score())

    # === Politique finale (greedy sur Q) ===
    learned_policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in q_table and len(q_table[s]) > 0:
            learned_policy[s] = int(np.argmax(q_table[s]))
        else:
            learned_policy[s] = 0  # Valeur par défaut

    return learned_policy, q_table, episode_scores


def plot_scores(scores, window=100, title="SARSA - Moyenne glissante des scores"):
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
