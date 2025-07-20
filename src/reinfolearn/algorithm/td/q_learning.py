import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def q_learning(env, nb_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    Q-Learning (Off-policy TD Control) - Version générique.

    Utilise env.score() pour calculer la récompense implicite.
    Compatible avec tous les environnements (y compris ceux sans p() ou reward()).
    """
    q_table = defaultdict(lambda: np.zeros(env.num_actions()))
    episode_scores = []

    for ep in tqdm(range(nb_episodes), desc="Q-Learning"):
        env.reset()
        state = env.state_id()
        score_before = env.score()

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
            score_after = env.score()

            # Reward implicite
            reward = score_after - score_before
            score_before = score_after

            # === Mise à jour Q(s,a) ===
            next_valid_actions = env.available_actions()
            if len(next_valid_actions) > 0:
                max_q_next = max([q_table[next_state][a] for a in next_valid_actions])
            else:
                max_q_next = 0.0

            q_table[state][action] += alpha * (reward + gamma * max_q_next - q_table[state][action])

            state = next_state

        episode_scores.append(env.score())

    # === Politique finale extraite de Q ===
    learned_policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in q_table and len(q_table[s]) > 0:
            learned_policy[s] = int(np.argmax(q_table[s]))
        else:
            learned_policy[s] = 0

    return learned_policy, q_table, episode_scores