# # rlearn/algorithms/pn/dyna_q.py

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def dyna_q(env, nb_episodes=3000, alpha=0.1, gamma=0.99, epsilon=0.1, planning_steps=50):
    """
    Implémentation robuste de Dyna-Q compatible avec tous les environnements.
    
    Si l'environnement ne définit pas p() et reward(), utilise env.score() pour dériver la récompense.
    """
    q_table = defaultdict(lambda: np.zeros(env.num_actions(), dtype=float))
    model = dict()  # (state, action) -> (next_state, reward)
    seen_state_actions = set()
    episode_scores = []

    for _ in tqdm(range(nb_episodes), desc="Dyna-Q"):
        env.reset()
        current_state = env.state_id()
        score_before = env.score()

        while not env.is_game_over():
            available_actions = env.available_actions()

            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(available_actions)
            else:
                q_values = np.array([q_table[current_state][a] for a in available_actions])
                best_actions = available_actions[np.flatnonzero(q_values == q_values.max())]
                action = np.random.choice(best_actions)

            env.step(action)
            next_state = env.state_id()
            score_after = env.score()

            # Reward implicite via la différence de score
            reward = score_after - score_before
            score_before = score_after

            # Mise à jour Q-learning classique
            next_valid_actions = env.available_actions()
            q_vals_next = [float(q_table[next_state][a]) for a in next_valid_actions]
            max_q_next = np.max(q_vals_next) if q_vals_next else 0.0

            q_table[current_state][action] += alpha * (reward + gamma * max_q_next - q_table[current_state][action])

            # Mise à jour du modèle
            model[(current_state, action)] = (next_state, reward)
            seen_state_actions.add((current_state, action))

            current_state = next_state

            # Étapes de planification (fictives)
            for _ in range(planning_steps):
                s_sim, a_sim = random.choice(list(seen_state_actions))
                s_prime_sim, r_sim = model[(s_sim, a_sim)]

                if s_prime_sim in q_table:
                    max_q_s_prime = np.max(q_table[s_prime_sim])
                else:
                    max_q_s_prime = 0.0

                q_table[s_sim][a_sim] += alpha * (r_sim + gamma * max_q_s_prime - q_table[s_sim][a_sim])

        # Score réel à la fin de l’épisode
        episode_scores.append(env.score())

    # Politique finale (greedy sur Q)
    learned_policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in q_table:
            q_values = np.array([q_table[s][a] for a in range(env.num_actions())])
            learned_policy[s] = np.argmax(q_values)
        else:
            learned_policy[s] = 0

    return learned_policy, q_table, episode_scores