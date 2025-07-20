import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def dyna_q_plus(env, nb_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, planning_steps=10, kappa=1e-4):
    """
    Dyna-Q+ : Apprentissage par planification avec exploration bonus (Sutton & Barto).
    
    Compatible avec tous les environnements, même sans reward() ou p().
    Ajoute un bonus basé sur le temps écoulé depuis la dernière visite d’un couple (état, action).
    """
    q_table = defaultdict(lambda: np.zeros(env.num_actions(), dtype=float))
    model = dict()               # (state, action) -> (next_state, reward)
    last_visit = dict()         # (state, action) -> time of last visit
    episode_scores = []
    t = 0  # temps global (nombre total de pas effectués)

    for _ in tqdm(range(nb_episodes), desc="Dyna-Q+"):
        env.reset()
        state = env.state_id()
        score_before = env.score()

        while not env.is_game_over():
            t += 1
            available_actions = env.available_actions()

            # ε-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(available_actions)
            else:
                q_values = np.array([q_table[state][a] for a in available_actions])
                best_actions = available_actions[np.flatnonzero(q_values == q_values.max())]
                action = np.random.choice(best_actions)

            env.step(action)
            next_state = env.state_id()
            score_after = env.score()

            # Reward implicite par différence de score
            reward = score_after - score_before
            score_before = score_after

            # Mise à jour Q-learning classique
            next_valid_actions = env.available_actions()
            q_vals_next = [q_table[next_state][a] for a in next_valid_actions]
            max_q_next = np.max(q_vals_next) if q_vals_next else 0.0

            q_table[state][action] += alpha * (reward + gamma * max_q_next - q_table[state][action])

            # Mise à jour du modèle
            model[(state, action)] = (next_state, reward)
            last_visit[(state, action)] = t

            # Planification fictive avec bonus d’exploration
            for _ in range(planning_steps):
                if not model:
                    break

                s_sim, a_sim = random.choice(list(model.keys()))
                s_prime_sim, r_sim = model[(s_sim, a_sim)]

                # ⏱️ Bonus d’exploration (temps écoulé)
                tau = t - last_visit.get((s_sim, a_sim), 0)
                bonus = kappa * np.sqrt(tau)
                total_reward = r_sim + bonus

                next_actions_sim = env.available_actions()
                q_vals_sim = [q_table[s_prime_sim][a] for a in next_actions_sim]
                max_q_sim = np.max(q_vals_sim) if q_vals_sim else 0.0

                q_table[s_sim][a_sim] += alpha * (total_reward + gamma * max_q_sim - q_table[s_sim][a_sim])

            state = next_state

        episode_scores.append(env.score())

    # Politique finale (greedy)
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        valid_actions = list(range(env.num_actions()))
        if s in q_table and len(valid_actions) > 0:
            q_vals = np.array([q_table[s][a] for a in valid_actions])
            policy[s] = valid_actions[np.argmax(q_vals)]
        else:
            policy[s] = 0

    return policy, q_table, episode_scores