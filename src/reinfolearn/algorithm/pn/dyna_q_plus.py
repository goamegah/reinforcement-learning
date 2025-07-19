# rlearn/algorithms/pn/dyna_q_plus.py

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def dyna_q_plus(env, nb_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, planning_steps=10, kappa=1e-4):
    """
    Dyna-Q+ : Apprentissage avec planification et exploration bonus.
    Conforme à Sutton & Barto, avec exploration optimiste via un bonus dépendant du temps écoulé.
    """
    q_table = defaultdict(lambda: np.zeros(env.num_actions(), dtype=float))
    model = dict()               # (s, a) -> (s', r)
    last_visit = dict()          # (s, a) -> last time seen
    episode_scores = []
    t = 0                        # temps global

    for _ in tqdm(range(nb_episodes), desc="Dyna-Q+"):
        env.reset()
        state = env.state_id()

        while not env.is_game_over():
            t += 1
            available_actions = env.available_actions()

            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(available_actions)
            else:
                q_values = np.array([q_table[state][a] for a in available_actions])
                best_actions = available_actions[np.flatnonzero(q_values == q_values.max())]
                action = np.random.choice(best_actions)

            # Avancer dans l'environnement
            env.step(action)
            next_state = env.state_id()

            # ✅ Reward immédiate (selon définition MDP)
            reward = sum(env.p(state, action, next_state, r_idx) * env.reward(r_idx)
                         for r_idx in range(env.num_rewards()))

            # ✅ Q-learning update
            next_valid_actions = env.available_actions()
            q_vals_next = [float(q_table[next_state][a]) for a in next_valid_actions]
            max_q_next = np.max(q_vals_next) if q_vals_next else 0.0

            q_table[state][action] += alpha * (reward + gamma * max_q_next - q_table[state][action])

            # ✅ Mise à jour du modèle + horodatage
            model[(state, action)] = (next_state, reward)
            last_visit[(state, action)] = t

            # ✅ Planification : générer des transitions fictives
            for _ in range(planning_steps):
                if not model:
                    break

                s_sim, a_sim = random.choice(list(model.keys()))
                s_prime_sim, r_sim = model[(s_sim, a_sim)]

                # ⏱️ Bonus d'exploration
                tau = t - last_visit.get((s_sim, a_sim), 0)
                bonus = kappa * np.sqrt(tau)
                total_reward = r_sim + bonus

                # Actions disponibles fictivement depuis s'
                next_actions_sim = env.available_actions()
                q_vals_sim = [float(q_table[s_prime_sim][a]) for a in next_actions_sim]
                max_q_sim = np.max(q_vals_sim) if q_vals_sim else 0.0

                # ✅ Q update fictif
                q_table[s_sim][a_sim] += alpha * (total_reward + gamma * max_q_sim - q_table[s_sim][a_sim])

            state = next_state

        episode_scores.append(env.score())

    # 🎯 Politique finale (greedy)
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        valid_actions = list(range(env.num_actions()))
        if s in q_table and len(valid_actions) > 0:
            q_vals = np.array([q_table[s][a] for a in valid_actions])
            policy[s] = valid_actions[np.argmax(q_vals)]
        else:
            policy[s] = 0

    return policy, q_table, episode_scores


def plot_scores(scores, window=100, title="Dyna-Q+ - Score moyen par épisode"):
    """
    Affiche la moyenne glissante des scores de Dyna-Q+.
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
