# rlearn/algorithms/mc/mc_off_policy.py

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

def mc_off_policy_control(env, nb_episodes=5000, gamma=0.99, epsilon=0.1):
    """
    Contrôle Monte Carlo Off-policy avec échantillonnage par importance pondérée.
    Apprend une politique optimale en suivant une autre politique exploratoire.

    :param env: Environnement MDP compatible
    :param nb_episodes: Nombre total d'épisodes
    :param gamma: Facteur d'actualisation
    :param epsilon: Taux d'exploration de la politique comportementale
    :return: politique apprise (greedy), Q-values, scores par épisode
    """
    q_values = defaultdict(dict)  # Q(s,a)
    cumulative_weights = defaultdict(lambda: defaultdict(float))  # C(s,a)
    episode_scores = []

    for _ in tqdm(range(nb_episodes), desc="MC Off-Policy Control"):
        env.reset()
        trajectory = []
        score_before = env.score()

        # === Génération de l’épisode avec politique comportementale ε-greedy ===
        while not env.is_game_over():
            state = env.state_id()
            actions = env.available_actions()

            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                q_vals = [q_values[state].get(a, 0.0) for a in actions]
                max_q = max(q_vals)
                best_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
                action = np.random.choice(best_actions)

            env.step(action)
            score_after = env.score()
            reward = score_after - score_before
            score_before = score_after

            trajectory.append((state, action, reward))

        episode_scores.append(env.score())

        # === Importance Sampling à rebours ===
        G = 0.0
        W = 1.0
        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = gamma * G + reward

            if action not in q_values[state]:
                q_values[state][action] = 0.0

            cumulative_weights[state][action] += W
            q_values[state][action] += (W / cumulative_weights[state][action]) * (G - q_values[state][action])

            # Politique cible (greedy)
            best_action = max(q_values[state], key=q_values[state].get)

            if action != best_action:
                break  # on arrête si la politique comportementale diverge de la cible

            # Poids d'importance
            prob_behavior = epsilon / len(actions)
            if action == best_action:
                prob_behavior += (1.0 - epsilon)

            W *= 1.0 / prob_behavior
            if W == 0.0:
                break

    # === Politique finale (greedy) ===
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in q_values and q_values[s]:
            policy[s] = max(q_values[s], key=q_values[s].get)
        else:
            policy[s] = 0

    return policy, q_values, episode_scores