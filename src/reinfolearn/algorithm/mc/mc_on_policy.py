# src/reinfolearn/algorithm/mc/mc_on_policy.py

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def mc_on_policy_first_visit(env, gamma=0.99, nb_episodes=5000, epsilon=0.1):
    """
    Monte Carlo On-Policy First-Visit (ε-soft policy).
    
    Implémentation pédagogique selon Sutton & Barto.

    :param env: environnement de type MDP
    :param gamma: facteur d'escompte
    :param nb_episodes: nombre d'épisodes pour l'apprentissage
    :param epsilon: taux d'exploration pour ε-greedy
    :return: (policy, Q_table, scores)
    """
    Q = defaultdict(lambda: {})              # Q(s,a)
    returns_by_sa = defaultdict(list)        # Liste des Gt pour chaque (s,a)
    episode_scores = []                      # Historique des scores

    for episode_idx in tqdm(range(nb_episodes), desc="MC On-Policy First-Visit"):
        env.reset()
        episode = []
        score_before = env.score()

        # === Génération de l'épisode avec politique ε-greedy ===
        while not env.is_game_over():
            state = env.state_id()
            actions = env.available_actions()

            if Q[state]:
                q_values = np.array([Q[state].get(a, 0.0) for a in actions])
                best_action = actions[np.argmax(q_values)]
                probs = np.ones(len(actions)) * (epsilon / len(actions))
                probs[np.argmax(q_values)] += (1.0 - epsilon)
                action = np.random.choice(actions, p=probs)
            else:
                action = np.random.choice(actions)

            env.step(action)
            score_after = env.score()
            reward = score_after - score_before
            score_before = score_after

            episode.append((state, action, reward))

        # === Mise à jour First-Visit Monte Carlo ===
        G = 0.0
        visited_state_actions = set()

        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t
            if (state_t, action_t) not in visited_state_actions:
                returns_by_sa[(state_t, action_t)].append(G)
                Q[state_t][action_t] = np.mean(returns_by_sa[(state_t, action_t)])
                visited_state_actions.add((state_t, action_t))

        episode_scores.append(env.score())

    # === Politique finale (greedy) ===
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if Q[s]:
            policy[s] = max(Q[s], key=Q[s].get)
        else:
            policy[s] = 0

    return policy, Q, episode_scores