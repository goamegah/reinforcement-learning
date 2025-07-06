# rlearn/algorithms/mc/mc_on_policy.py

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def mc_on_policy_first_visit(env, gamma=0.99, nb_episodes=5000, epsilon=0.1):
    """
    On-Policy First-Visit Monte Carlo Control (ε-soft policies).
    
    Implémentation pédagogique conforme à Sutton & Barto.
    
    :param env: environnement compatible
    :param gamma: facteur de discount
    :param nb_episodes: nombre d'épisodes
    :param epsilon: taux d'exploration ε
    :return: politique optimale, Q, historique des scores
    """
    Q = defaultdict(lambda: {})            # Fonction d'action-valeur
    returns = defaultdict(list)            # Liste des retours pour moyenne
    scores = []

    # Initialisation : politique ε-soft (implémentée via ε-greedy à chaque update)
    for ep in tqdm(range(nb_episodes), desc="MC On-Policy First Visit"):
        env.reset()
        episode = []  # Liste (s,a,r)

        # === Génération de l'épisode suivant π (ε-greedy) ===
        while not env.is_game_over():
            s = env.state_id()
            actions = env.available_actions()

            # ε-greedy policy π
            if s in Q and len(Q[s]) > 0:
                q_vals = np.array([Q[s].get(a_, 0.0) for a_ in actions])
                best_a = actions[np.argmax(q_vals)]
                probs = np.ones(len(actions)) * (epsilon / len(actions))
                probs[np.argmax(q_vals)] += (1 - epsilon)
                a = np.random.choice(actions, p=probs)
            else:
                a = np.random.choice(actions)

            # === Reward immédiate (via MDP) ===
            r = 0.0
            s_next = None
            env.step(a)
            s_next = env.state_id()
            for r_idx in range(env.num_rewards()):
                r += env.p(s, a, s_next, r_idx) * env.reward(r_idx)

            episode.append((s, a, r))

        # === Backward First-Visit MC update ===
        G = 0.0
        visited_sa = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            # First-visit : mise à jour uniquement si première visite du (s,a)
            if (s, a) not in visited_sa:
                returns[(s, a)].append(G)
                Q[s][a] = np.mean(returns[(s, a)])
                visited_sa.add((s, a))

        # === Politique π mise à jour pour être ε-soft greedy sur Q ===
        for s in Q:
            actions = list(Q[s].keys())
            q_vals = np.array([Q[s][a] for a in actions])
            best_a = actions[np.argmax(q_vals)]
            pi_s = {}
            for a in actions:
                if a == best_a:
                    pi_s[a] = 1 - epsilon + (epsilon / len(actions))
                else:
                    pi_s[a] = epsilon / len(actions)
            # Politique stockée si besoin (ici non utilisée directement)
        
        # Score de l'épisode (utile pour suivi convergence)
        scores.append(env.score())

    # === Politique finale greedy extraite de Q ===
    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if s in Q and Q[s]:
            policy[s] = max(Q[s], key=Q[s].get)
        else:
            policy[s] = 0

    return policy, Q, scores




# def mc_on_policy_first_visit(env, gamma=0.99, nb_episodes=5000, epsilon=0.1):
#     """
#     Monte Carlo On-Policy First-Visit avec epsilon-greedy et discount gamma
#     :param env: environnement compatible
#     :param gamma: facteur de discount
#     :param nb_episodes: nombre d'épisodes à exécuter
#     :param epsilon: exploration aléatoire (ε-greedy)
#     :return: politique apprise, Q, historique des scores
#     """
#     Q = defaultdict(lambda: {})  # Q[s][a]
#     returns = defaultdict(list)
#     scores = []

#     for ep in tqdm(range(nb_episodes), desc="MC On-Policy First Visit"):
#         env.reset()
#         episode = []  # (s, a, r)
#         visited = set()

#         while not env.is_game_over():
#             s = env.state_id()
#             actions = env.available_actions()

#             # ε-greedy
#             if s in Q and len(Q[s]) > 0:
#                 if np.random.rand() < epsilon:
#                     a = np.random.choice(actions)
#                 else:
#                     q_vals = np.array([Q[s].get(a_, 0.0) for a_ in actions])
#                     a = actions[np.argmax(q_vals)]
#             else:
#                 a = np.random.choice(actions)

#             episode.append((s, a))
#             env.step(a)

#         # @@ Calcul du retour G avec discount
#         G = 0.0
#         visited = set()
#         for t in reversed(range(len(episode))):
#             s, a = episode[t]
#             reward = 0.0
#             if t == len(episode) - 1:
#                 reward = env.score()
#             G = gamma * G + reward

#             if (s, a) not in visited:
#                 if a not in Q[s]:
#                     Q[s][a] = 0.0
#                 returns[(s, a)].append(G)
#                 Q[s][a] = np.mean(returns[(s, a)])
#                 visited.add((s, a))

#         scores.append(env.score())

#     # Politique finale : greedy
#     policy = np.zeros(env.num_states(), dtype=int)
#     for s in range(env.num_states()):
#         if s in Q and Q[s]:
#             policy[s] = max(Q[s], key=Q[s].get)
#         else:
#             policy[s] = 0

#     return policy, Q, scores


# def plot_scores(scores, window=100, title="MC On-Policy First-Visit - Score moyen"):
#     if len(scores) >= window:
#         moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
#     else:
#         moving_avg = scores
#     plt.figure(figsize=(8, 4))
#     plt.plot(moving_avg)
#     plt.title(title)
#     plt.xlabel("Épisode")
#     plt.ylabel("Score moyen")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
