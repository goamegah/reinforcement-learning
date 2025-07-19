import numpy as np
from collections import defaultdict
from tqdm import tqdm

def mc_on_policy_first_visit(env, gamma=0.99, nb_episodes=5000, epsilon=0.1):
    """
    Monte Carlo On-Policy First-Visit (ε-soft policy).
    
    Implémentation pédagogique selon Sutton & Barto.
    
    :param env: environnement de type MDP
    :param gamma: facteur d'escompte
    :param nb_episodes: nombre d'épisodes pour l'apprentissage
    :param epsilon: taux d'exploration pour ε-greedy
    :return: politique finale (greedy), dictionnaire Q(s,a), et scores par épisode
    """
    Q = defaultdict(lambda: {})              # Fonction d'action-valeur Q(s,a)
    returns_by_sa = defaultdict(list)        # Pour stocker tous les Gt(s,a)
    episode_scores = []                      # Pour suivre la performance

    for episode_idx in tqdm(range(nb_episodes), desc="MC On-Policy First-Visit"):
        env.reset()
        episode = []  # Liste de (state, action, reward)

        # === Génération de l'épisode selon π ε-greedy ===
        while not env.is_game_over():
            state = env.state_id()
            actions = env.available_actions()

            # Politique ε-greedy à partir de Q(s,a)
            if Q[state]:
                q_values = np.array([Q[state].get(a_, 0.0) for a_ in actions])
                best_action = actions[np.argmax(q_values)]
                probs = np.ones(len(actions)) * (epsilon / len(actions))
                probs[np.argmax(q_values)] += (1 - epsilon)
                action = np.random.choice(actions, p=probs)
            else:
                action = np.random.choice(actions)

            # Calcul du reward attendu (selon les probas du MDP)
            reward = 0.0
            env.step(action)
            next_state = env.state_id()
            for r_idx in range(env.num_rewards()):
                reward += env.p(state, action, next_state, r_idx) * env.reward(r_idx)

            episode.append((state, action, reward))

        # === Mise à jour First-Visit MC (à rebours) ===
        G = 0.0
        visited_state_actions = set()
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t
            if (state_t, action_t) not in visited_state_actions:
                returns_by_sa[(state_t, action_t)].append(G)
                Q[state_t][action_t] = np.mean(returns_by_sa[(state_t, action_t)])
                visited_state_actions.add((state_t, action_t))

        # Enregistrement du score final de l'épisode
        episode_scores.append(env.score())

    # === Politique finale déterministe extraite de Q (greedy) ===
    learned_policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        if Q[s]:
            learned_policy[s] = max(Q[s], key=Q[s].get)
        else:
            learned_policy[s] = 0

    return learned_policy, Q, episode_scores
