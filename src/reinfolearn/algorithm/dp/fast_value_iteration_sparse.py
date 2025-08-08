import numpy as np
from tqdm import tqdm
from collections import defaultdict

def extract_sparse_env(env):
    """
    Extrait une représentation sparse (paresseuse) de l'environnement.
    transitions[s][a] = liste de (s', r, p)
    """
    S = env.num_states()
    A = env.num_actions()
    R = env.num_rewards()

    transitions = defaultdict(lambda: defaultdict(list))
    for s in tqdm(range(S), desc="Extraction transitions"):
        for a in range(A):
            for sp in range(S):
                for r in range(R):
                    p = env.p(s, a, sp, r)
                    if p > 0:
                        transitions[s][a].append((sp, r, p))

    rewards = np.array([env.reward(r) for r in range(R)])
    return transitions, rewards


def fast_value_iteration_sparse(env, gamma=0.99, theta=1e-6, max_iterations=1000, use_progress=True, verbose=False):
    """
    Version mémoire-optimisée de Value Iteration pour les environnements très grands.
    Utilise une représentation sparse des transitions.
    """
    transitions, rewards = extract_sparse_env(env)
    S = env.num_states()
    A = env.num_actions()

    V = np.zeros(S)
    delta_history = []

    iterator = tqdm(range(max_iterations), desc="Fast Value Iteration") if use_progress else range(max_iterations)

    for i in iterator:
        delta = 0.0
        for s in range(S):
            v_old = V[s]
            action_values = np.zeros(A)
            for a in range(A):
                for sp, r, p in transitions[s][a]:
                    action_values[a] += p * (rewards[r] + gamma * V[sp])
            V[s] = np.max(action_values)
            delta = max(delta, abs(v_old - V[s]))
        
        delta_history.append(delta)
        
        if verbose:
            print(f"Iter {i}: Delta = {delta:.6f}, Mean V = {np.mean(V):.4f}")
            
        if delta < theta:
            break

    # Politique extraite de V
    policy = np.zeros(S, dtype=np.int32)
    for s in range(S):
        action_values = np.zeros(A)
        for a in range(A):
            for sp, r, p in transitions[s][a]:
                action_values[a] += p * (rewards[r] + gamma * V[sp])
        policy[s] = np.argmax(action_values)

    return policy, V, delta_history