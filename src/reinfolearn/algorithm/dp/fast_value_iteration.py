import numpy as np
from numba import njit
from tqdm import tqdm

@njit
def compute_action_value(env, s, a, V, gamma):
    q_sa = 0.0
    for s_p in range(env.num_states()):
        for r_idx in range(env.num_rewards()):
            prob = env.p(s, a, s_p, r_idx)
            reward = env.reward(r_idx)
            q_sa += prob * (reward + gamma * V[s_p])
    return q_sa

def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000, use_progress=True):
    """
    Value Iteration optimisée avec jauge de progression. Ne stocke pas toute la matrice de transition.
    """
    S = env.num_states()
    A = env.num_actions()
    V = np.zeros(S)
    delta_history = []

    range_fn = tqdm(range(max_iterations), desc="Value Iteration") if use_progress else range(max_iterations)

    for i in range_fn:
        delta = 0.0
        for s in range(S):
            v_old = V[s]
            q_values = np.zeros(A)
            for a in range(A):
                q_values[a] = compute_action_value(env, s, a, V, gamma)
            V[s] = np.max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        delta_history.append(delta)
        if delta < theta:
            break

    # Politique extraite de V
    policy = np.zeros(S, dtype=np.int32)
    for s in range(S):
        q_values = np.zeros(A)
        for a in range(A):
            q_values[a] = compute_action_value(env, s, a, V, gamma)
        policy[s] = np.argmax(q_values)

    return policy, V, delta_history
