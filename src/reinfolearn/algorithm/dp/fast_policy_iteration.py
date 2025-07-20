import numpy as np
from numba import njit
from tqdm import tqdm


def extract_env_matrices(env):
    """
    Extrait les matrices de transitions P et le vecteur de récompenses R à partir d’un environnement compatible.
    """
    S = env.num_states()
    A = env.num_actions()
    R = env.num_rewards()

    P = np.zeros((S, A, S, R))
    rewards = np.zeros(R)

    for r in range(R):
        rewards[r] = env.reward(r)

    for s in range(S):
        for a in range(A):
            for sp in range(S):
                for r in range(R):
                    P[s, a, sp, r] = env.p(s, a, sp, r)

    return P, rewards


@njit
def fast_policy_evaluation(policy, P, R_vec, gamma=0.99, theta=1e-6, max_iter=1000):
    """
    Évaluation de politique optimisée à l’aide de Numba.
    """
    S, A, S_, R_ = P.shape
    V = np.zeros(S)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(S):
            a = policy[s]
            v_new = 0.0
            for sp in range(S_):
                for r in range(R_):
                    v_new += P[s, a, sp, r] * (R_vec[r] + gamma * V[sp])
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < theta:
            break
    return V


def policy_evaluation_with_progress(policy, P, R_vec, gamma=0.99, theta=1e-6, max_iter=1000):
    """
    Évaluation de politique AVEC jauge de progression (pour le debug, mais non numba-compatible).
    """
    S, A, S_, R_ = P.shape
    V = np.zeros(S)
    for it in tqdm(range(max_iter), desc="Policy Evaluation"):
        delta = 0.0
        for s in range(S):
            a = policy[s]
            v_new = 0.0
            for sp in range(S_):
                for r in range(R_):
                    v_new += P[s, a, sp, r] * (R_vec[r] + gamma * V[sp])
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < theta:
            break
    return V


def fast_policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000, verbose=False, use_progress=False):
    """
    Policy Iteration accélérée avec Numba, jauge facultative.
    """
    P, R_vec = extract_env_matrices(env)
    S = env.num_states()
    A = env.num_actions()

    policy = np.zeros(S, dtype=np.int32)
    mean_value_history = []

    for i in tqdm(range(max_iterations), desc="Fast Policy Iteration"):
        if use_progress:
            V = policy_evaluation_with_progress(policy, P, R_vec, gamma, theta)
        else:
            V = fast_policy_evaluation(policy, P, R_vec, gamma, theta)

        mean_value_history.append(np.mean(V))

        policy_stable = True
        for s in range(S):
            old_action = policy[s]
            action_values = np.zeros(A)
            for a in range(A):
                for sp in range(S):
                    for r in range(R_vec.shape[0]):
                        action_values[a] += P[s, a, sp, r] * (R_vec[r] + gamma * V[sp])
            best_action = np.argmax(action_values)
            policy[s] = best_action
            if old_action != best_action:
                policy_stable = False

        if verbose:
            print(f"Iter {i}: Mean V = {np.mean(V):.4f}, Policy Stable = {policy_stable}")

        if policy_stable:
            break

    return policy, V, mean_value_history
