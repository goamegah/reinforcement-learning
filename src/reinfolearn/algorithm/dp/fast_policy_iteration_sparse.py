import numpy as np
from tqdm import tqdm
from collections import defaultdict

def extract_sparse_env(env):
    """
    Extrait une représentation sparse (paresseuse) de l’environnement.
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


def policy_evaluation_sparse(policy, transitions, rewards, gamma=0.99, theta=1e-6, max_iter=1000, use_progress=False):
    """
    Évaluation de politique sur environnement sparse
    """
    S = len(policy)
    V = np.zeros(S)

    iterator = tqdm(range(max_iter), desc="Policy Evaluation") if use_progress else range(max_iter)

    for _ in iterator:
        delta = 0.0
        for s in range(S):
            v = 0.0
            a = policy[s]
            for sp, r, p in transitions[s][a]:
                v += p * (rewards[r] + gamma * V[sp])
            delta = max(delta, abs(V[s] - v))
            V[s] = v
        if delta < theta:
            break
    return V


def fast_policy_iteration_sparse(env, gamma=0.99, theta=1e-6, max_iterations=1000, use_progress=True, verbose=False):
    """
    Version mémoire-optimisée de Policy Iteration pour les environnements très grands.
    Utilise une représentation sparse.
    """
    transitions, rewards = extract_sparse_env(env)
    S = env.num_states()
    A = env.num_actions()

    policy = np.zeros(S, dtype=np.int32)
    mean_value_history = []

    iterator = tqdm(range(max_iterations), desc="Fast Policy Iteration") if use_progress else range(max_iterations)

    for i in iterator:
        V = policy_evaluation_sparse(policy, transitions, rewards, gamma, theta, use_progress=False)
        mean_value_history.append(np.mean(V))

        policy_stable = True
        for s in range(S):
            old_action = policy[s]
            action_values = np.zeros(A)
            for a in range(A):
                for sp, r, p in transitions[s][a]:
                    action_values[a] += p * (rewards[r] + gamma * V[sp])
            best_action = np.argmax(action_values)
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        if verbose:
            print(f"Iter {i}: Mean V = {np.mean(V):.4f} | Stable = {policy_stable}")

        if policy_stable:
            break

    return policy, V, mean_value_history
