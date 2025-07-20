import numpy as np
from tqdm import tqdm

def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    V = np.zeros(env.num_states())
    delta_history = []

    while True:
        delta = 0
        for s in range(env.num_states()):
            v = V[s]
            a = policy[s]
            v_new = sum(
                env.p(s, a, s_p, r_idx) * (env.reward(r_idx) + gamma * V[s_p])
                for s_p in range(env.num_states())
                for r_idx in range(env.num_rewards())
            )
            V[s] = v_new
            delta = max(delta, abs(v - v_new))
        delta_history.append(delta)
        if delta < theta:
            break
    return V, delta_history


def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000, verbose=False):
    policy = np.zeros(env.num_states(), dtype=int)
    V = np.zeros(env.num_states())
    mean_value_history = []

    for i in tqdm(range(max_iterations), desc="Policy Iteration"):
        V, delta_history = policy_evaluation(env, policy, gamma, theta)
        mean_value_history.append(np.mean(V))

        policy_stable = True
        for s in range(env.num_states()):
            old_action = policy[s]
            action_values = np.zeros(env.num_actions())
            for a in range(env.num_actions()):
                action_values[a] = sum(
                    env.p(s, a, s_p, r_idx) * (env.reward(r_idx) + gamma * V[s_p])
                    for s_p in range(env.num_states())
                    for r_idx in range(env.num_rewards())
                )
            best_action = np.argmax(action_values)
            policy[s] = best_action
            if old_action != best_action:
                policy_stable = False

        if verbose:
            print(f"Iteration {i}: Mean V = {np.mean(V):.4f}, Policy Stable = {policy_stable}")

        if policy_stable:
            break

    return policy, V, mean_value_history
