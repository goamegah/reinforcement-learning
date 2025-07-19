import numpy as np

def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000):
    V = np.zeros(env.num_states())
    delta_history = []

    for i in range(max_iterations):
        delta = 0
        for s in range(env.num_states()):
            v = V[s]
            action_values = np.zeros(env.num_actions())
            for a in range(env.num_actions()):
                action_values[a] = sum(
                    env.p(s, a, s_p, r_idx) * (env.reward(r_idx) + gamma * V[s_p])
                    for s_p in range(env.num_states())
                    for r_idx in range(env.num_rewards())
                )
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        delta_history.append(delta)
        if delta < theta:
            break

    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        action_values = np.zeros(env.num_actions())
        for a in range(env.num_actions()):
            action_values[a] = sum(
                env.p(s, a, s_p, r_idx) * (env.reward(r_idx) + gamma * V[s_p])
                for s_p in range(env.num_states())
                for r_idx in range(env.num_rewards())
            )
        policy[s] = np.argmax(action_values)

    return policy, V, delta_history
