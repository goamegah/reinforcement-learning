from rlearn.environments.line_world import LineWorldMDP
from rlearn.algorithms.dp.policy_evaluation import policy_evaluation, plot_convergence

if __name__ == "__main__":
    env = LineWorldMDP(size=5)
    n = env.num_states()

    # Politique fixe : aller toujours à droite sauf au bout
    policy = [1 if s < n - 1 else 0 for s in range(n)]

    V, history = policy_evaluation(env, policy, gamma=0.9, theta=1e-5, verbose=True)
    print("Valeurs estimées :", V)
    plot_convergence(history)
