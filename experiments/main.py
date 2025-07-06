import argparse
import os
from secret_envs_wrapper import (
    SecretEnv0,
    SecretEnv1,
    SecretEnv2,
    SecretEnv3,
)

# === Algorithmes ===

# -- Dynamic Programming (DP)
from reinfolearn.algorithm.dp.policy_iteration import policy_iteration
from reinfolearn.algorithm.dp.value_iteration import value_iteration
# -- Monte Carlo (MC)
from reinfolearn.algorithm.mc.mc_on_policy import mc_on_policy_first_visit
from reinfolearn.algorithm.mc.mc_off_policy import mc_off_policy_control
from reinfolearn.algorithm.mc.mc_exploring_starts import mc_exploring_starts
# -- Temporal Difference (TD)
from reinfolearn.algorithm.td.q_learning import q_learning
from reinfolearn.algorithm.td.sarsa import sarsa
from reinfolearn.algorithm.td.expected_sarsa import expected_sarsa
# -- Planning
from reinfolearn.algorithm.pn.dyna_q_plus import dyna_q_plus
from reinfolearn.algorithm.pn.dyna_q import dyna_q


# === Utils ===
from reinfolearn.utils.io_utils import (
    save_scores,
    save_policy,
    save_q_table,
    save_values,
)
from reinfolearn.utils.plot_utils import plot_scores, plot_scores_overlay


def get_env(name):
    return {
        "env0": SecretEnv0(),
        "env1": SecretEnv1(),
        "env2": SecretEnv2(),
        "env3": SecretEnv3()
    }[name]


def main():
    parser = argparse.ArgumentParser(description="Run RL algorithm")
    parser.add_argument(
        "--env",
        choices=["env0", "env1", "env2", "env3"],
        type=str,
        help="Environnement à utiliser",
        default="env0",
        #required=True
    )
    parser.add_argument(
        "--algo", 
        choices=[
            "policy_iteration", "value_iteration",
            "mc_on", "mc_off", "mc_es",
            "q_learning", "sarsa", "expected_sarsa",
            "dyna_q", "dyna_q_plus"
        ],
        default="q_learning",
        #required=True
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.0)
    args = parser.parse_args()

    env = get_env(args.env)

    print(f"🔍 Environnement : {args.env} | Algo : {args.algo}")
    out_dir = f"outputs/{args.env}/{args.algo}"
    os.makedirs(out_dir, exist_ok=True)

    print("=== Informations sur l'environnement ===")
    print(env)
    print("Nombre d'états :", env.num_states())
    print("Nombre d'actions :", env.num_actions())
    print("Nombre de récompenses :", env.num_rewards())
    print(f"Liste des récompenses :{[env.reward(i) for i in range(env.num_rewards())]}")
    for i in range(env.num_rewards()):
        print(f"r[{i}] :", env.reward(i))
    print(env.p(s=0, a=0, s_p=0, r_index=0)) # Exemple de probabilité de transition
    print("Actions disponibles :", env.available_actions())
    print("========================================")

    if args.algo == "policy_iteration":
        policy, V, scores = policy_iteration(env, gamma=args.gamma, max_iterations=10)
        save_policy(policy, f"{out_dir}/policy.json")
        save_values(V, f"{out_dir}/values.npy")
        save_scores(scores, f"{out_dir}/scores.npy")
        plot_scores(scores, title="Policy Iteration")

    elif args.algo == "value_iteration":
        policy, V, scores = value_iteration(env, gamma=args.gamma, max_iterations=10)
        save_policy(policy, f"{out_dir}/policy.json")
        save_values(V, f"{out_dir}/values.npy")
        save_scores(scores, f"{out_dir}/scores.npy")
        plot_scores(scores, title="Value Iteration")

    elif args.algo == "mc_on":
        policy, Q, scores = mc_on_policy_first_visit(
            env, gamma=args.gamma, epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="MC On-Policy")
        # plot_scores_overlay(scores, title="MC On-Policy (First-Visit)")

    elif args.algo == "mc_off":
        policy, Q, scores = mc_off_policy_control(
            env, gamma=args.gamma, epsilon=args.epsilon,
            nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="MC Off-Policy (Weighted Importance Sampling)")
        # plot_scores_overlay(scores, title="MC Off-Policy (Ordinary Importance Sampling)")

    elif args.algo == "mc_es":
        policy, Q, scores = mc_exploring_starts(
            env, gamma=args.gamma, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="MC Exploring Starts")
        # plot_scores_overlay(scores, title="MC Exploring Starts")

    elif args.algo == "q_learning":
        policy, Q, scores = q_learning(
            env, gamma=args.gamma, alpha=args.alpha,
            epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="Q-Learning")
        # plot_scores_overlay(scores, title="Q-Learning")

    elif args.algo == "sarsa":
        policy, Q, scores = sarsa(
            env, gamma=args.gamma, alpha=args.alpha,
            epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="SARSA")
        # plot_scores_overlay(scores, title="SARSA")

    elif args.algo == "expected_sarsa":
        policy, Q, scores = expected_sarsa(
            env, gamma=args.gamma, alpha=args.alpha,
            epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="Expected SARSA")
        # plot_scores_overlay(scores, title="Expected SARSA")

    elif args.algo == "dyna_q":
        policy, Q, scores = dyna_q(
            env, gamma=args.gamma, alpha=args.alpha,
            epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        # plot_scores(scores, title="Dyna-Q")
        plot_scores(scores, title="Dyna-Q")

    elif args.algo == "dyna_q_plus":
        policy, Q, scores = dyna_q_plus(
            env, gamma=args.gamma, alpha=args.alpha,
            epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="Dyna-Q+")
        # plot_scores_overlay(scores, title="Dyna-Q+")

    print("[OK] Terminé. Résultats enregistrés dans :", out_dir)


if __name__ == "__main__":
    main()
