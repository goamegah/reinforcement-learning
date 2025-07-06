import argparse
import os
from reinfolearn.environment.grid_world import GridWorldMDP
from reinfolearn.environment.line_world import LineWorldMDP
from reinfolearn.environment.montyhall_level1 import MontyHallLevel1MDP
from reinfolearn.environment.montyhall_level2 import MontyHallLevel2MDP
from reinfolearn.environment.rock_paper_scissors import RockPaperScissorsMDP
from reinfolearn.environment.rock_paper_scissors_two_rounds import TwoRoundRPSMDP

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
from reinfolearn.utils.plot_utils import plot_scores


def get_env(name):
    return {
        "grid": GridWorldMDP(),
        "line": LineWorldMDP(),
        "monty": MontyHallLevel1MDP(),
        "monty_level2": MontyHallLevel2MDP(),
        "rps": RockPaperScissorsMDP(),
        "rps_two_rounds": TwoRoundRPSMDP(),
    }[name]


def main():
    parser = argparse.ArgumentParser(description="Run RL algorithm")
    parser.add_argument(
        "--env",
        choices=["grid", "line", "monty", "rps", "rps_two_rounds", "monty_level2"],
        type=str,
        help="Environnement à utiliser",
        default="grid",
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
        default="dyna_q_plus",
        #required=True
    )
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    env = get_env(args.env)

    print(f"🔍 Environnement : {args.env} | Algo : {args.algo}")
    out_dir = f"outputs/{args.env}/{args.algo}"
    os.makedirs(out_dir, exist_ok=True)

    if args.algo == "policy_iteration":
        policy, V, scores = policy_iteration(env, gamma=args.gamma)
        save_policy(policy, f"{out_dir}/policy.json")
        save_values(V, f"{out_dir}/values.npy")
        save_scores(scores, f"{out_dir}/scores.npy")
        plot_scores(scores, title="Policy Iteration")

    elif args.algo == "value_iteration":
        policy, V, scores = value_iteration(env, gamma=args.gamma)
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

    elif args.algo == "mc_off":
        policy, Q, scores = mc_off_policy_control(
            env, gamma=args.gamma, epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="MC Off-Policy")

    elif args.algo == "mc_es":
        policy, Q, scores = mc_exploring_starts(
            env, gamma=args.gamma, epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="MC Exploring Starts")

    elif args.algo == "q_learning":
        policy, Q, scores = q_learning(
            env, gamma=args.gamma, alpha=args.alpha,
            epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="Q-Learning")

    elif args.algo == "sarsa":
        policy, Q, scores = sarsa(
            env, gamma=args.gamma, alpha=args.alpha,
            epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="SARSA")

    elif args.algo == "expected_sarsa":
        policy, Q, scores = expected_sarsa(
            env, gamma=args.gamma, alpha=args.alpha,
            epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
        plot_scores(scores, title="Expected SARSA")

    elif args.algo == "dyna_q":
        policy, Q, scores = dyna_q(
            env, gamma=args.gamma, alpha=args.alpha,
            epsilon=args.epsilon, nb_episodes=args.episodes
        )
        save_policy(policy, f"{out_dir}/policy.json")
        save_scores(scores, f"{out_dir}/scores.npy")
        save_q_table(Q, f"{out_dir}/q_table.pkl")
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

    print("[OK] Terminé. Résultats enregistrés dans :", out_dir)


if __name__ == "__main__":
    main()
