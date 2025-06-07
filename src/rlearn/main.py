import argparse
import os
from rlearn.environments.grid_world import GridWorldMDP
from rlearn.environments.line_world import LineWorldMDP
from rlearn.environments.montyhall_level1 import MontyHallLevel1MDP
from rlearn.environments.montyhall_level2 import MontyHallLevel2MDP
from rlearn.environments.rock_paper_scissors import RockPaperScissorsMDP
from rlearn.environments.rock_paper_scissors_two_rounds import TwoRoundRPSMDP

# === Algorithmes ===

# -- Dynamic Programming (DP)
from rlearn.algorithms.dp.policy_iteration import policy_iteration
from rlearn.algorithms.dp.value_iteration import value_iteration
# -- Monte Carlo (MC)
from rlearn.algorithms.mc.mc_on_policy import mc_on_policy_first_visit
from rlearn.algorithms.mc.mc_off_policy import mc_off_policy_control
from rlearn.algorithms.mc.mc_exploring_starts import mc_exploring_starts
# -- Temporal Difference (TD)
from rlearn.algorithms.td.q_learning import q_learning
from rlearn.algorithms.td.sarsa import sarsa
from rlearn.algorithms.td.expected_sarsa import expected_sarsa

# === Utils ===
from rlearn.utils.io_utils import (
    save_scores,
    save_policy,
    save_q_table,
    save_values,
)
from rlearn.utils.plot_utils import plot_scores


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
        default="monty_level2",
        #required=True
    )
    parser.add_argument(
        "--algo", 
        choices=[
            "policy_iteration", "value_iteration",
            "mc_on", "mc_off", "mc_es",
            "q_learning", "sarsa", "expected_sarsa"
        ],
        default="mc_on",
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

    print("[OK] Terminé. Résultats enregistrés dans :", out_dir)


if __name__ == "__main__":
    main()
