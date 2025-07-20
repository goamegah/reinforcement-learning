# main.py

import argparse
import os
import sys
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Environnements ===
from reinfolearn.environment.grid_world import GridWorldMDP
from reinfolearn.environment.line_world import LineWorldMDP
from reinfolearn.environment.montyhall_level1 import MontyHallLevel1MDP
from reinfolearn.environment.montyhall_level2 import MontyHallLevel2MDP
from reinfolearn.environment.rock_paper_scissors import RockPaperScissorsMDP
from reinfolearn.environment.rock_paper_scissors_two_rounds import TwoRoundRPS

# === Algorithmes ===
from reinfolearn.algorithm.dp.policy_iteration import policy_iteration
from reinfolearn.algorithm.dp.value_iteration import value_iteration
from reinfolearn.algorithm.mc.mc_on_policy import mc_on_policy_first_visit
from reinfolearn.algorithm.mc.mc_off_policy import mc_off_policy_control
from reinfolearn.algorithm.mc.mc_exploring_starts import mc_exploring_starts
from reinfolearn.algorithm.td.q_learning import q_learning
from reinfolearn.algorithm.td.sarsa import sarsa
from reinfolearn.algorithm.td.expected_sarsa import expected_sarsa
from reinfolearn.algorithm.pn.dyna_q import dyna_q
from reinfolearn.algorithm.pn.dyna_q_plus import dyna_q_plus

# === Utilitaires ===
from reinfolearn.utils.io_utils import save_policy, save_q_table, save_values, save_scores
from reinfolearn.utils.plot_utils import plot_scores, plot_convergence
from reinfolearn.utils.io_utils import load_policy, load_q_table, load_scores, load_values



def get_env(env_name):
    envs = {
        "grid": GridWorldMDP(),
        "line": LineWorldMDP(),
        "monty": MontyHallLevel1MDP(),
        "monty_level2": MontyHallLevel2MDP(),
        "rps": RockPaperScissorsMDP(),
        "rps_two_rounds": TwoRoundRPS()
    }
    return envs[env_name]


def play_policy_step_by_step(env, policy):
    env.reset()
    env.display()
    while not env.is_game_over():
        state = env.state_id()
        action = policy[state]
        print(f"État courant: {state}, Action choisie: {action}")
        env.step(action)
        env.display()
        input("Appuyez sur Entrée pour continuer...")
    print(f" Fin de l’épisode. Score obtenu : {env.score()}")



def main():
    parser = argparse.ArgumentParser(description="Lanceur d'apprentissage RL")
    parser.add_argument("--env", choices=["grid", "line", "monty", "monty_level2", "rps", "rps_two_rounds"], default="grid")
    parser.add_argument("--algo", choices=[
            "policy_iteration", "value_iteration",
            "mc_on", "mc_off", "mc_es",
            "q_learning", "sarsa", "expected_sarsa",
            "dyna_q", "dyna_q_plus"
        ], 
        default="policy_iteration"
    )
    parser.add_argument("--play", action="store_true", help="Exécuter une politique déjà apprise")
    parser.add_argument("--load", action="store_true", help="Charger et afficher les résultats sauvegardés")

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1)
    args = parser.parse_args()

    env = get_env(args.env)
    print(f"[ℹ️] Environnement : {args.env} | Algorithme : {args.algo}")

    output_dir = f"outputs/{args.env}/{args.algo}"
    os.makedirs(output_dir, exist_ok=True)

        
    if args.play:
        try:
            policy_path = f"{output_dir}/policy.json"
            policy = load_policy(policy_path)
            print("Politique chargée.")

            if args.load:
                # Q-table
                q_path = f"{output_dir}/q_table.pkl"
                if os.path.exists(q_path):
                    Q = load_q_table(q_path)
                    print("Q-table chargée.")
                    print(Q)

                # Scores
                scores_path = f"{output_dir}/episode_scores.npy"
                if os.path.exists(scores_path):
                    scores = load_scores(scores_path)
                    print(f"Scores chargés ({len(scores)} épisodes)")
                    plot_scores(scores, title=f"{args.algo} - Score par épisode")

                # Valeurs V(s)
                values_path = f"{output_dir}/values.npy"
                if os.path.exists(values_path):
                    V = load_values(values_path)
                    print("Valeurs des états :")
                    print(V)

                # Convergence (value iteration ou policy iteration)
                convergence_path = (
                    f"{output_dir}/value_convergence.npy"
                    if "policy" in args.algo else f"{output_dir}/delta_convergence.npy"
                )
                if os.path.exists(convergence_path):
                    convergence = np.load(convergence_path)
                    ylabel = "Valeur moyenne V(s)" if "policy" in args.algo else "Delta max"
                    print("Historique de convergence chargé.")
                    plot_convergence(convergence, title=f"{args.algo} - Convergence", ylabel=ylabel)

            print("\n Exécution de la politique chargée...")
            play_policy_step_by_step(env, policy)

        except FileNotFoundError:
            print(" Politique non trouvée. Lance l’apprentissage d’abord.")
            return

        
    else:
        # === Algorithmes de Dynamic Programming ===
        if args.algo == "policy_iteration":
            policy, state_values, mean_state_values_per_iter = policy_iteration(env, gamma=args.gamma, verbose=True)
            save_policy(policy, f"{output_dir}/policy.json")
            save_values(state_values, f"{output_dir}/values.npy")
            save_scores(mean_state_values_per_iter, f"{output_dir}/value_convergence.npy")
            plot_convergence(mean_state_values_per_iter, title="Policy Iteration - Mean V(s)", ylabel="Valeur moyenne V(s)", save_path=f"{output_dir}/convergence_policy_iteration.png")

        elif args.algo == "value_iteration":
            policy, state_values, max_delta_per_iter = value_iteration(env, gamma=args.gamma)
            save_policy(policy, f"{output_dir}/policy.json")
            save_values(state_values, f"{output_dir}/values.npy")
            save_scores(max_delta_per_iter, f"{output_dir}/delta_convergence.npy")
            plot_convergence(max_delta_per_iter, title="Value Iteration - Max Δ", ylabel="Delta max", save_path=f"{output_dir}/convergence_policy_iteration.png")

        # === Algorithmes Monte Carlo ===
        elif args.algo == "mc_on":
            policy, Q, episode_scores = mc_on_policy_first_visit(env, gamma=args.gamma, epsilon=args.epsilon, nb_episodes=args.episodes)
            save_policy(policy, f"{output_dir}/policy.json")
            save_q_table(Q, f"{output_dir}/q_table.pkl")
            save_scores(episode_scores, f"{output_dir}/episode_scores.npy")
            plot_scores(episode_scores, title="MC On-Policy - Score par épisode", save_path=f"{output_dir}/mc_on_policy_scores.png")


        elif args.algo == "mc_off":
            policy, Q, total_scores_per_episode = mc_off_policy_control(env, gamma=args.gamma, epsilon=args.epsilon, nb_episodes=args.episodes)
            save_policy(policy, f"{output_dir}/policy.json")
            save_q_table(Q, f"{output_dir}/q_table.pkl")
            save_scores(total_scores_per_episode, f"{output_dir}/episode_scores.npy")
            plot_scores(total_scores_per_episode, title="MC Off-Policy - Score par épisode", save_path=f"{output_dir}/mc_off_policy_scores.png")

        elif args.algo == "mc_es":
            policy, Q, episode_scores = mc_exploring_starts(env, gamma=args.gamma, nb_episodes=args.episodes)
            save_policy(policy, f"{output_dir}/policy.json")
            save_q_table(Q, f"{output_dir}/q_table.pkl")
            save_scores(episode_scores, f"{output_dir}/episode_scores.npy")
            plot_scores(episode_scores, title="MC Exploring Starts - Score par épisode", save_path=f"{output_dir}/mc_exploring_starts_scores.png")

        # === Algorithmes TD ===
        elif args.algo == "q_learning":
            policy, Q, episode_scores = q_learning(env, gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon, nb_episodes=args.episodes)
            save_policy(policy, f"{output_dir}/policy.json")
            save_q_table(Q, f"{output_dir}/q_table.pkl")
            save_scores(episode_scores, f"{output_dir}/episode_scores.npy")
            plot_scores(episode_scores, title="Q-Learning - Score par épisode", save_path=f"{output_dir}/q_learning_scores.png")

        elif args.algo == "sarsa":
            policy, Q, episode_scores = sarsa(env, gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon, nb_episodes=args.episodes)
            save_policy(policy, f"{output_dir}/policy.json")
            save_q_table(Q, f"{output_dir}/q_table.pkl")
            save_scores(episode_scores, f"{output_dir}/episode_scores.npy")
            plot_scores(episode_scores, title="SARSA - Score par épisode", save_path=f"{output_dir}/sarsa_scores.png")

        elif args.algo == "expected_sarsa":
            policy, Q, episode_scores = expected_sarsa(env, gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon, nb_episodes=args.episodes)
            save_policy(policy, f"{output_dir}/policy.json")
            save_q_table(Q, f"{output_dir}/q_table.pkl")
            save_scores(episode_scores, f"{output_dir}/episode_scores.npy")
            plot_scores(episode_scores, title="Expected SARSA - Score par épisode", save_path=f"{output_dir}/expected_sarsa_scores.png")

        # === Algorithmes Planning ===
        elif args.algo == "dyna_q":
            policy, Q, episode_scores = dyna_q(env, gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon, nb_episodes=args.episodes)
            save_policy(policy, f"{output_dir}/policy.json")
            save_q_table(Q, f"{output_dir}/q_table.pkl")
            save_scores(episode_scores, f"{output_dir}/episode_scores.npy")
            plot_scores(episode_scores, title="Dyna-Q - Score par épisode", save_path=f"{output_dir}/dyna_q_scores.png")

        elif args.algo == "dyna_q_plus":
            policy, Q, episode_scores = dyna_q_plus(env, gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon, nb_episodes=args.episodes)
            save_policy(policy, f"{output_dir}/policy.json")
            save_q_table(Q, f"{output_dir}/q_table.pkl")
            save_scores(episode_scores, f"{output_dir}/episode_scores.npy")
            plot_scores(episode_scores, title="Dyna-Q+ - Score par épisode", save_path=f"{output_dir}/dyna_q_plus_scores.png")

        # Résumé des performances
        if "episode_scores" in locals():
            print(f"Moyenne des scores sur {args.episodes} épisodes : {np.mean(episode_scores):.2f}")
            print(f"Score max : {np.max(episode_scores)} | Score min : {np.min(episode_scores)}")

        print(f"Apprentissage terminé. Résultats disponibles dans : {output_dir}")


if __name__ == "__main__":
    main()
