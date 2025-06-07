from rlearn.environments.rock_paper_scissors import RockPaperScissorsMDP
from rlearn.algorithms.mc.mc_on_policy import mc_on_policy_first_visit, plot_scores

# Initialiser l'environnement
env = RockPaperScissorsMDP()

# Apprentissage avec Monte Carlo On-Policy
policy, Q, scores = mc_on_policy_first_visit(
    env,
    gamma=0.99,
    nb_episodes=5000,
    epsilon=0.1
)

# Affichage des résultats
print("\n✅ Politique apprise :")
for s in range(env.num_states()):
    print(f"État {s} → Action {policy[s]}")

print("\n✅ Valeurs Q :")
for s, q_vals in Q.items():
    print(f"État {s} : {q_vals}")

plot_scores(scores, window=100, title="MC On-Policy - Rock Paper Scissors")
