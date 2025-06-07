from rlearn.environments.line_world import LineWorldMDP
from rlearn.algorithms.mc.mc_on_policy import mc_on_policy_first_visit, plot_scores

# Créer l'environnement
env = LineWorldMDP(size=5)

# Appliquer l'apprentissage Monte Carlo On-Policy First-Visit
policy, Q, scores = mc_on_policy_first_visit(
    env,
    gamma=0.99,
    nb_episodes=5000,
    epsilon=0.1
)

# Affichage de la politique apprise
print("\n✅ Politique apprise (état → action) :")
for s in range(env.num_states()):
    print(f"État {s} → Action {policy[s]}")

# Affichage des valeurs Q
print("\n✅ Valeurs Q apprises :")
for s, q_vals in Q.items():
    print(f"État {s} : {q_vals}")

# Tracer la courbe de convergence
plot_scores(scores, window=100, title="MC On-Policy - LineWorld")
