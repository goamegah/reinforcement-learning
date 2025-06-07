from rlearn.environments.grid_world import GridWorldMDP
from rlearn.algorithms.mc.mc_off_policy import mc_off_policy_control, plot_scores

# Crée un environnement GridWorld 5x5
env = GridWorldMDP(size=5)

# Exécute l'apprentissage Monte Carlo Off-Policy
policy, Q, scores = mc_off_policy_control(
    env,
    nb_episodes=100_000,
    epsilon=0.1,
    weighted=True
)

# Affichage de la politique apprise
print("\n✅ Politique apprise (état → action) :")
for s in range(env.num_states()):
    print(f"État {s} → Action {policy[s]}")

# Affichage des valeurs Q
print("\n✅ Valeurs Q apprises :")
for s, actions in Q.items():
    print(f"État {s} : {actions}")

# Courbe de score moyen
plot_scores(scores, window=100, title="MC Off-Policy - GridWorld")
