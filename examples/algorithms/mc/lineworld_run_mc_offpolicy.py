from rlearn.environments.line_world import LineWorldMDP
from rlearn.algorithms.mc.mc_off_policy import mc_off_policy_control, plot_scores

# Créer un environnement LineWorld
env = LineWorldMDP(size=5)

# Exécuter Off-Policy Monte Carlo Control
policy, Q, scores = mc_off_policy_control(
    env,
    nb_episodes=5000,
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

# Affichage du score moyen
plot_scores(scores, window=100, title="MC Off-Policy - LineWorld")
