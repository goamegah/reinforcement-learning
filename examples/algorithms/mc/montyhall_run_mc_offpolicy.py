from rlearn.environments.montyhall_level1 import MontyHallLevel1MDP
from rlearn.algorithms.mc.mc_off_policy import mc_off_policy_control, plot_scores

# Initialisation de l'environnement
env = MontyHallLevel1MDP()

# Apprentissage via Off-Policy Monte Carlo
policy, Q, scores = mc_off_policy_control(
    env,
    nb_episodes=3000,
    epsilon=0.2,
    weighted=True
)

# Affichage des résultats
print("\n✅ Politique apprise :")
for s in range(env.num_states()):
    print(f"État {s} → Action {policy[s]}")

print("\n✅ Valeurs Q apprises :")
for s, q_vals in Q.items():
    print(f"État {s} : {q_vals}")

# Tracé des scores moyens
plot_scores(scores, title="MC Off-Policy - Monty Hall Level 1")
