# examples/algorithms/mc/montyhall_run_mc_onpolicy.py
from rlearn.environments.montyhall_level1 import MontyHallLevel1MDP
from rlearn.algorithms.mc.mc_on_policy import mc_on_policy_first_visit, plot_scores

env = MontyHallLevel1MDP()
policy, Q, scores = mc_on_policy_first_visit(env, nb_episodes=1_000, epsilon=0.1)

plot_scores(scores)

print("Politique apprise :")
for s in range(env.num_states()):
    print(f"État {s} → Action {policy[s]}")
