import os
import subprocess
import itertools

main_py_path = os.path.join("reinforcement-learning", "src", "reinfolearn", "main.py")
env_name = "grid"                  
algo_name = "sarsa"          
nb_episodes = 500

gammas = [0.9, 0.95, 0.99]
alphas = [0.1, 0.5]
epsilons = [0.1, 0.5]

combinations = list(itertools.product(gammas, alphas, epsilons))
total = len(combinations)


print(f" Lancement de {total} tests sur l’algo '{algo_name}' avec l’environnement '{env_name}'\n")

for i, (gamma, alpha, epsilon) in enumerate(combinations, start=1):
    print(f" [{i}/{total}] γ={gamma} | α={alpha} | ε={epsilon}")
    cmd = [
        "python", main_py_path,
        "--env", env_name,
        "--algo", algo_name,
        "--episodes", str(nb_episodes),
        "--gamma", str(gamma),
        "--alpha", str(alpha),
        "--epsilon", str(epsilon)
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Erreur lors de l'exécution pour : gamma={gamma}, alpha={alpha}, epsilon={epsilon}")
