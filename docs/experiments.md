# Documentation des Expérimentations

Ce document décrit les expérimentations réalisées avec le framework d'apprentissage par renforcement, les environnements utilisés, et les résultats obtenus.

## Notebooks d'expérimentation

Le répertoire `experiments/` contient plusieurs notebooks Jupyter qui permettent de comparer les performances des différents algorithmes sur divers environnements.

### experimentation_secret_envs.ipynb

Ce notebook compare les performances de plusieurs algorithmes d'apprentissage par renforcement sur des environnements secrets. Les algorithmes comparés sont :

- Monte Carlo On-Policy (First-Visit)
- Monte Carlo avec Exploring Starts
- Q-Learning
- SARSA
- Expected SARSA
- Dyna-Q
- Dyna-Q+
- Policy Iteration
- Value Iteration

#### Structure du notebook

1. **Importation des modules** : Importation des environnements, algorithmes et utilitaires nécessaires
2. **Fonctions utilitaires** : Définition de fonctions pour la visualisation et l'expérimentation
3. **Expérimentations sur les environnements** : Exécution des algorithmes sur chaque environnement secret
4. **Comparaison graphique** : Visualisation et interprétation des résultats

#### Fonction d'expérimentation

La fonction `run_experiment` est utilisée pour exécuter un algorithme sur un environnement et sauvegarder les résultats :

```python
def run_experiment(env_class, algo_fn, algo_name, env_name, **kwargs):
    env = env_class()
    output_dir = f"outputs/{env_name}/{algo_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Exécution de l'algorithme et sauvegarde des résultats
    if algo_name == "policy_iteration" or algo_name == "value_iteration":
        policy, V, history = algo_fn(env, **kwargs)
        save_policy(policy, f"{output_dir}/policy.json")
        save_values(V, f"{output_dir}/values.npy")
        save_scores(history, f"{output_dir}/convergence.npy")
    else:
        policy, Q, scores = algo_fn(env, **kwargs)
        save_policy(policy, f"{output_dir}/policy.json")
        save_q_table(Q, f"{output_dir}/q_table.pkl")
        save_scores(scores, f"{output_dir}/episode_scores.npy")
    
    return policy, Q if 'Q' in locals() else V, scores if 'scores' in locals() else history
```

## Environnements secrets

Les expérimentations sont réalisées sur quatre environnements secrets :

1. **SecretEnv0** : 16 états, 4 actions, 3 récompenses distinctes
2. **SecretEnv1** : 64 états, 4 actions, 3 récompenses distinctes
3. **SecretEnv2** : 100 états, 4 actions, 3 récompenses distinctes
4. **SecretEnv3** : 81 états, 4 actions, 3 récompenses distinctes

Ces environnements sont accessibles via des wrappers dans le module `experiments.secret_envs_wrapper`.

## Résultats des expérimentations

### Environnement SecretEnv0

Dans cet environnement, les meilleurs résultats sont atteints par l'algorithme SARSA, qui converge très rapidement vers un score élevé (supérieur à 8) et reste stable. Il est suivi de près par Q-Learning et Expected SARSA (scores autour de 7 à 7.5).

Dyna-Q+ progresse plus lentement mais finit par dépasser légèrement Expected SARSA, tandis que Dyna-Q reste en dessous avec des scores moyens autour de 5.5.

Les algorithmes Monte Carlo (MC On-Policy et MC Exploring Starts) montrent des performances inférieures, avec MC On-Policy plafonnant à un score d'environ 4 et MC Exploring Starts stagnant à un score fixe proche de 2.

### Environnement SecretEnv1

Dans cet environnement, Expected SARSA est clairement supérieur, atteignant rapidement un score moyen proche de 20 et s'y stabilisant. Q-Learning, SARSA et Dyna-Q suivent avec des scores moyens stabilisés autour de 16.

Dyna-Q+ montre une progression plus lente mais régulière, finissant autour de 15. MC On-Policy atteint un score final inférieur à 11, tandis que MC Exploring Starts reste à environ 3.

### Environnement SecretEnv2

Cet environnement est plus difficile, avec des scores initiaux très bas (autour de -55). Dyna-Q montre une progression impressionnante, atteignant un score proche de -15 et continuant à s'améliorer.

MC On-Policy et MC Exploring Starts se stabilisent vers -30, tandis que Q-Learning, Expected SARSA et SARSA convergent autour de -32 à -30. Dyna-Q+ affiche des performances fluctuantes, finissant au-dessus de -35 mais sans réelle stabilité.

### Environnement SecretEnv3

Dans cet environnement, tous les algorithmes débutent avec des scores moyens très négatifs (proches de -15). Expected SARSA atteint les meilleures performances finales avec un score moyen supérieur à +2.5, suivi de près par Q-Learning et SARSA (scores légèrement supérieurs à 2).

Dyna-Q plafonne autour de 1.5, tandis que Dyna-Q+ oscille autour de -2. MC On-Policy et MC Exploring Starts restent les moins performants, avec des scores moyens stagnant autour de -4 à -5.

## Interprétation générale

À travers ces expérimentations, on observe que :

1. **Expected SARSA** est généralement très performant, particulièrement dans les environnements complexes (Env1 et Env3).
2. **SARSA** et **Q-Learning** montrent également de bonnes performances dans la plupart des environnements.
3. **Dyna-Q** est particulièrement efficace dans l'environnement Env2, suggérant que sa capacité à combiner apprentissage direct et planification est avantageuse dans certains contextes.
4. **Dyna-Q+** montre des performances variables selon les environnements, parfois supérieures à Dyna-Q (Env0), parfois inférieures (Env2).
5. Les méthodes **Monte Carlo** (MC On-Policy et MC Exploring Starts) sont généralement moins performantes que les méthodes TD dans ces environnements, avec MC Exploring Starts montrant des performances particulièrement faibles.

Ces résultats soulignent l'importance de choisir l'algorithme approprié en fonction des caractéristiques de l'environnement et des objectifs spécifiques.

## Reproduction des expérimentations

Pour reproduire ces expérimentations, vous pouvez exécuter le notebook `experimentation_secret_envs.ipynb` :

```bash
jupyter notebook experiments/experimentation_secret_envs.ipynb
```

Les résultats seront sauvegardés dans le répertoire `experiments/outputs/` et peuvent être analysés à l'aide des outils de visualisation fournis dans le module `reinfolearn.utils.plot_utils`.