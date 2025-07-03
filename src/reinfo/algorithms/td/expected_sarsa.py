import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def expected_sarsa(env, nb_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    Expected SARSA algorithm for reinforcement learning.
    This algorithm learns the action-value function Q(s, a) and derives a policy from it.
    :param env: Environment compatible with the expected SARSA algorithm
    :param nb_episodes: Number of episodes to simulate
    :param gamma: Discount factor
    :param alpha: Learning rate
    :param epsilon: Exploration rate for ε-greedy policy
    :return: policy, Q, scores
    """
    # Initialize Q(s, a) for all states and actions
    # Using defaultdict to handle states not seen yet
    Q = defaultdict(lambda: np.zeros(env.num_actions()))
    scores = []

    for _ in tqdm(range(nb_episodes), desc="Expected SARSA"):
        env.reset()
        state = env.state_id()

        while not env.is_game_over():
            actions = env.available_actions()
            action = np.random.choice(actions) if np.random.rand() < epsilon else np.argmax(Q[state])
            env.step(action)
            next_state = env.state_id()
            next_actions = env.available_actions()

            probs = np.ones(env.num_actions()) * (epsilon / env.num_actions())
            best_action = np.argmax(Q[next_state])
            probs[best_action] += (1.0 - epsilon)

            expected_value = np.dot(probs, Q[next_state])
            reward = env.score() if env.is_game_over() else 0
            td_target = reward + gamma * expected_value
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state

        scores.append(env.score())

    policy = np.zeros(env.num_states(), dtype=int)
    for s in range(env.num_states()):
        policy[s] = np.argmax(Q[s]) if s in Q else 0

    return policy, Q, scores


def plot_scores(scores, window=100, title="Expected SARSA - Score moyen"):
    moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(moving_avg)
    plt.title(title)
    plt.xlabel("Épisode")
    plt.ylabel("Score moyen")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
