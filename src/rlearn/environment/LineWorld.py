""" LineWorld Environment
This environment simulates a simple line world where an agent can move forward or backward.
The agent starts at the beginning of the line and can move to the end.
The goal is to reach the end of the line, represented by an 'X'.
The environment provides methods to reset the state, display the current state,
step through actions, check if the game is over, and calculate the score.
The environment also implements methods for MDP (Markov Decision Process) related functionalities,
including the number of states, actions, rewards, and transition probabilities.
"""
# rlearn/environment/LineWorld.py
import numpy as np


class LineWorld:
    def __init__(self):
        self.position = 0
        self.size = 5

    def reset(self):
        self.position = 0

    def display(self):
        line = ['_' for _ in range(self.size)]
        line[self.position] = 'X'
        print(" ".join(line))

    def step(self, action):
        if action == 0 and self.position < self.size - 1:  # forward
            self.position += 1
        elif action == 1 and self.position > 0:  # backward
            self.position -= 1

    def is_game_over(self):
        return self.position == self.size - 1

    def score(self):
        return 1 if self.position == self.size - 1 else 0

    def available_actions(self):
        actions = []
        if self.position > 0:
            actions.append(1)  # backward
        if self.position < self.size - 1:
            actions.append(0)  # forward
        return np.array(actions)

    def state_id(self):
        return self.position

    def is_forbidden(self, action):
        return action not in self.available_actions()

    # MDP related methods
    def num_states(self):
        return self.size

    def num_actions(self):
        return 2  # 'forward' and 'backward'

    def num_rewards(self):
        return self.num_states()  # One reward per state

    def reward(self, i):
        return 1 if i == self.size - 1 else 0  # Reward only at the terminal state

    def p(self, s, a, s_p, r_index):
        if r_index != s_p:
            return 0.0  # Rewards index should match the state
        if a == 0 and s_p == s + 1 and s < self.size - 1:  # forward
            return 1.0
        elif a == 1 and s_p == s - 1 and s > 0:  # backward
            return 1.0
        return 0.0
