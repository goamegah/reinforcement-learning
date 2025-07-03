
# rlearn/environments/rock_paper_scissors.py
import numpy as np
from reinfo.environments.base_environment import BaseEnvironment

class RockPaperScissorsMDP(BaseEnvironment):
    def __init__(self):
        self.actions = ["rock", "paper", "scissors"]
        self.player_action = None
        self.opponent_action = None
        self._score = 0.0
        self._is_done = False

    def num_states(self) -> int:
        return 1  # toujours le même état

    def num_actions(self) -> int:
        return 3  # rock, paper, scissors

    def num_rewards(self) -> int:
        return 3  # -1, 0, +1

    def reward(self, index: int) -> float:
        return [-1.0, 0.0, 1.0][index]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s != 0 or s_p != 0:
            return 0.0
        for opp in range(3):
            result = self._resolve(a, opp)
            expected_r = {1: 2, 0: 1, -1: 0}[result]
            if expected_r == r_index:
                return 1.0 / 3
        return 0.0

    def _resolve(self, a1: int, a2: int) -> int:
        """Compare les actions : retourne +1 (gagné), -1 (perdu), 0 (égal)"""
        if a1 == a2: return 0
        if (a1 - a2) % 3 == 1:
            return 1  # a1 gagne
        else:
            return -1  # a1 perd

    def state_id(self) -> int:
        return 0

    def reset(self) -> None:
        self.player_action = None
        self.opponent_action = None
        self._is_done = False
        self._score = 0.0

    def is_forbidden(self, action: int) -> int:
        return 0 if 0 <= action <= 2 else 1

    def is_game_over(self) -> bool:
        return self._is_done

    def available_actions(self) -> np.ndarray:
        return np.array([0, 1, 2])

    def step(self, action: int) -> None:
        if self._is_done:
            return

        self.player_action = action
        self.opponent_action = np.random.choice([0, 1, 2])
        result = self._resolve(self.player_action, self.opponent_action)
        self._score += result
        self._is_done = True

    def display(self) -> None:
        from reinfo.display.display_rps import render_rps_round
        render_rps_round(self.player_action, self.opponent_action, self._score)

    def score(self) -> float:
        return self._score
    

if __name__ == "__main__":
    env = RockPaperScissorsMDP()
    env.reset()
    env.display()
    action = int(input("Ton choix (0=Rock, 1=Paper, 2=Scissors) ? "))
    env.step(action)
    env.display()

