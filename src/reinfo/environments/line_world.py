
# rlearn/environments/line_world.py
import numpy as np
from reinfo.environments.base_environment import BaseEnvironment

class LineWorldMDP(BaseEnvironment):
    def __init__(self, size: int = 5):
        assert size >= 3 and size % 2 == 1, "Size must be an odd integer >= 3"
        self.size = size
        self.start_pos = size // 2
        self.current_pos = self.start_pos
        self.terminal_left = 0
        self.terminal_right = size - 1
        self._is_done = False
        self._score = 0.0

    def __str__(self):
        return f"LineWorldMDP(size={self.size}, pos={self.current_pos}, score={self._score}, done={self._is_done})"

    def num_states(self) -> int:
        return self.size

    def num_actions(self) -> int:
        return 2  # 0: gauche, 1: droite

    def num_rewards(self) -> int:
        return 3  # -1, 0, +1

    def reward(self, index: int) -> float:
        return [-1.0, 0.0, 1.0][index]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s == self.terminal_left or s == self.terminal_right:
            return 0.0  # pas de transition après état terminal

        expected_s_p = s - 1 if a == 0 else s + 1
        if expected_s_p != s_p:
            return 0.0

        if s_p == self.terminal_left and r_index == 0:
            return 1.0
        elif s_p == self.terminal_right and r_index == 2:
            return 1.0
        elif 0 < s_p < self.size - 1 and r_index == 1:
            return 1.0
        return 0.0

    def state_id(self) -> int:
        return self.current_pos

    def reset(self) -> None:
        self.current_pos = self.start_pos
        self._is_done = False
        self._score = 0.0

    def display(self) -> None:
        from reinfo.display.display_line import render_line_world  # import local
        render_line_world(self.current_pos, self.size, self._is_done, self._score)


    def is_forbidden(self, action: int) -> int:
        if action == 0 and self.current_pos == self.terminal_left:
            return 1
        if action == 1 and self.current_pos == self.terminal_right:
            return 1
        if action not in [0, 1]:
            return 1
        return 0

    def is_game_over(self) -> bool:
        return self._is_done

    def available_actions(self) -> np.ndarray:
        return np.array([a for a in range(self.num_actions()) if not self.is_forbidden(a)])

    def step(self, action: int) -> None:
        if self._is_done or self.is_forbidden(action):
            return

        if action == 0:
            self.current_pos -= 1
        else:
            self.current_pos += 1

        if self.current_pos == self.terminal_left:
            self._score += -1.0
            self._is_done = True
        elif self.current_pos == self.terminal_right:
            self._score += 1.0
            self._is_done = True

    def score(self) -> float:
        return self._score

if __name__ == "__main__":
    env = LineWorldMDP()
    env.reset()

    while not env.is_game_over():
        env.display()
        try:
            action = int(input("Action (0=←, 1=→) : "))
        except ValueError:
            print("Entrée invalide.")
            continue
        if env.is_forbidden(action):
            print("/!\ Action interdite (mur)")
            continue
        env.step(action)

    env.display()

