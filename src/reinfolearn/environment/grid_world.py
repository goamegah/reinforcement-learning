# rlearn/environments/grid_world.py
import numpy as np
from reinfolearn.environment.base_environment import BaseEnvironment

class GridWorldMDP(BaseEnvironment):
    def __init__(self, size: int = 5):
        self.size = size
        self.num_states_ = size * size
        self.current_pos = (0, 0)
        self.terminal_states = {(0, size - 1): -3.0, (size - 1, size - 1): 1.0}
        self._is_done = False
        self._score = 0.0

    def _pos_to_id(self, row: int, col: int) -> int:
        return row * self.size + col

    def _id_to_pos(self, state_id: int) -> tuple:
        return divmod(state_id, self.size)

    def num_states(self) -> int:
        return self.num_states_

    def num_actions(self) -> int:
        return 4  # haut, bas, gauche, droite

    def num_rewards(self) -> int:
        return 3  # -3, 0, +1

    def reward(self, index: int) -> float:
        return [-3.0, 0.0, 1.0][index]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s >= self.num_states():
            return 0.0

        row, col = self._id_to_pos(s)

        if (row, col) in self.terminal_states:
            return 0.0

        next_row, next_col = row, col
        if a == 0 and row > 0: next_row -= 1      # haut
        elif a == 1 and row < self.size - 1: next_row += 1  # bas
        elif a == 2 and col > 0: next_col -= 1    # gauche
        elif a == 3 and col < self.size - 1: next_col += 1  # droite

        target_id = self._pos_to_id(next_row, next_col)

        if s_p != target_id:
            return 0.0

        if (next_row, next_col) in self.terminal_states:
            return float(r_index == (0 if self.terminal_states[(next_row, next_col)] == -3.0 else 2))
        else:
            return float(r_index == 1)  # reward 0

    def state_id(self) -> int:
        return self._pos_to_id(*self.current_pos)

    def reset(self) -> None:
        self.current_pos = (0, 0)
        self._is_done = False
        self._score = 0.0

    def is_forbidden(self, action: int) -> int:
        row, col = self.current_pos
        if action == 0 and row == 0: return 1
        if action == 1 and row == self.size - 1: return 1
        if action == 2 and col == 0: return 1
        if action == 3 and col == self.size - 1: return 1
        return 0

    def is_game_over(self) -> bool:
        return self._is_done

    def available_actions(self) -> np.ndarray:
        return np.array([a for a in range(self.num_actions()) if not self.is_forbidden(a)])

    def step(self, action: int) -> None:
        if self._is_done:
            return

        row, col = self.current_pos

        next_row, next_col = row, col
        if action == 0 and row > 0: next_row -= 1
        elif action == 1 and row < self.size - 1: next_row += 1
        elif action == 2 and col > 0: next_col -= 1
        elif action == 3 and col < self.size - 1: next_col += 1

        self.current_pos = (next_row, next_col)

        if self.current_pos in self.terminal_states:
            self._score += self.terminal_states[self.current_pos]
            self._is_done = True

    def display(self) -> None:
        from reinfolearn.display.display_grid import render_grid_world
        render_grid_world(self.current_pos, self.size, self.terminal_states, self._is_done, self._score)

    def score(self) -> float:
        return self._score
    
    
if __name__ == "__main__":
    env = GridWorldMDP()
    env.reset()
    while not env.is_game_over():
        env.display()
        action = int(input("[INST] Action (0=haut, 1=bas, 2=gauche, 3=droite) ? "))
        env.step(action)
    env.display()
    print(f"$$ Score final : {env.score()}")

