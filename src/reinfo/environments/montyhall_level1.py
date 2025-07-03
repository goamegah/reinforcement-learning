
# rlearn/environments/montyhall_level1.py
import numpy as np
from reinfo.environments.base_environment import BaseEnvironment

class MontyHallLevel1MDP(BaseEnvironment):
    def __init__(self):
        self.doors = [0, 1, 2]
        self.winning_door = None
        self.first_choice = None
        self.revealed_door = None
        self.second_choice = None
        self._is_done = False
        self._score = 0.0
        self.phase = 0  # 0 = choix initial, 1 = switch or keep

    def num_actions(self) -> int:
        return 3 if self.phase == 0 else 2  # choix initial (0-2), puis 0: garder, 1: changer

    def num_rewards(self) -> int:
        return 2  # 0.0 ou 1.0

    def reward(self, index: int) -> float:
        return [0.0, 1.0][index]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return 0.0  # non modélisable directement en tabulaire

    def num_states(self) -> int:
        return 6

    def state_id(self) -> int:
        return self.phase * 3 + (self.first_choice if self.first_choice is not None else 0)


    def reset(self) -> None:
        self.winning_door = np.random.choice(self.doors)
        self.first_choice = None
        self.revealed_door = None
        self.second_choice = None
        self.phase = 0
        self._is_done = False
        self._score = 0.0

    def is_forbidden(self, action: int) -> int:
        if self.phase == 0:
            return 0 if action in self.doors else 1
        else:
            return 0 if action in [0, 1] else 1

    def is_game_over(self) -> bool:
        return self._is_done

    def available_actions(self) -> np.ndarray:
        return np.array([a for a in range(self.num_actions()) if not self.is_forbidden(a)])

    def step(self, action: int) -> None:
        if self._is_done:
            return

        if self.phase == 0:
            self.first_choice = action
            # Révéler une porte qui n'est ni choisie ni gagnante
            remaining = [d for d in self.doors if d != self.first_choice and d != self.winning_door]
            self.revealed_door = np.random.choice(remaining)
            self.phase = 1
        else:
            if action == 0:
                final_choice = self.first_choice
            else:
                # Porte restante différente de first_choice et revealed
                final_choice = next(d for d in self.doors if d != self.first_choice and d != self.revealed_door)
            self.second_choice = final_choice
            self._score = 1.0 if self.second_choice == self.winning_door else 0.0
            self._is_done = True

    def display(self) -> None:
        from reinfo.display.display_monty import render_montyhall_round
        render_montyhall_round(self.phase, self.first_choice, self.revealed_door, self.second_choice, self._score, self._is_done)

    def score(self) -> float:
        return self._score


if __name__ == "__main__":
    env = MontyHallLevel1MDP()
    env.reset()

    while not env.is_game_over():
        env.display()
        action = int(input(f"[INST] Action ? ({env.available_actions().tolist()}) : "))
        env.step(action)

    env.display()
