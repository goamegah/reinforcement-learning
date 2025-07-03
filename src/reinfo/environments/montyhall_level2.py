
# rlearn/environments/montyhall_level2.py
import numpy as np
from reinfo.environments.base_environment import BaseEnvironment

class MontyHallLevel2MDP(BaseEnvironment):
    def __init__(self, n_doors=5):
        assert n_doors >= 3, "Il faut au moins 3 portes."
        self.n_doors = n_doors
        self.doors = list(range(n_doors))
        self.reset()

    def num_states(self) -> int:
        return (self.n_doors - 2 + 1) * self.n_doors  # +1 pour l'étape finale (switch/keep)

    def state_id(self) -> int:
        if not self.choice_history:
            return 0
        return self.step_count * self.n_doors + self.choice_history[-1]

    def num_actions(self) -> int:
        return self.n_doors if self.step_count < self.n_doors - 2 else 2

    def num_rewards(self) -> int:
        return 2

    def reward(self, index: int) -> float:
        return [0.0, 1.0][index]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return 0.0

    def reset(self) -> None:
        self.winning_door = np.random.choice(self.doors)
        self.remaining_doors = self.doors.copy()
        self.choice_history = []
        self._score = 0.0
        self._is_done = False
        self.step_count = 0
        self.final_choice = None

    def is_forbidden(self, action: int) -> int:
        if self.step_count < self.n_doors - 2:
            return 0 if action in self.remaining_doors else 1
        else:
            return 0 if action in [0, 1] else 1

    def is_game_over(self) -> bool:
        return self._is_done

    def available_actions(self) -> np.ndarray:
        if self.step_count < self.n_doors - 2:
            return np.array(self.remaining_doors)
        else:
            return np.array([0, 1])  # 0 = garder, 1 = switch

    def step(self, action: int) -> None:
        if self._is_done:
            return

        if self.step_count < self.n_doors - 2:
            # Phase de filtrage : garder une porte
            self.choice_history.append(action)

            # Révèle et supprime une perdante ≠ gagnante ≠ choix
            candidates = [d for d in self.remaining_doors if d != action and d != self.winning_door]
            if candidates:
                to_remove = np.random.choice(candidates)
                self.remaining_doors.remove(to_remove)

            self.step_count += 1

        else:
            # Phase finale : switch or keep
            last_choice = self.choice_history[-1]
            other_door = [d for d in self.remaining_doors if d != last_choice][0]
            self.final_choice = last_choice if action == 0 else other_door

            self._score = 1.0 if self.final_choice == self.winning_door else 0.0
            self._is_done = True

    def display(self) -> None:
        from reinfo.display.display_monty import render_montyhall_level2_final
        render_montyhall_level2_final(
            remaining_doors=self.remaining_doors,
            choice_history=self.choice_history,
            step_count=self.step_count,
            score=self._score,
            done=self._is_done,
            winning_door=self.winning_door if self._is_done else None,
            final_choice=self.final_choice,
            n_doors=self.n_doors
        )

    def score(self) -> float:
        return self._score
    

if __name__ == "__main__":
    env = MontyHallLevel2MDP(n_doors=5)
    env.reset()

    while not env.is_game_over():
        env.display()
        try:
            action = int(input("Votre choix : "))
        except ValueError:
            continue
        if env.is_forbidden(action):
            print("/!\ Action interdite.")
            continue
        env.step(action)

    env.display()

