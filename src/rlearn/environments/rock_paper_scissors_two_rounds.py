
# rlearn/environments/rock_paper_scissors_two_rounds.py
import numpy as np
from rlearn.environments.base_environment import BaseEnvironment

class TwoRoundRPSMDP(BaseEnvironment):
    def __init__(self):
        self.actions = ["rock", "paper", "scissors"]
        self.round = 0
        self.player_moves = []
        self.opponent_moves = []
        self._score = 0.0
        self._is_done = False

    def num_states(self) -> int:
        """
        6 états possibles :
        - Round 0 → état unique (0)
        - Round 1 → dépend du 1er coup de l'agent (3 possibilités)
        ⇒ 2 x 3 = 6
        """
        return 6

    def state_id(self) -> int:
        """
        Encodage : round * 3 + coup initial (si connu)
        - Round 0 → état 0
        - Round 1 → état ∈ {3, 4, 5}
        """
        if self.round == 0:
            return 0
        first_move = self.player_moves[0] if len(self.player_moves) > 0 else 0
        return self.round * 3 + first_move

    def num_actions(self) -> int:
        return 3  # rock, paper, scissors

    def num_rewards(self) -> int:
        return 3  # -1, 0, +1

    def reward(self, index: int) -> float:
        return [-1.0, 0.0, 1.0][index]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return 0.0  # non déterministe, dépendant de la stratégie adverse

    def _resolve(self, player: int, opponent: int) -> int:
        if player == opponent:
            return 0
        elif (player - opponent) % 3 == 1:
            return 1
        else:
            return -1

    def reset(self) -> None:
        self.round = 0
        self.player_moves = []
        self.opponent_moves = []
        self._score = 0.0
        self._is_done = False

    def is_forbidden(self, action: int) -> int:
        return 0 if action in [0, 1, 2] else 1

    def is_game_over(self) -> bool:
        return self._is_done

    def available_actions(self) -> np.ndarray:
        return np.array([0, 1, 2])

    def step(self, action: int) -> None:
        if self._is_done:
            return

        self.player_moves.append(action)

        if self.round == 0:
            opponent_action = np.random.choice([0, 1, 2])
        else:
            opponent_action = self.player_moves[0]  # copie du premier coup de l’agent

        self.opponent_moves.append(opponent_action)
        result = self._resolve(action, opponent_action)
        self._score += result
        self.round += 1

        if self.round >= 2:
            self._is_done = True

    def display(self) -> None:
        from rlearn.display.display_rps import render_rps_sequence
        render_rps_sequence(self.player_moves, self.opponent_moves, self._score)

    def score(self) -> float:
        return self._score


if __name__ == "__main__":
    env = TwoRoundRPSMDP()
    env.reset()

    while not env.is_game_over():
        env.display()
        try:
            action = int(input("[INST] Ton choix (0=Rock, 1=Paper, 2=Scissors) : "))
        except ValueError:
            print("/!\ Entrée invalide.")
            continue
        if env.is_forbidden(action):
            print("/!\ Action interdite.")
            continue
        env.step(action)

    env.display()
