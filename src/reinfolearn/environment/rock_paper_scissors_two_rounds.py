import numpy as np
from reinfolearn.environment.base_environment import BaseEnvironment
import random

class TwoRoundRPS(BaseEnvironment):
    def __init__(self):
        self.actions = [0, 1, 2]  # 0 = Rock, 1 = Paper, 2 = Scissors
        self.rewards_list = [-1, 0, 1]  # index 0 = -1, 1 = 0, 2 = +1
        self.reset()

    def reset(self) -> None:
        self.round = 1
        self.agent_actions = []
        self.opponent_actions = []
        self._score = 0
        self.done = False

    def num_states(self) -> int:
        return 10  # 1 état initial + 9 combinaisons possibles en round 2

    def num_actions(self) -> int:
        return 3  # Rock, Paper, Scissors

    def num_rewards(self) -> int:
        return 3  # -1, 0, +1

    def reward(self, index: int) -> float:
        return self.rewards_list[index]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        """
        Définir dynamiquement la probabilité d'une transition :
        - au round 1, adversaire joue aléatoirement
        - au round 2, adversaire copie l’action de l’agent au round 1
        """
        if self.round == 1:
            prob = 1 / 3 if 0 <= s_p - 1 < 9 else 0.0  # s_p = 1 to 9 pour round 2
        else:
            prob = 1.0 if s_p == 9 else 0.0  # terminal state (fin du jeu)
        return prob

    def state_id(self) -> int:
        if self.round == 1:
            return 0  # état initial
        else:
            agent_a1 = self.agent_actions[0]
            opp_a1 = self.opponent_actions[0]
            return 1 + 3 * agent_a1 + opp_a1  # 1 à 9

    def display(self) -> None:
        print(f"\n🧩 Round: {self.round}")
        print(f"👤 Agent: {self.agent_actions}")
        print(f"🤖 Opponent: {self.opponent_actions}")
        print(f"🎯 Score: {self._score}\n")

    def is_forbidden(self, action: int) -> int:
        return 0 if action in self.actions else 1

    def is_game_over(self) -> bool:
        return self.done

    def available_actions(self) -> np.ndarray:
        return np.array(self.actions)

    def step(self, action: int) -> None:
        if self.done:
            return

        if self.round == 1:
            opp_action = random.choice(self.actions)
            self.agent_actions.append(action)
            self.opponent_actions.append(opp_action)
            r = self._get_result(action, opp_action)
            self._score += r
            self.round += 1
        elif self.round == 2:
            # Opponent copie l'action du joueur au round 1
            opp_action = self.agent_actions[0]
            self.agent_actions.append(action)
            self.opponent_actions.append(opp_action)
            r = self._get_result(action, opp_action)
            self._score += r
            self.done = True

    def _get_result(self, a1: int, a2: int) -> int:
        # retourne -1 si perdu, 0 si égalité, 1 si gagné
        if a1 == a2:
            return 0
        if (a1 == 0 and a2 == 2) or (a1 == 1 and a2 == 0) or (a1 == 2 and a2 == 1):
            return 1
        return -1

    def score(self) -> float:
        return self._score


# Example usage
if __name__ == "__main__":
    env = TwoRoundRPS()
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
    print(f"$$ Score final : {env.score()}")
# End of code completion
# Note: The display function assumes a console output. Adjust as needed for GUI or other interfaces
# Note: The example usage at the end allows for manual testing of the environment.















# # rlearn/environments/rock_paper_scissors_two_rounds.py
# import numpy as np
# from reinfolearn.environment.base_environment import BaseEnvironment

# class TwoRoundRPSMDP(BaseEnvironment):
#     def __init__(self):
#         self.actions = ["rock", "paper", "scissors"]
#         self.round = 0
#         self.player_moves = []
#         self.opponent_moves = []
#         self._score = 0.0
#         self._is_done = False

#     def num_states(self) -> int:
#         """
#         6 états possibles :
#         - Round 0 → état unique (0)
#         - Round 1 → dépend du 1er coup de l'agent (3 possibilités)
#         ⇒ 2 x 3 = 6
#         """
#         return 6

#     def state_id(self) -> int:
#         """
#         Encodage : round * 3 + coup initial (si connu)
#         - Round 0 → état 0
#         - Round 1 → état ∈ {3, 4, 5}
#         """
#         if self.round == 0:
#             return 0
#         first_move = self.player_moves[0] if len(self.player_moves) > 0 else 0
#         return self.round * 3 + first_move

#     def num_actions(self) -> int:
#         return 3  # rock, paper, scissors

#     def num_rewards(self) -> int:
#         return 3  # -1, 0, +1

#     def reward(self, index: int) -> float:
#         return [-1.0, 0.0, 1.0][index]

#     def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
#         return 0.0  # non déterministe, dépendant de la stratégie adverse

#     def _resolve(self, player: int, opponent: int) -> int:
#         if player == opponent:
#             return 0
#         elif (player - opponent) % 3 == 1:
#             return 1
#         else:
#             return -1

#     def reset(self) -> None:
#         self.round = 0
#         self.player_moves = []
#         self.opponent_moves = []
#         self._score = 0.0
#         self._is_done = False

#     def is_forbidden(self, action: int) -> int:
#         return 0 if action in [0, 1, 2] else 1

#     def is_game_over(self) -> bool:
#         return self._is_done

#     def available_actions(self) -> np.ndarray:
#         return np.array([0, 1, 2])

#     def step(self, action: int) -> None:
#         if self._is_done:
#             return

#         self.player_moves.append(action)

#         if self.round == 0:
#             opponent_action = np.random.choice([0, 1, 2])
#         else:
#             opponent_action = self.player_moves[0]  # copie du premier coup de l’agent

#         self.opponent_moves.append(opponent_action)
#         result = self._resolve(action, opponent_action)
#         self._score += result
#         self.round += 1

#         if self.round >= 2:
#             self._is_done = True

#     def display(self) -> None:
#         from reinfolearn.display.display_rps import render_rps_sequence
#         render_rps_sequence(self.player_moves, self.opponent_moves, self._score)

#     def score(self) -> float:
#         return self._score


# if __name__ == "__main__":
#     env = TwoRoundRPSMDP()
#     env.reset()

#     while not env.is_game_over():
#         env.display()
#         try:
#             action = int(input("[INST] Ton choix (0=Rock, 1=Paper, 2=Scissors) : "))
#         except ValueError:
#             print("/!\ Entrée invalide.")
#             continue
#         if env.is_forbidden(action):
#             print("/!\ Action interdite.")
#             continue
#         env.step(action)

#     env.display()
