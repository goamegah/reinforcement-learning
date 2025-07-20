import numpy as np
from reinfolearn.environment.base_environment import BaseEnvironment
import random

class TwoRoundRPS(BaseEnvironment):
    def __init__(self):
        self.actions = [0, 1, 2]  # 0 = Rock, 1 = Paper, 2 = Scissors
        self.action_meanings = {0: "Rock", 1: "Paper", 2: "Scissors"}
        self.rewards_list = [-1, 0, 1]
        self.reset()

    def reset(self) -> None:
        self.round = 1
        self.agent_actions = []
        self.opponent_actions = []
        self._score = 0
        self.done = False

    def num_states(self) -> int:
        return 10

    def num_actions(self) -> int:
        return 3

    def num_rewards(self) -> int:
        return 3

    def reward(self, index: int) -> float:
        return self.rewards_list[index]

    def p(self, s, a, s_p, r_index):
        if s == 0:
            prob = 0
            for opp_a1 in range(3):
                expected_s_p = 1 + 3 * a + opp_a1
                if s_p == expected_s_p:
                    r = self._get_result(a, opp_a1)
                    idx = self.rewards_list.index(r)
                    if idx == r_index:
                        prob += 1 / 3
            return prob
        elif 1 <= s <= 9:
            agent_a1 = (s - 1) // 3
            opp_a1 = (s - 1) % 3
            opp_a2 = agent_a1
            r = self._get_result(a, opp_a2)
            idx = self.rewards_list.index(r)
            if s_p == s and r_index == idx:
                return 1.0
        return 0.0

    def state_id(self) -> int:
        if self.round == 1:
            return 0
        else:
            agent_a1 = self.agent_actions[0]
            opp_a1 = self.opponent_actions[0]
            return 1 + 3 * agent_a1 + opp_a1

    def display(self) -> None:
        print(f"\nRound: {self.round}")
        agent_display = [self.action_meanings[a] for a in self.agent_actions]
        opp_display = [self.action_meanings[a] for a in self.opponent_actions]
        print(f"Agent: {agent_display}")
        print(f"Opponent: {opp_display}")
        print(f"Score: {self._score}\n")

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
            opp_action = self.agent_actions[0]
            self.agent_actions.append(action)
            self.opponent_actions.append(opp_action)
            r = self._get_result(action, opp_action)
            self._score += r
            self.done = True

    def _get_result(self, a1: int, a2: int) -> int:
        if a1 == a2:
            return 0
        if (a1 == 0 and a2 == 2) or (a1 == 1 and a2 == 0) or (a1 == 2 and a2 == 1):
            return 1
        return -1

    def score(self) -> float:
        return self._score


# === INTERACTIVE TEST ===
if __name__ == "__main__":
    env = TwoRoundRPS()
    NUM_EPISODES = 1

    for ep in range(NUM_EPISODES):
        print(f"\n=== Épisode {ep + 1} ===")
        env.reset()
        while not env.is_game_over():
            state = env.state_id()
            actions = env.available_actions()
            print(f"État courant : {state}")
            while True:
                try:
                    print("Actions possibles :")
                    for a in actions:
                        print(f"  {a} = {env.action_meanings[a]}")
                    action = int(input("Choisissez votre action (0/1/2) : "))
                    if env.is_forbidden(action):
                        print("Action invalide. Essayez encore.")
                        continue
                    break
                except ValueError:
                    print("Entrée invalide. Entrez un nombre entre 0 et 2.")
            env.step(action)
            env.display()

        print(f"Score final : {env.score()}")
        print("-" * 40)