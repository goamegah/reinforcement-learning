import pytest
from rlearn.environments.montyhall_level1 import MontyHallLevel1


def test_reset_and_initial_state():
    env = MontyHallLevel1()
    state = env.reset()
    assert isinstance(state, int)
    assert state == env.state_id()
    assert env.index_to_state[state] == 'initial'


def test_first_step_transitions():
    env = MontyHallLevel1()
    env.reset()
    state1, reward, done = env.step(0)  # Choix de la porte A (index 0)
    assert reward == 0.0
    assert not done
    assert env.index_to_state[state1].startswith("first_")
    assert env.revealed_door is not None


def test_second_step_outcome():
    env = MontyHallLevel1()
    env.reset()
    env.step(0)  # Choix initial
    state2, reward, done = env.step(1)  # Switch
    assert done
    assert reward in [0.0, 1.0]
    assert env.index_to_state[state2].startswith("final_")


def test_available_actions():
    env = MontyHallLevel1()
    env.reset()
    assert env.available_actions() == [0, 1, 2]
    env.step(0)
    assert env.available_actions() == [0, 1]


def test_terminal_state_behavior():
    env = MontyHallLevel1()
    env.reset()
    env.step(0)
    env.step(1)
    assert env.is_game_over()
    assert env.available_actions() == []
