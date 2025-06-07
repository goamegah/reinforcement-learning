import os
import sys
import time
from rlearn.environments.line_world import LineWorldMDP

def clear_console():
    """Nettoie la console selon l’OS."""
    os.system('cls' if os.name == 'nt' else 'clear')

def render_line_world(pos, size, is_done, score):
    print("\n=== LineWorld ===")
    line = [" "] * size
    line[pos] = "A"  # A = Agent
    print("".join(f"[{c}]" for c in line))
    if is_done:
        print(f"🏁 Épisode terminé - Score : {score}")


def interactive_line_world():
    """Permet à l'utilisateur de jouer à LineWorld via le terminal."""
    env = LineWorldMDP(size=7)
    env.reset()

    while not env.is_game_over():
        clear_console()
        render_line_world(env.state_id(), env.size)
        print("\nUtilise les touches : ← (0) pour GAUCHE, → (1) pour DROITE")
        action = input("Action ? (0/1) : ").strip()

        if action not in ["0", "1"]:
            print("❌ Action invalide. Choisis 0 (←) ou 1 (→).")
            time.sleep(1)
            continue

        action = int(action)
        if env.is_forbidden(action):
            print("⚠️ Action interdite dans cette direction.")
            time.sleep(1)
            continue

        env.step(action)

    clear_console()
    render_line_world(env.state_id(), env.size, done=True, score=env.score())
