# # reinfolearn/display/display_grid.py

def render_grid_world(agent_pos, size, terminal_states, done=False, score=0.0):
    """
    Affiche la grille de GridWorld dans la console.

    Args:
        agent_pos (tuple): position actuelle de l'agent (row, col)
        size (int): taille de la grille (size x size)
        terminal_states (dict): {(row, col): reward}
        done (bool): si l'épisode est terminé
        score (float): score final (si terminé)
    """
    print("\n=== GridWorld ===")

    for row in range(size):
        line = ""
        for col in range(size):
            cell = "."
            if (row, col) == agent_pos:
                cell = " A "
            elif (row, col) in terminal_states:
                cell = "T+" if terminal_states[(row, col)] > 0 else "T-"
            else:
                cell = " . "
            line += cell
        print(line)

    if done:
        print(f"\n@@ Épisode terminé. Score final : {score}")
