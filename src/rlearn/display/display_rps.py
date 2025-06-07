def render_rps_round(player_action, opponent_action, score):
    actions = ["Rock", "Paper", "Scissors"]
    print("\n=== Rock Paper Scissors ===")

    if player_action is None:
        print("Prêt à jouer !")
        return

    print(f"🧠 Toi      : {actions[player_action]}")
    print(f"🤖 Adversaire : {actions[opponent_action]}")

    outcome = (player_action - opponent_action) % 3
    if outcome == 0:
        print("⚔️ Égalité !")
    elif outcome == 1:
        print("✅ Tu gagnes !")
    else:
        print("❌ Tu perds !")

    print(f"Score : {score}")


def render_rps_sequence(player_moves, opponent_moves, score):
    actions = ["Rock", "Paper", "Scissors"]
    print("\n=== Two-Round Rock Paper Scissors ===")

    for i, (p, o) in enumerate(zip(player_moves, opponent_moves)):
        print(f"\n🕹️ Round {i + 1}")
        print(f"Toi        : {actions[p]}")
        print(f"Adversaire : {actions[o]}")

        outcome = (p - o) % 3
        if outcome == 0:
            print("⚔️ Égalité")
        elif outcome == 1:
            print("✅ Gagné")
        else:
            print("❌ Perdu")

    print(f"\n🎯 Score total : {score}")

