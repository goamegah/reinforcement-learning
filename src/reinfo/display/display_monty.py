def render_montyhall_round(phase, first_choice, revealed_door, second_choice, score, done):
    doors = ['A', 'B', 'C']
    print("\n=== Monty Hall Level 1 ===")

    if phase == 0:
        print("[INST] Phase 1 : Choisissez une porte (0=A, 1=B, 2=C)")
        if first_choice is not None:
            print(f"🚪 Vous avez choisi : Porte {doors[first_choice]}")
    else:
        print("[INST] Phase 2 : Voulez-vous garder (0) ou changer (1) ?")
        print(f"*!* Porte révélée (vide) : {doors[revealed_door]}")
        print(f"*!* Votre choix initial : {doors[first_choice]}")

    if done:
        print(f"\n[OK] Porte finale choisie : {doors[second_choice]}")
        print(f"@@@ {'GAGNÉ' if score == 1.0 else 'PERDU'} (score = {score})")


def render_montyhall_level2_final(
    remaining_doors,
    choice_history,
    step_count,
    score,
    done,
    winning_door,
    final_choice,
    n_doors
):
    print("\n=== Monty Hall Level 2 ===")

    if step_count < n_doors - 2:
        print(f"Étape {step_count + 1}/{n_doors - 2} — Choix à conserver")
        print(f"*!* Portes restantes : {remaining_doors}")
        if choice_history:
            print(f"*!* Vos choix précédents : {choice_history}")
        print("[INST] Choisissez une porte à conserver.")
    elif not done:
        print(f"\nDernière étape : SWITCH (1) ou KEEP (0)")
        print(f"*!* Portes restantes : {remaining_doors}")
        print(f"*!* Dernier choix conservé : {choice_history[-1]}")
        print("[INST] Tapez 0 pour garder, 1 pour changer.")
    else:
        print(f"\n🏁 Partie terminée.")
        print(f"*!* Portes finales      : {remaining_doors}")
        print(f"*!* Porte gagnante      : {winning_door}")
        print(f"*!* Votre choix final   : {final_choice}")
        print(f"@@@ {'GAGNÉ' if score == 1.0 else 'PERDU'} (score = {score})")






