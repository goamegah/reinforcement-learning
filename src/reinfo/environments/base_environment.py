from abc import ABC, abstractmethod
import numpy as np

class BaseEnvironment(ABC):
    """
    Interface unique compatible avec les environnements secrets du professeur.
    Tous les environnements doivent implémenter ces méthodes.
    """

    # --- MDP related Methods ---
    @abstractmethod
    def num_states(self) -> int:
        """Retourne le nombre total d'états dans l'environnement."""
        pass

    @abstractmethod
    def num_actions(self) -> int:
        """Retourne le nombre total d'actions possibles."""
        pass

    @abstractmethod
    def num_rewards(self) -> int:
        """Retourne le nombre total de récompenses distinctes."""
        pass

    @abstractmethod
    def reward(self, index: int) -> float:
        """Retourne la récompense associée à un index donné."""
        pass

    @abstractmethod
    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        """Retourne la probabilité de transition P(s', r | s, a)."""
        pass

    # --- Monte Carlo and TD Methods ---
    @abstractmethod
    def state_id(self) -> int:
        """Retourne l'identifiant unique de l'état courant."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Réinitialise l'environnement."""
        pass

    @abstractmethod
    def display(self) -> None:
        """Affiche l'environnement à l'état courant (console ou GUI)."""
        pass

    @abstractmethod
    def is_forbidden(self, action: int) -> int:
        """
        Retourne 1 si l'action est interdite, 0 sinon.
        Utilisé pour filtrer les actions valides avant `step`.
        """
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """Retourne True si l'épisode est terminé."""
        pass

    @abstractmethod
    def available_actions(self) -> np.ndarray:
        """Retourne un tableau numpy des actions valides à cet instant."""
        pass

    @abstractmethod
    def step(self, action: int) -> None:
        """Exécute l’action (aucun retour immédiat)."""
        pass

    @abstractmethod
    def score(self) -> float:
        """Retourne le score global de l'épisode (utile à la fin du jeu)."""
        pass
