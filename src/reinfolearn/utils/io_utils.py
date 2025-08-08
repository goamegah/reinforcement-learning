
# rlearn/utils/io_utils.py
import os
import json
import pickle
import numpy as np

# === SCORES ===

def save_scores(scores, path):
    """
    Sauvegarde un tableau de scores (list ou np.array) au format .npy
    """
    np.save(path, np.array(scores))


def load_scores(path):
    """
    Charge un tableau de scores depuis un fichier .npy
    """
    return np.load(path)


# === POLICY ===

def save_policy(policy, path):
    """
    Sauvegarde une politique (tableau numpy) au format JSON
    """
    with open(path, "w") as f:
        json.dump(policy.tolist(), f)


def load_policy(path):
    """
    Charge une politique depuis un fichier JSON
    """
    with open(path, "r") as f:
        return np.array(json.load(f))

# === Q-TABLE ===

def save_q_table(Q, path):
    """
    Sauvegarde une Q-table (dictionnaire d'état -> actions) au format pickle
    Supporte également les tableaux numpy et autres types de données
    """
    with open(path, "wb") as f:
        if hasattr(Q, 'items') or isinstance(Q, dict):
            # Si Q est un dictionnaire ou un objet dict-like
            pickle.dump(dict(Q), f)
        else:
            # Si Q est un autre type (numpy array, etc.)
            pickle.dump(Q, f)


def load_q_table(path):
    """
    Charge une Q-table depuis un fichier pickle
    """
    with open(path, "rb") as f:
        return pickle.load(f)
    
# === VALUES ===

def save_values(V, path):
    """
    Sauvegarde un vecteur de valeurs V (np.array) au format .npy
    """
    np.save(path, np.array(V))


def load_values(path):
    """
    Charge un vecteur V depuis un fichier .npy
    """
    return np.load(path)
