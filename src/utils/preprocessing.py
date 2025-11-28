# Fichier : src/utils/preprocessing.py
import numpy as np

def normalize_position(box, width, height):
    """
    Calcule la position centrale d'une boîte et la normalise par les dimensions de l'image.

    :param box: Boîte englobante (x1, y1, x2, y2).
    :param width: Largeur de l'image.
    :param height: Hauteur de l'image.
    :return: Tuple (normalized_center_x, normalized_center_y).
    """
    x1, y1, x2, y2 = box
    
    # 1. Calcul du centre
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 2. Normalisation (ramener les valeurs entre 0 et 1)
    normalized_x = center_x / width
    normalized_y = center_y / height
    
    return (normalized_x, normalized_y)

def create_sequences(trajectory_data, sequence_length):
    """
    Crée des séquences temporelles de mouvement pour l'entraînement à partir 
    d'une longue trajectoire.

    :param trajectory_data: Liste de positions normalisées [(x1, y1), (x2, y2), ...].
    :param sequence_length: Taille de la fenêtre (N frames).
    :return: Liste de séquences prêtes pour le modèle.
    """
    sequences = []
    if len(trajectory_data) < sequence_length:
        return sequences

    # Fenêtre glissante
    for i in range(len(trajectory_data) - sequence_length + 1):
        sequence = trajectory_data[i:i + sequence_length]
        sequences.append(sequence)
        
    return sequences