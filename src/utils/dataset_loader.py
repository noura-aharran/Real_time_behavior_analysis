# Fichier : src/utils/dataset_loader.py
import os
import cv2
from .preprocessing import normalize_position, create_sequences
import numpy as np
# Dimensions d'image par défaut pour la simulation
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

def load_and_preprocess_training_data(videos_dir, sequence_length):
    """
    Simule le chargement et le prétraitement des données d'entraînement 
    à partir des vidéos brutes (trajectoires normales).
    
    :param videos_dir: Chemin vers le dossier des vidéos (data/videos/).
    :param sequence_length: Longueur de séquence requise pour le modèle.
    :return: Un array NumPy de séquences 3D (num_sequences, sequence_length, 2).
    """
    print(f"Chargement des données pour l'entraînement depuis {videos_dir}...")
    
    all_sequences = []
    
    # --- SIMULATION de l'extraction des trajectoires ---
    # En production, vous utiliseriez ici ObjectTracker pour extraire 
    # les trajectoires (positions x,y) des personnes dans chaque vidéo.
    
    # Simuler une trajectoire longue et normale (200 points)
    num_frames = 200
    simulated_trajectory = [
        normalize_position((i, i, i+50, i+100), DEFAULT_WIDTH, DEFAULT_HEIGHT) 
        for i in range(1, num_frames + 1)
    ]

    # Créer les séquences glissantes à partir de la trajectoire simulée
    sequences = create_sequences(simulated_trajectory, sequence_length)
    all_sequences.extend(sequences)
    
    if not all_sequences:
        print("ATTENTION: Aucune séquence d'entraînement générée.")
        return np.empty((0, sequence_length, 2))

    # Convertir en tableau NumPy (format requis par PyTorch/TensorFlow)
    data_array = np.array(all_sequences, dtype=np.float32)
    print(f"Données d'entraînement prêtes : {data_array.shape[0]} séquences de {sequence_length} frames.")
    
    return data_array