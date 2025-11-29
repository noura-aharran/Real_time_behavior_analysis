import numpy as np
import os
import json # Pourrait être utile pour lire la structure de certains masques
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import torch

# --- 1. Importation des Composants du Pipeline ---
# Assurez-vous que ces chemins sont corrects
from src.components.anomaly_detector import AnomalyDetector
from src.components.tracker import ObjectTracker
from src.components.sequence_builder import SequenceBuilder

# --- 2. Hyperparamètres de Validation ---
# Ces paramètres doivent correspondre à ceux de votre entraînement/détection
SEQUENCE_LENGTH = 30 
MODEL_PATH = 'C:\\Users\\LENOVO\\Documents\\Real_time_behavior_analysis\\models\\anomaly_model.pt'
TEST_DATA_DIR = 'C:\\Users\\LENOVO\\Documents\\Real_time_behavior_analysis\\data\\validation\\frames'# Dossier contenant les frames de test
MASK_DATA_DIR = 'C:\\Users\\LENOVO\\Documents\\Real_time_behavior_analysis\\data\\validation\\test_frame_mask' # Dossier contenant les masques

def load_masks(mask_dir):
    """
    Charge les masques d'anomalie de la vérité terrain (ground truth).
    Chaque fichier de masque doit être un tableau binaire (0=normal, 1=anomalie).
    """
    all_masks = {}
    
    # Exemple de lecture simple : Assurez-vous que les noms des fichiers correspondent aux vidéos
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.npy')] 
    
    for filename in mask_files:
        video_id = filename.split('.')[0] # Ex: '01' si le fichier est '01.npy'
        # Assurez-vous que le format de stockage (json, npy, txt) correspond à la réalité
        try:
            # Ici, on suppose que le masque est stocké sous forme de tableau NumPy
            mask = np.load(os.path.join(mask_dir, filename))
            all_masks[video_id] = mask.flatten() # Masque binaire aplati [0, 0, 1, 1, 0...]
        except Exception as e:
            print(f"Erreur de chargement du masque pour {filename}: {e}")
            continue
            
    return all_masks

def run_inference_and_collect_scores(test_data_dir, detector, all_masks):
    """
    Exécute le pipeline sur les données de test et collecte les scores d'anomalie
    avec leurs labels réels (vérité terrain).
    """
    # Initialisation des composants du pipeline
    tracker = ObjectTracker()
    builder = SequenceBuilder(sequence_length=SEQUENCE_LENGTH)
    
    all_anomaly_scores = []
    all_true_labels = []
    
    video_dirs = sorted([d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))])

    for video_id in video_dirs:
        print(f"Analyse de la vidéo : {video_id}")
        video_path = os.path.join(test_data_dir, video_id)
        
        # Le masque pour cette vidéo
        true_mask = all_masks.get(video_id)
        if true_mask is None:
            print(f"ATTENTION: Masque non trouvé pour {video_id}. Saut de la vidéo.")
            continue

        # La boucle de frame
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg') or f.endswith('.png')])
        
        for frame_idx, filename in enumerate(frame_files):
            frame_path = os.path.join(video_path, filename)
            
            # --- SIMULATION DE LECTURE DE FRAME ---
            # Dans une implémentation réelle, vous utiliseriez OpenCV pour lire l'image
            # frame = cv2.imread(frame_path) 
            # Pour l'instant, on simule une frame pour l'appel au tracker
            
            # --- Étape 1 : Tracking ---
            # Supposons ici que 'frame' est une image que le tracker peut traiter
            # Par souci de simplification, on ne va pas lire l'image, mais c'est l'idée générale.
            
            # SIMULATION TEMPORAIRE : Vous devez remplacer ceci par la lecture et l'appel réel
            # results = tracker.update(frame) 
            # SIMULATION de résultats du tracker (pour des fins d'illustration)
            if frame_idx % 10 == 0: # Simuler un objet suivi toutes les 10 frames
                 results = [(100, 100, 150, 150, 1, 0)] # x1, y1, x2, y2, track_id, class_id
            else:
                 results = []
            
            # --- Étape 2 : Construction de Séquence ---
            for x1, y1, x2, y2, track_id, class_id in results:
                builder.add_position(track_id, (x1, y1, x2, y2))
            
            # --- Étape 3 : Détection d'Anomalie (pour les séquences complètes) ---
            ready_sequences = builder.get_ready_sequences()
            
            for track_id, sequence in ready_sequences.items():
                is_anomaly, score = detector.analyze(sequence)
                
                # --- Étape 4 : Étiquetage du Score ---
                # On attribue le label de la VÉRITÉ TERRAIN à ce score d'anomalie
                
                # Vrai label = 1 si la frame actuelle ou une frame récente est anormale dans le masque
                # Simplification : On vérifie si la frame de fin de séquence est anormale
                
                true_label = 0
                if frame_idx < len(true_mask) and true_mask[frame_idx] == 1:
                    true_label = 1
                
                all_anomaly_scores.append(score)
                all_true_labels.append(true_label)
                
                # Note: Dans une validation plus rigoureuse, on vérifie si l'objet suivi est réellement l'anomalie
                
    return np.array(all_anomaly_scores), np.array(all_true_labels)


def plot_and_find_threshold(scores, true_labels):
    """
    Calcule et affiche la courbe Precision-Recall (PR) et trouve le seuil optimal (F1-Score).
    """
    # Calcul des courbes PR
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    
    # Calcul du F1-Score pour chaque seuil
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # Note: On doit gérer les divisions par zéro
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Trouver l'indice du seuil qui maximise le F1-Score
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1 = f1_scores[best_threshold_index]
    
    # Affichage de la courbe PR
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'Courbe PR (AUC={auc(recall, precision):.4f})')
    plt.scatter(recall[best_threshold_index], precision[best_threshold_index], 
                marker='o', color='red', label=f'Optimal (Seuil={best_threshold:.4f}, F1={best_f1:.4f})')
    plt.xlabel('Rappel (Recall)')
    plt.ylabel('Précision (Precision)')
    plt.title('Courbe Precision-Recall pour la Détection d\'Anomalie')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"\n Seuil Optimal (Max F1-Score) trouvé : {best_threshold:.6f}")
    print(f"   F1-Score correspondant : {best_f1:.4f}")
    
    return best_threshold

if __name__ == '__main__':
    
    # 1. Chargement du Modèle (avec un seuil initial arbitraire)
    detector = AnomalyDetector(model_path=MODEL_PATH, threshold=0.0) # Seuil à 0.0 pour collecter tous les scores

    if detector.model is None:
        print("Échec du chargement du modèle. Arrêt de la calibration.")
        exit()

    # 2. Chargement de la Vérité Terrain (Masques)
    # NOTE: Assurez-vous que la fonction 'load_masks' fonctionne correctement avec votre format de fichier réel
    masks = load_masks(MASK_DATA_DIR)
    
    if not masks:
         print("Aucun masque de vérité terrain chargé. Vérifiez le chemin et le format.")
         exit()

    # 3. Exécution de l'Inférence et Collecte des Scores
    print("\n--- Début de la collecte des scores sur les données de TEST ---")
    scores, labels = run_inference_and_collect_scores(TEST_DATA_DIR, detector, masks)
    
    if len(scores) == 0 or np.sum(labels) == 0:
        print("\nATTENTION: Aucune donnée valide ou aucune anomalie réelle détectée dans le jeu de test. Impossible de calibrer.")
        exit()

    # 4. Calibration et Détermination du Seuil
    best_threshold = plot_and_find_threshold(scores, labels)
    
    print("\n--- Calibration Terminée ---")
    print(f"Utilisez la valeur {best_threshold:.6f} comme nouveau seuil dans votre AnomalyDetector.")