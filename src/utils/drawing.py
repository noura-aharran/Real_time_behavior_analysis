# Fichier : src/utils/drawing.py
import cv2

def draw_results(frame, results, anomaly_scores, class_names=None):
    """
    Dessine les boîtes, les IDs, la classe et les scores d'anomalie sur l'image.

    :param frame: Image (array NumPy).
    :param results: Liste des résultats du tracker [x1, y1, x2, y2, track_id, class_id].
    :param anomaly_scores: Dictionnaire {track_id: (is_anomaly: bool, score: float)}.
    :param class_names: Liste optionnelle des noms de classe pour afficher les labels.
    :return: Image avec les résultats dessinés.
    """
    # Liste par défaut des noms de classes (COCO)
    if class_names is None:
        class_names = {0: 'person', 2: 'car', 24: 'backpack', 7: 'train', 56: 'chair'} # Exemple de quelques classes

    for x1, y1, x2, y2, track_id, class_id in results:
        
        # Récupérer l'état d'anomalie
        is_anomaly, score = anomaly_scores.get(track_id, (False, 0.0))
        
        # Déterminer la couleur (Rouge pour anomalie, Vert pour normal)
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        
        # Déterminer le label
        class_label = class_names.get(class_id, f"Cls:{class_id}")
        label = f"ID:{track_id} | {class_label} | Score:{score:.3f}"
        
        # Dessiner la boîte
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Dessiner le label
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame