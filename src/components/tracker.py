# Fichier : src/components/tracker.py

from ultralytics import YOLO
import numpy as np

class ObjectTracker:
    """
    Gère le suivi des objets (personnes et autres) à travers les images (frames)
    en utilisant le tracker intégré (ByteTrack) de YOLOv8.
    """
    
    def __init__(self, model_path='yolov8n.pt', tracker_config='botsort.yaml'):
        """
        Initialise le modèle YOLO et la configuration du tracker.
        
        :param model_path: Chemin vers les poids du modèle YOLO (yolov8n.pt, etc.).
        :param tracker_config: Fichier de configuration pour le tracker (e.g., 'botsort.yaml' ou 'bytetrack.yaml').
        """
        # Note: Nous chargeons le modèle YOLO ici car la méthode 'track' est attachée à lui.
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        
        # Le tracker de YOLOv8 utilise les IDs des classes COCO
        # Nous allons suivre toutes les classes (personnes, voitures, sacs, etc.)
        print(f"Tracker ByteTrack/BoTSORT initialisé avec la configuration : {tracker_config}")

    def update(self, frame, conf_threshold=0.5, iou_threshold=0.5):
        """
        Exécute la détection et le suivi en une seule étape.
        
        :param frame: L'image actuelle (array NumPy).
        :param conf_threshold: Seuil de confiance minimal pour les détections.
        :param iou_threshold: Seuil IoU pour le suivi.
        :return: Liste des résultats [x1, y1, x2, y2, track_id, class_id].
        """
        
        # --- Étape combinée de Détection et de Tracking ---
        
        # La méthode 'track' de YOLOv8 combine la détection (avec le modèle) et 
        # l'association des données (avec le tracker spécifié).
        results = self.model.track(
            source=frame,
            persist=True,               # Important : Maintient l'état du tracker entre les appels.
            tracker=self.tracker_config, # Spécifie le fichier de config du tracker (ex: 'bytetrack.yaml')
            conf=conf_threshold,        # Seuil de confiance pour les détections
            iou=iou_threshold,          # Seuil d'IoU pour l'association
            verbose=False               # Évite d'afficher les logs à chaque frame
        )
        
        tracked_results = []
        
        # La méthode 'track' retourne des objets 'Results'
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            
            # Utiliser .cpu().numpy() pour obtenir des données NumPy facilement manipulables
            boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) # [id1, id2, ...]
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int) # [cls1, cls2, ...]

            # Assembler les données
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Format de sortie : [x1, y1, x2, y2, track_id, class_id]
                tracked_results.append((x1, y1, x2, y2, track_id, class_id))
        
        return tracked_results

if __name__ == '__main__':
    # Exemple d'utilisation (Nécessite une vidéo et OpenCV)
    # import cv2
    # tracker = ObjectTracker()
    # cap = cv2.VideoCapture("chemin/vers/video.mp4")

    # while cap.isOpened():
    #     success, frame = cap.read()
    #     if success:
    #         # Le tracker exécute la détection et l'ID en une seule étape
    #         results = tracker.update(frame)
    #         
    #         # Affichage des résultats
    #         # for x1, y1, x2, y2, track_id, class_id in results:
    #         #     cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #         #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         
    #         # cv2.imshow("Tracking", frame)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #     else:
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    pass