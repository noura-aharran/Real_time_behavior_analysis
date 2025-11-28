# Fichier : src/inference/real_time_inference.py

import cv2
import numpy as np

# Importer tous les composants du pipeline
from src.components.tracker import ObjectTracker
from src.components.sequence_builder import SequenceBuilder
from src.components.anomaly_detector import AnomalyDetector

# L'importation correcte de la fonction de dessin externe
from src.utils.drawing import draw_results
from src.utils.preprocessing import normalize_position # Ajout pour une meilleure robustesse

class RealTimeInference:
    """
    Orchestre la détection, le suivi, la construction de séquences et l'analyse 
    des anomalies en temps réel sur un flux vidéo.
    """
    def __init__(self, video_source=0, sequence_length=30):
        # 1. Initialisation des composants
        self.tracker = ObjectTracker(model_path='yolov8n.pt') 
        self.builder = SequenceBuilder(sequence_length=sequence_length)
        self.detector = AnomalyDetector(model_path='models/anomaly_model.pt', threshold=0.05) 
        
        # 2. Configuration de la source vidéo
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la source vidéo: {video_source}")

        # Récupérer les dimensions du flux (nécessaire pour la normalisation)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Flux vidéo démarré : {self.width}x{self.height} pixels.")
        
        # Structure de stockage des scores pour l'affichage
        self.anomaly_scores = {} # {track_id: (is_anomaly, score)}

    # L'ancienne méthode self._draw_results EST SUPPRIMÉE, 
    # car elle est maintenant remplacée par la fonction externe draw_results.

    def run(self):
        """
        Boucle principale de traitement du flux vidéo.
        """
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            # 1. Étape de Tracking (Détection + Suivi)
            # results: [x1, y1, x2, y2, track_id, class_id]
            tracked_results = self.tracker.update(frame)

            # Liste des IDs qui étaient dans la frame actuelle
            current_ids = {r[4] for r in tracked_results}
            
            # 2. Construction des Séquences et Analyse
            for x1, y1, x2, y2, track_id, class_id in tracked_results:
                
                # Créer le tuple de boîte (format attendu par le builder)
                box = (x1, y1, x2, y2)
                
                # Utiliser la fonction de normalisation du dossier utils/
                # NOTE: Nous utilisons la fonction helper pour obtenir la position (x, y) normalisée,
                # mais le SequenceBuilder gère déjà la normalisation interne par défaut (1280x720). 
                # Pour un code robuste, vous passeriez width/height au builder ou utiliseriez 
                # le résultat de normalize_position dans add_position.
                
                self.builder.add_position(
                    track_id=track_id, 
                    box=box,
                    # Le SequenceBuilder utilise maintenant la fonction normalize_position 
                    # si nous le mettions à jour, pour l'instant on garde l'appel simple.
                )

            # 3. Analyse des séquences complètes
            ready_sequences = self.builder.get_ready_sequences()
            for track_id, sequence in ready_sequences.items():
                
                # Analyse de la séquence de 30 positions
                is_anomaly, score = self.detector.analyze(sequence)
                
                # Stocker le résultat pour l'affichage
                self.anomaly_scores[track_id] = (is_anomaly, score)

            # 4. Nettoyage des vieux scores (pour les IDs qui ont quitté la scène)
            self.anomaly_scores = {
                tid: score for tid, score in self.anomaly_scores.items() if tid in current_ids
            }

            # 5. Affichage des résultats
            # Appel de la fonction EXTERNE draw_results
            frame_with_results = draw_results(
                frame=frame, 
                results=tracked_results, 
                anomaly_scores=self.anomaly_scores
            )
            cv2.imshow("Analyse de Comportement en Temps Reel", frame_with_results)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 6. Nettoyage et arrêt
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Pour utiliser votre webcam, laissez 0. 
    # Pour utiliser un fichier vidéo, remplacez par le chemin: "path/to/video.mp4"
    VIDEO_SOURCE = 0 
    
    try:
        pipeline = RealTimeInference(video_source=VIDEO_SOURCE)
        pipeline.run()
    except IOError as e:
        print(f"Erreur fatale: {e}")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite: {e}")