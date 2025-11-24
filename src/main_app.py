import streamlit as st
import cv2
import numpy as np

# Importer les composants
from components.person_detector import PersonDetector
from components.tracker import ObjectTracker 
from components.sequence_builder import SequenceBuilder
from components.anomaly_detector import AnomalyDetector
# from src.utils.drawing import draw_boxes # Vous devrez créer cette utilitaire

# --- Initialisation des Composants ---
@st.cache_resource 
def initialize_components():
    """Initialise les modèles et les composants une seule fois."""
    detector = PersonDetector()
    tracker = ObjectTracker()
    builder = SequenceBuilder(sequence_length=30)
    # Assurez-vous que le modèle est entraîné et que le chemin est correct
    anomaly_model = AnomalyDetector(threshold=0.01) 
    return detector, tracker, builder, anomaly_model

detector, tracker, builder, anomaly_model = initialize_components()

# --- Interface Streamlit ---
st.set_page_config(page_title="Analyse Comportementale en Temps Réel", layout="wide")
st.title("🚨 Système de Détection d'Anomalies Comportementales")

# Colonnes pour l'affichage principal et le tableau de bord
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Flux Vidéo Analysé")
    # Placeholder pour l'affichage de la vidéo
    frame_placeholder = st.empty()

with col2:
    st.header("Tableau de Bord des Alertes")
    alert_log = []
    alert_placeholder = st.empty()

# Début de l'analyse (simuler l'entrée vidéo)
# Ceci devrait être remplacé par la lecture d'une webcam ou d'un fichier vidéo

video_file = st.file_uploader("Choisissez un fichier vidéo...", type=["mp4", "avi"])

if video_file:
    # Sauvegarder temporairement le fichier pour la lecture OpenCV
    tfile = "temp_video.mp4"
    with open(tfile, "wb") as f:
        f.write(video_file.getbuffer())
        
    cap = cv2.VideoCapture(tfile)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Détection de Personnes
        person_boxes = detector.detect(frame)

        # 2. Tracking ID
        tracked_results = tracker.update(person_boxes, frame) # [x1, y1, x2, y2, track_id]

        # Traitement pour chaque personne trackée
        frame_with_boxes = frame.copy()
        for x1, y1, x2, y2, track_id in tracked_results:
            
            # 3. Construction de Séquences
            builder.add_position(track_id, (x1, y1, x2, y2))
            
            is_anomaly = False
            mse_score = 0.0
            
            # 4. Analyse des Mouvements si la séquence est complète
            ready_sequences = builder.get_ready_sequences()
            if track_id in ready_sequences:
                sequence = ready_sequences[track_id]
                is_anomaly, mse_score = anomaly_model.analyze(sequence)

            # 5. Affichage et Alerte
            box_color = (0, 255, 0) # Vert par défaut
            if is_anomaly:
                box_color = (0, 0, 255) # Rouge si anomalie
                alert_message = f"🚨 Anomalie (ID: {track_id}, Score: {mse_score:.4f})"
                if alert_message not in alert_log:
                    alert_log.insert(0, alert_message) # Ajouter l'alerte
            
            # --- Dessin (Utilitaire draw_boxes à implémenter) ---
            # draw_boxes(frame_with_boxes, x1, y1, x2, y2, track_id, mse_score, box_color)
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame_with_boxes, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            cv2.putText(frame_with_boxes, f'MSE: {mse_score:.4f}', (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Mettre à jour l'affichage vidéo
        frame_placeholder.image(frame_with_boxes, channels="BGR")
        
        # Mettre à jour le tableau de bord des alertes
        with alert_placeholder.container():
            for alert in alert_log[:10]: # Afficher les 10 dernières
                st.error(alert)
                
    cap.release()
    st.success("Analyse vidéo terminée!")