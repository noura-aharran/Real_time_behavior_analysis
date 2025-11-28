# Fichier : src/main_app.py

import sys
import os
import streamlit as st
import cv2
import numpy as np
import time 

# --- CORRECTION MAJEURE DU CHEMIN DE RECHERCHE ---
# Ces lignes DOIVENT apparaître avant la première importation 'from src.x...'
# 1. Calcule le chemin absolu du dossier racine du projet (deux niveaux au-dessus de main_app.py)
#    Utilisation de os.path.join pour une meilleure compatibilité entre OS (Windows/Linux)
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(current_dir)

# 2. Ajoute la racine du projet au chemin de recherche des modules Python
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- Importation des Composants ---
# ATTENTION: Maintenant, ces imports absolus fonctionneront grâce aux lignes précédentes.
from src.components.tracker import ObjectTracker 
from src.components.sequence_builder import SequenceBuilder
from src.components.anomaly_detector import AnomalyDetector
from src.utils.drawing import draw_results # Utilisation de la fonction d'aide

# --- Initialisation des Composants (Utilisation de st.cache_resource pour la performance) ---
@st.cache_resource 
def initialize_components():
    """Initialise les modèles (Tracker et Detector) une seule fois."""
    
    try:
        # Tente d'initialiser les composants
        tracker = ObjectTracker(model_path='yolov8n.pt') 
        builder = SequenceBuilder(sequence_length=30)
        anomaly_model = AnomalyDetector(model_path='models/anomaly_model.pt', threshold=0.05) 
        
        # Le tracker contient les noms des classes COCO
        class_names = tracker.model.names 
        return tracker, builder, anomaly_model, class_names
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation des modèles : {e}")
        st.warning("Assurez-vous d'avoir exécuté 'train_anomaly.py' et que 'yolov8n.pt' est disponible.")
        return None, None, None, None

tracker, builder, anomaly_model, class_names = initialize_components()

# --- Interface Streamlit ---
st.set_page_config(page_title="Analyse Comportementale en Temps Réel", layout="wide")
st.title("🚨 Système de Détection d'Anomalies de Mouvement")

# Stockage de l'état du pipeline pour le temps réel
if 'anomaly_scores' not in st.session_state:
    st.session_state.anomaly_scores = {}
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# Colonnes pour l'affichage principal et le tableau de bord
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Flux Vidéo Analysé")
    frame_placeholder = st.empty()

with col2:
    st.header("Tableau de Bord des Alertes")
    alert_placeholder = st.empty()

# --- Contrôles de l'utilisateur ---
uploaded_file = st.file_uploader("Choisissez un fichier vidéo...", type=["mp4", "avi"])

# Utilisation d'un conteneur pour les boutons de contrôle pour mieux gérer l'état
control_container = st.container()

with control_container:
    col_start, col_stop = st.columns(2)
    start_button = col_start.button("Démarrer l'Analyse", disabled=st.session_state.is_running)
    stop_button = col_stop.button("Arrêter l'Analyse", disabled=not st.session_state.is_running)

# --- Logique de Démarrage ---
if start_button and not st.session_state.is_running:
    if tracker is None:
        st.error("Les modèles n'ont pas pu être initialisés. Veuillez vérifier la console.")
    else:
        st.session_state.is_running = True
        
        tfile_path = None
        cap = None
        
        # 1. Gestion de la source vidéo
        if uploaded_file:
            # Sauvegarder temporairement le fichier pour la lecture OpenCV
            tfile_path = "temp_video.mp4"
            try:
                with open(tfile_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                cap = cv2.VideoCapture(tfile_path)
            except Exception as e:
                st.error(f"Erreur lors de la sauvegarde du fichier : {e}")
                st.session_state.is_running = False

        else:
            # Tenter d'utiliser la webcam (source 0)
            cap = cv2.VideoCapture(0) 

        if not cap or not cap.isOpened():
            st.error("Impossible d'ouvrir la source vidéo. Vérifiez les permissions de la webcam ou le fichier.")
            st.session_state.is_running = False
        else:
            st.info("Analyse en cours... (Le temps réel peut varier selon les performances)")
            
            # --- Boucle Principale de Traitement ---
            while cap.isOpened() and st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Tracking
                tracked_results = tracker.update(frame) 
                current_ids = {r[4] for r in tracked_results}
                
                # 2. Construction de Séquences et Analyse
                for x1, y1, x2, y2, track_id, class_id in tracked_results:
                    box = (x1, y1, x2, y2)
                    builder.add_position(track_id, box)
                    
                    ready_sequences = builder.get_ready_sequences()
                    
                    if track_id in ready_sequences:
                        sequence = ready_sequences[track_id]
                        is_anomaly, mse_score = anomaly_model.analyze(sequence)
                        
                        st.session_state.anomaly_scores[track_id] = (is_anomaly, mse_score)
                        
                        if is_anomaly:
                            alert_message = f"🚨 Anomalie (ID:{track_id}, Classe:{class_names.get(class_id, 'N/A')}, Score:{mse_score:.4f})"
                            # Ajouter l'alerte récente en haut
                            st.session_state.alert_log.insert(0, alert_message)
                
                # 3. Nettoyage des IDs qui ont quitté
                st.session_state.anomaly_scores = {
                    tid: score for tid, score in st.session_state.anomaly_scores.items() if tid in current_ids
                }

                # 4. Affichage et Alerte
                frame_with_results = draw_results(
                    frame=frame, 
                    results=tracked_results, 
                    anomaly_scores=st.session_state.anomaly_scores,
                    class_names=class_names
                )
                
                frame_placeholder.image(frame_with_results, channels="BGR")
                
                with alert_placeholder.container():
                    st.markdown("### Alertes Récentes")
                    # Afficher uniquement les 10 dernières alertes
                    for alert in st.session_state.alert_log[:10]:
                        st.error(alert)
                        
                # time.sleep(0.01) # Mettre en pause pour éviter la surchauffe

            # Nettoyage après la boucle
            cap.release()
            # Supprimer le fichier temporaire s'il existe
            if tfile_path and os.path.exists(tfile_path):
                 os.remove(tfile_path)
            st.session_state.is_running = False
            st.success("Analyse vidéo terminée!")

# --- Logique d'Arrêt ---
if stop_button:
    st.session_state.is_running = False
    st.warning("Arrêt de l'analyse demandé.")
    # On vide les placeholders pour arrêter l'affichage
    if 'frame_placeholder' in locals():
        frame_placeholder.empty()
    if 'alert_placeholder' in locals():
        alert_placeholder.empty()