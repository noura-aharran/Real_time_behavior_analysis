# Fichier : src/components/anomaly_detector.py

import numpy as np
import torch
import torch.nn as nn
import os

class AnomalyDetector:
    """
    Charge un modèle (par exemple, un Auto-encodeur ou un LSTM) entraîné
    pour reconnaître les séquences de mouvement "normales" et calcule le score d'anomalie.
    """
    
    def __init__(self, model_path='models/anomaly_model.pt', threshold=0.05):
        """
        Initialise le détecteur et charge le modèle pré-entraîné.
        
        :param model_path: Chemin vers le fichier de poids du modèle (e.g., PyTorch .pt).
        :param threshold: Seuil de reconstruction/erreur au-delà duquel le comportement est jugé anormal.
        """
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        print(f"Détecteur d'anomalies chargé sur {self.device}. Seuil d'anomalie: {self.threshold}")
        
    def _load_model(self, path):
        """
        Charge l'architecture et les poids du modèle.
        NOTE: Vous devez définir et importer ici la classe de votre modèle (e.g., LSTMAutoEncoder).
        """
        if not os.path.exists(path):
             # Créez le dossier 'models' si nécessaire et assurez-vous que le modèle y est placé
            print(f"ATTENTION: Fichier modèle non trouvé à {path}. Le détecteur sera un MOCK.")
            return None # Retourne None si le modèle n'est pas trouvé
        
        # Exemple d'initialisation d'une architecture fictive pour charger les poids
        # Dans un vrai projet, remplacez 'YourModelArchitecture' par votre classe réelle
        try:
            # Ex: model = YourModelArchitecture(input_size=2, hidden_size=64, num_layers=2)
            # Puis: model.load_state_dict(torch.load(path))
            
            # Pour le squelette, on utilise un simple placeholder
            model = DummyModel() # Placeholder
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval() # Met le modèle en mode évaluation
            return model
        except Exception as e:
            print(f"Erreur lors du chargement du modèle PyTorch: {e}")
            return None
    
    def analyze(self, sequence):
        """
        Analyse une séquence de mouvement complète (N frames) et calcule le score d'anomalie.
        
        :param sequence: Liste de 30 positions normalisées (e.g., [(x0, y0), (x1, y1), ...]).
        :return: (is_anomaly: bool, score: float)
        """
        if self.model is None:
            # Comportement de secours si le modèle n'a pas pu être chargé (par exemple, retour d'une anomalie aléatoire)
            score = np.random.rand() * 0.2 
            return score > 0.1, score

        # 1. Préparation des données: Convertir la séquence en Tenseur PyTorch
        # La séquence doit avoir le format (longueur_sequence, batch_size, input_features)
        # Ici: (30 frames, 1 batch, 2 features (x, y))
        seq_array = np.array(sequence, dtype=np.float32)
        # Ajout de la dimension 'batch' (1) et conversion en Tenseur
        input_tensor = torch.from_numpy(seq_array).unsqueeze(1).to(self.device)
        
        # 2. Inférence (prédiction)
        with torch.no_grad(): # Désactive le calcul des gradients pour l'inférence
            # L'Auto-encodeur essaie de reconstruire la séquence d'entrée
            # Exemple: output, _ = self.model(input_tensor)
            output = self.model(input_tensor) # Utilisation simplifiée pour le placeholder
            
        # 3. Calcul de l'Erreur de Reconstruction (Metric pour l'anomalie)
        # L'erreur (MSE) est la distance entre la séquence d'entrée (input_tensor) et celle reconstruite (output)
        error = nn.functional.mse_loss(output, input_tensor, reduction='mean')
        
        # Le score est l'erreur moyenne
        anomaly_score = error.item()
        
        # 4. Détermination de l'anomalie
        is_anomaly = anomaly_score > self.threshold
        
        return is_anomaly, anomaly_score

# --- Modèle Placeholder (À remplacer par votre vraie architecture) ---
# Nécessaire pour que le code puisse être importé sans erreur si un vrai modèle est défini ailleurs.
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Un modèle bidon qui retourne simplement l'entrée
    def forward(self, x):
        return x # Simule une reconstruction parfaite (erreur ~ 0)
        
if __name__ == '__main__':
    # Exemple d'utilisation (doit simuler une séquence de 30 points)
    # detector = AnomalyDetector()
    # sequence_normale = [(i/30, i/30) for i in range(30)] # Ligne droite normale
    # sequence_anormale = [(i/30, 0.5) for i in range(15)] + [(0.5, i/30) for i in range(15, 30)] # Changement brusque

    # is_anomaly, score = detector.analyze(sequence_normale)
    # print(f"Séquence Normale: Score={score:.4f}, Anomaly={is_anomaly}")
    
    # is_anomaly, score = detector.analyze(sequence_anormale)
    # print(f"Séquence Anormale: Score={score:.4f}, Anomaly={is_anomaly}")
    pass