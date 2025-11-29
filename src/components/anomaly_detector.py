import numpy as np
import torch
import torch.nn as nn
import os
import math

# --- ARCHITECTURE DU TRANSFORMER AUTO-ENCODEUR (TAE) ---

class PositionalEncoding(nn.Module):
    """
    Ajoute l'information de position aux embeddings d'entrée (crucial pour les Transformers).
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        # Formule classique de l'encodage positionnel sinusoïdal
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tenseur d'entrée de forme (seq_len, batch_size, embed_dim)
        """
        # Ajout des valeurs d'encodage positionnel
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerAutoEncoder(nn.Module):
    """
    Auto-encodeur basé sur l'encodeur Transformer pour la reconstruction de séquences.
    """
    def __init__(self, seq_len, feature_size, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Encodage d'Entrée : Projection des features (e.g., 2 pour x, y) vers la dimension du modèle (d_model)
        self.linear_in = nn.Linear(feature_size, d_model)
        
        # 2. Encodage Positionnel : Ajout de l'information de l'ordre
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        
        # 3. Encodeur Transformer : La partie centrale qui apprend les dépendances séquentielles
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            batch_first=False # Attention : Le format attendu est (Seq_len, Batch, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 4. Décodage/Sortie : Projection de d_model vers la taille de feature originale (reconstruction)
        self.linear_out = nn.Linear(d_model, feature_size)

    def forward(self, src):
        # src forme attendue: (seq_len, batch_size, feature_size)

        # Projection des features et mise à l'échelle (norme du Transformer)
        src = self.linear_in(src) * math.sqrt(self.d_model) 
        
        # Ajout de l'encodage positionnel
        src = self.pos_encoder(src)
        
        # Passage par l'Encodeur Transformer
        output = self.transformer_encoder(src)
        
        # Reconstruction finale
        output = self.linear_out(output)
        
        # output forme: (seq_len, batch_size, feature_size)
        return output

# --- CLASSE DE DÉTECTION D'ANOMALIE ---

class AnomalyDetector:
    """
    Charge un modèle (Transformer Auto-encodeur) entraîné
    pour reconnaître les séquences de mouvement "normales" et calcule le score d'anomalie.
    """
    
    # Définition des paramètres par défaut qui doivent correspondre à l'entraînement
    DEFAULT_SEQ_LEN = 30
    DEFAULT_FEATURE_SIZE = 2
    DEFAULT_D_MODEL = 32
    DEFAULT_NHEAD = 4
    DEFAULT_NUM_LAYERS = 2
    
    def __init__(self, model_path='models/anomaly_model.pt', threshold=0.000493):
        """
        Initialise le détecteur et charge le modèle pré-entraîné.
        
        :param model_path: Chemin vers le fichier de poids du modèle (e.g., PyTorch .pt).
        :param threshold: Seuil de reconstruction/erreur au-delà duquel le comportement est jugé anormal.
        """
        self.threshold = threshold
        self.seq_len = self.DEFAULT_SEQ_LEN
        self.feature_size = self.DEFAULT_FEATURE_SIZE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        print(f"Détecteur d'anomalies (TAE) chargé sur {self.device}. Seuil d'anomalie: {self.threshold}")
        
    def _load_model(self, path):
        """
        Charge l'architecture et les poids du modèle.
        """
        if not os.path.exists(path):
            print(f"ATTENTION: Fichier modèle non trouvé à {path}. Le détecteur sera désactivé (None).")
            return None 

        try:
            # Initialisation du Transformer Auto-encodeur avec les hyperparamètres d'entraînement
            model = TransformerAutoEncoder(
                seq_len=self.seq_len, 
                feature_size=self.feature_size,
                d_model=self.DEFAULT_D_MODEL,
                nhead=self.DEFAULT_NHEAD,
                num_layers=self.DEFAULT_NUM_LAYERS
            ) 
            # Chargement des poids sur l'appareil (CPU ou GPU)
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
            # Comportement de secours si le modèle n'a pas pu être chargé
            score = np.random.rand() * 0.2 
            return score > 0.1, score

        # 1. Préparation des données: Convertir la séquence en Tenseur PyTorch
        # Format attendu pour le TAE: (seq_len, batch_size, input_features)
        
        seq_array = np.array(sequence, dtype=np.float32)
        
        # Vérification et ajout de la dimension 'batch' (1)
        # Forme (30, 2) -> Tenseur (30, 1, 2)
        input_tensor = torch.from_numpy(seq_array).unsqueeze(1).to(self.device)
        
        # 2. Inférence (prédiction)
        with torch.no_grad(): # Désactive le calcul des gradients
            # L'Auto-encodeur essaie de reconstruire la séquence d'entrée
            output = self.model(input_tensor) 
            
        # 3. Calcul de l'Erreur de Reconstruction (Score d'anomalie)
        # L'erreur (MSE) est la distance entre la séquence d'entrée et celle reconstruite
        error = nn.functional.mse_loss(output, input_tensor, reduction='mean')
        
        anomaly_score = error.item()
        
        # 4. Détermination de l'anomalie
        is_anomaly = anomaly_score > self.threshold
        
        return is_anomaly, anomaly_score

if __name__ == '__main__':
    # Exemple d'utilisation (doit simuler une séquence de 30 points (x, y))
    
    # 1. Test du chargement d'un modèle (échouera si le fichier n'existe pas)
    detector = AnomalyDetector(model_path='models/anomaly_model.pt')
    
    # 2. Simulation d'une séquence
    sequence_normale = [(i/30, i/30) for i in range(30)] # Mouvement linéaire normal
    
    # 3. Analyse
    is_anomaly, score = detector.analyze(sequence_normale)
    print(f"\nSéquence Test: Score={score:.6f}, Anomalie={is_anomaly}")