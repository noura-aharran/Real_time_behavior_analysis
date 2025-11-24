import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

class AnomalyDetector:
    """
    Autoencodeur basé sur LSTM pour la détection d'anomalies.
    """
    def __init__(self, sequence_length=30, n_features=2, model_path='models/anomaly_analyzer/lstm_ae.h5', threshold=0.01):
        self.sequence_length = sequence_length # Longueur de la séquence (N)
        self.n_features = n_features         # Nombre de features (x, y) = 2
        self.threshold = threshold           # Seuil d'erreur de reconstruction
        self.model = self._build_model()
        
        try:
            self.model.load_weights(model_path)
            print("Poids du modèle d'anomalie chargés.")
        except Exception:
            print("ATTENTION: Poids du modèle d'anomalie non trouvés. Le modèle doit être entraîné.")
            
    def _build_model(self):
        """Construit l'architecture de l'Autoencodeur LSTM."""
        # Encodeur
        inputs = Input(shape=(self.sequence_length, self.n_features))
        encoded = LSTM(128, activation='relu', return_sequences=True)(inputs)
        encoded = LSTM(64, activation='relu', name='encoder_output')(encoded)
        
        # Répéter le vecteur latent pour le Décodeur
        repeated_vector = RepeatVector(self.sequence_length)(encoded)
        
        # Décodeur
        decoded = LSTM(64, activation='relu', return_sequences=True)(repeated_vector)
        decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
        
        # Sortie (même forme que l'entrée)
        output = TimeDistributed(Dense(self.n_features))(decoded)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse') # Mean Squared Error (MSE)
        return model

    def analyze(self, sequence):
        """
        Analyse une séquence pour détecter une anomalie.
        
        :param sequence: Une séquence de mouvement (Numpy array, shape [N, 2]).
        :return: (is_anomaly, reconstruction_error)
        """
        # Préparer les données pour le modèle (ajouter la dimension de batch)
        X = np.array(sequence).reshape(1, self.sequence_length, self.n_features)
        
        # Reconstruction
        X_pred = self.model.predict(X, verbose=0)
        
        # Calcul de l'erreur de reconstruction (MSE)
        # Nous comparons les séquences originales et reconstruites
        mse = np.mean(np.power(X - X_pred, 2))
        
        is_anomaly = mse > self.threshold
        
        return is_anomaly, mse