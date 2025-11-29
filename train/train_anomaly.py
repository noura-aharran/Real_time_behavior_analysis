import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import math # Nécessaire pour l'initialisation du Transformer

# --- 1. Importation des Composants TAE (à partir de votre fichier anomaly_detector.py) ---
# Nous importons les classes d'architecture définies dans le fichier de détection
from src.components.anomaly_detector import TransformerAutoEncoder, PositionalEncoding 
from src.utils.dataset_loader import load_and_preprocess_training_data # Nécessaire pour le chargement des données

# Assurez-vous que le répertoire 'models' existe
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# L'architecture TransformerAutoEncoder est maintenant importée et utilisée.
# Nous n'avons plus besoin de la redéfinir ici, car elle est dans src/components/anomaly_detector.py.

# --- 2. Fonction d'Entraînement ---

def train_model(data_loader, model, epochs=50, learning_rate=1e-3):
    """
    Fonction principale pour l'entraînement du Transformer Auto-encodeur.
    Le modèle est entraîné à minimiser l'erreur de reconstruction (MSE) sur les séquences normales.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss(reduction='mean') # Erreur de reconstruction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for [batch] in data_loader:
            # Le Transformer attend l'ordre (Seq_len, Batch, Feature) par défaut.
            # Le DataLoader donne (Batch, Seq_len, Feature). Nous devons transposer.
            batch = batch.to(device)
            input_sequence = batch.transpose(0, 1).float() # (32, 30, 2) -> (30, 32, 2)
            
            optimizer.zero_grad()
            
            # Reconstruction
            outputs = model(input_sequence)
            
            # Calcul de l'erreur (outputs et input_sequence doivent être de même forme)
            loss = criterion(outputs, input_sequence)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(data_loader):.6f}')

    # Sauvegarde du modèle entraîné (État du Transformer)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'anomaly_model.pt'))
    print(f"\nModèle Transformer sauvegardé dans {os.path.join(MODEL_DIR, 'anomaly_model.pt')}")


if __name__ == '__main__':
    # --- 3. Paramètres et Démarrage de l'Entraînement ---
    
    # Ces paramètres doivent correspondre aux DEFAULT_... dans AnomalyDetector
    SEQ_LEN = 30
    BATCH_SIZE = 32
    N_FEATURES = 2 
    D_MODEL = 32 # Doit être cohérent avec AnomalyDetector

    print("--- Démarrage de l'Entraînement du Transformer Auto-encodeur (TAE) ---")
    
    # 1. Chargement des données (simulées - assurez-vous que load_and_preprocess_training_data fonctionne)
    # NOTE: Assurez-vous que le chemin 'data/videos/' est correct pour les 330 vidéos normales
    data_np = load_and_preprocess_training_data(videos_dir='data/videos/', sequence_length=SEQ_LEN)
    
    if data_np.shape[0] > 0:
        data_tensor = torch.from_numpy(data_np)
        dataset = TensorDataset(data_tensor)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 2. Initialisation du modèle Transformer
        model = TransformerAutoEncoder(
            seq_len=SEQ_LEN, 
            feature_size=N_FEATURES, 
            d_model=D_MODEL
        )
        
        # 3. Entraînement
        train_model(data_loader, model, epochs=20)
    else:
        print("Pas assez de données pour l'entraînement. Entraînement ignoré.")