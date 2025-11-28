# Fichier : train/train_anomaly.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# Assurez-vous que le répertoire 'models' existe
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Définition de l'architecture du modèle (LSTM Auto-encodeur)
# Fichier : train/train_anomaly.py (CLASSE CORRIGÉE)

class LSTMAutoEncoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoEncoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        # Encodeur (réduit la séquence à un vecteur 'latent')
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Décodeur (reconstruit la séquence)
        self.decoder = nn.LSTM(
            input_size=n_features, # L'entrée du décodeur est généralement l'output du pas précédent (ici, nous utilisons le vecteur latent)
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Encodeur
        # output_e: (batch_size, seq_len, embedding_dim)
        # hidden_e: (num_layers, batch_size, embedding_dim) (état caché final)
        _, (hidden_e, cell_e) = self.encoder(x)
        
        # 2. Préparation du Décodeur
        # Initialiser l'input du décodeur à zéro, de la taille (batch_size, seq_len, n_features)
        # Ceci est nécessaire pour que le décodeur fasse N pas de temps
        decoder_input = torch.zeros(batch_size, self.seq_len, self.n_features).to(x.device)
        
        # L'état initial du décodeur est l'état final de l'encodeur
        decoder_hidden = (hidden_e, cell_e)
        
        # 3. Décodeur
        # output_d: (batch_size, seq_len, embedding_dim)
        output_d, _ = self.decoder(decoder_input, decoder_hidden)
        
        # 4. Couche de Sortie (projection de embedding_dim vers n_features)
        output = self.output_layer(output_d)
        
        return output
    
def train_model(data_loader, model, epochs=50, learning_rate=1e-3):
    """
    Fonction principale pour l'entraînement.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss(reduction='mean') # Erreur de reconstruction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for [batch] in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Reconstruction
            outputs = model(batch)
            
            # Calcul de l'erreur
            loss = criterion(outputs, batch)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(data_loader):.6f}')

    # Sauvegarde du modèle entraîné
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'anomaly_model.pt'))
    print(f"Modèle sauvegardé dans {os.path.join(MODEL_DIR, 'anomaly_model.pt')}")


if __name__ == '__main__':
    from src.utils.dataset_loader import load_and_preprocess_training_data
    
    SEQ_LEN = 30
    BATCH_SIZE = 32
    N_FEATURES = 2 # (x, y)
    
    # 1. Chargement des données (simulées)
    data_np = load_and_preprocess_training_data(videos_dir='data/videos/', sequence_length=SEQ_LEN)
    
    if data_np.shape[0] > 0:
        data_tensor = torch.from_numpy(data_np)
        dataset = TensorDataset(data_tensor)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 2. Initialisation du modèle
        model = LSTMAutoEncoder(seq_len=SEQ_LEN, n_features=N_FEATURES)
        
        # 3. Entraînement
        print("Démarrage de l'entraînement de l'Auto-Encodeur...")
        train_model(data_loader, model, epochs=20)
    else:
        print("Pas assez de données pour l'entraînement. Entraînement ignoré.")