class SequenceBuilder:
    """
    Collecte les centres des boîtes (x, y) pour chaque ID sur N frames.
    """
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length # N frames nécessaires pour l'analyse
        self.sequences = {} # {track_id: [(x1, y1), (x2, y2), ...]}
        
    def add_position(self, track_id, box):
        """
        Ajoute la position centrale normalisée d'une personne à sa séquence.
        """
        # Calculer le centre (centre_x, centre_y)
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalisation (très important pour les RN)
        # Ici on suppose une normalisation simple, à adapter à votre image/scène
        normalized_position = (center_x / 1280.0, center_y / 720.0)
        
        if track_id not in self.sequences:
            self.sequences[track_id] = []
            
        self.sequences[track_id].append(normalized_position)
        
        # Maintenir la longueur de la séquence
        if len(self.sequences[track_id]) > self.sequence_length:
            self.sequences[track_id].pop(0) # Retirer la plus ancienne

    def get_ready_sequences(self):
        """
        Retourne les séquences qui sont prêtes à être analysées.
        """
        ready = {}
        for track_id, seq in self.sequences.items():
            if len(seq) == self.sequence_length:
                ready[track_id] = seq
        return ready