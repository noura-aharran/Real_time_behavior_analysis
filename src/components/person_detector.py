from ultralytics import YOLO

class PersonDetector:
    """
    Initialise et exécute la détection de personnes avec un modèle YOLOv8 pré-entraîné.
    """
    def __init__(self, model_path='yolov8n.pt'):
        # Charger le modèle YOLOv8 nano (rapide et léger)
        self.model = YOLO(model_path)
        # Assurer que l'on ne détecte que les personnes (classe COCO '0')
        self.target_class = 0 
        print(f"Détecteur de personnes YOLOv8 chargé.")

    def detect(self, frame):
        """
        Détecte les personnes dans une image (frame).
        
        :param frame: Image (array NumPy).
        :return: Liste des boîtes englobantes des personnes [x1, y1, x2, y2].
        """
        # Exécuter l'inférence
        results = self.model.predict(frame, verbose=False)
        
        person_boxes = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                # Filtrer par classe (doit être 'person')
                if int(box.cls) == self.target_class:
                    # Coordonnées au format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    person_boxes.append((x1, y1, x2, y2))
                    
        return person_boxes

if __name__ == '__main__':
    # Exemple d'utilisation (nécessite une image ou OpenCV pour la capturer)
    # detector = PersonDetector()
    # image = cv2.imread("chemin/vers/image.jpg")
    # boxes = detector.detect(image)
    # print(f"Boîtes détectées : {boxes}")
    pass