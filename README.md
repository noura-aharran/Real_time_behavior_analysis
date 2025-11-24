# Real-Time Behavior Analysis – Computer Vision System

Ce projet implémente un système complet de détection de comportements anormaux en temps réel à partir de flux vidéo. Il combine la détection de personnes, le tracking multi-objets, la construction de séquences temporelles et l’analyse de mouvement à l’aide d’un modèle d’anomalie basé sur la reconstruction.



## Objectif du projet

Développer un pipeline performant et modulable qui permet :

1. **Détection des personnes** dans une scène vidéo.
2. **Tracking des individus** avec attribution d'un identifiant unique (ID tracking).
3. **Extraction de séquences temporelles** propres à chaque individu (trajectoire + mouvements).
4. **Analyse du comportement** via un modèle d’anomalie entraîné uniquement sur des séquences normales.
5. **Détection d'anomalies en temps réel** (gestes brusques, comportements inhabituels, menaces, chutes, etc.).
6. **Affichage des résultats** à travers une interface Streamlit (visualisation + alertes).

Le système vise à être léger, temps réel, et utilisable sur une machine locale sans GPU haut de gamme.


## Pipeline fonctionnel

Le flux du système est organisé comme suit :

1. **Input caméra ou vidéo**  
2. **Person Detector** (ex : YOLOv8)  
   → extrait les bounding boxes des personnes  
3. **Tracking** (ByteTrack / DeepSORT)  
   → attribution d’un ID unique à chaque personne  
4. **Sequence Builder**  
   → construction d’une séquence temporelle pour chaque ID (mouvements + features)  
5. **Anomaly Detector**  
   → modèle entraîné uniquement sur des comportements normaux  
   → calcule l’erreur de reconstruction  
   → décide normal / anormal  
6. **Streamlit App**  
   → visualisation vidéo  
   → tracking des IDs  
   → affichage des alertes d’anomalie en temps réel  



## Détection d’anomalies : principe utilisé

Le projet utilise une approche **non supervisée** basée sur un **Autoencoder temporel**.

- Le modèle est **entraîné uniquement sur des séquences normales**.
- Lors de l’inférence :
  - Si une séquence est normale → facile à reconstruire → faible erreur
  - Si une séquence est anormale → reconstruction difficile → erreur élevée → anomalie

Cette méthode est robuste et fonctionne même sans étiquettes d’anomalies.


##  Modèles utilisés

- **Détection de personnes** : YOLOv8 / YOLO-NAS / YOLOv11 (au choix)
- **Tracking** : ByteTrack ou DeepSORT
- **Features extraction** : MobileNet / EfficientNet / MediaPipe Pose
- **Anomaly Model** : LSTM Autoencoder ou ConvLSTM Autoencoder  
  (selon les besoins et les ressources)




## Dataset recommandé

Pour l’entraînement de l’anomalie :

- UCSD Ped2 (léger, facile à manipuler)
- Avenue Dataset
- ShanghaiTech (plus complet mais très lourd) (https://svip-lab.github.io/dataset/campus_dataset.html)


## Fonctionnalités principales

- Détection et tracking en temps réel
- Séquences temporelles par personne
- Modèle d'anomalie non supervisé
- Alerte automatique sur comportements suspects
- Interface utilisateur simple via Streamlit
- Modularité complète pour ajouter de nouveaux modèles



## Contribution

Toute contribution est la bienvenue : amélioration du modèle, ajout d’un dataset, optimisation du tracking, etc.



## Licence

Projet disponible sous licence MIT.


