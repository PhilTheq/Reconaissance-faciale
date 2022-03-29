import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import argparse
from pathlib import Path
import os
import face_recognition as fr
from model import *




if __name__ == '__main__':
    #args = parser.parse_args()

    # On importe les visages qui sont dans notre base (dossier 'visage_connus')
    # print('[INFO] Importation des visages...')
    face_to_encode_path = Path("./visages_connus/") #args.input (line 118 & 12 & 13) for specific repo

    # On crée une variable tableau qui va stocker tous les visages connus
    files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]

    for file_ in face_to_encode_path.rglob('*.png'):
        files.append(file_)

    # On crée une variable de type tableau qui va stocker les nom des personnes dont le visage est dans la base

    path = "./visages_connus/"

    known_face_names = []
    known_name_encodings = []

    images = os.listdir(path)
    for _ in images:
        image = fr.load_image_file(path + _)
        image_path = path + _
        encoding = fr.face_encodings(image)[0]

        known_name_encodings.append(encoding)
        known_face_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())



    #known_face_names = ['Amar','Amine','Jules','Philippe']

    # Ce tableau va stocker le des encodages de chaque visage
    known_face_encodings = []

    # On parcourt la liste des fichiers des visages pour ouvrir chacun d'eux
    for file_ in files:
        image = PIL.Image.open(file_)
        image = np.array(image)

        # Encodage de chaque fichier
        face_encoded = encode_face(image)[0][0]
        known_face_encodings.append(face_encoded)

    print('[INFO] Visages importés')
    print('[INFO] Démarrage Webcam...')
    video_capture = cv2.VideoCapture(0)
    print('[INFO] Webcam démarré')
    print('[INFO] Détection...')
    while True:
        ret, frame = video_capture.read()
        easy_face_reco(frame, known_face_encodings, known_face_names)
        cv2.imshow('App de reconnaissance faciale', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Arrêt du système...')
    video_capture.release()
    cv2.destroyAllWindows()
