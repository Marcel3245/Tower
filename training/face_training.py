import cv2
import os
from pathlib import Path
import numpy as np
from PIL import Image

detector = cv2.CascadeClassifier("../Cascades/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

Path("pictures/training").mkdir(exist_ok=True)
Path("pictures/output").mkdir(exist_ok=True)
namesIdsFile = 'pictures/output/names-ids.txt'
encode_ids = open(namesIdsFile, 'w')

def encode_known_faces():
    names = []
    faceSamples = []
    ids = []
    id_counter = 0

    for filepath in Path("pictures/training").glob("*/*"):
        name = filepath.parent.name
        if name not in names:
            names.append(name)
            id_counter += 1
        
        # Load images from each class
        path = f"pictures/training/{name}"
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]     
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img,'uint8')
            faces = detector.detectMultiScale(img_numpy)
            
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w]) 
                ids.append(id_counter)

    for i in range(len(np.unique(ids))):
        encode_ids.write(f'{i}, {names[i]} \n')

    return faceSamples, ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces, ids = encode_known_faces()
recognizer.train(faces, np.array(ids))
recognizer.write('pictures/output/trainer.yml')

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
