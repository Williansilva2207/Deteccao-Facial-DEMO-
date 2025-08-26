import cv2 as cv
from mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet

# Caminhos absolutos
img1 = cv.imread(r"C:\Users\willi\OneDrive\Documentos\Estudos\I.A\RECONHECIMENTO FACIAL\pessoa.jpg")
img2 = cv.imread(r"C:\Users\willi\OneDrive\Documentos\Estudos\I.A\RECONHECIMENTO FACIAL\pessoa2.jpg")

# Conferir se carregou
if img1 is None or img2 is None:
    print(" Erro: verifique os caminhos das imagens.")
    exit()

# Converter para RGB
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

# Detector MTCNN
detector = MTCNN()

# Detectar rostos
faces1 = detector.detect_faces(img1_rgb)
faces2 = detector.detect_faces(img2_rgb)

x1, y1, w1, h1 = faces1[0]['box']
x2, y2, w2, h2 = faces2[0]['box']

face1 = img1_rgb[y1:y1+h1, x1:x1+w1]
face2 = img2_rgb[y2:y2+h2, x2:x2+w2]

# Carregar FaceNet
embedder = FaceNet()
embedding1 = embedder.embeddings([face1])[0]
embedding2 = embedder.embeddings([face2])[0]

# Comparar
distance = np.linalg.norm(embedding1 - embedding2)
print("Distância:", distance)
if distance < 0.9:
    print("Mesma pessoa")
else:
    print("Pessoa diferente")
