# run this manually to generate face_database.json
from facial.face_utils import build_face_database
from keras_facenet import FaceNet

embedder = FaceNet()
build_face_database("dataset/faces", embedder)
