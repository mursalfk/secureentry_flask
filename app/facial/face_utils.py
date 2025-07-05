import os, json
import numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

embedder = FaceNet()

def build_face_database(directory, embedder_instance):
    database = {}
    for filename in os.listdir(directory):
        if not filename.endswith('.jpg'): continue
        path = os.path.join(directory, filename)
        detections = embedder_instance.extract(path, threshold=0.95)
        if detections:
            embedding = detections[0]['embedding']
            identity = os.path.splitext(filename)[0]
            database[identity] = embedding.tolist()
    with open(os.path.join(directory, "face_database.json"), "w") as f:
        json.dump(database, f)
    return database

def load_face_database(path="app/dataset/faces/face_database.json"):
    with open(path, "r") as f:
        return json.load(f)

def find_match_in_database(image_path, database, threshold=0.5):
    detections = embedder.extract(image_path, threshold=0.95)
    if not detections:
        return "No face detected", float('inf')

    query_embedding = detections[0]['embedding']
    best_match = None
    min_dist = float("inf")

    for name, emb in database.items():
        dist = cosine(query_embedding, np.array(emb))
        if dist < min_dist:
            min_dist = dist
            best_match = name

    return (best_match, min_dist) if min_dist <= threshold else ("No match", min_dist)
