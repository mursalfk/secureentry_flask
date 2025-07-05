import os
import base64
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine
from keras_facenet import FaceNet

embedder = FaceNet()  # load once

def build_face_database(directory):
    db = {}
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        try:
            detections = embedder.extract(path, threshold=0.95)
            if detections:
                identity = os.path.splitext(file)[0]
                db[identity] = detections[0]['embedding']
        except Exception as e:
            print(f"❌ Error in {file}: {e}")
    return db

def find_match_in_database(image_path, database, threshold=0.5):
    try:
        detections = embedder.extract(image_path, threshold=0.95)
        if not detections:
            return "No face detected", float('inf')

        query_embedding = detections[0]['embedding']
        min_dist = float('inf')
        best_match = None

        for name, db_emb in database.items():
            dist = cosine(query_embedding, db_emb)
            if dist < min_dist:
                min_dist = dist
                best_match = name

        if min_dist <= threshold:
            return best_match, min_dist
        else:
            return "No match found", min_dist
    except Exception as e:
        return f"Error: {e}", float('inf')