import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine
from keras_facenet import FaceNet
import matplotlib.pyplot as plt

# --- 1. Initialize the FaceNet Embedder ---
# This will download the pre-trained model weights on its first run[3].
try:
    embedder = FaceNet()
    print("FaceNet embedder loaded successfully.")
except Exception as e:
    print(f"Error loading FaceNet embedder: {e}")
    exit()


def build_face_database(directory, embedder_instance):
    database = {}
    print(f"Building face database from '{directory}'...")
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        
        try:
            # The extract method performs face detection and returns detections[3].
            # Each detection includes the embedding.
            detections = embedder_instance.extract(path, threshold=0.95)
            
            if detections:
                # Use the embedding of the most confident face detection
                embedding = detections[0]['embedding']
                # Use filename as the key
                identity = os.path.splitext(filename)[0]
                database[identity] = embedding
            else:
                print(f"Warning: No face detected in {filename}. Skipping.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print(f"Database built successfully with {len(database)} entries.")
    # Save the database to a JSON file for future use
    with open(os.path.join(directory, 'face_database.json'), 'w') as f:
        import json
        # convert numpy arrays to lists for JSON serialization
        database = {name: embedding.tolist() for name, embedding in database.items()}
        json.dump(database, f, indent=4)
    print(f"Face database saved to {os.path.join(directory, 'face_database.json')}")
    return database

def find_match_in_database(query_image_path, database, embedder_instance, threshold=0.5):
    try:
        query_detections = embedder_instance.extract(query_image_path, threshold=0.95)
        if not query_detections:
            return "No face detected in query image", float('inf')
        
        query_embedding = query_detections[0]['embedding']
    except Exception as e:
        return f"Error processing query image: {e}", float('inf')

    
    best_match_name = None
    min_distance = float('inf')
    
    # Iterate through the database to find the closest match
    for name, db_embedding in database.items():
        # Calculate cosine distance
        distance = cosine(query_embedding, db_embedding)
        
        if distance < min_distance:
            min_distance = distance
            best_match_name = name
            
    # Check if the best match is within the verification threshold
    if min_distance <= threshold:
        return best_match_name, min_distance
    else:
        return "No match found", min_distance

# __main__ function to demonstrate usage
if __name__ == "__main__":
    db_path = "dataset"
    temp_path = os.path.join(db_path, "faces", "temp_image")
    # Find the latest image in the temp directory
    if not os.path.exists(temp_path):
        print(f"Temporary image directory '{temp_path}' does not exist.")
        exit()
    temp_images = [f for f in os.listdir(temp_path) if f.endswith('.jpg')]
    if not temp_images:
        print(f"No images found in temporary directory '{temp_path}'.")
        exit()
    temp_images.sort(key=lambda x: os.path.getmtime(os.path.join(temp_path, x)))
    latest_temp_image = os.path.join(temp_path, temp_images[-1])
    print(f"Using latest temporary image: {latest_temp_image}")
    
    # Code from Srinjan
    temp_image = latest_temp_image  # Path to a test image
    verification_threshold = 0.5  # Set a threshold for face verification
    
    # Building Face Database
    faces_path = os.path.join(db_path, "faces")
    face_db = build_face_database(faces_path, embedder)
    # print(f"Face database contains {len(face_db)} entries.")
    
        
    if not face_db:
        print("No valid face embeddings found in the database.")
    else:
        match_name, distance = find_match_in_database(
            temp_image,
            face_db,
            embedder,
            threshold=verification_threshold
        )
        
        print("\n --- Verification Result ---")
        if "No match" in match_name or "No face" in match_name or "Error" in match_name:
            print(f"Verification failed: {match_name}")
            print(f"Distance to closes face: {distance:.4f}" if distance != float('inf') else "")
        else:
            print(f"Face Matched")
            print(f"Identity: {match_name}")
            print(f"Cosine Distance: {distance:.4f} (Threshold: {verification_threshold})")
            
            try:
                query_img = Image.open(temp_image)
                # Find the original filename for the matched identity
                match_filename = ""
                for f in os.listdir(os.path.join(db_path, "faces")):
                    if os.path.splitext(f)[0] == match_name:
                        match_filename = f
                        break
                
                match_img = Image.open(os.path.join(db_path, "faces", match_filename))
                
                # Display the images
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(query_img)
                plt.title("Query Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(match_img)
                plt.title(f"Matched: {match_name}")
                plt.axis('off')
                
                plt.show()
            except Exception as e:
                print(f"Error displaying images: {e}")
    # db_directory = os.path.join("dataset", "faces")
    # print(f"Loading face images from: {db_directory}")
    # face_database = build_face_database(db_directory, embedder)
    
    # # Print the database
    # for name, embedding in face_database.items():
    #     print(f"{name}: {embedding[:5]}...")  # Print first 5 values of the embedding