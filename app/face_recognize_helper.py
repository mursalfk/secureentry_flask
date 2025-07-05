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

def find_match_in_database(query_image_path, database, embedder_instance, threshold=0.5):
    """
    Finds the best match for a query image in the face database.
    
    Args:
        query_image_path (str): Path to the image to verify.
        database (dict): The dictionary of known face embeddings.
        embedder_instance (FaceNet): The FaceNet embedder instance.
        threshold (float): The maximum cosine distance to be considered a match.
        
    Returns:
        A tuple of (best_match_name, min_distance).
    """
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


# --- 3. Main Execution ---

# if __name__ == "__main__":
#     # --- Configuration ---
#     # Path to the folder containing images of known people
#     DATABASE_DIR = 'path/to/your/database_folder'
#     # Path to the new image you want to verify
#     QUERY_IMAGE_PATH = 'path/to/your/query_image.jpg' # <-- IMPORTANT: Change these paths
#     # Verification threshold (lower is stricter). 0.5 is a good starting point for cosine distance.
#     VERIFICATION_THRESHOLD = 0.5

#     # --- Step 1: Build the database of known faces ---
#     face_db = build_face_database(DATABASE_DIR, embedder)

#     if not face_db:
#         print("Database is empty. Please add images to the database folder and try again.")
#     else:
#         # --- Step 2: Verify the query image against the database ---
#         match_name, distance = find_match_in_database(
#             QUERY_IMAGE_PATH, face_db, embedder, VERIFICATION_THRESHOLD
#         )
        
#         print("\n--- Verification Result ---")
#         if "No match" in match_name or "No face" in match_name or "Error" in match_name:
#             print(f"Result: {match_name}")
#             print(f"Distance to closest face: {distance:.4f}" if distance != float('inf') else "")
#         else:
#             print(f"Face Matched!")
#             print(f"Identity: {match_name}")
#             print(f"Cosine Distance: {distance:.4f} (Threshold: {VERIFICATION_THRESHOLD})")
            
#             # --- Step 3: Visualize the result ---
#             try:
#                 query_img = Image.open(QUERY_IMAGE_PATH)
#                 # Find the original filename for the matched identity
#                 match_filename = ""
#                 for f in os.listdir(DATABASE_DIR):
#                     if os.path.splitext(f)[0] == match_name:
#                         match_filename = f
#                         break
                
#                 match_img = Image.open(os.path.join(DATABASE_DIR, match_filename))
                
#                 fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#                 axes[0].imshow(query_img)
#                 axes[0].set_title("Query Image")
#                 axes[0].axis('off')
                
#                 axes[1].imshow(match_img)
#                 axes[1].set_title(f"Best Match: {match_name}")
#                 axes[1].axis('off')
                
#                 plt.suptitle(f'Verification Successful (Distance: {distance:.4f})', fontsize=16)
#                 plt.show()
#             except Exception as e:
#                 print(f"Could not display images. Error: {e}")