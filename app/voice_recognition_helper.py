import tensorflow as tf
import numpy as np
import librosa
import os
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# Paths
MODEL_PATH = os.path.join("app", "models", "voice-recognition-model.keras")
EMBEDDINGS_DIR = os.path.join("app", "models", "voice_embeddings")

# Load full model
base_model = tf.keras.models.load_model(MODEL_PATH)

# 🔍 Get embedding model (up to 'embedding_layer')
embedding_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("embedding_layer").output
)

# 🔊 Feature extraction
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfcc.shape[1] < 32:
        pad_width = 32 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :32]
    return np.expand_dims(mfcc.T, axis=0)  # (1, 32, 13)

# 🔐 Predict speaker using cosine similarity
def predict_user(audio_path):
    try:
        features = extract_features(audio_path)
        print("📦 Features shape:", features.shape)

        # Get embedding
        voice_embedding = embedding_model.predict(features, verbose=0)[0]
        print("🧠 Voice embedding shape:", voice_embedding.shape)

        best_match = None
        highest_similarity = -1
        threshold = 0.80

        for filename in os.listdir(EMBEDDINGS_DIR):
            if filename.endswith(".npy"):
                username = filename.replace("_voice_embedding.npy", "")
                saved_embedding = np.load(os.path.join(EMBEDDINGS_DIR, filename))

                print(f"🔍 Comparing with {username}")
                print(f"🔸 Saved embedding shape: {saved_embedding.shape}")

                similarity = cosine_similarity([voice_embedding], [saved_embedding])[0][0]
                print(f"📊 Similarity with {username}: {similarity:.4f}")

                if similarity > highest_similarity and similarity >= threshold:
                    highest_similarity = similarity
                    best_match = username

        return best_match

    except Exception as e:
        print("🚨 Voice prediction error:")
        traceback.print_exc()
        return None
