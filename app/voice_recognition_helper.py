import tensorflow as tf
import numpy as np
import os
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from sklearn.preprocessing import StandardScaler

# Paths
MODEL_PATH = os.path.join("app", "models", "voice-recognition-model.keras")
EMBEDDINGS_DIR = os.path.join("app", "models", "voice_embeddings")

# Load the saved model and warm it up
base_model = tf.keras.models.load_model(MODEL_PATH)
_ = base_model(np.zeros((1, 32, 13), dtype=np.float32))  # warmup call

# Now extract the correct embedding layer
embedding_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("embedding_layer").output
)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000, duration=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = StandardScaler().fit_transform(mfcc)
    if mfcc.shape[1] < 32:
        pad_width = 32 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :32]
    return mfcc.T  # (32, 13)


# 🔐 Predict speaker using cosine similarity
def predict_user(audio_path):
    try:
        features = extract_features(audio_path)
        print("📦 Features shape:", features.shape)

        # Get embedding
        features = np.expand_dims(features, axis=0)
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