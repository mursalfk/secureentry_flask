import os
import numpy as np
import tensorflow as tf
import librosa

# === Paths ===
DATASET_DIR = "app/voice_data"
MODEL_PATH = "app/models/voice-recognition-model.keras"
EMBEDDINGS_DIR = "app/models/voice_embeddings"

# === Load model and get embedding layer ===
base_model = tf.keras.models.load_model(MODEL_PATH)
embedding_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("embedding_layer").output
)

# === Ensure output directory exists ===
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# === Feature extraction ===
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfcc.shape[1] < 32:
        pad_width = 32 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :32]
    return np.expand_dims(mfcc.T, axis=0)  # (1, 32, 13)

# === Loop through users ===
speaker_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
print(f"🧑‍🤝‍🧑 Found {len(speaker_folders)} speaker folders.")

for speaker in speaker_folders:
    speaker_path = os.path.join(DATASET_DIR, speaker, "chunks")
    if not os.path.exists(speaker_path):
        print(f"⚠️ Skipping {speaker} — no chunks/ folder found.")
        continue

    files = [f for f in os.listdir(speaker_path) if f.endswith(".wav")]
    if len(files) == 0:
        print(f"⚠️ Skipping {speaker} — no audio files.")
        continue

    features = extract_features(os.path.join(speaker_path, files[0]))
    embedding = embedding_model.predict(features, verbose=0)[0]

    save_path = os.path.join(EMBEDDINGS_DIR, f"{speaker}_voice_embedding.npy")
    np.save(save_path, embedding)
    print(f"✅ Saved embedding for {speaker} → {save_path}")
