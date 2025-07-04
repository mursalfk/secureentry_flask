import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

# Patch np.complex for compatibility with librosa
if not hasattr(np, 'complex'):
    np.complex = complex

# Paths
DATASET_DIR = "app/dataset/voice_data"
MODEL_SAVE_PATH = "app/models/voice-recognition-model.keras"
LABEL_MAP_PATH = "app/models/label_map.npy"
EMBEDDINGS_DIR = "app/models/voice_embeddings"

print("🚀 Starting voice model retraining...")

# Feature extraction
def extract_features(audio_path, sr=16000, n_mfcc=13, max_len=32):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T  # (32, 13)

# Load data
X, y = [], []
label_map = {}

speaker_folders = sorted([
    f for f in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, f))
])

print(f"🧑‍🤝‍🧑 Found {len(speaker_folders)} speaker folders.")

if len(speaker_folders) < 2:
    print("⚠️ Not enough users to train the model. Skipping retraining.")
    exit()

for idx, speaker in enumerate(speaker_folders):
    label_map[idx] = speaker
    speaker_path = os.path.join(DATASET_DIR, speaker)
    sample_count = 0

    for root, _, files in os.walk(speaker_path):
        for file in files:
            if not file.endswith(".wav"):
                continue
            full_path = os.path.join(root, file)
            try:
                features = extract_features(full_path)
                X.append(features)
                y.append(idx)
                sample_count += 1
            except Exception as e:
                print(f"❌ Error processing {full_path}: {e}")

    print(f"📁 {speaker} → {sample_count} samples")

# Convert to arrays
X = np.array(X)
y = np.array(y)
print(f"✅ Dataset shape: X={X.shape}, y={y.shape}")

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
inputs = tf.keras.Input(shape=(32, 13))
x = tf.keras.layers.LSTM(128)(inputs)
x = tf.keras.layers.Dense(64, activation='relu', name="embedding_layer")(x)
outputs = tf.keras.layers.Dense(len(label_map), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
print("🧠 Training model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20, batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Save model & label map
model.save(MODEL_SAVE_PATH)
np.save(LABEL_MAP_PATH, label_map)
print("✅ Voice recognition model retrained and saved.")

# ========== Generate embeddings ==========
print("🎯 Generating updated voice embeddings...")

embedding_model = Model(
    inputs=model.input,
    outputs=model.get_layer("embedding_layer").output
)

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

for idx, speaker in label_map.items():
    chunks_dir = os.path.join(DATASET_DIR, speaker, "chunks")
    if not os.path.exists(chunks_dir):
        print(f"⚠️ No 'chunks/' folder for {speaker}, skipping.")
        continue

    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith(".wav")]
    if not chunk_files:
        print(f"⚠️ No chunks found for {speaker}, skipping embedding.")
        continue

    sample_path = os.path.join(chunks_dir, chunk_files[0])
    try:
        sample_features = extract_features(sample_path)
        sample_features = np.expand_dims(sample_features, axis=0)  # (1, 32, 13)
        embedding = embedding_model.predict(sample_features, verbose=0)[0]
        embedding_path = os.path.join(EMBEDDINGS_DIR, f"{speaker}_voice_embedding.npy")
        np.save(embedding_path, embedding)
        print(f"✅ Saved embedding for {speaker}")
    except Exception as e:
        print(f"❌ Failed to create embedding for {speaker}: {e}")
