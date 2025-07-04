import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import soundfile as sf

VOICE_DATA_DIR = os.path.join("app", "voice_data")
MODEL_OUTPUT_PATH = os.path.join("app", "models", "voice-recognition-model.keras")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfcc.shape[1] < 32:
        pad_width = 32 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :32]
    return mfcc.T  # Shape: (32, 13)

def retrain_model():
    print("🚀 Starting voice model retraining...")

    X, y, label_to_user = [], [], {}
    users = sorted(os.listdir(VOICE_DATA_DIR))
    for idx, user in enumerate(users):
        chunk_dir = os.path.join(VOICE_DATA_DIR, user, "chunks")
        if not os.path.exists(chunk_dir): continue
        label_to_user[idx] = user

        for filename in os.listdir(chunk_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(chunk_dir, filename)
                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(idx)
                except Exception as e:
                    print(f"❌ Skipping {file_path} due to error: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"✅ Dataset shape: X={X.shape}, y={y.shape}")
    if len(np.unique(y)) < 2:
        print("⚠️ Not enough users to train the model. Skipping retraining.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a simple LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20,
                        batch_size=32,
                        callbacks=[early_stop],
                        verbose=1)

    model.save(MODEL_OUTPUT_PATH)
    print(f"✅ Retrained model saved at: {MODEL_OUTPUT_PATH}")
