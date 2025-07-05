
import tensorflow as tf
import numpy as np
import os
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from sklearn.preprocessing import StandardScaler

# Paths
# MODEL_PATH = os.path.join("app", "models", "voice-recognition-model.keras")
# EMBEDDINGS_DIR = os.path.join("app", "models", "voice_embeddings")

# # Load the saved model and warm it up
# base_model = tf.keras.models.load_model(MODEL_PATH)
# _ = base_model(np.zeros((1, 32, 13), dtype=np.float32))  # warmup call

# # Now extract the correct embedding layer
# embedding_model = tf.keras.Model(
#     inputs=base_model.input,
#     outputs=base_model.get_layer("embedding_layer").output
# )

def extract_features(parent_dir, speaker_folders):
    features = []
    for i, speaker_folder in enumerate(speaker_folders):
        speaker_folder_path = os.path.join(parent_dir, speaker_folder)
        print(i, speaker_folder)

        for filename in os.listdir(speaker_folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(speaker_folder_path, filename)
                audio, sr = librosa.load(file_path, sr=None, duration=1)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                
                # Normalize MFCC features
                mfccs = StandardScaler().fit_transform(mfccs)
                
                features.append(mfccs.T)

    return np.array(features)



    # y, sr = librosa.load(file_path, sr=16000, duration=1)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # mfcc = StandardScaler().fit_transform(mfcc)
    # if mfcc.shape[1] < 32:
    #     pad_width = 32 - mfcc.shape[1]
    #     mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    # else:
    #     mfcc = mfcc[:, :32]
    # return mfcc.T  # (32, 13)

import tensorflow as tf
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import soundfile as sf

# 🔐 Predict speaker using cosine similarity
def predict_user(audio_path):

    speaker_folders = [
        "Benjamin_Netanyau",
        "Jens_Stoltenberg",
        "Julia_Gillard",
        "Magaret_Tarcher",
        "Nelson_Mandela",
        "Mursal_Furqan",
    ]

    test_audio_path = os.path.join(audio_path)


    test_audio, sr = librosa.load(test_audio_path, sr=16000)


    chunk_duration = 1  # seconds
    chunk_samples = sr * chunk_duration


    chunks = [test_audio[i:i + chunk_samples] for i in range(0, len(test_audio), chunk_samples)]


    print(f"Number of chunks before padding: {len(chunks)}")


    output_dir = os.path.join("app", "dataset",  "test", "noise", "Mursal_Furqan")
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
    # Pad with zeros if chunk is shorter than 1 second
        if len(chunk) < chunk_samples:
            pad_width = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), mode='constant')


        output_path = os.path.join(output_dir, f"{i}.wav")
        sf.write(output_path, chunk, sr)
        print(f"Saved: {output_path}")
        
    

    features = extract_features(os.path.join("app", "dataset",  "test", "noise"), ["Mursal_Furqan"])

    model_path = os.path.join("app", "models", "voice-recognition-model.keras")
    model = tf.keras.models.load_model(model_path)

    probabilites = model.predict(features)
    predictions = np.argmax(probabilites, axis=1)
    print('----------------------------------')
    predicted_class = np.bincount(predictions).argmax()
    print(predictions)
    print(speaker_folders[predicted_class])
    print(predicted_class)
    print('----------------------------------')
    
    return predicted_class

    # try:
    #     features = extract_features(audio_path)
    #     print("📦 Features shape:", features.shape)

    #     # Get embedding
    #     features = np.expand_dims(features, axis=0)
    #     voice_embedding = embedding_model.predict(features, verbose=0)[0]
    #     print("🧠 Voice embedding shape:", voice_embedding.shape)

    #     best_match = None
    #     highest_similarity = -1
    #     threshold = 0.80

    #     for filename in os.listdir(EMBEDDINGS_DIR):
    #         if filename.endswith(".npy"):
    #             username = filename.replace("_voice_embedding.npy", "")
    #             saved_embedding = np.load(os.path.join(EMBEDDINGS_DIR, filename))

    #             print(f"🔍 Comparing with {username}")
    #             print(f"🔸 Saved embedding shape: {saved_embedding.shape}")

    #             similarity = cosine_similarity([voice_embedding], [saved_embedding])[0][0]
    #             print(f"📊 Similarity with {username}: {similarity:.4f}")

    #             if similarity > highest_similarity and similarity >= threshold:
    #                 highest_similarity = similarity
    #                 best_match = username

    #     return best_match

    # except Exception as e:
    #     print("🚨 Voice prediction error:")
    #     traceback.print_exc()
    #     return None