# Standard libraries
import os
import shutil

# Data manipulation
import numpy as np
import pandas as pd

# Audio processing
import librosa
import soundfile as sf

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Deep learning
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from flask_login import login_required, current_user

def retrain_model():
   print("🚀 Starting voice model retraining...")

   INPUT_DIR = "./data/input"
   OUTPUT_DIR = "./data/output"
   
   # Output directory to clear
   output_dir = OUTPUT_DIR

   # Clear the contents of the output directory
   shutil.rmtree(output_dir, ignore_errors=True)
   os.makedirs(output_dir, exist_ok=True)

   print(f"Contents of {output_dir} cleared.")
   username = current_user.username.replace(" ", "_")
   
   path_temp_dir = os.path.join("app", "dataset", "voice_data", f"{username}", "temp")
   
   # Get the list of files in the temp directory
   temp_files = os.listdir(path_temp_dir)
   
   # get the latest file based on the timestamp in the filename
   latest_file = max(temp_files, key=lambda x: os.path.getctime(os.path.join(path_temp_dir, x)))
   latest_file_path = os.path.join(path_temp_dir, latest_file)
   print(f"Latest file in temp directory: {latest_file_path}")

   test_audio_path = latest_file_path

   bg_noise_path = os.path.join("app", "dataset", "_background_noise_", "running_tap.wav")

   test_audio, sr = librosa.load(test_audio_path, sr=16000)
   noise, _ = librosa.load(bg_noise_path, sr=sr)
   repeat_times = int(np.ceil(len(test_audio) / len(noise)))
   extended_noise = np.tile(noise, repeat_times)[:len(test_audio)]
   mixed = 0.8 * test_audio + 0.5 * extended_noise


   chunk_duration = 1  # seconds
   chunk_samples = sr * chunk_duration


   chunks = [mixed[i:i + chunk_samples] for i in range(0, len(mixed), chunk_samples)]


   print(f"Number of chunks before padding: {len(chunks)}")


   output_dir = os.path.join("app", "dataset", "noise", "Mursal_Furqan")
   os.makedirs(output_dir, exist_ok=True)


   for i, chunk in enumerate(chunks):
      # Pad with zeros if chunk is shorter than 1 second
      if len(chunk) < chunk_samples:
         pad_width = chunk_samples - len(chunk)
         chunk = np.pad(chunk, (0, pad_width), mode='constant')

      output_path = os.path.join(output_dir, f"{i}.wav")
      sf.write(output_path, chunk, sr)


   # Path to the dataset
   dataset_path = os.path.join("app", "dataset", "voice_data")


   # Output directory to save the combined files
   output_dir = os.path.join("app", "dataset", "noise")


   # Create the output directory if it doesn't exist
   os.makedirs(output_dir, exist_ok=True)


   # # List of speaker folders
   # speaker_folders = [
   #    "Benjamin_Netanyau",
   #    "Jens_Stoltenberg",
   #    "Julia_Gillard",
   #    "Magaret_Tarcher",
   #    "Nelson_Mandela"
   # ]


   # for speaker_folder in speaker_folders:
   #    speaker_folder_path = speaker_folder_path = os.path.join(dataset_path, speaker_folder)
   #    wav_files = [f for f in os.listdir(speaker_folder_path)]
   #    for wav_file in wav_files:
   #       wav_file_path = os.path.join(speaker_folder_path, wav_file)
   #       audio, sr = librosa.load(wav_file_path, sr=None)
   #       noise, _ = librosa.load(bg_noise_path, sr=sr)
   #       repeat_times = int(np.ceil(len(audio) / len(noise)))
   #       extended_noise = np.tile(noise, repeat_times)[:len(audio)]
         
   #       mixed = 0.8 * audio + 0.5 * extended_noise


   #       os.makedirs(os.path.join(output_dir,f"{speaker_folder}"), exist_ok=True)


   #       output_file_path = os.path.join(output_dir, f"{speaker_folder}", wav_file.split(".")[0]+".wav")
   #       sf.write(output_file_path, mixed, sr)


   # Convert data to MFCC
   # Set the parent directory for speaker folders
   parent_dir = os.path.join("app", "dataset", "noise")


   # List of speaker folders
   speaker_folders = [
      "Benjamin_Netanyau",
      "Jens_Stoltenberg",
      "Julia_Gillard",
      "Magaret_Tarcher",
      "Nelson_Mandela",
      "Mursal_Furqan"
   ]


   def extract_features(parent_dir, speaker_folders):
      features = []
      labels = []


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
                  labels.append(i)


      return np.array(features), np.array(labels)

   # Extract features and labels
   X, y = extract_features(parent_dir, speaker_folders)

   # Lable Encoding
   # Encode labels with explicit classes
   label_encoder = LabelEncoder()
   y = label_encoder.fit_transform(y)
   label_encoder.classes_ = np.array(speaker_folders)

   # Split the data into training, validation, and test sets
   X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
   X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

   # Print the shapes of training and validation data
   print("Training Data Shape:", X_train.shape)
   print("Validation Data Shape:", X_val.shape)

   # Define the RNN model
   model = tf.keras.Sequential([
      tf.keras.layers.LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(len(speaker_folders), activation='softmax')
   ])

   # Compile the model
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

   # Define the EarlyStopping callback
   early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

   # Train the model with EarlyStopping
   history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[early_stopping])

   # Check if EarlyStopping triggered
   if early_stopping.stopped_epoch > 0:
      print("Early stopping triggered at epoch", early_stopping.stopped_epoch + 1)
   else:
      print("Training completed without early stopping")

   # Saving the model
   model_output_path = os.path.join("app", "models", "voice-recognition-model.keras")
   model.save(model_output_path)



# import os
# import numpy as np
# import librosa
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping
# import soundfile as sf

# VOICE_DATA_DIR = os.path.join("app", "dataset", "voice_data")
# MODEL_OUTPUT_PATH = os.path.join("app", "models", "voice-recognition-model.keras")

# def extract_features(audio_path):
#     y, sr = librosa.load(audio_path, sr=16000)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     if mfcc.shape[1] < 32:
#         pad_width = 32 - mfcc.shape[1]
#         mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mfcc = mfcc[:, :32]
#     return mfcc.T  # Shape: (32, 13)

# def retrain_model():
#     print("🚀 Starting voice model retraining...")
#     X, y, label_to_user = [], [], {}
#     users = sorted(os.listdir(VOICE_DATA_DIR))
#     for idx, user in enumerate(users):
#         chunk_dir = os.path.join(VOICE_DATA_DIR, user, "chunks")
#         if not os.path.exists(chunk_dir): continue
#         label_to_user[idx] = user

#         for filename in os.listdir(chunk_dir):
#             if filename.endswith(".wav"):
#                 file_path = os.path.join(chunk_dir, filename)
#                 try:
#                     features = extract_features(file_path)
#                     X.append(features)
#                     y.append(idx)
#                 except Exception as e:
#                     print(f"❌ Skipping {file_path} due to error: {e}")

#     # USer's AUdio HERE
#     # test_audio_path = INPUT_DIR + "/test-data/WhatsApp Audio 2025-04-23 at 15.57.52.wav"

#     # # USer's AUdio HERE
#     # bg_noise_path = INPUT_DIR+ "/speaker-recognition-dataset/16000_pcm_speeches/_background_noise_/running_tap.wav"

#     # test_audio, sr = librosa.load(test_audio_path, sr=16000)
#     # noise, _ = librosa.load(bg_noise_path, sr=sr)
#     # repeat_times = int(np.ceil(len(test_audio) / len(noise)))
#     # extended_noise = np.tile(noise, repeat_times)[:len(test_audio)]
#     # mixed = 0.8 * test_audio + 0.5 * extended_noise
    
#     X = np.array(X)
#     y = np.array(y)

#     print(f"✅ Dataset shape: X={X.shape}, y={y.shape}")
#     if len(np.unique(y)) < 2:
#         print("⚠️ Not enough users to train the model. Skipping retraining.")
#         return

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     num_classes = len(np.unique(y))

#     # Build a simple LSTM model
#     inputs = tf.keras.Input(shape=(32, 13))
#     x = tf.keras.layers.LSTM(128)(inputs)
#     x = tf.keras.layers.Dense(64, activation='relu', name="embedding_layer")(x)
#     outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
#     model = tf.keras.Model(inputs, outputs)
    
#     # model = tf.keras.models.load_model(MODEL_OUTPUT_PATH)
#     # model = tf.keras.load_model(MODEL_OUTPUT_PATH)
#     # print("🔄 Loaded existing model for retraining.")

#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
#     history = model.fit(X_train, y_train,
#                         validation_data=(X_val, y_val),
#                         epochs=20,
#                         batch_size=32,
#                         callbacks=[early_stop],
#                         verbose=1)

#     model.save(MODEL_OUTPUT_PATH)
#     print(f"✅ Retrained model saved at: {MODEL_OUTPUT_PATH}")
