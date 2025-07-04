import os

DATASET_DIR = "app/voice_data"

print("🧑‍🤝‍🧑 Total speaker folders:", len(os.listdir(DATASET_DIR)))
print()

for speaker in sorted(os.listdir(DATASET_DIR)):
    speaker_path = os.path.join(DATASET_DIR, speaker)
    if not os.path.isdir(speaker_path):
        continue

    wav_count = 0
    for root, _, files in os.walk(speaker_path):
        wav_files = [f for f in files if f.endswith('.wav')]
        wav_count += len(wav_files)

    print(f"📁 {speaker} → {wav_count} files")
