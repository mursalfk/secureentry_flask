# app/voice_chunker.py
import os
import wave
import contextlib

def chunk_audio_file(audio_path, output_dir, chunk_duration=1):
    try:
        os.makedirs(output_dir, exist_ok=True)
        with contextlib.closing(wave.open(audio_path, 'rb')) as wf:
            framerate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            total_frames = wf.getnframes()
            total_duration = total_frames // framerate

            for i in range(0, total_duration):
                wf.setpos(i * framerate)
                frames = wf.readframes(framerate)

                chunk_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_chunk{i}.wav")
                with wave.open(chunk_path, 'wb') as chunk:
                    chunk.setnchannels(n_channels)
                    chunk.setsampwidth(sampwidth)
                    chunk.setframerate(framerate)
                    chunk.writeframes(frames)
    except Exception as e:
        print(f"❌ Failed to chunk {audio_path}: {e}")
