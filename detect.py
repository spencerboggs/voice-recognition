import numpy as np
import librosa
import sounddevice as sd
import joblib
import os

def load_models():
    models = {}
    for filename in os.listdir('models'):
        if filename.endswith('_model.pkl'):
            speaker_name = filename.split('_')[0]
            model_path = os.path.join('models', filename)
            gmm_model = joblib.load(model_path)
            models[speaker_name] = gmm_model
    print("Loaded models:", models)  
    return models

def extract_features(audio):
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def record_audio(duration=10, samplerate=16000):
    recording = np.zeros(int(duration * samplerate), dtype=np.int16)

    def callback(indata, frames, time, status):
        nonlocal recording
        recording[:frames] = indata[:, 0]

    with sd.InputStream(callback=callback, channels=1, samplerate=samplerate, dtype='int16'):
        print("Recording started.")
        sd.sleep(duration * 1000)
    print("Recording finished.")
    return recording

def detect_speaker():
    models = load_models()
    print("Start detecting speaker...")
    print("Listening for speech...")
    
    audio = record_audio(duration=5)
    features = extract_features(audio)
    
    best_score = -np.inf
    best_speaker = None
    
    for speaker, gmm in models.items():
        print(f"Speaker: {speaker}, Model type: {type(gmm)}")  
        score = gmm.score([features])
        if score > best_score:
            best_score = score
            best_speaker = speaker
    
    if best_speaker:
        print(f"Detected speaker: {best_speaker}")
    else:
        print("No match detected.")

if __name__ == "__main__":
    detect_speaker()
