import os
import numpy as np
import librosa
import joblib
from sklearn.mixture import GaussianMixture
import sounddevice as sd
import time

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
        print("Recording started. Press Enter when you're done.")
        start_time = time.time()
        while True:
            if time.time() - start_time > duration:
                break
            if input() == '':
                break
        print("Recording finished.")
    return recording

def train_model():
    speakers = {}
    predefined_sentence = "This is a test sentence to help the system recognize your voice. Please speak clearly and distinctly for best results."

    print("Start recording and labeling speakers...")
    speaker_name = input("Enter speaker's name: ")

    features_list = []

    for i in range(5):
        print(f"Please ask {speaker_name} to read the following sentence aloud:")
        print(f"\"{predefined_sentence}\"")
        input(f"Press Enter when ready to start recording (Recording {i + 1}/5)...")

        print(f"Recording {speaker_name}'s voice...")
        audio = record_audio(duration=10)
        
        features = extract_features(audio)
        features_list.append(features)

    features_array = np.array(features_list)

    gmm = GaussianMixture(n_components=2, covariance_type='diag', reg_covar=1e-4)
    gmm.fit(features_array)
    
    if not os.path.exists('models'):
        os.makedirs('models')

    model_filename = f"models/{speaker_name}_model.pkl"
    joblib.dump(gmm, model_filename)
    print(f"Training complete and model saved for {speaker_name} at {model_filename}.")

if __name__ == "__main__":
    train_model()
