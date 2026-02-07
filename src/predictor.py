import os
import wave
import json
import pickle
import numpy as np
import pandas as pd
from vosk import Model, KaldiRecognizer
from scipy.io import wavfile

# --- CONFIG ---
VOSK_PATH = "models/vosk-model-small-en-us-0.15"
MODEL_PATH = "models/neuro_model.pkl"

# 1. LOAD MODELS (Global variables to load only once)
print("⏳ Loading AI Models...")
if not os.path.exists(VOSK_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("CRITICAL: Missing Vosk or .pkl model in 'models/' folder.")

vosk_model = Model(VOSK_PATH)

with open(MODEL_PATH, 'rb') as f:
    clf = pickle.load(f)
print("✅ Models Loaded!")

def extract_features_from_audio(file_path):
    """
    The exact same math from Phase 2, but for a single file.
    """
    try:
        wf = wave.open(file_path, "rb")
    except:
        return None

    # Check formatting
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio format mismatch (Must be Mono 16kHz)")
        return None

    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)
    
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0: break
        if rec.AcceptWaveform(data):
            part = json.loads(rec.Result())
            if 'result' in part: results.extend(part['result'])
    
    final_part = json.loads(rec.FinalResult())
    if 'result' in final_part: results.extend(final_part['result'])
    wf.close()
    
    if not results: return None

    # --- MATH ---
    total_duration = results[-1]['end'] - results[0]['start']
    pause_time = 0.0
    for i in range(len(results) - 1):
        gap = results[i+1]['start'] - results[i]['end']
        if gap > 0.25: 
            pause_time += gap
    
    pause_rate = pause_time / total_duration if total_duration > 0 else 0
    
    words = [w['word'] for w in results]
    ttr = len(set(words)) / len(words) if words else 0
    
    # Return as DataFrame (required for Sklearn input)
    return pd.DataFrame([{
        "pause_rate": round(pause_rate, 3),
        "vocab_richness": round(ttr, 3),
        "word_count": len(words)
    }])

def get_prediction(audio_path):
    """
    Returns: (Risk Label, Confidence Score, Feature Data)
    """
    features = extract_features_from_audio(audio_path)
    
    if features is None:
        return "Error", 0.0, None

    # Predict
    prediction_class = clf.predict(features)[0] # 0, 1, 2, or 3
    probabilities = clf.predict_proba(features)[0] # e.g., [0.1, 0.8, 0.05, 0.05]
    confidence = max(probabilities)
    
    labels = {0: "Healthy", 1: "Mild Decline", 2: "Moderate Decline", 3: "Severe Decline"}
    result_text = labels.get(prediction_class, "Unknown")
    
    return result_text, confidence, features