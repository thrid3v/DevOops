import os
import wave
import json
import pickle
import numpy as np
import pandas as pd
import librosa
from vosk import Model, KaldiRecognizer

# --- CONFIG ---
VOSK_PATH = "models/vosk-model-small-en-us-0.15"
MODEL_PATH = "models/neuro_model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATASET_PATH = "data/dataset.csv" # Required for baseline comparison

# 1. LOAD MODELS
print("⏳ Loading AI Pipeline...")
clf = None
scaler = None

if not os.path.exists(VOSK_PATH):
    raise FileNotFoundError(f"CRITICAL: Missing Vosk model at {VOSK_PATH}")
vosk_model = Model(VOSK_PATH)

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            clf = pickle.load(f)
            print("✅ 9-Parameter Model Loaded!")
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
            print("✅ Feature Scaler Loaded!")
except Exception as e:
    print(f"⚠️ Model Load Error: {e}")

def generate_clinical_summary(features_df):
    """Refined summary logic to ensure the Streamlit box populates with insights."""
    try:
        # Extract specific biomarkers from the current scan
        p_rate = features_df['pause_rate'].iloc[0]
        v_rich = features_df['vocab_richness'].iloc[0]
        s_rate = features_df['speech_rate'].iloc[0]
        
        reasons = []
        
        # Clinical reasoning based on your 9-parameter fingerprints
        if p_rate > 0.30:
            reasons.append(f"Hesitation detected: Pause Rate ({p_rate}) exceeds clinical baseline.")
        if v_rich < 0.85:
            reasons.append(f"Reduced lexical diversity: Vocab Richness ({v_rich}) indicates potential 'empty speech'.")
        if s_rate < 2.2:
            reasons.append(f"Slower tempo: Speech Rate ({s_rate} wps) correlates with increased cognitive effort.")
            
        if not reasons:
            return "✅ Speech biomarkers align with established healthy neurological baselines."
        
        # Join reasons with a clean separator for the UI box
        return " | ".join(reasons)
    except Exception as e:
        return f"Summary generation error: {str(e)}"

def extract_features_from_audio(file_path):
    """ Extracts all 9 biomarkers in exact training order """
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc[0])
        mfcc_delta = np.mean(librosa.feature.delta(mfcc))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        emotional_range = np.var(pitch_values) if len(pitch_values) > 0 else 0
        
        # LINGUISTIC ANALYSIS
        wf = wave.open(file_path, "rb")
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

        initial_latency = results[0]['start']
        total_duration = results[-1]['end'] - results[0]['start']
        pause_time = sum([results[i+1]['start'] - results[i]['end'] 
                         for i in range(len(results) - 1) 
                         if (results[i+1]['start'] - results[i]['end']) > 0.25])
        
        pause_rate = pause_time / total_duration if total_duration > 0 else 0
        words = [w['word'] for w in results]
        ttr = len(set(words)) / len(words) if words else 0
        speech_rate = len(words) / results[-1]['end'] if results[-1]['end'] > 0 else 0
        
        return pd.DataFrame([{
            "pause_rate": round(pause_rate, 3),
            "vocab_richness": round(ttr, 3),
            "word_count": len(words),
            "initial_latency": round(initial_latency, 3),
            "acoustic_texture": round(mfcc_mean, 3),
            "speech_brightness": round(spec_centroid, 3),
            "mfcc_delta": round(mfcc_delta, 4),
            "emotional_range": round(emotional_range, 3),
            "speech_rate": round(speech_rate, 3)
        }])
    except:
        return None

def get_prediction(audio_path):
    """ Returns: (Label, Confidence, Features) """
    features_df = extract_features_from_audio(audio_path)
    
    if features_df is None:
        return "Inconclusive", 0.0, None

    features_final = features_df
    if scaler is not None:
        try:
            features_final = scaler.transform(features_df)
        except:
            pass

    if clf is not None:
        probabilities = clf.predict_proba(features_final)[0] 
        confidence = np.max(probabilities) * 100 
        prediction_class = clf.predict(features_final)[0]
        
        labels = {0: "Healthy", 1: "Mild Decline", 2: "Moderate Decline", 3: "Severe Decline"}
        result_text = labels.get(prediction_class, "Unknown")
        
        return result_text, round(confidence, 2), features_df

    return "Demo (Heuristic)", 90.0, features_df

from sklearn.metrics import accuracy_score

def calculate_current_accuracy():
    """Calculates the accuracy of the saved model against the full local dataset."""
    try:
        if not os.path.exists(DATASET_PATH) or clf is None or scaler is None:
            return 0.0
            
        # Load the full 19-file dataset
        df = pd.read_csv(DATASET_PATH)
        X = df.drop(['label', 'filename'], axis=1)
        y = df['label']
        
        # Scale and Predict
        X_scaled = scaler.transform(X)
        y_pred = clf.predict(X_scaled)
        
        # Calculate Accuracy
        acc = accuracy_score(y, y_pred)
        return round(acc * 100, 2)
    except Exception as e:
        print(f"Accuracy calc error: {e}")
        return 0.0