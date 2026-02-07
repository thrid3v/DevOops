import os
import json
import wave
import pandas as pd
import numpy as np
import librosa
from vosk import Model, KaldiRecognizer

# --- CONFIGURATION ---
MODEL_PATH = "models/vosk-model-small-en-us-0.15" 
DATA_DIR = "data"
OUTPUT_CSV = "data/dataset.csv"

# --- 1. LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    print(f"❌ CRITICAL ERROR: Model not found at {MODEL_PATH}")
    exit(1)

print("Loading Vosk Model...")
model = Model(MODEL_PATH)

def analyze_audio(file_path):
    try:
        # PART A: ACOUSTIC & PROSODIC ANALYSIS
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc[0])
        mfcc_delta = np.mean(librosa.feature.delta(mfcc))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        emotional_range = np.var(pitch_values) if len(pitch_values) > 0 else 0
        
        # PART B: LINGUISTIC TIMING
        wf = wave.open(file_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            wf.close()
            return None
            
        rec = KaldiRecognizer(model, wf.getframerate())
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

        # FEATURE EXTRACTION
        initial_latency = results[0]['start']
        total_duration = results[-1]['end'] - results[0]['start']
        pause_time = sum([results[i+1]['start'] - results[i]['end'] 
                         for i in range(len(results) - 1) 
                         if (results[i+1]['start'] - results[i]['end']) > 0.25])
        
        pause_rate = pause_time / total_duration if total_duration > 0 else 0
        words = [w['word'] for w in results]
        ttr = len(set(words)) / len(words) if words else 0
        speech_rate = len(words) / results[-1]['end'] if results[-1]['end'] > 0 else 0
        
        return {
            "pause_rate": round(pause_rate, 3),
            "vocab_richness": round(ttr, 3),
            "word_count": len(words),
            "initial_latency": round(initial_latency, 3),
            "acoustic_texture": round(mfcc_mean, 3),
            "speech_brightness": round(spec_centroid, 3),
            "mfcc_delta": round(mfcc_delta, 4),
            "emotional_range": round(emotional_range, 3),
            "speech_rate": round(speech_rate, 3)
        }
    except Exception as e:
        return None

# --- 2. PROCESSING LOOP WITH TRACKING ---
data_records = []
labels_map = {"healthy": 0, "mild": 1, "moderate": 2, "severe": 3}
scan_stats = {label: 0 for label in labels_map.keys()} 

print(f"\n📂 Scanning '{DATA_DIR}' for audio files...")

for folder_name, label_code in labels_map.items():
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        print(f"⚠️  Skipping missing folder: {folder_name}")
        continue
        
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    print(f"📁 Processing '{folder_name}' ({len(files)} files found)...")
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        features = analyze_audio(file_path)
        if features:
            features['label'] = label_code
            features['filename'] = filename
            data_records.append(features)
            scan_stats[folder_name] += 1
            print(f"   ✅ {filename} processed.")
        else:
            print(f"   ❌ {filename} FAILED (Check Mono/16kHz format).")

# --- 3. FINAL SUMMARY TABLE ---
print("\n" + "="*40)
print(f"{'CATEGORY':<15} | {'FILES PROCESSED':<15}")
print("-" * 40)
for label, count in scan_stats.items():
    print(f"{label.capitalize():<15} | {count:<15}")
print("-" * 40)
print(f"{'TOTAL':<15} | {len(data_records):<15} / 19 Expected")
print("="*40 + "\n")

if data_records:
    df = pd.DataFrame(data_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"🚀 SUCCESS! Dataset saved with all available parameters.")
else:
    print("🛑 ERROR: No data extracted. Ensure files are in the data/ subfolders.")