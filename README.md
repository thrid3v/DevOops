# NeuroSentinel - Cognitive Assessment Dashboard

![Version](https://img.shields.io/badge/version-1.2-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

**NeuroSentinel** is an AI-powered cognitive assessment system that analyzes speech patterns to detect early signs of cognitive decline. The system extracts 9 neurological biomarkers from voice recordings and uses machine learning to classify cognitive health status.

---

## 🚀 Quick Start - How to Run

### Prerequisites
- Python 3.8 or higher
- Microphone access (for live recording)
- ~500MB disk space (for Vosk speech recognition model)

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd hack_sphere
```

2. **Install system dependencies** (required for audio processing)

**macOS:**
```bash
brew install ffmpeg libsndfile
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1
```

**Windows:**
- Download and install [FFmpeg](https://ffmpeg.org/download.html)
- Or use Chocolatey: `choco install ffmpeg`

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch the Application**
```bash
cd src
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

> **Note**: The Vosk speech recognition model, trained ML model, and scaler are already included in the repository under the `models/` directory.

### Optional: Retrain the Model

If you want to retrain the model with new data:

```bash
cd src

# Step 1: Generate synthetic training data
python generate_synthetic_data.py

# Step 2: Train the model
python train_model.py
```

---

## 📊 9-Point Neurological Biomarker Panel

NeuroSentinel analyzes speech using **9 distinct biomarkers** across acoustic, prosodic, and linguistic dimensions:

### 1. **Pause Rate** 
- **Type**: Linguistic Timing
- **Description**: Ratio of pause time to total speech duration
- **Calculation**: `pause_time / total_duration` (pauses > 0.25s)
- **Clinical Significance**: Elevated pause rates (>0.4) indicate hesitations and word-finding difficulties, common in early cognitive decline
- **Healthy Range**: < 0.40

### 2. **Vocabulary Richness (Type-Token Ratio)**
- **Type**: Linguistic Complexity
- **Description**: Ratio of unique words to total words spoken
- **Calculation**: `len(unique_words) / len(total_words)`
- **Clinical Significance**: Reduced lexical diversity suggests semantic memory impairment
- **Healthy Range**: > 0.50

### 3. **Word Count**
- **Type**: Linguistic Fluency
- **Description**: Total number of words spoken in the recording
- **Calculation**: Direct count from speech-to-text transcription
- **Clinical Significance**: Lower word counts may indicate reduced verbal fluency
- **Healthy Range**: > 20 words (for typical assessment)

### 4. **Speech Rate**
- **Type**: Temporal Dynamics
- **Description**: Words spoken per second
- **Calculation**: `word_count / total_duration`
- **Clinical Significance**: Slowed speech rate (<2.0 wps) correlates with processing speed deficits
- **Healthy Range**: > 2.0 words per second

### 5. **Initial Latency**
- **Type**: Response Time
- **Description**: Time delay before first word is spoken
- **Calculation**: Timestamp of first word from recording start
- **Clinical Significance**: Delayed response (>0.5s) indicates cognitive processing delays
- **Healthy Range**: < 0.5 seconds

### 6. **Acoustic Texture (MFCC Mean)**
- **Type**: Acoustic Feature
- **Description**: Mean of first Mel-Frequency Cepstral Coefficient
- **Calculation**: `np.mean(librosa.feature.mfcc(y, sr, n_mfcc=13)[0])`
- **Clinical Significance**: Reflects voice quality and articulatory precision; changes may indicate motor speech disorders
- **Healthy Range**: Baseline comparison (typically > -350)

### 7. **MFCC Delta**
- **Type**: Acoustic Dynamics
- **Description**: Rate of change in MFCCs (spectral dynamics)
- **Calculation**: `np.mean(librosa.feature.delta(mfcc))`
- **Clinical Significance**: High variability suggests unstable articulation; low variability indicates monotonous speech
- **Healthy Range**: Stable values (|delta| < 0.05)

### 8. **Speech Brightness (Spectral Centroid)**
- **Type**: Acoustic Feature
- **Description**: Center of mass of the frequency spectrum
- **Calculation**: `np.mean(librosa.feature.spectral_centroid(y, sr))`
- **Clinical Significance**: Lower values indicate muffled or less energetic speech, associated with reduced vocal effort
- **Healthy Range**: > 1500 Hz

### 9. **Emotional Range (Pitch Variance)**
- **Type**: Prosodic Feature
- **Description**: Variance in fundamental frequency (F0)
- **Calculation**: `np.var(pitch_values)` from `librosa.piptrack()`
- **Clinical Significance**: Reduced pitch variance indicates flat affect and emotional blunting
- **Healthy Range**: > 1,000,000 (high expressiveness)

---

## 🧠 Machine Learning Models

### Primary Classification Model
- **Algorithm**: Random Forest Classifier
- **Implementation**: `sklearn.ensemble.RandomForestClassifier`
- **Training Data**: 2,000 synthetic samples (500 per class)
- **Feature Preprocessing**: StandardScaler normalization
- **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation

#### Optimized Hyperparameters
```python
{
    'n_estimators': [200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}
```

### Speech Recognition Model
- **Model**: Vosk (Kaldi-based ASR)
- **Version**: vosk-model-small-en-us-0.15
- **Purpose**: Speech-to-text transcription for linguistic feature extraction
- **Language**: English (US)
- **Size**: ~40MB

### Feature Extraction Libraries
- **Librosa**: Acoustic and prosodic feature extraction (MFCCs, spectral centroid, pitch tracking)
- **Vosk**: Real-time speech recognition with word-level timestamps
- **NumPy**: Statistical computations

---

## 🎯 Classification Categories

The system classifies cognitive health into **4 categories**:

| Label | Code | Description | Clinical Indicators |
|-------|------|-------------|---------------------|
| **Healthy** | 0 | Normal cognitive function | Low pause rate, high vocab richness, normal speech rate |
| **Mild Decline** | 1 | Early cognitive impairment | Slight hesitations, reduced lexical diversity |
| **Moderate Decline** | 2 | Moderate cognitive impairment | Noticeable pauses, slowed speech, reduced word count |
| **Severe Decline** | 3 | Advanced cognitive impairment | Significant delays, very low fluency, flat prosody |

---

## 📁 Project Structure

```
hack_sphere/
├── src/
│   ├── app.py                      # Main Streamlit application
│   ├── predictor.py                # Prediction engine and feature extraction
│   ├── extract_features.py         # Batch feature extraction from audio files
│   ├── train_model.py              # Model training with hyperparameter tuning
│   ├── generate_synthetic_data.py  # Synthetic data generation
│   ├── db.py                       # SQLite database operations
│   ├── report.py                   # PDF report generation
│   └── visualize_results.py        # Data visualization utilities
├── models/
│   ├── neuro_model.pkl             # Trained Random Forest model
│   ├── scaler.pkl                  # Feature scaler (StandardScaler)
│   └── vosk-model-small-en-us-0.15/  # Speech recognition model
├── data/
│   ├── dataset.csv                 # Original feature dataset
│   ├── synthetic_dataset.csv       # Augmented training data
│   └── [healthy/mild/moderate/severe]/  # Audio file folders
├── neuro.db                        # SQLite database for patient records
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🔬 Technical Workflow

### 1. **Audio Recording**
- User records voice sample via Streamlit audio input
- Audio saved as WAV file (16kHz, mono)

### 2. **Feature Extraction**
```python
# Acoustic features (Librosa)
- MFCC analysis
- Spectral centroid calculation
- Pitch tracking

# Linguistic features (Vosk)
- Speech-to-text transcription
- Word-level timestamps
- Pause detection
```

### 3. **Preprocessing**
- Feature scaling using saved `StandardScaler`
- Normalization ensures balanced feature importance

### 4. **Prediction**
- Random Forest classifier predicts cognitive status
- Confidence score calculated from class probabilities
- Results stored in SQLite database

### 5. **Visualization**
- Radar chart comparing patient data to healthy baseline
- Historical trend analysis (pause rate, word count, speech rate)
- PDF report generation with clinical interpretation

---

## 📈 Model Performance

- **Dataset Accuracy**: Calculated in real-time against local dataset
- **Live Trial Accuracy**: User-confirmed predictions during clinical use
- **Optimization Techniques**:
  - Feature scaling (StandardScaler)
  - Balanced synthetic data generation
  - Hyperparameter tuning via GridSearchCV
  - 5-fold cross-validation

---

## 🗄️ Database Schema

**Table**: `records`

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key (auto-increment) |
| timestamp | TEXT | Assessment date/time |
| patient_id | TEXT | Patient identifier |
| risk_label | TEXT | Classification result |
| confidence | REAL | Prediction confidence (%) |
| pause_rate | REAL | Biomarker value |
| vocab_richness | REAL | Biomarker value |
| word_count | INTEGER | Biomarker value |
| speech_rate | REAL | Biomarker value |
| initial_latency | REAL | Biomarker value |
| acoustic_texture | REAL | Biomarker value |
| mfcc_delta | REAL | Biomarker value |
| speech_brightness | REAL | Biomarker value |
| emotional_range | REAL | Biomarker value |

---

## 🎨 User Interface Features

### Tab 1: New Assessment
- Circular red recording button with live waveform
- One-click analysis with real-time biomarker extraction
- Audio playback for verification

### Tab 2: Biomarker Details
- 9-metric dashboard with color-coded indicators
- Neurological fingerprint radar chart
- Clinical observations in elderly-friendly format
- PDF report download
- Manual accuracy tracking (confirm correct/incorrect)

### Tab 3: Clinical History
- Longitudinal trend graphs for key biomarkers
- Historical data table for patient records
- Risk threshold indicators

---

## 🔧 Dependencies

```txt
streamlit          # Web application framework
pandas             # Data manipulation
plotly             # Interactive visualizations
numpy              # Numerical computations
librosa            # Audio feature extraction
vosk               # Speech recognition
scikit-learn       # Machine learning
fpdf               # PDF report generation
```

---

## 🧪 Development Commands

### Extract Features from Audio Files
```bash
cd src
python extract_features.py
```
Processes audio files in `data/healthy/`, `data/mild/`, `data/moderate/`, `data/severe/` folders.

### Generate Synthetic Training Data
```bash
python generate_synthetic_data.py
```
Creates 2,000 balanced samples (500 per class) based on original dataset statistics.

### Train Model
```bash
python train_model.py
```
Performs hyperparameter tuning and saves optimized model to `models/neuro_model.pkl`.

### Run Application
```bash
streamlit run app.py
```

---

## 📄 PDF Report Generation

Each assessment generates a clinical report containing:
- Patient ID and timestamp
- Risk assessment (color-coded)
- Complete biomarker table with reference ranges
- Clinical interpretation based on findings
- Recommendations for follow-up

---

## 🔒 Privacy & Security

- All processing occurs **locally** (edge AI)
- No data transmitted to external servers
- SQLite database stored locally
- HIPAA-compliant architecture (local deployment)

---

## 🎯 Clinical Use Cases

1. **Early Screening**: Detect subtle cognitive changes before clinical symptoms
2. **Longitudinal Monitoring**: Track cognitive trajectory over time
3. **Treatment Response**: Measure intervention effectiveness
4. **Research**: Collect standardized speech biomarkers for studies

---

## ⚠️ Limitations & Disclaimers

- **Not a diagnostic tool**: For research and screening purposes only
- **Requires clinical validation**: Predictions should be confirmed by healthcare professionals
- **Language limitation**: Currently supports English (US) only
- **Audio quality**: Requires clear recordings in quiet environments
- **Sample size**: Performance depends on training data quality

---

## 🤝 Contributing

To extend the system:
1. Add new audio samples to `data/` folders
2. Run `extract_features.py` to update dataset
3. Generate synthetic data with `generate_synthetic_data.py`
4. Retrain model using `train_model.py`

---

## 📚 References

### Biomarker Research
- Pause rate and cognitive decline: König et al. (2015)
- Lexical diversity in dementia: Fraser et al. (2016)
- Acoustic features in neurodegeneration: Skodda et al. (2011)

### Technologies
- [Librosa Documentation](https://librosa.org/)
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)

---

## 📞 Support

For technical issues or questions:
- **Missing audio libraries**: Ensure `ffmpeg` and `libsndfile` are installed (see Installation Steps)
- **Import errors**: Run `pip install -r requirements.txt` to install all Python dependencies
- **Audio format issues**: Check that files are WAV, 16kHz, mono format
- **Model not found**: Verify the Vosk model exists in `models/vosk-model-small-en-us-0.15/`
- **Runtime errors**: Review terminal logs for detailed error messages

### Common Issues

**"No module named 'librosa'" or audio loading fails:**
```bash
# Install system dependencies first
brew install ffmpeg libsndfile  # macOS
# Then reinstall librosa
pip install --upgrade librosa
```

**"Model not found" error:**
- Ensure you're in the `src/` directory when running the app
- Verify `models/vosk-model-small-en-us-0.15/` exists in the parent directory

---

## 📝 License

MIT License - See LICENSE file for details

---

**Version**: 1.2 (Clinical Research Build)  
**Last Updated**: 2026-02-08  
**System**: NeuroSentinel Edge AI


:0