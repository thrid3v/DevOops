import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG ---
DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/neuro_model.pkl"

# 1. Load Data
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found. Did you run extract_features.py?")
    exit()

df = pd.read_csv(DATA_PATH)

# Check if we have enough data
if len(df) < 5:
    print("⚠️ Warning: You have very little data. The model might be unstable.")

# 2. Prepare Features (X) and Labels (y)
# We drop 'filename' (it's text) and 'label' (it's the answer)
X = df.drop(['label', 'filename'], axis=1)
y = df['label']

# 3. Split Data (80% Train, 20% Test)
# random_state=42 ensures we get the same split every time (Reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the "Brain"
print(f"Training on {len(X_train)} samples...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)
print("Detailed Report:")
print(classification_report(y_test, y_pred))

# 6. Feature Importance (For the Judges!)
print("-" * 30)
print("🧠 What matters most?")
importances = model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"   - {feature}: {importance:.2f}")

# 7. Save the Model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
    
print(f"\n💾 Model saved to {MODEL_PATH}")
print("READY FOR PHASE 3 (The App)!")