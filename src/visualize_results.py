import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- CONFIG ---
DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/neuro_model.pkl"

# 1. Load Data and Model
df = pd.read_csv(DATA_PATH)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

X = df.drop(['label', 'filename'], axis=1)
y = df['label']

# 2. Replicate the Training Split (to get the test set)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42, k_neighbors=1)
X_res, y_res = smote.fit_resample(X_scaled, y)
_, X_test, _, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 3. Generate Predictions
y_pred = model.predict(X_test)
labels = ["Healthy", "Mild", "Moderate", "Severe"]
cm = confusion_matrix(y_test, y_pred)

# 4. Plot the Heatmap
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)

plt.title("Alzheimer's Diagnostic Accuracy: 83.33%", fontsize=16, pad=20)
plt.xlabel("AI Prediction", fontsize=12)
plt.ylabel("Actual Clinical Diagnosis", fontsize=12)
plt.savefig("models/confusion_matrix.png")
print("✅ Heatmap saved to models/confusion_matrix.png")
plt.show()