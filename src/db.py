import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "neuro.db"

def init_db():
    """Creates the table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            patient_id TEXT,
            risk_label TEXT,
            confidence REAL,
            pause_rate REAL,
            vocab_richness REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_result(patient_id, label, conf, pause, vocab):
    """Saves a new scan to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        INSERT INTO records (timestamp, patient_id, risk_label, confidence, pause_rate, vocab_richness)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, patient_id, label, conf, pause, vocab))
    
    conn.commit()
    conn.close()
    print(f"✅ Saved record for {patient_id}")

def get_history(patient_id):
    """Fetches all past records for a patient to plot the graph."""
    conn = sqlite3.connect(DB_NAME)
    # Pandas is great here - it turns SQL directly into a DataFrame for charts
    df = pd.read_sql_query(f"SELECT * FROM records WHERE patient_id='{patient_id}'", conn)
    conn.close()
    return df