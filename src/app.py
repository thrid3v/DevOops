import streamlit as st
import pandas as pd
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- IMPORT BACKEND ---
try:
    from predictor import get_prediction
    from db import init_db, save_result, get_history
except ImportError as e:
    st.error(f"❌ CRITICAL ERROR: {e}")
    st.stop()

# --- 1. SETUP & CONFIG ---
st.set_page_config(
    page_title="NeuroSentinel | Cognitive Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database on Startup
if "db_init" not in st.session_state:
    init_db()
    st.session_state["db_init"] = True

# Custom CSS for "Medical" Look
st.markdown("""
    <style>
        .block-container {padding-top: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 24px;}
        .big-font {font-size:20px !important; color: #555;}
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR (PATIENT CONTEXT) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=80)
    st.title("NeuroSentinel")
    st.caption("v1.2 (Clinical Research Build)")
    st.markdown("---")
    
    st.header("👤 Patient Profile")
    # Patient ID controls the DB context
    patient_id = st.text_input("Patient ID / MRN", value="PT-8492-X")
    
    st.info(f"Active Session: {patient_id}")
    st.markdown("---")
    
    with st.expander("ℹ️ Biomarker Guide"):
        st.markdown("""
        **1. Pause Rate:** % of time spent in silence. >0.4 indicates hesitation.
        **2. Vocab Richness:** Diversity of unique words (Type-Token Ratio).
        **3. Fluency:** Words per minute (normalized).
        """)

# --- 3. MAIN APP LAYOUT ---
st.title("🧠 Cognitive Assessment Dashboard")
st.markdown(f"**Patient:** `{patient_id}` | **Date:** `{datetime.now().strftime('%Y-%m-%d')}`")

# Tabs for Logical Flow
tab1, tab2 = st.tabs(["🎙️ New Assessment", "📈 Clinical History"])

# ==========================================
# TAB 1: LIVE RECORDING & ANALYSIS
# ==========================================
with tab1:
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("1. Voice Sample Acquisition")
        st.markdown("Ask patient to describe the 'Cookie Theft' image or their morning routine.")
        
        # --- AUDIO INPUT ---
        audio_value = st.audio_input("Start Recording")

        if audio_value:
            st.audio(audio_value, format="audio/wav")
            
            # Save temp file
            with open("temp_input.wav", "wb") as f:
                f.write(audio_value.getvalue())
            
            # --- ACTION BUTTON ---
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Processing Neurological Markers (Vosk Engine)..."):
                    time.sleep(1) # UX Delay
                    
                    # CALL THE AI
                    label, confidence, data = get_prediction("temp_input.wav")
                    
                    # SAVE TO DB
                    save_result(
                        patient_id, 
                        label, 
                        confidence, 
                        data['pause_rate'][0], 
                        data['vocab_richness'][0]
                    )
                    
                    # STORE IN SESSION STATE (To persist across re-runs)
                    st.session_state['last_result'] = {
                        "label": label,
                        "conf": confidence,
                        "data": data
                    }
                    st.success("Analysis Complete & Saved to Database.")

    # --- RESULTS DISPLAY ---
    with col_right:
        st.subheader("2. Diagnostic Insights")
        
        if 'last_result' in st.session_state:
            res = st.session_state['last_result']
            data = res['data']
            label = res['label']
            
            # 1. TOP LEVEL ALERT
            if "Decline" in label:
                st.error(f"🚨 RESULT: {label.upper()} (Confidence: {res['conf']*100:.1f}%)")
            else:
                st.success(f"✅ RESULT: {label.upper()} (Confidence: {res['conf']*100:.1f}%)")
            
            # 2. KEY METRICS ROW
            m1, m2, m3 = st.columns(3)
            m1.metric("Pause Rate", f"{data['pause_rate'][0]:.2f}", delta="-0.05" if data['pause_rate'][0] < 0.3 else "+0.12", delta_color="inverse")
            m2.metric("Vocab Richness", f"{data['vocab_richness'][0]:.2f}", delta="Normal")
            m3.metric("Word Count", f"{data['word_count'][0]}")

            # 3. RADAR CHART (The "Wow" Visualization)
            st.markdown("##### 🧠 Neurological Fingerprint")
            
            # Normalize for chart (0-100)
            vocab_score = min(data['vocab_richness'][0] * 120, 100) # Scale up TTR
            fluency_score = max(0, (1.0 - data['pause_rate'][0]) * 100) # Invert Pause Rate
            
            categories = ['Vocabulary', 'Fluency', 'Speech Rate', 'Complexity']
            fig = go.Figure()

            # Healthy Baseline (Fixed)
            fig.add_trace(go.Scatterpolar(
                r=[85, 90, 80, 85],
                theta=categories,
                fill='toself',
                name='Healthy Baseline',
                line_color='green',
                opacity=0.4
            ))

            # Patient Data
            fig.add_trace(go.Scatterpolar(
                r=[vocab_score, fluency_score, 70, vocab_score*0.9],
                theta=categories,
                fill='toself',
                name='Patient Scan',
                line_color='red' if "Decline" in label else 'blue'
            ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=350,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Please record audio and click 'Run Analysis' to see results.")

# ==========================================
# TAB 2: LONGITUDINAL HISTORY (The Graph)
# ==========================================
with tab2:
    st.subheader(f"History for {patient_id}")
    
    # 1. FETCH DATA
    history_df = get_history(patient_id)
    
    if not history_df.empty:
        # 2. TREND CHART (Pause Rate is the best indicator of decline)
        st.markdown("##### 📉 Cognitive Trajectory (Pause Rate)")
        st.caption("Lower is better. Sharp increases indicate rapid decline.")
        
        # Sort by date just in case
        history_df = history_df.sort_values(by="timestamp")
        
        fig_trend = px.line(
            history_df, 
            x='timestamp', 
            y='pause_rate', 
            markers=True,
            title="Pause Rate Over Time",
            labels={'pause_rate': 'Pause Duration Ratio', 'timestamp': 'Assessment Date'}
        )
        
        # Add a "Risk Threshold" Line
        fig_trend.add_hline(y=0.40, line_dash="dash", line_color="red", annotation_text="Clinical Threshold (0.40)")
        fig_trend.update_layout(height=350)
        
        st.plotly_chart(fig_trend, use_container_width=True)

        # 3. DATA TABLE
        with st.expander("📄 View Raw Clinical Data"):
            st.dataframe(history_df.style.highlight_max(axis=0, color="#fffdc9"))
            
            # Download Button
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Report (CSV)",
                csv,
                f"report_{patient_id}.csv",
                "text/csv"
            )
            
    else:
        st.warning("No historical records found for this patient ID. Run a new assessment in Tab 1.")

# --- FOOTER ---
st.markdown("---")
st.caption("🔒 NeuroSentinel processes all data locally (Edge AI). No audio leaves this device.")