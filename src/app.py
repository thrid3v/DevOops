import streamlit as st
import pandas as pd
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from report import generate_pdf
from datetime import datetime

# --- IMPORT BACKEND ---
try:
    from predictor import get_prediction, generate_clinical_summary, calculate_current_accuracy
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

if "db_init" not in st.session_state:
    init_db()
    st.session_state["db_init"] = True

current_accuracy = calculate_current_accuracy()

# Custom CSS for "Medical" Look
st.markdown("""
    <style>
        .block-container {padding-top: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 24px;}
        .report-box {background-color: white; color: black; padding: 15px; border-left: 5px solid #ff4b4b; border-radius: 5px; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=80)
    st.title("NeuroSentinel")
    st.caption("v1.2 (Clinical Research Build)")
    st.markdown("---")
    
    st.header("👤 Patient Profile")
    patient_id = st.text_input("Patient ID / MRN", value="PT-8492-X")
    st.info(f"Active Session: {patient_id}")
    st.markdown("---")
    
    if st.button("Reset Live Trial Stats"):
        st.session_state.session_correct = 0
        st.session_state.session_total = 0
        st.rerun()
    
    st.subheader("📊 System Benchmarks")
    st.metric("Live Dataset Accuracy", f"{current_accuracy}%", delta="Real-time")

# --- 3. MAIN APP LAYOUT ---
st.title("🧠 Cognitive Assessment Dashboard")
st.markdown(f"**Patient:** `{patient_id}` | **Date:** `{datetime.now().strftime('%Y-%m-%d')}`")

tab1, tab2 = st.tabs(["🎙️ New Assessment", "📈 Clinical History"])

# ==========================================
# TAB 1: LIVE RECORDING & ANALYSIS
# ==========================================
with tab1:
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("1. Voice Sample Acquisition")
        st.markdown("Ask patient to describe the 'Cookie Theft' image.")
        
        audio_input = st.audio_input("Start Recording")

        if audio_input:
            st.audio(audio_input, format="audio/wav")
            with open("temp_input.wav", "wb") as f:
                f.write(audio_input.getvalue())
            
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Extracting 9 Neurological Biomarkers..."):
                    label, confidence, data = get_prediction("temp_input.wav")
                    
                    if data is not None:
                        save_result(patient_id, label, confidence, 
                                    data['pause_rate'][0], data['vocab_richness'][0])
                        
                        report_path = generate_pdf(
                            patient_id, 
                            {"pause_rate": data['pause_rate'][0], 
                             "vocab_richness": data['vocab_richness'][0], 
                             "word_count": data['word_count'][0]},
                            label
                        )
                        
                        st.session_state['last_result'] = {
                            "label": label, "conf": confidence, "data": data,
                            "summary": generate_clinical_summary(data),
                            "report_path": report_path
                        }

    with col_right:
        st.subheader("2. Diagnostic Insights")
        
        if 'last_result' in st.session_state:
            res = st.session_state['last_result']
            data, label = res['data'], res['label']
            
            # --- Result Alert ---
            if "Decline" in label:
                st.error(f"🚨 RESULT: {label.upper()} (Confidence: {res['conf']:.1f}%)")
            else:
                st.success(f"✅ RESULT: {label.upper()} (Confidence: {res['conf']:.1f}%)")
            
            # --- Clinical Reasoning ---
            st.markdown(f"""<div class="report-box"><strong>Clinical Observations:</strong><br>{res['summary']}</div>""", unsafe_allow_html=True)

            # --- [UPDATED] METRICS ROW WITH TRIAL ACCURACY ---
            st.markdown("##### 🧪 Performance & Metrics")
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            
            m_c1.metric("Pause Rate", f"{data['pause_rate'][0]:.2f}", 
                        delta="Risk" if data['pause_rate'][0] > 0.4 else "Normal", delta_color="inverse")
            m_c2.metric("Vocab", f"{data['vocab_richness'][0]:.2f}")
            m_c3.metric("Speech Rate", f"{data['speech_rate'][0]:.1f} wps")
            
            # Calculate Trial Accuracy live
            correct = st.session_state.get('session_correct', 0)
            total = st.session_state.get('session_total', 0)
            trial_acc = (correct / total * 100) if total > 0 else 0.0
            m_c4.metric("Trial Accuracy", f"{trial_acc:.1f}%", delta=f"{total} Tests")

            # --- ACTIONS & PDF ---
            st.markdown("---")
            st.caption("👨‍⚕️ Judge's Verdict: Confirm diagnosis to update live session stats.")
            v1, v2, v3 = st.columns([1, 1, 1.5])
            with v1:
                if st.button("✅ Confirm Correct"):
                    st.session_state.session_correct = st.session_state.get('session_correct', 0) + 1
                    st.session_state.session_total = st.session_state.get('session_total', 0) + 1
                    st.rerun()
            with v2:
                if st.button("❌ Mark Incorrect"):
                    st.session_state.session_total = st.session_state.get('session_total', 0) + 1
                    st.rerun()
            with v3:
                if os.path.exists(res['report_path']):
                    with open(res['report_path'], "rb") as f:
                        st.download_button("📄 Download PDF Report", f, 
                                         file_name=f"NeuroSentinel_{patient_id}.pdf")

            # --- Radar Chart ---
            st.markdown("##### 🧠 9-Parameter Neurological Fingerprint")
            categories = ['Pause Rate', 'Vocab', 'Texture', 'Brightness', 'Delta', 'Emotional Range', 'Speech Rate', 'Latency', 'Count']
            r_vals = [(1-data['pause_rate'][0])*100, data['vocab_richness'][0]*100, abs(data['acoustic_texture'][0]), 
                      (data['speech_brightness'][0]/4000)*100, abs(data['mfcc_delta'][0])*500, (data['emotional_range'][0]/5000)*100,
                      (data['speech_rate'][0]/3)*100, (1/max(1, data['initial_latency'][0]))*100, min(data['word_count'][0], 100)]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=r_vals, theta=categories, fill='toself', name='Patient Scan', line_color='red' if "Decline" in label else 'blue'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=400, margin=dict(l=50, r=50, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Record audio and click 'Run Analysis' to see results.")

# ==========================================
# TAB 2: HISTORY
# ==========================================
with tab2:
    st.subheader(f"History for {patient_id}")
    h_df = get_history(patient_id)
    if not h_df.empty:
        fig_t = px.line(h_df, x='timestamp', y='pause_rate', markers=True, title="Pause Rate Trajectory")
        fig_t.add_hline(y=0.40, line_dash="dash", line_color="red", annotation_text="Risk Threshold")
        st.plotly_chart(fig_t, use_container_width=True)
        st.dataframe(h_df)
    else:
        st.warning("No historical records found.")

st.markdown("---")
st.caption(f"🔒 NeuroSentinel Edge AI | System-Wide Accuracy: {current_accuracy}%")