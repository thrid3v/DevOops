import streamlit as st
import os
import time
from predictor import get_prediction # Import our engine

# Page Config
st.set_page_config(page_title="NeuroSentinel", page_icon="🧠", layout="wide")

# --- UI HEADER ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2814/2814666.png", width=80)
with col2:
    st.title("NeuroSentinel")
    st.markdown("**Early detection of cognitive decline via linguistic biomarkers.**")

st.markdown("---")

# --- SIDEBAR (Patient Context) ---
with st.sidebar:
    st.header("Patient Profile")
    st.text_input("Patient ID", value="8492-X")
    st.date_input("Date of Birth")
    st.info("Status: Monitoring Recommended")
    
    st.divider()
    st.caption("NeuroSentinel v1.0 (Hackathon Build)")

# --- MAIN AREA ---
st.subheader("📝 Cognitive Stress Test")
st.write("Please read the 'Cookie Theft' description or describe your day.")

# THE MAGIC MIC
audio_value = st.audio_input("Record Voice Sample")

if audio_value:
    st.audio(audio_value)
    
    # 1. Save temp file for processing
    with open("temp_input.wav", "wb") as f:
        f.write(audio_value.getvalue())
    
    # 2. Analyze
    with st.spinner("Analyzing neural speech patterns..."):
        time.sleep(1) # Fake delay for dramatic effect
        label, confidence, data = get_prediction("temp_input.wav")

    # 3. Show Results
    if data is not None:
        # Create 3 Columns for metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Pause Rate", f"{data['pause_rate'][0]}")
        m2.metric("Vocab Richness", f"{data['vocab_richness'][0]}")
        m3.metric("Word Count", f"{data['word_count'][0]}")
        
        st.divider()
        
        # Big Result Banner
        if label == "Healthy":
            st.success(f"✅ Analysis Result: {label} ({(confidence*100):.1f}%)")
        elif "Mild" in label:
            st.warning(f"⚠️ Analysis Result: {label} ({(confidence*100):.1f}%)")
        else:
            st.error(f"🚨 Analysis Result: {label} ({(confidence*100):.1f}%)")
            
        st.info("Clinical Note: Elevated pause rates detected in conjunction with simplified vocabulary.")
        
    else:
        st.error("Could not analyze audio. Please try speaking longer.")