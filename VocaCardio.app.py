pip install streamlit scipy numpy pandas
import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
import time

# =====================================================================
# 🧠 LAYER 1: BACKEND MEDICAL SIGNAL PROCESSING ENGINE
# =====================================================================
class VoiceBiometricEngine:
    def __init__(self, sampling_rate: int = 22050):
        self.sr = sampling_rate

    def analyze_audio_stream(self, waveform: np.ndarray) -> dict:
        """
        Extracts micro-timing fluctuations (Jitter) and micro-amplitude 
        power shifts (Shimmer) to quantify structural vocal cord stability.
        """
        # Protect against empty array allocations or flat noise spikes
        if np.max(np.abs(waveform)) < 1e-4:
            return {"jitter": 0.001, "shimmer": 0.010, "instability_score": 5.0}

        # Normalize the signal matrix
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-6)
        
        # Cross-autocorrelation peak matching to capture fundamental pitch frequencies (F0)
        corr = signal.correlate(waveform, waveform, mode='full')
        corr = corr[len(corr)//2:]
        
        # Map indices directly bounding average human limits (80Hz - 300Hz)
        min_peak_dist = int(self.sr / 300)
        max_peak_dist = int(self.sr / 80)
        
        peaks, _ = signal.find_peaks(corr[min_peak_dist:max_peak_dist], prominence=0.05)
        peaks = peaks + min_peak_dist
        
        if len(peaks) < 3:
            return {"jitter": 0.002, "shimmer": 0.012, "instability_score": 8.5}
            
        # 1. Calculate Jitter (Period-to-period micro-timing variability)
        periods = np.diff(peaks) / self.sr
        jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
        
        # 2. Calculate Shimmer (Period-to-period micro-amplitude power deviations)
        peak_amplitudes = waveform[peaks]
        shimmer = float(np.mean(np.abs(np.diff(peak_amplitudes))) / np.mean(peak_amplitudes))
        
        # Synthesize a unified visual score bounded between 0% and 100%
        instability_score = min(float((jitter * 750) + (shimmer * 150)), 100.0)
        
        return {
            "jitter": jitter,
            "shimmer": shimmer,
            "instability_score": round(instability_score, 2)
        }

# =====================================================================
# 📊 LAYER 2: CLINICAL DATABASE STORAGE MOCKUP
# =====================================================================
class PatientDatabase:
    @staticmethod
    def get_demographics(patient_id: str) -> dict:
        """Returns baseline diagnostic information for selected profile."""
        profiles = {
            "PT-8802": {
                "name": "Eleanor Vance", "age": 67, 
                "condition": "Early-Stage Parkinson's / Chronic Asthma", 
                "tier": "High Risk Monitoring"
            },
            "PT-4419": {
                "name": "Marcus Chen", "age": 42, 
                "condition": "Post-Viral Respiratory Recovery Pipeline", 
                "tier": "Routine Screening"
            }
        }
        return profiles.get(patient_id, {"name": "Unknown", "age": 0, "condition": "N/A", "tier": "N/A"})

    @staticmethod
    def generate_longitudinal_records(patient_id: str, days: int = 30) -> pd.DataFrame:
        """Generates realistic time-series clinical entries containing anomalies."""
        np.random.seed(42 if patient_id == "PT-8802" else 101)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        # Simulate varying baseline trends dependent on target profile
        drift = np.linspace(0, 8, days) if patient_id == "PT-8802" else np.linspace(0, -2, days)
        
        heart_rate = np.random.normal(74, 3, days) + (drift * 0.5)
        spo2 = np.random.normal(97.5, 0.8, days) - (drift * 0.15)
        bp_systolic = np.random.normal(126, 4, days) + drift
        historical_instability = np.random.normal(14.2, 1.8, days) + (drift * 1.2)
        
        df = pd.DataFrame({
            "Date": dates,
            "Heart_Rate_BPM": heart_rate.round(1),
            "Oxygen_Sat_Percentage": np.clip(spo2, 85, 100).round(1),
            "Systolic_BP_mmHg": bp_systolic.round(1),
            "Voice_Instability_Index": np.clip(historical_instability, 0, 100).round(2)
        })
        return df.set_index("Date")

# =====================================================================
# 🏥 LAYER 3: CORE APPLICATION INTERFACE DESIGN
# =====================================================================
st.set_page_config(page_title="VocalTriage Pro Panel", page_icon="🏥", layout="wide")

# App Header
st.title("🏥 VocalTriage Pro: Comprehensive Medical Dashboard Engine")
st.markdown("Category: **Remote Patient Monitoring (RPM)** | Integrating Multi-Modal Physiological Streams with Acoustic Analytics.")
st.markdown("---")

# Sidebar Electronic Health Record (EHR) Navigator
st.sidebar.header("📁 Electronic Health Records Selector")
active_patient_id = st.sidebar.selectbox(
    "Select Target Patient File:",
    ["PT-8802", "PT-4419"]
)

# Initialize data components based on selection
patient_meta = PatientDatabase.get_demographics(active_patient_id)
clinical_history = PatientDatabase.generate_longitudinal_records(active_patient_id)

# SECTION 1: MASTER CLINICAL HEADER FILE
col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
with col_meta1:
    st.markdown(f"👤 **Patient Identification:**\n### {patient_meta['name']}")
with col_meta2:
    st.markdown(f"🎂 **Chronological Age:**\n### {patient_meta['age']} Years")
with col_meta3:
    st.markdown(f"🩺 **Primary Medical Condition:**\n`{patient_meta['condition']}`")
with col_meta4:
    st.markdown(f"🚨 **Assigned Clinical Tier:**\n`{patient_meta['tier']}`")

st.markdown("---")

# SECTION 2: SYSTEM ALERT COGNITION LAYER
st.subheader("🔔 Automated Pipeline Risk Alerts")
latest_vitals = clinical_history.iloc[-1]
seven_day_baseline = clinical_history.iloc[-7:-1].mean()

alerts_triggered = []
if latest_vitals['Oxygen_Sat_Percentage'] < 94.0:
    alerts_triggered.append(f"🚨 **CRITICAL HYPOXIA RISK:** SpO2 drop tracking below threshold standard ({latest_vitals['Oxygen_Sat_Percentage']}%).")
if (latest_vitals['Systolic_BP_mmHg'] - seven_day_baseline['Systolic_BP_mmHg']) > 8.0:
    alerts_triggered.append(f"⚠️ **CARDIOVASCULAR ANOMALY:** Systolic Blood Pressure surge monitored over standard deviation thresholds.")
if latest_vitals['Voice_Instability_Index'] > 22.0:
    alerts_triggered.append(f"🔍 **NEURO-RESPIRATORY EXCURSION:** Micro-vocal instability indexes are tracking abnormally high.")

if alerts_triggered:
    for alert in alerts_triggered:
        st.error(alert)
else:
    st.success("🟢 **SYSTEM NORMAL:** Longitudinal tracking indicators are within safe baseline variances.")

# SECTION 3: INTERACTIVE SCREENING ACQUISITION
st.subheader("🎙️ Live Patient Audio Triage Stream")
st.info("Clinical Instruction: Click the processing simulator below to capture 5 seconds of the patient sustaining a flat vowel 'Ah' phone sound pattern.")

if st.button("🔴 Initialize Real-Time Vocal Assessment", use_container_width=True):
    with st.status("Accessing microphone channels... Processing telemetry...", expanded=False) as status:
        time.sleep(1.2)
        status.update(label="Vocal biometrics computed!", state="complete")
    
    # Simulate extraction waveform context
    t = np.linspace(0, 5, 22050 * 5)
    # Add intentional variation vectors depending on patient state severity profile
    variance_factor = 0.12 if active_patient_id == "PT-8802" else 0.03
    simulated_wave = np.sin(2 * np.pi * 145 * t) + variance_factor * np.sin(2 * np.pi * 7 * t)
    
    # Process waveform through backend parsing engine
    engine = VoiceBiometricEngine()
    live_metrics = engine.analyze_audio_stream(simulated_wave)
    
    # Update temporary history data layer directly to demonstrate instant updates
    clinical_history.loc[pd.Timestamp.now(), "Voice_Instability_Index"] = live_metrics["instability_score"]
    latest_vitals["Voice_Instability_Index"] = live_metrics["instability_score"]
    
    st.toast("Patient vocal metric updated successfully in tracking graphs!", icon="📊")

# SECTION 4: HISTORICAL DATA KPI SUMMARY
st.subheader("📈 Current Vitals Dashboard (vs. 7-Day Running Baseline)")
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    hr_delta = latest_vitals['Heart_Rate_BPM'] - seven_day_baseline['Heart_Rate_BPM']
    st.metric(label="Heart Rate", value=f"{latest_vitals['Heart_Rate_BPM']:.0f} BPM", delta=f"{hr_delta:+.1f} vs baseline")
with col_kpi2:
    bp_delta = latest_vitals['Systolic_BP_mmHg'] - seven_day_baseline['Systolic_BP_mmHg']
    st.metric(label="Systolic Blood Pressure", value=f"{latest_vitals['Systolic_BP_mmHg']:.0f} mmHg", delta=f"{bp_delta:+.1f} mmHg", delta_color="inverse")
with col_kpi3:
    spo2_delta = latest_vitals['Oxygen_Sat_Percentage'] - seven_day_baseline['Oxygen_Sat_Percentage']
    st.metric(label="Oxygen Saturation (SpO2)", value=f"{latest_vitals['Oxygen_Sat_Percentage']:.1f}%", delta=f"{spo2_delta:+.1f}%", delta_color="inverse")
with col_kpi4:
    vocal_delta = latest_vitals['Voice_Instability_Index'] - seven_day_baseline['Voice_Instability_Index']
    st.metric(label="Voice Instability Index", value=f"{latest_vitals['Voice_Instability_Index']:.1f}%", delta=f"{vocal_delta:+.1f}%", delta_color="inverse")

# SECTION 5: VISUAL TRAJECTORY EXPLORER
st.markdown("---")
st.subheader("📊 30-Day Physiological Trend Analysis Windows")

tab_graph1, tab_graph2, tab_data = st.tabs(["🫁 Neuro-Acoustic Trajectory", "❤️ Cardio-Respiratory Matrix", "📋 Comprehensive Electronic Log Frame"])

with tab_graph1:
    st.markdown("**Longitudinal Voice Degradation Matrix** (Spikes represent structural voice stability drops)")
    st.line_chart(clinical_history["Voice_Instability_Index"], color="#FF4B4B")
