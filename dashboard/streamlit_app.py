"""Streamlit dashboard for Predictive Maintenance System - C-MAPSS Edition."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Predictive Maintenance - C-MAPSS FD001",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NASA / Aerospace Black & Orange Theme with Jet Engine Background
import base64
import os

def get_base64_image(image_path):
    """Convert image to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# Get the jet engine background
jet_engine_bg = get_base64_image(os.path.join(os.path.dirname(__file__), '..', 'assets', 'jet_engine.png'))

JET_ENGINE_CSS = """
<style>
    /* Jet engine blueprint background with dark overlay */
    .stApp {
        background: 
            linear-gradient(rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.92)),
            url('data:image/png;base64,JET_BG_PLACEHOLDER');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #ffffff;
    }
    
    /* Main content area */
    .stApp > header {
        background: transparent;
    }
    
    /* Sidebar - solid black with orange border */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%) !important;
        border-right: 3px solid #ff6b00;
    }
    
    section[data-testid="stSidebar"] .stRadio > div > label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ff6b00 !important;
    }
    
    /* Headers - bright orange */
    h1, h2, h3, h4, h5, h6 {
        color: #ff6b00 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
        font-weight: 700 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        border-bottom: 3px solid #ff6b00;
        padding-bottom: 10px;
    }
    
    /* Cards and metric containers - black glass effect */
    .stMetric {
        background: rgba(10, 10, 10, 0.9) !important;
        border: 2px solid #ff6b00;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(255, 107, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stMetric label {
        color: #ff9a56 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    /* Buttons - glowing orange */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b00 0%, #ff8c33 100%) !important;
        color: #000000 !important;
        border: none;
        border-radius: 8px;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 12px 30px;
        box-shadow: 0 0 20px rgba(255, 107, 0, 0.6);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(255, 107, 0, 0.9);
        transform: translateY(-2px);
        background: linear-gradient(135deg, #ff8c33 0%, #ffaa55 100%) !important;
    }
    
    /* Slider labels - HIGH VISIBILITY */
    .stSlider label,
    .stSlider > div > div > div > label,
    .stSlider > div > div > div > div > label,
    [data-baseweb="slider"] + label,
    .stMarkdown > p > strong {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);
    }
    
    /* Slider track */
    .stSlider > div > div > div > div > div {
        background-color: #ff6b00 !important;
    }
    
    /* Slider thumb */
    .stSlider > div > div > div > div > div > div {
        background: #ff6b00 !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 0 10px rgba(255, 107, 0, 0.8);
    }
    
    /* All text in slider containers */
    .stSlider p,
    .stSlider span,
    .stSlider div {
        color: #ffffff !important;
    }
    
    /* Success box - green glow */
    .stSuccess {
        background: rgba(0, 40, 0, 0.9) !important;
        border: 2px solid #00ff88;
        border-radius: 8px;
        color: #00ff88 !important;
    }
    
    /* Warning box - amber glow */
    .stWarning {
        background: rgba(40, 40, 0, 0.9) !important;
        border: 2px solid #ffaa00;
        border-radius: 8px;
        color: #ffaa00 !important;
    }
    
    /* Error box - red glow */
    .stError {
        background: rgba(40, 0, 0, 0.9) !important;
        border: 2px solid #ff4444;
        border-radius: 8px;
        color: #ff4444 !important;
    }
    
    /* Info box - blue glow */
    .stInfo {
        background: rgba(0, 20, 40, 0.9) !important;
        border: 2px solid #00aaff;
        border-radius: 8px;
        color: #00aaff !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 2px solid #ff6b00;
        border-radius: 8px;
        background: rgba(10, 10, 10, 0.95);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(10, 10, 10, 0.8);
        border-radius: 8px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(26, 26, 26, 0.9);
        border: 2px solid #ff6b00;
        border-radius: 6px;
        color: #ff9a56;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ff6b00 !important;
        color: #000000 !important;
        font-weight: 800;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(10, 10, 10, 0.9) !important;
        border: 2px solid #ff6b00;
        border-radius: 8px;
        color: #ff6b00 !important;
        font-weight: 600;
    }
    
    /* Dividers */
    hr {
        border-color: #ff6b00;
        border-width: 2px;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(10, 10, 10, 0.9);
        border-color: #ff6b00;
        color: #ffffff;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 3px dashed #ff6b00;
        border-radius: 12px;
        background: rgba(10, 10, 10, 0.8);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b00, #ffaa00);
    }
    
    /* Radio buttons */
    .stRadio > div > label {
        color: #ffffff !important;
    }
    
    /* Checkbox */
    .stCheckbox > label > span {
        color: #ffffff !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background: rgba(10, 10, 10, 0.9);
        border: 2px solid #ff6b00;
        color: #ffffff;
    }
    
    /* Number inputs */
    .stNumberInput > div > div > input {
        background: rgba(10, 10, 10, 0.9);
        border: 2px solid #ff6b00;
        color: #ffffff;
    }
    
    /* Subheaders */
    .stSubheader {
        color: #ff9a56 !important;
        border-left: 4px solid #ff6b00;
        padding-left: 15px;
    }
    
    /* Caption */
    .stCaption {
        color: #cccccc !important;
    }
    
    /* Code blocks */
    .stCode {
        background: rgba(10, 10, 10, 0.95);
        border: 1px solid #ff6b00;
    }
    
    /* Plotly charts background */
    .stPlotlyChart {
        background: rgba(10, 10, 10, 0.8);
        border-radius: 12px;
        border: 1px solid #ff6b00;
    }
</style>
"""

st.markdown(JET_ENGINE_CSS.replace('JET_BG_PLACEHOLDER', jet_engine_bg), unsafe_allow_html=True)

# C-MAPSS Sensor definitions
OP_SETTINGS = {
    "op_setting_1": {"name": "Operational Setting 1", "min": -0.01, "max": 0.01, "default": 0.0},
    "op_setting_2": {"name": "Operational Setting 2", "min": -0.01, "max": 0.01, "default": 0.0},
    "op_setting_3": {"name": "Operational Setting 3", "min": 99.0, "max": 101.0, "default": 100.0},
}

SENSORS = {
    "sensor_2": {"name": "Sensor 2 - Total temp at fan inlet", "min": 640.0, "max": 645.0, "default": 641.82},
    "sensor_3": {"name": "Sensor 3 - Total pressure at bypass duct", "min": 1580.0, "max": 1600.0, "default": 1589.70},
    "sensor_4": {"name": "Sensor 4 - Total pressure ratio", "min": 1395.0, "max": 1410.0, "default": 1400.60},
    "sensor_7": {"name": "Sensor 7 - Physical fan speed", "min": 550.0, "max": 560.0, "default": 554.36},
    "sensor_8": {"name": "Sensor 8 - Physical core speed", "min": 2385.0, "max": 2395.0, "default": 2388.06},
    "sensor_9": {"name": "Sensor 9 - Engine pressure ratio (PR)", "min": 9040.0, "max": 9055.0, "default": 9046.19},
    "sensor_11": {"name": "Sensor 11 - Static pressure at burner", "min": 46.0, "max": 49.0, "default": 47.47},
    "sensor_12": {"name": "Sensor 12 - Ratio of fuel flow to PS30", "min": 518.0, "max": 525.0, "default": 521.66},
    "sensor_13": {"name": "Sensor 13 - Corrected fan speed", "min": 2385.0, "max": 2392.0, "default": 2388.02},
    "sensor_14": {"name": "Sensor 14 - Corrected core speed", "min": 8125.0, "max": 8145.0, "default": 8138.62},
    "sensor_15": {"name": "Sensor 15 - Bypass ratio", "min": 8.3, "max": 8.5, "default": 8.4195},
    "sensor_17": {"name": "Sensor 17 - Burner fuel-air ratio", "min": 388.0, "max": 395.0, "default": 392.0},
    "sensor_20": {"name": "Sensor 20 - Bleed enthalpy", "min": 38.0, "max": 41.0, "default": 39.06},
    "sensor_21": {"name": "Sensor 21 - Requested fan speed", "min": 23.3, "max": 23.5, "default": 23.4190},
}

DROPPED_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]


def check_api_health():
    """Check if API is running."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200, resp.json()
    except:
        return False, None


def predict_rul(reading: dict):
    """Call RUL prediction API."""
    try:
        resp = requests.post(
            f"{API_URL}/api/v1/predict/rul",
            json={"single_reading": reading},
            timeout=10
        )
        return resp.status_code == 200, resp.json()
    except Exception as e:
        return False, {"error": str(e)}


def render_sidebar():
    """Render sidebar navigation."""
    st.sidebar.title("✈️ C-MAPSS RUL Predictor")
    st.sidebar.markdown("---")
    
    # API status
    api_ok, health = check_api_health()
    if api_ok:
        st.sidebar.success("✅ API Connected")
        if health:
            st.sidebar.info(f"Model: {health.get('model_type', 'unknown')}")
            st.sidebar.info(f"Features: {health.get('n_features', 0)}")
    else:
        st.sidebar.error("❌ API Not Available")
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        ["🔮 Predict RUL", "📊 Sensor Reference", "⚙️ Model Info", "📈 Batch Analysis"]
    )
    
    return page


def render_predict_rul():
    """Render RUL prediction page."""
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: rgba(10,10,10,0.9); 
                border: 2px solid #ff6b00; border-radius: 15px; margin-bottom: 20px;
                box-shadow: 0 0 30px rgba(255,107,0,0.4);'>
        <h1 style='color: #ff6b00; border: none; font-size: 2.2rem; margin-bottom: 5px;'>
            ✈️ JET ENGINE RUL PREDICTOR
        </h1>
        <p style='color: #ffffff; font-size: 1.1rem; margin: 0;'>
            NASA C-MAPSS FD001 Turbofan Degradation Analysis
        </p>
        <p style='color: #ff9a56; font-size: 0.9rem; margin: 5px 0 0 0;'>
            Adjust sensor readings below to predict Remaining Useful Life
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ⚙️ Operational Settings")
        op_readings = {}
        for key, info in OP_SETTINGS.items():
            op_readings[key] = st.slider(
                info["name"],
                min_value=info["min"],
                max_value=info["max"],
                value=info["default"],
                step=0.0001,
                format="%.4f"
            )
    
    with col2:
        st.markdown("### 📡 Sensor Readings")
        sensor_readings = {}
        for key, info in SENSORS.items():
            sensor_readings[key] = st.slider(
                info["name"],
                min_value=info["min"],
                max_value=info["max"],
                value=info["default"],
                step=(info["max"] - info["min"]) / 100
            )
    
    # Combine readings
    reading = {**op_readings, **sensor_readings}
    
    st.markdown("---")
    
    if st.button("🚀 Predict RUL", type="primary", use_container_width=True):
        with st.spinner("Analyzing sensor data..."):
            success, result = predict_rul(reading)
        
        if success:
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rul = result.get("rul_clipped", 0)
                st.metric("Predicted RUL", f"{rul:.1f} cycles")
            
            with col2:
                urgency = result.get("urgency", "unknown")
                color_map = {
                    "critical": "🔴",
                    "urgent": "🟠",
                    "warning": "🟡",
                    "monitoring": "🔵",
                    "healthy": "🟢"
                }
                st.metric("Urgency", f"{color_map.get(urgency, '⚪')} {urgency.upper()}")
            
            with col3:
                raw_rul = result.get("predicted_rul", 0)
                st.metric("Raw Prediction", f"{raw_rul:.1f}")
            
            with col4:
                features = result.get("features_used", 0)
                st.metric("Features Used", features)
            
            # Urgency gauge
            st.markdown("---")
            st.subheader("Health Assessment")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rul,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Remaining Useful Life (cycles)"},
                delta={"reference": 60, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                gauge={
                    "axis": {"range": [0, 125], "tickwidth": 1},
                    "bar": {"color": "darkblue"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 10], "color": "red"},
                        {"range": [10, 20], "color": "orange"},
                        {"range": [20, 40], "color": "yellow"},
                        {"range": [40, 80], "color": "lightblue"},
                        {"range": [80, 125], "color": "lightgreen"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 20
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")


def render_sensor_reference():
    """Render sensor reference page."""
    st.title("📊 C-MAPSS Sensor Reference")
    
    st.markdown("""
    ## NASA C-MAPSS FD001 Dataset
    
    The dataset simulates turbofan engine degradation. Each engine starts with 
    some degree of initial wear and manufacturing variation, which is unknown.
    
    ### Operational Settings
    - **op_setting_1**: Altitude setting
    - **op_setting_2**: Mach number setting  
    - **op_setting_3**: Throttle resolver angle setting
    """)
    
    st.subheader("Sensors Used by Model")
    
    sensor_data = []
    for key, info in SENSORS.items():
        sensor_data.append({
            "Sensor": key,
            "Description": info["name"],
            "Min": info["min"],
            "Max": info["max"],
            "Typical": info["default"]
        })
    
    df = pd.DataFrame(sensor_data)
    st.dataframe(df, use_container_width=True)
    
    st.subheader("Dropped Sensors (Uninformative)")
    st.markdown(f"`{', '.join(DROPPED_SENSORS)}`")
    st.info("These sensors are near-constant across all operating conditions and provide no predictive signal.")


def render_model_info():
    """Render model information page."""
    st.title("⚙️ Model Information")
    
    api_ok, health = check_api_health()
    
    if api_ok:
        st.success("API is connected and model is loaded")
        
        # Get model info
        try:
            resp = requests.get(f"{API_URL}/model/info", timeout=5)
            info = resp.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Details")
                st.write(f"**Type:** {info.get('model_type', 'N/A')}")
                st.write(f"**Dataset:** {info.get('dataset', 'N/A')}")
                st.write(f"**Features:** {info.get('n_features', 0)}")
                st.write(f"**RUL Clip:** {info.get('rul_clip', 125)}")
                st.write(f"**Rolling Window:** {info.get('rolling_window', 10)}")
            
            with col2:
                st.subheader("Hyperparameters")
                hp = info.get("hyperparameters", {})
                for key, value in hp.items():
                    st.write(f"**{key}:** {value}")
            
            st.markdown("---")
            st.subheader("Test Set Performance")
            
            metrics = info.get("test_metrics", {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}", help="Root Mean Squared Error (lower is better)")
            
            with col2:
                st.metric("MAE", f"{metrics.get('mae', 0):.2f}", help="Mean Absolute Error (lower is better)")
            
            with col3:
                st.metric("NASA Score", f"{metrics.get('nasa_score', 0):.0f}", help="Asymmetric scoring (lower is better, penalizes late predictions)")
            
        except Exception as e:
            st.error(f"Error fetching model info: {e}")
    else:
        st.error("API not available")


def render_batch_analysis():
    """Render batch analysis page."""
    st.title("📈 Batch Analysis")
    
    st.markdown("Upload a CSV file with sensor readings for batch RUL prediction.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} rows")
            st.dataframe(df.head())
            
            required_cols = list(OP_SETTINGS.keys()) + list(SENSORS.keys())
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing..."):
                        predictions = []
                        for idx, row in df.iterrows():
                            reading = row.to_dict()
                            success, result = predict_rul(reading)
                            if success:
                                predictions.append({
                                    "row": idx,
                                    "rul": result.get("rul_clipped", 0),
                                    "urgency": result.get("urgency", "unknown")
                                })
                        
                        if predictions:
                            pred_df = pd.DataFrame(predictions)
                            st.success(f"Completed {len(predictions)} predictions")
                            st.dataframe(pred_df)
                            
                            # Visualization
                            fig = px.bar(
                                pred_df, 
                                x="row", 
                                y="rul",
                                color="urgency",
                                color_discrete_map={
                                    "critical": "red",
                                    "urgent": "orange",
                                    "warning": "yellow",
                                    "monitoring": "blue",
                                    "healthy": "green"
                                },
                                title="RUL Predictions by Row"
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


def main():
    """Main application entry point."""
    page = render_sidebar()
    
    if page == "🔮 Predict RUL":
        render_predict_rul()
    elif page == "📊 Sensor Reference":
        render_sensor_reference()
    elif page == "⚙️ Model Info":
        render_model_info()
    elif page == "📈 Batch Analysis":
        render_batch_analysis()


if __name__ == "__main__":
    main()
