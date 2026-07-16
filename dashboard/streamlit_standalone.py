"""Standalone Streamlit dashboard for C-MAPSS Predictive Maintenance (no API required)."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import base64
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import xgboost as xgb

st.set_page_config(
    page_title="Predictive Maintenance - C-MAPSS FD001",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Model Loading ───────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21"
]

@st.cache_resource
def load_model():
    """Load XGBoost model from disk."""
    base_path = Path(__file__).parent.parent / "models"
    metadata_path = base_path / "model_metadata.json"
    features_path = base_path / "feature_columns.json"
    
    model = xgb.Booster()
    for fmt in ["xgboost_rul_model.ubj", "xgboost_rul_model.json", "xgboost_rul_model_new.json"]:
        model_path = base_path / fmt
        if model_path.exists():
            model.load_model(str(model_path))
            break
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    feature_cols = FEATURE_COLUMNS[:]
    if features_path.exists():
        with open(features_path, "r") as f:
            feature_cols = json.load(f)
    
    return model, metadata, feature_cols

def predict_single(model, reading, feature_cols):
    """Predict RUL from a single reading."""
    features = {}
    for col in FEATURE_COLUMNS:
        val = float(reading.get(col, 0.0))
        features[col] = val
        features[f"{col}_roll_mean"] = val
        features[f"{col}_roll_std"] = 0.0
    
    feature_values = [features.get(col, 0.0) for col in feature_cols]
    
    X = np.array([feature_values])
    dmatrix = xgb.DMatrix(X, feature_names=feature_cols)
    
    raw_pred = model.predict(dmatrix)[0]
    clipped = float(np.clip(raw_pred, 0, 125))
    
    if clipped <= 10:
        urgency = "critical"
    elif clipped <= 20:
        urgency = "urgent"
    elif clipped <= 40:
        urgency = "warning"
    elif clipped <= 80:
        urgency = "monitoring"
    else:
        urgency = "healthy"
    
    return {"predicted_rul": float(raw_pred), "rul_clipped": clipped, "urgency": urgency}

# ─── CSS Theme ───────────────────────────────────────────────────────────────
def get_base64_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

assets_path = Path(__file__).parent.parent / "assets" / "jet_engine.png"
if assets_path.exists():
    jet_bg = get_base64_image(assets_path)
else:
    jet_bg = ""

st.markdown(f"""
<style>
    .stApp {{
        background: 
            linear-gradient(rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.92)),
            url('data:image/png;base64,{jet_bg}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #ffffff;
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%) !important;
        border-right: 3px solid #ff6b00;
    }}
    section[data-testid="stSidebar"] .stRadio > div > label {{ color: #ffffff !important; font-weight: 500; }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ color: #ff6b00 !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: #ff6b00 !important; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8); font-weight: 700 !important; }}
    h1 {{ font-size: 2.5rem !important; border-bottom: 3px solid #ff6b00; padding-bottom: 10px; }}
    .stMetric {{ background: rgba(10, 10, 10, 0.9) !important; border: 2px solid #ff6b00; border-radius: 12px; padding: 20px; box-shadow: 0 8px 32px rgba(255, 107, 0, 0.3); }}
    .stMetric label {{ color: #ff9a56 !important; font-weight: 600 !important; }}
    .stMetric [data-testid="stMetricValue"] {{ color: #ffffff !important; font-weight: 700 !important; }}
    .stButton > button {{ background: linear-gradient(135deg, #ff6b00 0%, #ff8c33 100%) !important; color: #000000 !important; border: none; border-radius: 8px; font-weight: 800 !important; text-transform: uppercase; letter-spacing: 2px; padding: 12px 30px; box-shadow: 0 0 20px rgba(255, 107, 0, 0.6); }}
    .stButton > button:hover {{ box-shadow: 0 0 30px rgba(255, 107, 0, 0.9); transform: translateY(-2px); }}
    .stSlider label, .stSlider > div > div > div > label, .stSlider > div > div > div > div > label {{ color: #ffffff !important; font-weight: 700 !important; font-size: 1rem !important; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9); }}
    .stSlider > div > div > div > div > div {{ background-color: #ff6b00 !important; }}
    .stSlider > div > div > div > div > div > div {{ background: #ff6b00 !important; border: 3px solid #ffffff !important; box-shadow: 0 0 10px rgba(255, 107, 0, 0.8); }}
    .stSlider p, .stSlider span, .stSlider div {{ color: #ffffff !important; }}
    .stSuccess {{ background: rgba(0, 40, 0, 0.9) !important; border: 2px solid #00ff88; border-radius: 8px; color: #00ff88 !important; }}
    .stWarning {{ background: rgba(40, 40, 0, 0.9) !important; border: 2px solid #ffaa00; border-radius: 8px; color: #ffaa00 !important; }}
    .stError {{ background: rgba(40, 0, 0, 0.9) !important; border: 2px solid #ff4444; border-radius: 8px; color: #ff4444 !important; }}
    .stInfo {{ background: rgba(0, 20, 40, 0.9) !important; border: 2px solid #00aaff; border-radius: 8px; color: #00aaff !important; }}
    .stDataFrame {{ border: 2px solid #ff6b00; border-radius: 8px; background: rgba(10, 10, 10, 0.95); }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; background: rgba(10, 10, 10, 0.8); border-radius: 8px; padding: 5px; }}
    .stTabs [data-baseweb="tab"] {{ background: rgba(26, 26, 26, 0.9); border: 2px solid #ff6b00; border-radius: 6px; color: #ff9a56; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ background: #ff6b00 !important; color: #000000 !important; font-weight: 800; }}
    .streamlit-expanderHeader {{ background: rgba(10, 10, 10, 0.9) !important; border: 2px solid #ff6b00; border-radius: 8px; color: #ff6b00 !important; font-weight: 600; }}
    hr {{ border-color: #ff6b00; border-width: 2px; }}
    .stSelectbox > div > div {{ background: rgba(10, 10, 10, 0.9); border-color: #ff6b00; color: #ffffff; }}
    .stFileUploader {{ border: 3px dashed #ff6b00; border-radius: 12px; background: rgba(10, 10, 10, 0.8); }}
    .stRadio > div > label {{ color: #ffffff !important; }}
    .stCheckbox > label > span {{ color: #ffffff !important; }}
    .stTextInput > div > div > input {{ background: rgba(10, 10, 10, 0.9); border: 2px solid #ff6b00; color: #ffffff; }}
    .stSubheader {{ color: #ff9a56 !important; border-left: 4px solid #ff6b00; padding-left: 15px; }}
    .stPlotlyChart {{ background: rgba(10, 10, 10, 0.8); border-radius: 12px; border: 1px solid #ff6b00; }}
</style>
""", unsafe_allow_html=True)

# ─── Sensor Definitions ──────────────────────────────────────────────────────
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

# ─── Load Model ──────────────────────────────────────────────────────────────
model, metadata, feature_cols = load_model()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.title("✈️ C-MAPSS RUL Predictor")
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Model Loaded Locally")
    if metadata:
        st.sidebar.info(f"Model: {metadata.get('model_type', 'xgboost')}")
        st.sidebar.info(f"Features: {metadata.get('n_features', 51)}")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate to:",
        ["🔮 Predict RUL", "📊 Sensor Reference", "⚙️ Model Info", "📈 Batch Analysis"]
    )
    return page

# ─── Pages ───────────────────────────────────────────────────────────────────
def render_predict_rul():
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
            op_readings[key] = st.slider(info["name"], min_value=info["min"], max_value=info["max"], value=info["default"], step=0.0001, format="%.4f")
    
    with col2:
        st.markdown("### 📡 Sensor Readings")
        sensor_readings = {}
        for key, info in SENSORS.items():
            sensor_readings[key] = st.slider(info["name"], min_value=info["min"], max_value=info["max"], value=info["default"], step=(info["max"] - info["min"]) / 100)
    
    reading = {**op_readings, **sensor_readings}
    st.markdown("---")
    
    if st.button("🚀 Predict RUL", type="primary", use_container_width=True):
        with st.spinner("Analyzing sensor data..."):
            result = predict_single(model, reading, feature_cols)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rul = result["rul_clipped"]
            st.metric("Predicted RUL", f"{rul:.1f} cycles")
        with col2:
            urgency = result["urgency"]
            icons = {"critical": "🔴", "urgent": "🟠", "warning": "🟡", "monitoring": "🔵", "healthy": "🟢"}
            st.metric("Urgency", f"{icons.get(urgency, '⚪')} {urgency.upper()}")
        with col3:
            st.metric("Raw Prediction", f"{result['predicted_rul']:.1f}")
        with col4:
            st.metric("Features Used", 51)
        
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
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 20}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_sensor_reference():
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
    sensor_data = [{"Sensor": k, "Description": v["name"], "Min": v["min"], "Max": v["max"], "Typical": v["default"]} for k, v in SENSORS.items()]
    st.dataframe(pd.DataFrame(sensor_data), use_container_width=True)
    
    st.subheader("Dropped Sensors (Uninformative)")
    st.markdown(f"`{', '.join(DROPPED_SENSORS)}`")
    st.info("These sensors are near-constant across all operating conditions and provide no predictive signal.")

def render_model_info():
    st.title("⚙️ Model Information")
    st.success("✅ Model loaded locally (no API required)")
    
    if metadata:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Details")
            st.write(f"**Type:** {metadata.get('model_type', 'N/A')}")
            st.write(f"**Dataset:** {metadata.get('dataset', 'N/A')}")
            st.write(f"**Features:** {metadata.get('n_features', 0)}")
            st.write(f"**RUL Clip:** {metadata.get('rul_clip', 125)}")
            st.write(f"**Rolling Window:** {metadata.get('rolling_window', 10)}")
        with col2:
            st.subheader("Hyperparameters")
            for k, v in metadata.get("hyperparameters", {}).items():
                st.write(f"**{k}:** {v}")
        
        st.markdown("---")
        st.subheader("Test Set Performance")
        metrics = metadata.get("test_metrics", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
        with c2:
            st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
        with c3:
            st.metric("NASA Score", f"{metrics.get('nasa_score', 0):.0f}")

def render_batch_analysis():
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
                            result = predict_single(model, row.to_dict(), feature_cols)
                            predictions.append({
                                "row": idx,
                                "unit_number": row.get("unit_number", idx),
                                "rul": result["rul_clipped"],
                                "urgency": result["urgency"]
                            })
                        
                        pred_df = pd.DataFrame(predictions)
                        st.success(f"Completed {len(predictions)} predictions")
                        st.dataframe(pred_df)
                        
                        fig = px.bar(pred_df, x="unit_number", y="rul", color="urgency",
                                     color_discrete_map={"critical": "red", "urgent": "orange", "warning": "yellow", "monitoring": "blue", "healthy": "green"},
                                     title="RUL Predictions by Engine")
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
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
