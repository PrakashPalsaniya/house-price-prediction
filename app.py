import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    with open('house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 House Price Prediction System")
st.markdown("---")

# Load model
model, features = load_model()

# Sidebar
st.sidebar.header("📊 Model Information")
st.sidebar.info("""
This app predicts house prices based on various features:
- Median Income
- House Age
- Number of Rooms
- Number of Bedrooms
- Population
- Households
- Latitude & Longitude
""")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("🔢 Input Features")
    
    # California Housing features
    MedInc = st.slider("Median Income (in $10,000)", 0.5, 15.0, 3.0, 0.1)
    HouseAge = st.slider("House Age (years)", 1, 52, 25, 1)
    AveRooms = st.slider("Average Rooms", 2.0, 10.0, 5.0, 0.1)
    AveBedrms = st.slider("Average Bedrooms", 1.0, 5.0, 2.0, 0.1)
    Population = st.slider("Population", 100, 5000, 1000, 100)
    AveOccup = st.slider("Average Occupancy", 1.0, 10.0, 3.0, 0.1)
    Latitude = st.slider("Latitude", 32.0, 42.0, 37.0, 0.1)
    Longitude = st.slider("Longitude", -125.0, -114.0, -120.0, 0.1)

with col2:
    st.header("🎯 Prediction")
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'MedInc': [MedInc],
        'HouseAge': [HouseAge],
        'AveRooms': [AveRooms],
        'AveBedrms': [AveBedrms],
        'Population': [Population],
        'AveOccup': [AveOccup],
        'Latitude': [Latitude],
        'Longitude': [Longitude]
    })
    
    # Make prediction
    if st.button("🔮 Predict Price", type="primary"):
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.success(f"### Predicted House Price: ${prediction * 100000:,.2f}")
        
        # Additional info
        st.info(f"""
        **Price Breakdown:**
        - Price per room: ${(prediction * 100000) / AveRooms:,.2f}
        - Price per bedroom: ${(prediction * 100000) / AveBedrms:,.2f}
        """)

# Footer
st.markdown("---")
st.markdown("### 📈 Model Performance")

col3, col4, col5 = st.columns(3)
col3.metric("Model Type", "XGBoost")
col4.metric("R² Score", "~0.84")
col5.metric("RMSE", "~0.52")

# Display visualizations if they exist
st.markdown("---")
st.header("📊 Visualizations")

try:
    img1 = Image.open('model_evaluation.png')
    st.image(img1, caption='Model Evaluation', use_container_width=True)
except:
    st.warning("Run model_evaluation.py first to generate visualizations")
