import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="PropPredict Pro", layout="wide")

# -----------------------------
# Load Models
# -----------------------------
price_model = joblib.load("real_estate_rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")

try:
    investment_model = joblib.load("investment_classifier.pkl")
    classifier_columns = joblib.load("classifier_columns.pkl")
except:
    investment_model = None

# -----------------------------
# City Coordinates
# -----------------------------
city_coords = {
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Hyderabad": (17.3850, 78.4867)
}

# -----------------------------
# Title
# -----------------------------
st.title("🏠 PropPredict Pro")
st.markdown("### AI Real Estate Price & Investment Advisor")

st.markdown("---")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📥 Property Inputs")

size = st.sidebar.slider("Size (SqFt)", 500, 5000, 1200)
bhk = st.sidebar.selectbox("BHK", [1, 2, 3, 4, 5])
price_per_sqft = st.sidebar.slider("Price per SqFt", 1000, 20000, 8000)

property_type = st.sidebar.selectbox(
    "Property Type",
    ["Apartment", "Villa", "Independent House"]
)

furnishing = st.sidebar.selectbox(
    "Furnishing",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

city = st.sidebar.selectbox(
    "City",
    list(city_coords.keys())
)

# -----------------------------
# Predict Button
# -----------------------------
if st.sidebar.button("🚀 Predict"):

    # -----------------------------
    # PRICE INPUT
    # -----------------------------
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0

    input_data["Size_in_SqFt"] = size
    input_data["BHK"] = bhk
    input_data["Price_per_SqFt"] = price_per_sqft

    # Dropdown mapping
    prop_col = f"Property_Type_{property_type}"
    furn_col = f"Furnishing_{furnishing}"
    city_col = f"City_{city}"

    for col in [prop_col, furn_col, city_col]:
        if col in input_data.columns:
            input_data[col] = 1

    # -----------------------------
    # PRICE PREDICTION
    # -----------------------------
    price_pred = price_model.predict(input_data)[0]

    # -----------------------------
    # DISPLAY METRICS
    # -----------------------------
    st.markdown("## 📊 Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("💰 Predicted Price", f"{price_pred:.2f} Lakhs")

    with col2:
        manual = size * price_per_sqft / 100000
        st.metric("📌 Manual Estimate", f"{manual:.2f} Lakhs")

    with col3:
        if investment_model is not None:
            input_class = pd.DataFrame(columns=classifier_columns)
            input_class.loc[0] = 0

            input_class["Size_in_SqFt"] = size
            input_class["BHK"] = bhk
            input_class["Price_per_SqFt"] = price_per_sqft

            for col in [prop_col, furn_col, city_col]:
                if col in input_class.columns:
                    input_class[col] = 1

            prob = investment_model.predict_proba(input_class)[0][1]
            score = int(prob * 100)

            st.metric("🎯 Investment Score", f"{score}/100")

            if score > 70:
                st.success("🔥 Excellent Investment")
            elif score > 50:
                st.info("👍 Moderate Investment")
            else:
                st.warning("⚠️ Risky Investment")

    st.markdown("---")

    # -----------------------------
    # 🗺️ INTERACTIVE MAP (Plotly)
    # -----------------------------
    st.subheader("🗺️ Location Insight")

    lat, lon = city_coords[city]

    # Create nearby points for better visualization
    df_map = pd.DataFrame({
        "lat": lat + np.random.randn(20) * 0.02,
        "lon": lon + np.random.randn(20) * 0.02,
        "price": np.random.randint(50, 500, 20)
    })

    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        size="price",
        color="price",
        color_continuous_scale="viridis",
        zoom=10,
        height=500
    )

    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

    st.markdown("---")

    # -----------------------------
    # Insight
    # -----------------------------
    st.subheader("📈 Insight")

    st.info(
        "Predictions are based on property size, location, furnishing, and type. "
        "Map shows nearby simulated listings to visualize market density."
    )
