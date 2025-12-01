import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ======================
# 🔧 Config & Constants
# ======================

st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Final selected feature order – MUST match training
FEATURE_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "1stFlrSF",
    "TotalBsmtSF",
    "BsmtFinSF1",
    "LotArea",
    "GarageCars",
    "TotRmsAbvGrd",
    "2ndFlrSF",
    "YearBuilt",
    "GarageArea",
    "FullBath",
    "OverallCond",
    "YearRemodAdd",
    "MSSubClass",
]

# Nice labels + helper text for UI
FEATURE_META = {
    "OverallQual": {
        "label": "Overall Material & Finish Quality (1–10)",
        "help": "Overall quality of the house (10 = very excellent, 1 = very poor).",
        "min": 1, "max": 10, "step": 1, "default": 7,
    },
    "GrLivArea": {
        "label": "Above Ground Living Area (sq ft)",
        "help": "Total living area above ground level.",
        "min": 400, "max": 6000, "step": 10, "default": 1800,
    },
    "1stFlrSF": {
        "label": "1st Floor Area (sq ft)",
        "help": "Finished area of the first floor.",
        "min": 300, "max": 4000, "step": 10, "default": 1200,
    },
    "TotalBsmtSF": {
        "label": "Total Basement Area (sq ft)",
        "help": "Total basement area (finished + unfinished).",
        "min": 0, "max": 4000, "step": 10, "default": 800,
    },
    "BsmtFinSF1": {
        "label": "Finished Basement Area (sq ft)",
        "help": "Area of finished part of the basement.",
        "min": 0, "max": 4000, "step": 10, "default": 500,
    },
    "LotArea": {
        "label": "Lot Area (sq ft)",
        "help": "Total land area of the property.",
        "min": 1000, "max": 50000, "step": 50, "default": 9000,
    },
    "GarageCars": {
        "label": "Garage Capacity (cars)",
        "help": "Number of cars the garage can hold.",
        "min": 0, "max": 5, "step": 1, "default": 2,
    },
    "TotRmsAbvGrd": {
        "label": "Total Rooms Above Ground",
        "help": "Total rooms above ground (excluding bathrooms).",
        "min": 2, "max": 15, "step": 1, "default": 7,
    },
    "2ndFlrSF": {
        "label": "2nd Floor Area (sq ft)",
        "help": "Finished area of the second floor (0 if no second floor).",
        "min": 0, "max": 3000, "step": 10, "default": 400,
    },
    "YearBuilt": {
        "label": "Year Built",
        "help": "Original construction year of the house.",
        "min": 1870, "max": 2025, "step": 1, "default": 2005,
    },
    "GarageArea": {
        "label": "Garage Area (sq ft)",
        "help": "Floor area of the garage.",
        "min": 0, "max": 1500, "step": 10, "default": 450,
    },
    "FullBath": {
        "label": "Full Bathrooms Above Ground",
        "help": "Number of full bathrooms (above ground).",
        "min": 0, "max": 4, "step": 1, "default": 2,
    },
    "OverallCond": {
        "label": "Overall Condition (1–10)",
        "help": "Overall condition of the house (10 = excellent, 1 = very poor).",
        "min": 1, "max": 10, "step": 1, "default": 6,
    },
    "YearRemodAdd": {
        "label": "Year Remodeled",
        "help": "Most recent remodel/addition year (same as Year Built if never remodeled).",
        "min": 1950, "max": 2025, "step": 1, "default": 2010,
    },
    "MSSubClass": {
        "label": "Building Class (MSSubClass)",
        "help": "Building class (e.g., 20 = 1-story, 60 = 2-story).",
        "min": 20, "max": 190, "step": 5, "default": 60,
    },
}


# ======================
# 📦 Load Model
# ======================

@st.cache_resource
def load_model():
    # Adjust path if your model is elsewhere
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "ames_house_price_production.pkl"
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return joblib.load(model_path)


model = load_model()


# ======================
# 🎨 Custom Styling
# ======================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 0.8rem;
        background: linear-gradient(135deg, #1d976c, #93f9b9);
        color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    .prediction-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .footer {
        font-size: 0.8rem;
        color: #777;
        margin-top: 2rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================
# 🧱 Layout
# ======================

st.markdown('<div class="main-title">🏠 Ames House Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Estimate the sale price of a house using a production-grade ML model trained on the Ames Housing dataset.</div>',
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1.4, 1])

# -------- Sidebar info --------
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write(
        """
        This app uses a **CatBoost regression model** trained on the Ames Housing dataset.
        
        The model was:
        - Cross-validated using 5-fold CV  
        - Trained on **15 most important numeric features**  
        - Wrapped in a scikit-learn `Pipeline` for clean deployment
        """
    )
    st.markdown("---")
    st.subheader("📊 Model Performance (Validation)")
    st.write("- R² ≈ **0.91**")
    st.write("- MAE ≈ **$15.9K**")
    st.write("- RMSE ≈ **$22.2K**")
    st.write("- MAPE ≈ **≈9.6%**")

    st.markdown("---")
    st.caption("Tip: Start with the defaults and tweak values to see how price responds.")


# -------- Main Input Form --------
with left_col:
    st.subheader("📝 Enter Property Details")

    with st.form("house_form"):
        # Split inputs into two columns for nicer UI
        col1, col2 = st.columns(2)

        user_inputs = {}

        # First column fields
        with col1:
            for key in [
                "OverallQual",
                "GrLivArea",
                "1stFlrSF",
                "TotalBsmtSF",
                "BsmtFinSF1",
                "LotArea",
                "GarageCars",
                "TotRmsAbvGrd",
            ]:
                meta = FEATURE_META[key]
                user_inputs[key] = st.number_input(
                    meta["label"],
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=float(meta["default"]),
                    step=float(meta["step"]),
                    help=meta["help"],
                    key=key,
                )

        # Second column fields
        with col2:
            for key in [
                "2ndFlrSF",
                "YearBuilt",
                "GarageArea",
                "FullBath",
                "OverallCond",
                "YearRemodAdd",
                "MSSubClass",
            ]:
                meta = FEATURE_META[key]
                # Choose int or float input
                is_int = meta["step"].is_integer() and meta["min"].is_integer() and meta["max"].is_integer()
                if is_int:
                    user_inputs[key] = st.number_input(
                        meta["label"],
                        min_value=int(meta["min"]),
                        max_value=int(meta["max"]),
                        value=int(meta["default"]),
                        step=int(meta["step"]),
                        help=meta["help"],
                        key=key,
                    )
                else:
                    user_inputs[key] = st.number_input(
                        meta["label"],
                        min_value=float(meta["min"]),
                        max_value=float(meta["max"]),
                        value=float(meta["default"]),
                        step=float(meta["step"]),
                        help=meta["help"],
                        key=key,
                    )

        submitted = st.form_submit_button("🔮 Predict House Price")

# -------- Prediction & Output --------
with right_col:
    st.subheader("📌 Prediction")

    if submitted:
        # Build input DataFrame in correct column order
        input_data = pd.DataFrame([[user_inputs[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

        try:
            pred_price = model.predict(input_data)[0]
            pred_price = float(pred_price)

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Estimated Sale Price</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="prediction-value">${pred_price:,.0f}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                "<div class='metric-label'>This estimate is based on your inputs and the learned patterns from the Ames Housing dataset.</div>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("🔍 View input summary"):
                st.write(input_data.T.rename(columns={0: "Value"}))

        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")
    else:
        st.info("Fill in the property details on the left and click **Predict House Price** to see the model's estimate.")

st.markdown(
    "<div class='footer'>Built as an end-to-end ML project: data → model → evaluation → deployment.</div>",
    unsafe_allow_html=True
)
