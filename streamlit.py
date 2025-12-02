import requests
import streamlit as st
import pandas as pd

# ======================
# 🔧 CONFIG
# ======================

API_URL = "https://house-price-prediction-kox7.onrender.com/predict"

st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

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

FEATURE_META = {
    "OverallQual": {
        "label": "Overall Material & Finish Quality (1–10)",
        "help": "Overall quality of the house (10 = excellent, 1 = very poor).",
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
# 🎨 Styling
# ======================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
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
        border-radius: 0.9rem;
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
    '<div class="subtitle">Frontend in Streamlit • Backend FastAPI on Render • CatBoost regression model on top 15 numeric features.</div>',
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1.5, 1])

# ---- Sidebar ----
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        This app sends your inputs to a **FastAPI ML service** hosted on Render:

        - Model: CatBoost Regressor  
        - Trained on Ames Housing dataset  
        - Uses 15 most important numeric features  
        - Deployed as a REST API (`/predict`)
        """
    )
    st.markdown("---")
    st.subheader("📊 Validation Metrics (Reduced Model)")
    st.write("- R² ≈ **0.91**")
    st.write("- MAE ≈ **$15.9K**")
    st.write("- RMSE ≈ **$22.2K**")
    st.write("- MAPE ≈ **≈9.6%**")
    st.markdown("---")
    st.caption("Tip: Start with defaults and tweak values to see how price responds.")


# ---- Inputs ----
with left_col:
    st.subheader("📝 Enter Property Details")

    user_inputs = {}
    with st.form("house_form"):
        c1, c2 = st.columns(2)

        with c1:
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

        with c2:
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


# ---- Prediction ----
with right_col:
    st.subheader("📌 Prediction")

    if submitted:
        # Build payload exactly how the API expects it
        payload = {col: float(user_inputs[col]) for col in FEATURE_COLUMNS}

        try:
            with st.spinner("Calling prediction API..."):
                resp = requests.post(API_URL, json=payload, timeout=60)

            if resp.status_code == 200:
                data = resp.json()

                # Adjust this if your API returns a different field name
                pred  = data.get("Estimated Price")

                try:
                    price_val = float(pred)
                except Exception:
                    st.error("API response format is not as expected:")
                    st.write(data)
                else:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Estimated Sale Price</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="prediction-value">${price_val:,.0f}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "<div class='metric-label'>Powered by a CatBoost model deployed on FastAPI (Render).</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    with st.expander("🔍 View input summary"):
                        df_in = pd.DataFrame.from_dict(user_inputs, orient="index", columns=["Value"])
                        st.write(df_in)

            else:
                st.error(f"❌ API returned status code {resp.status_code}")
                st.write(resp.text)

        except requests.exceptions.RequestException as e:
            st.error("❌ Could not reach the prediction API.")
            st.write(str(e))
    else:
        st.info("Fill the details on the left and click **Predict House Price** to get an estimate.")

st.markdown(
    "<div class='footer'>Built as an end-to-end ML system: training → API → Streamlit frontend.</div>",
    unsafe_allow_html=True
)
