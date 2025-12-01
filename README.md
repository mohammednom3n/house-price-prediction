# 🏠 Ames House Price Prediction – Production ML System

A full end-to-end Machine Learning project that predicts residential sale prices using the Ames Housing dataset. The system demonstrates the complete ML engineering pipeline — from data preparation and model comparison to feature selection, deployment as an API, and integration with an interactive frontend.

**🔗Live App:** https://house-price-predicts.streamlit.app/

## 🔗 Live API
Swagger Docs: https://house-price-prediction-kox7.onrender.com/docs

## 🚀 Project Highlights
✅ Real-world regression problem on tabular data  
✅ Multiple model benchmarking (LightGBM vs CatBoost)  
✅ 5-Fold cross-validation for stable evaluation  
✅ Feature importance aggregation & selection  
✅ Reduced-feature retraining for production robustness  
✅ Final model deployed as a FastAPI REST API on Render  
✅ Streamlit frontend consuming the live API  
✅ Docker containerization for backend and frontend

## 🧠 Problem Statement
Predict the final sale price of a residential home based on architectural, size, and quality attributes.

## 🛠️ Tech Stack
Data & Modeling:
- Python
- Pandas, NumPy, Scikit-Learn
- CatBoost
- LightGBM

Deployment:
- FastAPI
- Uvicorn
- Joblib

Frontend:
- Streamlit

Infrastructure:
- Docker
- Render (cloud hosting)

## 🔄 Modeling Workflow

### Pipeline Construction
Initial pipelines were built using missing value imputation (median strategy), standard scaling for numeric variables, one-hot encoding for categorical variables, and integrated sklearn Pipelines with ColumnTransformer.

### Model Comparison

| Model | CV R² Mean | Std |
|------|-------------|------|
| LightGBM | 0.9087 | ±0.0080 |
| CatBoost | 0.9128 | ±0.0081 |

CatBoost was selected for the final system placement based on superior generalization and lower error metrics.

### Feature Selection
Model feature importance was extracted and aggregated back to original features after categorical expansion. The top 15 most predictive numeric features were selected to retrain the production model:

OverallQual, GrLivArea, 1stFlrSF, TotalBsmtSF, BsmtFinSF1, LotArea, GarageCars, TotRmsAbvGrd, 2ndFlrSF, YearBuilt, GarageArea, FullBath, OverallCond, YearRemodAdd, MSSubClass

This resulted in a simpler deployment input schema, higher interpretability, and reduced overfitting risk.

### Final Production Model Performance

| Metric | Result |
|-------|---------|
| R² | 0.9107 |
| MAE | $15,903 |
| RMSE | $22,211 |
| MAPE | 9.61% |

## 🚀 System Architecture

[User] → [Streamlit Frontend] → (HTTP POST) → [FastAPI REST API – Render] → [CatBoost Production Model]

## 🌐 Deployment

Backend API:
- Hosted on Render
- POST /predict endpoint
- Swagger docs: https://house-price-prediction-kox7.onrender.com/docs

Frontend:
- Interactive Streamlit web UI
- Numeric-form property input
- Sends prediction requests to the live API
- Displays formatted price output

## 🐳 Dockerized Deployment
Both backend and frontend are containerized, ensuring reproducibility, consistent environments, and cloud portability.

## 🗂️ Project Structure

.
├── api/main.py        (FastAPI backend)
├── models/ames_house_price_production.pkl
├── app.py             (Streamlit frontend)
├── notebooks/
│   ├── eda.ipynb
│   ├── training.ipynb
│   └── feature_selection.ipynb
├── requirements-dev.txt
├── requirements-prod.txt
├── requirements.txt
├── Dockerfile
└── README.md

## ▶️ Run Locally

Install dependencies:
pip install -r requirements.txt

Run frontend:
streamlit run app.py

Run backend:
uvicorn api.main:app --reload

## 🎯 Key ML Engineering Themes Demonstrated
✅ Real-world dataset modeling  
✅ K-fold cross validation  
✅ Feature importance aggregation & selection  
✅ Production retraining  
✅ Clean sklearn pipelines  
✅ ML REST microservice deployment  
✅ API/frontend integration  
✅ Dockerized environments

## 👨‍💻 Author
Mohammed Noman  
Machine Learning Engineer

## ⭐ Acknowledgments
Ames Housing Dataset (Kaggle)  
CatBoost & LightGBM open-source communities

## ✅ Project Status
✔ Final model trained  
✔ API deployed  
✔ Frontend integrated  
✔ Dockerized system complete  

THIS PROJECT IS PRODUCTION READY.
