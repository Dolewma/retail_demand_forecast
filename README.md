# Retail Demand Forecasting App

This project involved developing an interactive application to forecast product sales across retail stores. The underlying models were built and trained as part of a machine learning pipeline and integrated into a user-facing Streamlit web interface.

Project Goal
The objective was to predict product demand for different stores, enabling better decision-making in inventory, procurement, and supply planning.

Application Features
The web app allows users to select a store, item, and date (within January to March 2014). Based on this input, it generates a sales forecast using one of three available models.

Exploratory Data Analysis
Key insights included:

Seasonal and weekly patterns in sales

Influence of holidays and promotions

Trends in oil prices over time

Behavioral differences across stores and product families
Tools used: pandas, matplotlib, seaborn, plotly
Documented in: course_project_week_1.ipynb

Modeling and Optimization
Three model types were developed and evaluated:

XGBoost: Gradient boosted trees, tends to produce higher, more aggressive forecasts

LSTM: Deep learning model for time series, generates more stable and conservative predictions

Hybrid Model: Combines XGBoost and LSTM, dynamically adjusting weights based on the difference between individual predictions

Feature Engineering
Key features included:

Lag variables (1, 7, 14, 30 days)

Moving averages

Time-based features (weekday, month, holidays)

Oil price indicators
Development steps documented in: course_project_week_2_and_3_.ipynb

Streamlit App Capabilities

Select between XGBoost, LSTM, or Hybrid model

Input: store, item, and forecast date (Jan–Mar 2014)

Automatic data preparation and feature engineering

Forecast output with integrated error handling

Project Structure

retail_demand_forecast/
├── app/
│   ├── main.py
│   ├── config.py
├── model/
│   ├── model_utils.py
├── data/
│   ├── data_utils.py
│   └── __init__.py
├── models/       # .pkl and .h5 files
├── requirements.txt
├── README.md
└── course_project_week_*.ipynb


Conclusion
This project combines traditional machine learning (XGBoost) with deep learning (LSTM) in a production-oriented Streamlit interface. Through advanced feature engineering and model blending, it delivers realistic and interpretable demand forecasts that can be applied in real-world retail scenarios.

Link to project in comments or profile.

