# app/config.py

import os

# Basis-Verzeichnisse
DATA_PATH = r"C:\Users\49176\Desktop\Forecast_CSVs"
MODEL_PATH = r"C:\Users\49176\Desktop\Sales_Forecast\models"

# Einzelne Modellpfade korrekt zusammensetzen
MODEL_FILE_PATHS = {
    "xgboost": os.path.join(MODEL_PATH, "model_xgb_best.pkl"),
    "lstm": os.path.join(MODEL_PATH, "model_lstm_best.h5"),
    "hybrid": os.path.join(MODEL_PATH, "hybrid_spikeboost_model.pkl"),
    "scaler_x_lstm": os.path.join(MODEL_PATH, "scaler_x_lstm.pkl"),
    "scaler_y_lstm": os.path.join(MODEL_PATH, "scaler_y_lstm.pkl"),
}

