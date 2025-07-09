import pickle
import xgboost as xgb
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model as load_keras_model
from app.config import MODEL_FILE_PATHS

# =====================
# LSTM Feature-Konfiguration
# =====================

LSTM_FEATURES = [
    'store_nbr', 'day', 'week', 'month', 'day_of_week', 'is_weekend', 'onpromotion',
    'dcoilwtico', 'oil_price_1d_lag', 'oil_price_7d_avg', 'oil_price_14d_avg', 'oil_price_30d_avg',
    'is_holiday', 'unit_sales_1d_lag', 'unit_sales_7d_avg', 'unit_sales_14d_avg', 'unit_sales_30d_avg'
]

# =====================
# SCALER LADEN
# =====================

scaler_x = joblib.load(MODEL_FILE_PATHS["scaler_x_lstm"])
scaler_y = joblib.load(MODEL_FILE_PATHS["scaler_y_lstm"])

# =====================
# MODELLE LADEN
# =====================

def load_xgboost_model():
    with open(MODEL_FILE_PATHS["xgboost"], "rb") as f:
        return pickle.load(f)

def load_lstm_model():
    return load_keras_model(MODEL_FILE_PATHS["lstm"], compile=False)

def load_hybrid_model():
    return "HYBRID"  # Dummy-Objekt als Platzhalter für das Hybridmodell

# =====================
# PREDICT FUNKTION
# =====================

def predict(model, input_data, model_choice=None, lstm_model=None, xgb_model=None):
    input_data = input_data.copy()
    input_data = input_data.drop(columns=["unit_sales"], errors="ignore")

    if model_choice == "hybrid":
        if lstm_model is None or xgb_model is None:
            raise ValueError("Hybridmodell benötigt LSTM- und XGBoost-Modelle.")

        # LSTM vorbereiten
        input_lstm = input_data.sort_values("date").tail(30)
        missing = [f for f in LSTM_FEATURES if f not in input_lstm.columns]
        if missing:
            raise ValueError(f"Fehlende Features für LSTM: {missing}")
        input_scaled = scaler_x.transform(input_lstm[LSTM_FEATURES])
        input_array = np.reshape(input_scaled, (1, 30, len(LSTM_FEATURES)))
        y_pred_lstm = scaler_y.inverse_transform(lstm_model.predict(input_array))[0][0]

        # XGBoost vorbereiten
        input_xgb = input_data.drop(columns=["date"], errors="ignore")
        y_pred_xgb = xgb_model.predict(input_xgb)[0]

        # Kombination mit dynamischer Gewichtung
        delta = abs(y_pred_xgb - y_pred_lstm)
        alpha = 0.65 - 0.1 if delta > 8 else 0.65
        hybrid_pred = alpha * y_pred_lstm + (1 - alpha) * y_pred_xgb
        return hybrid_pred

    # LSTM Modell
    if hasattr(model, "predict") and "Sequential" in str(type(model)):
        if input_data.shape[0] < 30:
            raise ValueError("Mindestens 30 Tage Historie erforderlich für LSTM-Vorhersage.")
        input_data = input_data.sort_values("date").tail(30)
        missing = [f for f in LSTM_FEATURES if f not in input_data.columns]
        if missing:
            raise ValueError(f"Fehlende Features für LSTM: {missing}")
        input_scaled = scaler_x.transform(input_data[LSTM_FEATURES])
        input_array = np.reshape(input_scaled, (1, 30, len(LSTM_FEATURES)))
        prediction_scaled = model.predict(input_array)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        return prediction[0][0]

    # XGBoost Modell
    elif hasattr(model, "predict"):
        input_data = input_data.drop(columns=["date"], errors="ignore")
        return model.predict(input_data)[0]

    raise ValueError("Unbekannter Modelltyp für die Vorhersage.")

# =====================
# PREPROCESS-FUNKTION FÜR LSTM SEQUENZEN
# =====================

def preprocess_lstm_sequence(store_id, item_id, date, df_stores, df_items, df_filtered, df_oil, df_holidays):
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    date = pd.to_datetime(date)

    full_history = df_filtered[
        (df_filtered['store_nbr'] == store_id) &
        (df_filtered['item_nbr'] == item_id) &
        (df_filtered['date'] < date)
    ].sort_values("date").copy()

    if full_history.shape[0] < 35:
        raise ValueError(f"Mindestens 35 Tage Historie erforderlich für LSTM-Vorhersage. Aktuell: {full_history.shape[0]}")

    # Lag-Features berechnen
    full_history['unit_sales_1d_lag'] = full_history['unit_sales'].shift(1)
    full_history['unit_sales_7d_avg'] = full_history['unit_sales'].shift(1).rolling(7).mean()
    full_history['unit_sales_14d_avg'] = full_history['unit_sales'].shift(1).rolling(14).mean()
    full_history['unit_sales_30d_avg'] = full_history['unit_sales'].shift(1).rolling(30).mean()

    history_df = full_history.dropna(subset=[
        'unit_sales_1d_lag', 'unit_sales_7d_avg',
        'unit_sales_14d_avg', 'unit_sales_30d_avg'
    ]).tail(30).copy()

    if history_df.shape[0] < 30:
        raise ValueError(f"Nicht genügend gültige Lag-Daten. Nur {history_df.shape[0]} Zeilen nach Bereinigung.")

    # Zeitmerkmale
    history_df['day'] = history_df['date'].dt.day
    history_df['week'] = history_df['date'].dt.isocalendar().week
    history_df['month'] = history_df['date'].dt.month
    history_df['day_of_week'] = history_df['date'].dt.dayofweek
    history_df['is_weekend'] = history_df['day_of_week'].isin([5, 6]).astype(int)

    # Öl vorbereiten
    df_oil['date'] = pd.to_datetime(df_oil['date'])
    df_oil = df_oil.sort_values('date')
    df_oil['dcoilwtico'] = df_oil['dcoilwtico'].interpolate()
    df_oil['oil_price_1d_lag'] = df_oil['dcoilwtico'].shift(1)
    df_oil['oil_price_7d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(7).mean()
    df_oil['oil_price_14d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(14).mean()
    df_oil['oil_price_30d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(30).mean()

    history_df = history_df.merge(
        df_oil[['date', 'dcoilwtico', 'oil_price_1d_lag',
                'oil_price_7d_avg', 'oil_price_14d_avg', 'oil_price_30d_avg']],
        on='date', how='left'
    )

    # Feiertage
    df_holidays['date'] = pd.to_datetime(df_holidays['date'])
    holidays = df_holidays[
        (df_holidays['locale'] == 'National') &
        (df_holidays['transferred'] == False)
    ]
    history_df['is_holiday'] = history_df['date'].isin(holidays['date']).astype(int)

    # onpromotion behandeln
    history_df['onpromotion'] = history_df['onpromotion'].astype(str).str.lower().map({
        'true': 1, 'false': 0
    }).fillna(0).astype(int)

    # Letzter Check
    missing = [col for col in LSTM_FEATURES if col not in history_df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten im LSTM-Input: {missing}")

    return history_df[LSTM_FEATURES + ['date']]

