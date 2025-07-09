import streamlit as st
import datetime
import numpy as np
import pandas as pd

from app.config import MODEL_FILE_PATHS
from data.data_utils import load_data, preprocess_input_data, preprocess_lstm_sequence
from model.model_utils import (
    load_xgboost_model,
    load_lstm_model,
    load_hybrid_model,
    predict,
)

# ======================
# Caching für Daten
# ======================

@st.cache_data
def load_cached_data():
    return load_data()

@st.cache_data
def preprocess_oil(df_oil):
    df_oil['date'] = pd.to_datetime(df_oil['date'])
    df_oil = df_oil.sort_values('date')
    df_oil['dcoilwtico'] = df_oil['dcoilwtico'].interpolate()
    df_oil['oil_price_1d_lag'] = df_oil['dcoilwtico'].shift(1)
    df_oil['oil_price_7d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(7).mean()
    df_oil['oil_price_14d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(14).mean()
    df_oil['oil_price_30d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(30).mean()
    return df_oil

@st.cache_data
def get_valid_store_item_mapping(df_train, df_items):
    df_train['date'] = pd.to_datetime(df_train['date'])
    top_families = df_items['family'].value_counts().nlargest(3).index.tolist()
    top_items = df_items[df_items['family'].isin(top_families)]['item_nbr'].unique()

    store_item_map = {}
    for store in df_train['store_nbr'].unique():
        filtered = df_train[
            (df_train['store_nbr'] == store) &
            (df_train['item_nbr'].isin(top_items)) &
            (df_train['date'] < '2014-04-01')
        ]
        item_counts = filtered['item_nbr'].value_counts()
        valid_items = item_counts[item_counts >= 35].index.tolist()
        if valid_items:
            store_item_map[store] = valid_items

    return store_item_map

# ======================
# Main App
# ======================

def main():
    st.title("🛒 Guayas Sales Forecasting")

    model_ui_choice = st.selectbox("🧠 Wähle dein Prognosemodell:", [
        "🛍️ Aggressive Sales Estimate",      # XGBoost
        "🧾 Conservative Sales Estimate",    # LSTM
        "⚖️ Balanced Sales Estimate"         # Hybrid
    ])

    model_choice = {
        "🛍️ Aggressive Sales Estimate": "xgboost",
        "🧾 Conservative Sales Estimate": "lstm",
        "⚖️ Balanced Sales Estimate": "hybrid"
    }[model_ui_choice]

    # Datumsbereich für gültige Vorhersagen
    min_date = datetime.date(2014, 1, 1)
    max_date = datetime.date(2014, 3, 31)

    try:
        df_stores, df_items, df_transactions, df_oil, df_holidays, df_train = load_cached_data()
        df_oil = preprocess_oil(df_oil)
        st.success("✅ Daten geladen & vorbereitet")
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Daten: {e}")
        return

    store_item_map = get_valid_store_item_mapping(df_train, df_items)
    valid_stores = sorted(store_item_map.keys())

    if not valid_stores:
        st.error("❌ Keine gültigen Stores mit ausreichender Historie vorhanden.")
        return

    store_id = st.selectbox("📍 Store", valid_stores)
    valid_items = store_item_map.get(store_id, [])
    if not valid_items:
        st.error("❌ Kein Item mit ausreichender Historie gefunden für diesen Store.")
        return

    item_id = st.selectbox("📦 Item", sorted(valid_items))

    # Eingeschränkte Datumsauswahl
    date = st.date_input(
        "📅 Forecast Date (nur Jan–März 2014)",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    if not (min_date <= date <= max_date):
        st.error(f"❌ Das Vorhersagedatum muss zwischen {min_date} und {max_date} liegen.")
        return

    # Modell laden
    try:
        if model_choice == "hybrid":
            lstm_model = load_lstm_model()
            xgb_model = load_xgboost_model()
            model = load_hybrid_model()  # Dummy
        elif model_choice == "lstm":
            model = load_lstm_model()
            lstm_model = None
            xgb_model = None
        else:  # xgboost
            model = load_xgboost_model()
            lstm_model = None
            xgb_model = None
        st.success("✅ Modell geladen")
    except Exception as e:
        st.error(f"❌ Fehler beim Laden des Modells: {e}")
        return

    if st.button("📊 Get Forecast"):
        try:
            st.write("🔄 Eingabedaten werden vorbereitet...")

            if model_choice in ["lstm", "hybrid"]:
                input_data = preprocess_lstm_sequence(
                    store_id, item_id, date,
                    df_stores, df_items, df_train,
                    df_oil, df_holidays
                )
            else:
                input_data = preprocess_input_data(
                    store_id, item_id, date,
                    df_stores, df_items, df_train,
                    df_oil, df_holidays
                )

            prediction = predict(
                model=model,
                input_data=input_data,
                model_choice=model_choice,
                lstm_model=lstm_model,
                xgb_model=xgb_model
            )
            value = int(np.ceil(np.squeeze(prediction)))

            st.success(f"✅ Vorhersage für {date}: {value} Einheiten")

        except Exception as e:
            st.error(f"❌ Fehler bei der Vorhersage: {str(e)}")


if __name__ == "__main__":
    main()
