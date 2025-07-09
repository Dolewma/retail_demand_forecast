import pandas as pd
import os

# Lokaler Pfad zu den CSV-Dateien
LOCAL_DATA_PATH = r"C:\Users\49176\Desktop\Forecast_CSVs"

def load_data(data_path=LOCAL_DATA_PATH):
    stores = pd.read_csv(os.path.join(data_path, "stores.csv"))
    items = pd.read_csv(os.path.join(data_path, "items.csv"))
    transactions = pd.read_csv(os.path.join(data_path, "transactions.csv"))
    oil = pd.read_csv(os.path.join(data_path, "oil.csv"))
    holidays = pd.read_csv(os.path.join(data_path, "holidays_events.csv"))
    df_prepared = pd.read_csv(os.path.join(data_path, "df_prepared_guayas.csv"))
    return stores, items, transactions, oil, holidays, df_prepared


def preprocess_input_data(store_id, item_id, date, df_stores, df_items, df_filtered, df_oil, df_holidays):
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    date = pd.to_datetime(date)

    df_filtered = df_filtered[
        (df_filtered['store_nbr'] == store_id) &
        (df_filtered['item_nbr'] == item_id) &
        (df_filtered['date'] < date)
    ].sort_values("date")

    row = {
        'store_nbr': store_id,
        'day': date.day,
        'week': date.isocalendar().week,
        'month': date.month,
        'day_of_week': date.dayofweek,
        'is_weekend': int(date.dayofweek in [5, 6]),
        'onpromotion': 0
    }

    for col in [
        'unit_sales_1d_lag', 'unit_sales_7d_avg', 'unit_sales_14d_avg', 'unit_sales_30d_avg'
    ]:
        row[col] = df_filtered[col].iloc[-1] if col in df_filtered.columns and not df_filtered.empty else 0

    df_oil['date'] = pd.to_datetime(df_oil['date'])
    df_oil = df_oil.sort_values('date')
    df_oil['dcoilwtico'] = df_oil['dcoilwtico'].interpolate()
    df_oil['oil_price_1d_lag'] = df_oil['dcoilwtico'].shift(1)
    df_oil['oil_price_7d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(7).mean()
    df_oil['oil_price_14d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(14).mean()
    df_oil['oil_price_30d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(30).mean()

    oil_row = df_oil[df_oil['date'] < date].tail(1)
    if not oil_row.empty:
        row['dcoilwtico'] = oil_row['dcoilwtico'].values[0]
        row['oil_price_1d_lag'] = oil_row['oil_price_1d_lag'].values[0]
        row['oil_price_7d_avg'] = oil_row['oil_price_7d_avg'].values[0]
        row['oil_price_14d_avg'] = oil_row['oil_price_14d_avg'].values[0]
        row['oil_price_30d_avg'] = oil_row['oil_price_30d_avg'].values[0]
    else:
        for col in ['dcoilwtico', 'oil_price_1d_lag', 'oil_price_7d_avg', 'oil_price_14d_avg', 'oil_price_30d_avg']:
            row[col] = 0

    df_holidays['date'] = pd.to_datetime(df_holidays['date'])
    national_holidays = df_holidays[(df_holidays['locale'] == 'National') & (df_holidays['transferred'] == False)]
    row['is_holiday'] = int(date in national_holidays['date'].values)

    ordered_columns = [
        'store_nbr', 'day', 'week', 'month', 'day_of_week', 'is_weekend', 'onpromotion',
        'dcoilwtico', 'oil_price_1d_lag', 'oil_price_7d_avg', 'oil_price_14d_avg', 'oil_price_30d_avg',
        'is_holiday', 'unit_sales_1d_lag', 'unit_sales_7d_avg', 'unit_sales_14d_avg', 'unit_sales_30d_avg'
    ]

    return pd.DataFrame([row])[ordered_columns]


def preprocess_lstm_sequence(store_id, item_id, forecast_date, df_stores, df_items, df_full, df_oil, df_holidays):
    forecast_date = pd.to_datetime(forecast_date)
    df_full['date'] = pd.to_datetime(df_full['date'])

    # ⏳ Vollständige Historie für Lags laden
    history_all = df_full[
        (df_full['store_nbr'] == store_id) &
        (df_full['item_nbr'] == item_id) &
        (df_full['date'] < forecast_date)
    ].sort_values("date").copy()

    if history_all.shape[0] < 35:
        raise ValueError(f"Mindestens 35 Tage Historie erforderlich. Aktuell: {history_all.shape[0]}")

    # 🧠 Lag-Features berechnen
    history_all['unit_sales_1d_lag'] = history_all['unit_sales'].shift(1)
    history_all['unit_sales_7d_avg'] = history_all['unit_sales'].shift(1).rolling(7).mean()
    history_all['unit_sales_14d_avg'] = history_all['unit_sales'].shift(1).rolling(14).mean()
    history_all['unit_sales_30d_avg'] = history_all['unit_sales'].shift(1).rolling(30).mean()

    # Nur vollständige Zeilen
    df_seq = history_all.dropna(subset=[
        'unit_sales_1d_lag', 'unit_sales_7d_avg', 'unit_sales_14d_avg', 'unit_sales_30d_avg'
    ]).tail(30).copy()

    if df_seq.shape[0] < 30:
        raise ValueError(f"Nicht genügend gültige Lag-Zeilen. Nur {df_seq.shape[0]} vorhanden.")

    # Ölpreis-Merkmale
    df_oil['date'] = pd.to_datetime(df_oil['date'])
    df_oil = df_oil.sort_values('date')
    df_oil['dcoilwtico'] = df_oil['dcoilwtico'].interpolate()
    df_oil['oil_price_1d_lag'] = df_oil['dcoilwtico'].shift(1)
    df_oil['oil_price_7d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(7).mean()
    df_oil['oil_price_14d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(14).mean()
    df_oil['oil_price_30d_avg'] = df_oil['dcoilwtico'].shift(1).rolling(30).mean()

    df_seq = df_seq.merge(
        df_oil[['date', 'dcoilwtico', 'oil_price_1d_lag', 'oil_price_7d_avg',
                'oil_price_14d_avg', 'oil_price_30d_avg']],
        on='date', how='left'
    )

    # Feiertage
    df_holidays['date'] = pd.to_datetime(df_holidays['date'])
    national_holidays = df_holidays[(df_holidays['locale'] == 'National') & (df_holidays['transferred'] == False)]
    df_seq['is_holiday'] = df_seq['date'].isin(national_holidays['date']).astype(int)

    # onpromotion
    df_seq['onpromotion'] = df_seq['onpromotion'].astype(str).str.lower().map({'true': 1, 'false': 0}).fillna(0).astype(int)

    # Zeitmerkmale
    df_seq['day'] = df_seq['date'].dt.day
    df_seq['week'] = df_seq['date'].dt.isocalendar().week
    df_seq['month'] = df_seq['date'].dt.month
    df_seq['day_of_week'] = df_seq['date'].dt.dayofweek
    df_seq['is_weekend'] = df_seq['day_of_week'].isin([5, 6]).astype(int)

    ordered_columns = [
        'store_nbr', 'day', 'week', 'month', 'day_of_week', 'is_weekend', 'onpromotion',
        'dcoilwtico', 'oil_price_1d_lag', 'oil_price_7d_avg', 'oil_price_14d_avg', 'oil_price_30d_avg',
        'is_holiday', 'unit_sales_1d_lag', 'unit_sales_7d_avg', 'unit_sales_14d_avg', 'unit_sales_30d_avg'
    ]

    return df_seq[ordered_columns + ['date']]
