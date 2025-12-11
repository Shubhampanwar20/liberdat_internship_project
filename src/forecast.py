import os
import pandas as pd
import numpy as np
import joblib
from src import config
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb

def produce_forecast():
    # Load processed data
    daily = pd.read_csv(os.path.join(config.OUTPUT_DIR,'daily_steps.csv'), parse_dates=['date'])
    clin = pd.read_csv(os.path.join(config.OUTPUT_DIR,'daily_clinical_features.csv'), parse_dates=['date'])
    data = daily.merge(clin, on='date', how='left').fillna(0).sort_values('date').reset_index(drop=True)

    # Same feature engineering as modeling.py
    data['day_of_week'] = data['date'].dt.dayofweek
    data['week_of_year'] = data['date'].dt.isocalendar().week.astype(int)
    data['steps_t-1'] = data['Daily_Step_Count'].shift(1).fillna(0)
    data['steps_t-7'] = data['Daily_Step_Count'].shift(7).fillna(0)
    data['steps_t-30'] = data['Daily_Step_Count'].shift(30).fillna(0)
    data['roll_7_mean'] = data['Daily_Step_Count'].rolling(7, min_periods=1).mean().fillna(0)

    FEATURE_COLS = ['day_of_week','week_of_year','steps_t-1','steps_t-7','steps_t-30','roll_7_mean']

    X_full = data[FEATURE_COLS]
    y_full = data['Daily_Step_Count']

    # Load trained model
    model_path = os.path.join(config.OUTPUT_DIR, "xgb_model.pkl")
    model = joblib.load(model_path)

    # Exponential smoothing trend
    try:
        trend_model = ExponentialSmoothing(y_full, trend='add', seasonal='add', seasonal_periods=7).fit()
        trend_forecast = trend_model.forecast(steps=config.FORECAST_DAYS)
    except Exception:
        trend_forecast = np.repeat(y_full.iloc[-1], config.FORECAST_DAYS)

    # Prepare iterative forecast
    prev_vals = list(y_full.values[-30:])  # last 30 days
    future_dates = pd.date_range(data['date'].max() + pd.Timedelta(days=1),
                                 periods=config.FORECAST_DAYS, freq='D')
    rows = []

    for i, d in enumerate(future_dates):
        row = {}
        row['day_of_week'] = d.weekday()
        row['week_of_year'] = int(d.isocalendar()[1])
        row['steps_t-1'] = prev_vals[-1]
        row['steps_t-7'] = prev_vals[-7] if len(prev_vals) >= 7 else prev_vals[-1]
        row['steps_t-30'] = prev_vals[-30] if len(prev_vals) >= 30 else prev_vals[-1]
        row['roll_7_mean'] = np.mean(prev_vals[-7:])  # NEW: include rolling mean

        X_row = pd.DataFrame([row])[FEATURE_COLS]
        pred = model.predict(X_row)[0]
        pred = int(round(pred))

        prev_vals.append(pred)
        rows.append({'Date': d.date(), 'Predicted_Steps': pred})

    # Build forecast dataframe
    forecast_df = pd.DataFrame(rows)
    forecast_df['Trend_Component'] = trend_forecast[:len(forecast_df)]
    forecast_df['Exogenous_Impact'] = forecast_df['Predicted_Steps'] - forecast_df['Trend_Component']

    outpath = os.path.join(config.OUTPUT_DIR, "365_day_forecast.csv")
    forecast_df.to_csv(outpath, index=False)
    print("Saved forecast to", outpath)

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    produce_forecast()

if __name__ == "__main__":
    main()
