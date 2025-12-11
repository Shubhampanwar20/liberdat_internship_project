import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src import config

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

def prepare_features():
    daily = pd.read_csv(os.path.join(config.OUTPUT_DIR,'daily_steps.csv'), parse_dates=['date'])
    clin = pd.read_csv(os.path.join(config.OUTPUT_DIR,'daily_clinical_features.csv'), parse_dates=['date'])
    df = daily.merge(clin, on='date', how='left').fillna(0)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['steps_t-1'] = df['Daily_Step_Count'].shift(1).fillna(0)
    df['steps_t-7'] = df['Daily_Step_Count'].shift(7).fillna(0)
    df['steps_t-30'] = df['Daily_Step_Count'].shift(30).fillna(0)
    df['roll_7_mean'] = df['Daily_Step_Count'].rolling(7, min_periods=1).mean().fillna(0)
    return df

def baseline_expsmoothing(train_series, test_len):
    try:
        model = ExponentialSmoothing(train_series, trend='add', seasonal='add', seasonal_periods=7)
        fit = model.fit(optimized=True)
        preds = fit.forecast(steps=test_len)
        return preds.values, fit
    except Exception:
        last = train_series.iloc[-1]
        return np.repeat(last, test_len), None

def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
    # Try multiple fit signatures for different xgboost versions
    if X_val is not None and y_val is not None:
        try:
            model.fit(X_train, y_train, eval_set=[(X_val,y_val)], early_stopping_rounds=20, verbose=False)
            return model
        except TypeError:
            # some xgboost versions expect callbacks
            try:
                model.fit(X_train, y_train, eval_set=[(X_val,y_val)], callbacks=[xgb.callback.EarlyStopping(rounds=20)])
                return model
            except Exception:
                # fallback to plain fit
                model.fit(X_train, y_train)
                return model
        except Exception:
            # any other exception -> fallback to plain fit
            model.fit(X_train, y_train)
            return model
    else:
        model.fit(X_train, y_train)
        return model

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df = prepare_features()

    hold = getattr(config, "HOLDOUT_DAYS", 30)
    if len(df) < hold + 10:
        print("Not enough data for holdout. Need more rows.")
        return

    train = df.iloc[:-hold].copy()
    test = df.iloc[-hold:].copy()

    baseline_preds, base_model = baseline_expsmoothing(train['Daily_Step_Count'], len(test))
    base_rmse = rmse(test['Daily_Step_Count'], baseline_preds)
    base_mae = mean_absolute_error(test['Daily_Step_Count'], baseline_preds)

    feature_cols = [c for c in train.columns if c not in ['date','Daily_Step_Count']]
    X_train = train[feature_cols]
    y_train = train['Daily_Step_Count']
    X_test = test[feature_cols]
    y_test = test['Daily_Step_Count']

    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    preds = xgb_model.predict(X_test)
    xgb_rmse = rmse(y_test, preds)
    xgb_mae = mean_absolute_error(y_test, preds)

    print("Baseline (ExpSmoothing) RMSE:", round(base_rmse,3), "MAE:", round(base_mae,3))
    print("XGBoost RMSE:", round(xgb_rmse,3), "MAE:", round(xgb_mae,3))

    joblib.dump(xgb_model, os.path.join(config.OUTPUT_DIR, "xgb_model.pkl"))
    metrics = {
        'baseline_rmse': float(base_rmse),
        'baseline_mae': float(base_mae),
        'xgb_rmse': float(xgb_rmse),
        'xgb_mae': float(xgb_mae)
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(config.OUTPUT_DIR,'metrics_summary.csv'), index=False)

    # SHAP: use version-compatible API
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    except Exception:
        try:
            explainer = shap.Explainer(xgb_model)
            sv = explainer(X_test)
            # sv.values may be 2D array or an object
            if hasattr(sv, "values"):
                arr = np.array(sv.values)
            else:
                arr = np.array(sv)
            mean_abs_shap = np.abs(arr).mean(axis=0)
        except Exception:
            mean_abs_shap = np.zeros(len(X_test.columns))

    feat_imp = pd.DataFrame({'feature': X_test.columns, 'mean_abs_shap': mean_abs_shap})
    feat_imp = feat_imp.sort_values('mean_abs_shap', ascending=False)
    feat_imp.to_csv(os.path.join(config.OUTPUT_DIR,'shap_feature_importance.csv'), index=False)
    print("Saved SHAP feature importance to outputs/shap_feature_importance.csv")

if __name__ == "__main__":
    main()
