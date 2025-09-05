from flask_cors import CORS

from datetime import timedelta

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
port = int(os.environ.get('PORT', 5000))
app = Flask(__name__)
CORS(app)



 
def load_data(file_path, from_currency, to_currency):
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(by='ds')
    return df[(df['From'] == from_currency) & (df['To'] == to_currency)]
 
def train_arima(df, steps):
    model = ARIMA(df['y'], order=(1, 1, 1))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=steps)
    return forecast.tolist(), fitted_model
 
def train_prophet(df, steps):
    model = Prophet()
    df = df[['ds', 'y']]
    model.fit(df)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast['yhat'][-steps:].tolist(), model
 
def train_random_forest(df, steps):
    model = RandomForestRegressor()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['y'].values
    model.fit(X, y)
    future_X = np.arange(len(df), len(df) + steps).reshape(-1, 1)
    forecast = model.predict(future_X)
    return forecast.tolist(), model
 
def train_xgboost(df, steps):
    model = xgb.XGBRegressor()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['y'].values
    model.fit(X, y)
    future_X = np.arange(len(df), len(df) + steps).reshape(-1, 1)
    forecast = model.predict(future_X)
    return forecast.tolist(), model
 
def train_lstm(df, steps):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    X = np.arange(len(df)).reshape(-1, 1, 1)
    y = df['y'].values
    model.fit(X, y, epochs=50, verbose=0)
    future_X = np.arange(len(df), len(df) + steps).reshape(-1, 1, 1)
    forecast = model.predict(future_X)
    return forecast.flatten().tolist(), model
 
def train_sarima(df, steps):
    model = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=steps)
    return forecast.tolist(), fitted_model
 
def train_lightgbm(df, steps):
    model = LGBMRegressor()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['y'].values
    model.fit(X, y)
    future_X = np.arange(len(df), len(df) + steps).reshape(-1, 1)
    forecast = model.predict(future_X)
    return forecast.tolist(), model
 
def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }
 
@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    from_currency = data.get("from_currency")
    to_currency = data.get("to_currency")
    model_type = data.get("model", "arima")
    file_path = os.path.join(os.getcwd(), 'cleaned_exchange_rates.csv')
    start_date = pd.to_datetime(data.get("start_date"))
    end_date = pd.to_datetime(data.get("end_date"))
    steps = (end_date - start_date).days
    df = load_data(file_path, from_currency, to_currency)
    if df.empty:
        return jsonify({"error": "No data available for the selected currency pair"}), 400
   
    model_mapping = {
        "arima": train_arima,
        "prophet": train_prophet,
        "random_forest": train_random_forest,
        "xgboost": train_xgboost,
        "lstm": train_lstm,
        "sarima": train_sarima,
        "lightgbm": train_lightgbm
    }
   
    if model_type not in model_mapping:
        return jsonify({"error": "Invalid model type"}), 400
   
    forecasted_values, trained_model = model_mapping[model_type](df, steps)
 
    # Generate forecasted dates
    last_date = df['ds'].max()
    forecasted_dates = [last_date + timedelta(days=i) for i in range(1, steps + 1)]
 
    # Combine dates and forecasted values
    forecast_data = [{"date": date.strftime('%Y-%m-%d'), "forecast": value} for date, value in zip(forecasted_dates, forecasted_values)]
 
    return jsonify(forecast_data)
 
@app.route('/performance', methods=['POST'])
def performance():
    data = request.json
    from_currency = data.get("from_currency")
    to_currency = data.get("to_currency")
    model_type = data.get("model")
    file_path = os.path.join(os.getcwd(), 'cleaned_exchange_rates.csv')
   
    df = load_data(file_path, from_currency, to_currency)
    if df.empty:
        return jsonify({"error": "No data available for the selected currency pair"}), 400
   
    model_mapping = {
        "arima": train_arima,
        "prophet": train_prophet,
        "random_forest": train_random_forest,
        "xgboost": train_xgboost,
        "lstm": train_lstm,
        "sarima": train_sarima,
        "lightgbm": train_lightgbm
    }
   
    if model_type not in model_mapping:
        return jsonify({"error": "Invalid model type"}), 400
   
    forecasted_values, trained_model = model_mapping[model_type](df, 30)
    y_true = df['y'][-30:].values
    y_pred = np.array(forecasted_values)
    metrics = evaluate_model(y_true, y_pred)
   
    return jsonify(metrics)
 
app.run(debug=False,host='0.0.0.0', port=port)