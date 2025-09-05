import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, request, jsonify
import os
 
app = Flask(__name__)
 
# Load and preprocess data
def load_data(file_path, from_currency, to_currency):
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(by='ds')
    return df[(df['From'] == from_currency) & (df['To'] == to_currency)]
 
# Train ARIMA model and forecast
def train_arima(df, p=1, d=1, q=1, steps=30):
    df['y_diff'] = df['y'].diff().dropna()
    model = ARIMA(df['y'], order=(p, d, q))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=steps)
    return forecast.tolist()
 
@app.route('/forecast', methods=['GET'])
def forecast():
    data = request.json 
    from_currency = "INR"
    to_currency = "USD"
    file_path = os.path.join(r'C:\Users\I587436\Documents\New folder\Template\HCLTECH_USECASE\Backend_python\clean_financial_data.csv')
   
    df = load_data(file_path, from_currency, to_currency)
    if df.empty:
        return jsonify({"error": "No data available for the selected currency pair"}), 400
   
    forecasted_values = train_arima(df)
    return jsonify({"from_currency": from_currency, "to_currency": to_currency, "forecast": forecasted_values})
 
if __name__ == "__main__":
    app.run(debug=True)