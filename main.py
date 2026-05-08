from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
import joblib
import pandas as pd
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = FastAPI()

model = tf.keras.models.load_model('model_stock_price.keras')
scaler = joblib.load('scaler.pkl')

templates = Jinja2Templates(directory="templates")

class StockData(BaseModel):
    prices: list

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/stock-info")
async def stock_info():
    df = pd.read_csv('googl_data_2020_2025.csv', skiprows=3)  # skip 3 baris header
    df.columns = ['Date', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    last_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])
    change = last_price - prev_price
    pct = (change / prev_price) * 100
    
    return {
        "price": round(last_price, 2),
        "change": round(change, 2),
        "pct": round(pct, 2),
        "last_30": df['Close'].tail(30).tolist(),
        "volumes": df['Volume'].tail(30).tolist()
    }

@app.get("/model-metrics")
async def model_metrics():
    # Load data
    df = pd.read_csv('googl_data_2020_2025.csv', skiprows=3)
    df.columns = ['Date', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    close_prices = df['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close_prices)
    
    # Buat sequences (Windowing)
    X, y_true = [], []
    window_size = 30
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, 0])
        y_true.append(scaled[i, 0])
    
    X = np.array(X).reshape(-1, window_size, 1)
    y_true = np.array(y_true).reshape(-1, 1)
    
    # Predict
    y_pred_scaled = model.predict(X)
    
    # CRITICAL FIX: Pastikan y_pred hanya mengambil kolom pertama jika model multi-output
    # atau pastikan shape-nya sama dengan y_true
    if len(y_pred_scaled.shape) > 2: # Jika [samples, steps, features]
        y_pred_scaled = y_pred_scaled[:, 0, 0] 
    elif y_pred_scaled.shape[1] > 1: # Jika [samples, steps]
        y_pred_scaled = y_pred_scaled[:, 0]
        
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    
    # Inverse transform ke harga asli
    y_true_real = scaler.inverse_transform(y_true).flatten()
    y_pred_real = scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Metrics calculation
    mae = float(mean_absolute_error(y_true_real, y_pred_real))
    rmse = float(np.sqrt(mean_squared_error(y_true_real, y_pred_real)))
    
    # R2 Score
    ss_res = np.sum((y_true_real - y_pred_real) ** 2)
    ss_tot = np.sum((y_true_real - np.mean(y_true_real)) ** 2)
    r2 = float(1 - (ss_res / ss_tot))
    
    # 30D Volatility (Annualized)
    last_31 = df['Close'].tail(31).values
    log_returns = np.diff(np.log(last_31))
    volatility = float(np.std(log_returns) * np.sqrt(252) * 100)
    
    # Directional Accuracy (apakah naik/turunnya benar?)
    actual_diff = np.diff(y_true_real)
    pred_diff = np.diff(y_pred_real)
    dir_acc = float(np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100)
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "volatility": round(volatility, 2),
        "directional_accuracy": round(dir_acc, 2)
    }

@app.post("/predict")
async def predict(data: StockData):
    input_data = np.array(data.prices).reshape(-1, 1)   
    scaled_data = scaler.transform(input_data)           
    final_input = scaled_data.reshape(1, 30, 1)          
    
    prediction = model.predict(final_input)
    rescaled_prediction = scaler.inverse_transform(prediction)
    return {"prediction": float(rescaled_prediction[0][0])}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents), skiprows=3)  # tambah skiprows=3
    df.columns = ['Date', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    last_30_days = df['Close'].tail(30).values.reshape(-1, 1)

    if len(last_30_days) < 30:
        return {"error": f"Data kurang! Butuh 30 baris, hanya ada {len(last_30_days)}"}
    
    # Scaling
    scaled_data = scaler.transform(last_30_days)
    
    # Reshape & Predict
    final_input = scaled_data.reshape(1, 30, 1)
    prediction = model.predict(final_input)
    
    # Inverse Scale
    rescaled_prediction = scaler.inverse_transform(prediction)
    
    return {"prediction": float(rescaled_prediction[0][0])}

# run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)