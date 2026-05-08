from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
import joblib
import pandas as pd
import io

app = FastAPI()

model = tf.keras.models.load_model('model_stock_price.keras')
scaler = joblib.load('scaler.pkl')

templates = Jinja2Templates(directory="templates")

class StockData(BaseModel):
    prices: list

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(data: StockData):
    input_data = np.array(data.prices).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    final_input = scaled_data.reshape(1, 30, 1)
    prediction = model.predict(final_input)
    
    rescaled_prediction = scaler.inverse_transform(prediction)
    return {"prediction": float(rescaled_prediction[0][0])}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    contents =await file.read()
    df = pd.read_csv(io.BytesIO(contents))
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