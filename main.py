from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Lovable-ൽ നിന്നും മറ്റ് വെബ്‌സൈറ്റുകളിൽ നിന്നും ഉള്ള റിക്വസ്റ്റുകൾ സ്വീകരിക്കാൻ (CORS Enable ചെയ്യൽ)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model-ഉം Dataset-ഉം Load ചെയ്യുന്നു
model = pickle.load(open('car_model.pkl', 'rb'))
df = pd.read_csv('cardekho_dataset.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Frontend-ൽ നിന്ന് വരുന്ന വിവരങ്ങളുടെ ഘടന (Data Format)
class CarInput(BaseModel):
    brand: str
    model: str
    vehicle_age: int
    km_driven: int
    seller_type: str
    fuel_type: str
    transmission_type: str
    mileage: float
    engine: int
    max_power: float
    seats: int

@app.get("/")
def home():
    return {"status": "Car Price Prediction API is Live!"}

@app.post("/predict")
def predict_price(car: CarInput):
    car_name = car.brand + " " + car.model
    
    input_data = pd.DataFrame([[
        car_name, car.brand, car.model, car.vehicle_age, 
        car.km_driven, car.seller_type, car.fuel_type, car.transmission_type, 
        car.mileage, car.engine, car.max_power, car.seats
    ]], columns=[
        'car_name', 'brand', 'model', 'vehicle_age', 
        'km_driven', 'seller_type', 'fuel_type', 'transmission_type', 
        'mileage', 'engine', 'max_power', 'seats'
    ])
    
    # Label Encoding
    for col in ['car_name', 'brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        input_data[col] = input_data[col].astype(str)
        input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    prediction = model.predict(input_data)[0]
    return {"predicted_price": round(float(prediction), 2)}