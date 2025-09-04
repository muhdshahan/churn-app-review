from fastapi import FastAPI , HTTPException
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

model = joblib.load('models/model.joblib')

class BankFeatures(BaseModel):
    Age: int
    Gender: str
    Balance: float
    EstimatedSalary: float
    NumOfProducts: int
    CreditScore: int
    IsActiveMember: int

@app.post('/predict')
def predict(val: BankFeatures):
    data = pd.DataFrame([val.dict()])

    if val.Gender not in ['Male', 'Female']:
        raise HTTPException(status_code=400, detail='Invalid Gender')
    
    data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
    prediction = model.predict(data)

    if prediction==1:
        return {"Prediction": 1, "Message": "Churn"}
    else:
        return {"Prediction": 0, "Message": "Not Churn"}
