"""
Datos de entrada del modelo:
["review_scores_rating", "room_type_Shared room", "room_type_Private room", "property_type_Apartment", "property_type_Condominium",
       "property_type_House", "property_type_Other", "bedrooms", "bathrooms"]

{
    "review_scores_rating": int,
    "room_type": str (Entire home/apt, Shared room, Private room),
    "property_type": str (Apartment, Bed & Breakfast, Condominium, House, Other),
    "bedrooms": int,
    "bathrooms": int,
}

{
    "review_scores_rating": 70,
    "room_type": "Shared room",
    "property_type": "Apartment",
    "bedrooms": 2,
    "bathrooms": 1
}

{
    "review_scores_rating": 95,
    "room_type": "Entire home/apt",
    "property_type": "House",
    "bedrooms": 5,
    "bathrooms": 2
}

"""
from typing import Annotated
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("random_forest_model.joblib")

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class PredictionRequest(BaseModel):
    review_scores_rating: float
    room_type: str
    property_type: str
    bedrooms: int
    bathrooms: int

def room_type_encoding(message):
    room_type_encoded = {"room_type_Shared room": 0, "room_type_Private room": 0}
    if message["room_type"].lower() == "shared room":
        room_type_encoded["room_type_Shared room"] = 1
    elif message["room_type"].lower() == "private room":
        room_type_encoded["room_type_Private room"] = 1

    del message["room_type"]

    return message.update(room_type_encoded)

def property_type_encoding(message):
    property_type_encoded = {"property_type_Bed_&_Breakfast": 0, "property_type_Condominium": 0,
                         "property_type_House": 0, "property_type_Other": 0}

    if message["property_type"].lower() == "bed & breakfast":
        property_type_encoded["property_type_Bed_&_Breakfast"] = 1
    elif message["property_type"].lower() == "condominium":
        property_type_encoded["property_type_Condominium"] = 1
    elif message["property_type"].lower() == "house":
        property_type_encoded["property_type_House"] = 1
    elif message["property_type"].lower() == "other":
        property_type_encoded["property_type_Other"] = 1

    del message["property_type"]

    return message.update(property_type_encoded)

def data_prep(message):
    room_type_encoding(message)
    property_type_encoding(message)

    return pd.DataFrame(message, index=[0])


def price_prediction(message: dict):
    data = data_prep(message)
    label = model.predict(data)[0]
    return {"label":label}

@app.get("/")
def main():
    return {"message": "Hola"}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password
    if username == "user" and password == "password":
        return {"access_token": "mock_token", "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Invalid credentials")

@app.post("/price-prediction/")
async def predict_price(message: dict, token: Annotated[str, Depends(oauth2_scheme)]):
    try:
        data = data_prep(message)

        # Verifica que el número de características sea el esperado
        expected_features = model.n_features_in_
        if data.shape[1] != expected_features:
            raise ValueError(f"El modelo espera {expected_features} características, pero recibió {data.shape[1]}.")

        prediction = model.predict(data)[0]

        return {"prediction": prediction}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Falta la característica: {str(ke)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor")