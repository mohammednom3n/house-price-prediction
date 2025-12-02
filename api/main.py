from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference import predict_price

app = FastAPI(title = "House_Price_Pridiction API")

class PrdictionResquest(BaseModel):
    OverallQual: int
    GrLivArea: int
    FirstFlrSF: int = Field(..., alias="1stFlrSF")
    TotalBsmtSF: int
    BsmtFinSF1: int
    LotArea: int
    GarageCars: int
    TotRmsAbvGrd: int
    SecndFlrSF:int = Field(..., alias="2ndFlrSF")
    YearBuilt: int
    GarageArea: int
    FullBath: int
    OverallCond: int
    YearRemodAdd: int
    MSSubClass: int


@app.post("/predict")
def predict(req: PrdictionResquest):
    data = req.dict(by_alias=True)
    result = predict_price(data)
    return {"prediction": float(result)}
    