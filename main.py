from fastapi import FastAPI
from pydantic import BaseModel

import schemas
import MLModel


app = FastAPI()


@app.get("/health")
def health():
    health = schemas.Health()
    return health.dict()

@app.post("/input_data")  
def query_stock(stock_code: schemas.stock_code):
    x = MLModel.run_model(
        stock_code.stock_code, 
        stock_code.train_predict, 
        stock_code.cross_validation
        )
    return f'{x}'


@app.get("/predict")
async def iprediction():
    return({f'the test result is'})

