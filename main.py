from fastapi import FastAPI
from pydantic import BaseModel

import schemas
import MLModel


app = FastAPI()

results_to_post = dict()

@app.get("/health")
def health():
    health = schemas.Health()
    return health.dict()

@app.post("/input_data")  
def query_stock(stock_code: schemas.stock_code):
    x = MLModel.run_model(
        stock_code.stock_code, 
        stock_code.train_predict, 
        stock_code.cross_validation,
        stock_code.save_results,
        )
    return f'{x}'

@app.post("/stocktests")  
def query_stock_save(stock_code: schemas.stock_code):
    x = MLModel.save_file()
    return f'{x}'

@app.get("/predict")
async def prediction():
    return({f'the test result is'})

