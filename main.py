from fastapi import FastAPI, BackgroundTasks, Response
from pydantic import BaseModel

import schemas
import MLModel

import io
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


app = FastAPI()

result_from_training = dict()

@app.get("/health")
def health():
    health = schemas.Health()
    return health.dict()

@app.post("/input_data2")  
def query_stock_2(stock_code: schemas.stock_code):
    run_model_object = MLModel.run_supermodel(
        stock_code.stock_code, 
        stock_code.train_predict, 
        stock_code.cross_validation,
        stock_code.save_results,
        )
    x = run_model_object.train_predict()
    result_from_training[str(stock_code.stock_code)] = run_model_object
    plots_availables = [
        'baseline_models_plot','hiperparamter_plot', 
        'return_prediction_plot','future_prices_plot'
        ]

    y = ' the available plots are: ' + str(plots_availables) + 'indexed'
    return f'{x + y}'

@app.post("/vis")
def  get_vis(index: schemas.plot_schema,background_tasks: BackgroundTasks):

    results_to_post = {
        '0': result_from_training[index.index_stock].baseline_models_plot(),
        '1': result_from_training[index.index_stock].hiperparamter_plot(),
        '2': result_from_training[index.index_stock].return_prediction_plot(),
        '3': result_from_training[index.index_stock].future_prices_plot()

    }

    img_buf = results_to_post[index.plot_index]  
    background_tasks.add_task(img_buf.close)
    headers = {'Content-Disposition': 'inline; filename="out.png"'}
    return Response(img_buf.getvalue(), headers=headers, media_type='image/png')

