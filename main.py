from fastapi import FastAPI, BackgroundTasks, Response
from pydantic import BaseModel

import schemas
import MLModel

import io
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


app = FastAPI()

result_from_training = []

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
    result_from_training.append(run_model_object)
    plots_availables = [
        'baseline_models_plot','hiperparamter_plot', 
        'return_prediction_plot','future_prices_plot'
        ]

    y = ' the plots are: ' + str(plots_availables)
    return f'{x + y}'

@app.post("/vis")
def  get_vis(testito: schemas.testito,background_tasks: BackgroundTasks):

    results_to_post = dict()
    results_to_post['baseline_models_plot'] = result_from_training[0].baseline_models_plot()
    results_to_post['hiperparamter_plot'] = result_from_training[0].hiperparamter_plot()
    results_to_post['return_prediction_plot'] = result_from_training[0].return_prediction_plot()
    results_to_post['future_prices_plot'] = result_from_training[0].future_prices_plot()

    img_buf = results_to_post[testito.testito_name]  ### correctmeeee
    background_tasks.add_task(img_buf.close)
    headers = {'Content-Disposition': 'inline; filename="out.png"'}
    return Response(img_buf.getvalue(), headers=headers, media_type='image/png')

