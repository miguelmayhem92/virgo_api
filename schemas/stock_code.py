from pydantic import BaseModel

class stock_code_training(BaseModel):
    stock_code : str = 'name'
    train_predict : bool = False
    cross_validation : bool = False
    n_trials: int = 1
    save_results_mlflow : bool = False

class stock_code_production(BaseModel):
    stock_code : str = 'name'
