from pydantic import BaseModel

class stock_code(BaseModel):
    stock_code : str = 'DIS'
    train_predict : bool = False
    cross_validation : bool = False
    save_results : bool = False