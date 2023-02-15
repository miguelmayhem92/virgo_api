from pydantic import BaseModel

class train_bidfinder(BaseModel):
    train_bidfinder : bool = False

class predict_bidfinder(BaseModel):
    predict_bidfinder : list = ['None']

