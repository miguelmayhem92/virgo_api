from pydantic import BaseModel

class Health(BaseModel):
    name = 'project name'
    api_version='version project'
    model_version='model version'