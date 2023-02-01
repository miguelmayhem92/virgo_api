from pydantic import BaseModel

class plot_schema(BaseModel):
    index_stock :str = ''
    plot_index : str = '0'
