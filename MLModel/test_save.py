import pandas as pd
import tempfile

with tempfile.TemporaryDirectory() as tmpdirname:
     x = 'created temporary directory', tmpdirname


def save_file():
    path = 'C:\\Users\\Miguel\\Descargas\\'
    data_dict= {'type':['test']}
    data_ = pd.DataFrame(data_dict)
    save_in = getcwd()+ '\\test_save_fastapi.csv'

    """ with tempfile.TemporaryDirectory() as tmpdirname:
        save_in = 'created temporary directory ' + tmpdirname + '\\test_save_fastapi.csv'
        data_.to_csv(save_in)"""

    return save_in