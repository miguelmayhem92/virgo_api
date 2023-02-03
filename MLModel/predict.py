import mlflow.pyfunc
from mlflow import MlflowClient

def get_model_production_results(stock_code):

    client = MlflowClient()

    model_name = f'{stock_code}_model'
    latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
    latest_production_version = latest_version_info[0].version
    model_version = latest_production_version

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
