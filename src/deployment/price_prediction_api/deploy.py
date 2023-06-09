# Author: Ty Andrews
# Date: 2023-06-04

import os
import sklearn
from azureml.core import Environment
from azureml.core import Workspace
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.model import Model



def main():
    ws = Workspace.from_config(
        # assumed running from root of repo
        path=os.path.join("src", "deployment", "price_prediction_api", "config.json")
    )
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")

    env = Environment("price-prediction-env")
    env.python.conda_dependencies.add_pip_package("joblib")
    env.python.conda_dependencies.add_pip_package("pandas==1.5")
    env.python.conda_dependencies.add_pip_package("numpy==1.23")
    env.python.conda_dependencies.add_pip_package("inference-schema")
    env.python.conda_dependencies.add_pip_package("azureml-defaults")
    env.python.conda_dependencies.add_pip_package(
        "scikit-learn=={}".format(sklearn.__version__)
    )

    inference_config = InferenceConfig(
        entry_script="score.py",
        environment=env,
    )

    price_model = Model(ws, "all-vehicles-price-prediction", version=3)
    price_model_quant5 = Model(ws, "all-vehicles-price-prediction-quant5", version=1)
    price_model_quant95 = Model(ws, "all-vehicles-price-prediction-quant95", version=1)

    deployment_config = AciWebservice.deploy_configuration(cpu_cores=0.5, memory_gb=2, enable_app_insights=True)

    service = Model.deploy(
        ws,
        "price-prediction-service",
        [price_model, price_model_quant5, price_model_quant95],
        inference_config,
        deployment_config,
        overwrite=True,
    )

    service.wait_for_deployment(show_output=True)

    print("Deployment done, scoring URI: ", service.scoring_uri)

if __name__ == "__main__":
    main()
