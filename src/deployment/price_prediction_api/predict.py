# Author: Ty Andrews
# Date: 2023-06-08

import requests


def predict_price(vehicle_features: dict, api_url: str):
    """
    Makes a POST request to the deployed API to get a price prediction for a vehicle.

    Parameters
    ----------
    vehicle_features : dict
        A dictionary containing the vehicle features to be used in the prediction.
    api_url : str
        The URL of the deployed API.

    Returns
    -------
    predicted_price : float
        The predicted price of the vehicle in $CAD.
    price_upper_ci : float
        The upper 95% confidence interval of the predicted price of the vehicle in $CAD.
    price_lower_ci : float
        The lower 95% confidence interval of the predicted price of the vehicle in $CAD.
    """
    data = {"request_data": {"data": {"vehicle_data": vehicle_features}}}

    headers = {"Content-Type": "application/json"}

    resp = requests.post(api_url, json=data, headers=headers)

    predicted_price = resp.json()["predicted_price"]
    price_upper_ci = resp.json()["upper_ci"]
    price_lower_ci = resp.json()["lower_ci"]

    return predicted_price, price_upper_ci, price_lower_ci
