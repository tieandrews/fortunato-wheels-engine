# Author: Ty Andrews
# Date: 2023-03-23

import os
import sys

import pandas as pd
import requests
import random
import time
import datetime as dt
import logging
import json

cur_dir = os.getcwd()
SRC_PATH = cur_dir[: cur_dir.index("fortunato-wheels") + len("fortunato-wheels")]
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Create a custom logger
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# # create console handler and set level to debug
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# # create formatter
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# # add formatter to ch
# ch.setFormatter(formatter)
# # add ch to logger
# if len(logger.handlers) != 0:
#     logger.handlers.clear()
#     print("Cleared exisiting handlers")

# logger.addHandler(ch)
# logger.propagate = False


def parse_page_single_vehicle_ad(ad_json):
    """Parses the JSON returned by Kijiji for a single vehicle ad page.

    Parameters
    ----------
    ad_json : dict
        The raw json returned from Kijiji

    Returns
    -------
    dict
        The parsed and formatted dict with all mandatory fields
    """

    listing = dict()

    # MANDATORY fields
    try:
        listing["ad_id"] = ad_json["id"]
        listing["url"] = ad_json["url"]
        listing["title"] = ad_json["title"]
        listing["make"] = ad_json["make"]
        listing["model"] = ad_json["model"]
        listing["year"] = ad_json["attr"]["yc"]
        listing["mileage"] = int(
            ad_json["attr"]["ml"].split("\xa0")[0].replace(",", "")
        )
        listing["mileageUnit"] = ad_json["attr"]["ml"].split("\xa0")[1]
        listing["price"] = int(ad_json["prices"]["consumerPrice"]["amount"])
        listing["currency"] = ad_json["prices"]["consumerPrice"]["currency"]
        listing["modified"] = ad_json["modified"]
        listing["created"] = ad_json["created"]

        listing["location"] = {
            "country": ad_json["attr"]["cn"],
            "postalCode": ad_json["attr"]["z"],
            "city": ad_json["attr"]["loc"].split(",")[0],
            "stateProvince": ad_json["attr"]["loc"].split(",")[1][1:],
        }
    except KeyError:
        logger.warn(f"Ad id {ad_json['id']} missing one or more mandatory fields.")
        return None

    # OPTIONAL parameters from Ad
    if "condition" in ad_json:
        listing["condition"] = {
            "condition": ad_json["condition"],
        }
    else:
        listing["condition"] = {
            "condition": "unknown",
        }

    try:
        # get True/False flag of roadworthy status
        listing["condition"]["roadworthy"] = ad_json["roadworthy"] == "True"
    except:
        pass

    try:
        # get True/False flag of damage status
        listing["condition"]["hasDamage"] = ad_json["hasDamage"] == "True"
    except:
        pass

    # some adds don't have vehicle color added
    if "ecol" not in ad_json["attr"]:
        listing["color"] = None
    else:
        listing["color"] = ad_json["attr"]["ecol"]

    listing["images"] = [img["uri"] for img in ad_json["images"]]

    listing["features"] = [f.strip() for f in ad_json["features"]]

    listing["seller"] = {
        "sellerType": ad_json["st"],
        "sellerForeignId": ad_json["sellerId"],
        "contact": ad_json["contact"],
        "partnerId": ad_json["partnerId"],
        "customerId": ad_json["customerId"],
    }

    attribute_mappings = {
        "horsePower": "pw",
        "fuelType": "ft",
        "engineSize": "cc",
        "doors": "door",
        "seats": "sc",
        "driveTrain": "dt",
        "cylinders": "cylinders",
        "transmission": "tr",
        "fuelEconomy": "csmpt",
    }

    # parse the most common optional attributes on the ad
    listing["details"] = {}
    for key, value in attribute_mappings.items():
        try:
            listing["details"][key] = ad_json["attr"][value]
        except:
            pass

    if "fuelEconomy" in listing["details"]:
        listing["details"]["fuelEconomy"] = listing["details"]["fuelEconomy"].replace(
            "u\\2009", " "
        )

    # Future tracking of price changes initializattion
    listing["price_history"] = []

    return listing


def parse_direct_single_vehicle_ad(ad_json):
    listing = dict()

    listing["ad_id"] = ad_json["id"]
    listing["url"] = ad_json["url"]
    listing["title"] = ad_json["title"]
    listing["make"] = ad_json["make"]

    # some adds don't have model added
    if "model" not in ad_json:
        listing["model"] = None
    else:
        listing["model"] = ad_json["model"]

    if "trim" not in ad_json:
        listing["trim"] = None
    else:
        listing["trim"] = ad_json["trim"]

    for attribute in ad_json["quickFacts"]["attributes"]:
        # ensure attributes actually have a value
        if "value" in attribute:
            if attribute["tag"] == "condition":
                listing["condition"] = attribute["value"]
            elif attribute["tag"] == "mileage":
                listing["mileage"] = int(
                    attribute["value"].split("\xa0")[0].replace(",", "")
                )
                listing["mileageUnit"] = attribute["value"].split("\xa0")[1]
            elif attribute["tag"] == "transmission":
                listing["transmission"] = attribute["value"]
            elif attribute["tag"] == "fuelType":
                listing["fuelType"] = attribute["value"]
            elif attribute["tag"] == "driveTrain":
                listing["driveTrain"] = attribute["value"]

    listing["year"] = ad_json["year"]

    if "vin" in ad_json:
        listing["vin"] = ad_json["vin"]

    # some adds don't have vehicle color added
    if "exteriorColor" not in ad_json:
        listing["color"] = None
    else:
        listing["color"] = ad_json["exteriorColor"]

    if "consumerPrice" in ad_json["prices"]:
        listing["price"] = int(ad_json["prices"]["consumerPrice"]["amount"])
        listing["currency"] = ad_json["prices"]["consumerPrice"]["currency"]
    else:
        listing["price"] = None
        listing["currency"] = None

    listing["price_history"] = []

    if "images" in ad_json:
        listing["images"] = [img["uri"] for img in ad_json["images"]]
    else:
        listing["images"] = []

    listing["modified"] = ad_json["modified"]
    listing["created"] = ad_json["created"]

    listing["location"] = dict(
        stateProvince=ad_json["location"]["province"],
        postalCode=ad_json["location"]["zipCode"],
        city=ad_json["location"]["city"],
        country=ad_json["contact"]["country"],
    )

    listing["seller"] = {
        "sellerType": ad_json["contact"]["type"],
        "sellerForeignId": ad_json["sellerForeignId"],
        "contact": ad_json["contact"],
        "partnerId": ad_json["partnerId"],
        "customerId": ad_json["customerId"],
    }

    if "dealerRating" in ad_json:
        listing["seller"]["dealerRating"] = ad_json["dealerRating"]["rating"]

    listing["details"] = {}

    exclude_details = ["make", "model", "year"]

    for det in ad_json["vehicleDetails"]:
        for att in det["attributes"]:
            # exclude duplicated fields
            if att["values"][0].lower() not in exclude_details:
                # fix messed up characters and convert to numeric where possible
                if att["values"][0].lower() == "power":
                    att["values"][1] = int(att["values"][1].replace("\xa0hp", ""))
                elif ("fuel" in att["values"][0].lower()) & (
                    "consumption" in att["values"][0].lower()
                ):
                    att["values"][1] = att["values"][1].replace("\u2009", " ")

                if len(att["values"]) == 2:
                    listing["details"][
                        att["values"][0].lower().replace(" ", "_")
                    ] = att["values"][1]
                else:
                    listing["details"][att["values"][0].lower()] = True

    if "priceRating" in ad_json:
        listing["priceRating"] = ad_json["priceRating"]

    return listing
