# Author: Ty Andrews
# Date: 2023-03-28

import os
import sys

import logging
import pymongo
import datetime as dt
from dotenv import load_dotenv, find_dotenv

cur_dir = os.getcwd()
SRC_PATH = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


def connect_to_database():
    """Sets up connection to Cosmos DB.

    Returns
    -------
    tuple
        A tuple containing the client, database, and collection pymongo objects.
    """
    CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")

    DB_NAME = os.environ.get("ADS_DB_NAME")
    COLLECTION_NAME = os.environ.get("KIJIJI_COLLECTION_NAME")

    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    return client, db, collection


def upload_ads_batch(collection: pymongo.collection.Collection, ads: list) -> None:
    """Uploads a batch of ads to Cosmos DB.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        The collection to upload the ads to.
    ads : list
        A list of ads to upload.
    """

    # Check if ad already exists in Cosmos DB
    existing_ads = list(
        collection.find({"ad_id": {"$in": [ad["ad_id"] for ad in ads]}})
    )

    ads_to_be_inserted = []
    num_existing_ads = 0
    # If ad already exists, update price
    for ad in ads:
        # parsed_ad = parse_single_vehicle_ad(ad)

        existing_ad = next((x for x in existing_ads if x["ad_id"] == ad["ad_id"]), None)
        if existing_ad is None:
            ads_to_be_inserted.append(ad)
        elif existing_ad["price"] != ad["price"]:
            update_ad_price(collection, existing_ad, ad)
        else:
            num_existing_ads += 1

    logger.debug(f"{num_existing_ads} existing ads found in Cosmos DB")

    # Upload new ads to Cosmos DB
    if len(ads_to_be_inserted) > 0:
        upload_to_cosmos(collection, ads_to_be_inserted)
        logger.info(f"Successfully uploaded {len(ads_to_be_inserted)} ads to Cosmos DB")


def check_for_duplicate_ads(collection: pymongo.collection.Collection):
    """Checks for duplicate ads in the database.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        The collection to check for duplicate ads in.
    """
    duplicates = list(
        collection.aggregate(
            [
                {
                    "$group": {
                        "_id": {
                            "make": "$make",
                            "model": "$model",
                            "year": "$year",
                            "title": "$title",
                        },
                        "uniqueIds": {"$addToSet": "$_id"},
                        "count": {"$sum": 1},
                    }
                },
                {"$match": {"count": {"$gt": 1}}},
                {"$sort": {"count": -1}},
            ]
        )
    )

    if len(duplicates) == 0:
        logger.debug("No duplicate ads with make/model/year/title found.")
    else:
        logger.warning(
            "Duplicate ads found with same make/model/year/title. Check the database."
        )


def upload_to_cosmos(collection, data):
    """Uploads a list of documents to Cosmos DB

    Parameters
    ----------
    collection : pymongo.collection.Collection
        The collection to upload the documents to.
    data : list
        A list of documents to upload.

    Raises
    ------
    Exception
        If an error occurs while uploading the documents.
    """
    try:
        collection.insert_many(data)
    except Exception as e:
        logger.error("Unexpected error:", type(e), e)
        sys.exit(1)


def update_ad_price(collection, existing_ad, new_ad):
    """Updates the price of an ad in Cosmos DB and adds entry to price history

    Parameters
    ----------
    collection : pymongo.collection.Collection
        The collection to update the ad in.
    existing_ad : dict
        The existing ad in Cosmos DB.
    new_ad : dict
        The new ad to update the existing ad with.

    Raises
    ------
    Exception
        If an error occurs while updating the ad.
    """

    existing_ad["price_history"].append(
        {
            "price": existing_ad["price"],
            "data": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    collection.update_one(
        {"ad_id": new_ad["ad_id"]},
        {
            "$set": {
                "price_history": existing_ad["price_history"],
                "price": new_ad["price"],
            }
        },
    )

    logger.info(
        f"Price updated for ad_id: {new_ad['ad_id']} from {existing_ad['price']} to {new_ad['price']}"
    )
