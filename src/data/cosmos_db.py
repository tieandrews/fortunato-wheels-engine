# Author: Ty Andrews
# Date: 2023-03-28

import os
import sys

import pymongo
import datetime as dt
from dotenv import load_dotenv, find_dotenv


cur_dir = os.getcwd()
SRC_PATH = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.logs import get_logger

# Create a custom logger
logger = get_logger(__name__)

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

