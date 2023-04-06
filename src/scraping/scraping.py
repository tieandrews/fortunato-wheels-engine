# Author: Ty Andrews
# Date: 2023-03-27
"""This script runs the scraping functions for the fortunato-wheels-engine project.

Usage: scraping.py [--site=<site>] [--make=<make>] [--num_pages=<num_pages>] 

Options:
    --site=<site>               The website to scrape from. [default: all]
    --make=<make>               The car make to scrape. [default: all]
    --num_pages=<num_pages>     The number of pages to scrape. [default: standard]
"""

import os
import sys

import json
from docopt import docopt

opt = docopt(__doc__)

cur_dir = os.getcwd()
SRC_PATH = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.websites.kijiji import get_kijiji_car_ad_pages
from src.logs import get_logger

# Create a custom logger
logger = get_logger(__name__)

def scrape_kijiji_car_make_pages(make=None, num_pages=None):
    """Scrapes Kijiji ad pages with multiple ads per page and inserts them
    into Cosmos DB.

    Parameters
    ----------
    make : str, optional
        The make, aka brand, of vehicles to scrape, all options can be found
        in , by default None
    num_pages : _type_, optional
        _description_, by default None

    Raises
    ------
    ValueError
        _description_
    """
    if (num_pages != "standard") & (isinstance(num_pages, int)):
        high_volume_num_pages = int(num_pages)
        low_volume_num_pages = int(num_pages)
    else:
        high_volume_num_pages = 5
        low_volume_num_pages = 1

    # pull in list of all car makes
    with open(
        os.path.join(os.getcwd(), "data", "scraping-tracking", "car-brands.json")
    ) as f:
        car_manufacturers = json.load(f)

    # get make name/number from user input if it's corrrectly input
    if (make != "all") & (
        any(make in sublist for sublist in car_manufacturers["high_volume_car_makes"])
        | any(make in sublist for sublist in car_manufacturers["low_volume_car_makes"])
    ):
        car_manufacturers["high_volume_car_makes"] = [
            car_make
            for car_make in car_manufacturers["high_volume_car_makes"]
            if car_make[1] == make
        ]
        car_manufacturers["low_volume_car_makes"] = [
            car_make
            for car_make in car_manufacturers["low_volume_car_makes"]
            if car_make[1] == make
        ]
    elif make == "all":
        pass
    else:
        raise ValueError(
            f"Make {make} is not a valid car make. See data/scraping-tracking/car-brands.json for valid car makes."
        )

    for car_make in car_manufacturers["high_volume_car_makes"]:
        make = car_make[1].lower().replace(" ", "-")
        make_number = car_make[0]

        logger.debug(f"Scraping {make} car ads...")
        # scrape the kijiji car ads for each car make
        df, json_dict, next_page_url = get_kijiji_car_ad_pages(
            car_make=make,
            make_number=make_number,
            num_pages=high_volume_num_pages,
            page_size=20,
            batch_upload_size=20,
        )

    for car_make in car_manufacturers["low_volume_car_makes"]:
        make = car_make[1].lower().replace(" ", "-")
        make_number = car_make[0]

        logger.info(f"Scraping {make} car ads...")
        # scrape the kijiji car ads for each car make
        df, json_dict, next_page_url = get_kijiji_car_ad_pages(
            car_make=make,
            make_number=make_number,
            num_pages=low_volume_num_pages,
            page_size=40,
            batch_upload_size=20,
        )


# main function
if __name__ == "__main__":
    logger.info(
        f"Docopt options: --make={opt['--make']}, --num_pages={opt['--num_pages']}, --site={opt['--site']}"
    )

    if opt["--num_pages"] != "standard":
        try:
            opt["--num_pages"] = int(opt["--num_pages"])
        except:
            raise ValueError(f"num_pages must be an integer or 'standard'.")

    if opt["--site"] == "kijiji":
        scrape_kijiji_car_make_pages(make=opt["--make"], num_pages=opt["--num_pages"])
    else:
        logger.info("Scraping all sites...")
        scrape_kijiji_car_make_pages(make=opt["--make"], num_pages=opt["--num_pages"])
