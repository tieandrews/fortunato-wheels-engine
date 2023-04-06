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
import pymongo
from dotenv import load_dotenv, find_dotenv

cur_dir = os.getcwd()
SRC_PATH = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.logs import get_logger

from src.websites.kijiji_ad_parsing import (
    parse_direct_single_vehicle_ad,
    parse_page_single_vehicle_ad,
)

from src.data.upload_to_db import (
    connect_to_database,
    upload_ads_batch,
    check_for_duplicate_ads,
)

load_dotenv(find_dotenv())

# Create a custom logger
logger = get_logger(__name__)


def get_kijiji_request_header() -> dict:
    """Provides a request header with multiple random user agents.

    Returns
    -------
    dict
        The header dict for Kijiji requests.
    """

    user_agents = [
        "com.ebay.kijiji.ca 6.5.0 (samsung SM-G930U; Android 8.0.0; en_US)",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    ]

    user_agent = random.choice(user_agents)

    header = {
        "User-Agent": user_agent,
        "accept-language": "en-CA",
        "Accept": "*/*",
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip, deflate, br",
        "referer": "https://www.kijijiautos.ca/cars/",
        "x-client": "ca.move.web.app",
        "listings-only": "true",
    }

    return header


def get_kijiji_request_cookie(session):
    """Provides a new request cookie with authentication for Kijiji.

    Parameters
    ----------
    session : requests.Session
        The session object to use for the request.

    Returns
    -------
    dict
        The cookie dict for Kijiji requests.
    """

    response = session.get("https://www.kijijiautos.ca")

    cookies_dict = response.cookies.get_dict()

    cookie = {
        "mvcid": cookies_dict["mvcid"],
        "trty": "e",
        "locale": "en-CA",
        "GCLB": cookies_dict["GCLB"],
        "disableLoginPrompt": "true",
        "location": "%7B%7D",
    }

    return cookie


def parse_results_to_df(ads_dict):
    raw_df = pd.DataFrame(ads_dict)

    return raw_df


def get_proxy_details():
    """Provides a proxy dict for Kijiji requests.

    Returns
    -------
    dict
        The proxy dict for Kijiji requests.
    """

    proxy_domain = os.environ.get("PROXY_DOMAIN")
    proxy_port = os.environ.get("PROXY_PORT")
    proxy_username = os.environ.get("PROXY_USERNAME")
    proxy_password = os.environ.get("PROXY_PASSWORD")

    proxy = {
        "http": f"http://{proxy_username}:{proxy_password}@{proxy_domain}:{proxy_port}"
    }

    return proxy


def random_wait_time(min_duration=1.5, max_duration=3.5):
    """Provides a random wait time between requests.

    Parameters
    ----------
    min_duration : float, optional
        The minimum wait time, by default 1.5
    max_duration : float, optional
        The maximum wait time, by default 3.5
    """

    duration = random.uniform(min_duration, max_duration)

    time.sleep(duration)


def load_failed_ad_ids():
    """Returns the ad id's which failed ot be scraped.

    Returns
    -------
    dict
        Dictionary with ad id's as keys and http request response error as values.
    """

    with open(
        os.path.join(os.getcwd(), "data", "scraping-tracking", "failed-ad-ids.json")
    ) as f:
        failed_ad_ids = json.load(f)

    return failed_ad_ids


def save_failed_ad_ids(failed_ad_ids):
    """Saves the ad_ids of ads that failed to be scraped"""

    with open(
        os.path.join(os.getcwd(), "data", "scraping-tracking", "failed-ad-ids.json"),
        "w",
    ) as f:
        json.dump(failed_ad_ids, f, indent=4)


def get_kijiji_car_ad_pages(
    car_make,
    make_number,
    page_size=20,
    num_pages=5,
    batch_upload_size=20,
    next_page_url=None,
):
    base_url = f"https://www.kijijiautos.ca/consumer/srp/by-params?ms={str(make_number)}&psz={page_size}"
    # to start off before appending page tokens
    if next_page_url is None:
        next_page_url = base_url

    failed_ad_ids = load_failed_ad_ids()

    logger.info(f"Getting {num_pages} pages of {car_make} Kijiji car ads...")

    session = requests.Session()
    header = get_kijiji_request_header()
    cookie = get_kijiji_request_cookie(session)
    proxy = get_proxy_details()

    results_df = pd.DataFrame()
    ads_json_list = []
    pages_till_new_session = random.randint(2, 4)
    delay = 1
    previous_page_token = None

    client, db, collection = connect_to_database()

    try:
        for i in range(0, num_pages):
            logger.info(f"Getting page {i+1} of {num_pages}, url: {next_page_url}")

            if (i % pages_till_new_session == 0) & (i != 0):
                session = requests.Session()
                cookie = get_kijiji_request_cookie(session)
                header = get_kijiji_request_header()
                logger.debug(
                    f"Updating session and cookies after {pages_till_new_session} pages"
                )
                pages_till_new_session = random.randint(2, 4)
            # pause before requesting next page
            random_wait_time(min_duration=delay, max_duration=2 * delay)

            start_time = time.time()
            res = session.get(
                next_page_url, proxies=proxy, headers=header, cookies=cookie
            )
            end_time = time.time()
            delay = end_time - start_time

            if res.status_code != 200:
                logger.error(f"Error getting {car_make} page {i}: {res.status_code}")
                # failed_ad_ids[ad_id] = res.status_code
                random_wait_time(min_duration=delay, max_duration=2 * delay)
                continue

            json_results = res.json()
            ads_list = json_results["listings"]["items"]

            ads_json_list.extend(ads_list)

            # append json results to df using pd.concat
            results_df = pd.concat([results_df, parse_results_to_df(ads_list)])

            if len(ads_json_list) >= batch_upload_size:
                logger.debug(f"Uploading batch of {len(ads_json_list)} ads to db")
                parsed_ads = [parse_page_single_vehicle_ad(ad) for ad in ads_json_list]
                upload_ads_batch(collection, parsed_ads)
                ads_json_list = []

            if json_results["listings"]["nextPageToken"] == previous_page_token:
                logger.debug(f"Reached end of pages for {car_make}")
                break
            else:
                previous_page_token = json_results["listings"]["nextPageToken"]

            # update url to get next page
            next_page_url = (
                base_url + "&pageToken=" + json_results["listings"]["nextPageToken"]
            )

            # pause before requesting next page
            random_wait_time()
    except Exception as e:
        logger.error("Unexpected error:", type(e), e)
        logger.info("Next page URL: " + next_page_url)
        return results_df, ads_json_list, next_page_url

    # upload final batch of ads
    if len(ads_json_list) > 0:
        logger.info(f"Uploading final batch of {len(ads_json_list)} ads to db")
        parsed_ads = [parse_page_single_vehicle_ad(ad) for ad in ads_json_list]
        upload_ads_batch(collection, parsed_ads)

    # final check for potential duplicates added
    check_for_duplicate_ads(collection)

    # report next page URL for use in next run
    logger.debug(f"Next page URL: {next_page_url}")

    return results_df, ads_json_list, next_page_url


def get_kijiji_car_ads(max_requests=100, batch_upload_size=40, start_id=None):
    base_url = "https://www.kijijiautos.ca/consumer/svc/a/"

    if start_id is None:
        ad_id = 28600000
    else:
        ad_id = start_id

    failed_ad_ids = load_failed_ad_ids()

    session = requests.Session()
    header = get_kijiji_request_header()
    cookie = get_kijiji_request_cookie(session)
    proxy = get_proxy_details()

    results_df = pd.DataFrame()
    ads_json_list = []
    requests_till_new_session = random.randint(30, 60)

    client, db, collection = connect_to_database()

    num_requests = 0
    delay = 2  # initialize request delay time

    try:
        # for ad_id in range(start_id, start_id + max_requests):
        while num_requests < max_requests:
            if ad_id not in failed_ad_ids:
                url = base_url + str(ad_id)

                if (ad_id % requests_till_new_session == 0) & (ad_id != 0):
                    session = requests.Session()
                    cookie = get_kijiji_request_cookie(session)
                    header = get_kijiji_request_header()
                    logger.debug(
                        f"Updating session and cookies after {requests_till_new_session} requests"
                    )
                    requests_till_new_session = random.randint(30, 60)

                try:
                    # pause before requesting next page
                    random_wait_time(min_duration=delay, max_duration=2 * delay)

                    start_time = time.time()
                    res = session.get(
                        url, proxies=proxy, headers=header, cookies=cookie
                    )
                    end_time = time.time()
                    delay = end_time - start_time
                    num_requests += 1

                    if res.status_code != 200:
                        logger.error(
                            f"Error getting ad with id {ad_id}: {res.status_code}"
                        )
                        failed_ad_ids[ad_id] = res.status_code
                        random_wait_time(min_duration=delay, max_duration=2 * delay)
                        continue

                    json_results = res.json()

                except Exception as e:
                    logger.error(f"Error getting ad with id {ad_id}: {e}")
                    failed_ad_ids[ad_id] = 1
                    continue

                ads_json_list.append(json_results)

                # append json results to df using pd.concat
                results_df = pd.concat([results_df, parse_results_to_df(ads_json_list)])

                if len(ads_json_list) >= batch_upload_size:
                    logger.debug(f"Uploading batch of {len(ads_json_list)} ads to db")
                    parsed_ads = [
                        parse_direct_single_vehicle_ad(ad) for ad in ads_json_list
                    ]
                    upload_ads_batch(collection, parsed_ads)
                    ads_json_list = []

            ad_id += 1

            if round(num_requests / max_requests * 100, 2) % 20 == 0:
                logger.info(
                    f"Progress: {round(num_requests/max_requests*100, 0)}% done"
                )

    except Exception as e:
        logger.error("Unexpected error:", type(e), e)
        logger.debug("Last URL: " + url)
        return results_df, ads_json_list, ad_id

    # final check for potential duplicates added
    check_for_duplicate_ads(collection)

    save_failed_ad_ids(failed_ad_ids)

    # report next URL for use in next run
    logger.info(f"Last id/URL: {ad_id} {url}")

    return results_df, ads_json_list, ad_id
