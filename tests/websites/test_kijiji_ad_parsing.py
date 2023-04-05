# Author: Ty Andrews
# Date: April 2, 2023
import os
import sys

import pytest
import requests
import json

cur_dir = os.getcwd()
src_path = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
print(src_path)
if src_path not in sys.path:
    sys.path.append(src_path)

import src.websites.kijiji_ad_parsing as kijiji_ad_parsing


# a sample ad from a multi-page scraping result
@pytest.fixture
def valid_page_ad():
    # load valid ad from data/testing/multi-page-scraping-sample-ad.json
    with open(
        os.path.join("data", "testing", "multi-page-scraping-sample-ad.json"), "r"
    ) as f:
        VALID_PAGE_AD = json.load(f)

    return VALID_PAGE_AD


# a sample ad from a direct single-page scraping result
@pytest.fixture
def valid_single_ad():
    # load valid ad from data/testing/multi-page-scraping-sample-ad.json
    with open(
        os.path.join("data", "testing", "single-ad-scraping-sample-ad.json"), "r"
    ) as f:
        VALID_SINGLE_AD = json.load(f)

    return VALID_SINGLE_AD


# ensure we get back the mandaotry fields with correct dtypes
def test_parse_page_single_vehicle_ad(valid_page_ad):
    parsed_ad = kijiji_ad_parsing.parse_page_single_vehicle_ad(valid_page_ad)

    assert isinstance(parsed_ad["ad_id"], str)
    assert isinstance(parsed_ad["url"], str)
    assert isinstance(parsed_ad["title"], str)
    assert isinstance(parsed_ad["make"], str)
    assert isinstance(parsed_ad["model"], str)
    assert isinstance(parsed_ad["price"], int)
    assert isinstance(parsed_ad["currency"], str)
    assert isinstance(parsed_ad["modified"], int)
    assert isinstance(parsed_ad["created"], int)
    assert isinstance(parsed_ad["location"]["country"], str)
    assert isinstance(parsed_ad["location"]["postalCode"], str)
    assert isinstance(parsed_ad["location"]["stateProvince"], str)
    assert isinstance(parsed_ad["location"]["city"], str)


# ensure we get back the mandaotry fields with correct dtypes
def test_parse_single_vehicle_ad(valid_single_ad):
    parsed_ad = kijiji_ad_parsing.parse_direct_single_vehicle_ad(valid_single_ad)

    assert isinstance(parsed_ad["ad_id"], str)
    assert isinstance(parsed_ad["url"], str)
    assert isinstance(parsed_ad["title"], str)
    assert isinstance(parsed_ad["make"], str)
    assert isinstance(parsed_ad["model"], str)
    assert isinstance(parsed_ad["price"], int)
    assert isinstance(parsed_ad["currency"], str)
    assert isinstance(parsed_ad["modified"], int)
    assert isinstance(parsed_ad["created"], int)
    assert isinstance(parsed_ad["location"]["country"], str)
    assert isinstance(parsed_ad["location"]["postalCode"], str)
    assert isinstance(parsed_ad["location"]["stateProvince"], str)
    assert isinstance(parsed_ad["location"]["city"], str)
