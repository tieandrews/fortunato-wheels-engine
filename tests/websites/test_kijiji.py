import os
import sys

import pytest
import requests

cur_dir = os.getcwd()
src_path = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
print(src_path)
if src_path not in sys.path:
    sys.path.append(src_path)

import src.websites.kijiji as kijiji


# check that the function returns a dict a list of int items in the form
# {ad_id(str): request-response(int)}
def test_load_failed_ad_ids():
    failed_ad_ids = kijiji.load_failed_ad_ids()

    assert isinstance(failed_ad_ids, dict)
    assert isinstance(list(failed_ad_ids.values())[0], int)


# ensure the proxy details dict contains http proxy info and a realistic port
# number
def test_get_proxy_details():
    proxy = kijiji.get_proxy_details()

    port = int(proxy["http"].split(":")[-1])

    assert "http" in proxy.keys(), "proxy should have http info, does not currently"
    assert port > 0, "port number should be in range 0-9999"
    assert port <= 9999, "port number should be in range 0-9999"


# check a cookie with all mandatory fields is returned
def test_get_kijiji_request_cookie():
    session = requests.Session()

    cookie = kijiji.get_kijiji_request_cookie(session)

    assert isinstance(
        cookie["mvcid"], str
    ), "Site needs MVCID from a valid session call to kijiji"
    assert isinstance(
        cookie["GCLB"], str
    ), "Kijiji needs valid GCLB string from a request session to authenticate"


# Check the request header has basic scraping best practices implemented
def test_get_kijiji_request_header():
    request_header = kijiji.get_kijiji_request_header()

    assert (
        "python-requests" not in request_header["User-Agent"]
    ), "Python requests found in header, ensure to replace with custom browser version"
    assert (
        "kijijiautos" in request_header["referer"]
    ), "Referer should be coming from kijijiautos site"
    assert (
        "x-client" in request_header.keys()
    ), "x-client link of ca.move.web.app mandatory for session authentication"
