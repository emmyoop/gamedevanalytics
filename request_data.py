# standard library imports
import csv
import datetime as dt
import json
import os
import statistics
import time

# third-party imports
import numpy as np
import pandas as pd
import requests

import settings

# customisations - ensure tables show all columns
pd.set_option("max_columns", 100)

def get_request(url, parameters=None):
    """Return json-formatted response of a get request using optional parameters.

    Parameters
    ----------
    url : string
    parameters : {'parameter': 'value'}
        parameters to pass as part of get request

    Returns
    -------
    json_data
        json-formatted response (dict-like)
    """
    try:
        response = requests.get(url=url, params=parameters)
    except requests.exceptions.Timeout:
        for i in range(5, 0, -1):
            print('\rTimeout. Waiting... ({})'.format(i), end='')
            time.sleep(1)
        print('\rRetrying.' + ' ' * 10)
        return get_request(url, parameters)
    except requests.exceptions.TooManyRedirects:
        print('TooManyRedirects')
    except requests.exceptions.RequestException as e:
    # catastrophic error. bail.
        raise SystemExit(e)

    if response:
        return response.json()
    else:
        # response is none usually means too many requests. Wait and try again
        print('No response, waiting 10 seconds...')
        time.sleep(10)
        print('Retrying.')
        return get_request(url, parameters)


if __name__ == "__main__":
    json_data = get_request(settings.ALL_APPS_URL)