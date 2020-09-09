# standard library imports
import csv
import datetime as dt
import json
import os
import statistics
import datetime
import time

# third-party imports
import numpy as np
import pandas as pd
import requests
import sqlalchemy as db
from sqlalchemy.types import Integer, Text, String, DateTime, NVARCHAR

import settings
import database

# customisations - ensure tables show all columns
pd.set_option("max_columns", 100)



ALL_APPS_URL = "http://api.steampowered.com/ISteamApps/GetAppList/v0002/?key={STEAMKEY}&format=json".format(
        STEAMKEY=settings.STEAMKEY)


engine, conn = database.open_db_connection()



def convert_to_df(d):
    df = pd.DataFrame.from_dict(d, orient='index')
    df.reset_index(level=0, inplace=True)
    return df


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


# def write_applist_to_file(d):
#     for app in d['applist']['apps']:
#         appid = app['appid']
#         with open(settings.APP_FILE, 'a') as applist:
#             json.dump(appid, applist)
#             applist.write(",")
#
#     with open(settings.APP_FILE, 'a') as applist:
#         applist.seek(0, os.SEEK_END)  # Move to last
#         applist.seek(applist.tell() - 1, os.SEEK_SET)  # back One character
#         applist.truncate()  # Delete the last comma ","


# def write_file_row(filename, data):
#     with open(filename, 'a') as file:
#         json.dump(data, file)  # write api resp to .JSON file
#         file.write(",")  # add comma for JSON array element
#
#
# def start_json_file(filename):
#     with open(filename, 'a') as file:
#         file.write('{"responses":[')
#
#
# def end_json_file(filename):
#     with open(filename, 'a') as file:
#         file.seek(0, os.SEEK_END)  # Move to last
#         file.seek(file.tell() - 1, os.SEEK_SET)  # back One character
#         file.truncate()  # Delete the last comma ","
#         file.write(']}')
#
#
# def end_csv_file(filename):
#     with open(filename, 'a') as file:
#         file.seek(0, os.SEEK_END)  # Move to last
#         file.seek(file.tell() - 1, os.SEEK_SET)  # back One character
#         file.truncate()  # Delete the last comma ",
#
# def write_csv_file(filename, data):
#     try: # todo: there must be a better way...
#         file_filled = pd.read_csv(filename, index_col=False, header=0)
#         write_file_row(filename, data)
#     except:
#         print(settings.PROCESSED_APP_FILE, " is empty")
#
#         with open(filename, 'a') as file:
#             file.write(",")  # add comma after last element
#             json.dump(data, file)  # write api resp to .JSON file
#             file.write(",")  # add comma for JSON array element


def get_unprocessed_applist(since_date):
    '''
    Gets the list of all steam app ids that have not been processed since a given date, or have never been processed

    Returns a Series of those app ids
    '''

    query_string = "SELECT * FROM {table} WHERE last_update < CAST('{query_date}' AS DATETIME) OR last_update IS NULL".format(
        table=settings.Database_Tables['APP_LIST_TABLE'],
        query_date=since_date
    )

    unprocessed_app_list = pd.read_sql(
        query_string,
        con=engine,
        columns=[
            'app_id',
            'last_update'
        ],
        parse_dates=[
            'last_update'
        ],
    )

    return unprocessed_app_list['app_id']


def update_applist(replace_table=False):
    '''
    replace_table: if True, will replace all app_list table data, not update

    Pulls all app_ids from Steam, compares them to anything on the app_list table already and adds any new ones with a
    last_update of yesterday

    return nothing
    '''

    all_apps = get_request(ALL_APPS_URL)

    if engine.has_table(settings.Database_Tables['APP_LIST_TABLE']) and not replace_table:
        appid_dict_list = pd.read_sql_table(
            settings.Database_Tables['APP_LIST_TABLE'],
            con=engine,
            columns=[
                'app_id',
                'last_update'
            ],
            parse_dates=[
                'last_update'
            ],
        ).to_dict(orient='records')

    else:
        appid_dict_list = []

    yesterday = datetime.date.today() - datetime.timedelta(days=1)

    for app in all_apps['applist']['apps']:
        appid = app['appid']

        if not any(d.get('app_id', None) == appid for d in appid_dict_list):
            appid_dict_list.append({'app_id': appid, 'last_update': yesterday})

    app_list_df = pd.DataFrame.from_records(appid_dict_list)
    if app_list_df.empty:
        print('app list dataframe is empty!  Probbably need a new steam key')
        raise SystemExit(0)

    app_list_df['last_update'] = pd.to_datetime(app_list_df['last_update'])

    app_list_df.to_sql(
        settings.Database_Tables['APP_LIST_TABLE'],
        engine,
        if_exists='replace',
        index=False,
        chunksize=10000,
        dtype={
            "app_id": Integer,
            "last_update": DateTime
        }
    )


def retrieve_app_data(since_date=datetime.date.today(), limit=500):  #todo: converting to function

    unprocessed_appids = get_unprocessed_applist(since_date)
    all_apps, failed_apps = get_app_info(unprocessed_appids, limit=limit)

    return all_apps


def get_app_info(apps, limit=200):
    '''

    :param apps: series of steam appids
    :param limit: total number of requests to make

    creates a dataframe of app info for steam apps - raw data that needs to be processed
    '''
    # has logic to deal with 200 requests/5 min limits

    count = 0

    app_data = {}
    app_failure = {}

    for appid in apps:
        count += 1
        app_info = get_request("http://store.steampowered.com/api/appdetails?appids={APP_ID}".format(APP_ID=appid))
        time.sleep(1.6)  # should keep us below request limit

        if app_info[str(appid)]['success']:
            app_data.update(app_info)
        else:
            app_failure.update(app_info)

        if count >= limit:
            print("reached " + str(limit) + " requests, stopping")
            break

    success_df = convert_to_df(app_data)
    success_df = success_df.set_index('index')

    success_df = pd.concat([success_df.drop(['data'], axis=1), success_df['data'].apply(pd.Series)], axis=1)
    failure_df = convert_to_df(app_failure)

    return success_df, failure_df


if __name__ == "__main__":
    pass
    # update_applist()
    # print('written to table')
    # print(app_list_df.head())
    # unprocessed_apps = get_unprocessed_applist()
    # print('unprocessed recs pulled')
    # print(unprocessed_apps)

    # cnt = 0
    #
    # print('Getting list of all apps!')
    # all_apps = get_request(ALL_APPS_URL)
    # write_applist_to_file(all_apps)
    #
    # unprocessed_appids = get_unprocessed_applist()
    # print(unprocessed_appids.head())

    # start_json_file(settings.SUCCESS_FILE)
    # start_json_file(settings.FAILURE_FILE)

    # while len(unprocessed_appids) != 0:
    #     cnt += 1
    #
    #     print('Requesting app info batch ' + str(cnt))
    #     get_app_info(unprocessed_appids['index'], limit=500)
    #
    #     unprocessed_appids = get_unprocessed_applist()
    #     print(str(len(unprocessed_appids)) + ' appids remaining to process.')

    # need to make sure these lines execute when processing is interrupted
    # end_json_file(settings.SUCCESS_FILE)
    # end_json_file(settings.FAILURE_FILE)
