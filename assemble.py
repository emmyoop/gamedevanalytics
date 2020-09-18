import numpy as np
import pandas as pd
import sqlalchemy
import math

from sqlalchemy.types import Integer, Text, Boolean, DateTime, NVARCHAR, Float

import datetime
from os import sys

from ast import literal_eval

import settings
import request_data
import database

# APP_INFO_TABLE = 'dirty_app_info'
engine, conn = database.open_db_connection()


raw_data_dtype={
 'type': NVARCHAR(256),
 'game_name': NVARCHAR(256),
 'steam_appid': Integer,
 'required_age': Integer,
 'is_free': Boolean,
 'short_description': Text,
 'supported_languages': Text,
 'website': Text,
 'developers': Text,
 'publishers': Text,
 # 'dlc': Text,
 'controller_support': Boolean,
 'achievement_count': NVARCHAR(256),
 'drm_notice': Text,
 'ext_user_account_notice': Text,
 'price_currency': NVARCHAR(256),
 'price_discount_percent': Float,
 'price_final': Float,
 'price_initial': Float,
 # 'price_recurring_sub': Float,
 # 'price_recurring_sub_desc': Text,
 'windows_support': Boolean,
 'mac_support': Boolean,
 'linux_support': Boolean,
 'recommendations': Float,
 'screenshot_count': Float,
 'movie_count': Float,
 'coming_soon': Boolean,
 'release_date': DateTime,
 'fullgame_appid': NVARCHAR(256),
 'demo_appid': NVARCHAR(256),
 'dlc_count': Integer,
 'metacritic_score': Float
            }

# def load_data(filepath):
#     '''
#     not sure if we actually will need this or not - copied over from notebook
#     :return:
#     '''
#     def try_literal_eval(s):
#         try:
#             return literal_eval(s)
#         except ValueError:
#             return s
#
#
#     steam_data = pd.read_csv(filepath)
#     # steam_data = pd.json_normalize(steam_data, errors='ignore')
#     # all columns that are dicts are being read in as strings - look in to json_normalize as possibly better solution?
#     steam_data['price_overview'] = steam_data.price_overview.apply(try_literal_eval)
#     steam_data['platforms'] = steam_data.platforms.apply(try_literal_eval)
#     steam_data['recommendations'] = steam_data.recommendations.apply(try_literal_eval)
#     steam_data['screenshots'] = steam_data.screenshots.apply(try_literal_eval)
#     steam_data['movies'] = steam_data.movies.apply(try_literal_eval)
#     steam_data['genres'] = steam_data.genres.apply(try_literal_eval)
#     steam_data['release_date'] = steam_data.release_date.apply(try_literal_eval)
#     steam_data['fullgame'] = steam_data.fullgame.apply(try_literal_eval)
#     steam_data['demos'] = steam_data.demos.apply(try_literal_eval)
#     steam_data['categories'] = steam_data.categories.apply(try_literal_eval)
#     steam_data['metacritic'] = steam_data.metacritic.apply(try_literal_eval)
#     steam_data['achievements'] = steam_data.achievements.apply(try_literal_eval)
#
#     return steam_data


# more general functions - maybe pull them out somewhere else?
def flatten_field(df, field, rename_dict, drops_list):
    '''
    takes in a dataframe column that is a dict and separates it into
    separate columns per key/value pair.  Can rename cols and drop
    columns as specified

    df: dataframe to alter
    field: column to flatten
    rename_dict: dictionary of current_name: new_name pairs for updating
    drops_list: list of new columns to drop
    '''
    if df[field].dropna().empty:
        print(field + ' column was empty')
        df_clean = df.drop(axis=1, columns=drops_list)
        df_clean.rename(columns=rename_dict, inplace=True)

    else:
        df_clean = pd.concat([df, df[field].apply(pd.Series)], axis=1)
        df_clean.drop(axis=1, columns=drops_list, inplace=True)
        df_clean.rename(columns=rename_dict, inplace=True)

    return df_clean


def list_to_string(df, field):
    '''
    takes in a dataframe column that is a list and separates it into
    just the contents of the list, replacing the original columns.

    df: dataframe to alter
    field: column to remove list
    '''
    if df[field].dropna().empty:
        print(field + ' column was empty')
        return df

    df['liststring'] = [','.join(map(str, l)) for l in df[field]]

    df.drop(axis=1, columns=[field], inplace=True)
    df.rename(columns={'liststring': field}, inplace=True)

    return df


def convert_to_datetime(df, col, rename_dict, drops_list):
    '''
    Takes in a dataframe column that is a dict.  Pulls the
    dict into columns and takes the date column into a dattime
    object.  Assumes a date format pandas can distinguish.
    '''
    s = df[col].apply(pd.Series)
    s['date'] = pd.to_datetime(s['date'], errors='coerce')

    df_clean = pd.concat([df, s], axis=1)
    df_clean.drop(axis=1, columns=drops_list, inplace=True)
    df_clean.rename(columns=rename_dict, inplace=True)

    return df_clean


def remove_unused_data(df, column, valid_list):
    '''
    Removes rows that so not match any values in the valid_list
    for the column
    '''

    contains = [df['type'].str.contains(i) for i in valid_list]
    df_clean = df[np.any(contains, axis=0)]
    return df_clean


def map_to_bool(df, mapping, col):
    '''
    maps values in a column to be just bools
    df: dataframe
    mapping: dict of mappings ex: {np.nan: False, 'full': True}
    col: name of col to convert
    '''
    return df[col].map(mapping)


def replace_with_count(df, col):
    '''
    gets the length of a column that is of type list
    '''
    if df[col].dropna().empty:
        return df
    df = pd.concat([df, df[col].str.len()], axis=1)
    # this results in both old and new columns having the same name
    # so below code will remove the original column
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    return df


def create_unique_bool_cols(df, col, prefix):
    '''
    Takes in a single columns in a dataframe, detmerines all unique values,
    creates a column for each unique value in the dataframe and fills it
    with a bool for each row indicating if that values exists for that row

    assumes column splits out into ['id','description'] pairs for uniqueness

    new column names will all be delimited with underscore

    df: Dataframe
    col: column to split out into multiple bool columns
    prefix: prefix of the new column names (genre -> genre_action, genre_adventure...).
    Will use description to build new column name
    '''
    # first we need to create a table of all possible values then store those so we can access them
    # combine everything into single column

    s = df[col].apply(pd.Series)
    num_cols = s.shape[1]

    # from pandas docs: Iteratively appending to a Series can be more computationally intensive than a single
    # concatenate. A better solution is to append values to a list and then concatenate the list with the original
    # Series all at once.
    listified = s[0].tolist()
    for x in range(1, num_cols):
        sub_list = s[x].dropna().tolist()
        listified += sub_list

    y = pd.Series(listified)

    y = y.dropna()

    # pull the dict out to columns
    z = y.apply(pd.Series)
    z = z.drop_duplicates(keep="first")

    # create a new column for each unique value
    for index, row in z.iterrows():
        new_col = '{0} {1}'.format(prefix, row['description']).replace(" ", "_")
        df[new_col] = False

    # then fill those columns in the Dataframe with bools
    for index, row in df.iterrows():
        if type(row[col]) == float:
            continue
        for item in row[col]:
            new_name = '{0} {1}'.format(prefix, item['description']).replace(" ", "_")
            # because you can't update on  iterrows()
            df.at[index, new_name] = True

    # drop the original column at the end of processing
    df.drop(axis=1, columns=col, inplace=True)

    return df


def convert_col_to_bool_table(df, col, table_name):
    '''
    Takes in a single columns in a dataframe, determines all unique values,
    creates a column for each unique value in the dataframe and fills it
    with a bool for each row indicating if that values exists for that row.

    Stores this as a tables with steam_appid

    assumes column splits out into ['id','description'] pairs for uniqueness


    df: Dataframe
    col: column to split out into multiple bool columns
    table_name: table to append new values
    Will use description to build new column name
    '''
    # first we need to create a table of all possible values then store those so we can access them
    # combine everything into single column

    if col not in df.columns:
        return df

    new_df = df[['steam_appid', col]].copy()

    s = new_df[col].apply(pd.Series)
    if s.empty:
        print('Column ' + col + ' is empty.')
        df.drop(axis=1, columns=col, inplace=True)
        return df
    num_cols = s.shape[1]

    # from pandas docs: Iteratively appending to a Series can be more computationally intensive than a single
    # concatenate. A better solution is to append values to a list and then concatenate the list with the original
    # Series all at once.
    listified = s[0].tolist()
    for x in range(1, num_cols):
        sub_list = s[x].dropna().tolist()
        listified += sub_list

    y = pd.Series(listified)

    y = y.dropna()

    # pull the dict out to columns
    z = y.apply(pd.Series)

    z = z.drop_duplicates(keep="first")

    # build df of current table columns
    if engine.has_table(table_name):
        boolean_df = pd.read_sql_table(
            table_name,
            con=engine,
        )
    # drop any rows in the existing data that are being updated by the new df
        boolean_df = boolean_df[~boolean_df['steam_appid'].isin(new_df['steam_appid'])]
    else:
        boolean_df = pd.DataFrame()

    # create a new column for each unique value
    for index, row in z.iterrows():
        new_df[row['description']] = False

    # then fill those columns in the Dataframe with bools
    for index, row in df.iterrows():
        if type(row[col]) == float:
            continue
        for item in row[col]:
            # because you can't update on  iterrows()
            new_df.at[index, item['description']] = True

    # append the old dataframe onto the new, filling in False for any extra columns
    df_append = new_df.append(boolean_df, verify_integrity=True).fillna(False)

    # drop the original column at the end of processing
    df_append.drop(axis=1, columns=col, inplace=True)
    df.drop(axis=1, columns=col, inplace=True)

    write_to_table(df_append, table_name, replace=True)

    return df


# Steam specific dataset cleaning
def initial_cleanup(df, replace=False):

    del_cols = ['success',
                'detailed_description',
                'about_the_game',
                'header_image',
                'pc_requirements',
                'mac_requirements',
                'linux_requirements',
                'support_info',
                'background',
                'legal_notice',
                'reviews',
                'content_descriptors',

                'packages',  # is unclear if we want/need this
                'package_groups', # is unclear if we want/need this

                'ext_user_account_notice'

                ]

    num_type_list = ['required_age']

    rename_dict = {
        'name': 'game_name'
    }

    # set steam_appid as index
    df.set_index('steam_appid')

    # remove columns we don't care about
    print('remove columns we dont care about')
    df_clean = df.drop(columns=del_cols, axis=1, errors='ignore')

    # rename columns as appropriate
    df_clean.rename(columns=rename_dict, inplace=True)

    # update types to numeric
    print('update types to numeric')
    for i in num_type_list:
        df_clean[i] = pd.to_numeric(df_clean[i])

    # update types to datetime
    print('update types to datetime')
    df_clean = convert_to_datetime(df_clean, 'release_date', {'date': 'release_date'}, ['release_date'])

    # trim down to just below types
    print('trim down to just below types')
    valid_types = ['game', 'dlc', 'demo']
    df_clean = remove_unused_data(df_clean, 'type', valid_types)

    if df_clean.empty:
        print('no relevant types, stop processing batch')
        return df_clean

    print(df_clean.head)
    print("Pre processed Dataframe size: " + str(df_clean.shape))

    # flatten cols as possible
    print('flatten cols as possible')
    try:
        df_clean = flatten_field(df_clean,
                                 'fullgame',
                                 {'appid': 'fullgame_appid'},
                                 ['name', 'fullgame'])
    except:
        print('Error flattening {} from dict columns: {}'.format('fullgame', sys.exc_info()[0]))

    try:
        df_clean = flatten_price(df_clean)
    except:
        print('Error flattening {} from dict columns: {}'.format('price', sys.exc_info()[0]))

    try:
        df_clean = flatten_platform(df_clean)
    except:
        print('Error flattening {} from dict columns: {}'.format('platform', sys.exc_info()[0]))

    try:
        df_clean = flatten_field(df_clean,
                                 'recommendations',
                                 {'total': 'recommendations'},
                                 ['recommendations'])
    except:
        print('Error flattening {} from dict columns: {}'.format('recommendations', sys.exc_info()[0]))

    try:
        df_clean = flatten_field(df_clean,
                             'metacritic',
                             {'score': 'metacritic_score'},
                             ['metacritic', 'url'])
    except:
        print('Error flattening {} from dict columns: {}'.format('metacritic', sys.exc_info()[0]))

    try:
        df_clean = flatten_field(df_clean,
                             'achievements',
                             {"total": "achievement_count"},
                             ['achievements', 'highlighted'])
    except:
        print('Error flattening {} from dict columns: {}'.format('achievements', sys.exc_info()[0]))

    # convert col of lists to just the string contents
    # todo: not sure this is really how we want to handle errors!?
    print('developers to string')
    try:
        df_clean = list_to_string(df_clean, 'developers')
    except:
        print('failed to convert developer column, blanking out data')
        df_clean['developers'] = np.nan
    print('publishers to string')
    try:
        df_clean = list_to_string(df_clean, 'publishers')
    except:
        print('failed to convert publishers column, blanking out data')
        df_clean['publishers'] = np.nan

    print('pull demo id')
    # there seems to be only 1 demo in the subset i pulled so we'll just show that one demo id instead of the dict
    try:
        if df_clean['demos'].dropna().empty:
            print('demos column was empty')
            df_clean.rename(columns={'demos': 'demo_appid'}, inplace=True)
        else:
            s = df_clean['demos'].apply(pd.Series)
            s['demo_appid'] = s[0].apply(lambda x: str(x['appid']) if not pd.isnull(x) else np.nan)
            df_clean = pd.concat([df_clean, s['demo_appid']], axis=1)
            # drop the original column at the edn of processing
            df_clean.drop(axis=1, columns='demos', inplace=True)
    except:
        print('Error with column {}: {}'.format('demos', sys.exc_info()[0]))

    # convert cols to bool type
    try:
        bool_col = 'controller_support'
        controller_mapping = {np.nan: False, 'full': True}
        df_clean[bool_col] = map_to_bool(df_clean, controller_mapping, bool_col)
    except:
        print('Error with column {}: {}'.format('controller_support', sys.exc_info()[0]))

    # convert cols to just counts
    print('convert columns to just counts')
    try:
        df_clean = replace_with_count(df_clean, 'screenshots')
        df_clean.rename(columns={'screenshots': 'screenshot_count'}, inplace=True)
    except:
        print('Error with column {}: {}'.format('screenshots', sys.exc_info()[0]))

    try:
        df_clean = replace_with_count(df_clean, 'movies')
        df_clean.rename(columns={'movies': 'movie_count'}, inplace=True)
    except:
        print('Error with column {}: {}'.format('movies', sys.exc_info()[0]))

    try:
        df_clean = replace_with_count(df_clean, 'dlc')
        df_clean.rename(columns={'dlc': 'dlc_count'}, inplace=True)
    except:
        print('Error with column {}: {}'.format('dlc', sys.exc_info()[0]))

    # convert lists to bools for easy categorization
    print('convert lists to bools for easy categorization - store in separate tables')
    df_clean = convert_col_to_bool_table(df_clean, 'genres', settings.Database_Tables['GENRES_TABLE'])
    df_clean = convert_col_to_bool_table(df_clean, 'categories', settings.Database_Tables['CATEGORIES_TABLE'])

    df_clean = df_clean.drop(columns=[0], axis=1, errors='ignore')

    print('*** Dataframe columns***')
    print(df_clean.columns)
    for k, v in raw_data_dtype.items():
        if k not in df_clean.columns:
            print(k + ' not in dataframe, adding it')
            df_clean[k] = np.nan

    print('checking for extra columns and dropping them')
    extra = []
    for col in df_clean.columns:
        if col not in raw_data_dtype:
            print(str(col) + ' is extra column that should not exist')
            extra.append(col)
    df_clean = df_clean.drop(columns=extra, axis=1)

    return df_clean


def flatten_price(df):

    field_to_rename = {'currency': 'price_currency',
                       'discount_percent': 'price_discount_percent',
                       'final': 'price_final',
                       'initial': 'price_initial'}
    fields_to_drop = ['price_overview', 'final_formatted', 'initial_formatted']

    df_clean = flatten_field(df, 'price_overview', field_to_rename, fields_to_drop)

    df_clean['price_final'] = df_clean['price_final'] / 100
    df_clean['price_initial'] = df_clean['price_initial'] / 100

    return df_clean


def flatten_platform(df):
    fields_to_rename = {'windows': 'windows_support',
                        'mac': 'mac_support',
                        'linux': 'linux_support'}
    fields_to_drop = ['platforms']

    df_clean = flatten_field(df, 'platforms', fields_to_rename, fields_to_drop)

    return df_clean


def write_to_table(df, table_name, replace=False, table_dtype=None):
    # todo: need to add logic for main data store to store this to a temp table then do a join on the permenant and temp
    #  table so that we can update any records that are new

    if table_dtype == None:
        table_dtype = sqlcol(df)

    if replace:
        if_exists = 'replace'
    else:
        if_exists = 'append'

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,  # todo: this is not right, should be append, but leaving replace for testing - if i create
                              # tables with specific column names it should solve this
        index=False,
        dtype=table_dtype
    )


def write_last_update(appid_series, update_date):

    appid_df = pd.read_sql_table(
            settings.Database_Tables['APP_LIST_TABLE'],
            con=engine,
            columns=[
                'app_id',
                'last_update'
            ],
            parse_dates=[
                'last_update'
            ],
        )

    # todo: probably a one liner to do below
    for appid in appid_series:
        appid_df.loc[appid_df.app_id == appid, 'last_update'] = update_date

    appid_df['last_update'] = pd.to_datetime(appid_df['last_update'])

    appid_df.to_sql(
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


def sqlcol(dfparam):

    dtypedict = {}
    for i,j in zip(dfparam.columns, dfparam.dtypes):
        if "object" in str(j):
            dtypedict.update({i: sqlalchemy.types.TEXT})

        elif "datetime" in str(j):
            dtypedict.update({i: sqlalchemy.types.DateTime()})

        elif "float" in str(j):
            dtypedict.update({i: sqlalchemy.types.Float(precision=3, asdecimal=True)})

        elif "int" in str(j):
            dtypedict.update({i: sqlalchemy.types.INT()})

        elif "bool" in str(j):
            dtypedict.update({i: sqlalchemy.types.Boolean()})

        else:
            dtypedict.update({i: sqlalchemy.types.NVARCHAR(length=255)})

    return dtypedict


def build_new_tables(chunks=1000, records=20):

    print('drop all tables')
    connection = engine.raw_connection()
    cursor = connection.cursor()
    for k,v in settings.Database_Tables.items():
        command = "DROP TABLE IF EXISTS {};".format(v)
    cursor.execute(command)
    connection.commit()
    cursor.close()

    print('Build new app list')
    request_data.update_applist(replace_table=True)

    loop_through_records(datetime.date.today(), chunks, records)


def loop_through_records(since_date, chunks, records=20):
    '''

    :param since_date: date appid was last processed - generally today but this allow to
                        rerun for failed processing on other days
    records: number of records total to process
    :return: none
    '''

    if records > chunks:
        chunksize = chunks
    else:
        chunksize = records

    for i in range(0,math.ceil(records/chunksize)):
        print('iteration ' + str(i+1))

        print('Get raw app info')
        raw_app_data = request_data.retrieve_app_data(since_date, chunksize)

        print('clean up data for table insertion')
        parsed_data = initial_cleanup(raw_app_data, replace=True)

        if parsed_data.empty:
            print('**************Empty dataset, nothing to write to tables**************')
        else:
            print('write to table')
            write_to_table(parsed_data, settings.Database_Tables['APP_INFO_TABLE'], replace=False, table_dtype=raw_data_dtype)

        print('updating app list for last update')
        today = datetime.date.today()
        write_last_update(parsed_data['steam_appid'], today)

def update_existing_tables(since_date=datetime.date.today(), chunks=1000, records=20):

    #todo: query
    loop_through_records(since_date, chunks, records)


if __name__ == "__main__":
    # build_new_tables()
    build_new_tables(chunks=50, records=1500)


    # yesterday = datetime.date.today() - datetime.timedelta(days=1)

    # update_existing_tables(since_date=yesterday,chunks=50, records=1500)

