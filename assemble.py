import time
import numpy as np
import pandas as pd
import requests

import json
from pandas.io.json import json_normalize

from ast import literal_eval

def load_data(filepath):
    '''
    not sure if we actually will need this or not - copied over from notebook
    :return:
    '''
    def try_literal_eval(s):
        try:
            return literal_eval(s)
        except ValueError:
            return s


    steam_data = pd.read_csv(filepath)
    # steam_data = pd.json_normalize(steam_data, errors='ignore')
    # all columns that are dicts are being read in as strings - look in to json_normalize as possibly better solution?
    steam_data['price_overview'] = steam_data.price_overview.apply(try_literal_eval)
    steam_data['platforms'] = steam_data.platforms.apply(try_literal_eval)
    steam_data['recommendations'] = steam_data.recommendations.apply(try_literal_eval)
    steam_data['screenshots'] = steam_data.screenshots.apply(try_literal_eval)
    steam_data['movies'] = steam_data.movies.apply(try_literal_eval)
    steam_data['genres'] = steam_data.genres.apply(try_literal_eval)
    steam_data['release_date'] = steam_data.release_date.apply(try_literal_eval)
    steam_data['fullgame'] = steam_data.fullgame.apply(try_literal_eval)
    steam_data['demos'] = steam_data.demos.apply(try_literal_eval)
    steam_data['categories'] = steam_data.categories.apply(try_literal_eval)
    steam_data['metacritic'] = steam_data.metacritic.apply(try_literal_eval)
    steam_data['achievements'] = steam_data.achievements.apply(try_literal_eval)

    return steam_data

# more general functions - maybe pull them out somewhere else?
def flatten_field(df, field, rename_dict, drops_list):
    '''
    takes in a dataframe column that is a dict and seprates it into
    separate columns per key/value pair.  Can rename cols and drop
    columns as specified

    df: dataframe to alter
    field: column to flatten
    rename_dict: dictionary of current_name: new_name pairs for updating
    drops_list: list of new columns to drop
    '''
    df_clean = pd.concat([df, df[field].apply(pd.Series)], axis=1)
    df_clean.drop(axis=1, columns=drops_list, inplace=True)
    df_clean.rename(columns=rename_dict, inplace=True)

    return df_clean


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
    df = pd.concat([df, df[col].str.len()], axis=1)
    # this results in both old and new columns having the same name
    # so below code will remove the old screenshots dictionary column
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
    s = df[col].apply(pd.Series)
    # combine everything into single column
    # todo: add logic to know number of cols on the fly
    y = s[0].append([s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]], ignore_index=True).dropna()
    # split out dict to seprate columns
    z = y.apply(pd.Series)
    z = z.drop_duplicates(subset=['id', 'description'], keep="first")

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

    # drop the original column at the edn of processing
    df.drop(axis=1, columns=col, inplace=True)

    return df



# Steam specific dataset cleaning
def initial_cleanup(df):
    del_cols = ['index',
                'success',
                'header_image',
                'pc_requirements',
                'mac_requirements',
                'linux_requirements',
                'support_info',
                'background',
                'legal_notice',
                'reviews',
                'content_descriptors']

    num_type_list = ['required_age']

    rename_dict = {
        'name': 'game_name'
    }

    # set steam_appid as index
    df.set_index('steam_appid')

    # remove columns we don't care about
    df_clean = df.drop(columns=del_cols, axis=1)

    # rename columns as appropriate
    df_clean.rename(columns=rename_dict, inplace=True)

    # update types to numeric
    for i in num_type_list:
        df_clean[i] = pd.to_numeric(df_clean[i])

    # update types to datetime
    df_clean = convert_to_datetime(df_clean, 'release_date', {'date': 'release_date'}, ['release_date'])

    # trim down to just below types
    valid_types = ['game', 'dlc', 'demo']
    df_clean = remove_unused_data(df_clean, 'type', valid_types)

    # flatten cols as possible
    df_clean = flatten_price(df_clean)
    df_clean = flatten_platform(df_clean)
    df_clean = flatten_field(df_clean,
                             'recommendations',
                             {'total': 'recommendations'},
                             [0, 'recommendations'])
    df_clean = flatten_field(df_clean,
                             'metacritic',
                             {'score': 'metacritic_score'},
                             ['metacritic', 0, 'url'])
    df_clean = flatten_field(df_clean,
                             'fullgame',
                             {'appid': 'fullgame_appid'},
                             ['fullgame', 'name', 0])
    df_clean = flatten_field(df_clean,
                             'achievements',
                             {"total": "achievement_count"},
                             [0, 'achievements', 'highlighted'])

    # there seems to be only 1 demo in the subset i pulled so we'll just show that one demo id instead of the dict
    s = df_clean['demos'].apply(pd.Series)
    s['demo_appid'] = s[0].apply(lambda x: str(x['appid']) if not pd.isnull(x) else np.nan)
    df_clean = pd.concat([df_clean, s['demo_appid']], axis=1)
    # drop the original column at the edn of processing
    df_clean.drop(axis=1, columns='demos', inplace=True)

    # convert cols to bool type
    bool_col = 'controller_support'
    controller_mapping = {np.nan: False, 'full': True}
    df_clean[bool_col] = map_to_bool(df_clean, controller_mapping, bool_col)

    # convert cols to just counts
    df_clean = replace_with_count(df_clean, 'screenshots')
    df_clean.rename(columns={'screenshots': 'screenshot_count'}, inplace=True)

    df_clean = replace_with_count(df_clean, 'movies')
    df_clean.rename(columns={'movies': 'movie_count'}, inplace=True)

    df_clean = replace_with_count(df_clean, 'dlc')
    df_clean.rename(columns={'dlc': 'dlc_count'}, inplace=True)

    # convert lists to bools for easy categorization
    df_clean = create_unique_bool_cols(df_clean, 'genres', 'genre')
    df_clean = create_unique_bool_cols(df_clean, 'categories', 'category')

    return df_clean


def flatten_price(df):
    field_to_rename = {'currency': 'price_currency',
                       'discount_percent': 'price_discount_percent',
                       'final': 'price_final',
                       'initial': 'price_initial',
                       'recurring_sub': 'price_recurring_sub',
                       'recurring_sub_desc': 'price_recurring_sub_desc'}
    fields_to_drop = ['price_overview', 0, 'final_formatted', 'initial_formatted']

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

if __name__ == "__main__":
    steam_data = load_data('initial_preprocessed.csv')
    data1 = initial_cleanup(steam_data)
    data1.to_csv('consolidated_processed.csv', index=False)