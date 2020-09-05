
DATA_DIR = "data"
PROCESSED_DIR = "processed"

APP_FILE = 'response/applist.csv'
PROCESSED_APP_FILE ='response/processed_applist.csv'

SUCCESS_FILE = 'response/success.json'
FAILURE_FILE = 'response/failure.json'

INITIAL_DATAFRAME = 'processed/inital_structure.json'

APP_INFO_TABLE = 'dirty_app_info'
APP_LIST_TABLE = 'app_list'
SUCCESS_TABLE = 'app_info'
FAILURE_TABLE = 'request_failure'
GENRES_TABLE = 'genre'
CATEGORIES_TABLE = 'category'

# API URLs
ALL_APPS_URL = "http://api.steampowered.com/ISteamApps/GetAppList/v0002/?key=STEAMKEY&format=json"
APP_DETAILS_URL = "http://store.steampowered.com/api/appdetails?appids={APP_ID}"

import _local_settings

STEAMKEY = _local_settings.STEAMKEY

config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'newuser',
    'password': 'newpassword',
    'database': 'test_db'
}