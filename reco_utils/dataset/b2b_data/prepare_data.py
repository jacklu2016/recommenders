from typing import Union

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from pandas.core.arrays import ExtensionArray
from sklearn import preprocessing
import time

def convert_date_2_timestamp(date_str):
    time_array = time.strptime(date_str, "%Y%m%d")
    return int(time.mktime(time_array));


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def load_pandas_df(
    size="100k",
    header=None,
    local_cache_path=None,
    title_col=None,
    genres_col=None,
    year_col=None,
):
    return pd.read_csv('user_item_rating.csv')

if __name__ == '__main__':
    # user data
    user_df = pd.read_table('user_list.txt', header=None)
    user_df.columns = ['user_code']
    user_df['id'] = user_df.index
    user_map = dict(zip(user_df['user_code'].values, user_df['id'].values))

    # item data
    item_df = pd.read_table('item_list.txt', header=None)
    item_df.columns = ['item_code']
    item_df['id'] = item_df.index
    item_map = dict(zip(item_df['item_code'].values, item_df['id'].values))

    order_df = pd.read_csv('cust_prod_order_07-10.csv')
    order_df_train = order_df[order_df['DAY_WID'].isin(['20200706', '20200707', '20200707', '20200708', '20200709', '20200710', '20200711', '20200712', '20200713'])]
    order_df_train.sort_values(by='DAY_WID')

    order_df_train['reviewerID'] = order_df_train['CUST_INTERNAL_CODE'].map(user_map)
    order_df_train['asin'] = order_df_train['COMPANY_PRODUCT_CODE'].map(item_map)
    order_df_train = order_df_train.dropna(axis=0, how='any')
    order_df_train['reviewerID'] = order_df_train['reviewerID'].astype(int)
    order_df_train['asin'] = order_df_train['asin'].astype(int)
    user_item_train = pd.DataFrame(columns=['reviewerID', 'asin', 'DAY_WID'])
    user_item_train[['reviewerID', 'asin', 'DAY_WID']] = order_df_train[['reviewerID', 'asin', 'DAY_WID']]

    #生成rating数据
    min_max_scaler = preprocessing.MinMaxScaler()
    user_item_rating = pd.DataFrame(columns=['userID', 'itemID', 'rating', 'timestamp'])
    user_item_rating['userID'] = order_df_train['reviewerID']
    user_item_rating['itemID'] = order_df_train['asin']
    user_item_rating['rating'] = normalization(order_df_train['SALE_QTY'].values)
    user_item_rating['timestamp'] = order_df_train['DAY_WID'].map(lambda x: convert_date_2_timestamp(str(x)))
    user_item_rating.to_csv('user_item_rating.csv', index=None)

    user_item_test = user_item_train[user_item_train['DAY_WID'] == 20200713]
    user_item_train = user_item_train[user_item_train['DAY_WID'] != 20200713]
    user_item_test.drop(['DAY_WID'], axis=1, inplace=True)
    user_item_train.drop(['DAY_WID'], axis=1, inplace=True)
    user_item_train.to_csv('user_item_train.csv', index=None)
    user_item_test.to_csv('user_item_test.csv', index=None)
