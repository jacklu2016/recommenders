import pandas as pd


if __name__ == '__main__':
    item_df = pd.read_table('item_list_all.txt', header=None)
    item_df.columns = ['item_code']

    order_df = pd.read_csv('cust_prod_order_07-10.csv')
    order_df_train = order_df[order_df['DAY_WID'].isin(
        ['20200706', '20200707', '20200707', '20200708', '20200709', '20200710', '20200711', '20200712', '20200713'])]

    item_df['has_order'] = item_df['item_code'].isin(order_df_train['COMPANY_PRODUCT_CODE'].values)

    item_df_reduce = item_df[item_df['has_order']]
    item_df_reduce.drop(['has_order'], axis=1, inplace=True)

    item_df_reduce.to_csv('item_list.txt', sep='\t', index=False, header=None)
