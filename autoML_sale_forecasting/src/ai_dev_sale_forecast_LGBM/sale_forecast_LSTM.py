# coding: utf-8

"""
ThinkPage Retail Forecasting

LSTM version 1.0

dev:xingqiang chen
date:2018-03-20
"""
# coding: utf-8

import argparse
import datetime as DT
import gc
import json
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pymysql
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine


def parse_arguments(argv):
    DATE = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    parser = argparse.ArgumentParser()

    parser.add_argument('--DATE', type=str, help='the date of starting predction.', default=DATE)

    return parser.parse_args(argv)


def T(FORMAT_DATE, i):
    return (DT.datetime.strptime(FORMAT_DATE, '%Y-%m-%d')
            + DT.timedelta(i)).strftime('%Y%m%d')


def FT(FORMAT_DATE, i):
    return (DT.datetime.strptime(FORMAT_DATE, '%Y-%m-%d')
            + DT.timedelta(i)).strftime('%Y-%m-%d')


def date_dt(DAY):
    return date(int(DAY[0:4]), int(DAY[5:7]), int(DAY[8:]))


"""
Define The Global Variables and Constants
"""
TODAY_DATE = parse_arguments(sys.argv[1:]).DATE  ## for test TODO
print(u"## 当前日期：" + TODAY_DATE)
PRE_DAYS = 3
print(u"## 预测天数:" + str(PRE_DAYS))
DATA_START_DAY = FT(TODAY_DATE, -1000)
print(u"## 获取数据的长度180天,起点日期：" + DATA_START_DAY)

TRAIN_START_DAY = FT(TODAY_DATE, -600)
print(u"## 训练数据的长度76天,起点日期:" + TRAIN_START_DAY)

VAL_DATE = FT(TODAY_DATE, -30)
print(u"## 验证数据的起点日期：" + VAL_DATE)

PREDICTION_DAY = FT(TODAY_DATE, PRE_DAYS)
print(u"## 预测日期：" + PREDICTION_DAY)

TRAIN_DATE = date_dt(TRAIN_START_DAY)
VAL_DATE = date_dt(VAL_DATE)
PRE_DATE = date_dt(TODAY_DATE)
num_days = 10


def Get_Data_from_sql_online(TODAY_DATE, DATA_START_DAY):
    # db info
    host = 'xx.xx.xx.xx'
    user = 'xx'
    password = 'xx'
    port = 3306

    def get_df_actual_purchase_his(DATA_START_DAY):
        """
        """
        # purchase data from db
        cols = ['date', 'storeid', 'goodsid', 'cost', 'qty', 'purchase_num_of_days']
        database = 'MARKET_DB'
        db = pymysql.connect(host, user, password, database, port, charset='utf8')
        cursor = db.cursor()
        sql_command = """SELECT date,storeid, goodsid,cost,qty,purchase_num_of_days FROM mid_aggrbydate_actualpurchase_his WHERE date >='{start_day}'"""

        sql = sql_command.format(start_day=DATA_START_DAY)
        cursor.execute(sql)
        results = cursor.fetchall()
        df = pd.DataFrame(list(results), columns=cols)

        return df

    def get_df_aggrbydate_airq(DATA_START_DAY):
        """
        """
        # airq data from db
        cols = ['date', 'storeid', 'AQI']
        database = 'MARKET_DB'
        db = pymysql.connect(host, user, password, database, port, charset='utf8')
        cursor = db.cursor()
        sql_command = """SELECT date,storeid, AQI FROM  mid_aggrbydate_airq_his WHERE date >='{start_day}'"""

        sql = sql_command.format(start_day=DATA_START_DAY)
        cursor.execute(sql)
        results = cursor.fetchall()
        df = pd.DataFrame(list(results), columns=cols)

        return df

    def get_df_aggrbydate_saleprice(DATA_START_DAY):
        """
        """
        # price data from db
        cols = ['date', 'storeid', 'goodsid', 'discount_degree', 'price', 'x']
        database = 'MARKET_DB'
        db = pymysql.connect(host, user, password, database, port, charset='utf8')
        cursor = db.cursor()
        sql_command = """SELECT date,storeid,goodsid,discount_degree,price,x FROM  mid_aggrbydate_saleprice_his WHERE date >='{start_day}'"""

        sql = sql_command.format(start_day=DATA_START_DAY)
        cursor.execute(sql)
        results = cursor.fetchall()
        df = pd.DataFrame(list(results), columns=cols)

        return df

    def get_df_aggrbydate_saleqty(DATA_START_DAY):
        """
        """
        # price data from db
        cols = ['date', 'storeid', 'goodsid', 'sale_qty', 'sale_qty_adj', 'x']
        database = 'MARKET_DB'
        db = pymysql.connect(host, user, password, database, port, charset='utf8')
        cursor = db.cursor()
        sql_command = """SELECT date,storeid,goodsid,sale_qty,sale_qty_adj,x FROM  mid_aggrbydate_saleqty_his WHERE date >='{start_day}'"""

        sql = sql_command.format(start_day=DATA_START_DAY)
        cursor.execute(sql)
        results = cursor.fetchall()
        df = pd.DataFrame(list(results), columns=cols)

        return df

    def get_df_aggrbydate_weather_his(DATA_START_DAY):
        """
        """
        # price data from db
        cols = ['date', 'storeid', 'tem', 'windspeed', 'pre1h', 'rhu']
        database = 'MARKET_DB'
        db = pymysql.connect(host, user, password, database, port, charset='utf8')
        cursor = db.cursor()
        sql_command = """SELECT date,storeid,tem,windspeed,pre1h,rhu FROM  mid_aggrbydate_weather_his WHERE date >='{start_day}'"""

        sql = sql_command.format(start_day=DATA_START_DAY)
        cursor.execute(sql)
        results = cursor.fetchall()
        df = pd.DataFrame(list(results), columns=cols)

        return df

    def get_df_category_his(TODAY_DATE):
        """
        """
        # category data from db
        cols = ['deptid', 'name', 'name_up1', 'name_up2', 'name_up3', 'name_up4', 'deptlevelid', 'date']
        database = 'MARKET_DB'
        db = pymysql.connect(host, user, password, database, port, charset='utf8')
        cursor = db.cursor()
        sql_command = """SELECT deptid,name,name_up1,name_up2,name_up3,name_up4, deptlevelid, date FROM  raw_cleaned_category_his WHERE date ='{today_date}'"""

        sql = sql_command.format(today_date=TODAY_DATE)
        cursor.execute(sql)
        results = cursor.fetchall()
        df = pd.DataFrame(list(results), columns=cols)

        return df

    def get_df_goods_his(TODAY_DATE):
        """
        """
        # category data from db
        cols = ['goodsid', 'spname', 'ppname', 'deptid', 'goodstypeid', 'stocktype', 'saletaxrate', 'spec', 'unitname',
                'origin', 'keepdays', 'indate', 'date']
        database = 'MARKET_DB'
        db = pymysql.connect(host, user, password, database, port, charset='utf8')
        cursor = db.cursor()
        sql_command = """SELECT goodsid, spname,ppname,deptid,goodstypeid,stocktype,saletaxrate,spec,unitname,origin,keepdays,indate,date FROM  raw_cleaned_goods_his WHERE date ='{today_date}'"""

        sql = sql_command.format(today_date=TODAY_DATE)
        cursor.execute(sql)
        results = cursor.fetchall()
        df = pd.DataFrame(list(results), columns=cols)

        return df

    def get_df_canlendar_info(DATA_START_DAY):
        """
        """
        # category data from db
        cols = ['date', 'weekday', '24term_name_en', '24term_code', '24term_name_cn', 'holiday_bylaw']
        database = 'COMMONINFO_DB'
        db = pymysql.connect(host, user, password, database, port, charset='utf8')
        cursor = db.cursor()
        sql_command = """SELECT date, weekday,24term_name_en,24term_code,24term_name_cn,holiday_bylaw FROM  calendar WHERE date >='{start_day}' """

        sql = sql_command.format(start_day=DATA_START_DAY)
        cursor.execute(sql)
        results = cursor.fetchall()
        df = pd.DataFrame(list(results), columns=cols)

        return df

    return {'actual_purchase_his': get_df_actual_purchase_his(DATA_START_DAY),
            'airq': get_df_aggrbydate_airq(DATA_START_DAY),
            'saleprice': get_df_aggrbydate_saleprice(DATA_START_DAY),
            'weather_his': get_df_aggrbydate_weather_his(DATA_START_DAY),
            'saleqty': get_df_aggrbydate_saleqty(DATA_START_DAY),
            'category_his': get_df_category_his(TODAY_DATE),
            'goods_his': get_df_goods_his(TODAY_DATE),
            'canlendar_info': get_df_canlendar_info(DATA_START_DAY)
            }


def pre_data(data_sql):
    """
    :param data_sql:
    :return:
    """
    ## get the data
    airq = data_sql['airq'].set_index(['storeid', 'date'])
    weather = data_sql['weather_his'].set_index(['storeid', 'date'])
    canlendar_info = data_sql['canlendar_info']

    airq_weather = pd.concat([airq, weather], axis=1).fillna(0.0).reset_index()
    airq_weather_canlendar = pd.merge(airq_weather, canlendar_info, how='left', on=['date']).fillna(
        'Ordinary_day').drop(
        ['24term_name_en', '24term_code'], axis=1)
    airq_weather_canlendar['date'] = pd.to_datetime(airq_weather_canlendar['date'])
    airq_weather_canlendar['day_type'] = airq_weather_canlendar['weekday'].map(
        lambda x: 'work_day' if 1 <= x <= 5 else 'weekends')

    ##去除空位数据
    saleqty = data_sql['saleqty'].dropna()

    ## 对销量值进行转换
    saleqty['sale_qty'] = saleqty['sale_qty'].map(lambda x: np.log1p(
        float(x)) if float(x) > 0 else 0)
    saleqty['sale_qty_adj'] = saleqty['sale_qty_adj'].map(lambda x: np.log1p(
        float(x)) if float(x) > 0 else 0)

    ## 对折扣、价格进行处理
    saleprice = data_sql['saleprice'].dropna()
    ##增加一列 'onpromotion'
    saleprice['onpromotion'] = saleprice['discount_degree'].map(lambda x: 1 if x > 0 else 0).astype(bool)

    ## 构建训练集
    df_train_test = pd.merge(saleqty, saleprice, on=['date', 'storeid', 'goodsid', 'x'], how='left')
    df_train_test['date'] = pd.to_datetime(df_train_test['date'])
    df_train_test = df_train_test.drop(['sale_qty_adj', 'x'], axis=1)

    df_train = df_train_test[df_train_test.date < TODAY_DATE]  # validate_date
    df_test_temp = df_train_test[df_train_test.date >= TODAY_DATE]  ## validate_date
    ## 补缺数据 2017-08-31

    df_test_temp_app = df_test_temp[df_test_temp.date == '2016-08-31'].dropna()
    df_test_temp_app['date'] = pd.datetime(2017, 8, 31)
    df_test_temp = df_test_temp.append(df_test_temp_app).sort_values(by='date')

    df_test_temp = df_test_temp[df_test_temp.date <= PREDICTION_DAY]  ## validate_date+3 = pre_date
    df_test_temp[['date', 'storeid', 'goodsid', 'sale_qty']].to_csv('test.csv', index=None)

    df_test = df_test_temp.drop(['sale_qty'], axis=1).dropna().drop_duplicates().reset_index()
    df_test['id'] = df_test['index']
    df_test = df_test.drop('index', axis=1)
    df_test = df_test.set_index(['storeid', 'goodsid', 'date'])
    df_test = df_test.drop_duplicates()

    ## make goods info
    category_his = data_sql['category_his'].drop(['date'], axis=1)
    goods_his = data_sql['goods_his'].drop(['indate', 'date'], axis=1)

    ## 商品信息
    items_info = pd.merge(category_his, goods_his, on=['deptid'], how='right')
    items_info = items_info.set_index('goodsid')
    ##拷贝一份
    items_info_origin = items_info.copy()

    le = LabelEncoder()
    for col_name in items_info.columns:
        if items_info[[col_name]].dtypes[0] == 'object':
            items_info[col_name] = le.fit_transform(items_info[col_name].values)

    ##门店特征的汇总
    stores = pd.DataFrame()
    stores['storeid'] = ['A035', 'A206']
    stores['city'] = ['Beijing', 'Beijing']
    stores['state'] = ['open', 'open']
    stores['type'] = ['Middle', 'small']
    stores = stores.set_index("storeid")

    stores['city'] = le.fit_transform(stores['city'].values)
    stores['state'] = le.fit_transform(stores['state'].values)
    stores['type'] = le.fit_transform(stores['type'].values)

    ##日期特征
    for col_name in airq_weather_canlendar.columns:
        if airq_weather_canlendar[[col_name]].dtypes[0] == 'object' and col_name != 'storeid':
            airq_weather_canlendar[col_name] = le.fit_transform(airq_weather_canlendar[col_name].values)

    df_train['date'] = pd.to_datetime(df_train['date'])
    df_time = df_train.loc[df_train.date >= DATA_START_DAY]  ##获取数据的起点：start_date

    condition = airq_weather_canlendar
    condition_train = condition[condition.date < TODAY_DATE]  # validate_date
    condition_test_temp = condition[condition.date >= TODAY_DATE]  ## validate_date
    condition_test_temp = condition_test_temp[condition_test_temp.date <= PREDICTION_DAY]  ## validate_date+3 = pre_date
    condition_test = condition_test_temp
    condition_test = condition_test.set_index(['storeid', 'date'])
    condition_test = condition_test.drop_duplicates()
    condition_train['date'] = pd.to_datetime(condition_train['date'])
    condition_time = condition_train.loc[condition_train.date >= DATA_START_DAY]  ##获取数据的起点：start_date

    def make_time_series(data_df, data_test, index_name, feature_name):
        """
        输入：特征集合:data_df，DataFrame;测试部分的特征：data_test，DataFrame
             索引名称：list type；特征名称：字符串

        输出：时间序列化后的特征，DataFrame
        """
        temp_train = data_df.set_index(index_name)[[feature_name]].unstack(
            level=-1).fillna(0.0)
        temp_train.columns = temp_train.columns.get_level_values(1)
        temp_test = data_test[[feature_name]].unstack(level=-1).fillna(0.0)
        temp_test.columns = temp_test.columns.get_level_values(1)
        temp_test = temp_test.reindex(temp_train.index).fillna(0.0)
        temp = pd.concat([temp_train, temp_test], axis=1)
        del temp_test, temp_train
        return temp

    promo_time = make_time_series(df_time, df_test, ["storeid", "goodsid", "date"], "onpromotion")
    price_time = make_time_series(df_time, df_test, ["storeid", "goodsid", "date"], "price")
    discount_time = make_time_series(df_time, df_test, ["storeid", "goodsid", "date"], "discount_degree")

    # In[25]:

    AQI_time = make_time_series(condition_time, condition_test, ["storeid", "date"], "AQI")
    tem_time = make_time_series(condition_time, condition_test, ["storeid", "date"], "tem")
    pre1h_time = make_time_series(condition_time, condition_test, ["storeid", "date"], "pre1h")
    windspeed_time = make_time_series(condition_time, condition_test, ["storeid", "date"], "windspeed")
    rhu_time = make_time_series(condition_time, condition_test, ["storeid", "date"], "rhu")
    weekday_time = make_time_series(condition_time, condition_test, ["storeid", "date"], "weekday")
    term24_name_cn_time = make_time_series(condition_time, condition_test, ["storeid", "date"], "24term_name_cn")
    holiday_bylaw_time = make_time_series(condition_time, condition_test, ["storeid", "date"], "holiday_bylaw")
    day_type_time = make_time_series(condition_time, condition_test, ["storeid", "date"], "day_type")

    # In[26]:

    df_time = df_time.set_index(
        ["storeid", "goodsid", "date"])[["sale_qty"]].unstack(
        level=-1).fillna(0)
    df_time.columns = df_time.columns.get_level_values(1)

    # In[27]:

    items_info = items_info.reindex(df_time.index.get_level_values(1))
    items_info["perishable"] = items_info["keepdays"].map(lambda x: 1 if x < 3 else 0)
    stores = stores.reindex(df_time.index.get_level_values(0))

    df_time_item = df_time.groupby('goodsid')[df_time.columns].sum()
    promo_time_item = promo_time.groupby('goodsid')[promo_time.columns].sum()
    price_time_item = price_time.groupby('goodsid')[price_time.columns].sum()
    discount_time_item = discount_time.groupby('goodsid')[discount_time.columns].sum()

    condition_time_store = condition_time.groupby('storeid')[condition_time.columns].sum()
    AQI_time_store = AQI_time.groupby('storeid')[AQI_time.columns].sum()
    tem_time_store = tem_time.groupby('storeid')[tem_time.columns].sum()
    pre1h_time_store = pre1h_time.groupby('storeid')[pre1h_time.columns].sum()
    windspeed_time_store = windspeed_time.groupby('storeid')[windspeed_time.columns].sum()
    rhu_time_store = rhu_time.groupby('storeid')[rhu_time.columns].sum()
    weekday_time_store = weekday_time.groupby('storeid')[weekday_time.columns].sum()
    term24_name_cn_time_store = term24_name_cn_time.groupby('storeid')[term24_name_cn_time.columns].sum()
    holiday_bylaw_time_store = holiday_bylaw_time.groupby('storeid')[holiday_bylaw_time.columns].sum()
    day_type_time_store = day_type_time.groupby('storeid')[day_type_time.columns].sum()

    df_time_store_class = df_time.reset_index()
    df_time_store_class['keepdays'] = items_info['keepdays'].values
    df_time_store_class_index = df_time_store_class[['keepdays', 'storeid']]
    df_time_store_class = df_time_store_class.groupby(['keepdays', 'storeid'])[df_time.columns].sum()

    df_time_promo_store_class = promo_time.reset_index()
    df_time_promo_store_class['keepdays'] = items_info['keepdays'].values
    df_time_promo_store_class_index = df_time_promo_store_class[['keepdays', 'storeid']]
    df_time_promo_store_class = df_time_promo_store_class.groupby(['keepdays', 'storeid'])[promo_time.columns].sum()

    ##TODO 统计每天订单量

    def get_timespan(df, dt, minus, periods, freq='D'):
        return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

    def prepare_dataset(df, target_name, target_df, ttime, is_train=True, name_prefix=None):
        X = {
            target_name + "_20_time": get_timespan(target_df, ttime, 20, 20).sum(axis=1).values,
            target_name + "_30_time": get_timespan(target_df, ttime, 30, 30).sum(axis=1).values,
            target_name + "_60_time": get_timespan(target_df, ttime, 60, 60).sum(axis=1).values,
            target_name + "_3_time_aft": get_timespan(target_df, ttime + timedelta(days=PRE_DAYS), PRE_DAYS - 1,
                                                      PRE_DAYS).sum(axis=1).values,
            target_name + "_2_time_aft": get_timespan(target_df, ttime + timedelta(days=PRE_DAYS), PRE_DAYS - 1,
                                                      PRE_DAYS % 2 + round(PRE_DAYS / 2) - 1).sum(axis=1).values,
            target_name + "_1_time_aft": get_timespan(target_df, ttime + timedelta(days=PRE_DAYS), PRE_DAYS - 1,
                                                      PRE_DAYS - 2).sum(axis=1).values,
        }
        # print(X)

        for i in [1, 2, 3, 5, 7, 14, 20, 30, 60]:
            tmp1 = get_timespan(df, ttime, i, i)
            tmp2 = (get_timespan(target_df, ttime, i, i) > 0) * 1

            X['has_' + target_name + '_mean_%s' % i] = (tmp1 * tmp2.replace(0, np.nan)).mean(axis=1).values
            X['has_' + target_name + '_mean_%s_decay' % i] = (
                    tmp1 * tmp2.replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

            X['no_' + target_name + '_mean_%s' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan)).mean(axis=1).values
            X['no_' + target_name + '_mean_%s_decay' % i] = (
                    tmp1 * (1 - tmp2).replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

        for i in [1, 2, 3, 5, 7, 14, 20, 30, 60]:
            tmp = get_timespan(df, ttime, i, i)
            X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
            X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
            X['mean_%s' % i] = tmp.mean(axis=1).values
            X['median_%s' % i] = tmp.median(axis=1).values
            X['min_%s' % i] = tmp.min(axis=1).values
            X['max_%s' % i] = tmp.max(axis=1).values
            X['std_%s' % i] = tmp.std(axis=1).values

        for i in [1, 2, 3, 5, 7, 14, 20, 30, 60]:
            tmp = get_timespan(df, ttime + timedelta(days=-7), i, i)
            X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
            X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
            X['mean_%s_2' % i] = tmp.mean(axis=1).values
            X['median_%s_2' % i] = tmp.median(axis=1).values
            X['min_%s_2' % i] = tmp.min(axis=1).values
            X['max_%s_2' % i] = tmp.max(axis=1).values
            X['std_%s_2' % i] = tmp.std(axis=1).values

        for i in [1, 2, 3, 4, 5, 7, 14, 20, 30, 60]:
            tmp = get_timespan(df, ttime, i, i)
            X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
            X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
            X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

            tmp = get_timespan(target_df, ttime, i, i)
            X['has_' + target_name + '_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
            X['last_has_' + target_name + '_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
            X['first_has_' + target_name + '_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

        tmp = get_timespan(target_df, ttime + timedelta(days=PRE_DAYS), PRE_DAYS - 1, PRE_DAYS - 1)
        X['has_' + target_name + '_days_in_after_2_days'] = (tmp > 0).sum(axis=1).values
        X['last_has_' + target_name + '_day_in_after_2_days'] = i - ((tmp > 0) * np.arange(PRE_DAYS - 1)).max(
            axis=1).values
        X['first_has_' + target_name + '_day_in_after_2_days'] = ((tmp > 0) * np.arange(PRE_DAYS - 1, 0, -1)).max(
            axis=1).values

        for i in range(1, PRE_DAYS):
            X['day_%s_time' % i] = get_timespan(df, ttime, i, 1).values.ravel()

        for i in range(7):
            X['mean_4_dow{}_time'.format(i)] = get_timespan(df, ttime, 28 - i, 4, freq='7D').mean(axis=1).values
            X['mean_20_dow{}_time'.format(i)] = get_timespan(df, ttime, 70 - i, 10, freq='7D').mean(axis=1).values

        for i in range(-PRE_DAYS, PRE_DAYS):
            X["promo_{}".format(i)] = target_df[ttime + timedelta(days=i)].values.astype(np.uint8)

        X = pd.DataFrame(X)

        if is_train:
            y = df[
                pd.date_range(ttime, periods=PRE_DAYS)
            ].values
            return X, y
        if name_prefix is not None:
            X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
        return X

    print("Preparing dataset...")
    ttime = TRAIN_DATE
    X_l, y_l = [], []
    for i in range(num_days):
        delta = timedelta(days=7 * i)
        X_tmp, y_tmp = prepare_dataset(df_time, 'promotion_x', promo_time, ttime + delta, name_prefix='sales_quantity')

        X_tmp2 = prepare_dataset(df_time_item, 'promotion', promo_time_item, ttime + delta, is_train=False,
                                 name_prefix='promo_item')
        X_tmp2.index = df_time_item.index
        X_tmp2 = X_tmp2.reindex(df_time.index.get_level_values(1)).reset_index(drop=True)

        X_tmp3 = prepare_dataset(df_time_item, 'price', price_time_item, ttime + delta, is_train=False,
                                 name_prefix='price_item')
        X_tmp3.index = df_time_item.index
        X_tmp3 = X_tmp3.reindex(df_time.index.get_level_values(1)).reset_index(drop=True)

        X_tmp4 = prepare_dataset(df_time_item, 'discount', discount_time_item, ttime + delta, is_train=False,
                                 name_prefix='discount_item')
        X_tmp4.index = df_time_item.index
        X_tmp4 = X_tmp4.reindex(df_time.index.get_level_values(1)).reset_index(drop=True)

        ##condition features
        X_con1 = prepare_dataset(AQI_time_store, 'AQI', AQI_time_store, ttime + delta, is_train=False,
                                 name_prefix='AQI_store')
        X_con1.index = AQI_time_store.index
        X_con1 = X_con1.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

        X_con2 = prepare_dataset(AQI_time_store, 'tem', tem_time_store, ttime + delta, is_train=False,
                                 name_prefix='tem_store')
        X_con2.index = AQI_time_store.index
        X_con2 = X_con2.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

        X_con3 = prepare_dataset(AQI_time_store, 'rhu', rhu_time_store, ttime + delta, is_train=False,
                                 name_prefix='rhu_store')
        X_con3.index = AQI_time_store.index
        X_con3 = X_con3.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

        X_con4 = prepare_dataset(AQI_time_store, 'pre1h', pre1h_time_store, ttime + delta, is_train=False,
                                 name_prefix='pre1h_store')
        X_con4.index = AQI_time_store.index
        X_con4 = X_con4.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

        X_con5 = prepare_dataset(AQI_time_store, 'day_type', day_type_time_store, ttime + delta, is_train=False,
                                 name_prefix='day_type_store')
        X_con5.index = AQI_time_store.index
        X_con5 = X_con5.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

        X_con6 = prepare_dataset(AQI_time_store, 'windspeed', windspeed_time_store, ttime + delta, is_train=False,
                                 name_prefix='windspeed_store')
        X_con6.index = AQI_time_store.index
        X_con6 = X_con6.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

        X_con7 = prepare_dataset(AQI_time_store, 'term24_name_cn', term24_name_cn_time_store, ttime + delta,
                                 is_train=False,
                                 name_prefix='term24_name_cn_store')
        X_con7.index = AQI_time_store.index
        X_con7 = X_con7.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

        X_con8 = prepare_dataset(AQI_time_store, 'weekday', weekday_time_store, ttime + delta, is_train=False,
                                 name_prefix='weekday_store')
        X_con8.index = AQI_time_store.index
        X_con8 = X_con8.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

        X_con9 = prepare_dataset(AQI_time_store, 'holiday_bylaw', holiday_bylaw_time_store, ttime + delta,
                                 is_train=False,
                                 name_prefix='holiday_bylaw_store')
        X_con9.index = AQI_time_store.index
        X_con9 = X_con9.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

        X_tmp5 = prepare_dataset(df_time_store_class, 'keepdays_store_class', df_time_promo_store_class, ttime + delta,
                                 is_train=False, name_prefix='store_class')
        X_tmp5.index = df_time_store_class.index
        X_tmp5 = X_tmp5.reindex(df_time_store_class_index).reset_index(drop=True)

        X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3, X_tmp4, X_tmp5, items_info.reset_index(), stores.reset_index()],
                          axis=1)

        X_tmp_con = X_tmp
        X_tmp_con['storeid'] = X_tmp_con.storeid.map({'A035': 35})
        X_l.append(X_tmp_con)
        y_l.append(y_tmp)

        del X_tmp2, X_tmp3, X_tmp4, X_tmp5, X_con1, X_con2, X_con3, X_con4, X_con5, X_con6, X_con7, X_con8, X_con9
        gc.collect()

    # In[ ]:

    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)

    del X_l, y_l
    X_val, y_val = prepare_dataset(df_time, 'promotion_x', promo_time, VAL_DATE)

    X_val2 = prepare_dataset(df_time_item, 'promotion', promo_time_item, VAL_DATE, is_train=False,
                             name_prefix='promo_item')
    X_val2.index = df_time_item.index
    X_val2 = X_val2.reindex(df_time.index.get_level_values(1)).reset_index(drop=True)

    X_val3 = prepare_dataset(df_time_item, 'price', price_time_item, VAL_DATE, is_train=False,
                             name_prefix='price_item')
    X_val3.index = df_time_item.index
    X_val3 = X_val3.reindex(df_time.index.get_level_values(1)).reset_index(drop=True)

    X_val4 = prepare_dataset(df_time_item, 'discount', discount_time_item, VAL_DATE, is_train=False,
                             name_prefix='discount_item')
    X_val4.index = df_time_item.index
    X_val4 = X_val4.reindex(df_time.index.get_level_values(1)).reset_index(drop=True)

    ##condition features
    X_con_val1 = prepare_dataset(AQI_time_store, 'AQI', AQI_time_store, VAL_DATE, is_train=False,
                                 name_prefix='AQI_store')
    X_con_val1.index = AQI_time_store.index
    X_con_val1 = X_con_val1.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_val2 = prepare_dataset(AQI_time_store, 'tem', tem_time_store, VAL_DATE, is_train=False,
                                 name_prefix='tem_store')
    X_con_val2.index = AQI_time_store.index
    X_con_val2 = X_con_val2.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_val3 = prepare_dataset(AQI_time_store, 'rhu', rhu_time_store, VAL_DATE, is_train=False,
                                 name_prefix='rhu_store')
    X_con_val3.index = AQI_time_store.index
    X_con_val3 = X_con_val3.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_val4 = prepare_dataset(AQI_time_store, 'pre1h', pre1h_time_store, VAL_DATE, is_train=False,
                                 name_prefix='pre1h_store')
    X_con_val4.index = AQI_time_store.index
    X_con_val4 = X_con_val4.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_val5 = prepare_dataset(AQI_time_store, 'day_type', day_type_time_store, VAL_DATE, is_train=False,
                                 name_prefix='day_type_store')
    X_con_val5.index = AQI_time_store.index
    X_con_val5 = X_con_val5.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_val6 = prepare_dataset(AQI_time_store, 'windspeed', windspeed_time_store, VAL_DATE, is_train=False,
                                 name_prefix='windspeed_store')
    X_con_val6.index = AQI_time_store.index
    X_con_val6 = X_con_val6.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_val7 = prepare_dataset(AQI_time_store, 'term24_name_cn', term24_name_cn_time_store, VAL_DATE,
                                 is_train=False,
                                 name_prefix='term24_name_cn_store')
    X_con_val7.index = AQI_time_store.index
    X_con_val7 = X_con_val7.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_val8 = prepare_dataset(AQI_time_store, 'weekday', weekday_time_store, VAL_DATE, is_train=False,
                                 name_prefix='weekday_store')
    X_con_val8.index = AQI_time_store.index
    X_con_val8 = X_con_val8.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_val9 = prepare_dataset(AQI_time_store, 'holiday_bylaw', holiday_bylaw_time_store, VAL_DATE,
                                 is_train=False,
                                 name_prefix='holiday_bylaw_store')
    X_con_val9.index = AQI_time_store.index
    X_con_val9 = X_con_val9.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_val5 = prepare_dataset(df_time_store_class, 'keepdays_store_class', df_time_promo_store_class, VAL_DATE,
                             is_train=False, name_prefix='store_class')
    X_val5.index = df_time_store_class.index
    X_val5 = X_val5.reindex(df_time_store_class_index).reset_index(drop=True)

    X_val_tmp = pd.concat([X_val, X_val2, X_val3, X_val4, X_val5, items_info.reset_index(), stores.reset_index()],
                          axis=1)

    X_val = X_val_tmp
    X_val['storeid'] = X_val.storeid.map({'A035': 35})

    X_test = prepare_dataset(df_time, 'promotion_x', promo_time, PRE_DATE, is_train=False)

    X_test2 = prepare_dataset(df_time_item, 'promotion', promo_time_item, PRE_DATE, is_train=False,
                              name_prefix='promo_item')
    X_test2.index = df_time_item.index
    X_test2 = X_test2.reindex(df_time.index.get_level_values(1)).reset_index(drop=True)

    X_test3 = prepare_dataset(df_time_item, 'price', price_time_item, PRE_DATE, is_train=False,
                              name_prefix='price_item')
    X_test3.index = df_time_item.index
    X_test3 = X_test3.reindex(df_time.index.get_level_values(1)).reset_index(drop=True)

    X_test4 = prepare_dataset(df_time_item, 'discount', discount_time_item, PRE_DATE, is_train=False,
                              name_prefix='discount_item')
    X_test4.index = df_time_item.index
    X_test4 = X_test4.reindex(df_time.index.get_level_values(1)).reset_index(drop=True)

    ##condition features
    X_con_test1 = prepare_dataset(AQI_time_store, 'AQI', AQI_time_store, PRE_DATE, is_train=False,
                                  name_prefix='AQI_store')
    X_con_test1.index = AQI_time_store.index
    X_con_test1 = X_con_test1.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_test2 = prepare_dataset(AQI_time_store, 'tem', tem_time_store, PRE_DATE, is_train=False,
                                  name_prefix='tem_store')
    X_con_test2.index = AQI_time_store.index
    X_con_test2 = X_con_test2.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_test3 = prepare_dataset(AQI_time_store, 'rhu', rhu_time_store, PRE_DATE, is_train=False,
                                  name_prefix='rhu_store')
    X_con_test3.index = AQI_time_store.index
    X_con_test3 = X_con_test3.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_test4 = prepare_dataset(AQI_time_store, 'pre1h', pre1h_time_store, PRE_DATE, is_train=False,
                                  name_prefix='pre1h_store')
    X_con_test4.index = AQI_time_store.index
    X_con_test4 = X_con_test4.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_test5 = prepare_dataset(AQI_time_store, 'day_type', day_type_time_store, PRE_DATE, is_train=False,
                                  name_prefix='day_type_store')
    X_con_test5.index = AQI_time_store.index
    X_con_test5 = X_con_test5.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_test6 = prepare_dataset(AQI_time_store, 'windspeed', windspeed_time_store, PRE_DATE, is_train=False,
                                  name_prefix='windspeed_store')
    X_con_test6.index = AQI_time_store.index
    X_con_test6 = X_con_test6.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_test7 = prepare_dataset(AQI_time_store, 'term24_name_cn', term24_name_cn_time_store, PRE_DATE,
                                  is_train=False,
                                  name_prefix='term24_name_cn_store')
    X_con_test7.index = AQI_time_store.index
    X_con_test7 = X_con_test7.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_test8 = prepare_dataset(AQI_time_store, 'weekday', weekday_time_store, PRE_DATE, is_train=False,
                                  name_prefix='weekday_store')
    X_con_test8.index = AQI_time_store.index
    X_con_test8 = X_con_test8.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_con_test9 = prepare_dataset(AQI_time_store, 'holiday_bylaw', holiday_bylaw_time_store, PRE_DATE,
                                  is_train=False,
                                  name_prefix='holiday_bylaw_store')
    X_con_test9.index = AQI_time_store.index
    X_con_test9 = X_con_test9.reindex(AQI_time.index.get_level_values(0)).reset_index(drop=True)

    X_test5 = prepare_dataset(df_time_store_class, 'keepdays_store_class', df_time_promo_store_class, PRE_DATE,
                              is_train=False, name_prefix='store_class')
    X_test5.index = df_time_store_class.index
    X_test5 = X_test5.reindex(df_time_store_class_index).reset_index(drop=True)

    X_test_tmp = pd.concat([X_test, X_test2, X_test3, X_test4, X_test5, items_info.reset_index(), stores.reset_index()],
                           axis=1)

    X_test = X_test_tmp
    X_test['storeid'] = X_test.storeid.map({'A035': 35})

    del X_test2, X_test3, X_test4, X_test5, X_val2, df_time_item, promo_time_item, df_time_store_class, df_time_promo_store_class, df_time_store_class_index
    del X_con_test1, X_con_test2, X_con_test3, X_con_test4, X_con_test5, X_con_test6, X_val3, X_val4, X_val5
    del X_con_val1, X_con_val2, X_con_val3, X_con_val4, X_con_val5, X_con_val6, X_con_val7, X_con_val8, X_con_val9
    del condition_time_store, AQI_time_store, tem_time_store, pre1h_time_store, windspeed_time_store, rhu_time_store, weekday_time_store, term24_name_cn_time_store
    del holiday_bylaw_time_store, day_type_time_store, X_con_test8, X_con_test7, X_con_test9

    gc.collect()

    return X_train, y_train, X_test, X_val, y_val, items_info, items_info_origin, df_time, df_test


def build_model():
    model = Sequential()
    model.add(LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(.1))

    model.add(Dense(256))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.1))

    model.add(Dense(256))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.1))

    model.add(Dense(128))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(64))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(32))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(16))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(1))

    return model


"""生成报告"""


def sub_to_mysql(submission_df):
    """"

    """

    database_name = 'MARKET_DB'
    table_name = 'model_predct_result_dev'
    columns_nm = ['time_create', 'date', 'storeid', 'goodsid', 'goodsname', 'pred_sale_volume', 'model_param']

    submission_df.columns = columns_nm

    def DataFrame_to_MySQL(dataframe, database_name, table_name):
        yconnect = create_engine('mysql+pymysql://xx:xx@xx.xx.xx.xx:3306/?charset=utf8')
        pd.io.sql.to_sql(dataframe, table_name, yconnect, schema=database_name, if_exists='append', index=False)

    DataFrame_to_MySQL(submission_df, database_name, table_name)


data_sql = Get_Data_from_sql_online(TODAY_DATE, DATA_START_DAY)

X_train, y_train, X_test, X_val, y_val, items_info, items_info_origin, df_time, df_test = pre_data(data_sql)

X_train = pd.DataFrame(X_train)
X_val = pd.DataFrame(X_val)
X_test = pd.DataFrame(X_test)
scaler = StandardScaler()
data_df = pd.concat([X_train, X_val, X_test]).fillna(0.0)
scaler.fit(data_df)
X_train[:] = scaler.transform(X_train.fillna(0.0))
X_val[:] = scaler.transform(X_val.fillna(0.0))
X_test[:] = scaler.transform(X_test.fillna(0.0))

X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
X_val = X_val.as_matrix()
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

N_EPOCHS = 2000

val_pred = []
test_pred = []
# wtpath = 'weights.hdf5'  # To save best epoch. But need Keras bug to be fixed first.
sample_weights = np.array(pd.concat([items_info["perishable"]] * num_days) * 0.25 + 1)
for i in range(PRE_DAYS):
    print("=" * 50)
    print("Step %d" % (i + 1))
    print("=" * 50)
    y = y_train[:, i]
    y_mean = y.mean()
    xv = X_val
    yv = y_val[:, i]
    model = build_model()
    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-5, mode='min')
    ]
    model.fit(X_train, y - y_mean, batch_size=65536, epochs=N_EPOCHS, verbose=2,
              sample_weight=sample_weights, validation_data=(xv, yv - y_mean), callbacks=callbacks)
    val_pred.append(model.predict(X_val) + y_mean)
    test_pred.append(model.predict(X_test) + y_mean)

weight = items_info["perishable"] * 0.25 + 1
err = (y_val - np.array(val_pred).squeeze(axis=2).transpose()) ** 2
err = err.sum(axis=1) * weight
err = np.sqrt(err.sum() / weight.sum() / PRE_DAYS)
print('nwrmsle = {}'.format(err))

y_val = np.array(val_pred).squeeze(axis=2).transpose()
df_preds = pd.DataFrame(
    y_val, index=df_time.index,
    columns=pd.date_range(VAL_DATE, periods=PRE_DAYS)
).stack().to_frame("sale_qty")
df_preds.index.set_names(["storeid", "goodsid", "date"], inplace=True)
df_preds["sale_qty"] = np.clip(np.expm1(df_preds["sale_qty"]), 0, 10000000)
df_preds.reset_index().to_csv('nn_cv.csv', index=False)

print("Making submission...")
y_test = np.array(test_pred).squeeze(axis=2).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_time.index,
    columns=pd.date_range(TODAY_DATE, periods=PRE_DAYS)
).stack().to_frame("sale_qty")
df_preds.index.set_names(["storeid", "goodsid", "date"], inplace=True)

submission = df_preds.fillna(0)
submission["pred_sale_volume"] = np.clip(np.expm1(submission["sale_qty"]), 0, 10000000)
submission = submission.reset_index()

submission.insert(0, 'time_create', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
params = {}
params['Model_name'] = 'LSTM_V1'
params['Rediction_days'] = PRE_DAYS
params['RD'] = 'XingqiangChen'
params['N_EPOCHS'] = 2000
json_str = json.dumps(params)

submission['goodsid'] = submission['goodsid'].astype(str)
submission.insert(1, 'model_param', params['Model_name'])

submission_item = pd.merge(items_info_origin.reset_index()[['goodsid', 'name', 'spname']].astype(str),
                           submission.reset_index(), on=['goodsid'], how='right')
submission_item['goodsname'] = submission_item['name'] + [':'] + submission_item['spname']
submission_df = submission_item[
    ['time_create', 'date', 'storeid', 'goodsid', 'goodsname', 'pred_sale_volume', 'model_param']]

sub_to_mysql(submission_df)
