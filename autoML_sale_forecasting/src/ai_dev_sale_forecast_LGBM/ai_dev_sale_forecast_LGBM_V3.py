# coding: utf-8

"""
ThinkPage Retail Forecasting

version 4.0

dev:xingqiang chen
date:2018-03-30
"""
# coding=utf-8

import gc
import sys
import time

import argparse
import datetime as DT
import json
import numpy as np
import pandas as pd
import pymysql
import xgboost as xgb
from datetime import date, timedelta
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import ExtraTreeRegressor
from sqlalchemy import create_engine


def T(FORMAT_DATE, i):
    return (DT.datetime.strptime(FORMAT_DATE, '%Y-%m-%d')
            + DT.timedelta(i)).strftime('%Y%m%d')


def FT(FORMAT_DATE, i):
    return (DT.datetime.strptime(FORMAT_DATE, '%Y-%m-%d')
            + DT.timedelta(i)).strftime('%Y-%m-%d')


def formatdate_a(d, i):
    return (DT.datetime.strptime(d, '%Y-%m-%d')
            + DT.timedelta(i)).strftime('%Y-%m-%d')


def date_dt(DAY):
    return date(int(DAY[0:4]), int(DAY[5:7]), int(DAY[8:]))


def parse_arguments(argv):
    # DATE = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    DATE = '2018-03-12'
    parser = argparse.ArgumentParser()

    parser.add_argument('--DATE', type=str, help='the date of starting predction.', default=DATE)
    return parser.parse_args(argv)


"""
Define The Global Variables and Constants
"""
TODAY_DATE = parse_arguments(sys.argv[1:]).DATE  ## for test TODO
TODAY_DATE = '2018-03-12'

print("## 当前日期：" + TODAY_DATE)
PRE_DAYS = 3
print("## 预测天数:" + str(PRE_DAYS))
DATA_START_DAY = FT(TODAY_DATE, -2000)
print("## 获取数据的长度{}天,起点日期：".format(700) + DATA_START_DAY)

TRAIN_START_DAY = FT(TODAY_DATE, -365)
print("## 训练数据的长度{}天,起点日期:".format(365) + TRAIN_START_DAY)

PREDICTION_DAY = FT(TODAY_DATE, PRE_DAYS)
print("## 预测日期：" + PREDICTION_DAY)

num_days = 6

##是否加入GOOID infF:bool
SKU_INFO = False
LTD_weekends = 2000
LTD_job = 1400

##add week type
##工作日的序列长度
before_job_days = 9
## 休息日的序列长度
before_weekends_days = 16
## 预测天数
predict_days = 3
##是否考虑货品属性信息

MODEL = 'XGBOOST'


def Get_Data_from_sql_online(TODAY_DATE, DATA_START_DAY):
    # db info
    host = '47.97.17.51'
    user = 'sophon'
    password = 'IAmSophon_4523'
    port = 3306

    def get_df_actual_purchase_his(DATA_START_DAY):
        """
        """
        # purchase data from db
        cols = ['date', 'storeid', 'goodsid', 'cost', 'qty', 'purchase_num_of_days']
        database = 'CHAOSHIFA_DB'
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
        database = 'CHAOSHIFA_DB'
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
        database = 'CHAOSHIFA_DB'
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
        database = 'CHAOSHIFA_DB'
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
        database = 'CHAOSHIFA_DB'
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
        database = 'CHAOSHIFA_DB'
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
        database = 'CHAOSHIFA_DB'
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


def Get_Training_Data(df_train_test, condition, items_info):
    """
       制作训练数据
       输入：准备好的天气、空气质量和销量实际发生数据
       输出：构建好的预测数据集

    """
    tags_of_volume = ['date', 'storeid', 'goodsid', 'sale_qty']

    df_train_test['week'] = df_train_test['date'].map(
        lambda x: int(DT.datetime(int(x[:4]), int(x[5:7]), int(x[8:])).strftime("%w")))

    history_data = df_train_test

    df_0 = history_data[history_data['week'] == 0]
    df_1 = history_data[history_data['week'] == 6]

    history_weekends = pd.concat([df_0, df_1], axis=0)

    history_job = history_data[0 < history_data['week']]
    history_job = history_job[history_job['week'] < 6]

    history_weekends.sort_values(['date'], inplace=True, ascending=True)
    history_job.sort_values(['date'], inplace=True, ascending=True)

    ## make the time series

    ## do for weekends
    dates = sorted(list(set(history_weekends.date)), reverse=True)
    df_all_day = history_weekends[tags_of_volume]

    df_day_0 = df_all_day[df_all_day.date == dates[1]]
    df_day_1 = df_all_day[df_all_day.date == dates[2]]
    df_day_0 = df_day_0[['storeid', 'goodsid', 'sale_qty']]
    df_day_1 = df_day_1[['storeid', 'goodsid', 'sale_qty']]
    df_day_0 = df_day_0.rename(columns={'sale_qty': dates[1]})
    df_day_1 = df_day_1.rename(columns={'sale_qty': dates[2]})
    df_day_nw = pd.merge(df_day_0, df_day_1, how='outer', on=['storeid', 'goodsid'])

    for day in dates[3:]:
        df_day = df_all_day[df_all_day.date == day]
        df_day_nx = df_day[['storeid', 'goodsid', 'sale_qty']]
        df_day_nx = df_day_nx.rename(columns={'sale_qty': day})
        df_day_nw = pd.merge(df_day_nw, df_day_nx, how='outer', on=['storeid', 'goodsid'])
    df_day_nw = df_day_nw.fillna(0)

    ##do for job day

    dates = sorted(list(set(history_job.date)), reverse=True)
    df_all_day = history_job[tags_of_volume]

    df_day_0 = df_all_day[df_all_day.date == dates[1]]
    df_day_1 = df_all_day[df_all_day.date == dates[2]]
    df_day_0 = df_day_0[['storeid', 'goodsid', 'sale_qty']]
    df_day_1 = df_day_1[['storeid', 'goodsid', 'sale_qty']]
    df_day_0 = df_day_0.rename(columns={'sale_qty': dates[1]})
    df_day_1 = df_day_1.rename(columns={'sale_qty': dates[2]})
    df_day_nj = pd.merge(df_day_0, df_day_1, how='outer', on=['storeid', 'goodsid'])

    for day in dates[3:]:
        df_day = df_all_day[df_all_day.date == day]
        df_day_nx = df_day[['storeid', 'goodsid', 'sale_qty']]
        df_day_nx = df_day_nx.rename(columns={'sale_qty': day})
        df_day_nj = pd.merge(df_day_nj, df_day_nx, how='outer', on=['storeid', 'goodsid'])
    df_day_nj = df_day_nj.fillna(0)

    ### make training and prediction dataset
    week_one_hot_j = pd.get_dummies(history_job.week, 'week')

    ### make training and prediction dataset
    week_one_hot_w = pd.get_dummies(history_weekends.week, 'week')

    features_w = pd.concat([history_weekends, week_one_hot_w], axis=1)
    features_w = pd.merge(features_w, condition, on=['storeid', 'date'], how='left')
    features_w = pd.merge(features_w, items_info.reset_index(), on=['goodsid'], how='left')
    features_w = features_w.drop(features_w[['sale_qty']], axis=1)

    features_j = pd.concat([history_job, week_one_hot_j], axis=1)
    features_j = pd.merge(features_j, condition, on=['storeid', 'date'], how='left')
    features_j = pd.merge(features_j, items_info.reset_index(), on=['goodsid'], how='left')
    features_j = features_j.drop(features_j[['sale_qty']], axis=1)

    if before_job_days > before_weekends_days:

        Before_cols = ['before_volume_' + str(i).zfill(2) for i in range(0, before_job_days)]
    else:
        Before_cols = ['before_volume_' + str(i).zfill(2) for i in range(0, before_weekends_days)]

    dates = sorted(list(set(history_job.date)), reverse=True)
    before_0 = df_day_nj[df_day_nj.columns[-before_job_days:]]
    before_0.columns = Before_cols[0:before_job_days]
    before_title = df_day_nj[tags_of_volume[1:3]]
    before_end_0 = pd.concat([before_title, before_0], axis=1)
    before_end_0['date'] = df_day_nj.columns[-(before_job_days + 1)]

    history_jobday_0 = history_job[history_job.date.isin([df_day_nj.columns[-(before_job_days + 1)]])]
    history_jobday_0 = history_jobday_0[tags_of_volume]

    df_change = before_end_0['date']
    before_end_0 = before_end_0.drop('date', axis=1)
    before_end_0.insert(2, 'date', df_change)

    before_end_0 = pd.merge(history_jobday_0, before_end_0, how='left',
                            on=['storeid', 'goodsid', 'date'])
    before_end_0 = before_end_0.drop_duplicates()

    before_1 = df_day_nj[df_day_nj.columns[-(before_job_days + 1):-1]]
    before_1.columns = Before_cols[0:before_job_days]
    before_title = df_day_nj[tags_of_volume[1:3]]
    before_end_1 = pd.concat([before_title, before_1], axis=1)
    before_end_1['date'] = df_day_nj.columns[-(before_job_days + 2)]

    history_jobday_1 = history_job[history_job.date.isin([df_day_nj.columns[-(before_job_days + 2)]])]
    history_jobday_1 = history_jobday_1[tags_of_volume]

    df_change = before_end_1['date']
    before_end_1 = before_end_1.drop('date', axis=1)
    before_end_1.insert(2, 'date', df_change)

    before_end_1 = pd.merge(history_jobday_1, before_end_1, how='left',
                            on=['storeid', 'goodsid', 'date'])

    ## drop duplicartes
    before_end_1 = before_end_1.drop_duplicates()

    before_data_j = pd.concat([before_end_0, before_end_1], axis=0)

    for i in range(0, len(dates[1:-(before_job_days + 2)]) + 1):
        before = df_day_nj[df_day_nj.columns[-(before_job_days + 2) - i:-2 - i]]
        before.columns = Before_cols[0:before_job_days]
        before_title = df_day_nj[tags_of_volume[1:3]]
        before_end = pd.concat([before_title, before], axis=1)
        before_end['date'] = dates[-(before_job_days + 2) - i - 1]

        history_jobday = history_job[history_job.date.isin([dates[-(before_job_days + 2) - i - 1]])]
        history_jobday = history_jobday[tags_of_volume]

        df_change = before_end['date']
        before_end = before_end.drop('date', axis=1)
        before_end.insert(2, 'date', df_change)

        before_end = pd.merge(history_jobday, before_end, how='left',
                              on=['storeid', 'goodsid', 'date'])

        ## drop duplicartes
        before_end = before_end.drop_duplicates()

        before_data_j = pd.concat([before_data_j, before_end], axis=0)

    ## Make for weekends

    dates = sorted(list(set(history_weekends.date)), reverse=True)
    before_0 = df_day_nw[df_day_nw.columns[-before_weekends_days:]]
    before_0.columns = Before_cols[0:before_weekends_days]
    before_title = df_day_nw[tags_of_volume[1:3]]
    before_end_0 = pd.concat([before_title, before_0], axis=1)
    before_end_0['date'] = dates[-(before_weekends_days + 1)]

    history_weekday_0 = history_weekends[history_weekends.date.isin([df_day_nw.columns[-(before_weekends_days + 1)]])]
    history_weekday_0 = history_weekday_0[tags_of_volume]

    df_change = before_end_0['date']
    before_end_0 = before_end_0.drop('date', axis=1)
    before_end_0.insert(2, 'date', df_change)

    before_end_0 = pd.merge(history_weekday_0, before_end_0, how='left',
                            on=['storeid', 'goodsid', 'date'])
    before_end_0 = before_end_0.drop_duplicates()

    before_1 = df_day_nw[df_day_nw.columns[-(before_weekends_days + 1):-1]]
    before_1.columns = Before_cols[0:before_weekends_days]
    before_title = df_day_nw[tags_of_volume[1:3]]
    before_end_1 = pd.concat([before_title, before_1], axis=1)
    before_end_1['date'] = dates[-(before_weekends_days + 2)]

    history_weekday_1 = history_weekends[history_weekends.date.isin([df_day_nw.columns[-(before_weekends_days + 2)]])]
    history_weekday_1 = history_weekday_1[tags_of_volume]

    df_change = before_end_1['date']
    before_end_1 = before_end_1.drop('date', axis=1)
    before_end_1.insert(2, 'date', df_change)

    before_end_1 = pd.merge(history_weekday_1, before_end_1, how='left',
                            on=['storeid', 'goodsid', 'date'])

    ## drop duplicartes
    before_end_1 = before_end_1.drop_duplicates()

    before_data_w = pd.concat([before_end_0, before_end_1], axis=0)

    for i in range(0, len(dates[1:-(before_weekends_days + 2)]) + 1):
        before = df_day_nw[df_day_nw.columns[-(before_weekends_days + 2) - i:-2 - i]]
        before.columns = Before_cols[0:before_weekends_days]
        before_title = df_day_nw[tags_of_volume[1:3]]
        before_end = pd.concat([before_title, before], axis=1)
        before_end['date'] = dates[-(before_weekends_days + 2) - i - 1]

        history_weekday = history_weekends[history_weekends.date.isin([dates[-(before_weekends_days + 2) - i - 1]])]
        history_weekday = history_weekday[tags_of_volume]

        df_change = before_end['date']
        before_end = before_end.drop('date', axis=1)
        before_end.insert(2, 'date', df_change)

        before_end = pd.merge(history_weekday, before_end, how='left',
                              on=['storeid', 'goodsid', 'date'])

        ## drop duplicartes
        before_end = before_end.drop_duplicates()

        before_data_w = pd.concat([before_data_w, before_end], axis=0)

    features_end_weekends = pd.merge(before_data_w, features_w, how='left', on=['date', 'storeid', 'goodsid'])

    features_end_job = pd.merge(before_data_j, features_j, how='left', on=['date', 'storeid', 'goodsid'])

    ##add more features for goodsid level

    return {
        'features_weekends': features_end_weekends
        , 'features_job': features_end_job
    }


def pre_data(data_sql, items_info, stores, dates):
    """

    :param data_sql:
    :param items_info:
    :param stores:
    :return:
    """
    ## get the data

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

    df_train['date'] = pd.to_datetime(df_train['date'])
    df_time = df_train.loc[df_train.date >= DATA_START_DAY]  ##获取数据的起点：start_date

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

    df_time = df_time.set_index(
        ["storeid", "goodsid", "date"])[["sale_qty"]].unstack(
        level=-1).fillna(0)
    df_time.columns = df_time.columns.get_level_values(1)

    items_info = items_info.reindex(df_time.index.get_level_values(1))
    items_info["perishable"] = items_info["keepdays"].map(lambda x: 1 if x < 3 else 0)
    stores = stores.reindex(df_time.index.get_level_values(0))

    df_time_item = df_time.groupby('goodsid')[df_time.columns].sum()
    promo_time_item = promo_time.groupby('goodsid')[promo_time.columns].sum()
    price_time_item = price_time.groupby('goodsid')[price_time.columns].sum()
    discount_time_item = discount_time.groupby('goodsid')[discount_time.columns].sum()

    df_time_store_class = df_time.reset_index()
    df_time_store_class['keepdays'] = items_info['keepdays'].values
    df_time_store_class_index = df_time_store_class[['keepdays', 'storeid']]
    df_time_store_class = df_time_store_class.groupby(['keepdays', 'storeid'])[df_time.columns].sum()

    df_time_promo_store_class = promo_time.reset_index()
    df_time_promo_store_class['keepdays'] = items_info['keepdays'].values
    df_time_promo_store_class = df_time_promo_store_class.groupby(['keepdays', 'storeid'])[promo_time.columns].sum()

    ##TODO 统计每天订单量

    def get_timespan(df, dt, minus, periods, freq='D'):
        return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

    def prepare_dataset(df, target_name, target_df, ttime, is_train=True, name_prefix=None):
        X = {
            target_name + "_20_time": get_timespan(target_df, ttime, 2, 2).sum(axis=1).values,
            target_name + "_30_time": get_timespan(target_df, ttime, 3, 3).sum(axis=1).values,
            target_name + "_3_time_aft": get_timespan(target_df, ttime + timedelta(days=PRE_DAYS), PRE_DAYS - 1,
                                                      PRE_DAYS).sum(axis=1).values,
            target_name + "_2_time_aft": get_timespan(target_df, ttime + timedelta(days=PRE_DAYS), PRE_DAYS - 1,
                                                      PRE_DAYS % 2 + round(PRE_DAYS / 2) - 1).sum(axis=1).values,
            target_name + "_1_time_aft": get_timespan(target_df, ttime + timedelta(days=PRE_DAYS), PRE_DAYS - 1,
                                                      PRE_DAYS - 2).sum(axis=1).values,
        }
        # print(X)

        for i in [1, 2, 3, 5, 7]:
            tmp1 = get_timespan(df, ttime, i, i)
            tmp2 = (get_timespan(target_df, ttime, i, i) > 0) * 1

            X['has_' + target_name + '_mean_%s' % i] = (tmp1 * tmp2.replace(0, np.nan)).mean(axis=1).values
            X['has_' + target_name + '_mean_%s_decay' % i] = (
                    tmp1 * tmp2.replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

            X['no_' + target_name + '_mean_%s' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan)).mean(axis=1).values
            X['no_' + target_name + '_mean_%s_decay' % i] = (
                    tmp1 * (1 - tmp2).replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

        for i in [1, 2, 3, 5, 7]:
            tmp = get_timespan(df, ttime, i, i)
            X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
            X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
            X['mean_%s' % i] = tmp.mean(axis=1).values
            X['median_%s' % i] = tmp.median(axis=1).values
            X['min_%s' % i] = tmp.min(axis=1).values
            X['max_%s' % i] = tmp.max(axis=1).values
            X['std_%s' % i] = tmp.std(axis=1).values

        for i in [3, 5, 7]:
            tmp = get_timespan(df, ttime + timedelta(days=7), i, i)
            X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
            X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
            X['mean_%s_2' % i] = tmp.mean(axis=1).values
            X['median_%s_2' % i] = tmp.median(axis=1).values
            X['min_%s_2' % i] = tmp.min(axis=1).values
            X['max_%s_2' % i] = tmp.max(axis=1).values
            X['std_%s_2' % i] = tmp.std(axis=1).values

        for i in [1, 2, 3, 4, 5, 7]:
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

        for i in range(3):
            X['mean_4_dow{}_time'.format(i)] = get_timespan(df, ttime, 3 - i, 1, freq='3D').mean(axis=1).values
            X['mean_20_dow{}_time'.format(i)] = get_timespan(df, ttime, 6 - i, 2, freq='3D').mean(axis=1).values

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

    def make_features(PRE_DATE_str):

        PRE_DATE = date_dt(PRE_DATE_str)
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

        X_test5 = prepare_dataset(df_time_store_class, 'keepdays_store_class', df_time_promo_store_class, PRE_DATE,
                                  is_train=False, name_prefix='store_class')
        X_test5.index = df_time_store_class.index
        X_test5 = X_test5.reindex(df_time_store_class_index).reset_index(drop=True)

        X_test_tmp = pd.concat(
            [X_test, X_test2, X_test3, X_test4, X_test5, items_info.reset_index(), stores.reset_index()],
            axis=1)

        X_test = X_test_tmp

        X_test['date'] = PRE_DATE_str
        print(PRE_DATE_str)
        return X_test

    for PRE_DATE_str in dates:
        flag = 0
        X = make_features(PRE_DATE_str)
        if flag == 0:
            X_temp = X
            flag = 1
        else:
            X_temp = X_temp.append(X)

    del df_time_item, promo_time_item, df_time_store_class, df_time_promo_store_class, df_time_store_class_index

    gc.collect()

    return X_temp


def predict(training_data, TODAY_DATE):
    def data_combination(flag, num, pre_record, data_hat, final, final_result):
        features_end_job = training_data['features_job']
        features_end_weekends = training_data['features_weekends']
        train_job = features_end_job[features_end_job['date'] < TODAY_DATE]
        train_weekends = features_end_weekends[features_end_weekends['date'] < TODAY_DATE]

        if num == 0:
            if flag == '-job':
                predict_day = features_end_job[features_end_job['date'] == pre_record]
                data_to_model = pd.concat([train_job, predict_day], axis=0).fillna(0.0)
                data_hat = {'-job': data_to_model, '-weekends': train_weekends}


            elif flag == '-weekends':
                predict_day = features_end_weekends[features_end_weekends['date'] == pre_record]
                data_to_model = pd.concat([train_weekends, predict_day], axis=0).fillna(0.0)
                data_hat = {'-job': train_job, '-weekends': data_to_model}


        else:
            final = pd.DataFrame(final)
            final.columns = ['date', 'storeid', 'goodsid', 'float_quantity', 'sale_qty']
            final_result = pd.DataFrame(final_result)
            final_result.columns = ['date', 'storeid', 'goodsid', 'float_quantity', 'sale_qty']
            ## insert week
            final_result.insert(5, 'week', final_result['date'])
            final_result['week'] = final_result['week'].map(
                lambda x: int(DT.datetime(int(x[:4]), int(x[5:7]), int(x[8:])).strftime("%w")))

            df_0 = final_result[final_result['week'] == 0]
            df_1 = final_result[final_result['week'] == 6]

            final_weekends = pd.concat([df_0, df_1], axis=0)

            final_job = final_result[0 < final_result['week']]
            final_job = final_job[final_job['week'] < 6]

            pre_record_1 = formatdate_a(pre_record, -1)

            print(pre_record, pre_record_1)
            WhichWeek = str(
                DT.datetime(int(pre_record_1[:4]), int(pre_record_1[5:7]), int(pre_record_1[8:])).strftime("%w"))

            if WhichWeek == '0' or WhichWeek == '6':
                flag_l = '-weekends'

                data_fresh_a = data_hat[flag_l]
                latest_day_a = data_fresh_a[data_fresh_a['date'] == pre_record_1]
                latest_day_b = final[['sale_qty']]
                latest_day_c = latest_day_a.drop(['sale_qty'], axis=1)
                latest_day_c.reset_index(inplace=True)
                latest_day_c.drop('index', axis=1, inplace=True)
                latest_day_c.insert(3, 'sale_qty', latest_day_b)

                data_fresh_a.to_csv('data_fresh_a' + flag_l + '_')

                df1 = data_fresh_a[data_fresh_a['date'] < pre_record_1]
                df2 = latest_day_c
                data_fresh_b = pd.concat([df1, df2], join_axes=[df1.columns])

                data_hat = {'-job': data_hat['-job'], '-weekends': data_fresh_b}


            else:
                flag_l = '-job'

                data_fresh_a = data_hat[flag_l]
                latest_day_a = data_fresh_a[data_fresh_a['date'] == pre_record_1]
                latest_day_b = final[['sale_qty']]
                latest_day_c = latest_day_a.drop(['sale_qty'], axis=1)
                latest_day_c.reset_index(inplace=True)
                latest_day_c.drop('index', axis=1, inplace=True)
                latest_day_c.insert(3, 'sale_qty', latest_day_b)
                data_fresh_a.to_csv('data_fresh_a' + flag_l + '_.csv')
                df1 = data_fresh_a[data_fresh_a['date'] < pre_record_1]
                df2 = latest_day_c
                data_fresh_b = pd.concat([df1, df2], join_axes=[df1.columns])
                data_fresh_b.to_csv('data_fresh' + flag_l + '_.csv')

                data_hat = {'-job': data_fresh_b, '-weekends': data_hat['-weekends']}

            ##fresh for predict
            if flag == '-job':
                predict_day = features_end_job[features_end_job['date'] == pre_record]

                num_job = len(set(final_job.date))
                print('numbers of job days:', num_job)
                col_job = ['before_volume_' + str(i).zfill(2) for i in range(0, before_job_days)]
                train_data = data_hat[flag]
                latest_date = sorted(set(train_data.date.astype(str)))[-1]
                la_train_data = train_data[train_data['date'] == latest_date]

                la_train_data_a = la_train_data[la_train_data.columns[3:3 + before_job_days]]
                la_train_data_a.columns = col_job[0:before_job_days]
                la_train_data_b = la_train_data[la_train_data.columns[1:3]]
                la_train_data_c = pd.concat([la_train_data_b, la_train_data_a], axis=1)
                predict_day = predict_day.drop(col_job[0:before_job_days], axis=1)
                predict_day = pd.merge(predict_day, la_train_data_c, how='left', on=['storeid', 'goodsid'])
                df1 = train_data
                data_to_model = pd.concat([df1, predict_day], join_axes=[df1.columns]).fillna(0.0)
                data_hat = {'-job': data_to_model, '-weekends': data_hat['-weekends']}


            elif flag == '-weekends':
                num_weekends = len(set(final_weekends.date))
                print('numbers_weekends:', num_weekends)
                predict_day = features_end_weekends[features_end_weekends['date'] == pre_record]
                ##T final
                col = ['before_volume_' + str(i).zfill(2) for i in range(0, before_weekends_days)]
                train_data = data_hat[flag]
                latest_date = sorted(set(train_data.date.astype(str)))[-1]
                la_train_data = train_data[train_data['date'] == latest_date]

                la_train_data_a = la_train_data[la_train_data.columns[3:3 + before_weekends_days]]
                la_train_data_a.columns = col[0:before_weekends_days]
                la_train_data_b = la_train_data[la_train_data.columns[1:3]]

                la_train_data_c = pd.concat([la_train_data_b, la_train_data_a], axis=1)

                predict_day = predict_day.drop(col[0:before_weekends_days], axis=1)
                predict_day = pd.merge(predict_day, la_train_data_c, how='left', on=['storeid', 'goodsid'])
                df1 = data_hat[flag]
                data_to_model = pd.concat([df1, predict_day], join_axes=[df1.columns]).fillna(0.0)
                data_hat = {'-job': data_hat['-job'], '-weekends': data_to_model}

        return data_to_model, data_hat

    def exc_parall():
        final_result = []
        num = 0
        for i in range(0, predict_days):
            print('PREDICT FOR THE PREDICTED DAY:', FT(TODAY_DATE, i))

            pre_record = FT(TODAY_DATE, i)
            WhichWeek = str(DT.datetime(int(pre_record[:4]), int(pre_record[5:7]), int(pre_record[8:])).strftime("%w"))
            if WhichWeek == '0':
                flag = '-weekends'
                LTD_limit = LTD_weekends
                before_days = before_weekends_days
            elif WhichWeek == '6':
                flag = '-weekends'

                LTD_limit = LTD_weekends
                before_days = before_weekends_days
            else:
                flag = '-job'
                LTD_limit = LTD_job
                before_days = before_job_days
            if num == 0:
                data, data_hat = data_combination(flag, num, pre_record, 0, 0, 0)
                final = model(data, pre_record, LTD_limit, TODAY_DATE)
                final_result += final

            else:
                data, data_hat = data_combination(flag, num, pre_record, data_hat, final, final_result)

                final = model(data, pre_record, LTD_limit, TODAY_DATE)
                final_result += final
            num = 1
        return final_result

    ### defination of inition data variance

    final_result = exc_parall()

    return final_result


def model(data, pre_record, LTD_limit, TODAY_DATE):
    print('#predict date:', pre_record)
    data = data.fillna(0.0)
    data = data.replace('NULL', 0.0)

    data = data.reset_index().drop('index', axis=1)

    feature_columns = data.columns[4:]
    feature_map = {}
    for i in range(0, len(feature_columns)):
        feature_map['f' + str(i)] = feature_columns[i]

    data = data[data.date.astype(str) >= FT(TODAY_DATE, -LTD_limit)]

    label_list = data.iloc[:, 3].astype('float64')
    feature_list = data.iloc[:, 4:].astype('float64')
    store_list = data.iloc[:, 2].astype(str)
    main_list = data.iloc[:, 1].astype(str)
    date_list = data.iloc[:, 0].astype(str)

    label_list = np.array(label_list).tolist()
    feature_list = np.array(feature_list).tolist()
    store_list = np.array(store_list).tolist()
    main_list = np.array(main_list).tolist()
    date_list = np.array(date_list).tolist()

    start = 0
    end = len(date_list) - 1
    # print end
    end_training = end  # 训练集结尾

    # end即预测集结尾
    # 寻找训练集结尾,该位置下一位是预测集开头
    index = end_training
    while (index > 0):
        date1 = DT.datetime.strptime((str(date_list[index]).split(' ')[0]), '%Y-%m-%d')
        date2 = DT.datetime.strptime((str(date_list[end_training]).split(' ')[0]), '%Y-%m-%d')
        if ((date2 - date1).days > 0):
            end_training = index
            break;
        index -= 1

    print('训练集开头：', start)
    print('训练集结尾：', end_training)
    print('预测集结尾', end)

    training_feature = feature_list[start + 1:end_training + 1]
    training_label = label_list[start + 1:end_training + 1]

    predict_main = main_list[end_training + 1:end + 1]
    predict_store = store_list[end_training + 1:end + 1]
    predict_date = pre_record
    print('#############:', predict_date)
    predict_feature = feature_list[end_training + 1:end + 1]

    ### model training

    if MODEL == 'GBDT':
        model = GradientBoostingRegressor(n_estimators=650, learning_rate=0.01, max_depth=8, random_state=0,
                                          loss='lad').fit(training_feature, training_label)
        ### model predict result
        predict_result = model.predict(predict_feature)
        ##savetheMODEL
        joblib.dump(model, pre_record + '#' + ".m")
    elif MODEL == 'RFR':
        params = {}
        params['n_estimators'] = 200
        params['n_jobs'] = 50
        clf = RandomForestRegressor(n_estimators=200, n_jobs=50)

        clf.fit(training_feature, training_label)
        joblib.dump(clf, pre_record + '#' + ".m")
        predict_result = clf.predict(predict_feature)
    elif MODEL == 'ETR':

        clf = ExtraTreeRegressor(max_depth=8, random_state=1)

        clf.fit(training_feature, training_label)

        joblib.dump(clf, pre_record + '#' + ".m")
        predict_result = clf.predict(predict_feature)

    elif MODEL == 'KNN':
        clf = neighbors.KNeighborsRegressor()
        clf.fit(training_feature, training_label)
        joblib.dump(clf, pre_record + '#' + ".m")
        predict_result = clf.predict(predict_feature)


    elif MODEL == 'XGBOOST':
        # 训练集构建
        print('XGboost 预测')
        label = training_label
        dtrain = xgb.DMatrix(training_feature, label)

        num_round = 2000
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.05
        params["max_depth"] = 17
        params["eval_metric"] = "rmse"
        params["silent"] = 1
        params['nthread'] = 20
        params['lambda'] = 1.0
        params['gamma'] = 0.9
        params['subsample'] = 0.9
        params['colsample_bytree'] = 0.9
        params['min_child_weight'] = 12
        plst = list(params.items())  # Using 5000 rows for early stopping.

        bst = xgb.train(plst, dtrain, num_round)

        # 预测集构建
        data_pre = xgb.DMatrix(predict_feature)

        # 结果预测
        predict_result = bst.predict(data_pre)
        joblib.dump(bst, pre_record + '#' + ".m")

        feature_scores = pd.Series(bst.get_fscore()).sort_values(ascending=False)
        feature_scores = pd.DataFrame(feature_scores).reset_index()
        feature_scores.columns = ['feature', 'feature_score']
        feature_scores['feature_name'] = feature_scores['feature'].map(lambda x: feature_map[x])

        feature_scores_normal = pd.Series(bst.get_fscore()).sort_values(ascending=False) / sum(
            pd.Series(bst.get_fscore()).sort_values(ascending=False))  # 重要性归一化
        feature_scores.to_csv(pre_record + '_feature_scores.csv')
        feature_scores_normal.to_csv(pre_record + '_feature_scores_normal.csv')

        # get result data back
        csv_result = []
        inside = []
        for index in range(0, len(predict_result)):
            inside.append(str(predict_date))
            inside.append(str(predict_store[index]))
            inside.append(str(predict_main[index]))
            inside.append((predict_result[index]))
            forecast = round(predict_result[index])

            inside.append(forecast)
            csv_result.append(inside)
            inside = []

    return csv_result


def sub_to_mysql(submission_df):
    """"

    """

    database_name = 'CHAOSHIFA_DB'
    table_name = 'model_predct_result_dev'
    columns_nm = ['time_create', 'date', 'storeid', 'goodsid', 'goodsname', 'pred_sale_volume', 'model_param']

    submission_df.columns = columns_nm

    def DataFrame_to_MySQL(dataframe, database_name, table_name):
        yconnect = create_engine('mysql+pymysql://sophon:IAmSophon_4523@47.97.17.51:3306/?charset=utf8')
        pd.io.sql.to_sql(dataframe, table_name, yconnect, schema=database_name, if_exists='append', index=False)

    DataFrame_to_MySQL(submission_df, database_name, table_name)


def sub_to_mysql(submission_df):
    """"

    """

    database_name = 'CHAOSHIFA_DB'
    table_name = 'model_predct_result_dev'
    columns_nm = ['time_create', 'date', 'storeid', 'goodsid', 'goodsname', 'pred_sale_volume', 'model_param']

    submission_df.columns = columns_nm

    def DataFrame_to_MySQL(dataframe, database_name, table_name):
        yconnect = create_engine('mysql+pymysql://sophon:IAmSophon_4523@47.97.17.51:3306/?charset=utf8')
        pd.io.sql.to_sql(dataframe, table_name, yconnect, schema=database_name, if_exists='append', index=False)

    DataFrame_to_MySQL(submission_df, database_name, table_name)


def main():
    data_sql = Get_Data_from_sql_online(TODAY_DATE, DATA_START_DAY)

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

    def AQI_LEVEL(x):
        """
        空气污染指数的取值范围定为0～500，其中0～50、51～100、101～200、201～300和大于300，
        分别对应国家空气质量标准中日均值的 I级、II级、III级、IV级和V级标准的污染物浓度限定数值
        """
        if 0.0 <= x <= 50.0:
            return 'I'
        elif 50.0 < x <= 100.0:
            return 'II'
        elif 100.0 < x <= 200.0:
            return 'III'
        elif 200.0 < x < 300.0:
            return 'IV'
        elif x > 300.0:
            return 'V'

    airq_weather_canlendar['AQI_level'] = airq_weather_canlendar['AQI'].apply(AQI_LEVEL)

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
    ## 补缺数据 2017-08-31
    df_test_temp_app = df_train_test[df_train_test.date == '2016-08-31'].dropna()
    df_test_temp_app['date'] = '2017-08-31'
    df_train_test = df_train_test.append(df_test_temp_app).sort_values(by='date')
    df_train_test = df_train_test.drop(['sale_qty_adj', 'x'], axis=1)

    ##门店特征的汇总
    le = LabelEncoder()
    stores = pd.DataFrame()
    stores['storeid'] = ['A035', 'A206']
    stores['city'] = ['Beijing', 'Beijing']
    stores['state'] = ['open', 'open']
    stores['type'] = ['Middle', 'small']
    stores = stores.set_index("storeid")

    stores['city'] = le.fit_transform(stores['city'].values)
    stores['state'] = le.fit_transform(stores['state'].values)
    stores['type'] = le.fit_transform(stores['type'].values)

    ## make goods info
    category_his = data_sql['category_his'].drop(['date'], axis=1)
    goods_his = data_sql['goods_his'].drop(['indate', 'date'], axis=1)

    ## 商品信息
    items_info = pd.merge(category_his, goods_his, on=['deptid'], how='right')
    items_info = items_info.set_index('goodsid')
    ##拷贝一份
    items_info_origin = items_info.copy()

    for col_name in items_info.columns:
        if items_info[[col_name]].dtypes[0] == 'object':
            items_info[col_name] = le.fit_transform(items_info[col_name].values)

    ##日期特征
    for col_name in airq_weather_canlendar.columns:
        if airq_weather_canlendar[[col_name]].dtypes[0] == 'object' and col_name != 'storeid':
            airq_weather_canlendar[col_name] = le.fit_transform(airq_weather_canlendar[col_name].values)

    condition = airq_weather_canlendar

    train_data = Get_Training_Data(df_train_test, condition, items_info)
    dates = list(set(df_train_test.date))[0:-7]
    features_time = pre_data(data_sql, items_info, stores, dates)

    train_data['features_weekends'] = pd.merge(features_time, train_data['features_weekends'],
                                               on=['date', 'storeid', 'goodsid'], how='inner')
    train_data['features_job'] = pd.merge(features_time, train_data['features_job'], on=['date', 'storeid', 'goodsid'],
                                          how='inner')
    train_data['features_weekends'].to_csv('fe.csv')
    submission = predict(train_data, TODAY_DATE)

    submission = pd.DataFrame(submission)
    submission.columns = ['date', 'goodsid', 'storeid', 'sale_qty', 'sale_int_qty']
    submission["pred_sale_volume"] = np.clip(np.expm1(submission["sale_qty"]), 0, 10000000)

    submission.insert(0, 'time_create', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    params = {}
    params['Model_name'] = 'XGBOOST'
    params['Prediction_days'] = PRE_DAYS
    params['MAX_ROUND'] = 5000
    json_str = json.dumps(params)

    submission.insert(1, 'model_param', params['Model_name'])
    submission['goodsid'] = submission['goodsid'].astype(str)

    submission_item = pd.merge(items_info_origin.reset_index()[['goodsid', 'name', 'spname']].astype(str),
                               submission, on=['goodsid'], how='right')
    submission_item['goodsname'] = submission_item['name'] + [':'] + submission_item['spname']

    submission_df = submission_item[
        ['time_create', 'date', 'storeid', 'goodsid', 'goodsname', 'pred_sale_volume', 'model_param']]

    submission_df.to_csv('submission_data.csv')

    print(submission_df.head())

    sub_to_mysql(submission_df)


main()
