# -*- encoding: utf-8 -*-
import datetime as DT

__tmp_file_list = []
__atexit_registered = set()


def T(FORMAT_DATE, i):
    return (DT.datetime.strptime(FORMAT_DATE, '%Y-%m-%d')
            + DT.timedelta(i)).strftime('%Y%m%d')


def FT(FORMAT_DATE, i):
    return (DT.datetime.strptime(FORMAT_DATE, '%Y-%m-%d')
            + DT.timedelta(i)).strftime('%Y-%m-%d')


def formatdate_a(d, i):
    return (DT.datetime.strptime(d, '%Y-%m-%d')
            + DT.timedelta(i)).strftime('%Y-%m-%d')


##工作路径
Work_Dir = '/Users/xingqiangchen/PycharmProjects/analysis-jobs/jobs/chaoshifa_forecast_system/data/'
Run_Temp = '/Users/xingqiangchen/PycharmProjects/analysis-jobs/jobs/chaoshifa_forecast_system/run_temp/'
Result_Dir = '/Users/xingqiangchen/PycharmProjects/analysis-jobs/jobs/chaoshifa_forecast_system/result/'
Model_Dir = '/Users/xingqiangchen/PycharmProjects/analysis-jobs/jobs/chaoshifa_forecast_system/Model_Saved/'

## 模型选择： string. RFR:Random Forest Tree. XGB: XGBOOST,ect.
MODEL = 'XGBOOST'
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
predict_days = 16
##是否考虑货品属性信息

SKU_INFO = False
