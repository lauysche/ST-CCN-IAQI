# coding:utf-8

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time, datetime

from sklearn.preprocessing import MinMaxScaler

import utils

scaler = MinMaxScaler(feature_range=(-1, 1))
def main(center_id,step):
    # extract station id list in Beijing
    #df_airq = pd.read_csv('./data/microsoft_urban_air_data/airquality.csv')
    #station_id_list = np.unique(df_airq['station_id'])[:36]     # first 36 stations are in Beijing  np.unique去重
    station_id_list=[x for x in range(1,10)]


    
    # Calculate the influence degree (defined as the Pearson correlation coefficient) between the center station and other stations
    r_thred = 0.906  #相关性系数阈值
    center_station_id = center_id  #目标站点ID
    station_id_related_list = []  #相关站点ID
    df_one_station = pd.read_csv('./data/station_data/Shanghai_200{}_fiveTimes.csv'.format(center_station_id))
    v_list_1 = list(df_one_station['PM25_AQI_value'])
    # v_list_1 = list(df_one_station['PM10_AQI_value'])  # 目标站点PM10浓度时间序列
    # v_list_1 = list(df_one_station['NO2_AQI_value'])
    for station_id_other in station_id_list:
        df_one_station_other = pd.read_csv('./data/station_data/Shanghai_200{}_fiveTimes.csv'.format(station_id_other), error_bad_lines=False)
        v_list_2 = list(df_one_station_other['PM25_AQI_value'])
        r, p = stats.pearsonr(v_list_1, v_list_2)
        if r > r_thred:
            station_id_related_list.append(station_id_other)
        #print('{}  {}  {:.3f}'.format(center_station_id, station_id_other, r))
        print('{:.3f}'.format(r))
    print(len(station_id_related_list), station_id_related_list)

    
    # generate x and y
    # x_shape: [example_count, num_releated, seq_step, feat_size]
    # y_shape: [example_count,]
    print('Center station: {}\nRelated stations: {}'.format(center_station_id, station_id_related_list))
    #feat_names = ['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration', 'CO_Concentration', 'O3_Concentration', 'SO2_Concentration','weather', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']
    feat_names = ['PM25_AQI_value', 'PM10_AQI_value', 'NO2_AQI_value', 'temperature', 'pressure', 'humidity', 'wind']
    x_length = 24
    y_length = 1
    y_step = step
    x = []
    y = []
    l=['PM25_AQI_value','PM10_AQI_value','NO2_AQI_value','temperature','pressure','humidity','wind','weather']
    location=1/5  #测试集在数据集中的位置
    #将输入数据归一化
    for id in station_id_related_list:
        #df_one_station = pd.read_csv('./data/station_data/df_station_{}_workdays_fiveTimes.csv'.format(id))
        df_one_station = pd.read_csv('./data/station_data/Shanghai_200{}_fiveTimes.csv'.format(id))
        df = pd.read_csv('./data/station_data/dfff.csv')
        for i in range(0,8):
            data = df_one_station.iloc[:,i+2]
            #l = data.tolist()
            #print(data.values.reshape(-1,1))
            #print(data.values)
            d = scaler.fit_transform(data.values.reshape(-1,1))
            #print([x[0] for x in d.tolist()])
            df[l[i]] = [x[0] for x in d.tolist()]
        df.to_csv(r"df_station_scaler_{}.csv".format(id), mode='a', index=False)
        '''
        #变换测试集在数据集中的位置，探究对预测效果的影响
        df_scaler = pd.read_csv('df_station_scaler_{}.csv'.format(id))
        size=len(df_scaler['PM25_Concentration'])
        df_scaler_test = df_scaler[feat_names].iloc[0:int(size*location)]
        dataframe = pd.concat([df_scaler,df_scaler_test], axis=0)
        dataframe = dataframe[feat_names].iloc[int(size*location):]
        dataframe = pd.DataFrame(dataframe)
        dataframe.to_csv('df_station_{}_workdays_fiveTimes_changeTestSetLocation.csv'.format(id), mode='a', index=False)
        '''
    for station_id in station_id_related_list:
        #df_one_station = pd.read_csv('./data/station_data/df_station_{}_workdays_fiveTimes.csv'.format(station_id))
        df_one_station = pd.read_csv('df_station_scaler_{}.csv'.format(station_id))
        #df_one_station = pd.read_csv('df_station_{}_workdays_fiveTimes_changeTestSetLocation.csv'.format(station_id))
        x_one = []
        for start_id in range(0, len(df_one_station)-x_length-y_length+1-y_step+1, y_length):
            x_data = np.array(df_one_station[feat_names].iloc[start_id: start_id+x_length])
            y_list = np.array(df_one_station['PM25_AQI_value'].iloc[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1])
            # y_list = np.array(df_one_station['PM10_AQI_value'].iloc[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1])
            # y_list = np.array(df_one_station['NO2_AQI_value'].iloc[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1])
            if np.isnan(x_data).any() or np.isnan(y_list).any():
                continue
            x_one.append(x_data)
            if station_id == center_station_id:
                y.append(np.mean(y_list))
        if len(x_one) <= 0:
            continue
        x_one = np.array(x_one)
        x.append(x_one)

        print('station_id: {}  x_shape: {}'.format(station_id, x_one.shape))


    x = np.array(x)
    x = x.transpose((1, 0, 2, 3))
    y = np.array(y)
    print('x_shape: {}  y_shape: {}'.format(x.shape, y.shape))
    
    # Save the four dimensional data as pickle file
    utils.save_pickle('./data/xy/x_{}.pkl'.format(center_station_id), x)
    utils.save_pickle('./data/xy/y_{}.pkl'.format(center_station_id), y)
    print('x_shape: {}\ny_shape: {}'.format(x.shape, y.shape))


if __name__ == '__main__':
    main(2,1)
