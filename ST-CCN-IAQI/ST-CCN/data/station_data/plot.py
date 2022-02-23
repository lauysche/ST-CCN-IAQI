import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator

data = pd.read_csv('shanghai_2001.csv')
start=0
end=24*7

#寻找离散点
def find_anomalies(df):
    anomalies = []
    data_std = df.std()
    data_mean = df.mean()
    anomaly_cut_off = data_std * 3

    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
    for outlier in df[:]:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies

'''
for i in range(1,8):
    df = data['PM25_Concentration'].iloc[start:end]


    start+=24*7
    end+=24*7
    
    X=[x for x in range(1,169)]
    plt.plot(X,df,color='blue',label='dataset')
    x_major_locator = MultipleLocator(24)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlabel('hours')
    plt.ylabel('PM25_Concentration')
    plt.grid()
    plt.legend()
    plt.show()
    print(df.describe())
    
    
    print("第{}周的离群点为：{}".format(i,find_anomalies(df)))
'''

#线性插值
def insert_point(a,b):
    x1,y1=a
    x2,y2=b
    dy = y2-y1
    dx = x2-x1
    k=dy/dx
    b=y1-k*x1
    start=x1+1
    end=x2
    return k,b,start,end

x1=(1,16.080674)
y1=(3,33.31968257)
x2=(1,37.28696304)
y2=(5,29.16296014)
x3=(1,16.34775135)
y3=(7,26.43980026)
x4=(1,20.77934277)
y4=(4,19.91998863)
x5=(1,35.15483212)
y5=(3,30.68553841)
k,b,begin,stop = insert_point(x1,y1)
k2,b2,begin2,stop2 = insert_point(x2,y2)
k3,b3,begin3,stop3 = insert_point(x3,y3)
k4,b4,begin4,stop4 = insert_point(x4,y4)
k5,b5,begin5,stop5 = insert_point(x5,y5)
k_list=[]
k_list.append(k)
k_list.append(k2)
k_list.append(k3)
k_list.append(k4)
k_list.append(k5)
b_list=[]
b_list.append(b)
b_list.append(b2)
b_list.append(b3)
b_list.append(b4)
b_list.append(b5)
begin_list=[]
begin_list.append(begin)
begin_list.append(begin2)
begin_list.append(begin3)
begin_list.append(begin4)
begin_list.append(begin5)

stop_list=[]
stop_list.append(stop)
stop_list.append(stop2)
stop_list.append(stop3)
stop_list.append(stop4)
stop_list.append(stop5)

for i in range(len(k_list)):
    l = []
    for step in range(begin_list[i],stop_list[i]):
        l.append(float('%.3f'%(k_list[i]*step+b_list[i])))
    print(l)




