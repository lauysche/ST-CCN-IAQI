# coding:utf-8

import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import data_process

import models
import utils
import config as cfg
import data.station_data.inverse_data as inverse


def eval(net, x_test, y_test, plot=False):
    x_valid = x_test
    y_valid = y_test
    #print(y_valid)
    print('\nStart evaluating...\n')
    net.eval()
    rmse_train_list = []
    rmse_valid_list = []
    mae_valid_list = []
    y_valid_pred_final = []
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    h_state = None
    y_valid_pred_final = []
    rmse_valid = 0.0
    cnt = 0
    for start in range(len(x_valid) - cfg.batch_size + 1):
        x_input_valid = torch.tensor(x_valid[start:start + cfg.batch_size], dtype=torch.float32)
        y_true_valid = torch.tensor(y_valid[start:start + cfg.batch_size], dtype=torch.float32)
        if cfg.model_name == 'RNN' or cfg.model_name == 'GRU':
            y_valid_pred, _h_state = net(x_input_valid, h_state)
        else:
            y_valid_pred = net(x_input_valid)
            #print(y_valid_pred.data.numpy()[0][0])
            '''
            if y_valid_pred.data.numpy()[0][0] > 1:
                y_valid_pred.data.numpy()[0][0]=np.random.rand(1) * 0.1 + 0.9
            if y_valid_pred.data.numpy()[0][0] < -1:
                y_valid_pred.data.numpy()[0][0]=np.random.rand(1) * (-0.1) + (-0.9)
            '''
        y_valid_pred_final.extend(y_valid_pred.data.numpy())
        loss_valid = criterion(y_valid_pred, y_true_valid).data
        mse_valid_batch = loss_valid.numpy()
        rmse_valid_batch = np.sqrt(mse_valid_batch)
        rmse_valid += mse_valid_batch
        cnt += 1
    y_valid_pred_final = np.array(y_valid_pred_final).reshape((-1, 1))
    rmse_valid = np.sqrt(rmse_valid / cnt)
    mae_valid = metrics.mean_absolute_error(y_valid, y_valid_pred_final)
    r2_valid = metrics.r2_score(y_valid, y_valid_pred_final)
    MSE = metrics.mean_squared_error(y_valid, y_valid_pred_final)


    #y_valid =data_process.scaler.inverse_transform(y_valid)
    #y_valid_pred_final = data_process.scaler.inverse_transform(y_valid_pred_final)
    '''
    print(y_valid_pred_final)
    print('\nMSE_valid: {:.4f}  RMSE_valid: {:.4f}  MAE_valid: {:.4f}  R2_valid: {:.4f}\n'.format(MSE, rmse_valid, mae_valid, r2_valid))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(y_valid,color='blue',label='truth')
    plt.plot(y_valid_pred_final, color='pink', label='prediction')
    plt.title('阈值是0.82时模型的预测表现')
    plt.xlabel('hours')
    plt.ylabel('PM25_Concentration')
    plt.legend()
    plt.show()
    '''
    # 将min_max归一化后的数据，反归一化
    print(y_valid)
    print(y_valid_pred_final)  # 二维列表
    data = pd.read_csv('./data/station_data/Shanghai_2002_fiveTimes.csv')
    size = len(y_valid)
    data = data['PM25_AQI_value']
    l_pred, l_test = inverse.inverse_MinMaxScaler(data, y_valid_pred_final, y_valid)
    # l_pred, l_test = inverse.inverse_StandardScaler(data, y_valid_pred_final, y_valid)
    print("l_test :{}".format(l_test))

    print("l_pred :{}".format(l_pred))

    MSE=mean_squared_error(l_test, l_pred)
    RMSE=mean_squared_error(l_test, l_pred) ** 0.5
    MAE=mean_absolute_error(l_test, l_pred)
    R2=r2_score(l_test, l_pred)
    print("MSE valuse is {}".format(mean_squared_error(l_test, l_pred)))
    print("RMSE valuse is {}".format(mean_squared_error(l_test, l_pred) ** 0.5))
    print("MAE valuse is {}".format(mean_absolute_error(l_test, l_pred)))
    print("R2 valuse is {}".format(r2_score(l_test, l_pred)))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    l1=l_test
    l2=l_pred
    plt.plot(l1, color='blue', label='test')
    plt.plot(l2, color='pink', label='prediction')
    plt.title('对未来1个时刻的预测结果')
    plt.xlabel('hours')
    plt.ylabel('PM2.5AQI（μg/m3）')
    # plt.ylabel('PM10AQI（μg/m3）')
    # plt.ylabel('NO2AQI（μg/m3）')
    plt.legend()
    plt.show()
    print(l_pred)
    print(l_test)

    #return rmse_valid, mae_valid, r2_valid
    return l_pred,l_test,MSE,RMSE,MAE,R2


def main(step):
    # Hyper Parameters
    cfg.print_params()
    np.random.seed(cfg.rand_seed)
    torch.manual_seed(cfg.rand_seed)

    # Load data
    print('\nLoading data...\n')
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_data(f_x=cfg.f_x, f_y=cfg.f_y,step=step)

    # Generate model
    net = None
    if cfg.model_name == 'RNN':
        net = models.SimpleRNN(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'GRU':
        net = models.SimpleGRU(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'LSTM':
        net = models.SimpleLSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'TCN':
        net = models.TCN(input_size=cfg.input_size, output_size=cfg.output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'STCN':
        net = models.STCN(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                          num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, net))

    # Load model parameters
    net.load_state_dict(torch.load(cfg.model_save_pth))
    print(utils.get_param_number(net=net))

    # Evaluation
    lpred,ltest,MSE,RMSE,MAE,R2=eval(net, x_test, y_test)
    print('结束')



    return lpred,ltest,MSE,RMSE,MAE,R2


if __name__ == '__main__':
    main(1)
