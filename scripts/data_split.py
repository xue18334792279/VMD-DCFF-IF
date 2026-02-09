import pandas as pd
import PIL.Image
import matplotlib.pyplot as plt
from pyts.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyts.image import  GramianAngularField
from pyts.image import RecurrencePlot
from pyts.image import MarkovTransitionField

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


def data_cut_all_windows_nofill():
    input_size = 36
    forecasting_horizon = 12 #18

    # 读取文件数据
    path = 'D:/paper2/VMD/10/VMD.csv'
    df_in = pd.read_csv(path, sep=',', decimal='.',header=None)
    print(df_in.shape)


    # Initialize dataframes
    df_x_train = list([])
    df_y_train = list([])
    df_x_test = list([])

    for i in range(len(df_in)): #行数
        print(len(df_in))
        print('Series:', (i+1))

        # ==================================
        # ============= X_TEST =============
        # ==================================
        # 读取序列
        ts = np.asarray(df_in.iloc[i, :].dropna())

        ts = ts[-input_size:]
        if len(ts) < input_size:
            filler = np.full(shape=(input_size - len(ts)), fill_value=ts[0])
            ts = np.hstack((filler, ts)) #水平平铺

        # ts = np.reshape(ts, (-1, 1)).reshape(-1)
        # Scaling in interval [0,1]
        scaler = MinMaxScaler()
        ts = scaler.fit_transform(np.reshape(ts, (-1, 1))).reshape((-1))

        df_x_test.append(np.asarray(ts))

        # =============================================
        # ============= X_TRAIN & Y_TRAIN =============
        # =============================================
        ts = np.asarray(df_in.iloc[i, :].dropna())
        while len(ts) >= input_size + forecasting_horizon:
            # 处理 x_train & y_train
            y_train = ts[-forecasting_horizon:]
            x_train = ts[:-forecasting_horizon]
            x_train = x_train[-input_size:]

            # Scaling in interval [0,1]
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(np.reshape(x_train, (-1, 1))).reshape((-1))
            y_train = scaler.transform(np.reshape(y_train, (-1, 1))).reshape(-1)

            # Check for cases where the y_train scaling explodes
            # if (5 * np.sum(x_train[-forecasting_horizon:])) > np.sum(abs(y_train)):  # abs绝对值
            df_y_train.append(y_train)

            df_x_train.append(np.asarray(x_train))
            # Trim the series
            ts = ts[:-1]

    df_x_train = np.asarray(df_x_train)
    df_y_train = np.asarray(df_y_train)
    df_x_test = np.asarray(df_x_test)

    # 存储图片数据集
    path = 'D:/paper2/GRAM/data/4/'
    np.savetxt(path + 'x_train_all_windows.csv', df_x_train,delimiter=',')
    print('x_train:', df_x_train.shape)
    np.savetxt(path + 'x_test_all_windows.csv', df_x_test,delimiter=',')
    print('x_test:', df_x_test.shape)
    np.savetxt(path + 'y_train_all_windows.csv', df_y_train,delimiter=',')
    print('y_train:', df_y_train.shape)

    # 分割数据后保存各自的csv文件
    # npfile = np.load(r'D:/paper2/image/GASF/16/y_train_all_windows_line5.npy')
    # np_to_csv = pd.DataFrame(data=npfile)
    # np_to_csv.to_csv('D:/paper2/image/GASF/16/y_train.csv')
    # npfile = np.load(r'D:/paper2/image/GASF/16/x_train_all_windows_line5.npy')
    # np_to_csv = pd.DataFrame(data=npfile)
    # np_to_csv.to_csv('D:/paper2/image/GASF/16/x_train.csv')
    # npfile = np.load(r'D:/paper2/image/GASF/16/x_test_all_windows_line5.npy')
    # np_to_csv = pd.DataFrame(data=npfile)
    # np_to_csv.to_csv('D:/paper2/image/GASF/16/x_test.csv')

data_cut_all_windows_nofill()