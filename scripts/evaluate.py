from tqdm import tqdm
import pandas as pd
import numpy as np


# 该脚本用于评估模型的性能。它计算论文中报告的平均 MASE，以及中位 MASE、平均 sMAPE 和中位 sMAPE。

def smape(a, b):
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(200 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()


def mase(insample, y_test, y_hat_test, freq):
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])
    masep = np.mean(abs(insample[freq:] - y_hat_naive))
    return np.mean(abs(y_test - y_hat_test)) / masep


def calculate_error_metrics(frc_path, out_path, save_errors):
    # Read test data
    path = 'D:/paper2/data/train/allpeople_VMD.csv'  # 训练集
    df_insample = pd.read_csv(path, sep=',', decimal='.',header=None)
    path = 'D:/paper2/data/test/allpeople_test.csv'  # 测试集
    df_outsample = pd.read_csv(path, sep=',', decimal='.')

    # Load forecasts
    df_frc = pd.read_csv(frc_path, sep=',', decimal='.', header=None)
    frequency = 1
    errors_sm_m4 = []
    errors_ma_m4 = []
    errors_out = []
    for i in tqdm(range(len(df_frc))):
        # Read insample
        ts_insample = np.asarray(df_insample.iloc[i, 1:].dropna())  # 在将丢失值删除后，将数据切片索引，转换格式

        # Read outsample
        ts_outsample = np.asarray(df_outsample.iloc[i, 1:].dropna())

        # Get forecasts
        y_hat = np.asarray(df_frc.iloc[i,:])
        print(y_hat)

        # Calculate errors
        sm = smape(y_hat, ts_outsample)
        ma = mase(ts_insample, ts_outsample, y_hat, frequency)

        errors_sm_m4.append(sm)
        errors_ma_m4.append(ma)
        errors_out.append([(i + 1), sm, ma])

    if save_errors:
        errors_out = np.asarray(errors_out)
        np.savetxt(out_path, errors_out)

    print('Path:', frc_path)
    print('Mean sMAPE: ', np.round(np.mean(errors_sm_m4), 3))
    print('Median sMAPE: ', np.round(np.median(errors_sm_m4), 3))
    print('Mean MASE: ', np.round(np.mean(errors_ma_m4), 3))
    print('Median MASE: ', np.round(np.median(errors_ma_m4), 3))

file_num = 50
for i in range(file_num):
    print('模型：', i)
    # forecasts_path = 'D:/paper2/t/forecast/2/frc_forcnn' + str(i) + '.csv'  # forecast/6
    forecasts_path = 'D:/paper2/GRAM/forecast/8/1/frc_forcnn'+str(i)+'.csv' #forecast/6
    errors_path = 'D:/paper2/GRAM/error/2/'+str(i)+'.csv'
    # test_path = 'D:/paper2/data/origin_data_2019-test.csv'
    calculate_error_metrics(forecasts_path, errors_path, True)
# print('平均模型：')
# forecasts_path = 'D:/paper2/t/forecast/1/frc_forcnn.csv'  # forecast/6
# errors_path = 'D:/paper2/t/errors_forcnn_sd.csv'
# forecasts_path = 'D:/paper2/forecast/10/frc_forcnn.csv'
# errors_path = 'D:/paper2/error/10/errors_forcnn_sd.csv'
calculate_error_metrics(forecasts_path, errors_path, True)
