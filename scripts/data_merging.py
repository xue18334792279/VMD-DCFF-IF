import pandas as pd
import PIL.Image
import matplotlib.pyplot as plt
from pyts.preprocessing import MinMaxScaler
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_array
# from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence  # 稀疏矩阵中查找特征值/特征向量的函数

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyts.image import  GramianAngularField
from pyts.image import RecurrencePlot
from pyts.image import MarkovTransitionField
from pandas import Series

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
#
# 分割数据分为四个节点整理：（487,144（4x36））

def data1():
    path = 'D:/paper2/GRAM/data/4/x_train_all_windows.csv'
    df_in = pd.read_csv(path, sep=',', decimal='.',header=None)
    # print(df_in)
    df_in = np.asarray(df_in)
    x_train = []
    for i in range(487):
        a = df_in[i,:]
        b1 = df_in[i+487,:]
        # b2 = df_in[i+974,:]
        # b3 = df_in[i+1461,:]
        # b4 = df_in[i+1948, :]
        # b5 = df_in[i+2435, :]
        # b6 = df_in[i+2922, :]
        # b7 = df_in[i+3409, :]
        # b8 = df_in[i+3896, :]
        # b9 = df_in[i+4383, :]
        # b10 = df_in[i+4870, :]
        x = np.concatenate((a,b1),axis=0)
        # x = np.concatenate((x, b2), axis=0)
        # x = np.concatenate((x, b3), axis=0)
        # x = np.concatenate((x, b4), axis=0)
        # x = np.concatenate((x, b5), axis=0)
        # x = np.concatenate((x, b6), axis=0)
        # x = np.concatenate((x, b7), axis=0)
        # x = np.concatenate((x, b8), axis=0)
        # x = np.concatenate((x, b9), axis=0)
        # x = np.concatenate((x, b10), axis=0)
        x_train.append(x)
    for i in range(487):
        a = df_in[i, :]
        c = df_in[i + 974, :]
        x = np.concatenate((a, c), axis=0)
        x_train.append(x)
    for i in range(487):
        a = df_in[i, :]
        d = df_in[i + 1461, :]
        x = np.concatenate((a, d), axis=0)
        x_train.append(x)
    for i in range(487):
        a = df_in[i, :]
        e = df_in[i + 1948, :]
        x = np.concatenate((a, e), axis=0)
        x_train.append(x)
    for i in range(487):
        a = df_in[i, :]
        e = df_in[i + 2435, :]
        x = np.concatenate((a, e), axis=0)
        x_train.append(x)
    for i in range(487):
        a = df_in[i, :]
        e = df_in[i + 2922, :]
        x = np.concatenate((a, e), axis=0)
        x_train.append(x)
    for i in range(487):
        a = df_in[i, :]
        e = df_in[i + 3409, :]
        x = np.concatenate((a, e), axis=0)
        x_train.append(x)
    for i in range(487):
        a = df_in[i, :]
        e = df_in[i + 3896, :]
        x = np.concatenate((a, e), axis=0)
        x_train.append(x)
    for i in range(487):
        a = df_in[i, :]
        e = df_in[i + 4383, :]
        x = np.concatenate((a, e), axis=0)
        x_train.append(x)
    for i in range(487):
        a = df_in[i, :]
        e = df_in[i + 4870, :]
        x = np.concatenate((a, e), axis=0)
        x_train.append(x)
    x_train = np.asarray(x_train)
    print(x_train)
    return x_train

# 图像的4096数据（487,4096）
def data2():
    path = 'D:/paper2/GRAM/img/VMD/2/x_train1.csv'
    df_img1 = pd.read_csv(path, sep=',', decimal='.')
    df_img1 = np.asarray(df_img1)
    path = 'D:/paper2/GRAM/img/VMD/2/x_train.csv'
    df_img = pd.read_csv(path, sep=',', decimal='.')
    df_img = np.asarray(df_img)
    df_img = np.concatenate((df_img1,df_img),axis=0)
    print(df_img)
    return df_img
# 数据间相关系数边
def Edge(inp):
    def calc_corr(a, b):
        s1 = Series(a)
        s2 = Series(b)
        return s1.corr(s2)
    def create_graph(num_nodes, data):
        edge_index = [[], []]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                x, y = data[i, :], data[j, :]
                # print(x)
                # print(y)
                corr = calc_corr(x, y)
                # print(corr)
                if corr >= 0.4 or corr <= -0.4:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
        return edge_index
    x = np.reshape(inp,(4870,2,36))
    # print(x)
    edge = []
    for i in range(len(x)):
        edge_index = create_graph(2, x[i])
        edge.append(edge_index)
    # print(edge)
    return edge

def Graph(inp):
    gram = []
    def edgemat(edge, numnode):
        edge = np.asarray(edge)
        # print(edge.shape)
        link = []
        # print(edge[1])
        for i in range(len(edge[1])):
            a = edge[:, i]
            # print(a)
            link.append(a)
        # print(link)
        A = np.zeros((numnode, numnode))
        # print(A)
        for i, j in link:
            A[i, j] = 1
        b = A.T
        A = A + b
        A = A + sp.eye(A.shape[0])
        A = sp.coo_matrix(A,(2,2),dtype=np.int16)
        # print(A)
        d = sp.diags(np.power(np.array(A.sum(1)), -0.5).flatten(), 0)
        # tocsr()函数将矩阵转化为压缩稀疏行矩阵
        A = A.dot(d).transpose().dot(d).todense()
        # print(A)
        return A

    for i in range(4870):  # 487为数据长度
        A = edgemat(inp[i], 2)
        # print(A)
        gram.append(np.asarray(A).flatten())
    gram = np.asarray(gram)
    # print(gram)
    return gram


# gragh数据
x_train = data1()
print(x_train.shape)
# # img 数据
x_img = data2()
print(x_img.shape)
# # # 图矩阵
edge = Edge(x_train)
graph = Graph(edge)
print(graph.shape)
# # #
X = np.concatenate((x_img,x_train),axis=1)
print(X.shape)
X = np.concatenate((X,graph),axis=1)
print(X.shape)
X = X[:,1:]
print(X.shape)
print(X)
path = 'D:/paper2/GRAM/data/4/2/x_train_all_windows.csv'
np.savetxt(path,X,delimiter=',')
path = 'D:/paper2/GRAM/data/4/2/x_train_all_windows.npy'
np.save(path,X)

