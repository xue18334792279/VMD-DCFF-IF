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
# 分割数据分为两个节点整理：（487,144（4x36））

def data1():
    path = 'D:/paper2/GRAM/data/4/x_test_all_windows.csv'
    df_in = pd.read_csv(path, sep=',', decimal='.',header=None)
    print(df_in)
    x_train1 = []
    x_train = np.asarray(df_in)
    a = x_train[0]
    b1 = x_train[1]
    b2 = x_train[2]
    b3 = x_train[3]
    b4 = x_train[4]
    b5 = x_train[5]
    b6 = x_train[6]
    b7 = x_train[7]
    b8 = x_train[8]
    b9 = x_train[9]
    b10 = x_train[10]

    b1 = np.concatenate((a, b1), axis=0)
    x_train1.append(b1)
    b2 = np.concatenate((a, b2), axis=0)
    x_train1.append(b2)
    b3 = np.concatenate((a, b3), axis=0)
    x_train1.append(b3)
    b4 = np.concatenate((a, b4), axis=0)
    x_train1.append(b4)
    b5 = np.concatenate((a, b5), axis=0)
    x_train1.append(b5)
    b6 = np.concatenate((a, b6), axis=0)
    x_train1.append(b6)
    b7 = np.concatenate((a, b7), axis=0)
    x_train1.append(b7)
    b8 = np.concatenate((a, b8), axis=0)
    x_train1.append(b8)
    b9 = np.concatenate((a, b9), axis=0)
    x_train1.append(b9)
    b10 = np.concatenate((a, b10), axis=0)
    x_train1.append(b10)
    # print(x_train1)
    # x_train = np.reshape(x_train,(10,72))
    x_train1 = np.asarray(x_train1)
    print(x_train1.shape)
    return x_train1

# 图像的4096数据（487,4096）
def data2():
    path = 'D:/paper2/GRAM/img/VMD/2/x_test1.csv'
    df_img1 = pd.read_csv(path, sep=',', decimal='.')
    df_img1 = np.asarray(df_img1)
    print(df_img1)
    path = 'D:/paper2/GRAM/img/VMD/2/x_test.csv'
    df_img = pd.read_csv(path, sep=',', decimal='.')
    print(df_img)
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
    x = np.reshape(inp,(10,2,36))
    edge = []
    for i in range(len(x)):
        edge_index = create_graph(2, x[i])
        edge.append(edge_index)
    print(edge)
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
        print(link)
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
        print(A)
        return A

    for i in range(10):  # 487为数据长度
        A = edgemat(inp[i], 2)
        print(A)
        gram.append(np.asarray(A).flatten())
    gram = np.asarray(gram)
    print(gram.shape)
    return gram

def data3():
    path = 'D:/paper2/GRAM/img/VMD/2/y_train1.csv'
    df_img1 = pd.read_csv(path, sep=',', decimal='.')
    df_img1 = np.asarray(df_img1)
    print(df_img1)
    path = 'D:/paper2/GRAM/img/VMD/2/y_train.csv'
    df_img = pd.read_csv(path, sep=',', decimal='.')
    print(df_img)
    df_img = np.asarray(df_img)
    df_img = np.concatenate((df_img1, df_img), axis=0)
    print(df_img)
    return df_img



# x_train = data1()
# x_img = data2()
# print(x_img.shape)
# edge = Edge(x_train)
# graph = Graph(edge)
# # # # # #
# X = np.concatenate((x_img,x_train),axis=1)
# print(X.shape)
# X = np.concatenate((X,graph),axis=1)
# print(X.shape)
# X = X[:,1:]
# print(X.shape)
# print(X)
# path = 'D:/paper2/GRAM/data/4/2/x_test_all_windows.csv'
# np.savetxt(path,X,delimiter=',')
# path = 'D:/paper2/GRAM/data/4/2/x_test_all_windows.npy'
# np.save(path,X)
y = data3()
y=y[:,1:]
print(y.shape)
path = 'D:/paper2/GRAM/img/VMD/2/y_train_sum.csv'
np.savetxt(path,y,delimiter=',')
path = 'D:/paper2/GRAM/img/VMD/2/y_train_sum.npy'
np.save(path,y)