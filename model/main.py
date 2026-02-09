from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import gc
from keras import backend as K
from axial_attention import AxialAttention
from pandas import Series

def identity_block(inp, filters, res):
    # First block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(inp)  # 过滤器个数、kernel_size、same：边缘0填充，输入的形状
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Second block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Third block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # If it is a residual block -> implement the identity shortcut
    if res:
        x_shortcut = inp
        x = tf.keras.layers.Add()([x, x_shortcut])
    x = tf.keras.layers.LeakyReLU()(x)

    return x

def convolutional_block(inp, filters, res):
    # First block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Second block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Third block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # If it is a residual block -> implement the convolutional shortcut to match dimensions
    if res:
        x_shortcut = inp
        x_shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(x_shortcut)
        x_shortcut = tf.keras.layers.BatchNormalization()(x_shortcut)
        x = tf.keras.layers.Add()([x, x_shortcut])
    x = tf.keras.layers.LeakyReLU()(x)

    return x

def SE(inp, f):
    f = int(f)
    b, h, w, c = tf.shape(inp)
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(inp)
    # x = tf.keras.layers.Reshape((b, 1, 1, f))(x)
    # x = tf.keras.layers.Reshape((1,f))(x)
    # x = tf.reshape(x,(1, 1, c))
    x = tf.keras.layers.Dense(f//8 , activation='relu')(x)
    x = tf.keras.layers.Dense(f, activation='relu')(x)
    x = tf.keras.layers.Reshape((1, 1, f))(x)
    x = tf.keras.layers.Multiply()([x, inp])
    # x = tf.transpose(x,perm=(0,2,3,1))
    return x

def Timeblock(inp,filters,kernel_size,stride):
    #卷积 输入为每个时间步中每个节点的输入特征数量。输出每个时间步长中每个节点所需的输出通道数。核大小一维时间内核的大小
    #:param inp: Input data of shape (batch_size, num_nodes, num_timesteps,num_features=in_channels)
    #:return: Output data of shape (batch_size, num_nodes, num_timesteps_out, num_features_out=out_channels)
    # Convert into NCHW format for pytorch to perform convolutions.

    REGULARIZER = tf.keras.regularizers.l2(l=0.0001)
    INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2., mode="fan_out", distribution="truncated_normal")

    x = tf.keras.layers.BatchNormalization(axis=1)(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(filters,kernel_size=[1,kernel_size[0]],strides=[1,stride],padding='same',kernel_initializer=INITIALIZER,data_format='channels_first',kernel_regularizer=REGULARIZER)(x)
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    print(x.shape)

    return x

def SGCN(inp,A,filters,kernel_size,stride):
    REGULARIZER = tf.keras.regularizers.l2(l=0.0001)
    INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.,mode="fan_out",distribution="truncated_normal")
    x = tf.keras.layers.Conv2D(filters*kernel_size[1],kernel_size=1,padding='same',kernel_initializer=INITIALIZER,data_format='channels_first',kernel_regularizer=REGULARIZER)(inp)
    print(x.shape)
    if filters ==64 and stride == 1:
        x = tf.keras.layers.Reshape((kernel_size[1], filters, 2, 18))(x)
    elif filters ==64 and stride == 2:
        x = tf.keras.layers.Reshape((kernel_size[1], filters, 2, 36))(x)
    elif filters == 128 and stride == 2:
        x = tf.keras.layers.Reshape((kernel_size[1], filters, 2, 18))(x)
    elif filters == 128 and stride == 1:
        x = tf.keras.layers.Reshape((kernel_size[1], filters, 2, 9))(x)
    else:
        x = tf.keras.layers.Reshape((kernel_size[1], filters, 2, 36))(x)
        print(x.shape)
    x = tf.einsum('nvw,nkcvt->ncvt', A, x) # A = (b,2,2),x=(b,2,64,2,36)=(1,2,2,2,2)
    # print(x)
    return x,A

def STGCNblock(inp,A,filters,kernel_size,stride,res):
    #输入：每个时间步长中每个节点的输入特征数量。
    #输出：在每个时间步长中每个节点所需的输出特征数。
    #:param X: Input data of shape (batch_size, num_nodes, num_timesteps,num_features=in_channels).
    #:param A_hat: Normalized adjacency matrix.
    #:return: Output data of shape (batch_size, num_nodes,num_timesteps_out, num_features=out_channels).

    def Residual(x,filters,stride,res):
        if not res:
            residual = 0
            # print(residual)
        elif res and stride==1:
            residual = x
            # print(residual.shape)
        else:
            residual = tf.keras.layers.Conv2D(filters, kernel_size=[1,1],strides=[1,stride],padding='same',data_format='channels_first')(x)
            residual = tf.keras.layers.BatchNormalization(axis=1)(residual)
            print(residual.shape)
        return residual

    residual = Residual(inp,filters,stride,res)
    # print(residual.s)
    # x = Timeblock(inp,filters,kernel_size,stride)
    # print(t.shape)
    x,A = SGCN(inp,A,filters,kernel_size,stride)
    x = Timeblock(x,filters,kernel_size,stride)
    # print(x.shape)

    x += residual
    x = tf.keras.layers.LeakyReLU()(x)
    # x = Timeblock(x,filters,kernel_size,stride)
    # x = tf.keras.layers.BatchNormalization()(x)
    return x

def build_convolutional_model(bottleneck, blocks, l_per_block, starting_filters, residual):
    # Parameters
    f = starting_filters
    kernel_size = [3, 3]  #3.3/2.2
    stride = 1
    # AE construction
    inp = tf.keras.layers.Input(shape=(4172)) # 4613/4172
    # print(inp.shape)
    encoder = inp[:,:4096]
    print(encoder.shape)
    decoder = inp[:,4096:-4] #-121/4
    print(decoder.shape)
    A = inp[:,-4:] #121/4
    print(A.shape)
    encoder = (tf.cast(encoder, tf.float32) / 255)  # 张量数据转换类型 张量→目标类型
    encoder = tf.keras.layers.Reshape((64,64,1))(encoder)
    # print(encoder.shape)
    decoder = tf.keras.layers.Reshape((2,36))(decoder) # decoder = (batchsize,numnode,timestep) 11/2
    A = tf.keras.layers.Reshape((2,2))(A) #11.11/2.2
    # print(decoder.shape)
    # print(A.shape)

    x = tf.keras.layers.BatchNormalization()(encoder)
    dec = tf.keras.layers.BatchNormalization()(decoder)

# Encoder
    for i in range(blocks):
        # Convolutional layers
        x = convolutional_block(x, f, residual)
        # print(x.shape)
        for j in range(l_per_block - 1):
            x = identity_block(x, f, residual)
            # print(x.shape)
        # Pooling layer
        x = tf.keras.layers.Conv2D(f, (2, 2), strides=(2, 2), padding='same')(x)

        # print(x.shape)
        f = f * 2

    # Bottleneck
    # x = SE(x, int(f/2))
    x = identity_block(x, int(f / 2), residual)
    x = SE(x, f/2)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(bottleneck)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # # BILSTM module--
    x = tf.keras.layers.Reshape((1, bottleneck))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(bottleneck,return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(bottleneck))(x)#
    x = tf.keras.layers.Dense(bottleneck, activation='relu', kernel_initializer='he_uniform')(x)
    # x = tf.keras.layers.Dense(12, activation='linear', kernel_initializer='he_uniform')(x)
    # print(x.shape)
    # x = tf.keras.layers.Reshape((1, bottleneck))(x)
    # print(x.shape) # x = (b,1,bottleneck)

# Decoder
    dec = tf.expand_dims(dec,1) # dec = (b,c,4,36)
    # A =  tf.Variable(A, dtype=tf.float32,trainable=False,name='tf.reshape/Reshape:0')
    out = STGCNblock(dec,A,32,kernel_size,stride,res=False) #64
    out = STGCNblock(out, A, 32, kernel_size, stride, res=True)
    out = STGCNblock(out, A, 64, kernel_size, stride=2, res=True) #128
    out = STGCNblock(out, A, 64, kernel_size, stride, res=True)
    out = STGCNblock(out, A, 128, kernel_size, stride=2, res=True)#256
    out = STGCNblock(out, A, 128, kernel_size, stride, res=True)
    # print(out.shape)
    out = tf.keras.layers.Reshape((out.shape[2], -1))(out)
    # print(out.shape)
    # out = tf.keras.layers.Dense()
    out = tf.keras.layers.Dense(bottleneck, activation='relu', kernel_initializer='he_uniform')(out)
    out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(bottleneck,return_sequences=True))(out)
    out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(bottleneck))(out)
    # print(out.shape)
    # out = tf.keras.layers.Flatten()(out)
    # out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(bottleneck,return_sequences=True))(out)#
    # out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(bottleneck))(out)#


    out = tf.keras.layers.Dense(bottleneck, activation='relu', kernel_initializer='he_uniform')(out)#
    # out = tf.keras.layers.Dense(12, activation='linear', kernel_initializer='he_uniform')(out)
    print(out.shape)
# add
    x = tf.concat([x, out], 1)
    x = tf.keras.layers.Reshape((2, bottleneck))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(bottleneck, return_sequences=True))(x)
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(bottleneck))(x) #
    print(x.shape)
    # x = tf.keras.layers.Dense(bottleneck, activation='relu', kernel_initializer='he_uniform')(x)#
    # x=tf.keras.layers.Reshape((1,2048))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(bottleneck, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(bottleneck))(x)
    x = tf.keras.layers.Dense(bottleneck, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(12, activation='linear', kernel_initializer='he_uniform')(x)
#     a = tf.reduce_mean(out, 1)
#     out1 = out[:,1:]
#     out1 = tf.reduce_sum(out1,axis=1)
#     print(out1.shape)
#     a = out[:,0]
#     # print(a)
#     # x = tf.concat([x,out1],1)
#     x = tf.concat([x, out], 1)
#     x = tf.keras.layers.Reshape((2,12))(x)
# # # #
#     x = tf.reduce_mean(x,1)
#     print(x.shape)
    # x = tf.keras.layers.GRU(bottleneck, return_sequences='relu')(x)
    # x = tf.keras.layers.GRU(bottleneck, return_sequences=False)(x)
#     # x = tf.keras.layers.LSTM(bottleneck)(out)
#     # # print(x.shape)
#     x= tf.keras.layers.Dense(bottleneck, activation='relu', kernel_initializer='he_uniform')(x)

    return tf.keras.models.Model(inp, x)


def sample_generator(x_path, y_path, validation_split, batch_s):
    def process_images(x_im, y_h):
        # x_im = (tf.cast(x_im, tf.float32) / 255)  # 张量数据转换类型 张量→目标类型
        # x_im = tf.reshape(x_im, (64, 64, 1))
        return x_im, y_h

    # Load data
    x = np.load(x_path)  # 文件读写
    print(x.shape)
    y = np.load(y_path)
    print(y.shape)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split)  # 数据分测试集训练集
    print(x_train.shape)

    train_length = len(x_train)
    val_length = len(x_val)

    # Training samples
    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # 特征切片，将数据中的特征和标签组合成一个tuple
    print(data_train)
    data_train = data_train.shuffle(buffer_size=train_length)  # 打乱数据，程序会维持一个buffer_size大小的缓存，每次都会随机在这个缓存区抽取一定数量的数据
    data_train = data_train.map(process_images)  # 将序列中的每一个元素，输入函数，最后将映射后的每个值返回合并
    data_train = data_train.repeat()  # 将数据重复使用多少epoch
    data_train = data_train.batch(batch_size=batch_s)  # 将数据打包成batch_size
    data_train = data_train.prefetch(buffer_size=tf.data.AUTOTUNE)  # 数据预读取，提升IO性能

    # Validation samples
    data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    data_val = data_val.shuffle(buffer_size=val_length)
    data_val = data_val.map(process_images)
    data_val = data_val.repeat()
    data_val = data_val.batch(batch_size=batch_s)
    data_val = data_val.prefetch(buffer_size=tf.data.AUTOTUNE)

    return data_train, data_val, train_length, val_length


def train_models(cfg):
    number_of_models = 1  # 50

    # Unpack configuration
    bottleneck = cfg['bottleneck']
    num_blocks = cfg['number_blocks']
    num_layers = cfg['number_layers']
    batch_size = cfg['batch_size']
    val_split = cfg['validation_split']
    residual = cfg['residual']
    str_filters = cfg['starting_filters']
    x_path = cfg['x_path']
    y_path = cfg['y_path']
    # Build & Train the Forecasting models
    for i in range(number_of_models):
        print('Train Forecasting model:', (i + 1))

        train_set, val_set, train_length, val_length = sample_generator(x_path, y_path, val_split, batch_size)
        train_steps = train_length // batch_size + 1
        val_steps = val_length // batch_size + 1

        kmodel = build_convolutional_model(bottleneck, num_blocks, num_layers, str_filters, residual)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=80,decay_rate=0.9)
        opt = tf.keras.optimizers.Adam(lr=0.00001) # Adam(lr=0.001,amsgrad=True)/RMSprop(lr=0.001)

        # mode='min'：min模式是在监控指标值不再下降时停止训练；
        # patience：在监控指标没有提升的情况下，epochs 等待轮数。等待大于该值监控指标始终没有提升，则提前停止训练；
        # verbose: log输出方式
        # min_delta: 认为监控指标有提升的最小提升值。如果变化值小于该值，则认为监控指标没有提升；
        # restore_best_weights: 是否加载训练过程中保存的最优模型权重，如果为False，则使用在训练的最后一步获得的模型权重
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20,
                                              min_delta=0.0001, restore_best_weights=True)  # 避免过拟合，节省训练时间
        # 为搭建好的神经网络模型设置损失函数loss、优化器optimizer、准确性评价函数metrics。
        kmodel.compile(optimizer=opt, loss='mae')
        # 记录了loss和其他指标的数值随epoch变化的情况
        kmodel.fit(train_set, epochs=300, validation_data=val_set, verbose=1, callbacks=[es],
                   steps_per_epoch=train_steps, validation_steps=val_steps)
        # ts_path = 'D:/paper2/GRAM/img/univariate/x_test_all_windows_line5.npy'
        # df_x_test = np.load(ts_path)
        # y_hat_single = kmodel.predict(df_x_test)
        # print(y_hat_single)

        # path = 'D:/paper2/t/gcnCNN' + str(i) + '.h5'

        path = 'D:/GCN/8/gcnCNN'+ str(i)+'.h5'
        kmodel.save_weights(path)

        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()

def generate_forecasts(path_test, save_model_p, save_forecasts,cfg):
    number_of_models = 50  # 50
    bottleneck = cfg['bottleneck']
    num_blocks = cfg['number_blocks']
    num_layers = cfg['number_layers']
    batch_size = cfg['batch_size']
    val_split = cfg['validation_split']
    residual = cfg['residual']
    str_filters = cfg['starting_filters']
    # Read test data
    path = 'D:/paper2/GRAM/img/VMD/1/VMD.csv'  # 训练数据
    df_in = pd.read_csv(path, sep=',', decimal='.', header=None)
    print(df_in.shape)

    df_x_test = np.load(path_test)
    print(df_x_test.shape)
    # df_x_test = df_x_test[0]
    # df_x_test = df_x_test.astype(float) / 255
    # df_x_test = df_x_test.reshape((len(df_x_test),4256))
    # print(df_x_test.shape)

    # Load forecasting models & ensemble
    y_hat_all = list([])
    for i in range(number_of_models):
        print('Forecasting with model:', (i + 1))
        path = save_model_p + str(i) + '.h5'
        model = build_convolutional_model(bottleneck, num_blocks, num_layers, str_filters, residual)
        model.load_weights(path)

# 确保模型处于评估模式
#         model.compile(optimizer='adam', loss='mae')(path)
        y_hat_single = model.predict(df_x_test)
        # print(y_hat_single)
        y_hat_all.append(y_hat_single)
        y_hat_single = np.asarray(y_hat_single)

        fh = 12  # 18
        array_frc = np.array([]).reshape((0, fh))  # 0 保持原通道数不变
        for j in range(len(df_in)):
            ts_in = np.asarray(df_in.iloc[j, 1:].dropna())

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(np.reshape(ts_in[-36:], (-1, 1)))  # 测试数据最后36个进行按行展开 #36

            y_hat_1 = y_hat_single[j, :]
            y_hat_1 = scaler.inverse_transform(np.reshape(y_hat_1, (-1, 1))).reshape(-1)
            array_frc = np.vstack((array_frc, np.reshape(y_hat_1, (1, fh))))  # 按行堆叠数组
            # print(array_frc)

        if save_forecasts:
            path = 'D:/paper2/t/forecast/frc_forcnn' + str(i) + '.csv'
            # path = 'D:/paper2/GRAM/forecast/8/frc_forcnn' + str(i) + '.csv'
            np.savetxt(path, array_frc, delimiter=',')

    y_hat_all = np.asarray(y_hat_all)
    y_hat_all = np.median(y_hat_all, axis=0)  # 均值
    #          Process forecasts (scale back to original level)

    fh = 12  # 修改
    array_frc = np.array([]).reshape((0, fh))
    for i in range(len(df_in)):
        ts_in = np.asarray(df_in.iloc[i, 1:].dropna())

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.reshape(ts_in[-36:], (-1, 1)))  # 修改

        y_hat = y_hat_all[i, :]
        y_hat = scaler.inverse_transform(np.reshape(y_hat, (-1, 1))).reshape(-1)
        array_frc = np.vstack((array_frc, np.reshape(y_hat, (1, fh))))

    if save_forecasts:
        path = 'D:/paper2/t/forecast/frc_forcnn.csv'  # 预测数据存储地址
        np.savetxt(path, array_frc, delimiter=',')

configuration = {
    'bottleneck': 1024,  # 1024
    'number_blocks': 5,  # 5
    'number_layers': 3,  # 3
    'batch_size': 128,  # 64
    'validation_split': 0.2,  # 0.2
    'residual': True,
    'starting_filters': 8,  # 8
    'x_path': 'D:/paper2/GRAM/data/3/1/x_train_all_windows.npy',  # 数据图片处理train的x位置
    'y_path': 'D:/paper2/GRAM/img/VMD/1/y_train_all_windows_line5.npy'
}
# train_models(configuration)

ts_path = 'D:/paper2/GRAM/data/3/1/x_test_all_windows.npy'
md_path = 'D:/GCN/8/gcnCNN'
generate_forecasts(ts_path, md_path, True,configuration)

