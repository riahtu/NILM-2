import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
import zz_CNN as zz
from keras.utils import plot_model
# 全局变量
batch_size = 128
nb_classes = 2
epochs = 100
# input image dimensions
img_rows, img_cols = 4, int(2*60)
# number of convolutional filters to use
nb_filters = 50
# size of pooling area for max pooling
pool_size = 2
# convolution kernel size
kernel_size = 2
# define directory
dir = '../data/self_made_data/'
currentApp = 'class7'

#prep data, no transfer learning
# XTrain, YTrain, XTest, YTest = zz.prep_data('../data/self_made_data/P_data.csv',
#                                                     '../data/self_made_data/I_data.csv',
#                                                     '../data/self_made_data/DP_data.csv',
#                                                     '../data/self_made_data/PF_data.csv',
#                                                     '../data/self_made_data/DI_data.csv',
#                                                     '../data/self_made_data/U_data.csv',
#                                                     '../data/self_made_data/target_data.csv', trainRate=0.8)
#prep data, transfer learning
XTrain, YTrain, XTest, YTest = zz.prep_data_trans(P=dir+currentApp+'/P_data.csv',
                                                    I=dir+currentApp+'/I_data.csv',
                                                    DP=-1,
                                                    PF=dir+currentApp+'/PF_data.csv',
                                                    DI=-1,
                                                    U=dir+currentApp+'/U_data.csv',
                                                    R=dir+currentApp+'/R_data.csv',
                                                    Target=dir+currentApp+'/target_data.csv', trainRate=0.8)


# 根据不同的backend定下不同的格式
XTrain = XTrain.reshape(XTrain.shape[0], img_rows, img_cols)
XTest = XTest.reshape(XTest.shape[0], img_rows, img_cols)
input_shape = (img_rows, img_cols)

XTrain = XTrain.astype('float32')
XTest = XTest.astype('float32')
# XTrain /= max(np.amax(XTrain), np.amax(XTest))
# XTest /= max(np.amax(XTrain), np.amax(XTest))
print('XTrain shape:', XTrain.shape)
print(XTrain.shape[0], 'train samples')
print(XTest.shape[0], 'test samples')

# 构建模型
model = Sequential()
"""
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same',
                        input_shape=input_shape))
"""
model.add(Convolution1D(100, kernel_size=kernel_size,
                        padding='same',
                        input_shape=input_shape))  # 卷积层1
model.add(Activation('relu'))  # 激活层
model.add(Convolution1D(nb_filters, kernel_size=kernel_size,
                        padding='same',
                        input_shape=input_shape))  # 卷积层1
model.add(Activation('relu'))  # 激活层
model.add(Convolution1D(nb_filters, kernel_size=kernel_size))  # 卷积层2
model.add(Activation('relu'))  # 激活层
model.add(MaxPooling1D(pool_size=pool_size))  # 池化层
model.add(Dropout(0.25))  # 神经元随机失活
model.add(Flatten())  # 拉成一维数据
model.add(Dense(200))  # 全连接层1
model.add(Activation('relu'))  # 激活层
model.add(Dropout(0.5))  # 随机失活
model.add(Dense(nb_classes))  # 全连接层2
model.add(Activation('softmax'))  # Softmax评分

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
# 训练模型
model.fit(XTrain, YTrain, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(XTest, YTest))


# 评估模型
score = model.evaluate(XTest, YTest, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
# model.save('model_trans_water_heater.h5')
model.save('model_trans'+currentApp+'.h5')