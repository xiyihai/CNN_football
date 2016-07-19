#coding:utf-8

'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py

2016.06.06更新：
这份代码是keras开发初期写的，当时keras还没有现在这么流行，文档也还没那么丰富，所以我当时写了一些简单的教程。
现在keras的API也发生了一些的变化，建议及推荐直接上keras.io看更加详细的教程。

'''
#导入各种用到的模块组件
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
from data import load_data
import random
import numpy as np

from keras.models import model_from_json

np.random.seed(1024)  # for reproducibility

#加载数据
data, label = load_data()
#打乱数据
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
#输出图片的个数
print(data.shape[0], ' samples')

#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
label = np_utils.to_categorical(label, 6)

###############
#开始建立CNN模型
###############

#生成一个model
model = Sequential()

#第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
#border_mode可以是valid或者full，具体看这里说明：http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
#激活函数用tanh
#你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))

#4个卷积核，初始值不同，这样可以卷积出不同的特征
#需要修改卷积核个数4，大小5*5都需要改修  28*28
model.add(Convolution2D(10, 10, 10, border_mode='valid',input_shape=(3,320,180))) 
#model.add(Activation('relu'))
model.add(PReLU())
#随机丢弃数据集，使得结果不会过分拟合
model.add(Dropout(0.5))

#第二个卷积层，8个卷积核，每个卷积核大小3*3。4表示输入的特征图个数，等于上一层的卷积核个数
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)   311 171
model.add(Convolution2D(20, 10, 10, border_mode='valid'))
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.5))
#第三个卷积层，16个卷积核，每个卷积核大小3*3
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)  151 162
model.add(Convolution2D(40, 10, 10, border_mode='valid')) 
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

#71 77
model.add(Convolution2D(80, 10, 10, border_mode='valid')) 
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#31 34
model.add(Convolution2D(160, 10, 10, border_mode='valid')) 
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#11 12
model.add(Convolution2D(320, 5, 5, border_mode='valid')) 
model.add(PReLU())
model.add(Dropout(0.5))
#7 8

#全连接层，先将前一层输出的二维特征图flatten为一维的。
#Dense就是隐藏层。16就是上一层输出的特征图个数。4是根据每个卷积层计算出来的：(28-5+1)得到24,(24-3+1)/2得到11，(11-3+1)/2得到4
#全连接有128个神经元节点,初始化方式为normal

#对特征图进行分类，创建训练目标     
model.add(Flatten())
model.add(Dense(128, init='normal'))
model.add(PReLU())

#Softmax分类，输出是10类别
model.add(Dense(6, init='normal'))
model.add(Activation('softmax'))


#############
#开始训练模型
##############
#使用SGD + momentum
#model.compile里的参数loss就是损失函数(目标函数)


#SGD是一种训练方法，有很多种可选方法
#sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
#数据经过随机打乱shuffle=True。verbose=1，训练过程中输出的信息，0、1、2三种方式都可以，无关紧要。show_accuracy=True，训练时每一个epoch都输出accuracy。
#validation_split=0.2，将20%的数据作为验证集。

#batch_size是每一次抓取的图片个数，epoch是训练次数
model.fit(data, label, batch_size=210, nb_epoch=100,shuffle=True,verbose=1,validation_split=0.2)



#保存相关数据：weight + architecture
json_string = model.to_json()
open('architecture8.json', 'w').write(json_string)
model.save_weights('weights8.h5', overwrite=True)
