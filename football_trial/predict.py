#coding:utf-8
import theano, os
import numpy as np

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2
from keras.constraints import maxnorm
from keras.models import model_from_json

#定义加载文件路径
folder = "./Test8"
architecture = 'architecture8.json'
weights = 'weights8.h5'

#load the saved model
model = model_from_json(open(architecture).read())
model.load_weights(weights)

#加载训练方法
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#define theano funtion to get output of  FC layer
#get_activations = theano.function([model.layers[0].input], model.layers[0].output(), allow_input_downcast=True)

#define theano funtion to get output of  first Conv layer 
#get_featuremap = theano.function([model.layers[0].input],model.layers[2].output(train=False),allow_input_downcast=False) 
#data, label = load_tdata()

#visualize feature  of  Fully Connected layer
#data[0:10] contains 10 images
#feature = get_feature(data)  #visualize these images's FC-layer feature
#cPickle.dump(feature, open("./feature.pkl","wb"))

#featuremap = get_featuremap(data[0:10])
#cPickle.dump(featuremap, open("./featuremap.pkl","wb"))

#定义一个读取Test文件下图片的函数
def scan_files(directory,prefix=None,postfix=".jpg"):
	files_list=[]
	#返回三元组(根路径，所有子目录名字，所有文件名字)
	for root, sub_dirs, files in os.walk(directory):
		for special_file in files:
			if postfix:
				if special_file.endswith(postfix):
					#所有图片路径加入到数组中去
					files_list.append(os.path.join(root,special_file))
	return files_list

#加载数据
def load_tdata():
	files = scan_files(folder, postfix=".jpg")
	num = len(files)

	#定义一个4维空数组data放置图片数据，dtype是元素类型
	data = np.empty((num,3,320,180),dtype="float32")

	for i in range(num):
		#输入文件路径(包含文件名)
		print(files[i])
		#加载图片对象
		img = Image.open(files[i])
		#asarray将img对象转换成3维数组(像素点3个)表示，格式是float32
		arr = np.asarray(img,dtype="float32")
		arr = np.transpose(arr)
		#填充之前定义的4维数组，代表图片
		data[i,:,:,:] = arr
	return data, files

#test samples
tdata, files = load_tdata()
print "Number of samples: %d"%(tdata.shape[0])
#result是个数组，预测多少个图片就会有多少个result,每一个result中预测值有多个，分别有不同的权值(权值较大的是最有可能的)
result = model.predict_proba(tdata, batch_size=210, verbose=1)
count = 0
for i in result:
	print "Filename:%s =======> Predict_Number: %d"%(files[count],np.argmax(i))
	count = count + 1
