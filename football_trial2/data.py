#coding:utf-8
"""
Author:wepon
Source:https://github.com/wepe

"""


import os
from PIL import Image
import numpy as np

#读取文件夹mnist下的41994张图片，图片为灰度图，所以为1通道，图像大小28*28
#如果是将彩色图作为输入,则将1替换为3

def load_data():
	#定义一个4维空数组data放置图片数据，dtype是元素类型
	data = np.empty((1788,3,320,180),dtype="float32")
	#定义一个一维数组label放置图片训练结果
	label = np.empty((1788,),dtype="uint8")
	#读取mnlist下面所有图片
	imgs = os.listdir("./mnist8")
	num = len(imgs)

	for i in range(num):
		#加载图片对象
		img = Image.open("./mnist8/"+imgs[i])
		#asarray将img对象转换成3维数组(像素点3个)表示，格式是float32
		arr = np.asarray(img,dtype="float32")
		#填充之前定义的4维数组，代表图片
		arr = np.transpose(arr)
		data[i,:,:,:] = arr
		#填充标注信息数组，这里由于命名规范，所以直接获取对应数据即可
		#这里需要注意：label必须是0 1 2 3这样递增的数字   e0 q1 r2 space3 t4 w5
		label[i] = int(imgs[i].split('.')[0])
        
        #缩小像素范围
		#最大值
        data /= np.max(data)
        #平均值
        data -= np.mean(data)
	return data,label






