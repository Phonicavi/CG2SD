# -*- coding:utf-8 -*- 
from __future__ import division
from Config import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os,sys
import random
from sklearn.externals import joblib
import ctypes
from numpy.ctypeslib import ndpointer

class Model:
	@staticmethod
	def Rgb2greyFromFileName(filename):
		return Model.Rgb2grey(mpimg.imread(filename))

	@staticmethod
	def Rgb2grey(rgb):
		# call with Model.Rgb2grey(rgb)
		return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


	def __init__(self):
		self.clear();

	def clear(self):
		self.__width = 0;
		self.__height = 0;
		self.__sample_num = 0;
		self.__model = None;

	def __generate_model(self):
		## TODO:(Qiu Feng)
		# NEED TO set self.__width, self.__height, self.__sample_num, self.__model in this function
		# self.__model should be double nparray with size(width, height)
		m_default = 500
		n_default = 365
		files = os.listdir(DATA_SOURCE_PATH)
		grey_size = (m_default, n_default)
		for f in files:
			if f.endswith('.png'):
				Ii = Model.Rgb2greyFromFileName(DATA_SOURCE_PATH+f)
				grey_size = Ii.shape
				break
		grey_sum = np.zeros(grey_size)
		count = 0
		for f in files:
			if f.endswith('.png'):
				Ii = Model.Rgb2grey(mpimg.imread(DATA_SOURCE_PATH+f))
				grey_sum += Ii
				count += 1
		self.__height = grey_size[0]
		self.__width = grey_size[1]
		self.__sample_num = count
		albedo = grey_sum/float(count)
		self.__model = albedo

		assert self.__model.shape == (self.__height,self.__width);
		


	def get_model(self):
		if self.__model == None:
			self.__generate_model();
		return self.__model;


class Estimation:
	def __init__(self,filename,model,thereshold=3,k=1000):
		## read pic data and convert to grey
		self.__greyscale = Model.Rgb2greyFromFileName(DATA_SOURCE_PATH+filename);
		self.__model = model
		self.__height,self.__width = model.shape
		self.__R = None;    # should be a numpy array 
		self.__T = thereshold;
		self.__K = k;
		self.__labels = None;  # float [0,1], 0 - shadows, 1 - sunlits


	def get_shadows_label(self):
		if self.__labels == None:   # shape = (__height,__width)
			self.__labels = np.zeros((self.__height,self.__width));

			os.system("g++ --std=gnu++0x -O3 -fPIC -shared "+"./cUtils.cpp -o "+"./cUtils.so")
			_dll = ctypes.cdll.LoadLibrary('./cUtils.so')
			_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')
			_label = _dll.label 
			_label.argtypes = [ctypes.c_int, ctypes.c_int, _doublepp, _doublepp, _doublepp, ctypes.c_int, ctypes.c_int] 
			_label.restype = None 

			modelpp = (self.__model.__array_interface__['data'][0] \
				+ np.arange(self.__model.shape[0])*self.__model.strides[0]).astype(np.uintp) 
			gspp = (self.__greyscale.__array_interface__['data'][0] \
				+ np.arange(self.__greyscale.shape[0])*self.__greyscale.strides[0]).astype(np.uintp)
			lbpp = (self.__labels.__array_interface__['data'][0] \
				+ np.arange(self.__labels.shape[0])*self.__labels.strides[0]).astype(np.uintp)
			m = ctypes.c_int(self.__height) 
			n = ctypes.c_int(self.__width) 

			_label(m,n,modelpp,gspp,lbpp,self.__T,self.__K);

		return self.__labels;

if __name__ == '__main__':
	est = Estimation(filename = "meas-00001-00000.png",model = Model().get_model());
	print est.get_shadows_label();



