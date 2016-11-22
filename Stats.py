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
import gc

class Model:
	@staticmethod
	def Rgb2greyFromFileName(filename):
		return Model.Rgb2grey(mpimg.imread(filename))*1.0     # make it float

	@staticmethod
	def Rgb2grey(rgb):
		# call with Model.Rgb2grey(rgb)
		return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

	@staticmethod
	def GetIRModelPath():
		if not os.path.exists(RESULT_SOURCE_PATH+'model/'):
			os.mkdir(RESULT_SOURCE_PATH+'model/');
		return RESULT_SOURCE_PATH+'model/iRmodel.mdl';


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
		if os.path.exists(Model.GetIRModelPath()):
			print ">>>> Loading IR Model ..."
			self.__height,self.__width,self.__sample_num,self.__model = joblib.load(Model.GetIRModelPath())
		else:
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
			joblib.dump((self.__height,self.__width,self.__sample_num,self.__model),Model.GetIRModelPath(),compress = 3);

		assert self.__model.shape == (self.__height,self.__width);
		


	def get_model(self):
		if self.__model == None:
			self.__generate_model();
		return self.__model;


class Estimation:
	@staticmethod
	def GetLblDataPath(picfilename):
		if not os.path.exists(RESULT_SOURCE_PATH+'label_data/'):
			os.mkdir(RESULT_SOURCE_PATH+'label_data/');
		return RESULT_SOURCE_PATH+'label_data/('+picfilename+').lbldata';

	def __init__(self,model,thereshold=3,k=1000,c=0.25):
		## read pic data and convert to grey
	
		self.__model = model
		self.__height,self.__width = model.shape
		self.__R = None;    # should be a numpy array 
		self.__T = thereshold;
		self.__K = k;
		self.__C0 = c;
		self.__labels = None;  # float [0,1], 0 - shadows, 1 - sunlits

		os.system("g++ --std=gnu++0x -O3 -fPIC -shared "+"./cUtils.cpp -o "+"./cUtils.so")
		_dll = ctypes.cdll.LoadLibrary('./cUtils.so')
		_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')
		self.__label = _dll.label 
		self.__label.argtypes = [ctypes.c_int, ctypes.c_int, _doublepp, _doublepp, _doublepp, ctypes.c_int, ctypes.c_int] 
		self.__label.restype = None 


	def get_shadows_label_prob(self, filename):
		self.__greyscale = Model.Rgb2greyFromFileName(DATA_SOURCE_PATH+filename);
		if self.__labels == None:   # shape = (__height,__width)
			if not os.path.exists(Estimation.GetLblDataPath(filename)):
				self.__labels = np.zeros((self.__height,self.__width));

				modelpp = (self.__model.__array_interface__['data'][0] \
					+ np.arange(self.__model.shape[0])*self.__model.strides[0]).astype(np.uintp) 
				gspp = (self.__greyscale.__array_interface__['data'][0] \
					+ np.arange(self.__greyscale.shape[0])*self.__greyscale.strides[0]).astype(np.uintp)
				lbpp = (self.__labels.__array_interface__['data'][0] \
					+ np.arange(self.__labels.shape[0])*self.__labels.strides[0]).astype(np.uintp)
				m = ctypes.c_int(self.__height) 
				n = ctypes.c_int(self.__width) 

				self.__label(m,n,modelpp,gspp,lbpp,self.__T,self.__K);

				joblib.dump(self.__labels,Estimation.GetLblDataPath(filename),compress = 3);
			else:
				self.__labels = joblib.load(Estimation.GetLblDataPath(filename));
		return self.__labels;

	def clear_label(self):
		self.__labels = None;
		gc.collect();

	def get_shadows_label_tag(self, filename):
		self.get_shadows_label_prob(filename);
		label_tag = np.zeros_like(self.__labels);
		numLits = 0;
		numShadows = 0;
		for i in xrange(label_tag.shape[0]):
			for j in xrange(label_tag.shape[1]):
				if self.__labels[i,j] > 0.5+self.__C0:
					label_tag[i,j] = 1;
					numLits += 1;
				elif self.__labels[i,j] < 0.5-self.__C0:
					label_tag[i,j] = -1;
					numShadows += 1;
		print '######  %s shadow label #####' % filename
		print '#Valid: ', numLits+numShadows, '('+str(int((numLits+numShadows)/self.__model.size*100))+'%)';
		print '#Lits: ', numLits;
		print '#Shadows: ', numShadows;
		print '##################'

		return label_tag;

if __name__ == '__main__':
	est = Estimation(model = Model().get_model());
	for f in os.listdir(DATA_SOURCE_PATH):
		if f.endswith('.png'):
			est.clear_label();
			est.get_shadows_label_tag(filename = f);



