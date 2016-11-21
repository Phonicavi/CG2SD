# -*- coding:utf-8 -*- 
from Config import *
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

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

	def __generate_model():
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
				Ii = rgb2grey(mpimg.imread('./1/'+f))
				grey_sum += Ii
				count += 1
		self.__width = grey_size[0]
		self.__height = grey_size[1]
		self.__sample_num = count
		albedo = grey_sum/float(count)
		self.__model = albedo

		assert self.__model.shape == (self.__width,self.__height);
		


	def get_model(self):
		if self.__model == None:
			self.__generate_model();
		return self.__model;


class Estimation:
	def __init__(self,filename,model,thereshold=3,k=1000):
		## read pic data and convert to grey
		self.__greyscale = Model.Rgb2greyFromFileName(filename);
		self.__model = model
		self.__width,self.__height = model.shape
		self.__R = None;    # should be a numpy array 
		self.__T = thereshold;
		self.__K = k;
		self.__labels = None;  # float [0,1], 0 - shadows, 1 - sunlits

	def __POS(tp): # tuple -> int
		return tp[0]*self.__height+tp[1];

	def __index(pos): #int --> tuple
		return int(pos/self.__height), pos%self.__height;

	def get_R_at(self,x,y): #type(x) = tuple()
		return self.get_R()[self.__POS(x),self.__POS(y)];

	def get_R(self):
		if self.__R == None:
			self.__R = np.zeros((self.__model.size,self.__model.size));
			for i in xrange(self.__width):   # (i,j) : (k,l)
				for j in xrange(self.__height):
					for k in xrange(self.__width):
						for l in xrange(self.__height):
							self.__R[self.__POS((i,j)), self.__POS((k,l))] = \
								(self.__greyscale[i,j]/self.__model[i,j])/(self.__greyscale[k,l]/self.__model[k,l])

		return self.__R

	def get_shadows_label(self):
		if self.__labels == None:   # shape = (__width,__height)
			for i in xrange(self.__width):   # (i,j) : (k,l)
				for j in xrange(self.__height):
					self.__labels = self.__label(i,j);
		return self.__labels;

	def __label(i,j):
		import random
		TTL = [i for i in xrange (self.__model.size)];
		TTL.remove(self.__POS((i,j)));
		candidates = random.sample(TTL,self.__K);
		numLits = 0;
		numValid = 0;
		for c in candidates:
			if get_R_at((i,j),c) > self.__T:
				numLits += 1;
				numValid += 1;
			elif get_R_at((i,j),c) < 1.0/self.__T:
				numValid += 1;
		return float(numLits)/numValid;



