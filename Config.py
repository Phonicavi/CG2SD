# -*- coding:utf-8 -*- 
import os
from sklearn.externals import joblib
DATA_SOURCE_PATH = "./dataset/"
RESULT_SOURCE_PATH = "./result/"

for path in [DATA_SOURCE_PATH,RESULT_SOURCE_PATH]:
	if not os.path.exists(path):
		os.mkdir(path)