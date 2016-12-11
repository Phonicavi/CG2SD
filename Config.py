# -*- coding:utf-8 -*- 
import os
from sklearn.externals import joblib
DATA_SOURCE_PATH = "./dataset/"
RESULT_SOURCE_PATH = "./result/"
DYNAMIC_POLICY_PATH = "./dp/"

for path in [DATA_SOURCE_PATH,RESULT_SOURCE_PATH,DYNAMIC_POLICY_PATH]:
	if not os.path.exists(path):
		os.mkdir(path)

# params
ESTIMATION_THU = 0.75
ESTIMATION_THD = 0.40
DETECT_BOUNDARY_R_NEIGH = 1.2