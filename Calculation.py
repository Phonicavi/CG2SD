# -*- coding:utf-8 -*- 
from Config import *
from Stats import Model,Estimation
from Utils import CoordUtils, PlyUtils
import numpy as np
from numpy import linalg as LA
import math,sys

class Calculation:
	@staticmethod
	def GetBoundaryDataPath(picfilename):
		if not os.path.exists(RESULT_SOURCE_PATH+'boundary_data/'):
			os.mkdir(RESULT_SOURCE_PATH+'boundary_data/');
		return RESULT_SOURCE_PATH+'boundary_data/('+picfilename+').bddata';

	def __init__(self):
		self.__model = Model().get_model();
		self.__est = Estimation(model=self.__model,thu=0.75,thd=0.6);

	def process(self,fname):
		self.detect_boundary(fname, r_neigh=10);

	def __standard_normals(self,v):
		x,y,z = v;
		scal = math.sqrt(x**2 + y**2 + z**2);
		return x/scal, y/scal, z/scal;

	def __is_angle_less_than(self,v1,v2,deg=45):
		x1,y1,z1 = self.__standard_normals(v1);
		x2,y2,z2 = self.__standard_normals(v2);
		cosXita = abs(x1*x2+y1*y2+z1*z2)/(math.sqrt(x1**2 + y1**2 + z1**2)*math.sqrt(x2**2 + y2**2 + z2**2));
		return cosXita > math.cos(deg*math.pi/180);

	def __get_centroid(self,pointlist):
		num = len(pointlist);
		xSum = sum([tp[0] for tp in pointlist])
		ySum = sum([tp[1] for tp in pointlist])
		zSum = sum([tp[2] for tp in pointlist])
		return np.array([xSum/num, ySum/num, zSum/num]);

	def detect_boundary(self, fname, r_neigh, b0=0.2, xita_crease =45):
		if os.path.exists(Calculation.GetBoundaryDataPath(fname)):
			return joblib.load(Calculation.GetBoundaryDataPath(fname))

		self.__est.clear_label();
		label_tag = self.__est.get_shadows_label_tag(filename = fname);
		assert(self.__model.size == label_tag.size);

		pu = PlyUtils();
		cu = CoordUtils();

		all_points = pu.get_all_3d_points();   # currently all 3d points are considered
		BScores = {} # {(x,y,z,nx,ny,nz):Bpos}  type(Bpos):numpy.array(shape=(1,3))

		cnt = 0;
		for point in all_points:
			cnt += 1;
			if cnt % 10 == 0:
				print str(cnt)
			sys.stdout.flush();
			thisLabel = label_tag[cu.trans3d_2d(point[0],point[1])]
			if thisLabel == 0: continue;

			Nx = pu.find_point_within_rad(PlyUtils.Simplified(point),r_neigh); # TODO: (Qiu Feng)remove unlabled?

			Nx_star = [];
			for neigh in Nx:
				neiLabel = label_tag[cu.trans3d_2d(neigh[0],neigh[1])];
				# TODO: (Qiu Feng) specify the “contrary”
				if neiLabel != thisLabel and self.__is_angle_less_than(PlyUtils.Get_normals(neigh),PlyUtils.Get_normals(point),xita_crease):
					Nx_star.append(neigh)

			if (len(Nx_star) == 0):
				print 'o',
				sys.stdout.flush()
				continue;
			centroid = self.__get_centroid([PlyUtils.Simplified(p) for p in Nx_star]);

			Bpos = -1.0*thisLabel*(len(Nx_star)/len(Nx))*(centroid-np.array(PlyUtils.Simplified(point)))
			if LA.norm(Bpos) > b0:
				BScores[point] = Bpos;

		joblib.dump(BScores,Calculation.GetBoundaryDataPath(fname),compress=3);

		print "#Valid boundary pixel:",len(BScores)
		print BScores
		return BScores;

if __name__ == '__main__':
	calc = Calculation();
	calc.process(fname = "meas-00002-00000.png");



