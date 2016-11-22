# -*- coding:utf-8 -*- 
from __future__ import division
from Config import *
from plyfile import PlyData, PlyElement
import math
import os
from scipy import spatial

class CoordUtils:
	minX2d,maxX2d,minY2d,maxY2d = 12.0,352.0,9.0,498.0
	minX3d,maxX3d,minY3d,maxY3d = -39.018806, 29.731207,-29.078983, 70.853516

	def __init__(self):
		ax23 = (CoordUtils.maxX3d-CoordUtils.minX3d) / (CoordUtils.maxX2d-CoordUtils.minX2d);
		bx23 = (CoordUtils.minX3d*CoordUtils.maxX2d - CoordUtils.maxX3d*CoordUtils.minX2d) / (CoordUtils.maxX2d-CoordUtils.minX2d);
		ay23 = (CoordUtils.maxY3d-CoordUtils.minY3d) / (CoordUtils.minY2d-CoordUtils.maxY2d);
		by23 = (CoordUtils.maxY3d*CoordUtils.maxY2d - CoordUtils.minY3d*CoordUtils.minY2d) / (CoordUtils.maxY2d-CoordUtils.minY2d);

		ax32 = (CoordUtils.maxX2d-CoordUtils.minX2d) / (CoordUtils.maxX3d-CoordUtils.minX3d);
		bx32 = (CoordUtils.minX2d*CoordUtils.maxX3d - CoordUtils.maxX2d*CoordUtils.minX3d) / (CoordUtils.maxX3d-CoordUtils.minX3d);
		ay32 = (CoordUtils.maxY2d-CoordUtils.minY2d) / (CoordUtils.minY3d-CoordUtils.maxY3d);
		by32 = (CoordUtils.maxY2d*CoordUtils.maxY3d - CoordUtils.minY2d*CoordUtils.minY3d) / (CoordUtils.maxY3d-CoordUtils.minY3d);

		self.__cof23 = (ax23,bx23,ay23,by23);
		self.__cof32 = (ax32,bx32,ay32,by32);

	def trans2d_3d(self,x,y):
		(ax23,bx23,ay23,by23) = self.__cof23;
		return (ax23*y+bx23,ay23*x+by23);

	def trans3d_2d(self,x,y):
		(ax32,bx32,ay32,by32) = self.__cof32;
		return (int(round(ay32*y+by32)),int(round(ax32*x+bx32)));

class PlyUtils:
	_3dpoint_seemed = None;
	KDTree = None;
	_3dpoint_map = None;

	@staticmethod
	def GetPlyFileName():
		return DATA_SOURCE_PATH+'tentacle.ply'

	@staticmethod
	def Get3DPointDataName():
		return RESULT_SOURCE_PATH+'3dpoint.dat'

	@staticmethod
	def Simplified(point):
		return (point[0],point[1],point[2])
	@staticmethod
	def Get_normals(point):
		return (point[3],point[4],point[5])

	@classmethod
	def Build(cls):
		if cls.KDTree != None \
			and cls._3dpoint_seemed != None\
			and cls._3dpoint_map != None : return;

		if os.path.exists(PlyUtils.Get3DPointDataName()):
			print ">>>>> Loading 3d points ... ",
			cls._3dpoint_seemed = joblib.load(PlyUtils.Get3DPointDataName());
			print len(cls._3dpoint_seemed),'points in total'
		else:
			plydata = PlyData.read(PlyUtils.GetPlyFileName());
			dic = {};
			cls._3dpoint_seemed = []
			for i in xrange(plydata['vertex'].count):
				vertex = plydata['vertex'][i]
				(x, y, z, nx, ny, nz) = (vertex[t] for t in ('x', 'y', 'z', 'nx', 'ny', 'nz'))
				if dic.has_key((x,y)):
					if (z > dic[(x,y)][2]):
						dic[(x,y)] = (x, y, z, nx, ny, nz)
				else:
					dic[(x,y)] = (x, y, z, nx, ny, nz)
			cls._3dpoint_seemed = dic.values();
			joblib.dump(cls._3dpoint_seemed,PlyUtils.Get3DPointDataName(),compress=3)


		print ">>>>> Generating KDTree ..."
		pli = [(tp[0],tp[1],tp[2]) for tp in cls._3dpoint_seemed];
		cls._3dpoint_map = dict(zip(pli,cls._3dpoint_seemed))
		cls.KDTree = spatial.KDTree(pli);
		print ">>>>> Done Generating"


	def __init__(self, ):
		PlyUtils.Build();

	def get_all_3d_points(self):
		return PlyUtils._3dpoint_seemed;

	def find_point_within_rad(self,pos,rad):  # pos=(x,y,z)
		return [PlyUtils._3dpoint_seemed[id] for id in PlyUtils.KDTree.query_ball_point(list(pos),rad)];

	


	


if __name__ == '__main__':
	# cu = CoordUtils();
	# # print cu.trans2d_3d(352.0,498.0)
	# print cu.trans3d_2d(-39.1,70)
	pu = PlyUtils();
	print pu.standard_normals(8.026257391562132e-25, 7.754818242684634e-25, 7.331485945625383e-38);


