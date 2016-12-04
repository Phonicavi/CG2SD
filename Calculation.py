# -*- coding:utf-8 -*- 
from __future__ import division
from Config import *
from Stats import Model,Estimation
from Utils import CoordUtils, PlyUtils
import numpy as np
from numpy import linalg as LA
import math,sys
from progressbar import ProgressBar
from sklearn import linear_model

class Calculation:
	@staticmethod
	def GetBoundaryDataPath(picfilename,r_neigh,b0,xita_crease):
		if not os.path.exists(RESULT_SOURCE_PATH+'boundary_data/'):
			os.mkdir(RESULT_SOURCE_PATH+'boundary_data/');
		return RESULT_SOURCE_PATH+'boundary_data/('+picfilename+'_'+str((r_neigh,b0,xita_crease))+').bddata';

	def __init__(self):
		self.__model = Model().get_model();
		self.__est = Estimation(model=self.__model,thu=0.75,thd=0.40); # thu: thereshold-up	thd: thereshold-down
		self.__clf = linear_model.LinearRegression(n_jobs=4)

		# self.__ransac = None
		# os.system("g++ --std=gnu++0x -O3 -fPIC -shared "+"./cRANSAC.cpp -o "+"./cRANSAC.so")
		# _dll = ctypes.cdll.LoadLibrary('./cRANSAC.so')
		# _doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')
		# self.__ransac = _dll.ransac 
		# self.__ransac.argtypes = [ctypes.c_int, _doublepp, _doublepp] 
		# self.__ransac.restype = [ctypes.POINTER(ctypes.c_double)]

	def process(self,fname):
		bscores = self.detect_boundary(fname, r_neigh=1.2);
		# self.ransac(bscores);

	@classmethod
	def __standard_normals(cls,v):
		x,y,z = v;
		scal = math.sqrt(x**2 + y**2 + z**2);
		return x/scal, y/scal, z/scal;

	def __is_angel_less_than(self,v1,v2,deg=45):
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

	@classmethod
	def Get_angel_degree(cls,v1,v2):
		x1,y1,z1 = cls.__standard_normals(v1);
		x2,y2,z2 = cls.__standard_normals(v2);
		return math.acos(abs(x1*x2+y1*y2+z1*z2)/(math.sqrt(x1**2 + y1**2 + z1**2)*math.sqrt(x2**2 + y2**2 + z2**2)))*180/math.pi;

	def __report_boundary_status(self, bscores):
		angel_li = [0 for i in range(7)]
		# 0. >95
		# 1. 90-95
		# 2. 85-90
		# 3. 80-85
		# 4. 75-80
		# 5. 70-75
		# 6. <70
		for bnode in bscores.keys():
			angel = self.Get_angel_degree(PlyUtils.Get_normals(bnode),bscores[bnode]);
			# print angel
			if angel > 95:
				angel_li[0] += 1;
			elif angel < 70:
				angel_li[6] += 1;
			else:
				angel_li[int((95-angel)/5)+1] += 1;
		print "##### Report: boundary angel range #####"
		for i in range(len(angel_li)):
			print "[%d]: %.1f%%" % (i,angel_li[i]*100/len(bscores));
		print "################"


	def detect_boundary(self, fname, r_neigh, b0=0.2, xita_crease=45):
		BScores = {} # {(x,y,z,nx,ny,nz):Bpos}  type(Bpos):numpy.array(shape=(3,1))
		if os.path.exists(Calculation.GetBoundaryDataPath(fname,r_neigh,b0,xita_crease)):
			pass
			BScores = joblib.load(Calculation.GetBoundaryDataPath(fname,r_neigh,b0,xita_crease))
		else:
			self.__est.clear_label();
			label_tag = self.__est.get_shadows_label_tag(filename = fname);
			assert(self.__model.size == label_tag.size);

			pu = PlyUtils();
			cu = CoordUtils();

			all_points = pu.get_all_3d_points();   # currently all 3d points are considered
			

			cnt = 0
			pbar = ProgressBar(maxval=len(all_points)).start()
			for point in iter(all_points):
				cnt += 1;
				pbar.update(cnt)
				sys.stdout.flush();
				thisLabel = label_tag[cu.trans3d_2d(point[0],point[1])]
				if thisLabel == 0: continue;

				Nx = pu.find_point_within_rad(PlyUtils.Get_pos(point),r_neigh); 
				# if thisLabel == -1:     # if shadow, then ignore the unlabeled
				# 	for i in xrange(len(Nx)):
				# 		node = Nx[i];
				# 		if label_tag[cu.trans3d_2d(node[0],node[1])] == 0:   # means unlabeled
				# 			Nx[i] = [];
				# 	while [] in Nx: Nx.remove([]);

				Nx_star = [];
				for neigh in Nx:
					neiLabel = label_tag[cu.trans3d_2d(neigh[0],neigh[1])];
					if neiLabel != thisLabel and self.__is_angel_less_than(PlyUtils.Get_normals(neigh),PlyUtils.Get_normals(point),xita_crease):
						Nx_star.append(neigh)

				if (len(Nx_star) == 0):
					continue;
				centroid = self.__get_centroid([PlyUtils.Get_pos(p) for p in Nx_star]);

				Bpos = -1.0*thisLabel*(len(Nx_star)/len(Nx))*(centroid-np.array(PlyUtils.Get_pos(point)))
				if LA.norm(Bpos) > b0:
					BScores[point] = Bpos;
			pbar.finish();
			joblib.dump(BScores,Calculation.GetBoundaryDataPath(fname,r_neigh,b0,xita_crease),compress=3);

		print "#Valid boundary pixel:",len(BScores)
		self.__report_boundary_status(BScores)
		# cnt = 0;
		# for item in BScores:
		# 	if BScores[item][2] > 0: cnt += 1;
		# print cnt/len(BScores);
		return BScores;

	def __filter_boudary_by_angel(self, bscores, th=40):
		aftered = {};
		for bnode in bscores.keys():
			angel = self.Get_angel_degree(PlyUtils.Get_normals(bnode),bscores[bnode]);
			if angel > th:
				aftered[bnode] = bscores[bnode];
		return aftered;

	def get_bin_vectors(self):
		from scipy import spatial
		PI = math.pi;
		ret = []
		# 30,45,60,80
		alpha_xita = {
			PI/6:(10,PI/20),
			PI/4:(8,0),
			PI/3:(6,PI/12),
			PI*0.45:(4,0),
			PI/2:(1,0),
		}
		for a in alpha_xita.keys():
			diff,t = alpha_xita[a];
			while (t<2*PI):
				ret.append((math.cos(a)*math.cos(t),math.cos(a)*math.sin(t), math.sin(a)));
				t += 2*PI/diff;
		kdtree = spatial.KDTree([tuple(item) for item in ret]);
		return ret,kdtree;

	def __generate_hyp(self,inliers):
		x = [p[:2] for p in inliers];
		y = [p[-1] for p in inliers];
		self.__clf.fit(x,y);
		return self.__standard_normals((-self.__clf.coef_[0],-self.__clf.coef_[1],1))


	def ransac(self, bscores):
		# def dot_prod(x,y):
		# 	return np.dot(x,y);
		# def cross_prod(x,y):
		# 	return np.cross(x,y)

		bscores = self.__filter_boudary_by_angel(bscores, 40);
		normals = [0 for i in xrange(len(bscores))];
		bvs = [0 for i in xrange(len(bscores))];
		cnt = 0;
		for bnode in bscores.keys():
			normals[cnt] = np.array(self.__standard_normals(PlyUtils.Get_normals(bnode)));
			bvs[cnt] = np.array(self.__standard_normals(bscores[bnode]));
			cnt += 1;
		# normals = np.array(normals);
		# bvs = np.array(bvs);

		## Step-1: Initially, find two most likely vectors as the initial vectors
		best_two = [(-1,0),(-1,0)];
		for idx in xrange(len(normals)):
			angel = self.Get_angel_degree(normals[idx],bvc[idx]);
			if (angel > best_two[0][1]):
				best_two[1] = best_two[0];
				best_two[0] = (idx,angel);
			elif (angel > best_two[1][1]):
				best_two[1] = (idx,angel);
		cur_hyp = self.__standard_normals(np.cross(normals[best_two[0][0]], normals[best_two[1][0]]));
		if cur_hyp[2] < 0: cur_hyp = (-cur_hyp[0],-cur_hyp[1],-cur_hyp[2])   # recorrect the sun direction

		## Step-2: Get pre-computed bin-vectors and quantize the surface normals
		bin_vectors, bv_kdtree = self.get_bin_vectors();
		bv_bins = [[] for i in range(len(bin_vectors))];
		bv_bins_avers = [0 for i in range(len(bin_vectors))]
		bv_map = {};

		for idx in xrange(len(normals)):
			nearest_idx = bv_kdtree.query(normals[idx]);
			bv_bins[nearest_idx].append(idx);
			bv_map[idx] = nearest_idx;
		for i,Bin in enumerate(bv_bins):
			if len(Bin) == 0:
				bv_bins_avers[i] = None;
			else:
				bv_bins_avers[i] = self.__standard_normals(self.__get_centroid(Bin));

		## Step-3 loop
		# while 1:

			




		# normalspp = (normals.__array_interface__['data'][0] \
		# 		+ np.arange(normals.shape[0])*normals.strides[0]).astype(np.uintp) 
		# bvspp = (bvs.__array_interface__['data'][0] \
		# 		+ np.arange(bvs.shape[0])*bvs.strides[0]).astype(np.uintp) 

		# res = self.__ransac(len(bscores),normalspp,bvspp)




if __name__ == '__main__':
	calc = Calculation();
	# print calc.get_bin_vectors();
	calc.process(fname = "meas-00043-00000.png");



