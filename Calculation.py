# -*- coding:utf-8 -*- 
from __future__ import division
from Config import *
from Stats import Model,Estimation,Test
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

	@staticmethod
	def GetDirectionResultPath(picfilename,direction):
		if not os.path.exists(RESULT_SOURCE_PATH+'direction_result_data/'):
			os.mkdir(RESULT_SOURCE_PATH+'direction_result_data/');
		return RESULT_SOURCE_PATH+'direction_result_data/'+picfilename[:-4]+str(direction)+'.png';

	@staticmethod
	def GetBoundaryGraphPath(picfilename):
		if not os.path.exists(RESULT_SOURCE_PATH+'boundary_result_data/'):
			os.mkdir(RESULT_SOURCE_PATH+'boundary_result_data/');
		return RESULT_SOURCE_PATH+'boundary_result_data/';


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
		print ">>>>> begin process",fname
		bscores = self.detect_boundary(fname, r_neigh=1.2);
		print ">>>>> show boundary ... "
		originRGB = Model.RgbFromFileName(DATA_SOURCE_PATH+fname)
		originGrey = Model.Rgb2grey(originRGB)
		multigrey = Model.Multigrey(originGrey)
		cu = CoordUtils()
		for p3 in bscores:
			x3 = p3[0]
			y3 = p3[1]
			(x, y) = cu.trans3d_2d(x3, y3)
			multigrey[x, y, 0] = 1
			multigrey[x, y, 1] = 0
			multigrey[x, y, 2] = 0
		Test.drawRGB(multigrey, save_path=Calculation.GetBoundaryGraphPath(fname))
		print ">>>>> boundary result finished"
		target_hyp,max_inlier = self.ransac(bscores);
		print ">>>>> final sun direction:", target_hyp," with %d inliers" % max_inlier
		center = (originRGB.shape[0]/2, originRGB.shape[1]/2)
		dx,dy,dz = self.__standard_normals((target_hyp[0], target_hyp[1], 0))
		Test.drawDirection(originRGB,center=center,directVect=(dx,dy),save_path=Calculation.GetDirectionResultPath(fname,target_hyp))
		print ">>>>> direction result finished"

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
		tmp = min(1.0,abs(x1*x2+y1*y2+z1*z2)/(math.sqrt(x1**2 + y1**2 + z1**2)*math.sqrt(x2**2 + y2**2 + z2**2)))
		return math.acos(tmp)*180/math.pi;

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


	def detect_boundary(self, fname, r_neigh, b0=0.1, xita_crease=45):
		BScores = {} # {(x,y,z,nx,ny,nz):Bpos}  type(Bpos):numpy.array(shape=(3,1))
		if os.path.exists(Calculation.GetBoundaryDataPath(fname,r_neigh,b0,xita_crease)):
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
				# 	i=0
				# 	while i<len(Nx):
				# 		node = Nx[i];
				# 		if label_tag[cu.trans3d_2d(node[0],node[1])] == 0:   # means unlabeled
				# 			del x[i];
				# 		else:
				# 			i+=1;

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
		# self.__report_boundary_status(BScores)
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
			# PI/6:(200,0),
			PI/4:(200,0),
			PI/3:(200,0),
			PI*0.48:(200,0),
			PI/2:(1,0),
		}
		for a in alpha_xita.keys():
			diff,t = alpha_xita[a];
			while (t<2*PI):
				ret.append((math.cos(a)*math.cos(t),math.cos(a)*math.sin(t), math.sin(a)));
				t += 2.0*PI/diff;
		kdtree = spatial.KDTree(ret);
		# print ret
		return ret,kdtree;

	def __generate_hyp(self,inliers):
		x = [p[:2] for p in inliers];
		y = [p[-1] for p in inliers];
		self.__clf.fit(x,y);
		return self.__standard_normals((-self.__clf.coef_[0],-self.__clf.coef_[1],1))


	def ransac(self, bscores, angel_e=15, stop_e=0.2, inner_max_iter=60, outter_max_iter=50):
		import random
		## Policy
		USE_BINS = True;
		FILTER_BOUNDERY_VEC = False;
		print "[USE_BINS]:",USE_BINS
		print "[FILTER_BOUNDERY_VEC]:",FILTER_BOUNDERY_VEC


		if FILTER_BOUNDERY_VEC: bscores = self.__filter_boudary_by_angel(bscores, 30);
		normals = [0 for i in xrange(len(bscores))];
		pos = [0 for i in xrange(len(bscores))];
		bvs = [0 for i in xrange(len(bscores))];
		cnt = 0;
		for bnode in bscores.keys():
			normals[cnt] = np.array(self.__standard_normals(PlyUtils.Get_normals(bnode)));
			bvs[cnt] = np.array(self.__standard_normals(bscores[bnode]));
			pos[cnt] = np.array(PlyUtils.Get_pos(bnode))
			cnt += 1;

		
		out_iter = 0;
		opt_hyp = None;
		opt_max_inner_cnt = -1;
		while out_iter<outter_max_iter:
			print "\n##### OUTTER ITER %d ######\n" % out_iter
			out_iter += 1;
			## Step-1: Initially, randomly find two vectors as the initial vectors

			init_one, init_two = random.sample(set([str(list(item)) for item in normals]),2);
			init_one = eval(init_one);
			init_two = eval(init_two)

			cur_hyp = self.__standard_normals(np.cross(init_one, init_two));

			if cur_hyp[2] < 0: cur_hyp = (-cur_hyp[0],-cur_hyp[1],-cur_hyp[2])   # recorrect the sun direction
			pre_hyp = None;

			## Step-2: Get pre-computed bin-vectors and quantize the surface normals
			bin_vectors, bv_kdtree = self.get_bin_vectors();
			bv_map = {};
			for idx in xrange(len(normals)):
				dis,nearest_idx = bv_kdtree.query(normals[idx]);
				# print normals[idx]
				bv_map[idx] = nearest_idx;
			

			## Step-3 loop
			inner_iter = 0;
			repeat_code_set = set();
			while inner_iter < inner_max_iter:
				inner_iter += 1;
				### Step-3.1 filter out specific boundary points and throw into bins
				select_bins = set();
				# helper = set()
				bv_bins = [[] for i in range(len(bin_vectors))];
				bv_bins_avers = [0 for i in range(len(bin_vectors))]
				for i in xrange(len(normals)):
					angel = self.Get_angel_degree(normals[i],cur_hyp);
					if angel<90+angel_e and angel>90-angel_e and np.dot(np.asarray(cur_hyp),bvs[i])>0:
						bv_bins[bv_map[i]].append(normals[i]);
						# helper.add(str(normals[i]));
						select_bins.add(bv_map[i]);
				# print ">>>>>> [helper] distinct normal inliers", helper
				
				### Step-3.2 calc inliers
				if USE_BINS:
					for i,Bin in enumerate(bv_bins):
						if len(Bin) == 0:
							bv_bins_avers[i] = None;
						else:
							bv_bins_avers[i] = self.__standard_normals(self.__get_centroid(Bin));


					inliers = []
					for b in iter(select_bins):
						if bv_bins_avers[b] != None:
							inliers.append(bv_bins_avers[b]);
					print '>>>>> inliers cnt', sum([len(item) for item in bv_bins])
					print '>>>>> inliers bins cnt', len(inliers)
					inliers_cnt = sum([len(item) for item in bv_bins])
				else:
					inliers = []
					for b in iter(bv_bins):
						for n in b:
							inliers.append(n);
					print '>>>>> inliers cnt', len(inliers)
					inliers_cnt = len(inliers)

				if len(inliers) == 0: break;

				### Step-3.3 update hypothesis
				pre_hyp = cur_hyp;
				cur_hyp = self.__generate_hyp(inliers);
				inner_max_inliers_cnt = inliers_cnt
				hashcode = hash(str(inliers_cnt) + str(cur_hyp) + str(pre_hyp));
				if hashcode in repeat_code_set:
					break;
				else:
					repeat_code_set.add(hashcode)
				print ">>>>> inner_max_inliers_cnt",inner_max_inliers_cnt;

				print ">>>>> update hyp", pre_hyp,'===>>',cur_hyp
				# print select_bins
				if (self.Get_angel_degree(pre_hyp,cur_hyp)<stop_e):
					# print self.Get_angel_degree(pre_hyp,cur_hyp)
					break;

			if inner_max_inliers_cnt > opt_max_inner_cnt:
				opt_max_inner_cnt = inner_max_inliers_cnt;
				opt_hyp = cur_hyp
		return opt_hyp,opt_max_inner_cnt



if __name__ == '__main__':
	calc = Calculation()
	cnt = 0
	total = 60
	for i in xrange(200,383):
		fn = "meas-%05d-00000.png" % i
		try:
			calc.process(fname=fn)
			cnt += 1
			if cnt >= total:
				break
		except:
			pass




