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
	def GetDirectionResultPath(picfilename,direction,use_bins):
		path = RESULT_SOURCE_PATH+'direction_result_data/'+("USE_BINS" if use_bins else "NO_BINS")+'/'
		if not os.path.exists(path):
			os.makedirs(path);
		return path+picfilename[:-4]+str(direction)+'.png';

	@staticmethod
	def GetBoundaryGraphPath(picfilename):
		if not os.path.exists(RESULT_SOURCE_PATH+'boundary_result_data/'):
			os.mkdir(RESULT_SOURCE_PATH+'boundary_result_data/');
		return RESULT_SOURCE_PATH+'boundary_result_data/'+picfilename;


	def __init__(self):
		self.__model = Model().get_model();
		self.__est = Estimation(model=self.__model,thu=ESTIMATION_THU,thd=ESTIMATION_THD); # thu: thereshold-up	thd: thereshold-down
		self.__clf = linear_model.LinearRegression(n_jobs=4)

		# self.__ransac = None
		# os.system("g++ --std=gnu++0x -O3 -fPIC -shared "+"./cRANSAC.cpp -o "+"./cRANSAC.so")
		# _dll = ctypes.cdll.LoadLibrary('./cRANSAC.so')
		# _doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')
		# self.__ransac = _dll.ransac 
		# self.__ransac.argtypes = [ctypes.c_int, _doublepp, _doublepp] 
		# self.__ransac.restype = [ctypes.POINTER(ctypes.c_double)]

	def process(self,fname, detect_boundary_only=False):
		print ">>>>> begin process",fname
		bscores = self.detect_boundary(fname, r_neigh=DETECT_BOUNDARY_R_NEIGH);
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
		if detect_boundary_only == True: return
		## Policy
		USE_BINS = True;
		FILTER_BOUNDERY_VEC = False;
		target_hyp,max_inlier = self.ransac(bscores,USE_BINS=USE_BINS,FILTER_BOUNDERY_VEC=FILTER_BOUNDERY_VEC);
		print ">>>>> final sun direction:", target_hyp," with %d inliers" % max_inlier

		# tmpPathPrefix = fname[:16];
		# # print tmpPathPrefix
		# for file in os.listdir("./result/direction_result_data/USE_BINS/"):
		# 	if file.startswith(tmpPathPrefix):
		# 		aaa = file[16:-4];
		# 		# print aaa;
		# 		target_hyp = eval(aaa);
		# 		break;


		# print target_hyp
		center = (originRGB.shape[1]/2, originRGB.shape[0]/2)
		dx,dy = cu.scale3d_2d(target_hyp[0], target_hyp[1])
		dx,dy,dz = self.__standard_normals((dx,dy, 0))
		Test.drawDirection(originRGB,center=center,directVect=(dx,-dy),save_path=Calculation.GetDirectionResultPath(fname,target_hyp,USE_BINS))
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


	def ransac(self, bscores, angel_e=15, stop_e=0.2, inner_max_iter=60, outter_max_iter=50, USE_BINS=True, FILTER_BOUNDERY_VEC = False):
		import random
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
		init_cur_hyp_set = set()
		while out_iter<outter_max_iter:
			print "\n##### OUTTER ITER %d ######\n" % out_iter
			out_iter += 1;
			## Step-1: Initially, randomly find two vectors as the initial vectors

			init_one, init_two = random.sample(set([str(list(item)) for item in normals]),2);
			init_one = eval(init_one);
			init_two = eval(init_two)

			cur_hyp = self.__standard_normals(np.cross(init_one, init_two));

			if cur_hyp[2] < 0: cur_hyp = (-cur_hyp[0],-cur_hyp[1],-cur_hyp[2])   # recorrect the sun direction
			if str(cur_hyp) in init_cur_hyp_set:
				continue;
			else:
				init_cur_hyp_set.add(str(cur_hyp))
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
			inner_max_inliers_cnt = -1;
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

class DynamicPolicy:
	@staticmethod
	def GetModelPath():
		return DYNAMIC_POLICY_PATH+"dynamicPolicy.mdl";
	@staticmethod
	def GetFeaturePath():
		return DYNAMIC_POLICY_PATH+"dynamicPolicy_fea.data";
	@staticmethod
	def GetLabelPath():
		return DYNAMIC_POLICY_PATH+"label-comfine.csv";
	@staticmethod
	def GetDPResultPath():
		path = RESULT_SOURCE_PATH+"direction_result_data/dp/"
		if not os.path.exists(path):
			os.makedirs(path);
		return path
	@staticmethod
	def GetDPResultCSVPath():
		path = RESULT_SOURCE_PATH+"score/"
		if not os.path.exists(path):
			os.makedirs(path);
		return path+"dp_result.csv"

	@staticmethod
	def GetScorePath(isdp=False):
		path = RESULT_SOURCE_PATH+"score/"
		if not os.path.exists(path):
			os.makedirs(path);
		return (path+'dynamic_policy.csv') if isdp else (path+'single_policy.csv');

	def __init__(self):
		self.clear();
		self.dpmodel = self.__load_model() if os.path.exists(DynamicPolicy.GetModelPath()) else self.__generate_model();

	def clear(self):
		from sklearn.tree import DecisionTreeClassifier
		from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
		from sklearn.naive_bayes import GaussianNB
		from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
		from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
		from sklearn.linear_model import RidgeClassifier

		self.testdata = None;
		self.traindata = None;
		self.split_ratio=0.80
		self.classifiers = [
		    # ("Random Forest(entropy)",RandomForestClassifier(criterion = 'entropy',n_jobs= -1)),
		    # ("Random Forest(gini)",RandomForestClassifier(criterion = 'gini', max_features = 'auto',n_jobs= -1)),
		    ("Decision Tree",DecisionTreeClassifier()),
		    ("RidgeClassifier",RidgeClassifier()),
		    # ("AdaBoost",AdaBoostClassifier()),
		    ("QDA",QDA()),
		    ("LDA",LDA()),
		    ("GBDT",GradientBoostingClassifier(max_features = 'auto')),
		    ]

	def evaluate_model(self,times=50,classifier=None):
		import random
		from sklearn.metrics import accuracy_score
		fea = self.__generate_feature() if not os.path.exists(DynamicPolicy.GetFeaturePath()) else joblib.load(DynamicPolicy.GetFeaturePath());
		clf = self.dpmodel if classifier == None else classifier
		ttl_acc = 0;

		for i in xrange(times):
			random.shuffle(fea);
			traindata = fea[0:int(len(fea)*self.split_ratio)];
			print traindata[0];
			testdata = fea[int(len(fea)*self.split_ratio):];
			TrainX = [item[:-1] for item in traindata]
			TrainY = [item[-1] for item in traindata]
			TestX = [item[:-1] for item in testdata]
			TestY = [item[-1] for item in testdata]

			clf.fit(TrainX, TrainY)
			PredY = clf.predict(TestX)
			ttl_acc += accuracy_score(TestY,PredY)
			print accuracy_score(TrainY,clf.predict(TrainX)),accuracy_score(TestY,PredY)
		print '##### Model Evaluation #####';
		print '>>>>> Aver accuracy_score: %.2f%%' % (ttl_acc*1.0/times*100);
		return ttl_acc/times
	def __load_model(self):
		return joblib.load(DynamicPolicy.GetModelPath());

	def __generate_feature(self):
		## step1: get all the name,label from label data
		raw_data = {};
		with open(DynamicPolicy.GetLabelPath(),'r+') as f1:
			f1.readline();
			for line in f1.readlines():
				tmp = line.strip().split(',');
				raw_data[tmp[0]] = [int(tmp[1])];

		## step2: get average grayscale for every row
		for fname in os.listdir(DATA_SOURCE_PATH):
			if raw_data.has_key(fname[:-4]):
				gs = Model.Rgb2greyFromFileName(DATA_SOURCE_PATH+fname);
				raw_data[fname[:-4]].append(np.average(gs));
		for item in raw_data.values():
			print item
			assert(len(item) == 2);

		##step3: get label(shadow and sunlit) number, validnum
		est = Estimation(model=Model().get_model(),thu=ESTIMATION_THU,thd=ESTIMATION_THD);
		for fname in raw_data.keys():
			est.clear_label();
			nl,ns,vn = est.get_shadows_label_tag(filename = fname+'.png', stats_only=True);
			raw_data[fname] += [nl,ns];
		for item in raw_data.values():
			print item
			assert(len(item) == 4);

		##step4 get boundary data
		ca = Calculation();
		for fname in raw_data.keys():
			bscores = ca.detect_boundary(fname+'.png', r_neigh=DETECT_BOUNDARY_R_NEIGH);
			bsnum = len(bscores);
			aver_norm = 0;
			mat = []
			mat1 = []
			for bs in bscores.keys():
				aver_norm += LA.norm(bscores[bs])
				mat.append(bscores[bs]);
				mat1.append([bs[3],bs[4],bs[5]]);
			aver_norm /= bsnum;
			mat = np.array(mat);
			mat1 = np.array(mat1);
			vec_aver = np.mean(mat,0);  # 3
			vec_aver1 = np.mean(mat1,0);  # 3
			vec_var = np.var(mat,0); # 3
			vec_var1 = np.var(mat1,0); # 3
			raw_data[fname] += [bsnum, aver_norm,vec_aver[0],vec_aver[1],vec_aver[2], vec_var[0], vec_var[1], vec_var[2], \
					vec_aver1[0],vec_aver1[1],vec_aver1[2], vec_var1[0], vec_var1[1], vec_var1[2]];
			raw_data[fname] += list(np.cov(mat.T).reshape(-1));
			raw_data[fname] += list(np.cov(mat1.T).reshape(-1));
		for item in raw_data.values():
			print item
			assert(len(item) == 36);

		fea = [0 for i in xrange(len(raw_data))]

		cnt = 0;
		for item in raw_data.keys():
			fea[cnt] = raw_data[item][1:];
			fea[cnt].append(raw_data[item][0]);
			cnt += 1;
		joblib.dump(fea,DynamicPolicy.GetFeaturePath(),compress=3);
		return fea


	def __generate_model(self,repeat=100):
		import random
		from sklearn.metrics import classification_report as clfr
		from sklearn.metrics import accuracy_score
		
		best_clf = None;
		best_acc = -1;
		for name, clf in self.classifiers:
			acc = self.evaluate_model(times=repeat,classifier=clf)
			print name, acc
			if acc > best_acc:
				best_acc = acc;
				best_clf = clf;
			# print(clfr(TestY, PredY))
		joblib.dump(best_clf,DynamicPolicy.GetModelPath(),compress=3);

		return best_clf
	def select(self,fname):
		print '>>>>> Selecting',fname
		fn = fname[:-4];
		x = []
		for name in os.listdir(DATA_SOURCE_PATH):
			if name.startswith(fn):
				gs = Model.Rgb2greyFromFileName(DATA_SOURCE_PATH+name);
				x.append(np.average(gs));
				break;

		est = Estimation(model=Model().get_model(),thu=ESTIMATION_THU,thd=ESTIMATION_THD);
		est.clear_label();
		nl,ns,vn = est.get_shadows_label_tag(filename = fname, stats_only=True);
		x += [nl,ns];

		ca = Calculation();
		bscores = ca.detect_boundary(fname, r_neigh=DETECT_BOUNDARY_R_NEIGH);
		bsnum = len(bscores);
		aver_norm = 0;
		mat = []
		mat1 = []
		for bs in bscores.keys():
			aver_norm += LA.norm(bscores[bs])
			mat.append(bscores[bs]);
			mat1.append([bs[3],bs[4],bs[5]]);
		aver_norm /= bsnum;
		mat = np.array(mat);
		mat1 = np.array(mat1);
		vec_aver = np.mean(mat,0);  # 3
		vec_aver1 = np.mean(mat1,0);  # 3
		vec_var = np.var(mat,0); # 3
		vec_var1 = np.var(mat1,0); # 3
		x += [bsnum, aver_norm,vec_aver[0],vec_aver[1],vec_aver[2], vec_var[0], vec_var[1], vec_var[2], \
				vec_aver1[0],vec_aver1[1],vec_aver1[2], vec_var1[0], vec_var1[1], vec_var1[2]];
		x += list(np.cov(mat.T).reshape(-1));
		x += list(np.cov(mat1.T).reshape(-1));

		assert len(x) == 35
		lbl = self.dpmodel.predict([x])[0];
		path = RESULT_SOURCE_PATH+'direction_result_data/'+("USE_BINS" if lbl else "NO_BINS")+'/';
		import shutil
		for name in os.listdir(path):
			if name.startswith(fn):
				shutil.copyfile(path+name, DynamicPolicy.GetDPResultPath()+name);
				break;
		return (fn,lbl)

	def score(self):
		cnt = 0;
		use_bins_score = 0;
		no_bins_score = 0;
		dp_score = 0;
		with open(DynamicPolicy.GetScorePath(isdp=False),'r+') as f1, open(DynamicPolicy.GetDPResultCSVPath(),'r+') as f2:
			# name, use_bin_score, no_bin_score
			select = []
			for line in f2.readlines():
				select.append(int(line.strip().split(',')[1]))
			for line in f1.readlines():
				li = line.strip().split(',');
				use_bins_score += int(li[1]);
				no_bins_score += int(li[2]);
				dp_score += int(li[1]) if select[cnt] == 1 else int(li[2]);
				cnt += 1
		print ">>>>> Performance metrics: [USE_BIN] %.2f, [NO_BINS] %.2f, [DP] %.2f" % \
			(use_bins_score/cnt, no_bins_score/cnt, dp_score/cnt);
		return use_bins_score/cnt, no_bins_score/cnt, dp_score/cnt;







if __name__ == '__main__':
	# calc = Calculation()
	# cnt = 0
	# total = 1
	# for i in xrange(358,383):
	# 	fn = "meas-%05d-00000.png" % i
	# 	try:
	# 		calc.process(fname=fn,detect_boundary_only=False)
	# 		cnt += 1
	# 		if cnt >= total:
	# 			break
	# 	except Exception,e:
	# 		print e
	# 		pass
	dp = DynamicPolicy();
	# dp.evaluate_model(times=200);
	dp.score();
	# cnt = 0
	# total = 10000
	# ret = []
	# for i in xrange(1,383):
	# 	fn = "meas-%05d-00000.png" % i
	# 	try:
	# 		ret.append(dp.select(fn))
	# 		cnt += 1
	# 		if cnt >= total:
	# 			break
	# 	except Exception,e:
	# 		print e
	# with open(DynamicPolicy.GetDPResultCSVPath(),'w+') as f1:
	# 	for item in ret:
	# 		f1.write(item[0]+','+str(item[1])+'\n');
			




