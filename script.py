import os
from Config import *
from Stats import Model, Estimation, Test
from Calculation import Calculation
from Utils import CoordUtils, PlyUtils


if __name__ == '__main__':
	calc = Calculation()
	cu = CoordUtils()
	est = Estimation(model=Model().get_model(),thu=0.75,thd=0.45);
	for f in os.listdir(DATA_SOURCE_PATH):
		if f.endswith('.png'):
			est.clear_label();
			tag_mat = est.get_shadows_label_tag(filename = f);
			BS = calc.detect_boundary(fname=f, r_neigh=1.2, b0=0.2, xita_crease=45)
			for p3 in BS:
				x3 = p3[0]
				y3 = p3[1]
				(x, y) = cu.trans3d_2d(x3, y3)
				tag_mat[x, y] = 2 # boundary: 2
			# teson = Test(tag_mat)
			# teson.draw()
			# break

