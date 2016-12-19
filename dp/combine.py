import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

NO_BINS = os.listdir('./NO_BINS/')
USE_BINS = os.listdir('./USE_BINS/')

i = 0
for f0 in NO_BINS:
	name = f0[0:16]
	f1 = USE_BINS[i]
	i += 1
	assert (name == f1[0:16])
	I0 = mpimg.imread('./NO_BINS/'+f0)
	I1 = mpimg.imread('./USE_BINS/'+f1)
	I = np.column_stack((I0, I1))
	plt.imshow(I)
	plt.savefig('./COMBINE/'+name+'.png')
	plt.close()