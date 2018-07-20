import numpy as np

def help(module):
	if module=="cal_q":
		print("This function is used to calculate frequency unit q, also defined by (k'-k)/2pi")
		print("    -> q = 2*sin(theta/2)/lamda")
		print("    -> Input: detd (mm) , lamda (A), det_r (pixel), pixsize (mm)")
		print("    -> Output: q ( array, shape=(det_r,) )")
		return
	elif module=="cal_q_pat":
		print("This function is used to calculate frequency unit q of a pattern, also defined by (k'-k)/2pi")
		print("    -> q = 2*sin(theta/2)/lamda")
		print("    -> Input: detd (mm) , lamda (A), pixsize (mm)")
		print("              det_size ( detector size in pixels, list [x-size, y-size] )")
		print("      option: center ( center of pattern, in pixels, list [Cx, Cy], default=None and use det_size/2 as center )")
		print("    -> Output: q ( array, shape = det_r )")
		return
	elif module=="cal_r":
		print("This function is the inverse calculation of cal_q")
		print("    -> Input: qlist (numpy.ndarray, shape=(Nr,))")
		print("              detd (mm) , lamda (A), det_r (pixel), pixsize (mm)")
		print("    -> Output: det_r ( array, shape=qlist.shape )")
		return
	elif module=="oversamp_rate":
		print("This function calculates oversampling rate")
		print("    -> Input: sample_size ( diameter of your experiment sample (nm) )")
		print("              detd (mm) , lamda (A), pixsize (mm)")
		print("    -> Output: float")
	else:
		raise ValueError("No module names "+str(module))

# calculate frequency unit q, also named by k'-k
# q = 2*sin(theta/2)/lamda
def cal_q(detd, lamda, det_r, pixsize):
	# input: detd (mm) , lamda (A), det_r (pixel), pixsize (mm)
	lamda = lamda/10.0
	r = np.arange(det_r).astype(float) * pixsize
	theta = np.arctan(r/detd)
	q = 2*np.sin(theta/2.0)/lamda
	return q

def cal_q_pat(detd, lamda, pixsize, det_size, center=None):
	if center is None:
		center = np.array(det_size)/2.0
	lamda = lamda/10.0
	if len(det_size) == 2:
		x, y = np.indices(det_size)
		x = x - center[0]
		y = y - center[1]
		r = np.sqrt(x**2 + y**2)
	elif len(det_size) == 3:
		x, y, z = np.indices(det_size)
		x = x - center[0]
		y = y - center[1]
		z = z - center[2]
		r = np.sqrt(x**2 + y**2 + z**2)
	else:
		raise ValueError('Input det_size/center is not valid')
	theta = np.arctan(r/detd)
	q = 2*np.sin(theta/2.0)/lamda
	return q

def cal_r(qlist, detd, lamda, det_r, pixsize):
	lamda = lamda/10.0
	theta = 2*np.arcsin(qlist*lamda/2.0)
	rlist = np.tan(theta)*detd
	return rlist

def oversamp_rate(sample_size, detd, lamda, pixsize):
	q0 = cal_q(detd, lamda, 2, pixsize)[1]
	return 1.0/(sample_size*q0)