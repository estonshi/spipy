import numpy as np

def help(module):
	if module=="cal_q":
		print("This function is used to calculate frequency unit q, also defined by k'-k")
		print("    -> q = 2*sin(theta/2)/lamda")
		print("    -> Input: detd (mm) , lamda (A), det_r (pixel), pixsize (mm)")
		return
	elif module=="cal_r":
		print("This function is inverse calculation of cal_q")
		print("    -> Input : qlist (numpy.ndarray, shape=(Nr,))")
		print("               detd (mm) , lamda (A), det_r (pixel), pixsize (mm)")
		return
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

def cal_r(qlist, detd, lamda, det_r, pixsize):
	lamda = lamda/10.0
	theta = 2*np.arcsin(qlist*lamda/2.0)
	rlist = np.tan(theta)*detd
	return rlist
