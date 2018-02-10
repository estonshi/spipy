# generate grid from input pattern's shape
def grid(input_patt):
	import numpy as np
	if type(input_patt)!=np.ndarray:
		print("This finction is used to calculate grid of a pattern")
		print("    -> Input: input_patt (numpy.ndarray, shape = (Nx,Ny))")
		return
	size = input_patt.shape
	if len(size)==2:
		return np.mgrid[0:size[0],0:size[1]]
	elif len(size)==3:
		return np.mgrid[0:size[0],0:size[1],0:size[2]]
	else:
		return

# frediel search of SPI pattern to find center
def frediel_search(pattern, estimated_center=None):
	import numpy as np
	if type(pattern)!=np.ndarray:
		print("This function is used to find the zeros frequency point of an SPI pattern")
		print("    -> Input: pattern (none negative numpy.ndarray, shape=(Nx, Ny))")
		print("              estimated_center (estimated center of the pattern : (Cx, Cy))")
		return
	size = np.array(pattern.shape)
	search_z = [max(size[0]/20, 6), max(size[1]/20, 6)]
	search_zone = np.mgrid[estimated_center[0]-search_z[0]/2:estimated_center[0]+search_z[0]/2,\
					estimated_center[1]-search_z[1]/2:estimated_center[1]+search_z[1]/2]
	fred_z = [min(max(size[0]/2, 50),size[0]), min(max(size[1]/2, 50),size[1])]
	score = np.inf
	center = [0,0]
	for cen in search_zone.T.reshape(np.product(search_zone[0].shape),2):
		fred_zone = np.mgrid[cen[0]-fred_z[0]/2:cen[0]+fred_z[0]/2,cen[1]-fred_z[1]/2:cen[1]+fred_z[1]/2]
		inv_fred_zone = np.array([2*cen[0] - fred_zone[0], 2*cen[1] - fred_zone[1]])
		this_score = np.sum(np.abs(pattern[fred_zone[0],fred_zone[1]] - pattern[inv_fred_zone[0], inv_fred_zone[1]])\
					* (pattern[fred_zone[0],fred_zone[1]]>0).astype(float) * (pattern[inv_fred_zone[0],inv_fred_zone[1]]>0).astype(float))
		if score>this_score:
			center = cen
			score = this_score
	return center

# calculate accumulate intensity profile of SPI patterns
def inten_profile_vaccurate(dataset, *exp_param):
	import numpy as np
	if type(dataset)!=str or dataset == 'help':
		print("This finction is used to calculate accumulate intensity profile of SPI patterns")
		print("    -> Input: dataset (numpy.ndarray, shape=(Nd,Nx,Ny)) ")
		print("              *exp_param (detd (mm) , lamda (A), det_r (pixel), pixsize (mm))")
		print("[Notice] We don't recommend you to use this function as it is very slow. Use 'inten_profile_vfast' instead.")
		return
	import h5py
	import sys
	sys.path.append(__file__.split('/analyse/saxs.py')[0]+'/image/')
	import q
	qinfo = q.cal_q(exp_param[0], exp_param[1], exp_param[2]*4, exp_param[3]/4.0)
	if not qinfo:
		raise ValueError("Please be sure to give all exp_param ! Exit")
	data = dataset
	num = data.shape[0]
	size = data.shape[1:]
	intens = []
	rofq = np.inf
	import radp
	import scipy.ndimage.interpolation as ndint
	for ind,pat in enumerate(data):
		newpat = ndint.zoom(pat, 4)
		newcenter = frediel_search(newpat, [size[0]*2, size[1]*2])
		intens_one = radp.radial_profile_2d(newpat, newcenter)
		intens.append(intens_one)
		rofq = min(rofq, len(intens_one))
		sys.stdout.write(str(ind)+'/'+str(num)+' patterns\r')
		sys.stdout.flush()
	final = np.zeros(rofq)
	for p in intens:
		final += p[0:rofq]
	final = final/float(num)
	return np.vstack((qinfo,final[:len(qinfo)])).T

def inten_profile_vfast(dataset, *exp_param):
	import numpy as np
	if type(file_h5)!=str or file_h5 == 'help':
		print("This finction is used to calculate accumulate intensity profile of SPI patterns")
		print("The patterns stored in .h5 file should be a ndarray (num, Nx, Ny)")
		print("    -> Input: dataset (numpy.ndarray, shape=(Nd,Nx,Ny)) ")
		print("              *exp_param (detd (mm) , lamda (A), det_r (pixel), pixsize (mm))")
		return
	import sys
	sys.path.append(__file__.split('/analyse/saxs.py')[0]+'/image/')
	import q
	qinfo = q.cal_q(exp_param[0], exp_param[1], exp_param[2]*4, exp_param[3]/4.0)
	if not qinfo:
		raise ValueError("Please be sure to give all exp_param ! Exit")
	import h5py
	saxs = cal_saxs(dataset)
	center = frediel_search(saxs, [saxs.shape[0]/2, saxs.shape[1]/2])
	import radp
	import scipy.ndimage.interpolation as ndint
	newsaxs = newpat = ndint.zoom(saxs, 4)
	newcenter = center * 4
	intens = radp.radial_profile_2d(newsaxs, newcenter)
	return np.vstack((qinfo,intens[:len(qinfo)])).T

# calculate the saxs pattern of an SPI data set
def cal_saxs(data):
	import numpy as np
	if type(data)!=np.ndarray:
		print("This finction is used to calculate the saxs pattern of an SPI data set")
		print("    -> Input: data (patterns, numpy.ndarray, shape=(Nd,Nx,Ny)) ")
		return
	return np.sum(data, axis=0)/float(len(data))
