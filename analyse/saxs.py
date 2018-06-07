import numpy as np
import sys
sys.path.append(__file__.split('/analyse/saxs.py')[0]+'/image/')

def help(module):
	if module=="grid":
		print("This finction is used to calculate grid of a pattern")
		print("    -> Input: input_patt (numpy.ndarray, shape = (Nx,Ny))")
		return
	elif module=="frediel_search":
		print("This function is used to find the zeros frequency point of an SPI pattern")
		print("    -> Input: pattern (none negative numpy.ndarray, shape=(Nx, Ny))")
		print("              estimated_center (estimated center of the pattern : (Cx, Cy), error in 20 pixels )")
		print("      option: mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		return
	elif module=="inten_profile_vaccurate":
		print("This finction is used to calculate accumulate intensity profile of SPI patterns")
		print("    -> Input: dataset (numpy.ndarray, shape=(Nd,Nx,Ny)) ")
		print("              mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area)")
		print("              *exp_param (detd (mm) , lamda (A), det_r (pixel), pixsize (mm))")
		print("[Notice] We don't recommend you to use this function as it is very slow. Use 'inten_profile_vfast' instead.")
		return
	elif module=="inten_profile_vfast":
		print("This finction is used to calculate accumulate intensity profile of SPI patterns")
		print("The patterns stored in .h5 file should be a ndarray (num, Nx, Ny)")
		print("    -> Input: dataset (numpy.ndarray, shape=(Nd,Nx,Ny)) ")
		print("              mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area)")
		print("              *exp_param (detd (mm) , lamda (A), det_r (pixel), pixsize (mm))")
		return
	elif module=="cal_saxs":
		print("This finction is used to calculate the saxs pattern of an SPI data set")
		print("    -> Input: data (patterns, numpy.ndarray, shape=(Nd,Nx,Ny)) ")
		return
	elif module=="centering":
		print("This function shift the input pattern to its center. Output pattern is a part of original one.")
		print("    -> Input: pat (input pattern, a 2d numpy array, shape=(Nx,Ny))")
		print("              estimated_center (estimated center of original pattern, )")
		print("      option: mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		print("    -> Output: newpat ( output pattern, a 2d numpy array, shape=(Nx',Ny'))")
		print("               newmask ( ONLY if mask!=None, then the cut mask pattern will be output)")
	elif module=="particle_size":
		print("This function uses self-correlation transformation method to estimate particle size of saxs pattern")
		print("    -> Input: saxs (saxs pattern, numpy.2darray, shape=(Nx,Ny))")
		print("              estimated_center (the estimated center of saxs pattern)")
		print("     #option: exparam (experimetnal parameters, str, 'detector-sample distance(mm),lambda(A),pixel length(mm)', such as '578,7.9,0.3', default=None)")
		print("     #option: high_filter_cut (float between (0,1), which determine the FWHM of high pass filter, larger value means smaller filter width, default=0.3)")
		print("     #option: power (positive float 0~1, a power conducted on pattern to enhance the contribution of high q data, default=0.7)")
		print("     #option: mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		print("    -> Output: list, [particle_size(float), auto_correlation_radial_profile(2darray)]. particle size is not garunteed to be correct, \
			RECOMMOND to watch auto-correlation radial profile and find the locations of peaks, which are most trustable values of possible particle sizes. \
			auto_correlation_radial_profile is a 2-d array, where the 1st colum is particle size in nm (if exparam is given) and 2nd colum is the auto-correlation value.")
		print("[NOTICE] is your give exparam, the output particle_size is in nanometer; otherwise it is in pixels. Auto correlation radial profile is in pixels")
	else:
		raise ValueError("No module names "+str(module))


# generate grid from input pattern's shape
def grid(input_patt):	
	size = input_patt.shape
	if len(size)==2:
		return np.mgrid[0:size[0],0:size[1]]
	elif len(size)==3:
		return np.mgrid[0:size[0],0:size[1],0:size[2]]
	else:
		return

# frediel search of SPI pattern to find center
def frediel_search(pattern, estimated_center, mask=None):
	if mask is not None:
		maskpattern = pattern * (1 - mask)
	else:
		maskpattern = pattern
	size = np.array(maskpattern.shape)
	estimated_center = np.array(estimated_center).astype(int)
	search_z = [max(size[0]/20, 20), max(size[1]/20, 20)]
	searchzone_top = max(0, estimated_center[0]-search_z[0]/2)
	searchzone_bottom = min(size[0], estimated_center[0]+search_z[0]/2)
	searchzone_left = max(0, estimated_center[1]-search_z[1]/2)
	searchzone_right = min(size[1], estimated_center[1]+search_z[1]/2)
	search_zone = np.mgrid[searchzone_top : searchzone_bottom,\
					searchzone_left : searchzone_right]
	fred_z = [min(max(size[0]/2, 50),size[0]), min(max(size[1]/2, 50),size[1])]
	score = np.inf
	center = [0,0]
	for cen in search_zone.T.reshape(np.product(search_zone[0].shape),2):
		fred_zone = np.mgrid[cen[0]-fred_z[0]/2:cen[0]+fred_z[0]/2,cen[1]-fred_z[1]/2:cen[1]+fred_z[1]/2]
		inv_fred_zone = np.array([2*cen[0] - fred_zone[0], 2*cen[1] - fred_zone[1]])
		this_score = np.sum(np.abs(maskpattern[fred_zone[0],fred_zone[1]] - maskpattern[inv_fred_zone[0], inv_fred_zone[1]])\
					* (maskpattern[fred_zone[0],fred_zone[1]]>0).astype(float) * (maskpattern[inv_fred_zone[0],inv_fred_zone[1]]>0).astype(float))
		if score>this_score:
			center = cen
			score = this_score
	return center

# calculate accumulate intensity profile of SPI patterns
def inten_profile_vaccurate(dataset, mask, *exp_param):
	import q
	if len(exp_param)<4:
		raise ValueError("Please be sure to give all exp_param ! Exit")
	qinfo = q.cal_q(exp_param[0], exp_param[1], exp_param[2]*4, exp_param[3]/4.0)
	data = dataset
	num = data.shape[0]
	size = data.shape[1:]
	intens = []
	rofq = np.inf
	import radp
	import scipy.ndimage.interpolation as ndint
	if mask is not None:
		newmask = np.round(ndint.zoom(mask, 4)).astype(int)
	else:
		newmask = None
	for ind,pat in enumerate(data):
		newpat = ndint.zoom(pat, 4)
		newcenter = frediel_search(newpat, [size[0]*4, size[1]*4], mask)
		intens_one = radp.radial_profile_2d(newpat, newcenter, newmask)
		intens.append(intens_one[:,1])
		rofq = min(rofq, len(intens_one[:,1]))
		sys.stdout.write(str(ind)+'/'+str(num)+' patterns\r')
		sys.stdout.flush()
	final = np.zeros(rofq)
	for p in intens:
		final += p[0:rofq]
	final = final/float(num)
	return np.vstack((qinfo,final[:len(qinfo)])).T

def inten_profile_vfast(dataset, mask, *exp_param):
	import q
	if len(exp_param)<4:
		raise ValueError("Please be sure to give all exp_param ! Exit")
	qinfo = q.cal_q(exp_param[0], exp_param[1], exp_param[2]*4, exp_param[3]/4.0)
	saxs = cal_saxs(dataset)
	center = frediel_search(saxs, [saxs.shape[0]/2, saxs.shape[1]/2], mask)
	import radp
	import scipy.ndimage.interpolation as ndint
	newsaxs = newpat = ndint.zoom(saxs, 4)
	if mask is not None:
		newmask = np.round(ndint.zoom(mask, 4)).astype(int)
	else:
		newmask = None
	newcenter = center * 4
	intens = radp.radial_profile_2d(newsaxs, newcenter, newmask)
	return np.vstack((qinfo,intens[0:len(qinfo),1])).T

# calculate the saxs pattern of an SPI data set
def cal_saxs(data):
	return np.sum(data, axis=0)/float(len(data))

def centering(pat, estimated_center, mask=None):
	center = frediel_search(pat, estimated_center, mask)
	ori_shape = pat.shape
	c2left = center[1]
	c2right = ori_shape[1] - center[1] - 1
	c2top = center[0]
	c2bottom = ori_shape[0] - center[0] - 1
	if c2right>c2left:
		left = 0
		right = center[1] + c2left + 1
	else:
		right = ori_shape[1]
		left = center[1] - c2right
	if c2top>c2bottom:
		bottom = ori_shape[0]
		top = center[0] - c2bottom
	else:
		top = 0
		bottom = center[0] + c2top + 1
	err = (bottom - top) - (right - left)
	if err>0:
		bottom = bottom - err/2
		top = top + err/2
	elif err<0:
		right = right + err/2
		left = left - err/2
	else:
		pass
	if mask is None:
		return pat[top:bottom,left:right]
	else:
		return pat[top:bottom,left:right], mask[top:bottom,left:right]

def particle_size(saxs, estimated_center, exparam=None, high_filter_cut=0.3, power=0.7, mask=None):
	import radp
	# high pass filter
	csaxs, cmask = centering(saxs, estimated_center, mask)
	center = np.array(csaxs.shape)/2
	Iq = radp.radial_profile_2d(csaxs, center, cmask)[:,1]
	cut = Iq.max()*high_filter_cut
	width = np.where(Iq>cut)[0][-1]
	x, y = grid(csaxs)
	r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
	filter_s = 1 - np.exp(-(r/width)**16)
	saxs_filtered = np.abs(np.fft.ifft2( np.fft.fftshift(np.fft.fftshift(np.fft.fft2(csaxs)) * filter_s) ))
	# auto correlation
	auto_coor = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(saxs_filtered**power))))
	# detect particle size
	radp_auto_coor = radp.radial_profile_2d(auto_coor, center)[:,1]
	# find peak
	derive = (radp_auto_coor[1:] - radp_auto_coor[:-1])/radp_auto_coor[:-1]
	peak = np.argmax(derive) + 1
	if type(exparam)==str:
		try:
			import q
			param = np.array(exparam.split(',')).astype(float)
			q = q.cal_q(param[0], param[1], len(radp_auto_coor), param[2])
			print("resolution : "+str(1.0/q[-1])+" nm")
			peak = peak * 1.0/q[-1]
			sizes = 1.0/q[-1] * np.arange(len(radp_auto_coor))
		except:
			print('error')
			peak = np.argmax(derive) + 1
			sizes = np.arange(len(radp_auto_coor))
	else:
		sizes = np.arange(len(radp_auto_coor))
	return [peak, np.vstack([sizes, radp_auto_coor]).T]
