import numpy as np

def help(module):
	if module=="radial_profile_2d":
		print("This function is used to do averaged radial integration of an image")
		print("    -> Input: data (input image, numpy.ndarray, shape = (Nx, Ny))")
		print("              center (the zero point of your radial profile)")
		print("     #option: mask ( 0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		print("    -> Return: radial_p ( two columns array, first column is radius (pixel) and the second one is radial profile)")
		return
	elif module=="radial_profile_3d":
		print("This function is used to do averaged radial integration of an volume")
		print("    -> Input: data (input image, numpy.ndarray, shape = (Nx, Ny, Nz))")
		print("              center (the zero point of your radial profile)")
		print("     #option: mask ( 0/1 binary pattern, shape=(Nx, Ny, Nz), 1 means masked area, 0 means useful area, default=None)")
		print("    -> Return: radial_p ( two columns array, first column is radius (pixel) and the second one is radial profile)")
		return
	elif module=="shells_2d":
		print("This function returns indices in a pattern which forms a shell/circle when radius=rads")
		print("    -> Input: rads (int/float list[], a set of radius in pixels)")
		print("              data_shape (int turple, (sizex, sizey))")
		print("              center (data center, int/float turple/list, [cx, cy])")
		print("    -> Return: re (list, [shell1(numpy.ndarray,shape=(N1,2)), shell2(numpy.ndarray,shape=(N2,2)), ...])")
		return
	elif module=="shells_3d":
		print("This function returns indices in a pattern which forms a shell when radius=rads")
		print("    -> Input: rads (int/float list, a set of radius in pixels)")
		print("              data_shape (int turple, (sizex, sizey, sizez))")
		print("              center (data center, int/float turple/list, [cx, cy, cz])")
		print("    -> Return: re (list, [ shell1(numpy.ndarray), shell2(numpy.ndarray), ...])")
		return
	elif module=="radp_norm_2d":
		print("This function normalize pattern intensities (averaged inside r shells) using a given radial profile")
		print("    -> Input: ref_Iq (Reference radial intensity profile, numpy.ndarray, shape=(Nr,))")
		print("              data (pattern 2, numpy.ndarray, shape=(Nx,Ny)")
		print("              center (center of data, shape=[Cx,Cy])")
		print("     #option: mask ( 0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		print("[Notice] zeros point of ref_Iq should locate on the center of input data")
		return
	elif module=="radp_norm_3d":
		print("This function normalize volume intensities (averaged inside r shells) using a given radial profile")
		print("    -> Input: ref_Iq (Reference radial intensity profile, numpy.ndarray, shape=(Nr,))")
		print("              data (pattern 2, numpy.ndarray, shape=(Nx,Ny,Nz)")
		print("              center (center of data, shape=[Cx,Cy,Cz])")
		print("     #option: mask ( 0/1 binary pattern, shape=(Nx, Ny, Nz), 1 means masked area, 0 means useful area, default=None)")
		print("[Notice] While in normalization, zeros point of ref_Iq are forced to locate on the center of input data")
		return
	elif module=="circle":
		print("This function generates a circle/sphere area with given radius, centered by origin of coordinates")
		print("    -> Input: data_shape (int, 2 or 3, output dimension)")
		print("              rad (float/int, radius)")
		print("    -> Return: index of points inside given radius ( numpy.array, shape=(Np,data_shape))")
	else:
		raise ValueError("No module names "+str(module))


def radial_profile_2d(data, center, mask=None):
	center = np.round(center)
	x, y = np.indices((data.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	r = r.astype(np.int)
	if mask is not None:
		maskdata = data * (1-mask)
	else:
		maskdata = data

	tbin = np.bincount(r.ravel(), maskdata.ravel())
	nr = np.bincount(r.ravel())
	if mask is not None:
		r_mask = np.zeros(r.shape)
		r_mask[np.where(mask==1)] = 1
		nr_mask = np.bincount(r.ravel(), r_mask.ravel())
		nr = nr - nr_mask
	radialprofile = np.zeros(len(nr))
	r_pixel = np.sort(list(set(r.ravel())))
	nomaskr = np.where(nr>0)
	radialprofile[nomaskr] = tbin[nomaskr] / nr[nomaskr]
	return np.vstack([r_pixel,radialprofile]).T

def radial_profile_3d(data, center, mask=None):
	center = np.round(center)
	x, y, z = np.indices((data.shape))
	r = np.sqrt((x-center[0])**2 + (y-center[1])**2 +(z-center[2])**2)
	r = r.astype(np.int)
	if mask is not None:
		maskdata = data * (1-mask)
	else:
		maskdata = data

	tbin = np.bincount(r.ravel(),maskdata.ravel())
	nr = np.bincount(r.ravel())
	if mask is not None:
		r_mask = np.zeros(r.shape)
		r_mask[np.where(mask==1)] = 1
		nr_mask = np.bincount(r.ravel(), r_mask.ravel())
		nr = nr - nr_mask
	radialprofile = np.zeros(len(nr))
	r_pixel = np.sort(list(set(r.ravel())))
	nomaskr = np.where(nr>0)
	radialprofile[nomaskr] = tbin[nomaskr] / nr[nomaskr]
	return np.vstack([r_pixel,radialprofile]).T

def shells_2d(rads, data_shape, center):
	x, y = np.indices(data_shape)
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	re = []
	for rad in rads:
		selx ,sely = np.where(np.abs(r-rad)<0.5)
		re.append(np.vstack((selx,sely)).T)
	return re

def shells_3d(rads, data_shape, center):
	x, y, z = np.indices(data_shape)
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
	re = []
	for rad in rads:
		selx ,sely, selz = np.where(np.abs(r-rad)<0.5)
		re.append(np.vstack((selx,sely,selz)).T)
	return re

def radp_norm_2d(ref_Iq, data, center, mask=None):
	center = np.round(center)
	x, y = np.indices((data.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	r = r.astype(np.int)
	if mask is not None:
		maskdata = data * (1-mask)
	else:
		maskdata = data

	tbin = np.bincount(r.ravel(), data.ravel())
	nr = np.bincount(r.ravel())
	if mask is not None:
		r_mask = np.zeros(r.shape)
		r_mask[np.where(mask==1)] = 1
		nr_mask = np.bincount(r.ravel(), r_mask.ravel())
		nr = nr - nr_mask
	radialprofile = np.zeros(len(nr))
	r_pixel = np.sort(list(set(r.ravel())))
	nomaskr = np.where(nr>0)
	radialprofile[nomaskr] = tbin[nomaskr] / nr[nomaskr]

	norm_factor = np.zeros(radialprofile.shape)
	normed_len = min(len(ref_Iq), len(radialprofile))
	stop_rad = max(np.where(np.cumsum(ref_Iq)>1e-2)[0][0], np.where(np.cumsum(radialprofile)>1e-2)[0][0])
	norm_factor[stop_rad:normed_len] = ref_Iq[stop_rad:normed_len]/radialprofile[stop_rad:normed_len]

	newdata = np.zeros(data.shape)
	for ind,rad in enumerate(np.arange(r.min(), r.max()+1)):
		newdata[np.where(r==rad)] = data[np.where(r==rad)] * norm_factor[ind]
	return newdata

def radp_norm_3d(ref_Iq, data, center, mask=None):
	center = np.round(center)
	x, y, z = np.indices((data.shape))
	r = np.sqrt((x-center[0])**2 + (y-center[1])**2 +(z-center[2])**2)
	r = r.astype(np.int)
	if mask is not None:
		maskdata = data * (1-mask)
	else:
		maskdata = data

	tbin = np.bincount(r.ravel(),maskdata.ravel())
	nr = np.bincount(r.ravel())
	if mask is not None:
		r_mask = np.zeros(r.shape)
		r_mask[np.where(mask==1)] = 1
		nr_mask = np.bincount(r.ravel(), r_mask.ravel())
		nr = nr - nr_mask
	radialprofile = np.zeros(len(nr))
	r_pixel = np.sort(list(set(r.ravel())))
	nomaskr = np.where(nr>0)
	radialprofile[nomaskr] = tbin[nomaskr] / nr[nomaskr]

	norm_factor = np.zeros(radialprofile.shape)
	normed_len = min(len(ref_Iq), len(radialprofile))
	stop_rad = max(np.where(np.cumsum(ref_Iq)>1e-2)[0][0], np.where(np.cumsum(radialprofile)>1e-2)[0][0])
	norm_factor[stop_rad:normed_len] = ref_Iq[stop_rad:normed_len]/radialprofile[stop_rad:normed_len]

	newdata = np.zeros(data.shape)
	for ind,rad in enumerate(np.arange(r.min(), r.max()+1)):
		newdata[np.where(r==rad)] = data[np.where(r==rad)] * norm_factor[ind]
	return newdata

def circle(data_shape, rad):
	dsize = data_shape
	if dsize==3:
		cube = [2*int(np.ceil(rad)) + 1]*3
		center = np.array([int(np.ceil(rad))]*3)
		x, y, z = np.indices(cube)
		r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
		selx, sely, selz = np.where(r < rad)
		return np.vstack((selx,sely,selz)).T - center
	elif dsize==2:
		cube = [2*int(np.ceil(rad)) + 1]*2
		center = np.array([int(np.ceil(rad))]*2)
		x, y = np.indices(cube)
		r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
		selx, sely = np.where(r < rad)
		return np.vstack((selx,sely)).T - center
	else:
		raise RuntimeError("Only support 2d or 3d input!")
