def radial_profile_2d(data, center=None):
	import numpy as np
	if type(data)!=np.ndarray or center is None:
		print("This function is used to do averaged radial integration of an image")
		print("    -> Input: data (input image, numpy.ndarray, shape = (Nx, Ny))")
		print("              center (the zero point of your radial profile)")
		return

	x, y = np.indices((data.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	r = r.astype(np.int)

	tbin = np.bincount(r.ravel(), data.ravel())
	nr = np.bincount(r.ravel())
	radialprofile = tbin / nr
	return radialprofile

def radial_profile_3d(data, center=None):
	import numpy as np
	if type(data)!=np.ndarray or center==None:
		print("This function is used to do averaged radial integration of an volume")
		print("    -> Input: data (input image, numpy.ndarray, shape = (Nx, Ny, Nz))")
		print("              center (the zero point of your radial profile)")
		return

	x, y, z = np.indices((data.shape))
	r = np.sqrt((x-center[0])**2 + (y-center[1])**2 +(z-center[2])**2)
	r = r.astype(np.int)

	tbin = np.bincount(r.ravel(),data.ravel())
	nr = np.bincount(r.ravel())
	radialprofile = tbin/nr
	return radialprofile

def shells_2d(rads, data_shape=None, center=None):
	import numpy as np
	if type(rads)==str and rads=="help":
		print("This function returns indices in a pattern which forms a shell/circle when radius=rads")
		print("    -> Input: rads (int/float list[], a set of radius in pixels)")
		print("              data_shape (int turple, (sizex, sizey))")
		print("              center (data center, int/float turple/list, [cx, cy])")
		print("    -> Return: re (list, [shell1(numpy.ndarray,shape=(N1,2)), shell2(numpy.ndarray,shape=(N2,2)), ...])")
		return
	x, y = np.indices(data_shape)
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	re = []
	for rad in rads:
		selx ,sely = np.where(np.abs(r-rad)<0.5)
		re.append(np.vstack((selx,sely)).T)
	return re

def shells_3d(rads, data_shape=None, center=None):
	import numpy as np
	if type(rads)==str and rads=="help":
		print("This function returns indices in a pattern which forms a shell when radius=rads")
		print("    -> Input: rads (int/float list, a set of radius in pixels)")
		print("              data_shape (int turple, (sizex, sizey, sizez))")
		print("              center (data center, int/float turple/list, [cx, cy, cz])")
		print("    -> Return: re (list, [ shell1(numpy.ndarray), shell2(numpy.ndarray), ...])")
		return
	x, y, z = np.indices(data_shape)
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
	re = []
	for rad in rads:
		selx ,sely, selz = np.where(np.abs(r-rad)<0.5)
		re.append(np.vstack((selx,sely,selz)).T)
	return re

def radp_norm_2d(ref_Iq, data=None, center=None):
	import numpy as np
	if type(ref_Iq)==str and ref_Iq=="help":
		print("This function normalize pattern intensities (averaged inside r shells) using a given radial profile")
		print("    -> Input: ref_Iq (Reference radial intensity profile, numpy.ndarray, shape=(Nr,))")
		print("              data (pattern 2, numpy.ndarray, shape=(Nx,Ny)")
		print("              center (center of data, shape=[Cx,Cy])")
		print("[Notice] zeros point of ref_Iq locates on the center of input data")
		return
	x, y = np.indices((data.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	r = r.astype(np.int)
	tbin = np.bincount(r.ravel(),data.ravel())
	nr = np.bincount(r.ravel())
	radialprofile = tbin/nr

	norm_factor = np.zeros(radialprofile.shape)
	normed_len = min(len(ref_Iq), len(radialprofile))
	stop_rad = max(np.where(np.cumsum(ref_Iq)>1e-2)[0][0], np.where(np.cumsum(radialprofile)>1e-2)[0][0])
	norm_factor[stop_rad:normed_len] = ref_Iq[stop_rad:normed_len]/radialprofile[stop_rad:normed_len]

	newdata = np.zeros(data.shape)
	for ind,rad in enumerate(np.arange(r.min(), r.max()+1)):
		newdata[np.where(r==rad)] = data[np.where(r==rad)] * norm_factor[ind]
	return newdata

def radp_norm_3d(ref_Iq, data=None, center=None):
	import numpy as np
	if type(ref_Iq)==str and ref_Iq=="help":
		print("This function normalize volume intensities (averaged inside r shells) using a given radial profile")
		print("    -> Input: ref_Iq (Reference radial intensity profile, numpy.ndarray, shape=(Nr,))")
		print("              data (pattern 2, numpy.ndarray, shape=(Nx,Ny,Nz)")
		print("              center (center of data, shape=[Cx,Cy,Cz])")
		print("[Notice] While in normalization, zeros point of ref_Iq are forced to locate on the center of input data")
		return
	x, y, z = np.indices((data.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
	r = r.astype(np.int)
	tbin = np.bincount(r.ravel(),data.ravel())
	nr = np.bincount(r.ravel())
	radialprofile = tbin/nr

	norm_factor = np.zeros(radialprofile.shape)
	normed_len = min(len(ref_Iq), len(radialprofile))
	stop_rad = max(np.where(np.cumsum(ref_Iq)>1e-2)[0][0], np.where(np.cumsum(radialprofile)>1e-2)[0][0])
	norm_factor[stop_rad:normed_len] = ref_Iq[stop_rad:normed_len]/radialprofile[stop_rad:normed_len]

	newdata = np.zeros(data.shape)
	for ind,rad in enumerate(np.arange(r.min(), r.max()+1)):
		newdata[np.where(r==rad)] = data[np.where(r==rad)] * norm_factor[ind]
	return newdata

