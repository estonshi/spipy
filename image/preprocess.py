import sys
import numpy as np
sys.path.append(__file__.split("/image/preprocess.py")[0] + "/analyse")
import saxs
import radp

def help(module):
	if module=="fix_artifact":
		print("This function reduces artifacts of an adu dataset, whose patterns share the same artifacts")
		print("    -> Input: dataset (FLOAT adu patterns, numpy.ndarray, shape=(Nd,Nx,Ny))")
		print("              estimated_center (estimated pattern center, (Cx,Cy))")
		print("              artifacts (artifact location in pattern, numpy.ndarray, shape=(Na,2))")
		print("     *option: mask (mask area of patterns, 0/1 numpy.ndarray where 1 means masked, shape=(Nx,Ny), default=None)")
		print("    -> Return: newdataset (To save RAM, your input dataset is modified directly)")
		print("[Notice] This function cannot reduce backgroud noise, try preprocess.adu2photon instead")
		print("Help exit.")
		return
	elif module=="adu2photon":
		print("This function is used to evaluate adu value per photon and transfer adu to photon")
		print("    -> Input: dataset ( patterns of adu values, numpy.ndarray, shape=(Nd,Nx,Ny) )")
		print("     *option: mask ( masked area in patterns, shape=(Nx,Ny), a 0/1 2d-array where 1 means masked point, default=None )")
		print("     *option: photon_percent ( estimated percent of pixels that has photons, default=0.1)")
		print("     *option: nproc ( number of processes running in parallel, default=2)")
		print("     *option: transfer ( bool, Ture -> evaluate adu unit and transfer to photon, False -> just evaluate, default=True)")
		print("     *option: force_poisson ( bool, whether to determine photon numbers at each pixel according to poisson distribution, default=False, ignored if transfer=False )")
		print("    -> Return: adu (float) or [adu, data_photonCount] ( [float, int numpy.ndarray(Nd,Nx,Ny)] )")
		print("[Notice] This function is implemented with multi-processes. Nd is recommened to be >1k")
		print("Help exit.")
		return
	elif module=="hit_find":
		print("This function is used for hit finding, based on chi-square score. High score means hit")
		print("    -> Input: dataset ( raw patterns for intput, numpy.ndarray, shape=(Nd,Nx,Ny) )")
		print("              background ( averaged running background pattern, numpy.ndarray, shape=(Nx,Ny))")
		print("              radii_range ( radii of annular area used for hit-finding, list/array, [inner_r, outer_r], unit=pixel)")
		print("     *option: mask (mask area of patterns, 0/1 numpy.ndarray where 1 means masked, shape=(Nx,Ny), default=None)")
		print("     *option: cut_off ( chi-square cut-off, positve int/float, default=None and a mix-gaussian analysis is used for clustering)")
		print("    -> Return: label ( 0/1 array, 1 stands for hit, the same order with 'dataset' )")
		print("[Notice] if use cut_off=None, it's better for input dataset to contain over 100 patterns")
	else:
		raise ValueError("No module names "+str(module))

def _detect_artifact():
	pass

def fix_artifact(dataset, estimated_center, artifacts, mask=None):

	if estimated_center is None or artifacts is None:
		raise RuntimeError("no estimated_center or artifacts")
	try:
		dataset[0, artifacts[:,0], artifacts[:,1]]
	except:
		raise RuntimeError("Your input artifacts is not valid")

	print("\nAnalysing artifact locations ...")
	dataset[np.isnan(dataset)] = 0
	dataset[np.isinf(dataset)] = 0
	dataset = np.abs(dataset)
	powder = saxs.cal_saxs(dataset)
	center = np.array(saxs.frediel_search(powder, estimated_center, mask))
	inv_art_loc = 2*center - artifacts
	print("Data center : " + str(center))
	# whether inv_art_loc exceed pattern size
	normal_inv_art_loc = (inv_art_loc[:,0]<powder.shape[0]).astype(int) & (inv_art_loc[:,0]>=0).astype(int) \
		& (inv_art_loc[:,1]<powder.shape[1]).astype(int) & (inv_art_loc[:,1]>=0).astype(int)
	# whether a pair of artifact points is symmetried by center point
	art_pat = np.zeros(powder.shape)
	art_pat[artifacts] = 1
	pair_inv_art_loc_index = np.where(art_pat[inv_art_loc[:,0],inv_art_loc[:,1]]==1)[0]
	# whether inv_art_loc is in mask area
	if mask is not None:
		mask_inv_art_loc_index = np.where(mask[inv_art_loc[:,0],inv_art_loc[:,1]]==1)[0]
	else:
		mask_inv_art_loc_index = None
	# normal and unique locations
	print("Fix normal artifacts ...")
	normal_inv_art_loc[pair_inv_art_loc_index] = 0
	if mask is not None:
		normal_inv_art_loc[mask_inv_art_loc_index] = 0
	uniq_inv_art_loc = 1 - normal_inv_art_loc
	normal_artifacts = np.where(normal_inv_art_loc==1)[0]
	uniq_artifacts = np.where(uniq_inv_art_loc==1)[0]
	# fix artifacts at normal locations
	dataset[:, artifacts[normal_artifacts,0], artifacts[normal_artifacts,1]] = \
			dataset[:, inv_art_loc[normal_artifacts,0], inv_art_loc[normal_artifacts,1]]
	# fix artifacts at unique locations
	print("Fix unique artifacts ...")
	for loc in artifacts[uniq_inv_art_loc]:
		r = np.linalg.norm(loc)
		shell = radp.shells_2d([r], powder.shape, center)[0]
		mean_intens = np.mean(dataset[:, shell[:,0], shell[:,1]], axis=1)
		dataset[:, loc[0], loc[1]] = mean_intens
	return dataset

def adu2photon(dataset, mask=None, photon_percent=0.1, nproc=2, transfer=True, force_poisson=False):

	print("\nEvaluating adu units ...")
	no_photon_percent = 1 - photon_percent
	dataset[np.isnan(dataset)] = 0
	dataset[np.isinf(dataset)] = 0
	dataset = np.abs(dataset)
	if mask is not None:
		mindex = np.where(mask==1)
		dataset[:,mindex[0],mindex[1]] = 0
	powder = saxs.cal_saxs(dataset)
	countp = np.bincount(np.round(powder.ravel()).astype(int))
	if mask is not None:
		countp[0] = countp[0] - len(np.where(mask==1))
	sumc = np.cumsum(countp)
	percentc = sumc/sumc[-1].astype(float)
	adu = np.where(np.abs(percentc-no_photon_percent)<0.01)[0][0]
	print("Estimated adu value is " + str(adu) + ". Done.\n")

	if transfer:
		import multiprocessing as mp
		print("Transferring adu patterns to photon count patterns ...")
		result = []
		partition = range(0, len(dataset), np.ceil(len(dataset)/float(nproc)).astype(int))
		if len(partition)==nproc:
			partition.append(len(dataset))
		pool = mp.Pool(processes = nproc)
		for i in np.arange(nproc):
			data_part = dataset[partition[i]:partition[i+1]]
			result.append(pool.apply_async(_transfer, args=(data_part,no_photon_percent,adu,force_poisson,)))
			print("Start process " + str(i) + " .")
		pool.close()
		pool.join()
		out = np.zeros(dataset.shape, dtype='i4')
		for ind,p in enumerate(result):
			out[partition[ind]:partition[ind+1]] = p.get()
		print("Done.\n")
		return adu, out
	else:
		return adu

def _transfer(data, no_photon_percent, adu, force_poisson):

	def poisson(lamb):
		return np.random.poisson(lamb,1)[0]

	if data == []:
		return np.array([])
	re = np.zeros(data.shape, dtype='i4')
	for ind,pat in enumerate(data):
		countp = np.bincount(np.round(pat.ravel()).astype(int))
		sumc = np.cumsum(countp)
		percentc = sumc/sumc[-1].astype(float)
		adu_mine = np.where(np.abs(percentc-no_photon_percent)<0.01)[0][0]
		real_adu = 0.6*adu_mine + 0.4*adu
		if force_poisson:
			newp = np.frompyfunc(poisson,1,1)
			re[ind] = newp(pat/real_adu)
		else:
			newp = np.round(pat/real_adu).astype(int)
			re[ind] = newp
	return re

def hit_find(dataset, background, radii_range, mask=None, cut_off=None):
	dataset[np.isnan(dataset)] = 0
	dataset[np.isinf(dataset)] = 0
	dataset = np.abs(dataset)
	background[np.isnan(background)] = 0
	background[np.isinf(background)] = 0
	background = np.abs(background)
	if mask is not None:
		maskdataset = dataset * (1-mask)
		maskbackground = background * (1 - mask)
	else:
		maskdataset = dataset
		maskbackground = background
	dsize = maskdataset.shape
	if len(dsize)!=3 or background.shape!=dsize[1:]:
		raise RuntimeError("Input a set of 2d patterns! background should have the same shape with input!")
	center = saxs.frediel_search(saxs.cal_saxs(maskdataset), np.array(dsize[1:])/2, mask)
	inner_shell = radp.circle(2, radii_range[0]) + np.array(center)
	outer_shell = radp.circle(2, radii_range[1]) + np.array(center)
	shell = np.zeros(dsize[1:])
	shell[outer_shell[:,0], outer_shell[:,1]] = 1
	shell[inner_shell[:,0], inner_shell[:,1]] = 0
	# calculate chi square
	chi = np.zeros((dsize[0],1))
	for ind,p in enumerate(maskdataset):
		chi[ind,0] = np.sum( (p - maskbackground)**2 * shell ) / np.sum( (maskbackground - np.mean(maskbackground))**2 * shell)
	# predict
	if type(cut_off)==float or type(cut_off)==int:
		# cut-off
		label = np.zeros(dsize[0])
		label[np.where(chi>cut_off)[0]] = 1
	else:
		# clustering
		from sklearn import mixture
		clust = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(chi)
		label = clust.predict(chi)
		if np.mean(chi[np.where(label==0)[0]]) > np.mean(chi[np.where(label==1)[0]]):
			label = 1 - label
	return label
