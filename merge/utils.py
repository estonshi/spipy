import numpy as np
from spipy.image import quat
import numexpr as ne
from scipy.linalg import get_blas_funcs
import sys


def help(module):
	if module == "get_slice":
		print("Generate slices from a 3D model according to given quaternions (orientations)")
		print("    -> Input : model ( the model, a 3D numpy array )")
		print("               quaternions ( a set of quaternions, np.array([[w,qx,qy,qz],...]) ) ")
		print("               det_size ( the size of generated patterns (in pixels), [Npx, Npy] )")
		print("     *option : det_center ( the center of generated patterns (in pixels), default=None and the geometry center is used )")
		print("     *option : mask ( pattern mask , 2d numpy array where 1 means masked area and 0 means useful area, default is None )")
		print("    -> Output: slices, shape=(Nq, Npx, Npy) , Nq is the number of quaternions")
	elif module == "merge_slice":
		print("Merge sereval slices into a 3D model according to given quaternions (orientations)")
		print("    -> Input : model ( the model, a 3D numpy array )")
		print("               quaternions ( a set of quaternions, np.array([[w,qx,qy,qz],...]) ) ")
		print("               slices ( slices to merge into model, numpy array, shape=(N,Npx,Npy) )")
		print("     *option : weights ( inital interpolation weights for every pixel of input model, shape=model.shape, default is None and weights=ones is used )")
		print("     *option : det_center ( the center of generated patterns (in pixels), default=None and the geometry center is used )")
		print("     *option : mask ( pattern mask , 2d numpy array where 1 means masked area and 0 means useful area, default is None )")
		print("    -> Output: None, model is modified directly")
	elif module == "poisson_likelihood":
		print("Calculate poisson likelihood between a model slice and exp pattern")
		print("    -> Input : W_j ( model slice in orientation j, numpy 1d/2d array, do masking in advance )")
		print("               K_k ( experiment pattern, numpy 1d/2d array, do masking in advance )")
		print("     *option : beta ( float, suggested values are from 1 to 50, default=1 )")
		print("     *option : weight ( float, the weight of orientation j, if orientations are not strictly uniformly sampled, default is None )")
		print("    -> Output: float, R_jk = weight * Product{W_j**K_k*exp(-W_j)} ** beta")
	elif module == "maximization":
		print("Calculate updated slice of orientation j after likelihood maximization")
		print("    -> Input : K_ks ( all useful experiment patterns, numpy array, shape=(N,Np), reshape pattern to array or do masking in advance !)")
		print("               Prob_ks ( probabilities of all useful patterns (after normalizing in every orientation) in orientation j, shape=(N,) )")
		print("    -> Output: W_j_prime, updated slice in orientation j (flattened), length = K_ks.shape[1]")



def get_slice(model, quaternions, det_size, det_center=None, mask=None):
	'''
	get one slice from a 3D matrix (model), whose orientation depends on a quaternion
	model : 3d numpy array
	quaternions : 1d/2d array, [[w, qx, qy, qz],...], shape=(Nq, 4)
	det_size : [size_x, size_y]
	det_center : [cx, cy] or None, start from 0
	'''

	if len(np.array(quaternions).shape) == 1:
		quaternions = [quaternions]

	# make slice
	this_slice = np.zeros( (len(quaternions),det_size[0]*det_size[1]), dtype=np.float32 )
	slice_x, slice_y = np.mgrid[0:det_size[0], 0:det_size[1]]
	slice_z = np.zeros(det_size)
	if det_center is None:
		det_center = (np.array(det_size)-1)/2.0
	slice_x = slice_x - det_center[0]
	slice_y = slice_y - det_center[1]
	slice_coor_ori = np.vstack([slice_x.flatten(), slice_y.flatten(), slice_z.flatten()])   # shape=(3,N),  N=det_size[0]*det_size[1]
	# make mask
	if mask is not None:
		this_mask = mask.flatten()
		masked_index = np.where(this_mask == 0)[0]
	maxR = np.linalg.norm( [ max(det_size[0]-det_center[0]-1, det_center[0]), max(det_size[1]-det_center[1]-1, det_center[1]) ] )
	del slice_x, slice_y, slice_z

	for ind, quaternion in enumerate(quaternions):
		# make rotation
		rot_mat = np.array(quat.quat2rot(quaternion))   # dtype is np.matrix, shape=(3,3)
		gemm = get_blas_funcs("gemm",[rot_mat, slice_coor_ori])
		slice_coor = gemm(1, rot_mat, slice_coor_ori)
		slice_coor += np.reshape((np.array(model.shape)-1)/2.0, (3,1))  # np.array, shape=(3,N)

		# drop pixels which are out of bound
		drop = 0
		if mask is None:
			if np.ceil(maxR) > np.floor((min(model.shape)-1)/2.0):
				slice_index = np.where( (slice_coor[0]>=0) & (slice_coor[0]<=model.shape[0]-1) & \
										(slice_coor[1]>=0) & (slice_coor[1]<=model.shape[1]-1) & \
										(slice_coor[2]>=0) & (slice_coor[2]<=model.shape[2]-1) )[0]
				slice_coor = slice_coor[:, slice_index]       # shape=(3,N')
				drop = 1
		else:
			if np.ceil(maxR) > np.floor((min(model.shape)-1)/2.0):
				slice_index = np.where( (slice_coor[0]>=0) & (slice_coor[0]<=model.shape[0]-1) & \
										(slice_coor[1]>=0) & (slice_coor[1]<=model.shape[1]-1) & \
										(slice_coor[2]>=0) & (slice_coor[2]<=model.shape[2]-1) & \
										(this_mask==0))[0]
			else:
				slice_index = masked_index
			slice_coor = slice_coor[:, slice_index]       # shape=(3,N')
			drop = 1

		# interpolate
		slice_neighbor = np.zeros(slice_coor.shape, dtype=int)
		weights = np.zeros(slice_coor.shape[1])
		if drop:
			temp_slice = np.zeros(len(slice_index))
		else:
			temp_slice = np.zeros(det_size[0]*det_size[1])
		for i in range(8):
			if i==0:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==1:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==2:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==3:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==4:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==5:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==6:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==7:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			temp = -np.linalg.norm(slice_neighbor - slice_coor, axis=0)/0.3
			w = ne.evaluate('exp(temp)')
			weights += w
			temp_slice += model[slice_neighbor[0], slice_neighbor[1], slice_neighbor[2]] * w

		if drop:
			this_slice[ind][slice_index] = temp_slice / weights
		else:
			this_slice[ind] = temp_slice / weights

	if len(np.array(quaternions).shape) == 1:
		return this_slice.reshape((ind+1, det_size[0], det_size[1]))[0]
	else:
		return this_slice.reshape((ind+1, det_size[0], det_size[1]))



def merge_slice(model, quaternions, slices, weights=None, det_center=None, mask=None):
	'''
	merge slices into a given model, the orientations depend on quaternions
	model : 3d numpy array, original model
	quaternions : array, [[w, qx, qy, qz],...]
	this_slice : 3d numpy array, some patterns
	weights : same shape with model, initial interpolation weights of all voxels in the model
	det_center : [cx, cy] or None
	[NOTICE] "model" is modified directly, no return
	'''

	if len(np.array(quaternions).shape) == 1:
		quaternions = [quaternions]
	if len(np.array(slices).shape) == 2:
		slices = np.array([slices])

	# make slice
	slices_flat = slices.reshape((slices.shape[0], slices.shape[1]*slices.shape[2]))
	det_size = slices.shape[1:]
	slice_x, slice_y = np.mgrid[0:det_size[0], 0:det_size[1]]
	slice_z = np.zeros(det_size)

	if det_center is None:
		det_center = (np.array(det_size)-1)/2.0
	if weights is None:
		weights = np.ones(model.shape, dtype=np.float32)
	# make mask
	if mask is not None:
		this_mask = mask.flatten()
		masked_index = np.where(this_mask == 0)[0]

	slice_x = slice_x - det_center[0]
	slice_y = slice_y - det_center[1]
	slice_coor_ori = np.vstack([slice_x.flatten(), slice_y.flatten(), slice_z.flatten()])   # shape=(3,N)
	maxR = np.linalg.norm( [ max(det_size[0]-det_center[0]-1, det_center[0]), max(det_size[1]-det_center[1]-1, det_center[1]) ] )
	del slice_x, slice_y, slice_z

	for ind, quaternion in enumerate(quaternions):
		# make rotation
		rot_mat = np.array(quat.quat2rot(quaternion))   # dtype is np.matrix, shape=(3,3)
		gemm = get_blas_funcs("gemm",[rot_mat, slice_coor_ori])
		slice_coor = gemm(1, rot_mat, slice_coor_ori)
		slice_coor += np.reshape((np.array(model.shape)-1)/2.0, (3,1))  # np.array, shape=(3,N)

		# drop pixels which are out of bound
		if mask is None:
			if np.ceil(maxR) > np.floor((min(model.shape)-1)/2.0):
				slice_index = np.where( (slice_coor[0]>=0) & (slice_coor[0]<=model.shape[0]-1) & \
										(slice_coor[1]>=0) & (slice_coor[1]<=model.shape[1]-1) & \
										(slice_coor[2]>=0) & (slice_coor[2]<=model.shape[2]-1) )[0]
				slice_coor = slice_coor[:, slice_index]       # shape=(3,N')
				this_slice_flat = slices_flat[ind][slice_index]  # shape=(N',)
		else:
			if np.ceil(maxR) > np.floor((min(model.shape)-1)/2.0):
				slice_index = np.where( (slice_coor[0]>=0) & (slice_coor[0]<=model.shape[0]-1) & \
										(slice_coor[1]>=0) & (slice_coor[1]<=model.shape[1]-1) & \
										(slice_coor[2]>=0) & (slice_coor[2]<=model.shape[2]-1) & \
										(this_mask==0))[0]
			else:
				slice_index = masked_index
			slice_coor = slice_coor[:, slice_index]       # shape=(3,N')
			this_slice_flat = slices_flat[ind][slice_index]  # shape=(N',)

		# interpolate
		slice_neighbor = np.zeros(slice_coor.shape, dtype=int)
		for i in range(8):
			if i==0:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==1:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==2:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==3:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==4:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==5:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==6:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==7:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)

			temp = -np.linalg.norm(slice_neighbor - slice_coor, axis=0)
			w = ne.evaluate("exp(temp/0.3)")
			
			temp_weight = weights[slice_neighbor[0], slice_neighbor[1], slice_neighbor[2]]
			temp_model = model[slice_neighbor[0], slice_neighbor[1], slice_neighbor[2]]
			temp_model = ne.evaluate('temp_model * temp_weight + this_slice_flat * w')
			temp_weight = temp_weight + w
			temp_model = ne.evaluate('temp_model / temp_weight')
			weights[slice_neighbor[0], slice_neighbor[1], slice_neighbor[2]] = temp_weight
			model[slice_neighbor[0], slice_neighbor[1], slice_neighbor[2]] = temp_model



def poisson_likelihood(W_j, K_k, beta=1, weight=None):
	'''
	calculate poisson likelihood R_jk between model slice (W_j) and experimental pattern (K_k)

	Do masking ahead of using this function.
	Final return is the value weight*(R_jk^beta)
	'''

	temp = ne.evaluate('sum(K_k*log(W_j)-W_j)')/np.product(W_j.shape)
	if weight is not None:
		R_jk = ne.evaluate('exp(temp*beta)*weight')
	else:
		R_jk = ne.evaluate('exp(temp*beta)')

	return float(R_jk)



def maximization(K_ks, Prob_ks):
	'''
	Calculate updated tomograph of one orientation and return

	Input : K_ks , all useful patterns, please reshape pattern to array or do masking in advance
			Prob_ks , probabilities of all useful patterns (after normalizing in every orientation) in orientation j
	'''

	assert len(K_ks.shape) == 2, "please reshape pattern to array or do masking in advance"

	Prob_norm = (Prob_ks/sum(Prob_ks)).reshape((len(Prob_ks),1))
	W_j_prime = ne.evaluate('sum(K_ks * Prob_norm, axis=0)')

	return W_j_prime




