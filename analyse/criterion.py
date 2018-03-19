import numpy as np
import sys
from spipy.image import radp

def help(module):
	if module=="r_factor":
		print("This function is used to calculate overall r-factor of two models")
		print("    -> Input: F_cal ( voxel models from calculation, 3d numpy.ndarray )")
		print("              F_obs ( voxel models from observation, 3d numpy.ndarray )")
		print("    -> Output: r-factor ( float )")
		return
	elif module=="r_factor_shell":
		print("This function is used to calculate r-factor in shells between two models")
		print("    -> Input: F_cal ( voxel models from calculation, 3d numpy.ndarray )")
		print("              F_obs ( voxel models from obversation, 3d numpy.ndarray )")
		print("              rlist ( list/array, radius of shells)")
		print("    -> Output: r-factor ( np.array, the same shape with rlist )")
		print("[NOTICE] (Size_x/2, Size_y/2, Size_z/2) is used as the center of input matrix")
		return
	elif module=="fsc":
		print("Calculate Fourier Shell Correlation between two models (in frequency space)")
		print("    -> Input: F1 ( the first voxel model, 3d numpy.array )")
		print("              F2 ( the second voxel model, 3d numpy.array)")
		print("              rlist ( list/array, radius of shells)")
		print("    -> Output: fsc ( np.array, the same shape with rlist )")
		return
	elif module=="r_split":
		print("Calculate r-split factors between two models (in frequency space)")
		print("    -> Input: F1 ( the first voxel model, 3d numpy.array )")
		print("              F2 ( the second voxel model, 3d numpy.array)")
		print("              rlist ( list/array, radius of shells)")
		print("    -> Output: rs ( np.array, the same shape with rlist )")
	else:
		raise ValueError("No module names "+str(module))

def r_factor(F_cal, F_obs):
	if F_cal.shape != F_obs.shape:
		raise RuntimeError("F1 and F2 should be in the same size!")
	return np.sum(np.abs(np.abs(F_obs) - np.abs(F_cal))) / np.sum(np.abs(F_obs))

def r_factor_shell(F_cal, F_obs, rlist):
	if F_cal.shape != F_obs.shape:
		raise RuntimeError("F1 and F2 should be in the same size!")
	size = np.array(F_cal.shape)
	center = (size-1)/2.0
	shells = radp.shells_3d(rlist, size, center)
	Rf = np.zeros(len(rlist))
	for ind,shell in enumerate(shells):
		shell_f_cal = F_cal[shell[:,0],shell[:,1],shell[:,2]] + 1e-15
		shell_f_obs = F_obs[shell[:,0],shell[:,1],shell[:,2]] + 1e-15
		Rf[ind] = np.sum(np.abs(np.abs(shell_f_obs) - np.abs(shell_f_cal))) / np.sum(np.abs(shell_f_obs))
	return Rf

def fsc(F1, F2, rlist):
	if F1.shape != F2.shape:
		raise RuntimeError("F1 and F2 should be in the same size!")
	size = np.array(F1.shape)
	center = (size-1)/2.0
	shells = radp.shells_3d(rlist, size, center)
	FSC = np.zeros(len(rlist))
	for ind,shell in enumerate(shells):
		shell_f1 = F1[shell[:,0],shell[:,1],shell[:,2]] + 1e-15
		shell_f2 = F2[shell[:,0],shell[:,1],shell[:,2]] + 1e-15
		up = np.sum( shell_f1 * np.conj(shell_f2) )
		down = np.sqrt( np.sum( np.abs(shell_f1)**2 ) * np.sum( np.abs(shell_f2)**2 ) )
		FSC[ind] = np.abs(up) / down
	return FSC

def r_split(F1, F2, rlist):
	if F1.shape != F2.shape:
		raise RuntimeError("F1 and F2 should be in the same size!")
	size = np.array(F1.shape)
	center = (size-1)/2.0
	shells = radp.shells_3d(rlist, size, center)
	rs = np.zeros(len(rlist))
	for ind,shell in enumerate(shells):
		shell_f1 = F1[shell[:,0],shell[:,1],shell[:,2]] + 1e-15
		shell_f2 = F2[shell[:,0],shell[:,1],shell[:,2]] + 1e-15
		rs[ind] = 2 * np.sum(np.abs(np.abs(F1) - np.abs(F2))) / np.sum(np.abs(F1) + np.abs(F2))
	return rs
