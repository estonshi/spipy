import mrcfile
import numpy as np
import os

def help(module):
	if module=="writeccp4":
		print("Use this function to write volume data into ccp4/mrc format")
		print("    -> Input: volume ( volume data, shape=(Nx,Ny,Nz) )")
		print("              save_file ( output file, str, '...(path)/filename.ccp4' )")
		print("    -> Output: None")
	elif module=="readccp4":
		print("Read ccp4/mrc files")
		print("    -> Input: file_path ( path of file )")
		print("    -> Output: data ( a dict, {'volume':numpy.3darray, 'header':object} )")
	else:
		raise ValueError("No module names "+str(module))


def writeccp4(volume, save_file):
	save_path = os.path.dirname(save_file)
	if not os.path.exists(save_path):
		raise ValueError("Directory '" + save_path is "' invalid !")
	if not len(volume.shape)==3:
		raise ValueError("Input volume should be 3D data !")

	vold = np.nan_to_num(volume)
	vold = np.float32(vold)
	with mrcfile.new(save_file) as mrcf:
		mrcf.set_data(vold)
		mrcf.update_header_from_data()

def readccp4(file_path):
	if not os.path.exists(file_path):
		raise ValueError("Your file path is invalid !")

	data = {'volume':None, 'header':None}
	with mrcfile.open(file_path) as mrcf:
		data['volume'] = mrcf.data
		data['header'] = mrcf.header
	return data