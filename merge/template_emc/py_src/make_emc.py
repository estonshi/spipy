# transfer experimental images to emc file
import numpy as np
import h5py
import sys


if __name__ == '__main__':
	
	print('--> Read data...')

	readfile = sys.argv[1]
	path = sys.argv[2]
	savepath = sys.argv[3]

	"""
	########### IMPORTANT ###########
	image_size = (260,257)                   # input pattern shape
	num_data = 'all'                          # input pattern number
	readfile = './taubin46.h5'                  # input h5 file
	mask_file = ''
	if_preprocess = 0
	########### IMPORTANT ###########
	"""
	# Read the original exp data file
	expfile = h5py.File(readfile,'r')
	exp = expfile[path][...]
	expfile.close()
	if not (exp.dtype == np.dtype('int64') or exp.dtype == np.dtype('int32')):
		raise ValueError("The data type of input file is incorrect. It should be 'int64' or 'int32'")
	image_size = exp[0].shape

	print("--> Transfer data...")

	# function to get photon events and locations
	def photondata(expd):
		one_photon_events = []
		multi_photon_events = []
		multi_photon_loca = []
		multi_photon_counts = []
		for d in expd:
			index = np.where(d==1)
			one_photon_events.append(len(index[0]))
			index2 = np.where(d>1)
			multi_photon_events.append(len(index2[0]))
		print('first loop ends.')
		one_photon_loca = np.zeros(np.sum(one_photon_events),dtype='i4')
		multi_photon_loca = np.zeros(np.sum(multi_photon_events),dtype='i4')
		multi_photon_counts = np.zeros(np.sum(multi_photon_events),dtype='i4')
		index1_all = 0
		index2_all = 0
		for i,d in enumerate(expd):
			index1 = np.where(d==1)
			one_photon_loca[index1_all:index1_all+len(index1[0])] = index1[0]*image_size[1]+index1[1]
			index1_all += len(index1[0])
			index2 = np.where(d>1)
			multi_photon_loca[index2_all:index2_all+len(index2[0])] = index2[0]*image_size[1]+index2[1]
			multi_photon_counts[index2_all:index2_all+len(index2[0])] = d[index2]
			index2_all += len(index2[0])
			if np.mod(i+1,1000)==0:
				print(i+1)
		one_photon_events = np.array(one_photon_events,dtype='i4')
		one_photon_loca = one_photon_loca.astype('i4')
		multi_photon_events = np.array(multi_photon_events,dtype='i4')
		multi_photon_loca = multi_photon_loca.astype('i4')
		multi_photon_counts = multi_photon_counts.astype('i4')
		print('second loop ends.')
		return [one_photon_events,one_photon_loca,multi_photon_events,multi_photon_loca,multi_photon_counts]

	# calculate
	[one_photon_events,one_photon_loca,multi_photon_events,multi_photon_loca,multi_photon_counts] = photondata(exp)

	# write emc file
	newfile = open(savepath,'wb')
	num_ = np.zeros(256,dtype='i4')
	num_[0] = len(exp)
	num_[1] = image_size[0]*image_size[1]
	num_.tofile(newfile)
	one_photon_events.tofile(newfile)
	multi_photon_events.tofile(newfile)
	one_photon_loca.tofile(newfile)
	multi_photon_loca.tofile(newfile)
	multi_photon_counts.tofile(newfile)
	newfile.close()
	print('Transfer ends.')
