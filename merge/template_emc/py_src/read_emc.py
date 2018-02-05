import numpy as np
import scipy.io as sio
import h5py
import sys

########### IMPORTANT ###########
# image_size = (128,128)                   # input pattern shape
# image_num = 100						   # total pattern number
# mask_file = ''
# savefile = '1aon'
########### IMPORTANT ###########

if __name__ == '__main__':
	readfile = __file__.split('read_emc.py')[0] + '../data/photons.emc'  # input emc file
	readquat = __file__.split('read_emc.py')[0] + '../data/quaternion_buffer' # input quaternion file
	readdet = __file__.split('read_emc.py')[0] + '../data/det_sim.dat'
	readdensity = __file__.split('read_emc.py')[0] + '../data/densityMap.bin'
	readinten = __file__.split('read_emc.py')[0] + '../data/intensities.bin'
	try:
		savefile = sys.argv[1]
		print('saving to '+savefile+'.h5')
		image_num = int(sys.argv[2])
		print('pattern num : '+str(image_num))
	except:
		raise ValueError('Input params required ! (save_path and image_num)')

	data = np.fromfile(readfile,dtype='i4')
	num_data = data[0]
	pixs = data[1]
	ext = np.zeros((image_num,pixs),dtype='i4')

	one_photon_events = data[256:num_data+256]
	multi_photon_events = data[num_data+256:num_data*2+256]

	total_mul_events = np.sum(multi_photon_events)
	total_one_events = np.sum(one_photon_events)
	pc_one = num_data*2+256
	pc_mul = pc_one+total_one_events
	pc_mul_counts = pc_one+total_one_events+total_mul_events
	for i in range(image_num):
		one_event = one_photon_events[i]
		one_loca = data[pc_one:pc_one+one_event]
		ext[i][one_loca] = 1
		pc_one += one_event

		multi_event = multi_photon_events[i]
		multi_loca = data[pc_mul:pc_mul+multi_event]
		multi_counts = data[pc_mul_counts:pc_mul_counts+multi_event]
		ext[i][multi_loca] = multi_counts
		pc_mul += multi_event
		pc_mul_counts += multi_event
	image_size = int(np.round(np.sqrt(pixs)))
	ext.shape = ((image_num,image_size,image_size))

	quat = np.fromfile(readquat,dtype='double')
	quat.shape = (image_num,4)

	density = np.fromfile(readdensity, dtype=float)
	density_size = np.int(np.round(len(density)**(1.0/3.0)))
	density.shape = (density_size, density_size, density_size)

	intensity = np.fromfile(readinten, dtype=float)
	inten_size = np.int(np.round(len(intensity)**(1.0/3.0)))
	intensity.shape = (inten_size, inten_size, inten_size)

	detector = np.loadtxt(readdet, skiprows=1)

	f = h5py.File(savefile+'.h5','w')
	f.create_dataset('patterns',data=ext)
	f.create_dataset('quaternions',data=quat)
	f.create_dataset('electron density',data=density)
	f.create_dataset('scattering intensity',data=intensity)
	f.create_dataset('detector mapping', data=detector)
	f.close()
