import numpy as np
from spipy.simulate import sim_adu

if __name__ == '__main__':
	pdb="./1N0U1.pdb"
	config_param = {'parameters|detd' : 200, 'parameters|lambda' : 2.5, \
	'parameters|detsize' : 128, 'parameters|pixsize' : 0.3, \
	'parameters|stoprad' : 6, 'parameters|polarization' : 'x', \
	'make_data|num_data' : 10, 'make_data|fluence' : 1e14, \
	'make_data|scatter_factor' : True}
	euler_range = np.array([[0,np.pi/2.0],[0,np.pi/2.0],[0,np.pi/2.0]])
	'''
	sim_adu.single_process(pdb_file=pdb, param=config_param, \
		euler_mode='helix', euler_order='zxz', euler_range=euler_range, predefined=None, save_dir='./')
	'''
	sim_adu.multi_process(save_dir='./', pdb_file=pdb, param=config_param, \
		euler_mode='random', euler_order='zxz', euler_range=euler_range, predefined=None, njobs=2)
	