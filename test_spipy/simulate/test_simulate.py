import sys
from spipy.simulate import sim

if __name__=="__main__":
	
	config_default = {'parameters|detd' : 200, 'parameters|lambda' : 1.0, \
                  'parameters|detsize' : 128, 'parameters|pixsize' : 0.3, \
                  'parameters|stoprad' : 5, 'parameters|polarization' : 'x', \
                  'make_data|num_data' : 100, 'make_data|fluence' : 1e16}

	print("Create new project ...")
	sim.generate_config_files(pdb_file='./Fe_nano.pdb', workpath=None, name='simu_test', params=config_default)
	
	print("Simulating ...")
	sim.run_simulation()
