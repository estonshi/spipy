from spipy.phase import phase3d
import sys

if __name__=="__main__":

	params_essential = {'input|shape' : '125,125,125', 'input|padd_to_pow2' : True, \
		'input|inner_mask' : 4, 'input|outer_mask' : 55, \
		'input|outer_outer_mask' : 64, 'input|mask_edges' : True, \
		'phasing|repeats' : 20, 'phasing|iters' : '200ERA 200DM 300ERA', \
		'phasing_parameters|support_size' : 2000}
	params_optional = {'input|subtract_percentile' : None, 'input|spherical_support' : None, \
		'phasing_parameters|background' : 'True'}

	print("\nCreate new project ...")
	phase3d.new_project(data_path='3dvolume.bin', path='./', name='mytest')

	print("\nConfiguring ...")
	parameters = dict(params_essential, **params_optional)
	phase3d.config(params = parameters)

	print("Start !")
	phase3d.run(num_proc=4,nohup=False,cluster=True)

	sys.exit(0)
	phase3d.use_project('./mytest')
	phase3d.show_result(outpath=None, exp_param='581,7.9,128,0.3')
