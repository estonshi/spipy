from spipy.phase import phase2d
import sys

if __name__=="__main__":

	params_essential = {'input|shape' : '123,123', 'input|padd_to_pow2' : True, \
		'input|inner_mask' : 6, 'input|outer_mask' : 64, \
		'input|outer_outer_mask' : None, 'input|mask_edges' : True, \
		'phasing|repeats' : 40, 'phasing|iters' : '300ERA 200DM 300ERA', \
		'phasing_parameters|support_size' : 200}
	params_optional = {'input|subtract_percentile' : None, 'input|spherical_support' : None, \
		'phasing_parameters|background' : 'True'}

	print("\nCreate new project ...")
	phase2d.new_project(data_mask_path=['pattern.bin','pat_mask.npy'], path='./', name=None)

	print("\nConfiguring ...")
	parameters = dict(params_essential, **params_optional)
	phase2d.config(params = parameters)

	print("Start !")
	phase2d.run(nohup=True)

	sys.exit(0)
	phase2d.show_result(outpath=None, exp_param='581,7.9,128,0.3')
