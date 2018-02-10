_workpath = None

def use_project(project_path):
	global _workpath
	if type(project_path)!=str or project_path=="help":
		print("This function is used to switch to a existing project")
		print("    -> Input: project_path ( str, the project directory you want to switch to)")
		return
	import os
	import sys
	temp = None
	if project_path[0] == '/' or project_path[0:2] == '~/':
		temp = os.path.abspath(project_path)
		if os.path.exists(temp):
			_workpath = temp
		else:
			raise ValueError("The project " + temp + " doesn't exists. Exit")
	else:
		nowfolder = os.path.abspath(sys.path[0])
		temp = os.path.join(nowfolder, project_path)
		if os.path.exists(temp):
			_workpath = os.path.abspath(temp)
		else:
			raise ValueError("The project " + temp + " doesn't exists. Exit")

def new_project(data_path, path=None, name=None):
	global _workpath
	import sys
	import os
	if type(data_path)!=str or data_path == "help":
		print("This function is used to create a new project directory at your given path")
		print("    -> Input: data_path (path of your original intensity data)")
		print("     *option: path (create work directory at your give path, default as current dir)")
		print("     *option: name (give a name to your project, default is an number)")
		print("[Notice] Your original intensity file should be 3D matrix '.npy' or '.mat', or Dragonfly output '.bin'")
		print("[Notice] 'path' must be absolute path !")
		return
	import subprocess
	import numpy as np
	code_path = __file__.split('/phase3d.py')[0]
	if not os.path.exists(data_path):
		raise ValueError("\nYour data path is incorrect. Try ABSOLUTE PATH. Exit\n")
	if path == None or path == "./":
		path = os.path.abspath(sys.path[0])
	else:
		if not os.path.exists(path):
			raise ValueError('\n Your path is incorrect. Try ABSOLUTE PATH. Exit\n')
		else:
			path = os.path.abspath(path)
	if name is not None:
		_workpath = os.path.join(path, name)
	else:
		all_dirs = os.listdir(path)
		nid = 0
		for di in all_dirs:
			if di[0:8] == "phase3d_" and str.isdigit(di[8:]):
				nid = max(nid, int(di[8:]))
		nid += 1
		_workpath = os.path.join(path, 'phase3d_' + format(nid, '02d'))
	cmd = code_path + '/template_3d/new_project ' + _workpath
	subprocess.check_call(cmd, shell=True)
	# now change output|path in config.ini
	import ConfigParser
	config = ConfigParser.ConfigParser()
	config.read(os.path.join(_workpath, 'config.ini'))
	config.set('output', 'path', _workpath)
	config.set('input', 'fnam', os.path.join(_workpath,'data.bin'))
	with open(os.path.join(_workpath, 'config.ini'), 'w') as f:
		config.write(f)
	# now load data
	if data_path.split('.')[-1] == 'npy':
		data = np.load(data_path)
		data.tofile(_workpath+'/ori_intens/intensity.bin')
	elif data_path.split('.')[-1] == 'bin':
		cmd = 'cp ' + data_path + ' ' + _workpath + '/ori_intens/intensity.bin'
		subprocess.check_call(cmd, shell=True)
	elif data_path.split('.')[-1] == 'mat':
		import scipy.io as sio
		dfile = sio.loadmat(data_path)
		data = dfile.values()[0]
	else:
		raise ValueError('\n Error while loading your data ! Exit\n')
	cmd = 'ln -fs ' + _workpath + '/ori_intens/intensity.bin ' + _workpath + '/data.bin'
	subprocess.check_call(cmd, shell=True)
	print("\nAll work done ! ")
	print("Now please confirm running parameters. Your can re-edit it by calling function phase3d.config(...) or eidt config.ini directly.\n")

def config(params):
	global _workpath
	if params == {} or type(params)!=dict:
		print("This function is used to edit configure file")
		print("    -> Input (dict, parameters yout want to modified.)")
		print("params format : ")
		print("    {\n\
					'input|shape' : '120,120,120', \n\
					'input|padd_to_pow2' : 'True', \n\
					... \n\
					}")
		print("You can look into 'config.ini' for detail information")
		return
	import os
	if not os.path.exists(os.path.join(_workpath,'config.ini')):
		raise ValueError("I can't find your configure file, please run phase3d.new_project(...) first !")
	import ConfigParser
	config = ConfigParser.ConfigParser()
	config.read(os.path.join(_workpath,'config.ini'))
	for k in params.keys():
		section, par = k.split("|")
		config.set(section, par, params[k])
	with open(os.path.join(_workpath,'config.ini'), 'w') as f:
		config.write(f)
	print('\n Configure finished.')

def run(num_proc=1, nohup=False, cluster=True):
	global _workpath
	if type(num_proc)!=int:
		print("Call this function to start phasing")
		print("    -> *option: num_proc (int, how many processes to run in parallel, default=1)")
		print("       *option: nohup (bool, whether run it in the background, default=False)")
		print("       *option: cluster (bool, whether you will submit jobs using job scheduling system, if yes, the function will only generate a command file at your work path without submitting it, and ignore nohup value. default=True))")
		return
	import os
	import subprocess
	if not os.path.exists(os.path.join(_workpath,'config.ini')):
		raise ValueError("Please call phase3d.new_project(...) and phase3d.config(...) first ! Exit")
	import sys
	code_path = __file__.split('/phase3d.py')[0] + '/template_3d'
	if nohup == True:
		cmd = "python " + os.path.join(code_path,'make_input.py') + ' '+ os.path.join(_workpath,'config.ini') + ' >' + os.path.join(_workpath,'make_input.log')
	else:
		cmd = "python " + os.path.join(code_path,'make_input.py') + ' '+ os.path.join(_workpath,'config.ini')
	subprocess.check_call(cmd, shell=True)
	if nohup == True:
		cmd = "python " + os.path.join(code_path, 'phase.py') + ' ' + os.path.join(_workpath, 'input.h5') + ' ' + str(num_proc) + ' &>' + os.path.join(_workpath, 'phase.log')+'&'
	else:
		cmd = "python " + os.path.join(code_path, 'phase.py') + ' ' + os.path.join(_workpath, 'input.h5') + ' ' + str(num_proc)
	if cluster:
		print("\n Dry run on cluster, check submit_job.sh for details.\n")
		submitfile = open(os.path.join(_workpath, "submit_job.sh"), 'w')
		submitfile.write("#! /bin/bash\n\n")
		submitfile.write("# Submit the command below to your job submitting system to run 3d phasing\n")
		submitfile.write(cmd + '\n')
		submitfile.close()
	else:
		subprocess.check_call(cmd, shell=True)

def show_result(outpath=None, exp_param=None):
	global _workpath
	if type(outpath)==str and outpath == "help":
		print("This function is used to plot phasing results in a figure")
		print("    -> Input: ")
		print("     *option: outpath (IF you move output.h5 to another folder, please give me its path)")
		print("     *option: exp_param (list detd, lambda, det_size, pix_size in a string. Used to calculate q value.")
		print("                         e.g. '200,2.5,128,0.3'. If you don't need q info, leave it as default (None))")
		return
	if outpath is not None and type(outpath)!=str:
		raise ValueError("Input 'outpath should be a string. Exit'")
	import sys
	import os
	import subprocess
	code_path = __file__.split('/phase3d.py')[0] + '/template_3d'

	if outpath is None:
		cmd = "python " + os.path.join(code_path, 'show_result.py') + ' ' + os.path.join(_workpath, 'output.h5')
	else:
		cmd = "python " + os.path.join(code_path, 'show_result.py') + ' ' + outpath
	if exp_param is not None:
		cmd = cmd + ' ' + str(exp_param)

	subprocess.check_call(cmd, shell=True)
