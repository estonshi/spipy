#!/usr/bin/env python

work_dir = None
config_default = {'parameters|detd' : 200, 'parameters|lambda' : 2.5, \
                  'parameters|detsize' : 128, 'parameters|pixsize' : 0.3, \
                  'parameters|stoprad' : 0, 'parameters|polarization' : 'x', \
                  'make_data|num_data' : 100, 'make_data|fluence' : 1e14}

def generate_config_files(pdb_file, workdir = None, params = {}):
    import numpy as np
    import os
    import subprocess
    import argparse
    import logging
    import ConfigParser
    import sys
    global work_dir
    if type(pdb_file)!=str or pdb_file=="help":
        print("This function is used to configure simulation parameters")
        print("    -> Input: pdb_file (Path of your pdb file used in simulation [/..../xx.pdb])")
        print("     *option: workdir (Choose your work directory, ABSOLUTE PATH !)")
        print("     *option: params (A dict, {'section_1|param_1': value_1, ...})\n")
        print("[Notice] Give 'workdir' parameter a None value to use current directory")
        print("[Notice] If your 'params' is a empty dict, a default config file will be generated : ")
        print(config_default)
        print("\n---- Meanings of Parameters ----")
        print("'parameters|detd' : distance between sample and detector [unit : mm]")
        print("'parameters|lambda' : wave length of laser [unit : angstrom]")
        print("'parameters|detsize' : detector size in width/height [unit : pixel]")
        print("'parameters|pixsize' : pixel size of detector [unit : mm]")
        print("'parameters|stoprad' : radius of a circle region at the center of pattern that to be masked out [unit : pixel]")
        print("'parameters|polarization' : correction due to incident beam polarization, value from 'x', 'y' or 'none'")
        print("'make_data|num_data' : how many patterns do you want to generate")
        print("'make_data|fluence' : laser fluence [unit : photons/mm^2] (usually 1e10 ~ 1e14 is reasonable for most proteins)")
        return
    if not os.path.exists(pdb_file):
        print("Error!\nGive me a valid pdb file.\nExit.")
        return
    if workdir!=None and workdir[0]!='/':
        print("Error!\nUse ABSOLUTE PATH for workdir.\nExit.")
        return
    elif workdir!=None and not os.path.exists(workdir):
        print("Error!\nGive me a valid work directory.(Do not end with '/')\nExit.")
        return
    # copy config.ini to work dir
    work_dir = os.path.abspath(sys.path[0])
    path = __file__.split('/sim.py')[0] + '/config.ini'
    if workdir!=None:
        work_dir = workdir
    config = ConfigParser.ConfigParser()
    config.read(path)
    if params!={}:
        for key in params.keys():
            section, par = key.split('|')
            config.set(section, par, params[key])
    else:
        print("\n Generate default config file. Please run generate_config_file('help') to see details.")
    config.set("make_densities","pdb_code", pdb_file.split('/')[-1].split('.')[0])
    with open(work_dir + '/config.ini', 'w') as f:
        config.write(f)
    # copy /aux dir to work dir
    path = __file__.split('/sim.py')[0] + '/aux'
    cmd = "cp -r " + path + " " + work_dir
    subprocess.call(cmd, shell=True)
    cmd = "cp " + pdb_file + " " + work_dir + "/aux"
    subprocess.call(cmd, shell=True)
    # copy other files to work dir
    path = __file__.split('/sim.py')[0] + '/code/*'
    cmd = "cp -r " + path + " " + work_dir
    subprocess.call(cmd, shell=True)
    # mkdir
    cmd = "mkdir " + work_dir + "/data"
    subprocess.call(cmd, shell=True)
    # compile make_data
    print("\nPlease run 'sim.run_simulation()' after generation.")
    path = __file__.split('/sim.py')[0] + '/src'
    cmd = "cp -r " + path + " " + work_dir
    subprocess.call(cmd, shell=True)
    subprocess.call(work_dir + '/src/compile.sh', shell=True)

def run_simulation(help = None):
    import numpy as np
    import os
    import subprocess
    import argparse
    import logging
    import ConfigParser
    import sys
    global work_dir
    if help:
        print("This function is used to start simulation")
        print("    -> No input required")
        return
    if work_dir==None:
        print("Please run 'sim.generate_config_files(pdb_file, workdir = None, params = {})' first !")
        return
    
    config = ConfigParser.ConfigParser()
    config.read(work_dir + '/config.ini')
    save_pdb = work_dir + '/data/' + config.get("make_densities","pdb_code")
    pat_num = config.get("make_data","num_data")
    cmd = 'python ' + work_dir + '/sim_setup.py'
    subprocess.call(cmd, shell=True)
    cmd = 'python ' + work_dir + '/py_src/read_emc.py ' + save_pdb + " " + pat_num
    subprocess.call(cmd, shell=True)
    # delete tmp file
    cmd = 'rm ' + work_dir + '/data/densityMap.bin ' + work_dir + '/data/det_sim.dat '\
        + work_dir + '/data/intensities.bin ' + work_dir + '/data/photons.emc '\
        + work_dir + '/data/quaternion_buffer'
    subprocess.call(cmd, shell=True)

    print("\n== Complete ! ==")
    print("The simulation results are stored in '" + save_pdb + ".h5'")
