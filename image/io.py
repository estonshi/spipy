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
	elif module=="pdb2density":
		print("Calculate voxel density map model according to a given pdb file")
		print("    -> Input: pdb_file ( str, the path of pdb file )")
		print("              resolution ( float, the resolution of density map, in Angstrom )")
		print("    -> Output: densitymap ( numpy.3darray, voxel model of electron density map )")
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
	with mrcfile.new(save_file, overwrite=True) as mrcf:
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

def _readpdb(pdb_file):
	# return [mass,x,y,z,0]
	from spipy.simulate.code import process_pdb
	from collections import OrderedDict
	atom_table_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simulate/aux/henke_table')

	atom_types = process_pdb.find_atom_types(pdb_file)
	atom_scatt = OrderedDict()
	atom_radii = OrderedDict()
	for atom in atom_types:
		index, mass = process_pdb.find_mass(atom_table_path, atom)
		atom_scatt[atom.upper()] = [index,mass]
	atoms = process_pdb.get_atom_coords(pdb_file, atom_scatt)
	return atoms

def pdb2density(pdb_file, resolution):
	from spipy.simulate.code import process_pdb
	atoms = _readpdb(pdb_file)
	# process symmetry
	(s_l, t_l)  = process_pdb.read_symmetry(pdb_file)
	if len(s_l)*len(t_l)>0:
		all_atoms = process_pdb.apply_symmetry(atoms, s_l, t_l)
	else:
		all_atoms = atoms
	# process density
	density = process_pdb.atoms_to_density_map(all_atoms, resolution)
	density = np.abs(process_pdb.low_pass_filter_density_map(density))
	# put it at center
	points = np.array(np.where(density>1))
	center = np.round(np.mean(points,axis=1)).astype(int)
	R = int(np.ceil(np.linalg.norm(points.T-center, axis=1).max()))
	ext_part = density[max(center[0]-R,0):min(center[0]+R,density.shape[0]),\
						max(center[1]-R,0):min(center[1]+R,density.shape[1]),\
						max(center[2]-R,0):min(center[2]+R,density.shape[2])]
	ext_cen = (np.array(ext_part.shape)+1)/2
	box = np.zeros((2*R+1,2*R+1,2*R+1))
	box[R-ext_cen[0]:R+ext_part.shape[0]-ext_cen[0],\
		R-ext_cen[1]:R+ext_part.shape[1]-ext_cen[1],\
		R-ext_cen[2]:R+ext_part.shape[2]-ext_cen[2]] = ext_part

	return box

class _CXIDB():

	def print_path(self, groups, depth):
		for g in groups:
			if str(type(g)).split('.')[-2]=="group":
				children = g.keys()
				self.print_path(children, depth+2)
			elif str(type(g)).split('.')[-2]=="dataset":
				print(" "*depth+"|--"+g)
			else:
				continue

	def parser(self, cxifile, stdout='std'):
		import sys
		import h5py
		f = h5py.File(cxifile,'r')
		groups = f
		depth = 2
		if stdout!='std':
			f = open(stdout,'w')
			oldstd = sys.stdout
			sys.stdout = f
		print(cxifile)
		self.print_path(groups, depth)
		if stdout!='std':
			sys.stdout = oldstd
		f.close()

def cxi_parser(cxifile, out='std'):
	cxidb = _CXIDB()
	cxidb.parser(cxifile, out)