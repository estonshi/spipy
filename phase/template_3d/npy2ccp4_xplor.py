from iotbx import ccp4_map
import iotbx.xplor.map as xplor_map
from scitbx.array_family import flex
from cctbx import uctbx, sgtbx, maptbx
import numpy as np
import sys
from scipy import ndimage

def ccp4_map_type(map, N, radius,file_name='map.ccp4'):
  grid = flex.grid(N*2+1, N*2+1,N*2+1)
  map.reshape( grid )
  ccp4_map.write_ccp4_map(
      file_name=file_name,
      unit_cell=uctbx.unit_cell(" %s"%(radius*2.0)*3+"90 90 90"),
      space_group=sgtbx.space_group_info("P1").group(),
      gridding_first=(0,0,0),
      gridding_last=(N*2, N*2, N*2),
      map_data=map,
      labels=flex.std_string(["generated from zernike moments"]))

def xplor_map_type(m, N, radius, file_name='map.xplor'):
  gridding = xplor_map.gridding( [N*2+1]*3, [0]*3, [2*N]*3)
  grid = flex.grid(N*2+1, N*2+1,N*2+1)
  m.reshape( grid )
  uc = uctbx.unit_cell(" %s"%(radius*2.0)*3+"90 90 90")
  xplor_map.writer( file_name, ['no title lines'],uc, gridding,m)

def mat2pdb(mat, cpos, shift, files='map.pdb'):
  x,y,z = np.where(mat>0)
  #intens = mat[x,y,z]
  #r = np.sqrt(np.max((x-cpos)**2+(y-cpos)**2+(z-cpos)**2))
  x = shift+(x-cpos)#/r*rmax+cpos
  y = shift+(y-cpos)#/r*rmax+cpos
  z = shift+(z-cpos)#/r*rmax+cpos
  xyz = np.vstack((x,y,z)).T
  res_id = 1
  out = open(files, 'w')
  for ind,cor in enumerate(xyz):
    if ind==1000 or ind==1500 or ind==2000:
        print>>out, "ATOM  %5d  CB  GLU  %4d    %8.3f%8.3f%8.3f%6.2f%6.2f"%(res_id, res_id, cor[0], cor[1], cor[2], 1, 1)
        continue
    print>>out, "ATOM  %5d  CA  ASP  %4d    %8.3f%8.3f%8.3f%6.2f%6.2f"%(res_id, res_id, cor[0], cor[1], cor[2], 1, 1)
    res_id = res_id + 1
  out.close()


if __name__ == "__main__":
  filename = sys.argv[1]
  T_type = sys.argv[2]
  data = np.load(filename)
  #data = data[100:265,100:265,100:265]
  #data = ndimage.zoom(data,0.2)
  try:
    thre_s = float(sys.argv[3])
    thre_l = float(sys.argv[4])
    data[data>thre_l] = 0
    data[data<thre_s] = 0
    data[data>=thre_s] = 1
    print('-> binary.',data.sum())
  except:
    print('-> continuous.')

  (N,N,N)=np.shape(data)  
  data1 = flex.double( data.flatten('F') )
  print N, type(N)
  N = int( N/2 )
  radius = 20
  if T_type=='ccp4':
      out_name = filename+'.ccp4'
      ccp4_map_type( data1, N, radius, file_name=out_name )
  elif T_type=='xplor':
      out_name = filename+'.xplor'
      xplor_map_type( data1, N,radius, out_name )
  elif T_type=='pdb':
      out_name = filename+'.pdb'
      mat2pdb(data,N,radius,out_name)

