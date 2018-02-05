import numpy as np
import h5py
import pyqtgraph as pg
import io_utils
import noise
import sys, os

sys.path.append(os.path.abspath('.'))
from src import era

if __name__ == '__main__':
    fnam = sys.argv[1]
    
    # read the h5 file 
    diff, diff_ret, support, support_ret, \
    good_pix, solid_unit, solid_units_ret, \
    emods, econs, efids, T, T_rav, B_rav     = io_utils.read_output_h5(fnam)
    
    pow = 1.0
    # display the radial average of the data
    diff_rav = noise.rad_av(diff * good_pix)
    plot = pg.plot(diff_rav**pow)

    # display the radial average of the retrieved diff
    diff_ret = np.abs(np.fft.fftn(solid_units_ret[0]))**2
    diff_ret_rav = noise.rad_av(diff_ret) + B_rav
    plot.plot(diff_rav**pow, pen=pg.mkPen('b'))

    # display the radial average of the background
    plot.plot(B_rav**pow, pen=pg.mkPen('g'))

    # display the radial average of the background
    plot.plot(noise.rad_av(diff_ret)**pow, pen=pg.mkPen('r'))

    
    ########### Make a large background filled with the radial average
    i = np.fft.fftfreq(diff.shape[0]) * diff.shape[0]
    j = np.fft.fftfreq(diff.shape[1]) * diff.shape[1]
    k = np.fft.fftfreq(diff.shape[2]) * diff.shape[2]
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    rs      = np.sqrt(i**2 + j**2 + k**2).astype(np.int16)
    
    rs = rs.ravel()
    background = B_rav[rs].reshape(diff.shape)


