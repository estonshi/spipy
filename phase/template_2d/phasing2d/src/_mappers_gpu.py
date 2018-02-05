import numpy as np
import afnumpy
import afnumpy.fft

from mappers import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

c = afnumpy.arrayfire.get_device_count()
afnumpy.arrayfire.set_device(rank % c)

class Mapper():
    
    def __init__(self, I, **args):
        modes = Modes()
        
        # check if there is a background
        if isValid('background', args):
            if args['background'] is True :
                modes['B'] = np.random.random((I.shape)).astype(args['dtype'])
            else :
                modes['B'] = np.sqrt(args['background']).astype(args['dtype'])
        
            modes['B'] = afnumpy.array(modes['B'])

        if isValid('O', args):
            modes['O'] = args['O']
        else :
            modes['O'] = np.random.random(I.shape).astype(args['c_dtype'])

        modes['O'] = afnumpy.array(modes['O'])

        # this is the radial value for every pixel 
        # in the volume
        self.rs    = None 
        
        self.mask = 1
        if isValid('mask', args):
            self.mask = args['mask']
            if args['mask'] is not 1 :
                self.mask = afnumpy.array(self.mask)

        self.alpha = 1.0e-10
        if isValid('alpha', args):
            self.alpha = args['alpha']

        if isValid('mask', args) :
            self.I_norm = (args['mask'] * I).sum()
        else :
            self.I_norm = I.sum()
        
        self.amp   = afnumpy.sqrt(afnumpy.array(I.astype(args['dtype'])))

        # define the data projection
        # --------------------------
        if 'B' in modes.keys() :
            self.Pmod = self.Pmod_back
        else :
            self.Pmod = self.Pmod_single
    
        # define the support projection
        # -----------------------------
        if isValid('voxel_number', args) :
            self.voxel_number = args['voxel_number']
        else :
            self.voxel_number = False
            self.S    = afnumpy.array(args['support'])
        
        self.support = None
        if isValid('support', args):
            self.support = afnumpy.array(args['support'])

        self.modes = modes

    def object(self, modes):
        return np.array(modes['O'])

    def Psup(self, modes):
        out = modes.copy()

        if self.voxel_number :
            O = out['O']
            self.S = choose_N_highest_pixels( (O * O.conj()).real, self.voxel_number, support = self.support)

        out['O'] *= self.S

        if 'B' in modes.keys() :
            back, self.rs, self.r_av = radial_symetry(np.array(out['B']), rs = self.rs)
            out['B'] = afnumpy.array(back)
        return out

    def Pmod_single(self, modes):
        out = modes.copy()
        out['O'] = pmod_single(self.amp, modes['O'], self.mask, alpha = self.alpha)
        return out
    
    def Pmod_back(self, modes):
        out = modes.copy()
        out['O'], out['B'] = pmod_back(self.amp, modes['B'], modes['O'], self.mask, alpha = self.alpha)
        return out

    def Imap(self, modes):
        O = afnumpy.fft.fftn(modes['O']) 
        if 'B' in modes.keys() :
            I = (O.conj() * O).real + modes['B']**2
        else :
            I = (O.conj() * O).real 
        return I
    
    def Emod(self, modes):
        M         = self.Imap(modes)
        eMod      = afnumpy.sum( self.mask * ( afnumpy.sqrt(M) - self.amp )**2 )
        eMod      = afnumpy.sqrt( eMod / self.I_norm )
        return eMod

    def finish(self, modes):
        out = {}
        out['support'] = np.array(self.S)
        out['I']       = np.array(self.Imap(modes))

        if 'B' in modes.keys() :
            out['background'] = np.array(modes['B']**2)
            out['r_av']       = self.r_av
        return out

    def l2norm(self, delta, array0):
        num = 0
        den = 0
        for k in delta.keys():
            num += afnumpy.sum( (delta[k] * delta[k].conj()).real ) 
            den += afnumpy.sum( (array0[k] * array0[k].conj()).real ) 
        return np.sqrt(num / den)


def pmod_single(amp, O, mask = 1, alpha = 1.0e-10):
    O = afnumpy.fft.fftn(O)
    O = Pmod_single(amp, O, mask = mask, alpha = alpha)
    O = afnumpy.fft.ifftn(O)
    return O
    
def Pmod_single(amp, O, mask = 1, alpha = 1.0e-10):
    out  = mask * O * amp / (afnumpy.abs(O) + alpha)
    out += (1 - mask) * O
    return out

def pmod_back(amp, background, O, mask = 1, alpha = 1.0e-10):
    O = afnumpy.fft.fftn(O)
    O, background = Pmod_back(amp, background, O, mask = mask, alpha = alpha)
    O = afnumpy.fft.ifftn(O)
    return O, background
    
def Pmod_back(amp, background, O, mask = 1, alpha = 1.0e-10):
    M = mask * amp / afnumpy.sqrt((O.conj() * O).real + background**2 + alpha)
    out         = O * M
    background *= M
    out += (1 - mask) * O
    return out, background

def choose_N_highest_pixels_old(array, N):
    percent = (1. - float(N) / float(array.size)) * 100.
    thresh  = np.percentile(array, percent)
    support = array > thresh
    # print '\n\nchoose_N_highest_pixels'
    # print 'percentile         :', percent, '%'
    # print 'intensity threshold:', thresh
    # print 'number of pixels in support:', np.sum(support)
    return support

def choose_N_highest_pixels(array, N, tol = 1.0e-5, maxIters=1000, support=None):
    """
    Use bisection to find the root of
    e(x) = \sum_i (array_i > x) - N

    then return (array_i > x) a boolean mask

    This is faster than using percentile (surprising)
    
    If support0 is not None then values outside the support
    are ignored. 
    """
    s0 = array.max()
    s1 = array.min()
    
    if support is not None :
        a = array[support > 0]
    else :
        a = array
        support = 1
    
    for i in range(maxIters):
        s = (s0 + s1) / 2.
        e = afnumpy.sum(a > s) - N
    
        if np.abs(e) < tol :
            break

        if e < 0 :
            s0 = s
        else :
            s1 = s
        
    S = (array > s) * support
    #print 'number of pixels in support:', afnumpy.sum(support), i, s, e, type(support)
    return S

def radial_symetry(background, rs = None, is_fft_shifted = True):
    """
    Use arrayfire's histogram to calculate the radial averages.
    """
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.sqrt(i**2 + j**2 + k**2).astype(np.int16)
        
        if is_fft_shifted is False :
            rs = np.fft.fftshift(rs)
        rs = rs.ravel()
    
    ########### Find the radial average
    # get the r histogram
    r_hist = np.bincount(rs)
    # get the radial total 
    r_av = np.bincount(rs, background.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)

    ########### Make a large background filled with the radial average
    background = r_av[rs].reshape(background.shape)
    return background, rs, r_av

def _radial_symetry(background, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.sqrt(i**2 + j**2 + k**2).astype(np.int16)
        
        if is_fft_shifted is False :
            rs = np.fft.fftshift(rs)
        rs = rs.ravel()
    
    ########### Find the radial average
    # get the r histogram
    r_hist = np.bincount(rs)
    # get the radial total 
    r_av = np.bincount(rs, background.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)

    ########### Make a large background filled with the radial average
    background = r_av[rs].reshape(background.shape)
    return background, rs, r_av
