import numpy as np

# read in a 'class 1 powder sum' (the sum of all 2D scatter patterns)
# get a radial sum
# rescale to show the number of photons per speckle

def rad_av(diff, rs = None, is_fft_shifted = True, output_diff = False):
    if rs is None :
        i = np.fft.fftfreq(diff.shape[0]) * diff.shape[0]
        j = np.fft.fftfreq(diff.shape[1]) * diff.shape[1]
        i, j = np.meshgrid(i, j, indexing='ij')
        rs      = np.sqrt(i**2 + j**2).astype(np.int16).ravel()
        
        if is_fft_shifted is False :
            rs = np.fft.ifftshift(rs)
    
    ########### Find the radial average
    # get the r histogram
    r_hist = np.bincount(rs)
    # get the radial total 
    r_av = np.bincount(rs, diff.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)

    if output_diff :
        diff_out = np.zeros_like(diff)
        diff_out = r_av[rs].reshape(diff.shape)
        return r_av, diff_out
    else :
        return r_av

def simulate_powder(photons = 1e10):
    # make some diffraction pattern
    im = np.random.random((1024, 1024))
    
    im[256 :, 256 :] = 0.0
    
    diff = np.abs(np.fft.fftn(im))**2

    # if this is a powder of a sample in 
    # random orientations then we should 
    # radially average it
    r_av, diff = rad_av(diff, output_diff = True)
    
    # apply photon counting
    diff = diff / np.sum(diff)
    diff = np.random.poisson(lam = photons * diff)
    
    return diff, r_av

def photons_per_speckle(powder, diam = 1./4., z = 1000., lamb = 1.0e-3, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(powder.shape[0]) * powder.shape[0]
        j = np.fft.fftfreq(powder.shape[1]) * powder.shape[1]
        i, j = np.meshgrid(i, j, indexing='ij')
        rs   = np.sqrt(i**2 + j**2).astype(np.int16).ravel()
        
        if is_fft_shifted is False :
            rs = np.fft.ifftshift(rs)
    
    # find the pixel radius for each half speckle
    rmax = rs.max()
    r  = 0.0
    n  = 0
    rD = []
    while r < i.max() :
        r  = z * np.tan(2. * np.arcsin(n * lamb / (2. * diam) ) )
        rD.append(r)
        n += 0.5
    
    # and... once more
    r  = z * np.tan(2. * np.arcsin(n * lamb / (2. * diam) ) )
    rD.append(r)
    
    # now get the number of speckles in each speckle shell 
    # this is analytical
    # 2 pi n^2 + pi / 6
    """
    no  = []
    i_s = range(0, len(rD), 2)
    print i_s
    print len(rD)
    for i in i_s :
        if i == 0 :
            t = 1.
        
        elif i == i_s[-1] and (len(rD) % 2 == 1):
            t = 4./3.*np.pi * ((2.*rD[i] - rD[i-1])**3 - rD[i-1]**3) \
                            / (2.*rD[i] - 2.*rD[i-1])**3   
        else :
            print i
            t = 4./3.*np.pi * (rD[i+1]**3 - rD[i-1]**3) / (rD[i+1] - rD[i-1])**3   
        no.append(t)
    """
    
    # now we have a set of rings centred (presumably) on the speckles
    # let's get the radial sum of the powder pattern
    i_s = range(0, len(rD), 2)
    phot_per_speckle = np.zeros((len(i_s),), dtype=np.float)
    no = np.zeros((len(i_s),), dtype=np.float)
    for i in i_s :
        # get the radial range
        if i == 0 :
            rmin = 0
            rmax = rD[i+1]
        elif i == i_s[-1] and (len(rD) % 2 == 1):
            rmin = rD[i-1] 
            rmax = rD[i] 
        else :
            rmin = rD[i-1] 
            rmax = rD[i+1] 
        
        print rmax, rs.shape, powder.shape
        # get the number of photons in this radial range
        rss    = np.where( (rmin <= rs) * (rs < rmax) )
        #r_sum  = np.bincount(rs[rss], powder[rss].ravel())
        r_sum  = np.sum(powder.ravel()[rss])
        
        # now divide the number of photons in each speckle shell 
        # by the number of speckles in each speckle shell
        
        # 3D
        no[i/2] = float( 2.*np.pi*(i/2)**2 + np.pi/6.) / 2.
        
        # 2D
        #no[i/2] = max(float( 8.*np.pi*(i/2)), 1.) / 2.
        
        phot_per_speckle[i/2] = float(np.sum(r_sum)) / no[i/2]
    
    return rD, no, phot_per_speckle

    



