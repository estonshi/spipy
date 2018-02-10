import numpy as np

def expand_region_by(mask, frac):
    import scipy.ndimage
    
    N = np.sum(mask)
    for sig in range(1, 20):
        # convolve mask with a gaussian
        mask_out = scipy.ndimage.filters.gaussian_filter(mask.copy().astype(np.float), sig, mode = 'constant')

        # progressively truncate until the sesired number of pixels is reached
        M = frac * N
        threshs = np.linspace(mask_out.max(), mask_out.min(), 100)
        for thresh in threshs:
            s = np.sum(mask_out > thresh)
            #print thresh, s
            if s > M :
                return (mask_out > thresh)

        # we did not find a good candidate 


def shrinkwrap(sample, start_pix, stop_pix, steps, step, sigma = 2.):
    from scipy.special import erfc
    from scipy import ndimage
    t = np.abs(sample)
    t = ndimage.gaussian_filter(t, sigma, mode='wrap')
    
    x   = float(step)/float(steps-1)
    pix = (start_pix - stop_pix) * erfc(4*x - 2)/2. + stop_pix
    
    # find the 'pix' highest pixel values in the sample
    value = np.percentile(t, 100. * (1. - pix / float(sample.size)))
     
    support = np.zeros(sample.shape, dtype=np.bool)
    
    support[np.where(t > value)] = True
    return support
