import numpy as np

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def oneonr(x,a,c):
    c = 0
    return a / np.abs(x + 1.0e-5)**3 + c

def fit_oneonr(x, y):
    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp

    a = (y[1] - y[0]) / (1./x[1] - 1./x[0])
    c = y[0] - a / x[0]

    print a, c

    popt,pcov = curve_fit(oneonr,x,y,p0=[a, c])

    return popt, pcov

def fit_gaus(x, y):
    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp

    #y = ar([0,1,2,3,4,5,4,3,2,1])

    n     = len(x)                      #the number of data
    mean  = sum(x*y)/n                  #note this correction
    sigma = sum(y*(x-mean)**2)/n        #note this correction

    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])

    return popt, pcov

def fit_gaus_to_min(diff):
    l0 = np.fft.fftshift(diff[0, 0, :])
    l1 = np.fft.fftshift(diff[0, :, 0])
    l2 = np.fft.fftshift(diff[:, 0, 0])
    i = np.fft.fftshift(np.fft.fftfreq(l0.shape[0], 1./l0.shape[0]))

    # find minima
    m0 = np.r_[True, l0[1:] < l0[:-1]] & np.r_[l0[:-1] < l0[1:], True]
    m1 = np.r_[True, l1[1:] < l1[:-1]] & np.r_[l1[:-1] < l1[1:], True]
    m2 = np.r_[True, l2[1:] < l2[:-1]] & np.r_[l2[:-1] < l2[1:], True]

    # exclude centre
    m0[i == 0] = False
    m1[i == 0] = False
    m2[i == 0] = False

    # generate x, y list
    x = np.hstack((i[m0], i[m1], i[m2])) 
    y = np.hstack((l0[m0], l1[m1], l2[m2]))

    # fit gaus
    popt, pcov = fit_gaus(x, y)
    
    return i, l0, popt

def fit_oneonr_to_min(diff):
    l = np.fft.fftshift(diff[0, 0, :])
    i = np.fft.fftshift(np.fft.fftfreq(l.shape[0], 1./l.shape[0]))

    # find minima
    m = np.r_[True, l[1:] < l[:-1]] & np.r_[l[:-1] < l[1:], True]

    # exclude centre
    m[i == 0] = False

    # generate x, y list
    x = i[m]
    y = l[m]

    # fit gaus
    popt, pcov = fit_oneonr(x, y)

    return i, l, popt

def gaussian_subtract(diff, scale_gaus = 0.9, is_fft_shifted = True):
    i, l, popt = fit_gaus_to_min(diff)
    
    shape = diff.shape
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    k = np.fft.fftfreq(shape[2]) * shape[2]
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    rs      = (i**2 + j**2 + k**2).astype(np.float)
        
    if is_fft_shifted is False :
        rs = np.fft.ifftshift(rs)
    
    gaus = scale_gaus * popt[0] * np.exp(-rs/(2.*popt[2]**2))

    diff -= gaus

    diff[diff < 0.0] = 0.0
    return diff
    
def oneonr_subtract(diff, scale_gaus = 0.9, is_fft_shifted = True):
    i, l, popt = fit_oneonr_to_min(diff)
    
    shape = diff.shape
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    k = np.fft.fftfreq(shape[2]) * shape[2]
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    rs      = np.sqrt((i**2 + j**2 + k**2).astype(np.float))
        
    if is_fft_shifted is False :
        rs = np.fft.ifftshift(rs)
    
    gaus = scale_gaus * popt[0] / np.abs(rs + 1.0e-5)**3 

    diff -= gaus

    diff[diff < 0.0] = 0.0
    return diff

if __name__ == '__main__':
    import h5py
    import numpy as np
    import pyqtgraph as pg

    f = h5py.File('spi/ivan/input.h5', 'r')
    diff = oneonr_subtract(f['data'].value)
    """
    i, l, popt = fit_gaus_to_min(f['data'].value)
    i, l, popt2 = fit_oneonr_to_min(f['data'].value)
    plot = pg.plot(i, l)
    plot.plot(i, gaus(i, *popt), pen=pg.mkPen('b'))
    y = oneonr(i, *popt2)
    y[i == 0] = 0
    plot.plot(i, y, pen=pg.mkPen('g'))
    """
