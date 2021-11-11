import numpy as np
import scipy.ndimage as ndimage
from astropy.convolution import convolve,convolve_fft, kernels

def quick_background(img,s=31):
    bkg_min = ndimage.minimum_filter(img,size=(s,s))
    bkg_max = ndimage.maximum_filter(bkg_min,size=(s,s))
    return ndimage.gaussian_filter(bkg_max, s//2)

def background2D(image,rms,s=31,fill=0,experimental=False, v=True):
    image = np.copy(image)
    oldnan = np.isnan(image)

    sp = s + s//4
    sp = sp+1 if sp%2==0 else sp
    s1 = sp//2
    s1 = s1+1 if s1%2==0 else s1
    s2 = s1//2
    arr=np.ones((sp,sp))
    arr[s1-s2:s1+s2,s1-s2:s1+s2]=0
    expand =convolve_fft(image,kernels.CustomKernel(arr),boundary='wrap')
    image[np.isnan(image)] = expand[np.isnan(image)]

    #image[oldnan] = fill
    s = int(s)
    s = s+1 if s%2==0 else s
    bkg_min = ndimage.minimum_filter(image,size=(s,s))
    bkg_max = ndimage.maximum_filter(bkg_min,size=(s,s))
    kernel = kernels.Box2DKernel(s)
    bkg_mean = convolve(bkg_max,kernel,boundary='extend',)


    ## OLD METHOD
    if experimental:
        resid = np.abs(bkg_mean-image)
        mask = resid > 3*rms#np.std(resid)
        mask = resid > 3*rms#np.std(resid[mask])
        resid[mask] = np.nan
        resid_smooth = np.copy(resid)
        while np.any(np.isnan(resid_smooth)):
            # print('hi')
            resid_smooth = convolve(resid_smooth,kernel)
        bkg = bkg_mean+ resid_smooth
    else:
        bkg = bkg_mean

    bkg[oldnan] = np.nan

    return bkg



def background1d(image, s=31, fill=0, experimental=False):
    image = np.copy(image)
    oldnan = np.isnan(image)

    sp = s + s//4
    sp = sp+1 if sp % 2 == 0 else sp
    s1 = sp//2
    s1 = s1+1 if s1 % 2 == 0 else s1
    s2 = s1//2
    arr = np.ones((sp,))
    arr[s1-s2:s1+s2] = 0
    expand = convolve_fft(image, kernels.CustomKernel(arr), boundary='wrap')
    image[np.isnan(image)] = expand[np.isnan(image)]

    # image[oldnan] = fill
    s = int(s)
    s = s+1 if s % 2 == 0 else s
    bkg_min = ndimage.minimum_filter(image, size=(s,))
    bkg_max = ndimage.maximum_filter(bkg_min, size=(s,))
    kernel = kernels.Box1DKernel(s)
    bkg_mean = convolve(bkg_max, kernel, boundary='extend',)

    if experimental:
        bkg_mean = np.min(np.vstack([[bkg_max], [bkg_mean]]), axis=0)
        bkg_new = np.copy(bkg_mean)
        # print(bkg_new.shape)
        s = s//2
        while s > 2:
            s = s+1 if s % 2 == 0 else s
            kernel2 = kernels.Box1DKernel(s)
            bkg_mean = convolve_fft(bkg_new, kernel2, boundary='wrap')
            bkg_new = np.min(np.array([bkg_mean, bkg_new]), axis=0)
            s = s//2

        bkg_new = np.min(np.vstack([[bkg_mean], [bkg_new]]), axis=0)
        kernel3 = kernels.CustomKernel(np.ones((1,)))
        kernel3.normalize()
        bkg_mean = convolve_fft(bkg_new, kernel, boundary='wrap')

    bkg = bkg_mean

    bkg[oldnan] = np.nan

    return bkg