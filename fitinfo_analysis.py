
# coding: utf-8

# In[1]:


# In[2]:

from scipy import ndimage
from weighted import median as wmedian
from weighted import quantile as wquantile
import weightedstats as wstats


# In[3]:

import glob


# In[4]:

import os
import sys
import numpy as np
from sedfitter.fit_info import FitInfoFile
from sedfitter.models import load_parameter_table
from astropy.table import Table,join
import astropy.units as u
import astropy.constants as const
from itertools import chain


def get_menv(model_number, model_numbers, m_env):
    '''
    return envelope mass for a given
    model number
    '''
    has = np.in1d(model_numbers, [int(model_number)])
    if np.sum(has) == 0:
        return 0
    else:
        return m_env[has][0]
    # return m_env[np.where(model_numbers == int(model_number))][0]

def get_menv_2(tnum):
    root = '/Users/johnlewisiii/Charlie/sedfitter_cf/OrionProtostars/'
    menv = np.loadtxt(root+'menv')
    mnum = np.loadtxt(root+'model_numbers',dtype=np.str)
    # -- searchsorted(x,y) returns index of every y that is in x
    # it is asking - where does y_i first appear in x
    return menv[np.searchsorted(mnum,tnum)]

def append_menv(param_table):
    '''
    Appends MENV column to
    parameter table
    '''
    tnum = np.array([num[:7] for num in param_table['MODEL_NAME']])
    param_table['MENV'] = get_menv_2(tnum)
    return param_table

def minmax(arr, w=None):
    if w is not None:
        arr = np.array(arr)[w]
    try:
        return np.nanmin(arr), np.nanmax(arr)
    except:
        return np.min(arr), np.max(arr)

def minbestmax(arr):
    try:
        return np.nanmin(arr), arr[0], np.nanmax(arr)
    except:
        try:
            return np.min(arr), arr[0], np.max(arr)
        except:
            arr[0]

def jmin(arr):
    try:
        return np.nanmin(arr)
    except:
        return min(arr)

def jmax(arr):
    try:
        return np.nanmax(arr)
    except:
        return max(arr)

def mean(arr, w=None):
    if w is not None:
        if w[0]:
            return arr[0]
        else:
            arr = np.array(arr)[w]
    try:
        return np.nanmean(arr)
    except:
        return np.mean(arr)

def get_av(name, avs = None):
    '''
    return av for a given
    source
    '''
    return avs[np.where(avs[:, 0] == int(name))[0], 1][0]


# In[5]:

def jjmin(arr):
    return np.min(arr[np.isfinite(arr)])

def jjmax(arr):
    return np.max(arr[np.isfinite(arr)])

def jj_median(arr,weights,w=None):
    '''
    return weighted median
    '''
    if w is not None:
        arr = np.array(arr)[w]
        weights = np.array(weights)[w]
    return wmedian(np.asarray(arr),np.asarray(weights))
    #return arr[0]

def jj_avg_std(values, weights,w=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.

    Currently set to return [50,16,84] percentiles

    w allows you to select certain elements of list
    """
    if w is not None:
        values = np.array(values)[w]
        weights = np.array(weights)[w]

    values = np.asarray(values)
    weights = np.asarray(weights)

    median=jj_median(values,weights)
    #mad = jj_median(np.abs(values-median), weights=weights)
    #average = np.average(values, weights=weights)
    #variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    #V1 = np.sum(weights)
    #V2 = np.sum(weights**2)
    #unbiased_variance = variance/(1. - (V2/(V1**2))) #use unbiased variance for small populations
    #standard_deviation = np.sqrt(unbiased_variance)
    q1 = wquantile(values,weights,.16)
    q2 = wquantile(values,weights,.84)
    #q1,q2 = minmax(values,w=w)
    #q1 = values[0]
    return (median,q1,q2)#, median, mad*1.4826)


def return_av_array(av_file, convert_from_ak=True):
    '''
    allows user to either pass in a list of avs
    or a file containing source names and avs.
    will convert from ak to av if told to
    '''
    if isinstance(av_file,str):
        avs = np.loadtxt(os.path.expanduser(av_file))
        if convert_from_ak:
            avs[:, 1] = avs[:, 1] / .11
    else:
        avs=av_file
        if convert_from_ak:
            avs = avs / .11

    return avs


# In[7]:

def get_stage(t, ratio):
    '''
    return stage for a table of models
    '''
    menv = t['MENV']
    massc = t['MASSC']
    mdot = t['MDOT']
    mdisk = t['MDISK']

    stageI   =  (mdot / massc >= 1.0e-6) & (menv  / massc >= ratio)
    stageII  =  (mdot / massc  < 1.0e-6) & (mdisk / massc >=  1e-6)
    stageIII = ((mdot / massc  < 1.0e-6) & (mdisk / massc  < 1e-6)) | ~(stageII | stageI)

    if stageI:
        return 'I'
    elif stageII:
        return 'II'
    elif stageIII:
        return 'III'
    else:
        return 'Fail'



def source_classifier(menv, massc, mdot, mdisk, chi2, ratio = 0.05):
    '''
    return class for a given set of models for a source
    '''
    menv = np.asarray(menv)
    massc = np.asarray(massc)
    mdot = np.asarray(mdot)
    mdisk = np.asarray(mdisk)

    stageI   = (jj_median(mdot / massc,1./chi2) >= 1.0e-6) & (jj_median(menv / massc,1./chi2)  >= ratio)
    stageII  = (jj_median(mdot / massc,1./chi2)  < 1.0e-6) & (jj_median(mdisk / massc,1./chi2) >= 1e-6)
    stageIII = (jj_median(mdot / massc,1./chi2)  < 1.0e-6) & (jj_median(mdisk / massc,1./chi2) <  1e-6)

    if stageI:
        return 'P'
    elif stageII:
        return 'D'
    elif stageIII:
        return 'S'
    else:
        return 'Dex'


# In[8]:

def write_val(*nums):
    '''
    Prints out values appropriately scaled
    '''
    if len(nums)==1:
        nums = nums[0]
    if not hasattr(nums, '__iter__'):
        nums = (nums,)

    returns = ()
    for num in nums:
        if num == 0:
            returns += ('%i' % num,)
        elif (np.log10(num) < -2) ^ (np.log10(num) >= 3 ):
            val = np.log10(num)
            returns += ('$%0.1f \\times 10^{%i}$ ' % (10**np.mod(val,1),int(val)),)
        else:
            returns += ('%0.1f' % num,)
    return returns


# In[9]:

def new_results_final(input_fits, verbose=True, output=True,
                av_file=None, keep=('D', 1), convert_from_ak = True,
                scale_chi2=True,fname='',prot_only=False, ratio=0.05):

    print('----- Analyzing %s ----'%(os.path.basename(input_fits)))

    avs = return_av_array(av_file, convert_from_ak = convert_from_ak)

    fin = FitInfoFile(input_fits, 'r')
    source=[]
    t = load_parameter_table(fin.meta.model_dir)
    t['MODEL_NAME'] = np.char.strip(t['MODEL_NAME'])
    t = append_menv(t)
    t.sort('MODEL_NAME')

    result, source, av, averr, infos, avints, incl, menvs, avlos, lum, mdot, massc, mdisk, stage, tstar, ndata = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],[]

    tkeys = t.colnames[1:]
    maxt = [b+'+1sig' for b in tkeys]
    mint = [b+'-1sig' for b in tkeys]
    pkeys = list(chain(*zip(tkeys,mint,maxt)))
    keys = ['Source ID', 'class', 'av', 'scale','AVLOS'] + tkeys
    dtypes = [t[n].dtype for n in tkeys]
    [dtypes.insert(0,i) for i in [float,float,float,'|S5','|S5']]

    output_table = Table(names = keys, masked=True,dtype=dtypes)

    params = {k: [] for k in keys}


    fname = fname + '%2.2f'%ratio
    if output:
        if fname is None:
            if prot_only:
                fname = '_prot_only'
            fname = input_fits[:-8] + '_out_%s%i%s.md'%(keep[0],keep[1],fname)
        else:
            if prot_only:
                fname = fname + '_prot_only'
            fname = input_fits[:-8] + '_out_%s%i_%s.md'%(keep[0],keep[1],fname)

        fout = open(fname , 'w')
        fout.write("| Source | Class |  $\chi^2_{best}$ |  $N_{data}$ |  $N_{fits}$ | $N_{P}$ | $N_{D}$ | $M\_{env}/M\_{\\star}$ ($\\times 10^{-2}$) |       | $\\dot{M}/M_{\\star}$ ($\\times 10^{-6}$) |       | $M_{\\star}$ |       |\n")
        fout.write("|:------:|:-----:|:----------------:|:-----------:|:-----------:|:-------:|:-------:|:------------------------------------------:|:-----:|:-----------------------------------------:|:-----:|:------------:|:-----:|\n")
        fout.write("|        |       |                  |             |             |         |         |       Median                               | Range |      Median                               | Range |  Median      | Range |\n")

    # ----------- Loop over fin -------------- #
    for info in fin:
        param = { k: [] for k in keys }
        if not np.isnan(np.nanmean(info.av)):
            source.append(info.source)

            minchi=info.chi2.min()
            if (scale_chi2) | (keep[0] == 'J'):
                info.chi2 = info.chi2/minchi
                keep=('D',keep[1])
            info.keep(keep[0], keep[1])


            param['Source ID'] = info.source.name
            param['av'] = np.float32(get_av(info.source.name, avs))

            tsorted = info.filter_table(t, additional={})

            get_st = lambda tab: get_stage(tab,ratio=ratio)
            stages = map(get_st, tsorted) # get stages of selected fits
            stageI = np.array(stages) == 'I'
            stageII = (np.array(stages) == 'II') | (np.array(stages) == 'III')

            nfits=np.sum(stageI)+np.sum(stageII)
            # create lists that contain source properties of interest
            source.append(info.source)
            #infos.append(info)
            ndata.append(info.source.n_data)
            av.append(get_av(info.source.name, avs))
            #avints.append(tsorted['AV_INT'].tolist())
            #avlos.append(info.av)
            #incl.append(tsorted['INCL.'].tolist())
            #lum.append(tsorted['LTOT'].tolist())
            mdot.append(tsorted['MDOT'].tolist())
            massc.append(tsorted['MASSC'].tolist())
            mdisk.append(tsorted['MDISK'].tolist())
            #tstar.append(tsorted['TSTAR'].tolist())
            menvs.append(tsorted['MENV'].tolist())

            # determine class based on quantities of interest using classification criteria
            # these classification criteria may be different than what is required for
            # identfying Model stages. Will return a single class
            classification = source_classifier(menvs[-1], massc[-1], mdot[-1], mdisk[-1],info.chi2, ratio=ratio)

            result.append(classification)


        # ------------ Print important parameters to file -----------#
            if prot_only:
                writeout = output and (result[-1]=='P')
            else:
                writeout = output
            if writeout:

                fout.write('| %s ' % info.source.name) # source name
                fout.write(' | ')
                fout.write('%s ' % result[-1]) # source class
                fout.write(' | ')
                fout.write('%5.1f' % (minchi)) # lowest chi^2 value
                fout.write(' | ')
                fout.write('%3i' % (ndata[-1])) # number of data points
                fout.write(' | ')
                fout.write('%3i' % nfits) # number of fits in range
                fout.write(' | ')
                fout.write('%3i' % np.sum(stageI)) # number of stage I
                fout.write(' | ')
                fout.write('%3i' % np.sum(stageII)) # number of stage II
                fout.write(' | ')

                #Menv/Massc
                qavg, qmin, qmax = jj_avg_std(np.array(menvs[-1])/np.array(massc[-1]), 1./info.chi2)
                power = -2
                factor = 10.**power
                fout.write('%s' % write_val(qavg/factor) )
                fout.write(' | ')
                fout.write('(%s $-$ %s)' % (write_val(qmin/factor, qmax/factor)) )
                #fout.write(' %0.f2 ' % (np.log10(qmin)-np.log10(qavg)) )
                fout.write(' | ')

                #Mdot/Massc
                qavg, qmin, qmax = jj_avg_std(np.array(mdot[-1])/np.array(massc[-1]), 1./info.chi2)
                power = -6
                factor = 10.**power
                fout.write('%s' % write_val(qavg/factor))
                fout.write(' | ')
                fout.write('(%s $-$ %s)' % (write_val(qmin/factor,qmax/factor)) )
                fout.write(' | ')

                #Massc
                qavg, qmin, qmax = jj_avg_std(massc[-1], 1./info.chi2)
                fout.write('%s' % write_val(qavg) )
                fout.write(' | ')
                fout.write('(%s $-$ %s)' % (write_val(qmin,qmax)) )
                fout.write(' | ')

                fout.write('\n')
        # ------------ [END] Print important parameters to file ----------- #


            param['class'] = classification
            med, q1, q2 = jj_avg_std(info.av,1./info.chi2)
            #param['AVLOS-1sig'] = q1
            param['AVLOS'] = med
            #param['AVLOS+1sig'] = q2
            param['scale'] = wmedian(info.sc,1./info.chi2)
            for k in tkeys:
                med, q1, q2 = jj_avg_std(tsorted[k].__array__(),1./info.chi2)
                param[k] = med
                #param[k+'-1sig'] = q1
                #param[k+'+1sig'] = q2


        else:
            param['Source ID'] = info.source.name
            param['av'] = np.float32(get_av(info.source.name, avs))
            param['scale'] = np.nan
            param['class'] = 'U'
            #param['AVLOS-1sig'] = np.nan
            param['AVLOS'] = np.nan
            #param['AVLOS+1sig'] = np.nan
            for k in tkeys:
                #param[k+'-1sig'] = np.nan
                param[k] = np.nan
                #param[k+'+1sig'] = np.nan

        output_table.add_row(param)
        for k in tkeys:
                params[k].append(param[k])

    # ----------- [END] Loop over fin -------------- #
    if output:
        fout.close()
    #t = Table(params)
    #t = t[keys]

    return  output_table,source


