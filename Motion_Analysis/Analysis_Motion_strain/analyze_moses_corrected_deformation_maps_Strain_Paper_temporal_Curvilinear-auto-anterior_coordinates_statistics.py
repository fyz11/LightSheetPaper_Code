#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:26:50 2019

@author: felix
"""

import numpy as np 

def meantracks2D_to_3D( meantracks, ref_map):
    """
    ref map gives the matching corresponding look up table in 3D coordinates. 
    """
    
    meantracks3D = ref_map[meantracks[:,:,0], meantracks[:,:,1],:]
    
    return meantracks3D
    

def exponential_decay_correction_time(vid, average_fnc=None, f_scale=100.):
    """
    Fits an equation of form y=Ae^(Bt)+C on the mean intensities of video
    """
    from scipy.stats import linregress
    from scipy.optimize import least_squares
    
    if average_fnc is None:
        average_fnc = np.mean
        
        
    I_vid = np.hstack([average_fnc(v) for v in vid])
    I_time = np.arange(len(I_vid))
    
    # fit equation. y =A*e^(-Bt)
    log_I_vid = np.log(I_vid)
    slope, intercept, r_value, p_value, std_err = linregress(I_time, log_I_vid)

    # initial fit. 
    A = np.exp(intercept)
    B = slope
    
    # refined robust fitting. 
    def exp_decay(t,x):
        return (x[0] * np.exp(x[1] * t) + x[2])
        
    def res(x, t, y):
        return exp_decay(t,x) - y
    
    x0 = [A, B, 0]
    res_robust = least_squares(res, x0, loss='soft_l1', f_scale=f_scale, args=(I_time, I_vid))
        
    robust_y = exp_decay(I_time, res_robust.x)
    correction = float(robust_y[0]) / robust_y
    
#    plt.figure()
#    plt.plot(robust_y)
#    plt.show()
    vid_corrected = np.zeros(vid.shape, np.float32)
    
    for frame in range(vid.shape[0]):
        vid_corrected[frame, ...] = vid[frame, ...] * correction[frame]
    
    return vid_corrected, res_robust.x


def locate_unwrapped_im_folder(rootfolder, key='geodesic-rotmatrix'):
    
    found_dir = []
    
    for dirName, subdirList, fileList in os.walk(rootfolder):
#        print('Found directory: %s' % dirName)
        for directory in subdirList:
            if key in directory and 'unwrap_params' not in directory:
                found_dir.append(os.path.join(dirName, directory))
                
    return found_dir

def parse_condition_fnames_lifeact_unwrapped(vidfile):
    
    import os 
    import re 
    
    fname = os.path.split(vidfile)[-1]
    
    """ get the embryo number """
    emb_no = re.findall('L\d+', fname)

    if emb_no:
        emb_no = emb_no[0].split('L')[1]
    else:
        emb_no = np.nan
       
    emb_id = re.findall('Emb\d+', fname)
    
    if emb_id:
        emb_id = emb_id[0]
        """.split('Emb')[1]"""
    else:
        emb_id = ''
    
    """
    """
        
    """ get the angle of the unwrapping """
    ang_cand = re.findall(r"\d+", fname)

    ang = '000'
        
    for a in ang_cand:
        if len(a) == 3 and a!=emb_no:
            ang = a
            
    """ get the tissue type """
    tissue = []
    
    if 've' in fname and 'HEX' not in fname:
        tissue = 've'
    if 'epi' in fname:
        tissue = 'epi'
    if 've' in fname and 'HEX' in fname:
        tissue = 'hex'
        
    """ get  the projection type """
    if 'rect' in fname:
        proj_type = 'rect'
    if 'polar' in fname:
        proj_type = 'polar'
            
    return tissue, emb_no, emb_id, ang, proj_type


def parse_condition_fnames_hex_unwrapped(vidfile):
    
    import os 
    import re 
    
    fname = os.path.split(vidfile)[-1]
    
    """ get the embryo number """
    emb_no = re.findall('L\d+', fname)

    if emb_no:
        emb_no = emb_no[0].split('L')[1]
    else:
        emb_no = np.nan
        
    emb_id = re.findall('Emb\d+', fname)
    
    if emb_id:
        emb_id = emb_id[0]
        """.split('Emb')[1]"""
    else:
        emb_id = ''
        
    """ get the angle of the unwrapping """
    ang_cand = re.findall(r"\d+", fname)

    ang = '000'
        
    for a in ang_cand:
        if len(a) == 3 and a!=emb_no:
            ang = a
            
    """ get the tissue type """
    tissue = []
    
    if 've' in fname and 'HEX' not in fname:
        tissue = 've'
    if 'epi' in fname:
        tissue = 'epi'
    if 've' in fname and 'HEX' in fname:
        tissue = 'hex'
        
    """ get  the projection type """
    if 'rect' in fname:
        proj_type = 'rect'
    if 'polar' in fname:
        proj_type = 'polar'
            
    return tissue, emb_no, emb_id, ang, proj_type

def pair_meta_file_lifeact_mtmg_embryo(files, meta_files):
    
    uniq_vid_conds = np.unique(np.hstack(['-'.join(f[1:]) for f in meta_files]))
    file_conds = np.hstack(['-'.join(f) for f in meta_files])
    
    file_sets = []
    file_set_conditions = []
    
    for ii in range(len(uniq_vid_conds)):
        uniq_vid_cond = uniq_vid_conds[ii]
        
        file_set = []
        
        for tissue_type in ['ve','epi']:
            query_cond = tissue_type+'-'+uniq_vid_cond
            select_file = file_conds == query_cond
            select_file = files[select_file]
            
            file_set.append(select_file[0])
            
        file_sets.append(file_set)
        file_set_conditions.append(uniq_vid_conds[ii])
    
    return np.hstack(file_set_conditions), np.array(file_sets)


def pair_meta_file_hex_embryo(files, meta_files):
    
    uniq_vid_conds = np.unique(np.hstack(['-'.join(f[1:]) for f in meta_files]))
    file_conds = np.hstack(['-'.join(f) for f in meta_files])
    
    file_sets = []
    file_set_conditions = []
    
    for ii in range(len(uniq_vid_conds)):
        uniq_vid_cond = uniq_vid_conds[ii]
        
        file_set = []
        
        for tissue_type in ['ve','epi','hex']:
            query_cond = tissue_type+'-'+uniq_vid_cond
            select_file = file_conds == query_cond
            select_file = files[select_file]
            
            file_set.append(select_file[0])
            
        file_sets.append(file_set)
        file_set_conditions.append(uniq_vid_conds[ii])
    
    return np.hstack(file_set_conditions), np.array(file_sets)



def cross_corr_tracksets(tracks1, tracks2, return_curves=False, normalize=True, mode='full'):
    
    eps = np.finfo(np.float32).eps
    
    corrs = []
    lags = []
    
    for ii in range(len(tracks1)):
        
        sig1 = tracks1[ii].copy()
        sig2 = tracks2[ii].copy()
        
        if normalize:
            
            ccf = np.vstack([np.correlate((sig1[...,jj] - np.mean(sig1[...,jj])) / ((np.std(sig1[...,jj]) + eps) * len(sig1)), 
                                          (sig2[...,jj] - np.mean(sig2[...,jj])) / ((np.std(sig2[...,jj]) + eps)), mode=mode) for jj in range(sig1.shape[-1])])
#        else:
#            ccf = np.vstack([np.correlate(sig1, 
#                                          (sig2[...,jj] - np.mean(sig2[...,jj])) / ((np.std(sig2[...,jj]) + eps)), mode=mode) for jj in range(sig1.shape[-1])])
        ccf_sum = np.mean(ccf, axis=0); #print(ccf_sum.shape)
        max_ind = np.argmax(np.abs(ccf_sum))
        
        lag = max_ind - len(ccf_sum) //2 
        lags.append(lag)
        
        if return_curves:
            corrs.append(ccf_sum)
        else:
            corrs.append(ccf_sum[max_ind])
    
            
#    ccf_x = np.correlate((sig1_x-np.mean(sig1_x))/((np.std(sig1_x)+eps)*len(sig1_x)), (sig2_x-np.mean(sig2_x))/(np.std(sig2_x)+eps), mode=mode)
#    ccf_y = np.correlate((sig1_y-np.mean(sig1_y))/((np.std(sig1_y)+eps)*len(sig1_y)), (sig2_y-np.mean(sig2_y))/(np.std(sig2_y)+eps), mode=mode)
#    ccf_top = (ccf_x + ccf_y)
    return lags, corrs


def cross_corr_tracksets_multivariate(tracks1, tracks2, return_curves=False, normalize=True, mode='full'):
    
    eps = np.finfo(np.float32).eps
    n_dim = tracks1.shape[-1] # 2 dimensions.
    
    corrs = []
    lags = []
    
    print('using multivariation cross-correlation')
    
    for ii in range(len(tracks1)):
        
        sig1 = tracks1[ii].copy()
        sig2 = tracks2[ii].copy()
        
        if normalize:
            
            ccf_matrix_2d = []
            
            for mat_i in range(n_dim):
                for mat_j in range(n_dim):
            
#                    ccf = np.vstack([np.correlate((sig1[...,jj] - np.mean(sig1[...,jj])) / ((np.std(sig1[...,jj]) + eps) * len(sig1)), 
#                                                  (sig2[...,jj] - np.mean(sig2[...,jj])) / ((np.std(sig2[...,jj]) + eps)), mode=mode) for jj in range(sig1.shape[-1])])
                    ccf = np.correlate((sig1[...,mat_i] - np.mean(sig1[...,mat_i])) / ((np.std(sig1[...,mat_i]) + eps) * len(sig1)), 
                                       (sig2[...,mat_j] - np.mean(sig2[...,mat_j])) / ((np.std(sig2[...,mat_j]) + eps)), mode=mode)
                    ccf_matrix_2d.append(ccf)
                
            ccf_matrix_2d = np.vstack(ccf_matrix_2d).T
#            print(ccf_matrix_2d.shape)
#            ccf_eigenvalues = np.vstack([np.linalg.eigh(np.array([[ccf_matrix[0], ccf_matrix[1]], 
#                                                                  [ccf_matrix[2], ccf_matrix[3]]]))[0] for ccf_matrix in ccf_matrix_2d])
            ccf_eigenvalues = np.vstack([np.linalg.eigh(ccf_matrix.reshape((n_dim,n_dim)))[0] for ccf_matrix in ccf_matrix_2d])
#            print(ccf_eigenvalues.shape)        
            pos_max = np.argmax(np.abs(ccf_eigenvalues), axis=1)
#            pos_max = np.argmax(ccf_eigenvalues, axis=1)
#            print(pos_max.shape)
            ccf_sum = ccf_eigenvalues[np.arange(len(ccf_eigenvalues)), pos_max]
            ccf_sum = ccf_sum/float(n_dim)
#            print(ccf_sum.shape)
        else:
            ccf_matrix_2d = []
            
            for mat_i in range(n_dim):
                for mat_j in range(n_dim):
            
#                    ccf = np.vstack([np.correlate((sig1[...,jj] - np.mean(sig1[...,jj])) / ((np.std(sig1[...,jj]) + eps) * len(sig1)), 
#                                                  (sig2[...,jj] - np.mean(sig2[...,jj])) / ((np.std(sig2[...,jj]) + eps)), mode=mode) for jj in range(sig1.shape[-1])])
                    ccf = np.correlate((sig1[...,mat_i] + eps) / len(sig1), 
                                       (sig2[...,mat_j] + eps), mode=mode)
                    ccf_matrix_2d.append(ccf)
                
            ccf_matrix_2d = np.vstack(ccf_matrix_2d).T
#            print(ccf_matrix_2d.shape)
#            ccf_eigenvalues = np.vstack([np.linalg.eigh(np.array([[ccf_matrix[0], ccf_matrix[1]], 
#                                                                  [ccf_matrix[2], ccf_matrix[3]]]))[0] for ccf_matrix in ccf_matrix_2d])
            ccf_eigenvalues = np.vstack([np.linalg.eigh(ccf_matrix.reshape((n_dim,n_dim)))[0] for ccf_matrix in ccf_matrix_2d])
#            print(ccf_eigenvalues.shape)        
            pos_max = np.argmax(np.abs(ccf_eigenvalues), axis=1)
#            pos_max = np.argmax(ccf_eigenvalues, axis=1)
#            print(pos_max.shape)
            ccf_sum = ccf_eigenvalues[np.arange(len(ccf_eigenvalues)), pos_max]
            ccf_sum = ccf_sum/float(n_dim)
#        else:
#            ccf = np.vstack([np.correlate(sig1, 
#                                          (sig2[...,jj] - np.mean(sig2[...,jj])) / ((np.std(sig2[...,jj]) + eps)), mode=mode) for jj in range(sig1.shape[-1])])
#        ccf_sum = np.mean(ccf, axis=0); #print(ccf_sum.shape)
        max_ind = np.argmax(np.abs(ccf_sum))
        
        lag = max_ind - len(ccf_sum) //2 
        lags.append(lag)
        
        if return_curves:
            corrs.append(ccf_sum)
        else:
            corrs.append(ccf_sum[max_ind])
    
    return lags, corrs


#def cross_corr_tracksets3D(tracks1, tracks2, return_curves=False, normalize=True, mode='full'):
#    
#    eps = np.finfo(np.float32).eps
#    n_dim = tracks1.shape[-1] # 2 dimensions.
#    
#    corrs = []
#    lags = []
#    
#    print('using multivariation cross-correlation')
#    
#    for ii in range(len(tracks1)):
#        
#        sig1 = tracks1[ii].copy()
#        sig2 = tracks2[ii].copy()
#        
#        if normalize:
#            
#            ccf_matrix_2d = []
#            
#            for mat_i in range(n_dim):
#                for mat_j in range(n_dim):
#            
##                    ccf = np.vstack([np.correlate((sig1[...,jj] - np.mean(sig1[...,jj])) / ((np.std(sig1[...,jj]) + eps) * len(sig1)), 
##                                                  (sig2[...,jj] - np.mean(sig2[...,jj])) / ((np.std(sig2[...,jj]) + eps)), mode=mode) for jj in range(sig1.shape[-1])])
#                    ccf = np.correlate((sig1[...,mat_i] - np.mean(sig1[...,mat_i])) / ((np.std(sig1[...,mat_i]) + eps) * len(sig1)), 
#                                       (sig2[...,mat_j] - np.mean(sig2[...,mat_j])) / ((np.std(sig2[...,mat_j]) + eps)), mode=mode)
#                    ccf_matrix_2d.append(ccf)
#                
#            ccf_matrix_2d = np.vstack(ccf_matrix_2d).T
##            print(ccf_matrix_2d.shape)
#            ccf_eigenvalues = np.vstack([np.linalg.eigh(np.array([[ccf_matrix[0], ccf_matrix[1]], 
#                                                                  [ccf_matrix[2], ccf_matrix[3]]]))[0] for ccf_matrix in ccf_matrix_2d])
##            print(ccf_eigenvalues.shape)        
#            pos_max = np.argmax(np.abs(ccf_eigenvalues), axis=1)
##            pos_max = np.argmax(ccf_eigenvalues, axis=1)
##            print(pos_max.shape)
#            ccf_sum = ccf_eigenvalues[np.arange(len(ccf_eigenvalues)), pos_max]
##            print(ccf_sum.shape)
##        else:
##            ccf = np.vstack([np.correlate(sig1, 
##                                          (sig2[...,jj] - np.mean(sig2[...,jj])) / ((np.std(sig2[...,jj]) + eps)), mode=mode) for jj in range(sig1.shape[-1])])
##        ccf_sum = np.mean(ccf, axis=0); #print(ccf_sum.shape)
#        max_ind = np.argmax(np.abs(ccf_sum))
#        
#        lag = max_ind - len(ccf_sum) //2 
#        lags.append(lag)
#        
#        if return_curves:
#            corrs.append(ccf_sum)
#        else:
#            corrs.append(ccf_sum[max_ind])
#    
#    return lags, corrs




def smooth_track_set(tracks, win_size = 3, polyorder = 1):
    
    from scipy.signal import savgol_filter
                
    tracks_smooth = []
    
    for tra in tracks:
        tra_smooth = np.vstack([savgol_filter(tra[...,i], window_length=win_size, polyorder=polyorder) for i in range(tracks.shape[-1])]).T
        tracks_smooth.append(tra_smooth)
    
    return np.array(tracks_smooth)

def parse_start_end_times(vals):
    
    start_end = []
    
    for v in vals:
        v = str(v)
        if v=='nan':
            start_end.append([np.nan, np.nan])
        else:
#            print(v)
            start_end.append(np.hstack(v.split('_')).astype(np.float))

    return np.vstack(start_end)


def test_VE_piecewise_gradient(gradients, break_points, thresh_grad_buffer=0.5):
    
    # add the start and end stretch 
    breaks = np.hstack([0, break_points, len(gradients)+1])
    break_gradients = np.zeros(len(breaks)-1)
    break_gradient_signal = np.zeros(len(gradients))
    
    for ii in range(len(break_gradients)):
        
        break_start = breaks[ii]
        break_end = breaks[ii+1]
        
        break_grad = gradients[break_start:break_end]
        break_gradients[ii] = np.mean(break_grad)
        break_gradient_signal[break_start:break_end] = np.mean(break_grad)
        
    # conduct the test to classify segments into pre, migrate, post.
    grad_nonzero = break_gradients >= thresh_grad_buffer # must be positive for it to be migrating 
    
    pre_breaks = []
    migrate_breaks = []
    post_breaks = []
    
    # allocate all of these to migrate or post breaks. 
    for i in range(len(grad_nonzero)):
        # this step is necessary to allocate to one of these. 
        if i == 0:
            if grad_nonzero[i] == False:
                pre_breaks.append([i, breaks[i], breaks[i+1]])
            else:
                # if true then it will be migrate break. 
                migrate_breaks.append([i, breaks[i], breaks[i+1]])
        else:
            if grad_nonzero[i] == False and len(pre_breaks)>0:
                # check if we extend pre break. 
                if i - pre_breaks[-1][0] == 1:
                    pre_breaks.append([i, breaks[i], breaks[i+1]])
                else:
                    # belongs to post if there is something in migrate
                    if len(migrate_breaks) > 0: 
                        post_breaks.append([i, breaks[i], breaks[i+1]])
                    
            # migrate must come before post. 
            if grad_nonzero[i] == True and len(post_breaks) == 0:
                migrate_breaks.append([i, breaks[i], breaks[i+1]])        
                
            if grad_nonzero[i] == False and len(pre_breaks)==0 and len(migrate_breaks)>0:
                post_breaks.append([i, breaks[i], breaks[i+1]])
    
    all_breaks = [pre_breaks, migrate_breaks, post_breaks]
        
    return break_gradients, break_gradient_signal, all_breaks

def baseline_als(y, lam, p, niter=10):
    """ Code adapted from https://stackoverflow.com/questions/29156532/python-baseline-correction-library. 
    Implements paper of "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens (2005), [1].
    
    Parameters
    ----------
    y : numpy array
        numpy vector of observations to fit.
    lam : float 
        smoothness parameter, paper recommends 10**2 ≤ λ ≤ 10**9 for most applications.
    p : float
        asymmetry parameter, the higher this is the more asymmetric we consider the fitting errors, paper recommends 0.001 ≤ p ≤ 0.1 for positive peaks.
    niter : int
        the number of iterations to run the iterative fitting algorithm, the longer you run this, the more flat the baseline. Only a few iterations is needed. 
    
    Returns
    -------
    z : numpy array
        fitted baseline, same numpy vector size as y.
    References
    ----------
    .. [1] Eilers PH, Boelens HF. "Baseline correction with asymmetric least squares smoothing." Leiden University Medical Centre Report. 2005 Oct 21;1(1):5.
    
    """
    from scipy import sparse
    import numpy as np 
    from scipy.sparse.linalg import spsolve
    
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)

    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)

    return z


def smooth_area_measures(area_array, p=0.5, lam=1e3):
    
    ii, jj = area_array.shape
    
    area_array_smooth = np.zeros_like(area_array)
    
    for j in range(jj):
        area_array_smooth[:,j] = baseline_als(area_array[:,j], lam=lam, p=p)

    return area_array_smooth

def merge_time_points(all_times, times):
    
    out_times = []

    # do pre first:
    for per in times:
#        print(per)
        if len(per) == 0 :
            out_times.append(per)
        if len(per) == 1:
            out_times.append(per[0][1:])
        if len(per) > 1: 
            start = per[0][1]
            end = per[0][2]
            
            for ii in range(len(per)-1):
                if per[ii][2] == per[ii+1][1]:
                    end = per[ii+1][2]
                    
            out_times.append([start,end])

    return out_times                    
    

def tile_uniform_windows_radial_custom(imsize, n_r, n_theta, max_r=None, mask_distal=None, center=None, bound_r=True):
    
    from skimage.segmentation import mark_boundaries
    m, n = imsize
    spixels = np.zeros((m,n), dtype=np.int)
    
    XX, YY = np.meshgrid(range(n), range(m))
    
    if center is None:
        center = np.hstack([m/2., n/2.])

    r2 = (XX - center[1])**2  + (YY - center[0])**2
    r = np.sqrt(r2)
    theta = np.arctan2(YY-center[0],XX-center[1]) #+ np.pi
    
#    theta = np.arctan2(XX-center[1],YY-center[0]) + np.pi    
    if max_r is None:
        if bound_r: 
            max_r = np.minimum(np.abs(np.max(XX-center[1])),
                               np.abs(np.max(YY-center[0])))
        else:
            max_r = np.maximum(np.max(np.abs(XX-center[1])),
                               np.max(np.abs(YY-center[0])))
#    plt.figure()
#    plt.imshow(r)
#    plt.figure()
#    plt.imshow(theta)
#    plt.show()
#    print(theta.min())
#    r_bounds = np.linspace(0, .5*(m+n)/2., n_r+1)
    if mask_distal is not None:
        if bound_r:
            if mask_distal > 1:
                start_r = mask_distal
            else:
                start_r = int(mask_distal*max_r)
        else:
            if mask_distal > 1:
                start_r = mask_distal
            else:
                start_r = int(mask_distal*max_r)
    else:
        start_r = 0
    
    """ partition r """ 
    if bound_r: 
        r_bounds = np.linspace(start_r, max_r, n_r+1)
    else:
        r_bounds = np.linspace(start_r, max_r, n_r+1)
    """ partition theta """
    theta_bounds = np.linspace(-np.pi, np.pi, n_theta+1)
    
    """ label partition """ 
    if mask_distal is not None:
        counter = 2 # 0 is background.
        mask = r <= start_r
        spixels[mask] = 1
    else:
        counter = 1 # start from one. 
        
    for ii in range(len(r_bounds)-1):
        for jj in range(len(theta_bounds)-1):
            mask_r = np.logical_and(r>=r_bounds[ii], r <= r_bounds[ii+1])
            mask_theta = np.logical_and( theta>=theta_bounds[jj], theta <= theta_bounds[jj+1])
            mask = np.logical_and(mask_r, mask_theta)

            spixels[mask] = counter 
            counter += 1
    
#    fig, ax=  plt.subplots(figsize=(15,15))
#    ax.imshow(mark_boundaries(np.dstack([spixels,spixels,spixels]), spixels))
    return spixels


def realign_grid_center_polar( tile_grid, n_angles, center_dist_r, center_angle):
    
    """
    by default the -180 degrees will be labelled as '0', so have to rotate by 180.
    
    be default the image rotation is anticlockwise..... FFS.
    """
    from skimage.transform import rotate
    m, n = tile_grid.shape
    tile_grid_new = tile_grid.copy()
    
    YY, XX = np.indices(tile_grid.shape)

    """
    step 1: just adjust the rotation for now. 
    """
    # the first value is to make the center of the first quadrant lie on the x line. 
    tile_grid_new = np.rint(rotate(tile_grid_new, angle= (180+360./n_angles/2.) + center_angle, preserve_range=True)).astype(np.int)
#    tile_grid_new = rotate(tile_grid_new, angle=center_angle, preserve_range=True)
    tile_grid_new[tile_grid == 0] = 0 
    
    return tile_grid_new
   
def get_largest_connected_component( pts, dist_thresh):

    from sklearn.metrics.pairwise import pairwise_distances
    import networkx as nx
    A = pairwise_distances(pts) <= dist_thresh
 
    G = nx.from_numpy_matrix(A)
    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc = np.hstack(list(largest_cc))
                    
    return largest_cc

def postprocess_polar_labels(pos0, labels0, dist_thresh):
    
    select = np.arange(len(labels0))[labels0>0]
    pos_pt_ids = get_largest_connected_component( pos0[select], dist_thresh)
    
    labels_new = np.zeros_like(labels0)
    labels_new[select[pos_pt_ids]] = 1
    
    return labels_new

def parse_temporal_scaling(tformfile):
    
    import scipy.io as spio 
    from transforms3d.affines import decompose44
    
    obj = spio.loadmat(tformfile)
    scales = np.hstack([1./decompose44(matrix)[2][0] for matrix in obj['tforms']])
    
    return scales


##### functions for creating a custom gridding scheme given the boundary masks. 
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def draw_polygon_mask(xy, shape):
    
    from skimage.draw import polygon
    img = np.zeros(shape, dtype=np.bool)
    rr, cc = polygon(xy[:,1].astype(np.int),
                     xy[:,0].astype(np.int))
    img[rr, cc] = 1
    
    return img


# def tile_uniform_windows_radial_guided_line(imsize, n_r, n_theta, max_r, mid_line, center=None, bound_r=True):
    
#     from skimage.segmentation import mark_boundaries
#     m, n = imsize
    
#     XX, YY = np.meshgrid(range(n), range(m))
    
#     if center is None:
#         center = np.hstack([m/2., n/2.])

#     r2 = (XX - center[1])**2  + (YY - center[0])**2
#     r = np.sqrt(r2)
#     theta = np.arctan2(YY-center[0],XX-center[1])
    
#     if max_r is None:
#         if bound_r: 
#             max_r = np.minimum(np.abs(np.max(XX-center[1])),
#                                np.abs(np.max(YY-center[0])))
#         else:
#             max_r = np.maximum(np.max(np.abs(XX-center[1])),
#                                np.max(np.abs(YY-center[0])))

#     """
#     construct contour lines that bipartition the space. 
#     """
#     mid_line_polar = np.vstack(cart2pol(mid_line[:,0] - center[1], mid_line[:,1] - center[0])).T
#     mid_line_central = np.vstack(pol2cart(mid_line_polar[:,0], mid_line_polar[:,1])).T
    
#     # derive lower and upper boundary lines -> make sure to 
#     contour_r_lower_polar = np.array([np.linspace(0, l, n_r//2+1) for l in mid_line_polar[:,0]]).T
#     contour_r_upper_polar = np.array([np.linspace(l, max_r, n_r//2+1) for l in mid_line_polar[:,0]]).T
    
#     contour_r_lower_lines = [np.vstack(pol2cart(l, mid_line_polar[:,1])).T for l in contour_r_lower_polar][:-1]
#     contour_r_upper_lines = [np.vstack(pol2cart(l, mid_line_polar[:,1])).T for l in contour_r_upper_polar][1:]
    
#     all_dist_lines = contour_r_lower_lines + [mid_line_central] + contour_r_upper_lines
#     all_dist_lines = [np.vstack([ll[:,0] + center[1], ll[:,1] + center[0]]).T  for ll in all_dist_lines]
#     all_dist_masks = [draw_polygon_mask(ll, imsize) for ll in all_dist_lines]
#     # now construct binary masks. 
#     all_dist_masks = [np.logical_xor(all_dist_masks[ii+1], all_dist_masks[ii]) for ii in range(len(all_dist_masks)-1)] # ascending distance. 
    
#     """
#     construct angle masks to partition the angle space. 
#     """
#     # 2. partition the angular direction
#     angle_masks_list = [] # list of the binary masks in ascending angles. 
#     theta_bounds = np.linspace(-np.pi, np.pi, n_theta+1)
    
#     for ii in range(len(theta_bounds)-1):
#         mask_theta = np.logical_and( theta>=theta_bounds[ii], theta <= theta_bounds[ii+1])
#         angle_masks_list.append(mask_theta)
        
#     """
#     construct the final set of masks.  
#     """  
#     spixels = np.zeros((m,n), dtype=np.int)
    
#     counter = 1
#     for ii in range(len(all_dist_masks)):
#         for jj in range(len(angle_masks_list)):
#             mask = np.logical_and(all_dist_masks[ii], angle_masks_list[jj])

#             spixels[mask] = counter 
#             counter += 1
        
#     return spixels 


def tile_uniform_windows_radial_guided_line(imsize, n_r, n_theta, max_r, mid_line, center=None, bound_r=True, zero_angle=None, return_grid_pts=False):
    
    from skimage.segmentation import mark_boundaries
    m, n = imsize
    
    XX, YY = np.meshgrid(range(n), range(m))
    
    if center is None:
        center = np.hstack([m/2., n/2.])

    r2 = (XX - center[1])**2  + (YY - center[0])**2
    r = np.sqrt(r2)
    theta = np.arctan2(YY-center[0],XX-center[1])
    
    if max_r is None:
        if bound_r: 
            max_r = np.minimum(np.abs(np.max(XX-center[1])),
                               np.abs(np.max(YY-center[0])))
        else:
            max_r = np.maximum(np.max(np.abs(XX-center[1])),
                               np.max(np.abs(YY-center[0])))

    """
    construct contour lines that bipartition the space. 
    """
    mid_line_polar = np.vstack(cart2pol(mid_line[:,0] - center[1], mid_line[:,1] - center[0])).T
    mid_line_central = np.vstack(pol2cart(mid_line_polar[:,0], mid_line_polar[:,1])).T
    
    # derive lower and upper boundary lines -> make sure to 
    contour_r_lower_polar = np.array([np.linspace(0, l, n_r//2+1) for l in mid_line_polar[:,0]]).T
    contour_r_upper_polar = np.array([np.linspace(l, max_r, n_r//2+1) for l in mid_line_polar[:,0]]).T
    
    contour_r_lower_lines = [np.vstack(pol2cart(l, mid_line_polar[:,1])).T for l in contour_r_lower_polar][:-1]
    contour_r_upper_lines = [np.vstack(pol2cart(l, mid_line_polar[:,1])).T for l in contour_r_upper_polar][1:]
    
    all_dist_lines = contour_r_lower_lines + [mid_line_central] + contour_r_upper_lines
    all_dist_lines = [np.vstack([ll[:,0] + center[1], ll[:,1] + center[0]]).T  for ll in all_dist_lines]
    all_dist_masks = [draw_polygon_mask(ll, imsize) for ll in all_dist_lines]
    # now construct binary masks. 
    all_dist_masks = [np.logical_xor(all_dist_masks[ii+1], all_dist_masks[ii]) for ii in range(len(all_dist_masks)-1)] # ascending distance. 
    
    """
    construct angle masks to partition the angle space. 
    """
    # 2. partition the angular direction
    angle_masks_list = [] # list of the binary masks in ascending angles.
    
    theta = theta + np.pi
    
    if zero_angle is None:
        theta_bounds = np.linspace(0, 2*np.pi, n_theta+1)
    else:
#        print(np.linspace(0, 2*np.pi, n_theta+1) + (180+360./n_theta/2.)/180.*np.pi + zero_angle/180.*np.pi)
#        theta_bounds = np.mod(np.linspace(0, 2*np.pi, n_theta+1), 2*np.pi) 
        theta_bounds = np.mod(np.linspace(0, 2*np.pi, n_theta+1) + (180.-360./n_theta/2.)/180.*np.pi - zero_angle/180.*np.pi, 2*np.pi)
        print(theta_bounds)

    for ii in range(len(theta_bounds)-1):
        
        #### this works if all angles are within the 0 to 2 pi range. 
        if theta_bounds[ii+1] > theta_bounds[ii]:
            mask_theta = np.logical_and( theta>=theta_bounds[ii], theta <= theta_bounds[ii+1])
        else:
            mask_theta = np.logical_or(np.logical_and(theta>=theta_bounds[ii], theta<=2*np.pi), 
                                       np.logical_and(theta>=0, theta<=theta_bounds[ii+1]))
        angle_masks_list.append(mask_theta)
        
    """
    construct the final set of masks (which is for the angles. )  
    """  
    spixels = np.zeros((m,n), dtype=np.int)
    
    counter = 1
    for ii in range(len(all_dist_masks)):
        for jj in range(len(angle_masks_list)):
            mask = np.logical_and(all_dist_masks[ii], angle_masks_list[jj])

            spixels[mask] = counter 
            counter += 1
        
    if return_grid_pts:
        return spixels, [all_dist_lines, theta_bounds]
    else:
        return spixels 

def resample_curve(x,y,s=0, n_samples=10):
    
    import scipy.interpolate
    
    tck, u = scipy.interpolate.splprep([x,y], s=s)
    unew = np.linspace(0, 1., n_samples)
    out = scipy.interpolate.splev(unew, tck) 
    
    return np.vstack(out).T

def infer_control_pts_xy(radial_lines,thetas, ref_pt,s=0, resolution=360):

    from sklearn.metrics.pairwise import pairwise_distances
    from scipy.optimize import linear_sum_assignment
    # given closed curves and a set of angles, pull out the intersection points.
    grid_pts = []

    for rad_line in radial_lines[1:]:

        rad_line_interp = resample_curve(rad_line[:,0], rad_line[:,1], s=s, n_samples=resolution+1)
        rad_line_thetas = np.arctan2(rad_line_interp[:,1]-ref_pt[1], 
                                     rad_line_interp[:,0]-ref_pt[0])

        # rad_line_thetas[rad_line_thetas<0]  = rad_line_thetas[rad_line_thetas<0] + 2*np.pi # make [0, 2*np.pi]
        rad_line_thetas = rad_line_thetas + np.pi # to be consistent with the way grid is done.

        # find the minimum distance to the query thetas.
        dist_matrix = pairwise_distances(thetas[:,None], rad_line_thetas[:,None])

        i, j = linear_sum_assignment(dist_matrix)
        # print(i,j)
        
        found_pts = np.squeeze(rad_line_interp[j])
        dist_ref = np.linalg.norm(found_pts - ref_pt[None,:], axis=-1)

        # the reconstruction doesn't quite match...
        constructed_pts = np.vstack([dist_ref*np.cos(thetas-np.pi) + ref_pt[0], 
                                     dist_ref*np.sin(thetas-np.pi) + ref_pt[1]]).T
        grid_pts.append(constructed_pts[:-1])
        # grid_pts.append(found_pts[:-1])

    return np.array(grid_pts)


def lookup_3D_to_2D_directions(unwrap_params, ref_pos_2D, disp_3D, scale=10.):
#    
    disp_3D_ = disp_3D / float(np.linalg.norm(disp_3D) + 1e-8) # MUST normalise
    ref_pos_3D = unwrap_params[int(ref_pos_2D[0]), 
                               int(ref_pos_2D[1])]
    
    new_pos_3D = ref_pos_3D + scale*disp_3D_
    
    min_dists = np.linalg.norm(unwrap_params - new_pos_3D[None,None,:], axis=-1) # require the surface normal ? 
    
#    plt.figure()
#    plt.imshow(min_dists, cmap='coolwarm')
#    plt.plot
#    plt.show()
    min_pos = np.argmin(min_dists)
    new_pos_2D = np.unravel_index(min_pos, unwrap_params.shape[:-1]) # returns as the index coords.
    
##    print(new_pos_2D)
#    plt.figure()
#    plt.imshow(min_dists, cmap='coolwarm')
#    plt.plot(ref_pos_2D[1], ref_pos_2D[0], 'ko')
#    plt.plot(new_pos_2D[1], new_pos_2D[0], 'go')
#    plt.show()
    
#    print(new_pos_2D)
    direction_2D = new_pos_2D - ref_pos_2D
    return direction_2D, np.arctan2(direction_2D[1], direction_2D[0])


def lookup_3D_to_2D_loc_and_directions(unwrap_params, mean_3D, disp_3D_1, disp_3D_2, scale=10.):

    from sklearn.neighbors import NearestNeighbors
    
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(unwrap_params.reshape(-1,3))
    
    # normalise the 3D displacement.
    disp_3D_1_ = disp_3D_1 / (np.linalg.norm(disp_3D_1, axis=-1)[...,None] + 1e-8) # MUST normalise
    new_pos_3D_1 = mean_3D + scale*disp_3D_1_
    disp_3D_2_ = disp_3D_2 / (np.linalg.norm(disp_3D_2, axis=-1)[...,None] + 1e-8) # MUST normalise
    new_pos_3D_2 = mean_3D + scale*disp_3D_2_
    
    pos2D_map_indices = nbrs.kneighbors(mean_3D, return_distance=False)
    pos2D_map_indices_disp_1 = nbrs.kneighbors(new_pos_3D_1, return_distance=False)
    pos2D_map_indices_disp_2 = nbrs.kneighbors(new_pos_3D_2, return_distance=False)
    
    pos2D_map_indices = pos2D_map_indices.ravel()
    pos2D_map_indices_disp_1 = pos2D_map_indices_disp_1.ravel()
    pos2D_map_indices_disp_2 = pos2D_map_indices_disp_2.ravel()
#    print(pos2D_map_indices).shape
#    print(pos2D_map_indices_disp.shape)
#    print(pos2D_map_indices[0])
    
    pos2D_map_indices_ij = np.unravel_index(pos2D_map_indices, unwrap_params.shape[:-1])
    pos2D_map_indices_disp_ij_1 = np.unravel_index(pos2D_map_indices_disp_1, unwrap_params.shape[:-1])
    pos2D_map_indices_disp_ij_2 = np.unravel_index(pos2D_map_indices_disp_2, unwrap_params.shape[:-1])

    pos2D_map_indices_ij = np.vstack(pos2D_map_indices_ij).T
    pos2D_map_indices_disp_ij_1 = np.vstack(pos2D_map_indices_disp_ij_1).T
    pos2D_map_indices_disp_ij_2 = np.vstack(pos2D_map_indices_disp_ij_2).T
    
#    print(pos2D_map_indices_disp_ij.shape)

    direction_2D_1 = pos2D_map_indices_disp_ij_1 - pos2D_map_indices_ij
    direction_2D_1 = direction_2D_1 / (np.linalg.norm(direction_2D_1, axis=-1)[...,None] + 1e-8 )
    direction_2D_2 = pos2D_map_indices_disp_ij_2 - pos2D_map_indices_ij
    direction_2D_2 = direction_2D_2 / (np.linalg.norm(direction_2D_2, axis=-1)[...,None] + 1e-8 )
    
    return pos2D_map_indices_ij, direction_2D_1, direction_2D_2


def lookup_3D_mean(pts2D, unwrap_params):
    
    pts3D = unwrap_params[pts2D[:,0].astype(np.int), 
                          pts2D[:,1].astype(np.int)]
    
    pts3D_mean = np.median(pts3D, axis=0)
    
    pts2D_mean_remap = np.argmin(np.linalg.norm(unwrap_params - pts3D_mean[None,None,:], axis=-1))
    pts2D_mean_remap = np.unravel_index(pts2D_mean_remap, 
                                        unwrap_params.shape[:-1])
    
    return np.hstack(pts2D_mean_remap)

def geodesic_distance(pt1, pt2, unwrap_params, n_samples = 10):

    # helper function to compute a geodesic line along the 3D surface.
    from sklearn.neighbors import NearestNeighbors 
    nbrs_model = NearestNeighbors(n_neighbors=1, algorithm='auto') 

    pts3D_all = unwrap_params.reshape(-1,unwrap_params.shape[-1])
    nbrs_model.fit(pts3D_all)

    pt1_3D = unwrap_params[int(pt1[0]), int(pt1[1])]
    pt2_3D = unwrap_params[int(pt2[0]), int(pt2[1])]

    line_x = np.linspace(pt1_3D[0], pt2_3D[0], n_samples)
    line_y = np.linspace(pt1_3D[1], pt2_3D[1], n_samples)
    line_z = np.linspace(pt1_3D[2], pt2_3D[2], n_samples)

    line_xyz = np.vstack([line_x,line_y,line_z]).T

    _, neighbor_indices = nbrs_model.kneighbors(line_xyz)
    
    neighbor_indices = np.hstack(neighbor_indices)

    line_surf_pts = pts3D_all[neighbor_indices]

    print(line_surf_pts.shape)
    geodesic_dist = np.sum(np.linalg.norm(line_surf_pts[1:] - line_surf_pts[:-1], axis=-1))

    return line_surf_pts, geodesic_dist


def rotate_pts(pts, angle=0, center=[0,0]):
    
    angle_rads = angle / 180. * np.pi
    rot_matrix = np.zeros((2,2))
    rot_matrix[0,0] = np.cos(angle_rads)
    rot_matrix[0,1] = -np.sin(angle_rads)
    rot_matrix[1,0] = np.sin(angle_rads)
    rot_matrix[1,1] = np.cos(angle_rads)
    
#    print(rot_matrix)
    center_ = np.hstack(center)[None,:]
    pts_ = rot_matrix.dot((pts-center_).T).T + center_
    
    return pts_


def rotate_tracks(tracks, angle=0, center=[0,0], transpose=True):
    
    tracks_ = tracks.copy()
    if transpose:
        tracks_ = tracks_[...,::-1]

    tracks_rot = []
    
    for frame in range(tracks.shape[1]):
        track_pts = tracks_[:,frame]
        track_pts_rot = rotate_pts(track_pts, angle=angle, center=center)
        tracks_rot.append(track_pts_rot[:,None,:])
        
    tracks_rot = np.concatenate(tracks_rot, axis=1)
    
    if transpose:
        tracks_rot = tracks_rot[...,::-1]
    
    return tracks_rot

def smooth_flow_ds(flow, sigma, ds=4):

    import skimage.transform as sktform
    from scipy.signal import medfilt, medfilt2d, convolve2d
    import skimage.morphology as skmorph

    m,n,ch = flow.shape
    flow_frame = sktform.resize(flow, (m//ds, n//ds, ch), preserve_range=True)

    mean_kernel = skmorph.disk(sigma).astype(np.float32) 
    mean_kernel = mean_kernel / float(np.sum(mean_kernel))

    flow_frame[...,0] = convolve2d(flow_frame[...,0], mean_kernel, mode='same', boundary='symm')
    flow_frame[...,1] = convolve2d(flow_frame[...,1], mean_kernel, mode='same', boundary='symm')
                            
    flow_frame = sktform.resize(flow_frame, flow.shape, preserve_range=True)

    return flow_frame

def plot_grid_ax(ax, gridlines, color='w', lw=1, style='-.'):

    n_r, n_theta, _ = gridlines.shape

    for ii in range(n_r):
        ax.plot(gridlines[ii,:,0], 
                gridlines[ii,:,1], style, color=color, lw=lw)

    for jj in range(n_theta):
        ax.plot(gridlines[:,jj,0], 
                gridlines[:,jj,1], style, color=color, lw=lw)

    return []


def parse_grid_squares_from_pts(gridlines):

    # points are arranged by increasing radial distance, increasing polar angle. 
    # therefore should be a case of --->|
    #                                   |
    #                                   \/

    grid_squares = []

    n_r, n_theta, _ = gridlines.shape

    for ii in range(n_r-1):
        for jj in range(n_theta-1):
            top_left = gridlines[ii,jj]
            top_right = gridlines[ii,jj+1]
            bottom_right = gridlines[ii+1,jj+1]
            bottom_left = gridlines[ii+1,jj]

            grid_coords = np.vstack([top_left, top_right, bottom_right, bottom_left, top_left]) # create a complete circulation.
            grid_squares.append(grid_coords)

    grid_squares = np.array(grid_squares)

    return grid_squares


def plot_grid_squares_ax(ax, squarelines, color='w', lw=1):

    for squareline in squarelines:
        if color is not None:
            ax.plot(squareline[:,0], 
                    squareline[:,1], color=color, lw=lw)
        else:
            ax.plot(squareline[:,0], 
                    squareline[:,1], lw=lw)

    return []


def area_3D_mesh(grid_squares_2D, unwrap_params_3D, emb_centroid, nNeighbours=50):

    area_squares = []

    for sq in grid_squares_2D:

        # vertices of the square.
        yy = sq[:,1].astype(np.int)
        xx = sq[:,0].astype(np.int)

        sq_verts_3D = unwrap_params_3D[yy,xx]
        centroid_3D = np.mean(sq_verts_3D[:-1], axis=0)

        # surf_normal, surf_curvature = findPointNormals((centroid_3D)[None,:], 
        #                                                 unwrap_params_3D.reshape(-1,3), 
        #                                                 nNeighbours=nNeighbours, 
        #                                                 viewPoint=emb_centroid, 
        #                                                 dirLargest=True)
        # surf_normal = surf_normal[0]
        # surf_normal = surf_normal / (np.linalg.norm(surf_normal)+ 1e-8)

        surf_area = poly_area(sq_verts_3D, normal=None)
        area_squares.append(np.abs(surf_area))

    return np.hstack(area_squares)

def area_3D_mesh_fast(grid_squares_2D, unwrap_params_3D):

    from skimage.measure import mesh_surface_area
    # verts : (V, 3) array of floats
    #     Array containing (x, y, z) coordinates for V unique mesh vertices.
    # faces : (F, 3) array of ints
    #     List of length-3 lists of integers, referencing vertex coordinates as
    #     provided in `verts`

    areas = []

    # counter = 0 

    for sq in grid_squares_2D:

        # vertices of the square.
        yy = sq[:,1].astype(np.int)
        xx = sq[:,0].astype(np.int)

        sq_verts_3D = unwrap_params_3D[yy,xx]
        # verts.append(sq_verts_3D[:4])
        # verts_ids = counter + np.arange(4)

        verts = sq_verts_3D[:-1]
        faces = np.vstack([[0,1,2], [2,3,0]])
        # tri_1 = verts_ids[[0,1,2]]
        # tri_2 = verts_ids[[2,3,0]]
        # faces.append(tri_1)
        # faces.append(tri_2)

        # counter += 4
        area = mesh_surface_area(verts,faces)
        areas.append(area)

    # verts = np.vstack(verts)
    # faces = np.vstack(faces)

    # areas_faces = mesh_surface_area(verts,faces)

    return np.hstack(areas)

def heatmap_quad_vals(grid_squares_2D, vals, shape):

    from skimage.draw import polygon
    
    blank = np.zeros(shape); blank[:,:] = np.nan

    for sq_ii, sq in enumerate(grid_squares_2D):

        rr,cc = polygon(sq[:,1], sq[:,0], shape)
        blank[rr,cc] = vals[sq_ii]

    return blank


def avg_quad_vals(grid_squares_2D, vals, shape, avg_func=np.mean):

    from skimage.draw import polygon
    
    blank = np.zeros(shape); blank[:,:] = np.nan

    for sq_ii, sq in enumerate(grid_squares_2D):

        rr,cc = polygon(sq[:,1], sq[:,0], shape)
        blank[rr,cc] = avg_func(vals[rr,cc])

    return blank


#  helper coder for polygon area
def findPointNormals(query_points, ref_points, nNeighbours, viewPoint=[0,0,0], dirLargest=True):
    
    """
    construct kNN and estimate normals from the local PCA
    
    reference: https://uk.mathworks.com/matlabcentral/fileexchange/48111-find-3d-normals-and-curvature
    """
    # construct kNN object to look for nearest neighbours. 
    from sklearn.neighbors import NearestNeighbors
    
    neigh = NearestNeighbors(n_neighbors=nNeighbours+1)
    neigh.fit(ref_points)
    
    nn_inds = neigh.kneighbors(query_points, return_distance=False) # also get the distance, the distance is used for cov computation.
    nn_inds = nn_inds[:,1:] # remove self
    
    # find difference in position from neighbouring points (#technically this should be relative to the centroid of the patch!)
    # refine this computation. to take into account the central points. 
#    p = points[:,None,:] - points[nn_inds]
    p = ref_points[nn_inds] - (ref_points[nn_inds].mean(axis=1))[:,None,:]
    
    # compute covariance
    C = np.zeros((len(query_points), 6))
    C[:,0] = np.sum(p[:,:,0]*p[:,:,0], axis=1)
    C[:,1] = np.sum(p[:,:,0]*p[:,:,1], axis=1)
    C[:,2] = np.sum(p[:,:,0]*p[:,:,2], axis=1)
    C[:,3] = np.sum(p[:,:,1]*p[:,:,1], axis=1)
    C[:,4] = np.sum(p[:,:,1]*p[:,:,2], axis=1)
    C[:,5] = np.sum(p[:,:,2]*p[:,:,2], axis=1)
    C = C / float(nNeighbours)
    
    # normals and curvature calculation 
    normals = np.zeros(query_points.shape)
    curvature = np.zeros((len(query_points)))
    
    for i in range(len(query_points))[:]:
        
        # form covariance matrix
        Cmat = np.array([[C[i,0],C[i,1],C[i,2]],
                [C[i,1],C[i,3],C[i,4]],
                [C[i,2],C[i,4],C[i,5]]])
        
        # get eigen values and vectors
        [d,v] = np.linalg.eigh(Cmat);
        
#        d = np.diag(d);
        k = np.argmin(d)
        lam = d[k]
        
        # store normals
        normals[i,:] = v[:,k]
        
        #store curvature
        curvature[i] = lam / (np.sum(d) + 1e-15);

    # flipping normals to point towards viewpoints
    #ensure normals point towards viewPoint
    query_points = query_points - np.array(viewPoint).ravel()[None,:]; # this is outward facing

    if dirLargest:
        idx = np.argmax(np.abs(normals), axis=1)
        dir = normals[np.arange(len(idx)),idx]*query_points[np.arange(len(idx)),idx] < 0;
    else:
        dir = np.sum(normals*query_points,axis=1) < 0;
    
    normals[dir,:] = -normals[dir,:];
    
    return normals, curvature

#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#area of polygon poly
def poly_area(poly, normal=None):
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
        
    if normal is None:
        result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    else:
        result = np.dot(total, normal)
    return np.abs(result/2)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    return []


def pca(data, numComponents=None):
    """Principal Components Analysis

    From: http://stackoverflow.com/a/13224592/834250

    Parameters
    ----------
    data : `numpy.ndarray`
        numpy array of data to analyse
    numComponents : `int`
        number of principal components to use

    Returns
    -------
    comps : `numpy.ndarray`
        Principal components
    evals : `numpy.ndarray`
        Eigenvalues
    evecs : `numpy.ndarray`
        Eigenvectors
    """
    m, n = data.shape
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    if numComponents is not None:
        evecs = evecs[:, :numComponents]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs


#def decompose_grid_strain_changes(grid_time, unwrap_params, neighbours=35):
#
#    """
#    """
    
# this is actually a radial and theta gradient function. hm...
def compute_directional_vectors(unwrap_params, r_dist=5, theta_dist=1, smooth=False, smooth_theta_dist=10):
    """
    use taylor expansion.
    """
    from Geometry.meshtools import smooth_vects_neighbours
    
    m, n = unwrap_params.shape[:2]
    XX, YY = np.meshgrid(range(unwrap_params.shape[1]), range(unwrap_params.shape[0]))
    
    disps_XX = (XX-XX.shape[1]/2.)
    disps_YY = (YY-YY.shape[0]/2.)
    
    # parametrise the circular grid.
    dist_grid = np.sqrt((disps_XX)**2 + (disps_YY)**2)
    arg_grid = np.arctan2(disps_YY, disps_XX)
    
    # move infinitesimally in the outwards direction.
    XX_r = XX + (r_dist*disps_XX/(dist_grid+1e-8))
    YY_r = YY + (r_dist*disps_YY/(dist_grid+1e-8))
    
    XX_theta = dist_grid * np.cos(arg_grid+theta_dist/180.*np.pi) + XX.shape[1]/2.
    YY_theta = dist_grid * np.sin(arg_grid+theta_dist/180.*np.pi) + YY.shape[0]/2.

    XX_r = np.clip(XX_r, 0, n-1); XX_theta = np.clip(XX_theta,0,n-1)
    YY_r = np.clip(YY_r, 0, m-1); YY_theta = np.clip(YY_theta,0,m-1)
    
    radial_vects = unwrap_params[YY_r.astype(np.int), XX_r.astype(np.int)] - unwrap_params[YY.astype(np.int), XX.astype(np.int)]
#    radial_vects = radial_vects / (np.linalg.norm(radial_vects, axis=-1)+1e-8)[...,None]
    theta_vects = unwrap_params[YY_theta.astype(np.int), XX_theta.astype(np.int)] - unwrap_params[YY.astype(np.int), XX.astype(np.int)]

    if smooth:
        
        # can also do distance smoothing.
        
        
        # we do radial smoothing. 
        # smooth in a line manner. 
        XX_theta_line = dist_grid[...,None] * (np.cos(arg_grid[...,None]+np.arange(-smooth_theta_dist//2, smooth_theta_dist//2+1)[None,None,:]/180.*np.pi)) + XX.shape[1]/2.
        YY_theta_line = dist_grid[...,None] * (np.sin(arg_grid[...,None]+np.arange(-smooth_theta_dist//2, smooth_theta_dist//2+1)[None,None,:]/180.*np.pi)) + YY.shape[0]/2.

        XX_theta_line = np.clip(XX_theta_line, 0, n-1); 
        YY_theta_line = np.clip(YY_theta_line, 0, m-1); 

        radial_vects = np.nanmean(np.array([radial_vects[YY_theta_line[...,jj].astype(np.int), XX_theta_line[...,jj].astype(np.int)] for jj in range(XX_theta_line.shape[-1])]), axis=0)
        theta_vects = np.nanmean(np.array([theta_vects[YY_theta_line[...,jj].astype(np.int), XX_theta_line[...,jj].astype(np.int)] for jj in range(XX_theta_line.shape[-1])]), axis=0)

    radial_vects = radial_vects / (np.linalg.norm(radial_vects, axis=-1)+1e-8)[...,None]
    theta_vects = theta_vects / (np.linalg.norm(theta_vects, axis=-1)+1e-8)[...,None]
    
    return radial_vects, theta_vects


def compute_directional_gradients(vals, unwrap_params, r_dist=5, theta_dist=1, smooth=False, smooth_theta_dist=10):
    """
    use taylor expansion.
    """
    from Geometry.meshtools import smooth_vects_neighbours
    
    m, n = unwrap_params.shape[:2]
    XX, YY = np.meshgrid(range(unwrap_params.shape[1]), range(unwrap_params.shape[0]))
    
    disps_XX = (XX-XX.shape[1]/2.)
    disps_YY = (YY-YY.shape[0]/2.)
    
    # parametrise the circular grid.
    dist_grid = np.sqrt((disps_XX)**2 + (disps_YY)**2)
    arg_grid = np.arctan2(disps_YY, disps_XX)
    
    # move infinitesimally in the outwards direction.
    XX_r = XX + (r_dist*disps_XX/(dist_grid+1e-8))
    YY_r = YY + (r_dist*disps_YY/(dist_grid+1e-8))
    
    XX_theta = dist_grid * np.cos(arg_grid+theta_dist/180.*np.pi) + XX.shape[1]/2.
    YY_theta = dist_grid * np.sin(arg_grid+theta_dist/180.*np.pi) + YY.shape[0]/2.

    XX_r = np.clip(XX_r, 0, n-1); XX_theta = np.clip(XX_theta,0,n-1)
    YY_r = np.clip(YY_r, 0, m-1); YY_theta = np.clip(YY_theta,0,m-1)

    radial_vals = vals[YY_r.astype(np.int), XX_r.astype(np.int)] - vals[YY.astype(np.int), XX.astype(np.int)]
    theta_vals = vals[YY_theta.astype(np.int), XX_theta.astype(np.int)] - vals[YY.astype(np.int), XX.astype(np.int)]

    radial_vects = unwrap_params[YY_r.astype(np.int), XX_r.astype(np.int)] - unwrap_params[YY.astype(np.int), XX.astype(np.int)]
    theta_vects = unwrap_params[YY_theta.astype(np.int), XX_theta.astype(np.int)] - unwrap_params[YY.astype(np.int), XX.astype(np.int)]

    radial_vects = np.linalg.norm(radial_vects, axis=-1)
    theta_vects = np.linalg.norm(theta_vects, axis=-1)
    
    radial_vals = radial_vals/(radial_vects + 1e-8)
    theta_vals = theta_vals/(radial_vects + 1e-8)

    if smooth:
        
        # we do radial smoothing. 
        # smooth in a line manner. 
        XX_theta_line = dist_grid[...,None] * (np.cos(arg_grid[...,None]+np.arange(-smooth_theta_dist//2, smooth_theta_dist//2+1)[None,None,:]/180.*np.pi)) + XX.shape[1]/2.
        YY_theta_line = dist_grid[...,None] * (np.sin(arg_grid[...,None]+np.arange(-smooth_theta_dist//2, smooth_theta_dist//2+1)[None,None,:]/180.*np.pi)) + YY.shape[0]/2.

        XX_theta_line = np.clip(XX_theta_line, 0, n-1); 
        YY_theta_line = np.clip(YY_theta_line, 0, m-1); 

        radial_vals = np.nanmean(np.array([radial_vals[YY_theta_line[...,jj].astype(np.int), XX_theta_line[...,jj].astype(np.int)] for jj in range(XX_theta_line.shape[-1])]), axis=0)
        theta_vals = np.nanmean(np.array([theta_vals[YY_theta_line[...,jj].astype(np.int), XX_theta_line[...,jj].astype(np.int)] for jj in range(XX_theta_line.shape[-1])]), axis=0)

#    radial_vects = radial_vects / (np.linalg.norm(radial_vects, axis=-1)+1e-8)[...,None]
#    theta_vects = theta_vects / (np.linalg.norm(theta_vects, axis=-1)+1e-8)[...,None]
    return radial_vals, theta_vals


def compute_directional_gradients_time(vals, unwrap_params, r_dist=5, theta_dist=1, smooth=False, smooth_theta_dist=10):
    """
    use taylor expansion.
    """
    from Geometry.meshtools import smooth_vects_neighbours
    
    m, n = unwrap_params.shape[:2]
    XX, YY = np.meshgrid(range(unwrap_params.shape[1]), range(unwrap_params.shape[0]))
    
    disps_XX = (XX-XX.shape[1]/2.)
    disps_YY = (YY-YY.shape[0]/2.)
    
    # parametrise the circular grid.
    dist_grid = np.sqrt((disps_XX)**2 + (disps_YY)**2)
    arg_grid = np.arctan2(disps_YY, disps_XX)
    
    # move infinitesimally in the outwards direction.
    XX_r = XX + (r_dist*disps_XX/(dist_grid+1e-8))
    YY_r = YY + (r_dist*disps_YY/(dist_grid+1e-8))
    
    XX_theta = dist_grid * np.cos(arg_grid+theta_dist/180.*np.pi) + XX.shape[1]/2.
    YY_theta = dist_grid * np.sin(arg_grid+theta_dist/180.*np.pi) + YY.shape[0]/2.

    XX_r = np.clip(XX_r, 0, n-1); XX_theta = np.clip(XX_theta,0,n-1)
    YY_r = np.clip(YY_r, 0, m-1); YY_theta = np.clip(YY_theta,0,m-1)

    radial_vals = vals[:,YY_r.astype(np.int), XX_r.astype(np.int)] - vals[:,YY.astype(np.int), XX.astype(np.int)]
    theta_vals = vals[:,YY_theta.astype(np.int), XX_theta.astype(np.int)] - vals[:,YY.astype(np.int), XX.astype(np.int)]

    radial_vects = unwrap_params[YY_r.astype(np.int), XX_r.astype(np.int)] - unwrap_params[YY.astype(np.int), XX.astype(np.int)]
    theta_vects = unwrap_params[YY_theta.astype(np.int), XX_theta.astype(np.int)] - unwrap_params[YY.astype(np.int), XX.astype(np.int)]

    radial_vects = np.linalg.norm(radial_vects, axis=-1)
    theta_vects = np.linalg.norm(theta_vects, axis=-1)
    
    radial_vals = radial_vals/(radial_vects[None,:] + 1e-8)
    theta_vals = theta_vals/(radial_vects[None,:] + 1e-8)

    if smooth:
        
        # we do radial smoothing. 
        # smooth in a line manner. 
        XX_theta_line = dist_grid[...,None] * (np.cos(arg_grid[...,None]+np.arange(-smooth_theta_dist//2, smooth_theta_dist//2+1)[None,None,:]/180.*np.pi)) + XX.shape[1]/2.
        YY_theta_line = dist_grid[...,None] * (np.sin(arg_grid[...,None]+np.arange(-smooth_theta_dist//2, smooth_theta_dist//2+1)[None,None,:]/180.*np.pi)) + YY.shape[0]/2.

        XX_theta_line = np.clip(XX_theta_line, 0, n-1); 
        YY_theta_line = np.clip(YY_theta_line, 0, m-1); 

        radial_vals = np.nanmean(np.array([radial_vals[:,YY_theta_line[...,jj].astype(np.int), XX_theta_line[...,jj].astype(np.int)] for jj in range(XX_theta_line.shape[-1])]), axis=0)
        theta_vals = np.nanmean(np.array([theta_vals[:,YY_theta_line[...,jj].astype(np.int), XX_theta_line[...,jj].astype(np.int)] for jj in range(XX_theta_line.shape[-1])]), axis=0)

#    radial_vects = radial_vects / (np.linalg.norm(radial_vects, axis=-1)+1e-8)[...,None]
#    theta_vects = theta_vects / (np.linalg.norm(theta_vects, axis=-1)+1e-8)[...,None]
    return radial_vals, theta_vals
    


def radial_smoothing(img, bins_r, bins_theta, max_r=1.1):
    
    from Geometry.meshtools import smooth_vects_neighbours
    
    m, n = img.shape[:2]
    XX, YY = np.meshgrid(range(n), 
                         range(n))
    
    disps_XX = (XX-XX.shape[1]/2.)
    disps_YY = (YY-YY.shape[0]/2.)
    
    # parametrise the circular grid.
    dist_grid = np.sqrt((disps_XX)**2 + (disps_YY)**2)
    arg_grid = np.arctan2(disps_YY, disps_XX)
    
    
    bin_dist = np.linspace(0, n/2./max_r, bins_r+1)
    bin_theta = np.linspace(-np.pi, np.pi, bins_theta+1)
    bin_theta = np.hstack([bin_theta, bin_theta[0]]) # periodic
    
    img_new = np.zeros_like(img)
    
    # smooth in radial first.... 
    for ii in range(len(bin_dist)-1):
        for jj in range(len(bin_theta)-2):
            
            # construct bin 0.
            select_dist_0 = np.logical_and(dist_grid>=bin_dist[ii], dist_grid<=bin_dist[ii+1])
            select_theta_0 = np.logical_and(arg_grid>=bin_theta[jj], arg_grid<=bin_theta[jj+1])
            select_0 = np.logical_and(select_dist_0, select_theta_0)
            
            select_dist_1 = np.logical_and(dist_grid>=bin_dist[ii], dist_grid<=bin_dist[ii+1])
            select_theta_1 = np.logical_and(arg_grid>=bin_theta[jj+1], arg_grid<=bin_theta[jj+2])
            select_1 = np.logical_and(select_dist_1, select_theta_1)
            
            vals0 = img[select_0]
            vals1 = img[select_1]
            
            
            img_new[select_0] = np.mean(np.vstack([vals0, vals1]), axis=0)  
    
    
    return img_new


#def decompose_grid_strain_rate(grid_pts_time, unwrap_params_3D, decomposition_directions):
#    
#    """
#    decomposition comes from formula give in
#    "Myosin-II-mediated cell shape changes and cell intercalation contribute to primitive streak formation" - https://www.nature.com/articles/ncb3138
#    
#    """
##    directions_r, directions_theta = decomposition_directions 
#    
#    # the square is closed. n_time x n_squares x n_vertex points.
#    grid_pts_1 = unwrap_params_3D[grid_pts_time[1:,...,1].astype(np.int), 
#                                  grid_pts_time[1:,...,0].astype(np.int)]
#    grid_pts_0 = unwrap_params_3D[grid_pts_time[:-1,...,1].astype(np.int), 
#                                  grid_pts_time[:-1,...,0].astype(np.int)]
#    
#    directions_r = decomposition_directions[0][grid_pts_time[:-1,:,:-1,1].astype(np.int), 
#                                                       grid_pts_time[:-1,:,:-1,0].astype(np.int)]
#    directions_theta = decomposition_directions[1][grid_pts_time[:-1,:,:-1,1].astype(np.int), 
#                                                           grid_pts_time[:-1,:,:-1,0].astype(np.int)]
#    
#    diff_grid_pts_3D = grid_pts_1 - grid_pts_0
#    diff_grid_pts_3D = diff_grid_pts_3D[:,:,:4] # since the 5th point is only used to close.
#    
#    
#    print(diff_grid_pts_3D.shape)
#    print(directions_theta.shape)
#    
#    diff_grid_pts_3D_dr = np.mean(np.sum(diff_grid_pts_3D * directions_r, axis=-1), axis=-1)
#    diff_grid_pts_3D_theta = np.mean(np.sum(diff_grid_pts_3D * directions_theta, axis=-1), axis=-1)
#    
#    # technically the remainder radial differentiation 
#    # can only be computed using approximate methods.... 
#    # we now construct the elements of the deformation in principal directions.
#    dr_grid_pts_3D_dr = np.sum(diff_grid_pts_3D * directions_r, axis=-1)
#    dr_grid_pts_3D_dtheta = np.sum(diff_grid_pts_3D * directions_theta, axis=-1)
#    dtheta_grid_pts_3D_dr = np.sum(diff_grid_pts_3D * directions_r, axis=-1)
#    dtheta_grid_pts_3D_dtheta = np.sum(diff_grid_pts_3D * directions_theta, axis=-1)
#    
#    print(dr_grid_pts_3D_dr.shape)
    
    
def continuous_plane_strain_rate(velocity_time, unwrap_params_3D, decomposition_directions):
    
    # project 2D velocities into 3D velocities. 
    nframes, m, n, _ = velocity_time.shape

    XX, YY = np.meshgrid(range(n), range(m))
    
    YY1 = np.clip((YY[None,:]+velocity_time[...,1]).astype(np.int), 0, m-1)
    XX1 = np.clip((XX[None,:]+velocity_time[...,0]).astype(np.int), 0, n-1)
    velocity_time_3D = unwrap_params_3D[YY1, XX1] - unwrap_params_3D[YY,XX]
    
    # project 3D velocities into radial and angular directions
    velocity_time_3D_radial = np.sum(velocity_time_3D * decomposition_directions[0][None,:] , axis=-1)
    velocity_time_3D_theta = np.sum(velocity_time_3D * decomposition_directions[1][None,:] , axis=-1)
    
    
#    velocity_time_3D_radial_dr_time = []
#    velocity_time_3D_radial_dtheta_time = []
#    velocity_time_3D_theta_dr_time = []
#    velocity_time_3D_theta_dtheta_time = []
    # compute the radial and angular derivatives via local taylor expansions.
#    for frame in range(nframes):
    velocity_time_3D_radial_dr_time, velocity_time_3D_radial_dtheta_time = compute_directional_gradients_time(velocity_time_3D_radial, 
                                                                                           unwrap_params_3D, r_dist=5, theta_dist=1, smooth=True, smooth_theta_dist=10)
    velocity_time_3D_theta_dr_time, velocity_time_3D_theta_dtheta_time = compute_directional_gradients_time(velocity_time_3D_theta, 
                                                                                           unwrap_params_3D, r_dist=5, theta_dist=1, smooth=True, smooth_theta_dist=10)

#        velocity_time_3D_radial_dr_time.append(velocity_time_3D_radial_dr)
        
    # return the ma
    return (velocity_time_3D_radial_dr_time, velocity_time_3D_radial_dtheta_time, velocity_time_3D_theta_dr_time, velocity_time_3D_theta_dtheta_time)
    

def direct_inverse(A):
    """Compute the inverse of matrices in an array of shape (M,N,N)"""
    return np.linalg.inv(A) 

def statistical_strain_rate_grid(grid_squares_time, unwrap_params_3D):
    
    # we can do this in (x,y,z) coordinates.
    # see, http://graner.net/francois/publis/graner_tools.pdf
    
    links_squares_time_3D = unwrap_params_3D[grid_squares_time[:,:,1:,1].astype(np.int), 
                                          grid_squares_time[:,:,1:,0].astype(np.int)] - unwrap_params_3D[grid_squares_time[:,:,:-1,1].astype(np.int), 
                                          grid_squares_time[:,:,:-1,0].astype(np.int)]
    
    d_links_squares_time_3D = links_squares_time_3D[1:] - links_squares_time_3D[:-1]
    
    M_matrix_00 = np.mean( links_squares_time_3D[...,0] ** 2, axis=-1)
    M_matrix_01 = np.mean( links_squares_time_3D[...,0] * links_squares_time_3D[...,1], axis=-1)
    M_matrix_02 = np.mean( links_squares_time_3D[...,0] * links_squares_time_3D[...,2], axis=-1)
    M_matrix_10 = np.mean( links_squares_time_3D[...,1] * links_squares_time_3D[...,0], axis=-1)
    M_matrix_11 = np.mean( links_squares_time_3D[...,1] **2, axis=-1)
    M_matrix_12 = np.mean( links_squares_time_3D[...,1] * links_squares_time_3D[...,2], axis=-1)
    M_matrix_20 = np.mean( links_squares_time_3D[...,2] * links_squares_time_3D[...,0], axis=-1)
    M_matrix_21 = np.mean( links_squares_time_3D[...,2] * links_squares_time_3D[...,1], axis=-1)
    M_matrix_22 = np.mean( links_squares_time_3D[...,2] **2, axis=-1)
    
    # compute the inverse 3 x 3 matrix using fomulae.
    M_matrix = np.array([[M_matrix_00, M_matrix_01, M_matrix_02], 
                         [M_matrix_10, M_matrix_11, M_matrix_12], 
                         [M_matrix_20, M_matrix_21, M_matrix_22]])
    
#    print(M_matrix.max(), M_matrix.min())
    M_matrix = M_matrix.transpose(2,3,0,1)
    
#    print(M_matrix[0,0])
#    print(np.linalg.det(M_matrix[0,0]))
    M_inv = np.linalg.pinv(M_matrix.reshape(-1,3,3)).reshape(M_matrix.shape)
    
    print(np.allclose(M_matrix[0,0], np.dot(M_matrix[0,0], np.dot(M_inv[0,0], M_matrix[0,0]))))
#    print(M_inv[0,0])
#    print(np.linalg.inv(M_matrix[0,0]))
#    print(np.dot(M_matrix[0,0], np.linalg.inv(M_matrix[0,0])))
    
    C_matrix_00 = np.mean( links_squares_time_3D[:-1,...,0] * d_links_squares_time_3D[...,0], axis=-1)
    C_matrix_01 = np.mean( links_squares_time_3D[:-1,...,0] * d_links_squares_time_3D[...,1], axis=-1)
    C_matrix_02 = np.mean( links_squares_time_3D[:-1,...,0] * d_links_squares_time_3D[...,2], axis=-1)
    C_matrix_10 = np.mean( links_squares_time_3D[:-1,...,1] * d_links_squares_time_3D[...,0], axis=-1)
    C_matrix_11 = np.mean( links_squares_time_3D[:-1,...,1] * d_links_squares_time_3D[...,1], axis=-1)
    C_matrix_12 = np.mean( links_squares_time_3D[:-1,...,1] * d_links_squares_time_3D[...,2], axis=-1)
    C_matrix_20 = np.mean( links_squares_time_3D[:-1,...,2] * d_links_squares_time_3D[...,0], axis=-1)
    C_matrix_21 = np.mean( links_squares_time_3D[:-1,...,2] * d_links_squares_time_3D[...,1], axis=-1)
    C_matrix_22 = np.mean( links_squares_time_3D[:-1,...,2] * d_links_squares_time_3D[...,2], axis=-1)

    C_matrix = np.array([[C_matrix_00, C_matrix_01, C_matrix_02], 
                         [C_matrix_10, C_matrix_11, C_matrix_12], 
                         [C_matrix_20, C_matrix_21, C_matrix_22]])
    C_matrix = C_matrix.transpose(2,3,0,1)
    C_matrix_T = C_matrix.transpose(0,1,3,2) # obtain the matrix transpose.
    
#    V = 1./2 *( np.matmul(M_inv.reshape(-1,3,3), C_matrix.reshape()) + np.matmul(C_matrix_T, M_inv))
    
#    MM = M_inv.reshape(-1,3,3)
#    CC = C_matrix.reshape(-1,3,3)
#    CC_T = C_matrix_T.reshape(-1,3,3)
    V = 1./2 *( np.matmul(M_inv[:-1], C_matrix) + 
                np.matmul(C_matrix_T, M_inv[:-1]))
    
    return V

def statistical_strain_rate_grid_surface(grid_squares_time, unwrap_params_3D, direction_vectors, growth_correction=None):
    
    # we can do this in (x,y,z) coordinates.
    # see, http://graner.net/francois/publis/graner_tools.pdf
    
    # use to reduce the 3D displacement to 2D. 
    radial_direction_vector, theta_direction_vector = direction_vectors
    
    # lookup using the grid squares. 
    radial_direction_vector = np.mean(radial_direction_vector[grid_squares_time[:,:,:-1,1].astype(np.int), 
                                                              grid_squares_time[:,:,:-1,0].astype(np.int)], axis=-2)
    theta_direction_vector = np.mean(theta_direction_vector[grid_squares_time[:,:,:-1,1].astype(np.int), 
                                                            grid_squares_time[:,:,:-1,0].astype(np.int)], axis=-2)
    
    # renormalize directions.
    radial_direction_vector = radial_direction_vector / (np.linalg.norm(radial_direction_vector, axis=-1)[...,None] + 1e-8)
    theta_direction_vector = theta_direction_vector / (np.linalg.norm(theta_direction_vector, axis=-1)[...,None] + 1e-8)
    
    """
    Main code.
    """
    # (x,y,z) convention, l_matrix. -> get the displacement matrix. 
    links_squares_time_3D = unwrap_params_3D[grid_squares_time[:,:,1:,1].astype(np.int), 
                                             grid_squares_time[:,:,1:,0].astype(np.int)] - unwrap_params_3D[grid_squares_time[:,:,:-1,1].astype(np.int), 
                                             grid_squares_time[:,:,:-1,0].astype(np.int)]
    
    if growth_correction is not None:
        links_squares_time_3D = links_squares_time_3D*growth_correction[:,None,None,None]
    
#    # computation 1 -> take the average for the displacement. 
#    links_squares_time_3D_r = np.sum(links_squares_time_3D * radial_direction_vector[:,:,None,:], axis=-1)
#    links_squares_time_3D_theta = np.sum(links_squares_time_3D * theta_direction_vector[:,:,None,:], axis=-1)
#    links_squares_time_3D = np.concatenate([links_squares_time_3D_r[...,None],
#                                            links_squares_time_3D_theta[...,None]], axis=-1 )
    
    # computation 2 -> use the individual projections.
    radial_squares_ind = direction_vectors[0][grid_squares_time[:,:,:-1,1].astype(np.int), 
                                                 grid_squares_time[:,:,:-1,0].astype(np.int)]
    
    theta_squares_ind = direction_vectors[1][grid_squares_time[:,:,:-1,1].astype(np.int), 
                                               grid_squares_time[:,:,:-1,0].astype(np.int)]
    
    
    links_squares_time_3D_r = np.sum(links_squares_time_3D * radial_squares_ind, axis=-1)
    links_squares_time_3D_theta = np.sum(links_squares_time_3D * theta_squares_ind, axis=-1)
    links_squares_time_3D = np.concatenate([links_squares_time_3D_r[...,None],
                                            links_squares_time_3D_theta[...,None]], axis=-1 )
    
    
    # dl_matrix/dt (x,y,z)
    d_links_squares_time_3D = links_squares_time_3D[1:] - links_squares_time_3D[:-1]
#    d_links_squares_time_3D_r = np.sum(d_links_squares_time_3D * radial_direction_vector[:,:,None,:], axis=-1)
#    d_links_squares_time_3D_theta = np.sum(d_links_squares_time_3D * theta_direction_vector[:,:,None,:], axis=-1)
#    d_links_squares_time_3D = np.concatenate([d_links_squares_time_3D_r[...,None],
#                                              d_links_squares_time_3D_theta[...,None]], axis=-1 )
    
    # the M_matrix and C_matrix reduce to 2 x 2 matrices. 
    M_matrix_00 = np.mean( links_squares_time_3D[...,0] ** 2, axis=-1)
    M_matrix_01 = np.mean( links_squares_time_3D[...,0] * links_squares_time_3D[...,1], axis=-1)
    M_matrix_10 = np.mean( links_squares_time_3D[...,1] * links_squares_time_3D[...,0], axis=-1)
    M_matrix_11 = np.mean( links_squares_time_3D[...,1] **2, axis=-1)
    
    # compute the inverse 3 x 3 matrix using fomulae.
    M_matrix = np.array([[M_matrix_00, M_matrix_01], 
                         [M_matrix_10, M_matrix_11]])
    
#    print(M_matrix.max(), M_matrix.min())
    M_matrix = M_matrix.transpose(2,3,0,1) # last dimensions is 2x2.
    
#    print(M_matrix[0,0])
#    print(np.linalg.det(M_matrix[0,0]))
    M_inv = np.linalg.inv(M_matrix.reshape(-1,2,2)).reshape(M_matrix.shape)
    
    print(M_inv[0,0])
    print(np.allclose(M_matrix[0,0], np.dot(M_matrix[0,0], np.dot(M_inv[0,0], M_matrix[0,0]))))
#    print(M_inv[0,0])
#    print(np.linalg.inv(M_matrix[0,0]))
#    print(np.dot(M_matrix[0,0], np.linalg.inv(M_matrix[0,0])))
    
    # C_matrix in the change of basis also reduces to 2 x 2 
    C_matrix_00 = np.mean( links_squares_time_3D[:-1,...,0] * d_links_squares_time_3D[...,0], axis=-1)
    C_matrix_01 = np.mean( links_squares_time_3D[:-1,...,0] * d_links_squares_time_3D[...,1], axis=-1)
    C_matrix_10 = np.mean( links_squares_time_3D[:-1,...,1] * d_links_squares_time_3D[...,0], axis=-1)
    C_matrix_11 = np.mean( links_squares_time_3D[:-1,...,1] * d_links_squares_time_3D[...,1], axis=-1)
    
    C_matrix = np.array([[C_matrix_00, C_matrix_01], 
                         [C_matrix_10, C_matrix_11]])
    C_matrix = C_matrix.transpose(2,3,0,1)
    C_matrix_T = C_matrix.transpose(0,1,3,2) # obtain the matrix transpose.
    
    # matrix multiply assuming conserved links. 
    V = 1./2 *( np.matmul(M_inv[:-1], C_matrix) + 
                np.matmul(C_matrix_T, M_inv[:-1]))
    
    Omega = 1./2 * ( np.matmul(M_inv[:-1], C_matrix) - 
                     np.matmul(C_matrix_T, M_inv[:-1]))
    
    return V, Omega, (radial_direction_vector, theta_direction_vector)


def statistical_strain_rate_grid_surface_sparse(grid_squares_time, unwrap_params_3D, direction_vectors, growth_correction=None):
    
    # we can do this in (x,y,z) coordinates.
    # see, http://graner.net/francois/publis/graner_tools.pdf
    
    # use to reduce the 3D displacement to 2D. 
    radial_direction_vector, theta_direction_vector = direction_vectors
    
    # lookup using the grid squares. 
    radial_direction_vector = np.median(radial_direction_vector[:,:,:-1], axis=-2)
    theta_direction_vector = np.median(theta_direction_vector[:,:,:-1], axis=-2)
    
    # renormalize directions.
    radial_direction_vector = radial_direction_vector / (np.linalg.norm(radial_direction_vector, axis=-1)[...,None] + 1e-8)
    theta_direction_vector = theta_direction_vector / (np.linalg.norm(theta_direction_vector, axis=-1)[...,None] + 1e-8)
    
    """
    Main code.
    """
    # (x,y,z) convention, l_matrix. -> get the displacement matrix. 
    links_squares_time_3D = unwrap_params_3D[grid_squares_time[:,:,1:,1].astype(np.int), 
                                             grid_squares_time[:,:,1:,0].astype(np.int)] - unwrap_params_3D[grid_squares_time[:,:,:-1,1].astype(np.int), 
                                             grid_squares_time[:,:,:-1,0].astype(np.int)]
    
    if growth_correction is not None:
        links_squares_time_3D = links_squares_time_3D*growth_correction[:,None,None,None]
    
#    # computation 1 -> take the average for the displacement. 
#    links_squares_time_3D_r = np.sum(links_squares_time_3D * radial_direction_vector[:,:,None,:], axis=-1)
#    links_squares_time_3D_theta = np.sum(links_squares_time_3D * theta_direction_vector[:,:,None,:], axis=-1)
#    links_squares_time_3D = np.concatenate([links_squares_time_3D_r[...,None],
#                                            links_squares_time_3D_theta[...,None]], axis=-1 )
    
    # computation 2 -> use the individual projections.
    radial_squares_ind = direction_vectors[0][:,:,:-1]#[grid_squares_time[:,:,:-1,1].astype(np.int), 
                                                 #grid_squares_time[:,:,:-1,0].astype(np.int)]
    
    theta_squares_ind = direction_vectors[1][:,:,:-1]#[grid_squares_time[:,:,:-1,1].astype(np.int), 
                                               #grid_squares_time[:,:,:-1,0].astype(np.int)]
    #####
    
    links_squares_time_3D_r = np.sum(links_squares_time_3D * radial_squares_ind, axis=-1)
    links_squares_time_3D_theta = np.sum(links_squares_time_3D * theta_squares_ind, axis=-1)
    links_squares_time_3D = np.concatenate([links_squares_time_3D_r[...,None],
                                            links_squares_time_3D_theta[...,None]], axis=-1 )
    
    
    # dl_matrix/dt (x,y,z)
    d_links_squares_time_3D = links_squares_time_3D[1:] - links_squares_time_3D[:-1]
#    d_links_squares_time_3D_r = np.sum(d_links_squares_time_3D * radial_direction_vector[:,:,None,:], axis=-1)
#    d_links_squares_time_3D_theta = np.sum(d_links_squares_time_3D * theta_direction_vector[:,:,None,:], axis=-1)
#    d_links_squares_time_3D = np.concatenate([d_links_squares_time_3D_r[...,None],
#                                              d_links_squares_time_3D_theta[...,None]], axis=-1 )
    
    # the M_matrix and C_matrix reduce to 2 x 2 matrices. 
    M_matrix_00 = np.mean( links_squares_time_3D[...,0] ** 2, axis=-1)
    M_matrix_01 = np.mean( links_squares_time_3D[...,0] * links_squares_time_3D[...,1], axis=-1)
    M_matrix_10 = np.mean( links_squares_time_3D[...,1] * links_squares_time_3D[...,0], axis=-1)
    M_matrix_11 = np.mean( links_squares_time_3D[...,1] **2, axis=-1)
    
    # compute the inverse 3 x 3 matrix using fomulae.
    M_matrix = np.array([[M_matrix_00, M_matrix_01], 
                         [M_matrix_10, M_matrix_11]])
    
#    print(M_matrix.max(), M_matrix.min())
    M_matrix = M_matrix.transpose(2,3,0,1) # last dimensions is 2x2.
    
#    print(M_matrix[0,0])
#    print(np.linalg.det(M_matrix[0,0]))
    M_inv = np.linalg.inv(M_matrix.reshape(-1,2,2)).reshape(M_matrix.shape)
    
    print(M_inv[0,0])
    print(np.allclose(M_matrix[0,0], np.dot(M_matrix[0,0], np.dot(M_inv[0,0], M_matrix[0,0]))))
#    print(M_inv[0,0])
#    print(np.linalg.inv(M_matrix[0,0]))
#    print(np.dot(M_matrix[0,0], np.linalg.inv(M_matrix[0,0])))
    
    # C_matrix in the change of basis also reduces to 2 x 2 
    C_matrix_00 = np.mean( links_squares_time_3D[:-1,...,0] * d_links_squares_time_3D[...,0], axis=-1)
    C_matrix_01 = np.mean( links_squares_time_3D[:-1,...,0] * d_links_squares_time_3D[...,1], axis=-1)
    C_matrix_10 = np.mean( links_squares_time_3D[:-1,...,1] * d_links_squares_time_3D[...,0], axis=-1)
    C_matrix_11 = np.mean( links_squares_time_3D[:-1,...,1] * d_links_squares_time_3D[...,1], axis=-1)
    
    C_matrix = np.array([[C_matrix_00, C_matrix_01], 
                         [C_matrix_10, C_matrix_11]])
    C_matrix = C_matrix.transpose(2,3,0,1)
    C_matrix_T = C_matrix.transpose(0,1,3,2) # obtain the matrix transpose.
    
    # matrix multiply assuming conserved links. 
    V = 1./2 *( np.matmul(M_inv[:-1], C_matrix) + 
                np.matmul(C_matrix_T, M_inv[:-1]))
    
    Omega = 1./2 * ( np.matmul(M_inv[:-1], C_matrix) - 
                     np.matmul(C_matrix_T, M_inv[:-1]))
    
    return V, Omega, (radial_direction_vector, theta_direction_vector)

#def set_axes_equal(ax):
#    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
#    cubes as cubes, etc..  This is one possible solution to Matplotlib's
#    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
#
#    Input
#      ax: a matplotlib axis, e.g., as output from plt.gca().
#    '''
#
#    x_limits = ax.get_xlim3d()
#    y_limits = ax.get_ylim3d()
#    z_limits = ax.get_zlim3d()
#
#    x_range = abs(x_limits[1] - x_limits[0])
#    x_middle = np.mean(x_limits)
#    y_range = abs(y_limits[1] - y_limits[0])
#    y_middle = np.mean(y_limits)
#    z_range = abs(z_limits[1] - z_limits[0])
#    z_middle = np.mean(z_limits)
#
#    # The plot bounding box is a sphere in the sense of the infinity
#    # norm, hence I call half the max range the plot radius.
#    plot_radius = 0.5*max([x_range, y_range, z_range])
#
#    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
#    
#    return
def baseline_als(y, lam, p, niter=10, winsize=3, mode='edge'):
    """ Code adapted from https://stackoverflow.com/questions/29156532/python-baseline-correction-library. 
    Implements paper of "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens (2005), [1].
    
    Parameters
    ----------
    y : numpy array
        numpy vector of observations to fit.
    lam : float 
        smoothness parameter, paper recommends 10**2 ≤ λ ≤ 10**9 for most applications.
    p : float
        asymmetry parameter, the higher this is the more asymmetric we consider the fitting errors, paper recommends 0.001 ≤ p ≤ 0.1 for positive peaks.
    niter : int
        the number of iterations to run the iterative fitting algorithm, the longer you run this, the more flat the baseline. Only a few iterations is needed. 
    
    Returns
    -------
    z : numpy array
        fitted baseline, same numpy vector size as y.
    References
    ----------
    .. [1] Eilers PH, Boelens HF. "Baseline correction with asymmetric least squares smoothing." Leiden University Medical Centre Report. 2005 Oct 21;1(1):5.
    
    """
    from scipy import sparse
    import numpy as np 
    from scipy.sparse.linalg import spsolve
    
    N = len(y)
    if winsize is not None:
        y_ = np.pad(y, [[winsize//2, winsize//2]], mode=mode)
    else:
        y_ = y.copy()
    
    L = len(y_)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)

    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y_)
        w = p * (y_ > z) + (1-p) * (y_ < z)

    if winsize is not None:
        return z[winsize//2:winsize//2+N]
    else: 
        return z

# temporal moving average method. 
def moving_average( curve, winsize, avg_fnc):
    
    N = len(curve)
    curve_ = np.pad(curve, [[winsize//2, winsize//2]], mode='edge')
    curve_out = np.hstack([avg_fnc(curve_[ii:ii+winsize]) for ii in range(N)])
    
    return curve_out

def smooth_curves(curves, avg_fnc=None, winsize=3, win_mode='edge', polyorder=3, p=0.5, lam=1, method='als'):
    
    from scipy.signal import savgol_filter
    out = []
    
    for ii in range(curves.shape[1]):
        
        curve = curves[:,ii]
        if method == 'als':
            smooth_curve = baseline_als(curve,p=p, lam=lam, winsize=winsize, mode=win_mode)
        if method =='ma':
            smooth_curve = moving_average(curve, winsize=winsize, avg_fnc=avg_fnc)
        if method == 'savitzky':
            smooth_curve = savgol_filter(curve, window_length=winsize, polyorder=polyorder, mode=win_mode)
        out.append(smooth_curve[None,:])
        
    out = np.vstack(out).T
    
    return out

# add a quad heatmap function
def heatmap_vals_radial(polys, vals, shape, alpha=0):
    
    from skimage.draw import polygon
    n = len(polys)
    
#    heatmap = np.ones((shape[0], shape[1], 4)) * alpha # this is a color mappable.... 
    heatmap = np.zeros(shape)
    heatmap[:] = np.nan
    
    for ii in range(n):
        poly = polys[ii]
        val = vals[ii]
        
        rr, cc = polygon(poly[:,0], poly[:,1], shape)
        heatmap[rr,cc] = val
    
    return heatmap


def compute_change_area(areas):
    
    delta_areas = areas[1:] - areas[:-1]
    
    # obtain in terms of %
    delta_areas = delta_areas / (areas[:-1] +1.)#[None,:]
    delta_areas = np.vstack([np.zeros(areas.shape[1])[None,:], 
                                        delta_areas])
    return delta_areas


if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    import glob 
    import os 
    import scipy.io as spio
    from skimage.io import imsave
    
    import MOSES.Utility_Functions.file_io as fio
#    from MOSES.Optical_Flow_Tracking.optical_flow import farnebackflow, TVL1Flow, DeepFlow
#    from MOSES.Optical_Flow_Tracking.superpixel_track import compute_vid_opt_flow, compute_grayscale_vid_superpixel_tracks, compute_grayscale_vid_superpixel_tracks_w_optflow
#    from MOSES.Visualisation_Tools.track_plotting import plot_tracks
    from flow_vis import make_colorwheel, flow_to_color
    
    from skimage.filters import threshold_otsu
    import skimage.transform as sktform
    from tqdm import tqdm 
    import pandas as pd 
    
    from skimage.draw import polygon
    from skimage.filters import gaussian
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.exposure import rescale_intensity, equalize_hist
    from mpl_toolkits.mplot3d import Axes3D
    import skimage.morphology as skmorph

    from scipy.signal import medfilt, medfilt2d, convolve2d
    from mpl_toolkits.mplot3d import Axes3D
    
    import Geometry.meshtools as meshtools
    
    from MOSES.Motion_Analysis import tracks3D_tools as tra3Dtools
                    
#    master_analysis_save_folder = '/media/felix/Srinivas4/All_results/Migration_Analysis/Staged_Analysis'
#    fio.mkdir(master_analysis_save_folder)
    
#    # master_plot_save_folder = '/media/felix/My Passport/Shankar-2/All_Results/Visualizations/Radial_deformation_analysis_example'
#     master_plot_save_folder = '/media/felix/My Passport/Shankar-2/All_Results/Visualizations/Radial_deformation_analysis_example'
#    
#     fio.mkdir(master_plot_save_folder) 
    master_plot_save_folder = '/media/felix/My Passport/Shankar-2/All_Results/Visualizations/Radial_deformation_analysis_strain_rate'
    fio.mkdir(master_plot_save_folder)
    
    Ts = 10./60. # universal sampling time. 

    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Set up a nearest neighbour model 
    from sklearn.neighbors import NearestNeighbors 
    from skimage.segmentation import mark_boundaries
    from sklearn.decomposition import PCA
    
    from matplotlib.patches import Ellipse
    
    nbrs_model = NearestNeighbors(n_neighbors=9, algorithm='auto') # how many neighbours to include? 9 = 8-connected?
    
    
    all_rootfolders = ['/media/felix/Srinivas4/LifeAct', 
                       '/media/felix/Srinivas4/MTMG-TTR',
                       '/media/felix/Srinivas4/MTMG-HEX',
                       '/media/felix/Srinivas4/MTMG-HEX/new_hex']

    all_embryo_folders = []
    
    for rootfolder in all_rootfolders:
    # =============================================================================
    #     1. Load all embryos 
    # =============================================================================
        embryo_folders = np.hstack([os.path.join(rootfolder,folder) for folder in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, folder)) and 'new_hex' not in folder])
        all_embryo_folders.append(embryo_folders)
        
    all_embryo_folders = np.hstack(all_embryo_folders)
    
    # =============================================================================
    #     2. Get the staging files and times... (not absolutely necessary since this info is included in the tracks. )
    # =============================================================================
    n_spixels = 5000 # set the number of pixels. 
    smoothwinsize = 3
#    n_spixels = 5000    
#    all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES.xlsx', 
#                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG_TTR-smooth-lam1.xlsx',
#                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-smooth-lam1.xlsx',
#                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-new_hex-smooth-lam1.xlsx']
    # all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-LifeAct-smooth-lam10-correct_norot.xlsx', 
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-MTMG-TTR-smooth-lam10-correct_norot.xlsx',
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-MTMG-HEX-smooth-lam10-correct_norot.xlsx',
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-new_hex-smooth-lam10-correct_norot.xlsx']
    all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-LifeAct-smooth-lam5-correct_norot_consensus.xlsx', 
                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-TTR-smooth-lam5-correct_norot_consensus.xlsx',
                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-HEX-smooth-lam5-correct_norot_consensus.xlsx',
                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-new-HEX-smooth-lam5-correct_norot_consensus.xlsx']

    staging_tab = pd.concat([pd.read_excel(staging_file) for staging_file in all_staging_files], ignore_index=True)
    all_stages = staging_tab.columns[1:4]
    
    # =============================================================================
    #     3. Get the initial estimation of the VE migration directions to build the quadranting... 
    # =============================================================================
#    all_angle_files = ['/media/felix/Srinivas4/correct_angle_LifeAct.csv',
#                       '/media/felix/Srinivas4/correct_angle_MTMG-TTR.csv',
#                       '/media/felix/Srinivas4/correct_angle_MTMG-HEX.csv',
#                       '/media/felix/Srinivas4/correct_angle_new_hex.csv']
#    all_angle_files = ['/media/felix/Srinivas4/correct_angle-global-1000_LifeAct.csv',
#                       '/media/felix/Srinivas4/correct_angle-global-1000_MTMG-TTR.csv',
#                       '/media/felix/Srinivas4/correct_angle-global-1000_MTMG-HEX.csv',
#                       '/media/felix/Srinivas4/correct_angle-global-1000_new_hex.csv']
    
    # Use newest consensus angles. !
    # all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles/LifeAct_polar_angles_AVE_classifier.-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles/MTMG-TTR_polar_angles_AVE_classifier.-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles/MTMG-HEX_polar_angles_AVE_classifier.-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles/new_hex_polar_angles_AVE_classifier.-consensus.csv']
    all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/LifeAct_polar_angles_AVE_classifier-consensus.csv',
                       '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-TTR_polar_angles_AVE_classifier-consensus.csv',
                       '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-HEX_polar_angles_AVE_classifier-consensus.csv',
                       '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/new_hex_polar_angles_AVE_classifier-consensus.csv']
    

    angle_tab = pd.concat([pd.read_csv(angle_file) for angle_file in all_angle_files], ignore_index=True)
#    all_stages = staging_tab.columns[1:4]
    
    
#    analyse_stage = str(all_stages[0]) # Pre-migration stage
#    analyse_stage = str(all_stages[1]) # Migration stage
    analyse_stage = str(all_stages[2]) # Post-migration stage

    # analyse_stage = 'All'    
    print(analyse_stage)
    
#    master_analysis_save_folder = os.path.join('/media/felix/Srinivas4/All_results/Migration_Analysis/Staged_Analysis-growth_correct', analyse_stage)
    # master_analysis_save_folder = os.path.join('/media/felix/Srinivas4/All_results/Migration_Analysis/Staged_Analysis-growth_correct-Epi_Grid-auto-polar_Paper-test', analyse_stage)
    # fio.mkdir(master_analysis_save_folder)
    
    """
    Iterate over experimental folders to build quadranting space.....
    """
    supersample_r = 4
    supersample_theta = 4
    
    
    all_analysed_embryos = []

    all_mean_isotropic_strain_ve = []
    all_mean_isotropic_strain_epi = []
    
    all_mean_anisotropic_strain_ve = []
    all_mean_anisotropic_strain_epi = []

    all_mean_curl_strain_ve = []
    all_mean_curl_strain_epi = []
    
#    for embryo_folder in all_embryo_folders[4:5]:
    for embryo_folder in all_embryo_folders[:]:
#    for embryo_folder in embryo_folders[4:5]:
        
        # find the geodesic-rotmatrix. 
        unwrapped_folder = locate_unwrapped_im_folder(embryo_folder, key='geodesic-rotmatrix')
        
        # shouldn't include the lefty embryo anyway in the analysis
        if len(unwrapped_folder) == 1 and '_LEFTY_' not in unwrapped_folder[0]:
            print('processing: ', unwrapped_folder[0])
            
            # locate the demons folder to obtain the x,y,z dimensions for normalization.         
            embryo_im_folder = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step6_' in name or 'step6tf_' in name or 'step6tf-' in name or 'step6-' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
            embryo_im_folder_files = np.hstack(glob.glob(os.path.join(embryo_im_folder[0], '*.tif')))
            embryo_im_shape = np.hstack(fio.read_multiimg_PIL(embryo_im_folder_files[0]).shape)
            embryo_im_shape = embryo_im_shape[[1,0,2]]#.transpose([1,0,2])
            
            unwrapped_folder = unwrapped_folder[0]
            unwrapped_params_folder = unwrapped_folder.replace('geodesic-rotmatrix', 
                                                               'unwrap_params_geodesic-rotmatrix')
            # determine the savefolder. 
            saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned')
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-5000_VE_seg_directions') # this was the segmentation folder. 
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
            # savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-new')
            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-curvilinear_Tcorrected')

#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s' %(n_spixels, analyse_stage))
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_stage-aligned-correct_uniform_scale-%d_%s' %(n_spixels, analyse_stage))
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s-manual' %(n_spixels, analyse_stage))
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_stage-auto-polar-aligned-Paper-%d_%s' %(n_spixels, analyse_stage))
            # savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels))
            # savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s-manual' %(n_spixels, analyse_stage))
            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_stage-auto-polar-curvilinear-aligned-Paper-%d_%s' %(n_spixels, analyse_stage))
            
            # saveoptflowfolder = 
#            , 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')

            """
            To Do: switch to the manual extracted tracks. 
            """
            saveboundaryfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow-epi-boundary')
            saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
            
            if os.path.exists(savetrackfolder) and len(os.listdir(savetrackfolder)) > 0:
                # only tabulate this statistic if it exists. 
                # all_analysed_embryos.append(embryo_folder)

                # only analyse when this stage exists.
                unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))

                # =============================================================================
                # Set up the unwrapped files. 
                # =============================================================================
                
                if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
                    unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
                    paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
                
                else:
                    unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_hex_unwrapped(ff)) for ff in unwrap_im_files])
                    paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_hex_embryo(unwrap_im_files, unwrap_meta_files)
                
                # iterate over the pairs 
                for ii in tqdm(range(len(paired_unwrap_file_condition))[:]):
                    
                    unwrapped_condition = paired_unwrap_file_condition[ii]
                    print(unwrapped_condition)
                    
                    # =============================================================================
                    #   get scaling transforms from registration           
                    # =============================================================================
                    """
                    get the temporal transforms and extract the scaling parameters. 
                    """
                    temp_tform_files = glob.glob(os.path.join(embryo_folder, '*temp_reg_tforms*.mat'))
                    assert(len(temp_tform_files) > 0)
                    
                    temp_tform_scales = np.vstack([parse_temporal_scaling(tformfile) for tformfile in temp_tform_files])
                    temp_tform_scales = np.prod(temp_tform_scales, axis=0)
                    
                    # =============================================================================
                    #   Specific analyses.  
                    # =============================================================================
                    # only analyse if the prereq condition is met 
                    if unwrapped_condition.split('-')[-1] == 'polar':
                        # we only do the analysis for polar views. 
                        if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
                            ve_unwrap_file, epi_unwrap_file = paired_unwrap_files[ii]
                        else:
                            ve_unwrap_file, epi_unwrap_file, hex_unwrap_file = paired_unwrap_files[ii]
                                
                    # =============================================================================
                    #        Load the relevant videos.          
                    # =============================================================================
                        vid_ve = fio.read_multiimg_PIL(ve_unwrap_file)
                        vid_epi = fio.read_multiimg_PIL(epi_unwrap_file)
                        vid_epi_resize = sktform.resize(vid_epi, vid_ve.shape, preserve_range=True)
                        
                        # """
                        # 1. load the flow file for VE and Epi surfaces. 
                        # """
                        # flowfile_ve = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(ve_unwrap_file)[-1].replace('.tif', '.mat'))
                        # flow_ve = spio.loadmat(flowfile_ve)['flow']
                    
                        # flowfile_epi = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(epi_unwrap_file)[-1].replace('.tif', '.mat'))
                        # flow_epi = spio.loadmat(flowfile_epi)['flow']
                        # flow_epi_resize = sktform.resize(flow_epi, flow_ve.shape, preserve_range=True)
                        
                        if 'MTMG-HEX' in embryo_folder:
                            vid_hex = fio.read_multiimg_PIL(hex_unwrap_file)
                            # flowfile_hex = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(hex_unwrap_file)[-1].replace('.tif', '.mat'))
                            # flow_hex = spio.loadmat(flowfile_hex)['flow']
                            
                    # =============================================================================
                    #       Load the relevant saved pair track file.               
                    # =============================================================================
                        trackfile = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_%s_' %(n_spixels, analyse_stage) + unwrapped_condition+'.mat')
#                        trackfile = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')
                        
                        meantracks_ve = spio.loadmat(trackfile)['meantracks_ve']
                        meantracks_epi = spio.loadmat(trackfile)['meantracks_epi']
                        
                        proj_condition = spio.loadmat(trackfile)['condition'][0].split('-')
                        # proj_condition = unwrapped_condition.split('-')
                        proj_type = proj_condition[-1]
                        start_end_times = spio.loadmat(trackfile)['start_end'].ravel()
        
    #                        if proj_type == 'polar':
    #                        all_analysed_embryos.append(embryo_folder)
                    # =============================================================================
                    #       Load the relevant directionality info     
                    # =============================================================================
    #                        if proj_type == 'polar':
    #                            ve_angle_move = angle_tab.loc[angle_tab['polar_conditions'].values == '-'.join(proj_condition)]['correct_angles_vertical']
    #                        if proj_type == 'rect':
    #                            ve_angle_move = angle_tab.loc[angle_tab['rect_conditions'].values == '-'.join(proj_condition)]['correct_angles_vertical']
                        
                        if proj_type == 'polar':
                            ve_angle_move = angle_tab.loc[angle_tab['Embryo'].values == '-'.join(proj_condition)]['Angle_consenus']
                        if proj_type == 'rect':
                            ve_angle_move = angle_tab.loc[angle_tab['Embryo'].values == '-'.join(proj_condition)]['Angle_consenus']
                        
    #                        if proj_type == 'polar':
    #                            ve_angle_move = angle_tab.loc[angle_tab['Embryo'].values == '-'.join(proj_condition)]['Angle']
    #                        if proj_type == 'rect':
    #                            ve_angle_move = angle_tab.loc[angle_tab['Embryo'].values == '-'.join(proj_condition)]['Angle']
        
                        # this angle should already take inversion into account! 
                        ve_angle_move = 90 - ve_angle_move # to get back to the rectangular angle coordinate? 
                        print('ve movement direction: ', ve_angle_move)
                        
                    # =============================================================================
                    #       Load the boundary file
                    # ============================================================================
                        # grab the relevant boundary file. 
                        epi_boundary_file = os.path.join(saveboundaryfolder,unwrapped_condition,'inferred_epi_boundary-'+unwrapped_condition+'.mat')
                        epi_boundary_time = spio.loadmat(epi_boundary_file)['contour_line']
                        
                        epi_contour_line = epi_boundary_time[start_end_times[0]]
                        # epi_contour_line = epi_boundary_time[0] # just want the initial division line of the Epi.
                        epi_contour_line = epi_contour_line[start_end_times[0]:start_end_times[1]] # not sure we need this.
                    # =============================================================================
                    #       Load the unwrapping 3D geometry coordinates 
                    # =============================================================================
                
                        # load the associated unwrap param files depending on polar/rect coordinates transform. 
                        unwrap_param_file_ve = ve_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
                        unwrap_params_ve = spio.loadmat(unwrap_param_file_ve); unwrap_params_ve = unwrap_params_ve['ref_map_'+proj_type+'_xyz']
                        
                        unwrap_param_file_epi = epi_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
                        unwrap_params_epi = spio.loadmat(unwrap_param_file_epi); unwrap_params_epi = unwrap_params_epi['ref_map_'+proj_type+'_xyz']
                        unwrap_params_epi = sktform.resize(unwrap_params_epi, unwrap_params_ve.shape, preserve_range=True)
                        
                        
                        if proj_type == 'rect':
                            unwrap_params_ve = unwrap_params_ve[::-1] # reverse ....
                            unwrap_params_epi = unwrap_params_epi[::-1]
                        
                    # =============================================================================
                    #       Load the relevant VE segmentation from trained classifier    
                    # =============================================================================
                        trackfile_full = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels), 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')
    #                        trackfile_full = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels), 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')
                        
    #                        ve_seg_params_file = os.path.join(savevesegfolder, os.path.split(trackfile_full)[-1].replace('rect','polar').replace('.mat', '-ve_aligned.mat'))
    #                        ve_seg_params_obj = spio.loadmat(ve_seg_params_file)                             
                        
                        ve_tracks_full = spio.loadmat(trackfile_full)['meantracks_ve']
    #                        ve_seg_params_file = os.path.join(savevesegfolder, os.path.split(trackfile_full)[-1].replace('rect','polar').replace('.mat', '-ve_aligned.mat'))
    #                        ve_seg_params_obj = spio.loadmat(ve_seg_params_file) 

                        """
                        Load the relevant ve segmentation. 
                        """               
                        ve_seg_params_file = 'deepflow-meantracks-1000_' + os.path.split(unwrap_param_file_ve)[-1].replace('_unwrap_params-geodesic.mat',  '_polar-ve_aligned.mat')
                        ve_seg_params_file = os.path.join(savevesegfolder, ve_seg_params_file)
                        ve_seg_params_obj = spio.loadmat(ve_seg_params_file)                             
                
                
                        # load the polar mask and the times at which this was calculated to infer the positions at earlier and later times.  
                        ve_seg_select_rect_valid = ve_seg_params_obj['density_rect_pts_mask']>0 # get the polar associated mask. 
                        ve_seg_select_rect_valid_svm = ve_seg_params_obj['density_rect_pts_mask_SVM']>0
                        
                        ve_seg_center = ve_seg_params_obj['ve_center'].ravel()
                        
                        # use the ve_seg_center + unwrap_params_ve to figure out the 3D VE direction from angle! .
                        ve_seg_migration_times = ve_seg_params_obj['migration_stage_times'][0]
                        embryo_inversion = ve_seg_params_obj['embryo_inversion'].ravel()[0] > 0 
                        
                        # to do also load the 3D migration VE vector
    #                        mean_ve_direction_3D = ve_seg_params_obj['ve_vector_3D'].ravel()
                        
        #                ve_seg_select_rect = ve_seg_params_obj['ve_rect_select'].ravel() > 0
        #                ve_seg_select_rect_valid = ve_seg_params_obj['ve_rect_select_valid'].ravel() > 0
    #                        ve_seg_select_rect_valid = ve_seg_params_obj['density_rect_pts']
    #                        ve_seg_select_rect_valid = ve_seg_select_rect_valid >= np.mean(ve_seg_select_rect_valid) + 1 * np.std(ve_seg_select_rect_valid)
    #                        ve_seg_select_rect = ve_seg_select_rect_valid[meantracks_ve[:,0,0], 
    #                                                                      meantracks_ve[:,0,1]]
        
        
                        """
                        invert back the segmentation image to the first frame.
                        """
                    # =============================================================================
                    #    Embryo Inversion                      
                    # =============================================================================
                        if embryo_inversion:
                            """
                            NOTE: THIS IS NOT USED!
                            """
                            if proj_type == 'rect':
                                unwrap_params_ve = unwrap_params_ve[:,::-1] # reverse x direction 
                                unwrap_params_epi = unwrap_params_epi[:,::-1]

                                # properly invert the 3D. 
                                unwrap_params_ve[...,2] = embryo_im_shape[2] - unwrap_params_ve[...,2]
                                unwrap_params_epi[...,2] = embryo_im_shape[2] - unwrap_params_epi[...,2]
                                
                                # invert the 2D.
                                ve_tracks_full[...,1] = vid_ve[0].shape[1]-1 - ve_tracks_full[...,1] # reverse the x direction 
                                meantracks_ve[...,1] = vid_ve[0].shape[1]-1 - meantracks_ve[...,1] # reverse the x direction !. 
                                meantracks_epi[...,1] = vid_epi_resize[0].shape[1]-1 - meantracks_epi[...,1]
                                
                                vid_ve = vid_ve[:,:,::-1]
                                vid_epi = vid_epi[:,:,::-1]
                                vid_epi_resize = vid_epi_resize[:,:,::-1]

                                # check.... do we need to invert the epi boundary too !. 
                                epi_contour_line[...,0] = (vid_ve[0].shape[1]-1) - epi_contour_line[...,0] # correct x-coordinate.  
                                
                                
                                if 'MTMG-HEX' in embryo_folder:
                                    vid_hex = vid_hex[:,:,::-1]
                                    # flow_hex = flow_hex[:,:,::-1]
                                    # flow_hex[...,0] = -flow_hex[...,0]

                                # # apply correction to optflow.
                                # flow_ve = flow_ve[:,:,::-1]
                                # flow_ve[...,0] = -flow_ve[...,0] #(x,y)
                                # flow_epi = flow_epi[:,:,::-1]
                                # flow_epi[...,0] = -flow_epi[...,0] #(x,y)
                                # flow_epi_resize = flow_epi_resize[:,:,::-1]
                                # flow_epi_resize[...,0] = -flow_epi_resize[...,0] #(x,y)
                                
                            else:

                                unwrap_params_ve = unwrap_params_ve[::-1] # reverse y direction
                                unwrap_params_epi = unwrap_params_epi[::-1]
                                
                                # properly invert the 3D. 
                                unwrap_params_ve[...,2] = embryo_im_shape[2] - unwrap_params_ve[...,2]
                                unwrap_params_epi[...,2] = embryo_im_shape[2] - unwrap_params_epi[...,2]

                                # invert the 2D.
                                ve_tracks_full[...,0] = vid_ve[0].shape[0]-1 - ve_tracks_full[...,0] # switch the y direction. 
                                meantracks_ve[...,0] = vid_ve[0].shape[0]-1 - meantracks_ve[...,0]
                                meantracks_epi[...,0] = vid_epi_resize[0].shape[0]-1 - meantracks_epi[...,0]
                                
                                vid_ve = vid_ve[:,::-1]
                                vid_epi = vid_epi[:,::-1]
                                vid_epi_resize = vid_epi_resize[:,::-1]

                                epi_contour_line[...,1] = (vid_ve[0].shape[0]-1) - epi_contour_line[...,1] # correct the y-coordinate 
                                
                                if 'MTMG-HEX' in embryo_folder:
                                    vid_hex = vid_hex[:,::-1]
                                    # flow_hex = flow_hex[:,::-1]
                                    # flow_hex[...,1] = -flow_hex[...,1]

                                # # apply correction to optflow.
                                # flow_ve = flow_ve[:,::-1]
                                # flow_ve[...,1] = -flow_ve[...,1] #(x,y)
                                # flow_epi = flow_epi[:,::-1]
                                # flow_epi[...,1] = -flow_epi[...,1] #(x,y)
                                # flow_epi_resize = flow_epi_resize[:,::-1]
                                # flow_epi_resize[...,1] = -flow_epi_resize[...,1] #(x,y)
                            
                    # =============================================================================
                    #   Load the respective deformation map calculation for analysis.           
                    # =============================================================================
                        
                        savedefmapfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'deformation_maps', 'motion_only', analyse_stage)
                        # fio.mkdir(savedefmapfolder)
                        savedefmapfile = os.path.join(savedefmapfolder, 'deformation_quadrants_'+unwrapped_condition+'.mat')
                        
                        defmapobj = spio.loadmat(savedefmapfile)

                        # savedefmapfile = os.path.join(savedefmapfolder, 'deformation_quadrants_'+unwrapped_condition+'.mat')
                        # spio.savemat(savedefmapfile, {'pts_ve': pts_ve_all, 
                        #                             'squares_ve':squares_ve_all,
                        #                             'areas_ve':areas_ve_all, 
                        #                             'pts_epi': pts_epi_all, 
                        #                             'squares_epi':squares_epi_all, 
                        #                             'areas_epi':areas_epi_all, 
                        #                             've_angle':ve_angle_move.values[0],
                        #                             'polar_grid': polar_grid,
                        #                             'polar_grid_fine':polar_grid_fine,
                        #                             'supersample_radii': supersample_r, 
                        #                             'supersample_theta': supersample_theta, 
                        #                             'analyse_stage': analyse_stage, 
                        #                             'polar_grid_fine_coordinates':polar_grid_fine_coordinates, 
                        #                             'coarse_line_index_r':coarse_line_index_r,
                        #                             'coarse_line_index_theta':coarse_line_index_theta,
                        #                             'epi_line_time':epi_contour_line,
                        #                             'start_end_time':start_end_time})
                        
                        # Load the primary points. 
                        start_end_time = defmapobj['start_end_time'].astype(np.int)

                        start_, end_ = start_end_time[0]
#                        start_ = start_ - 1
#                        end_ = end_ - 1
#                        if end_ == len(vid_ve)-1:
#                            end_ = end_ + 1
                        supersample_radii = defmapobj['supersample_radii']
                        supersample_theta = defmapobj['supersample_theta']

                        polar_grid_fine_coordinates = defmapobj['polar_grid_fine_coordinates']

                        pts_ve = defmapobj['pts_ve']
                        squares_ve = defmapobj['squares_ve']
                        areas_ve = defmapobj['areas_ve']

                        pts_epi = defmapobj['pts_epi']
                        squares_epi = defmapobj['squares_epi']
                        areas_epi = defmapobj['areas_epi']
                        
                        ve_angle = defmapobj['ve_angle'][0]

                        polar_grid = defmapobj['polar_grid']
                        polar_grid_fine = defmapobj['polar_grid_fine']

                        epi_line_time = defmapobj['epi_line_time']
                        coarse_line_index_r = defmapobj['coarse_line_index_r'].ravel()
                        coarse_line_index_theta = defmapobj['coarse_line_index_theta'].ravel()
                        

                        """
                        Main Analysis:, Statistical strain rate grid computation. 
                        """
                        growth_correction = temp_tform_scales[start_:end_] # not that relevant but hm.... 

                        """
                        Load the computed strain rate statistics.
                        """

                        save_strain_rate_file = os.path.join(savedefmapfolder, 'deformation_strain_rate_fix_'+unwrapped_condition+'.mat')
                        
                        strain_rate_obj = spio.loadmat(save_strain_rate_file)
                        # spio.savemat(save_strain_rate_file, {'embryo': unwrapped_condition, 
                        #                                      'start_end_time': start_end_time,
                        #                                      've_angle_vector_xy': ve_angle_vector, 
                        #                                      've_norm_angle_vector_xy':ve_norm_angle_vector,
                        #                                      'growth_correction':growth_correction, 
                        #                                      'AVE_surf_dir_ve_full_3D': AVE_surf_dir_ve_full, 
                        #                                      'AVE_surf_dir_ve_norm_full_3D': AVE_surf_dir_ve_norm_full, 
                        #                                      'AVE_surf_dir_epi_full_3D':AVE_surf_dir_epi_full, 
                        #                                      'AVE_surf_dir_epi_norm_full_3D':AVE_surf_dir_epi_norm_full,
                        #                                      'V_2D_VE': V_2D_VE, 
                        #                                      'Omega_2D_VE':Omega_2D_VE,
                        #                                      'direct_r_3D_VE_grid':direct_r_3D_VE,
                        #                                      'direct_theta_3D_VE_grid':direct_theta_3D_VE,
                        #                                      'V_2D_Epi':V_2D_Epi,
                        #                                      'Omega_2D_Epi':Omega_2D_Epi,
                        #                                      'direct_r_3D_Epi_grid':direct_r_3D_Epi,
                        #                                      'direct_theta_3D_Epi_grid':direct_theta_3D_Epi,
                        #                                      'V_2D_isotropic_VE':V_2D_isotropic_VE_raw,
                        #                                      'V_2D_anistropic_VE':V_2D_anistropic_VE,
                        #                                      'w_VE':w_VE,
                        #                                      'v_VE':v_VE,
                        #                                      'V_2D_isotropic_Epi':V_2D_isotropic_Epi_raw,
                        #                                      'V_2D_anistropic_Epi':V_2D_anistropic_Epi,
                        #                                      'w_Epi':w_Epi,
                        #                                      'v_Epi':v_Epi,
                        #                                      'plot_points3D_ve':plot_points3D_ve,
                        #                                      'plot_points3D_epi':plot_points3D_epi,
                        #                                      'centroid_2D_ve_proj':centroid_2D_ve_proj,
                        #                                      'centroid_2D_prinvec_ve_proj':centroid_2D_prinvec_ve_proj,
                        #                                      'centroid_2D_secvec_ve_proj':centroid_2D_secvec_ve_proj,
                        #                                      'centroid_2D_epi_proj':centroid_2D_epi_proj,
                        #                                      'centroid_2D_prinvec_epi_proj':centroid_2D_prinvec_epi_proj,
                        #                                      'centroid_2D_secvec_epi_proj':centroid_2D_secvec_epi_proj})

                        if 'LifeAct' in embryo_folder:
                            temporal_time = 2./Ts 
                        else:
                            temporal_time = 1./Ts 

                        isotropic_VE_strain = strain_rate_obj['V_2D_isotropic_VE'][...,0,0] * temporal_time
                        isotropic_Epi_strain = strain_rate_obj['V_2D_isotropic_Epi'][...,0,0] * temporal_time

                        anisotropic_VE_strain = strain_rate_obj['w_VE'][...,1] * temporal_time
                        anisotropic_Epi_strain = strain_rate_obj['w_Epi'][...,1] * temporal_time

                        curl_VE_strain = strain_rate_obj['Omega_2D_VE'][...,0,1] * temporal_time
                        curl_Epi_strain = strain_rate_obj['Omega_2D_Epi'][...,0,1] * temporal_time


                        """
                        Conduct some smoothing.
                        """
                        isotropic_VE_strain = smooth_curves(isotropic_VE_strain, 
                                                            p=0.5, lam=100, 
                                                            winsize=np.maximum(len(isotropic_VE_strain)//10, 3), 
                                                            win_mode='edge', method='als')

                        isotropic_Epi_strain = smooth_curves(isotropic_Epi_strain, 
                                                                p=0.5, lam=100, 
                                                                winsize=np.maximum(len(isotropic_Epi_strain)//10, 3), 
                                                                win_mode='edge', method='als')

                        anisotropic_VE_strain = smooth_curves(anisotropic_VE_strain, 
                                                                p=0.5, lam=100, 
                                                                winsize=np.maximum(len(anisotropic_VE_strain)//10, 3), 
                                                                win_mode='edge', method='als')

                        anisotropic_Epi_strain = smooth_curves(anisotropic_Epi_strain, 
                                                                p=0.5, lam=100, 
                                                                winsize=np.maximum(len(anisotropic_Epi_strain)//10, 3), 
                                                                win_mode='edge', method='als')

                        curl_VE_strain = smooth_curves(curl_VE_strain, 
                                                                p=0.5, lam=100, 
                                                                winsize=np.maximum(len(curl_VE_strain)//10, 3), 
                                                                win_mode='edge', method='als')

                        curl_Epi_strain = smooth_curves(curl_Epi_strain, 
                                                                p=0.5, lam=100, 
                                                                winsize=np.maximum(len(curl_Epi_strain)//10, 3), 
                                                                win_mode='edge', method='als')


                        all_analysed_embryos.append(unwrapped_condition)

                        all_mean_isotropic_strain_ve.append(isotropic_VE_strain.mean(axis=0))
                        all_mean_isotropic_strain_epi.append(isotropic_Epi_strain.mean(axis=0))
                        
                        all_mean_anisotropic_strain_ve.append(anisotropic_VE_strain.mean(axis=0))
                        all_mean_anisotropic_strain_epi.append(anisotropic_Epi_strain.mean(axis=0))

                        all_mean_curl_strain_ve.append(curl_VE_strain.mean(axis=0))
                        all_mean_curl_strain_epi.append(curl_Epi_strain.mean(axis=0))

                    
    # stack all. 
    all_analysed_embryos = np.hstack(all_analysed_embryos)

    all_mean_isotropic_strain_ve = np.vstack(all_mean_isotropic_strain_ve)
    all_mean_isotropic_strain_epi = np.vstack(all_mean_isotropic_strain_epi)
    
    all_mean_anisotropic_strain_ve = np.vstack(all_mean_anisotropic_strain_ve)
    all_mean_anisotropic_strain_epi = np.vstack(all_mean_anisotropic_strain_epi)

    all_mean_curl_strain_ve = np.vstack(all_mean_curl_strain_ve)
    all_mean_curl_strain_epi = np.vstack(all_mean_curl_strain_epi)
    
    """
    Do some plots to check ... and save out. 
    """
    # construct the average area change map for both. 
    mean_all_mean_def_maps_ve = np.mean(all_mean_isotropic_strain_ve, axis=0)
    mean_all_mean_def_maps_epi = np.mean(all_mean_isotropic_strain_ve, axis=0)
    


    """
    Compute the equivalent 4 x 8 quadrant equivalent measurements. i.e. which quadrants are which.
    """
    polar_grid_region = tile_uniform_windows_radial_custom( (640,640), 
                                                             n_r=4, 
                                                             n_theta=8, 
                                                             max_r=640/2./1.2, 
                                                             mask_distal=None, # leave a set distance
                                                             center=None, 
                                                             bound_r=True)
    
    
    polar_grid_fine = tile_uniform_windows_radial_custom( (640,640), 
                                                             n_r=int(supersample_r)*4, 
                                                             n_theta=(int(supersample_theta)+1)*8,
                                                             max_r=640/2./1.2, 
                                                             mask_distal=None, # leave a set distance
                                                             center=None, 
                                                             bound_r=True)
    
    theta_spacing = int(supersample_theta)+1
    r_spacing = int(supersample_r)
    
    polar_grid_fine_2_coarse_ids = [] 
    
    for jj in range(4):
        for ii in range(8):
        
            start_ind = jj*r_spacing*theta_spacing*8
            # this builds the inner ring. 
            ind = np.mod(np.arange(-(theta_spacing//2), (theta_spacing//2)+1) + ii*theta_spacing, theta_spacing*8)
        
            ind = np.hstack([ind+kk*theta_spacing*8 for kk in range(r_spacing)])
 #            print(ind)
            all_quad_ind = ind + start_ind

            polar_grid_fine_2_coarse_ids.append(all_quad_ind)
            
    polar_grid_fine_2_coarse_ids = np.array(polar_grid_fine_2_coarse_ids)
            
#     # test the conversion .
#     from skimage.segmentation import mark_boundaries
#    
#     all_mask = np.zeros(polar_grid.shape, dtype=np.int)
#    
# #    all_mask[:] = np.nan
#     for ii in range(len(polar_grid_fine_2_coarse_ids)):
#         ii_id = polar_grid_fine_2_coarse_ids[ii]
#         grid_ii = ve_grid_rot[ii_id]
#
#         heat_delta_area_ve_viz = heatmap_quad_vals(grid_ii, 
#                                                    (ii+1)*np.ones(len(grid_ii)), 
#                                                    polar_grid.shape)
#         all_mask[~np.isnan(heat_delta_area_ve_viz)] = ii +1
#    
#     fig, ax = plt.subplots()
#     ax.imshow(all_mask, cmap='coolwarm')
#     plt.show()
#    
#     blank = np.zeros((polar_grid.shape[0], 
#                       polar_grid.shape[1], 3))
#    
#     fig, ax = plt.subplots(figsize=(15,15))
#     ax.imshow(mark_boundaries(blank, all_mask))
#     plt.show()
    
        
    """
    Compute the coarse changes.
    """
    mean_all_mean_def_maps_ve_coarse = np.hstack([np.mean(all_mean_isotropic_strain_ve[:,quad_ind].mean(axis=1), axis=0) for quad_ind in polar_grid_fine_2_coarse_ids])
    mean_all_mean_def_maps_epi_coarse = np.hstack([np.mean(all_mean_isotropic_strain_epi[:,quad_ind].mean(axis=1), axis=0) for quad_ind in polar_grid_fine_2_coarse_ids])
    
    # map on a standard plot. 
    
    from skimage.measure import regionprops, find_contours
    
    rotate_angle = 270 + 360/8./2.
    
    regpropgrid = regionprops(polar_grid_region)
    reg_centroids = np.vstack([re.centroid for re in regpropgrid])
    
    rot_center = np.hstack([polar_grid_region.shape[0]//2, 
                             polar_grid_region.shape[1]//2])
    
    # find the boundaries of this point and rotate instead. 
    uniq_regions = np.unique(polar_grid_region)[1:]
    polar_grid_cont_lines = [find_contours(polar_grid_region == reg, 0)[0] for reg in uniq_regions]
    
    polar_grid_cont_lines = [rotate_pts(cc[:,[1,0]], angle=-rotate_angle, center=rot_center)[:,[1,0]] for cc in polar_grid_cont_lines]


    rotate_angle_fine = 270 + 360/(8.*int(supersample_theta))/2.
    
    regpropgrid = regionprops(polar_grid_fine)
    reg_centroids_fine = np.vstack([re.centroid for re in regpropgrid])
    
    rot_center = np.hstack([polar_grid_region.shape[0]//2, 
                             polar_grid_region.shape[1]//2])
    
     # find the boundaries of this point and rotate instead. 
    uniq_regions_fine = np.unique(polar_grid_fine)[1:]
    polar_grid_cont_lines_fine = [find_contours(polar_grid_fine == reg, 0)[0] for reg in uniq_regions_fine]
    
    polar_grid_cont_lines_fine = [rotate_pts(cc[:,[1,0]], angle=-rotate_angle_fine, center=rot_center)[:,[1,0]] for cc in polar_grid_cont_lines_fine]

    
    
    fig, ax = plt.subplots()
    plt.imshow(polar_grid_fine)
    ax.plot(polar_grid_cont_lines_fine[0][:,1], 
             polar_grid_cont_lines_fine[0][:,0],
             'r-')
    plt.show()
    
    
    polar_grid_lin_mean_areas_ve = heatmap_vals_radial(polar_grid_cont_lines, 
                                                         mean_all_mean_def_maps_ve_coarse, 
                                                         polar_grid_region.shape, alpha=0)
    
    polar_grid_lin_mean_areas_epi = heatmap_vals_radial(polar_grid_cont_lines, 
                                                         mean_all_mean_def_maps_epi_coarse, 
                                                         polar_grid_region.shape, alpha=0)
    
    polar_grid_lin_mean_areas_ve_fine = heatmap_vals_radial(polar_grid_cont_lines_fine, 
                                                         mean_all_mean_def_maps_ve, 
                                                         polar_grid_region.shape, alpha=0)
    
    polar_grid_lin_mean_areas_epi_fine = heatmap_vals_radial(polar_grid_cont_lines_fine, 
                                                         mean_all_mean_def_maps_epi, 
                                                         polar_grid_region.shape, alpha=0)
    
    
    plt.figure(figsize=(10,10))
    plt.title('VE')
    plt.imshow(polar_grid_lin_mean_areas_ve, cmap='coolwarm_r', vmin = -0.05, vmax=0.05)
    
    for cc in polar_grid_cont_lines:
         plt.plot(cc[:,1], 
                  cc[:,0], 'k-', lw=3)
    plt.show()
    
    
    plt.figure(figsize=(10,10))
    plt.title('Epi')
    plt.imshow(polar_grid_lin_mean_areas_epi, cmap='coolwarm_r', vmin = -0.05, vmax=0.05)
    
    for cc in polar_grid_cont_lines:
        plt.plot(cc[:,1], 
                  cc[:,0], 'k-', lw=3)
    plt.show()
    
    
    plt.figure(figsize=(10,10))
    plt.title('VE')
    plt.imshow(polar_grid_lin_mean_areas_ve_fine, cmap='coolwarm_r', vmin = -0.1, vmax=0.1)
    
    for cc in polar_grid_cont_lines:
         plt.plot(cc[:,1], 
                  cc[:,0], 'k-', lw=3)
    plt.show()
    
    
    plt.figure(figsize=(10,10))
    plt.title('VE')
    plt.imshow(polar_grid_lin_mean_areas_epi_fine, cmap='coolwarm_r', vmin = -0.1, vmax=0.1)
    
    for cc in polar_grid_cont_lines:
        plt.plot(cc[:,1], 
                  cc[:,0], 'k-', lw=3)
    plt.show()
    
    
    plt.figure(figsize=(10,10))
    plt.title('VE')
    plt.imshow(polar_grid_lin_mean_areas_epi_fine<0, cmap='coolwarm_r', vmin = -0.01, vmax=0.01)
    
    for cc in polar_grid_cont_lines:
        plt.plot(cc[:,1], 
                  cc[:,0], 'k-', lw=3)
    plt.show()

        
    """
    Save as .csv
    """
    master_analysis_save_folder = os.path.join('/media/felix/Srinivas4/All_results/Deformation_Analysis/Staged_Strain_Rate_Change_Analysis-growth_correct-Epi_Grid-auto-polar_Paper', analyse_stage)
    fio.mkdir(master_analysis_save_folder)
 #    
 #    """
 #    Load the particular save file 
 #    """
 #    
    statsfile = os.path.join(master_analysis_save_folder, '%s_strain_rate_change_statistics.mat' %(analyse_stage))
 #    statsobj = spio.loadmat(statsfile)    
#    
#    all_mean_isotropic_strain_ve = np.vstack(all_mean_isotropic_strain_ve)
#    all_mean_isotropic_strain_epi = np.vstack(all_mean_isotropic_strain_epi)
#    
#    all_mean_anisotropic_strain_ve = np.vstack(all_mean_anisotropic_strain_ve)
#    all_mean_anisotropic_strain_epi = np.vstack(all_mean_anisotropic_strain_epi)
#
#    all_mean_curl_strain_ve = np.vstack(all_mean_curl_strain_ve)
#    all_mean_curl_strain_epi = np.vstack(all_mean_curl_strain_epi)
#    
    
    all_analysed_embryos = np.hstack(all_analysed_embryos)
#    all_mean_def_maps_ve = np.vstack(all_mean_def_maps_ve)
#    all_mean_def_maps_epi = np.vstack(all_mean_def_maps_epi)
    spio.savemat(statsfile, {'all_analysed_embryos':all_analysed_embryos, 
                              'all_mean_isotropic_strain_ve': all_mean_isotropic_strain_ve,
                              'all_mean_isotropic_strain_epi': all_mean_isotropic_strain_epi,
                              'all_mean_anisotropic_strain_ve': all_mean_anisotropic_strain_ve,
                              'all_mean_anisotropic_strain_epi':all_mean_anisotropic_strain_epi,
                              'all_mean_curl_strain_ve':all_mean_curl_strain_ve,
                              'all_mean_curl_strain_epi':all_mean_curl_strain_epi,
#                              'mean_all_mean_def_maps_ve':mean_all_mean_def_maps_ve,
#                              'mean_all_mean_def_maps_epi':mean_all_mean_def_maps_epi,
                              'polar_grid_fine_2_coarse_ids':polar_grid_fine_2_coarse_ids})

#     """
#     Draw the mean change maps. 
#     """
#     fig, ax = plt.subplots(figsize=(10,10))
#     heat_delta_area_ve_viz = heatmap_quad_vals(ve_grid_rot, 
#                                                mean_all_mean_def_maps_ve, 
#                                                polar_grid.shape)
        
#     ax.imshow(heat_delta_area_ve_viz, cmap='coolwarm_r', vmin=-0.02, vmax=0.02)
#     plt.show()
                    
#     fig, ax = plt.subplots(figsize=(10,10))
#     heat_delta_area_epi_viz = heatmap_quad_vals(ve_grid_rot, 
#                                                 mean_all_mean_def_maps_epi, 
#                                                 polar_grid.shape)
        
#     ax.imshow(heat_delta_area_epi_viz, cmap='coolwarm_r', vmin=-0.02, vmax=0.02)
#     plt.show()
    

    
#    
#    # apply PCA here. 
#    from sklearn.decomposition import PCA
#    
#    pca_model = PCA(n_components=5, whiten=True)
#    pca_model.fit(all_mean_def_maps_epi)
#    
#    
#    for jj in range(5):
#        fig, ax = plt.subplots(figsize=(10,10))
#        heat_delta_area_ve_viz = heatmap_quad_vals(ve_grid_rot, 
#                                                   pca_model.components_[jj], 
#                                                   polar_grid.shape)
#            
#        ax.imshow(heat_delta_area_ve_viz, cmap='coolwarm_r', vmin=-0.1, vmax=0.1)
#        plt.show()
#        
#        
#    from sklearn.decomposition import PCA
#    
#    pca_model = PCA(n_components=5, whiten=True)
#    pca_model.fit(all_mean_def_maps_epi)
#    
#    
#    for jj in range(5):
#        fig, ax = plt.subplots(figsize=(10,10))
#        heat_delta_area_ve_viz = heatmap_quad_vals(ve_grid_rot, 
#                                                   pca_model.components_[jj], 
#                                                   polar_grid.shape)
#            
#        ax.imshow(heat_delta_area_ve_viz, cmap='coolwarm_r', vmin=-0.02, vmax=0.02)
#        plt.show()
#    
    
    
    
                    