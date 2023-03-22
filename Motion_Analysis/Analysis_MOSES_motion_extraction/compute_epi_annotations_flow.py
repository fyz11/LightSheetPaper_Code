#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:26:50 2019

@author: felix
"""

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

    
def parse_polygon_lines(line_text):
    
    line = line_text[1:-1].split(',"')
    
    line_x = []
    line_y = []
    
    for ll in line:
        if 'all_points_x' in ll:
#            print(ll.split(':')[-1][1:-1])
            line_x = np.hstack(ll.split(':')[-1][1:-1].split(',')).astype(np.float)
        if 'all_points_y' in ll:
            line_y = np.hstack(ll.split(':')[-1][1:-1].split(',')).astype(np.float)
    return np.vstack([line_x, line_y]).T


def draw_polygon_mask(xy, shape):
    
    from skimage.draw import polygon
    img = np.zeros(shape, dtype=np.bool)
    rr, cc = polygon(xy[:,1].astype(np.int),
                     xy[:,0].astype(np.int))
    img[rr, cc] = 1
    
    return img


def retrieve_line_ids(fnames, frames, ve_fname, hex_present=False):
    
    if hex_present:
        wanted_frame_files = np.hstack([ve_fname.split('_params')[0]+'_ve-epi-hex_vid_geodesic_polar'+str(f+1).zfill(3)+'.jpg' for f in frames])
    else:
        wanted_frame_files = np.hstack([ve_fname.split('_params')[0]+'_ve-epi_vid_geodesic_polar'+str(f+1).zfill(3)+'.jpg' for f in frames])
#    print(wanted_frame_files)
    wanted_ids = np.hstack([np.arange(len(fnames))[fnames==f] for f in wanted_frame_files])
    
    return wanted_ids
    

def prop_boundary_lines_flow(line, flow, start_frame=0, smoothing_sigma=15):
    
    from scipy.interpolate import RegularGridInterpolator
    from skimage.filters import gaussian
#    from skimage.filters.rank import median 
    from scipy.signal import medfilt, medfilt2d, convolve2d
    import skimage.morphology as skmorph
    
    n_frames_flow = len(flow)
    
    all_lines = [line]
    
    line_x = line[:,0].copy()
    line_y = line[:,1].copy()
    
    print(line_x.shape)
    
    mean_kernel = skmorph.disk(smoothing_sigma).astype(np.float32) 
    mean_kernel = mean_kernel / float(np.sum(mean_kernel))
    
    for ii in range(n_frames_flow-start_frame):
        
        flow_frame = flow[ii+start_frame]
#        max_mag = np.max(np.abs(flow_frame))
#        
#        print(max_mag)
#        flow_frame = flow_frame/max_mag
#        f_u = RegularGridInterpolator((range(flow_frame.shape[0]),
#                                       range(flow_frame.shape[1])), 
#                                       flow_frame[...,0], method='linear')
#        f_v = RegularGridInterpolator((range(flow_frame.shape[0]),
#                                       range(flow_frame.shape[1])),   
#                                       flow_frame[...,1], method='linear')
#
#        line_x = line_x + f_u(np.vstack([line_y, line_x]).T)
#        line_y = line_y + f_v(np.vstack([line_y, line_x]).T)
#        line_x = line_x + gaussian(flow_frame[...,0], sigma=smoothing_sigma, preserve_range=True)[line_y.astype(np.int), line_x.astype(np.int)]
#        line_y = line_y + gaussian(flow_frame[...,1], sigma=smoothing_sigma, preserve_range=True)[line_y.astype(np.int), line_x.astype(np.int)]
#        line_x = line_x + max_mag*median(flow_frame[...,0], skmorph.disk(smoothing_sigma))[line_y.astype(np.int), line_x.astype(np.int)]
#        line_y = line_y + max_mag*median(flow_frame[...,1], skmorph.disk(smoothing_sigma))[line_y.astype(np.int), line_x.astype(np.int)]
#        line_x = line_x + medfilt(flow_frame[...,0], kernel_size=[smoothing_sigma, smoothing_sigma])[line_y.astype(np.int), line_x.astype(np.int)]
#        line_y = line_y + medfilt(flow_frame[...,1], kernel_size=[smoothing_sigma, smoothing_sigma])[line_y.astype(np.int), line_x.astype(np.int)]
#        line_x = line_x + medfilt2d(flow_frame[...,0], kernel_size=smoothing_sigma)[line_y.astype(np.int), line_x.astype(np.int)]
#        line_y = line_y + medfilt2d(flow_frame[...,1], kernel_size=smoothing_sigma)[line_y.astype(np.int), line_x.astype(np.int)]
        line_x = line_x + convolve2d(flow_frame[...,0], mean_kernel, mode='same', boundary='symm')[line_y.astype(np.int), line_x.astype(np.int)]
        line_y = line_y + convolve2d(flow_frame[...,1], mean_kernel, mode='same', boundary='symm')[line_y.astype(np.int), line_x.astype(np.int)]
        
        all_lines.append(np.vstack([line_x,line_y]).T)
        
    return np.array(all_lines)
        
def supersample_lines(line, factor=100):
    
    from scipy import interpolate 
    # applies spline based period interpolation as given here. https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
    x = np.hstack([line[:,0], line[0,0]])
    y = np.hstack([line[:,1], line[0,1]])
    
    tck, u = interpolate.splprep([x, y], s=0, per=True)

    # evaluate the spline fits for 1000 evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, factor), tck)
    
    return np.vstack([xi, yi]).T

    
if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    import glob 
    import os 
    import scipy.io as spio
    from skimage.io import imsave
    
    import MOSES.Utility_Functions.file_io as fio
    from MOSES.Optical_Flow_Tracking.optical_flow import farnebackflow, TVL1Flow, DeepFlow
    from MOSES.Optical_Flow_Tracking.superpixel_track import compute_vid_opt_flow, compute_grayscale_vid_superpixel_tracks, compute_grayscale_vid_superpixel_tracks_w_optflow
    from MOSES.Visualisation_Tools.track_plotting import plot_tracks
    from flow_vis import make_colorwheel, flow_to_color
    
    from skimage.filters import threshold_otsu, gaussian
    from scipy.ndimage.morphology import binary_fill_holes
    import skimage.transform as sktform
    from tqdm import tqdm 
    import pandas as pd 
    
    
#    master_analysis_save_folder = '/media/felix/Srinivas4/All_results/Migration_Analysis/Staged_Analysis'
#    fio.mkdir(master_analysis_save_folder)
    
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Set up a nearest neighbour model 
    from sklearn.neighbors import NearestNeighbors 
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
    #     2. Get the relevant flow
    # =============================================================================
    n_spixels = 5000 # set the number of pixels. 
    smoothwinsize = 3


#    for embryo_folder in embryo_folders[-2:-1]:
    for embryo_folder in all_embryo_folders[10:11]: # L401, ; done L945, L921
#    for embryo_folder in embryo_folders[4:5]:
        print(embryo_folder)
        # find the geodesic-rotmatrix. 
        unwrapped_folder = locate_unwrapped_im_folder(embryo_folder, key='geodesic-rotmatrix')
        
        if len(unwrapped_folder) == 1:
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
            saveannotfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow-epi-boundary')
            
            hollyannotfolder = '/media/felix/My Passport/Shankar-2/All_Manual_Annotations/all_midline_annotations'
            print(saveannotfolder)
            fio.mkdir(saveannotfolder) # create this folder. 
            
            unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))
            

            if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
            
            else:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_hex_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_hex_embryo(unwrap_im_files, unwrap_meta_files)
            
            
            # iterate over the pairs and parse just the polar ones, for the rectangular -> we map the polar (upsampling ? ) -> periodic based linear sampling. 
            for ii in tqdm(range(len(paired_unwrap_file_condition))[:]):
                
                unwrapped_condition = paired_unwrap_file_condition[ii]
                
                if unwrapped_condition.split('-')[-1] == 'polar':
                
                    if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
                        ve_unwrap_file, epi_unwrap_file = paired_unwrap_files[ii]
                    else:
                        ve_unwrap_file, epi_unwrap_file, hex_unwrap_file = paired_unwrap_files[ii]
                        
    #                unwrap_im_file = unwrap_im_files[ii]
    #                print(unwrap_im_file)
                    vid_ve = fio.read_multiimg_PIL(ve_unwrap_file)
                    vid_epi = fio.read_multiimg_PIL(epi_unwrap_file)
                    vid_epi_resize = sktform.resize(vid_epi, vid_ve.shape, preserve_range=True)
                    
                    if 'MTMG-HEX' in embryo_folder:
                        vid_hex = fio.read_multiimg_PIL(hex_unwrap_file)
                    
                    """
                    1. load the flow file for VE and Epi surfaces. 
                    """
                    flowfile_ve = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(ve_unwrap_file)[-1].replace('.tif', '.mat'))
                    flow_ve = spio.loadmat(flowfile_ve)['flow']
                    
                    flowfile_epi = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(epi_unwrap_file)[-1].replace('.tif', '.mat'))
                    flow_epi = spio.loadmat(flowfile_epi)['flow']
                    flow_epi_resize = sktform.resize(flow_epi, flow_ve.shape, preserve_range=True)
                    
                    
                    if 'MTMG-HEX' in embryo_folder:
                        flowfile_hex = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(hex_unwrap_file)[-1].replace('.tif', '.mat'))
                        flow_hex = spio.loadmat(flowfile_hex)['flow']
                        
                    
                    """
                    2. load the annotated Epi boundary files 
                    """
                    if len(unwrapped_condition.split('-')[1]) > 0: 
                        contourfile = os.path.join(hollyannotfolder, os.path.split(embryo_folder)[-1].split('_TP')[0]+'_via_region_data.csv')
                    else:
                        if '_Emb' in os.path.split(embryo_folder)[-1]:
                            contourfile = os.path.join(hollyannotfolder, os.path.split(embryo_folder)[-1].split('_Emb')[0]+'_via_region_data.csv')
                        else:
                            if '_1View' in os.path.split(embryo_folder)[-1]:
                                contourfile = os.path.join(hollyannotfolder, os.path.split(embryo_folder)[-1].split('_1View')[0]+'_via_region_data.csv')
                            else:
                                contourfile = os.path.join(hollyannotfolder, os.path.split(embryo_folder)[-1].split('_TP')[0]+'_via_region_data.csv')
                   
                    contourtab = pd.read_csv(contourfile) # read it 
                    contour_lines = dict(contourtab['region_shape_attributes'])
                    
                    contourfnames = contourtab['filename'].values
                    contourtabframes = np.hstack([int(os.path.split(f)[-1].split('polar')[1].split('.jpg')[0])-1 for f in contourfnames])
    
                    # load the fnames and frames relevant for this unwrapped condition. 
                    
                    if 'MTMG-HEX' in embryo_folder:
                        line_ids = retrieve_line_ids(contourfnames, np.unique(contourtabframes), os.path.split(ve_unwrap_file)[-1].split('.tif')[0], hex_present=True) # get the relevant lines. 
                    else:
                        line_ids = retrieve_line_ids(contourfnames, np.unique(contourtabframes), os.path.split(ve_unwrap_file)[-1].split('.tif')[0]) # get the relevant lines. 
                    contourtabframes_line_ids = contourtabframes[line_ids]
                    
                    line_ids_sort = line_ids[np.argsort(contourtabframes_line_ids)]
                    plot_lines = [parse_polygon_lines(contour_lines[line]) for line in line_ids_sort]
                    
                    # supersample the lines to increase accuracy
                    plot_lines = [supersample_lines(line, factor=50) for line in plot_lines]
                    
                    
    #                contour_lines = contour_lines[line_ids]# fetch the relevant contour lines
                    # plot the 0th frame line. 
                    fig, ax = plt.subplots()
                    plt.title('Frame-%s-VE' %(str(line_ids_sort[0]+1).zfill(3)))
                    ax.imshow(vid_ve[contourtabframes[line_ids_sort[0]]], cmap='gray')
                    ax.plot(np.hstack([plot_lines[0][:,0], plot_lines[0][0,0]]), 
                            np.hstack([plot_lines[0][:,1], plot_lines[0][0,1]]), 'r--', alpha=1, lw=3)
                    plt.show()
                    
                    fig, ax = plt.subplots()
                    plt.title('Frame-%s-Epi' %(str(line_ids_sort[0]+1).zfill(3)))
                    ax.imshow(vid_epi_resize[contourtabframes[line_ids_sort[0]]], cmap='gray')
                    ax.plot(np.hstack([plot_lines[0][:,0], plot_lines[0][0,0]]), 
                            np.hstack([plot_lines[0][:,1], plot_lines[0][0,1]]), 'r--', alpha=1, lw=3)
                    plt.show()
                    
                    
                    """
                    3. Prop the annotated Epi boundary files from the 0th frame to all other frames using optical flow. 
                    """
                    # use smoothing_sigma = 25 for the case of LifeAct. 
                    plot_lines_prop_vid = prop_boundary_lines_flow(plot_lines[0], flow_epi_resize, smoothing_sigma=25) # smoothing is very important for the boundary. 
                    
                    finalsavefolder = os.path.join(saveannotfolder, unwrapped_condition); fio.mkdir(finalsavefolder)
                    finalvizsavefolder = os.path.join(saveannotfolder, unwrapped_condition, 'visualisation'); fio.mkdir(finalvizsavefolder)
                    
                    spio.savemat(os.path.join(finalsavefolder, 'inferred_epi_boundary-'+unwrapped_condition+'.mat'), 
                                                                 {'contour_line': plot_lines_prop_vid.astype(np.float32)})
                    
                    for tt in range(len(vid_epi_resize)):
                        
                        if tt in contourtabframes_line_ids:
                            
                            print('true')
                            fig, ax = plt.subplots()
                            plt.title('Frame-%s-Epi' %(str(tt+1).zfill(3)))
                            ax.imshow(vid_epi_resize[tt], cmap='gray')
                            ax.plot(np.hstack([plot_lines_prop_vid[tt][:,0], plot_lines_prop_vid[tt][0,0]]), 
                                    np.hstack([plot_lines_prop_vid[tt][:,1], plot_lines_prop_vid[tt][0,1]]), 'r--', alpha=1, lw=3)
                            select_manual_line = np.arange(len(contourtabframes_line_ids))[contourtabframes_line_ids==tt][0]
                            ax.plot(np.hstack([plot_lines[select_manual_line][:,0], plot_lines[select_manual_line][0,0]]), 
                                    np.hstack([plot_lines[select_manual_line][:,1], plot_lines[select_manual_line][0,1]]), 'b--', alpha=1, lw=3)
                            fig.savefig(os.path.join(finalvizsavefolder, 'inferred_epi_boundary_overlay-'+unwrapped_condition+'-%s.svg' %(str(tt+1).zfill(3))), dpi=300, bbox_inches='tight')
                            plt.show()
                            
                        else:
                            
                            fig, ax = plt.subplots()
                            plt.title('Frame-%s-Epi' %(str(tt+1).zfill(3)))
                            ax.imshow(vid_epi_resize[tt], cmap='gray')
                            ax.plot(np.hstack([plot_lines_prop_vid[tt][:,0], plot_lines_prop_vid[tt][0,0]]), 
                                    np.hstack([plot_lines_prop_vid[tt][:,1], plot_lines_prop_vid[tt][0,1]]), 'r--', alpha=1, lw=3)
    #                        ax.plot(np.hstack([plot_lines[2][:,0], plot_lines[1][0,0]]), 
    #                                np.hstack([plot_lines[2][:,1], plot_lines[1][0,1]]), 'b--', alpha=1, lw=3)
                            fig.savefig(os.path.join(finalvizsavefolder, 'inferred_epi_boundary_overlay-'+unwrapped_condition+'-%s.svg' %(str(tt+1).zfill(3))), dpi=300, bbox_inches='tight')
                            plt.show()
                        
                
                
                
                
                
                
                
                
                    
                
                
                    
                    
