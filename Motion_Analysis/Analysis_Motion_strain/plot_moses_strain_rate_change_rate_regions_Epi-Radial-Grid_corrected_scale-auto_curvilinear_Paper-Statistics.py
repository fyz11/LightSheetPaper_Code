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


def tile_uniform_windows_radial_guided_line(imsize, n_r, n_theta, max_r, mid_line, center=None, bound_r=True, zero_angle=None):
    
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
        
    return spixels 


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


# add a rotate_pts_function
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


def find_quad_control_lines(polar_grid_region):
   
    from skimage.measure import find_contours
    
    uniq_regions = np.unique(polar_grid_region)[1:]
    polar_grid_cont_lines =  []
    
    for reg in uniq_regions:
        cont = find_contours(polar_grid_region == reg, 0)
        cont = cont[np.argmax([len(cc) for cc in cont])]
        
        polar_grid_cont_lines.append(cont)
        
    return polar_grid_cont_lines
    

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
    
    from skimage.filters import threshold_otsu
    import skimage.transform as sktform
    from tqdm import tqdm 
    import pandas as pd 
    
    from skimage.filters import gaussian
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.measure import find_contours
    
#    master_analysis_save_folder = '/media/felix/Srinivas4/All_results/Migration_Analysis/Staged_Analysis'
#    fio.mkdir(master_analysis_save_folder)
    
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Set up a nearest neighbour model 
    from sklearn.neighbors import NearestNeighbors 
    from skimage.segmentation import mark_boundaries
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
#    all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-LifeAct-smooth-lam10-correct_norot.xlsx', 
#                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-MTMG-TTR-smooth-lam10-correct_norot.xlsx',
#                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-MTMG-HEX-smooth-lam10-correct_norot.xlsx',
#                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-new_hex-smooth-lam10-correct_norot.xlsx']
    all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-LifeAct-smooth-lam5-correct_norot_consensus.xlsx', 
                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-TTR-smooth-lam5-correct_norot_consensus.xlsx',
                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-HEX-smooth-lam5-correct_norot_consensus.xlsx',
                         '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-new-HEX-smooth-lam5-correct_norot_consensus.xlsx']
    
    staging_tab = pd.concat([pd.read_excel(staging_file) for staging_file in all_staging_files], ignore_index=True)
    all_stages = staging_tab.columns[1:4]
#    
#    # =============================================================================
#    #     3. Get the initial estimation of the VE migration directions to build the quadranting... 
#    # =============================================================================
##    all_angle_files = ['/media/felix/Srinivas4/correct_angle_LifeAct.csv',
##                       '/media/felix/Srinivas4/correct_angle_MTMG-TTR.csv',
##                       '/media/felix/Srinivas4/correct_angle_MTMG-HEX.csv',
##                       '/media/felix/Srinivas4/correct_angle_new_hex.csv']
##    all_angle_files = ['/media/felix/Srinivas4/correct_angle-global-1000_LifeAct.csv',
##                       '/media/felix/Srinivas4/correct_angle-global-1000_MTMG-TTR.csv',
##                       '/media/felix/Srinivas4/correct_angle-global-1000_MTMG-HEX.csv',
##                       '/media/felix/Srinivas4/correct_angle-global-1000_new_hex.csv']
#    
    # Use newest consensus angles. !
#    all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles/LifeAct_polar_angles_AVE_classifier.-consensus.csv',
#                       '/media/felix/Srinivas4/Data_Info/Migration_Angles/MTMG-TTR_polar_angles_AVE_classifier.-consensus.csv',
#                       '/media/felix/Srinivas4/Data_Info/Migration_Angles/MTMG-HEX_polar_angles_AVE_classifier.-consensus.csv',
#                       '/media/felix/Srinivas4/Data_Info/Migration_Angles/new_hex_polar_angles_AVE_classifier.-consensus.csv']
#    all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/LifeAct_polar_angles_AVE_classifier-consensus.csv',
#                       '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-TTR_polar_angles_AVE_classifier-consensus.csv',
#                       '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-HEX_polar_angles_AVE_classifier-consensus.csv',
#                       '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/new_hex_polar_angles_AVE_classifier-consensus.csv']
#    
#    angle_tab = pd.concat([pd.read_csv(angle_file) for angle_file in all_angle_files], ignore_index=True)
#    all_stages = staging_tab.columns[1:4]
    
    
#    plotsavefolder = '/media/felix/Srinivas4/All_results/Deformation_Analysis/Staged_Area_Change_Analysis-growth_correct-Epi_Grid-auto-polar_Paper/Plots'
#    plotsavefolder = '/media/felix/Srinivas4/All_results/Deformation_Analysis/Staged_Strain_Rate_Change_Analysis-growth_correct-Epi_Grid-auto-polar_Paper/Plots'
    plotsavefolder = '/media/felix/Srinivas4/All_results/Deformation_Analysis/Staged_Strain_Rate_Change_Analysis-growth_correct-Epi_Grid-auto-polar_Paper/Plots'
    
    fio.mkdir(plotsavefolder)
    
    analyse_stages = [str(s) for s in all_stages]
    
    data_stages_coarse = []
    data_stages_fine = []
    
    
    plotkey = 'all_mean_isotropic_strain'
#    plotkey = 'all_mean_anisotropic_strain'
#    plotkey = 'all_mean_curl_strain'
    
    
    # iterate over all stages to load in the velocity data
    for analyse_stage in analyse_stages:
        print(analyse_stage)
        
#        master_analysis_save_folder = os.path.join('/media/felix/Srinivas4/All_results/Deformation_Analysis/Staged_Area_Change_Analysis-growth_correct-Epi_Grid-auto-polar_Paper', analyse_stage)
        master_analysis_save_folder = os.path.join('/media/felix/Srinivas4/All_results/Deformation_Analysis/Staged_Strain_Rate_Change_Analysis-growth_correct-Epi_Grid-auto-polar_Paper', analyse_stage)
  
        # these are already corrected for the temporal sampling. (should be h^(-1))
#        file_obj = spio.loadmat(os.path.join(master_analysis_save_folder,'%s_area_change_statistics.mat' %(analyse_stage))) 
        file_obj = spio.loadmat(os.path.join(master_analysis_save_folder, '%s_strain_rate_change_statistics.mat' %(analyse_stage))) 
        
        print(file_obj.keys())
        
#        spio.savemat(statsfile, {'all_analysed_embryos':all_analysed_embryos, 
#                              'all_mean_isotropic_strain_ve': all_mean_isotropic_strain_ve,
#                              'all_mean_isotropic_strain_epi': all_mean_isotropic_strain_epi,
#                              'all_mean_anisotropic_strain_ve': all_mean_anisotropic_strain_ve,
#                              'all_mean_anisotropic_strain_epi':all_mean_anisotropic_strain_epi,
#                              'all_mean_curl_strain_ve':all_mean_curl_strain_ve,
#                              'all_mean_curl_strain_epi':all_mean_curl_strain_epi,
##                              'mean_all_mean_def_maps_ve':mean_all_mean_def_maps_ve,
##                              'mean_all_mean_def_maps_epi':mean_all_mean_def_maps_epi,
#                              'polar_grid_fine_2_coarse_ids':polar_grid_fine_2_coarse_ids})
        
        
        all_analysed_embryos = file_obj['all_analysed_embryos']
        
        all_mean_data_ve = file_obj['%s_ve' %(plotkey)]
        all_mean_data_epi = file_obj['%s_epi' %(plotkey)]
        
        """
        csv export. 
        """
        column_titles = np.hstack([['Embryos'],
                                   ['Quad_%s' %str(jj+1) for jj in np.arange(all_mean_data_ve.shape[1])]])
        data_table = pd.DataFrame(np.hstack([all_analysed_embryos[:,None], 
                                             all_mean_data_ve]), index=None,
                                             columns = column_titles)
        
        savefile = os.path.join(master_analysis_save_folder,'%s_strain_rate_change_statistics_%s_per_h-VE.csv' %(analyse_stage, plotkey))
        data_table.to_csv(savefile, index=None)
        
        column_titles = np.hstack([['Embryos'],
                                   ['Quad_%s' %str(jj+1) for jj in np.arange(all_mean_data_epi.shape[1])]])
        data_table = pd.DataFrame(np.hstack([all_analysed_embryos[:,None], 
                                             all_mean_data_epi]), index=None,
                                             columns = column_titles)
        
        savefile = os.path.join(master_analysis_save_folder,'%s_strain_rate_change_statistics_%s_per_h-Epi.csv' %(analyse_stage, plotkey))
        data_table.to_csv(savefile, index=None)
        
        
        """
        Compute the mean statistics for plot. 
        """
        mean_ve = np.mean(all_mean_data_ve, axis=0)
        mean_epi = np.mean(all_mean_data_epi, axis=0)
        
        data_stage_fine = [(mean_ve, mean_epi)]
        data_stages_fine.append(data_stage_fine)
        
        """
        Conversion to coarse and saving this out too. 
        """
        coarse_to_fine_ids = file_obj['polar_grid_fine_2_coarse_ids']
#        mean_ve_coarse = np.hstack([ np.mean( np.mean(all_mean_delta_area_ve[:,kkk], axis=0)) for kkk in coarse_to_fine_ids])
#        mean_epi_coarse = np.hstack([ np.mean( np.mean(all_mean_delta_area_epi[:,kkk], axis=0)) for kkk in coarse_to_fine_ids])
#        mean_ve_nosmooth_coarse = np.hstack([ np.mean( np.mean(all_mean_delta_area_ve_raw[:,kkk], axis=0)) for kkk in coarse_to_fine_ids])
#        mean_epi_nosmooth_coarse = np.hstack([ np.mean( np.mean(all_mean_delta_area_epi_raw[:,kkk], axis=0)) for kkk in coarse_to_fine_ids])
        mean_ve_coarse = np.hstack([ np.mean( np.mean(all_mean_data_ve[:,kkk], axis=1), axis=0) for kkk in coarse_to_fine_ids])
        mean_epi_coarse = np.hstack([ np.mean( np.mean(all_mean_data_epi[:,kkk], axis=1), axis=0) for kkk in coarse_to_fine_ids])
        
        mean_ve_coarse_individual = np.vstack([np.mean(all_mean_data_ve[:,kkk], axis=1) for kkk in coarse_to_fine_ids]).T
        mean_epi_coarse_individual = np.vstack([np.mean(all_mean_data_epi[:,kkk], axis=1) for kkk in coarse_to_fine_ids]).T
        
        
        column_titles = np.hstack([['Embryos'],
                                   ['Quad_%s' %str(jj+1) for jj in np.arange(mean_ve_coarse_individual.shape[1])]])
        data_table = pd.DataFrame(np.hstack([all_analysed_embryos[:,None], 
                                             mean_ve_coarse_individual]), index=None,
                                             columns = column_titles)
        
        savefile = os.path.join(master_analysis_save_folder,'%s_strain_rate_change_statistics_%s_per_h-VE-8quad.csv' %(analyse_stage, plotkey))
        data_table.to_csv(savefile, index=None)
        
        column_titles = np.hstack([['Embryos'],
                                   ['Quad_%s' %str(jj+1) for jj in np.arange(mean_ve_coarse_individual.shape[1])]])
        data_table = pd.DataFrame(np.hstack([all_analysed_embryos[:,None], 
                                             mean_epi_coarse_individual]), index=None,
                                             columns = column_titles)
        
        savefile = os.path.join(master_analysis_save_folder,'%s_strain_rate_change_statistics_%s_per_h-Epi-8quad.csv' %(analyse_stage, plotkey))
        data_table.to_csv(savefile, index=None)
        
        
        data_stage_coarse = [(mean_ve_coarse, mean_epi_coarse)]        
        data_stages_coarse.append(data_stage_coarse)
        
        
# =============================================================================
#     Set up the normalised plots 
# =============================================================================
        
    # setup the universal grid lines.
    """
    Custom radial plot with background 
    """
    plotsize = 720
    supersample_theta = 4
    supersample_r = 4
    
    theta_spacing = int(supersample_theta)+1
    r_spacing = int(supersample_r)
    
#    max_plot_range = 0.2
#    max_plot_range_coarse = 0.2
    # isotropic
    max_plot_range = .1
    max_plot_range_coarse = .1
#    # anisotropic
#    max_plot_range = 1.5
#    max_plot_range_coarse = 1.5
    max_plot_range_0 = 1.
    max_plot_range_0_coarse = 1.
    
    from skimage.measure import regionprops
    
    rotate_angle = 270 + 360/8./2.
    
    polar_grid_region = tile_uniform_windows_radial_custom( (plotsize,plotsize), 
                                                            n_r=4, 
                                                            n_theta=8, 
                                                            max_r=plotsize/2./1.2, 
                                                            mask_distal=None, # leave a set distance
                                                            center=None, 
                                                            bound_r=True)
    
    regpropgrid = regionprops(polar_grid_region)
    reg_centroids = np.vstack([re.centroid for re in regpropgrid])
    
    rot_center = np.hstack([polar_grid_region.shape[0]//2, 
                            polar_grid_region.shape[1]//2])
    
    # find the boundaries of this point and rotate instead. 
    uniq_regions = np.unique(polar_grid_region)[1:]
    polar_grid_cont_lines = find_quad_control_lines(polar_grid_region)
#    polar_grid_cont_lines = [find_contours(polar_grid_region == reg, 0)[0] for reg in uniq_regions]

    polar_grid_cont_lines = [rotate_pts(cc[:,[1,0]], angle=-rotate_angle, center=rot_center)[:,[1,0]] for cc in polar_grid_cont_lines]

#    # get the centroid coordinates for each region.
#    regpropgrid = regionprops(rotated_grid)
    reg_centroids = rotate_pts(reg_centroids[:,[1,0]], angle=-rotate_angle, center=rot_center)[:,[1,0]]
    


    polar_grid_fine = tile_uniform_windows_radial_custom( (plotsize,plotsize), 
                                                            n_r=int(supersample_r)*4, 
                                                            n_theta=(int(supersample_theta)+1)*8,
                                                            max_r=plotsize/2./1.2, 
                                                            mask_distal=None, # leave a set distance
                                                            center=None, 
                                                            bound_r=True)
    
    # also the fine grid.
    rotate_angle_fine = 270 + 360/(8.*int(supersample_theta+1.))/2.
    
    regpropgrid = regionprops(polar_grid_fine)
    reg_centroids_fine = np.vstack([re.centroid for re in regpropgrid])
    
    rot_center = np.hstack([polar_grid_region.shape[0]//2, 
                            polar_grid_region.shape[1]//2])
    
    # find the boundaries of this point and rotate instead. 
    uniq_regions_fine = np.unique(polar_grid_fine)[1:]
#    polar_grid_cont_lines_fine = [find_contours(polar_grid_fine == reg, 0)[0] for reg in uniq_regions_fine]
    polar_grid_cont_lines_fine = find_quad_control_lines(polar_grid_fine)
    
    polar_grid_cont_lines_fine = [rotate_pts(cc[:,[1,0]], angle=-rotate_angle_fine, center=rot_center)[:,[1,0]] for cc in polar_grid_cont_lines_fine]

    
    
    fig, ax = plt.subplots()
    plt.imshow(polar_grid_fine)
    ax.plot(polar_grid_cont_lines_fine[0][:,1], 
            polar_grid_cont_lines_fine[0][:,0],
            'r-')
    plt.show()
    
# =============================================================================
#     Draw the fine results.
# =============================================================================
    
    for ii in range(len(data_stages_fine)):
        
#        (strain_ve, strain_epi), (strain_ve_0, strain_epi_0) = data_stages_fine[ii]
        [(strain_ve, strain_epi)] = data_stages_fine[ii]
        
        polar_grid_areas_ve = heatmap_vals_radial(polar_grid_cont_lines_fine, 
                                                  strain_ve, 
                                                  polar_grid_region.shape, alpha=0)
    
    
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('VE - %s - $\Delta$Strain/h' %(analyse_stages[ii]))
        cmap = ax.imshow(polar_grid_areas_ve, vmin=-max_plot_range, vmax=max_plot_range, cmap='coolwarm_r')
        

        for cc in polar_grid_cont_lines_fine:
            plt.plot(cc[:,1], 
                     cc[:,0], 'k-', lw=1)
            
        for cc in polar_grid_cont_lines:
            plt.plot(cc[:,1], 
                     cc[:,0], 'y-', lw=2, zorder=100)
            
            
        plt.colorbar(cmap, fraction=0.05, pad=0.02, aspect=4)
#        fig.savefig(os.path.join(plotsavefolder, 'VE - %s - change_Area_fraction_per_h.svg' %(analyse_stages[ii])), dpi=300, bbox_inches='tight')
        plt.grid('off')
        plt.axis('off')
        fig.savefig(os.path.join(plotsavefolder, 'VE - %s - change_strain_%s_fraction_per_h.svg' %(analyse_stages[ii], plotkey)), dpi=300, bbox_inches='tight')
        plt.show()
        
        
        polar_grid_areas_epi = heatmap_vals_radial(polar_grid_cont_lines_fine, 
                                                  strain_epi, 
                                                  polar_grid_region.shape, alpha=0)
    
    
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('Epi - %s - $\Delta$Area/h' %(analyse_stages[ii]))
        cmap = ax.imshow(polar_grid_areas_epi, vmin=-max_plot_range, vmax=max_plot_range, cmap='coolwarm_r')
        for cc in polar_grid_cont_lines_fine:
            plt.plot(cc[:,1], 
                     cc[:,0], 'k-', lw=1)
            
        for cc in polar_grid_cont_lines:
            plt.plot(cc[:,1], 
                     cc[:,0], 'y-', lw=2, zorder=100)
        plt.colorbar(cmap, fraction=0.05, pad=0.02, aspect=4)
        plt.grid('off')
        plt.axis('off')
        fig.savefig(os.path.join(plotsavefolder, 'Epi - %s -  change_strain_%s_fraction_per_h.svg' %(analyse_stages[ii], plotkey)), dpi=300, bbox_inches='tight')
        plt.show()
        
        
#    
#        """
#        Mean change rate from state 0 
#        """
#        polar_grid_areas_ve = heatmap_vals_radial(polar_grid_cont_lines_fine, 
#                                                  area_ve_0, 
#                                                  polar_grid_region.shape, alpha=0)
#        
#        fig, ax = plt.subplots(figsize=(10,10))
#        plt.title('VE - %s - $\Delta$Area/h - TP0' %(analyse_stages[ii]))
#        cmap = ax.imshow(polar_grid_areas_ve, vmin=-max_plot_range_0, vmax=max_plot_range_0, cmap='coolwarm_r')
#        
#
#        for cc in polar_grid_cont_lines_fine:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=1)
#            
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'y-', lw=2, zorder=100)
#            
#            
#        plt.colorbar(cmap, fraction=0.05, pad=0.02, aspect=4)
#        plt.grid('off')
#        plt.axis('off')
#        fig.savefig(os.path.join(plotsavefolder, 'VE - tp0 - %s - change_Area_fraction_per_h.svg' %(analyse_stages[ii])), dpi=300, bbox_inches='tight')
#        
#        plt.show()
#        
#        
#        polar_grid_areas_epi = heatmap_vals_radial(polar_grid_cont_lines_fine, 
#                                                  area_epi_0, 
#                                                  polar_grid_region.shape, alpha=0)
#    
#    
#        fig, ax = plt.subplots(figsize=(10,10))
#        plt.title('Epi - %s - $\Delta$Area/h - TP0' %(analyse_stages[ii]))
#        cmap = ax.imshow(polar_grid_areas_epi, vmin=-max_plot_range_0, vmax=max_plot_range_0, cmap='coolwarm_r')
#        for cc in polar_grid_cont_lines_fine:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=1)
#            
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'y-', lw=2, zorder=100)
#        plt.colorbar(cmap, fraction=0.05, pad=0.02, aspect=4)
#        plt.grid('off')
#        plt.axis('off')
#        fig.savefig(os.path.join(plotsavefolder, 'Epi - tp0 - %s - change_Area_fraction_per_h.svg' %(analyse_stages[ii])), dpi=300, bbox_inches='tight')
#        
#        plt.show()
#        
#        
# =============================================================================
#         Start of coarse plotting. 
# =============================================================================
        
    for ii in range(len(data_stages_coarse)):
        
#        (area_ve, area_epi), (area_ve_nosmooth, area_epi_nosmooth), (area_ve_0, area_epi_0) = data_stages_coarse[ii]
        [(strain_ve, strain_epi)] = data_stages_coarse[ii]
        
        polar_grid_areas_ve = heatmap_vals_radial(polar_grid_cont_lines, 
                                                  strain_ve, 
                                                  polar_grid_region.shape, alpha=0)
    
    
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('VE - %s - $\Delta$Strain/h' %(analyse_stages[ii]))
        cmap = ax.imshow(polar_grid_areas_ve, vmin=-max_plot_range_coarse, vmax=max_plot_range_coarse, cmap='coolwarm_r')
        

#        for cc in polar_grid_cont_lines_fine:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=1)
            
        for cc in polar_grid_cont_lines:
            plt.plot(cc[:,1], 
                     cc[:,0], 'k-', lw=2, zorder=100)
            
            
        plt.colorbar(cmap, fraction=0.05, pad=0.02, aspect=4)
        plt.grid('off')
        plt.axis('off')
        fig.savefig(os.path.join(plotsavefolder, 'VE - coarse - %s - change_strain_%s_fraction_per_h.svg' %(analyse_stages[ii], plotkey)), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        
        polar_grid_areas_epi = heatmap_vals_radial(polar_grid_cont_lines, 
                                                  strain_epi, 
                                                  polar_grid_region.shape, alpha=0)
    
    
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('Epi - %s - $\Delta$Strain/h' %(analyse_stages[ii]))
        cmap = ax.imshow(polar_grid_areas_epi, vmin=-max_plot_range_coarse, vmax=max_plot_range_coarse, cmap='coolwarm_r')
#        for cc in polar_grid_cont_lines_fine:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=1)
            
        for cc in polar_grid_cont_lines:
            plt.plot(cc[:,1], 
                     cc[:,0], 'k-', lw=2, zorder=100)
        plt.colorbar(cmap, fraction=0.05, pad=0.02, aspect=4)
        plt.grid('off')
        plt.axis('off')
        fig.savefig(os.path.join(plotsavefolder, 'Epi - coarse - %s - change_strain_%s_fraction_per_h.svg' %(analyse_stages[ii], plotkey)), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        
        
#        polar_grid_areas_ve = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                  area_ve_nosmooth, 
#                                                  polar_grid_region.shape, alpha=0)
#    
#    
#        fig, ax = plt.subplots(figsize=(10,10))
#        plt.title('VE - %s - $\Delta$Area/h' %(analyse_stages[ii]))
#        cmap = ax.imshow(polar_grid_areas_ve, vmin=-max_plot_range_coarse, vmax=max_plot_range_coarse, cmap='coolwarm_r')
#        
#
##        for cc in polar_grid_cont_lines_fine:
##            plt.plot(cc[:,1], 
##                     cc[:,0], 'k-', lw=1)
#            
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=2, zorder=100)
#            
#            
#        plt.colorbar(cmap, fraction=0.05, pad=0.02, aspect=4)
#        plt.grid('off')
#        plt.axis('off')
#        fig.savefig(os.path.join(plotsavefolder, 'VE - coarse_raw - %s - change_Area_fraction_per_h.svg' %(analyse_stages[ii])), dpi=300, bbox_inches='tight')
#        
#        plt.show()
#        
#        
#        polar_grid_areas_epi = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                  area_epi_nosmooth, 
#                                                  polar_grid_region.shape, alpha=0)
#    
#        fig, ax = plt.subplots(figsize=(10,10))
#        plt.title('Epi - %s - $\Delta$Area/h' %(analyse_stages[ii]))
#        cmap = ax.imshow(polar_grid_areas_epi, vmin=-max_plot_range_coarse, vmax=max_plot_range_coarse, cmap='coolwarm_r')
##        for cc in polar_grid_cont_lines_fine:
##            plt.plot(cc[:,1], 
##                     cc[:,0], 'k-', lw=1)
#            
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=2, zorder=100)
#        plt.colorbar(cmap, fraction=0.05, pad=0.02, aspect=4)
#        plt.grid('off')
#        plt.axis('off')
#        fig.savefig(os.path.join(plotsavefolder, 'Epi - coarse_raw - %s - change_Area_fraction_per_h.svg' %(analyse_stages[ii])), dpi=300, bbox_inches='tight')
#        
#        plt.show()
#        
#        
#        
#        """
#        Mean change rate from state 0 
#        """
#        polar_grid_areas_ve = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                  area_ve_0, 
#                                                  polar_grid_region.shape, alpha=0)
#        
#        fig, ax = plt.subplots(figsize=(10,10))
#        plt.title('VE - %s - $\Delta$Area/h - TP0' %(analyse_stages[ii]))
#        cmap = ax.imshow(polar_grid_areas_ve, vmin=-max_plot_range_0_coarse, vmax=max_plot_range_0_coarse, cmap='coolwarm_r')
#        
#
##        for cc in polar_grid_cont_lines_fine:
##            plt.plot(cc[:,1], 
##                     cc[:,0], 'k-', lw=1, alpha=0.2, zorder=50)
#            
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=2, zorder=100)
#            
#            
#        plt.colorbar(cmap, fraction=0.05, pad=0.02, aspect=4)
#        plt.grid('off')
#        plt.axis('off')
#        fig.savefig(os.path.join(plotsavefolder, 'VE - coarse_tp0 - %s - change_Area_fraction_per_h.svg' %(analyse_stages[ii])), dpi=300, bbox_inches='tight')
#        
#        plt.show()
#        
#        
#        polar_grid_areas_epi = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                  area_epi_0, 
#                                                  polar_grid_region.shape, alpha=0)
#    
#    
#        fig, ax = plt.subplots(figsize=(10,10))
#        plt.title('Epi - %s - $\Delta$Area/h - TP0' %(analyse_stages[ii]))
#        cmap = ax.imshow(polar_grid_areas_epi, vmin=-max_plot_range_0_coarse, vmax=max_plot_range_0_coarse, cmap='coolwarm_r')
##        for cc in polar_grid_cont_lines_fine:
##            plt.plot(cc[:,1], 
##                     cc[:,0], 'k-', lw=1)
#            
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=2, zorder=100)
#        plt.colorbar(cmap, fraction=0.05, pad=0.02, aspect=4)
#        plt.grid('off')
#        plt.axis('off')
#        fig.savefig(os.path.join(plotsavefolder, 'Epi - coarse_tp0 - %s - change_Area_fraction_per_h.svg' %(analyse_stages[ii])), dpi=300, bbox_inches='tight')
#        
#        plt.show()
#    
#    
    
    
#    """
#    Individual plots of statistics.
#    """
#    max_total_speed = np.max(data_stages[1][0][0])
#    max_mean_lin_speed = np.max(data_stages[1][1][0]) ###### lower bounded by 0. 
#    max_mean_curve_speed = np.max(data_stages[1][2][0])
#    max_mean_anterior_speed = np.max(np.abs(data_stages[1][3][0]))
#    
#    max_length_arrow = 45.
#    length_ve = np.sqrt(data_stages[1][4][0][0]**2 + data_stages[1][4][0][1]**2)
#    scale_arrow = max_length_arrow / (float(length_ve.max()))
#    
#    
#    for iii in range(len(analyse_stages)):
#        
#        total_speed_ve, total_speed_epi = data_stages[iii][0]
#        (x_disps_ve, y_disps_ve), (x_disps_epi, y_disps_epi) = data_stages[iii][4]
#        
#        polar_grid_lin_mean_speeds_ve = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                            total_speed_ve, 
#                                                            polar_grid_region.shape, alpha=0)
#        polar_grid_lin_mean_speeds_epi = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                             total_speed_epi, 
#                                                             polar_grid_region.shape, alpha=0)
#    
#        plt.figure(figsize=(10,10))
#        plt.title('VE - Total_Speed - %s' %(analyse_stages[iii]))
#        plt.imshow(polar_grid_lin_mean_speeds_ve, cmap='coolwarm', vmin = -max_total_speed, vmax=max_total_speed)
#        
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=3)
#        
#        for ii in range(len(x_disps_ve)):
#    #        plt.plot(reg_centroids[ii,1],
#    #                 reg_centroids[ii,0], 'k.')
#            plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#                                 reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#                                xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#                                          reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#                                          arrowprops={'arrowstyle': '-|>', 'lw': 3, 'ec': 'k'}, va='center', ha='center')
#    #        plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#    #                             reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#    #                            xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#    #                                      reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#    #                                      arrowprops={'width':.3,'headlength':.5, 'headwidth':.5 , 'lw': 2, 'ec': 'k'}, va='center', ha='center')
#        plt.savefig(os.path.join(plotsavefolder, 'total_speed_VE_radial_polar_%s.svg' %(analyse_stages[iii])), bbox_inches='tight')
#        plt.show()
#        
#        
#        plt.figure(figsize=(10,10))
#        plt.title('Epi - Total_Speed - %s' %(analyse_stages[iii]))
#        plt.imshow(polar_grid_lin_mean_speeds_epi, cmap='coolwarm', vmin = -max_total_speed, vmax=max_total_speed)
#        
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=3)
#        
#        for ii in range(len(x_disps_ve)):
#    #        plt.plot(reg_centroids[ii,1],
#    #                 reg_centroids[ii,0], 'k.')
#            plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_epi[ii],
#                                 reg_centroids[ii,0]-.5*scale_arrow*y_disps_epi[ii]), 
#                                xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_epi[ii],
#                                          reg_centroids[ii,0]+.5*scale_arrow*y_disps_epi[ii]),
#                                          arrowprops={'arrowstyle': '-|>', 'lw': 3, 'ec': 'k'}, va='center', ha='center')
#    #        plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#    #                             reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#    #                            xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#    #                                      reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#    #                                      arrowprops={'width':.3,'headlength':.5, 'headwidth':.5 , 'lw': 2, 'ec': 'k'}, va='center', ha='center')
#        plt.savefig(os.path.join(plotsavefolder, 'total_speed_Epi_radial_polar_%s.svg' %(analyse_stages[iii])), bbox_inches='tight')    
#        plt.show()
#    
#    
## =============================================================================
##     Plot Mean Speed
## =============================================================================
#    for iii in range(len(analyse_stages)):
#        
#        total_speed_ve, total_speed_epi = data_stages[iii][1]
#        (x_disps_ve, y_disps_ve), (x_disps_epi, y_disps_epi) = data_stages[iii][4]
#        
#        polar_grid_lin_mean_speeds_ve = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                            total_speed_ve, 
#                                                            polar_grid_region.shape, alpha=0)
#        polar_grid_lin_mean_speeds_epi = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                             total_speed_epi, 
#                                                             polar_grid_region.shape, alpha=0)
#    
#        plt.figure(figsize=(10,10))
#        plt.title('VE - Mean_Linear_Speed - %s' %(analyse_stages[iii]))
#        plt.imshow(polar_grid_lin_mean_speeds_ve, cmap='coolwarm', vmin = -max_mean_lin_speed, vmax=max_mean_lin_speed)
#        
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=3)
#        
#        for ii in range(len(x_disps_ve)):
#    #        plt.plot(reg_centroids[ii,1],
#    #                 reg_centroids[ii,0], 'k.')
#            plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#                                 reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#                                xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#                                          reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#                                          arrowprops={'arrowstyle': '-|>', 'lw': 3, 'ec': 'k'}, va='center', ha='center')
#    #        plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#    #                             reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#    #                            xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#    #                                      reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#    #                                      arrowprops={'width':.3,'headlength':.5, 'headwidth':.5 , 'lw': 2, 'ec': 'k'}, va='center', ha='center')
#        plt.savefig(os.path.join(plotsavefolder, 'mean_lin_speed_VE_radial_polar_%s.svg' %(analyse_stages[iii])), bbox_inches='tight')        
#        plt.show()
#        
#        
#        plt.figure(figsize=(10,10))
#        plt.title('Epi - Mean_Linear_Speed - %s' %(analyse_stages[iii]))
#        plt.imshow(polar_grid_lin_mean_speeds_epi, cmap='coolwarm', vmin = -max_mean_lin_speed, vmax=max_mean_lin_speed)
#        
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=3)
#        
#        for ii in range(len(x_disps_ve)):
#    #        plt.plot(reg_centroids[ii,1],
#    #                 reg_centroids[ii,0], 'k.')
#            plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_epi[ii],
#                                 reg_centroids[ii,0]-.5*scale_arrow*y_disps_epi[ii]), 
#                                xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_epi[ii],
#                                          reg_centroids[ii,0]+.5*scale_arrow*y_disps_epi[ii]),
#                                          arrowprops={'arrowstyle': '-|>', 'lw': 3, 'ec': 'k'}, va='center', ha='center')
#    #        plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#    #                             reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#    #                            xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#    #                                      reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#    #                                      arrowprops={'width':.3,'headlength':.5, 'headwidth':.5 , 'lw': 2, 'ec': 'k'}, va='center', ha='center')
#        plt.savefig(os.path.join(plotsavefolder, 'mean_lin_speed_Epi_radial_polar_%s.svg' %(analyse_stages[iii])), bbox_inches='tight')            
#        plt.show()
#        
#        
## =============================================================================
##     Plot Mean Speed
## =============================================================================
#    for iii in range(len(analyse_stages)):
#        
#        total_speed_ve, total_speed_epi = data_stages[iii][2]
#        (x_disps_ve, y_disps_ve), (x_disps_epi, y_disps_epi) = data_stages[iii][4]
#        
#        polar_grid_lin_mean_speeds_ve = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                            total_speed_ve, 
#                                                            polar_grid_region.shape, alpha=0)
#        polar_grid_lin_mean_speeds_epi = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                             total_speed_epi, 
#                                                             polar_grid_region.shape, alpha=0)
#    
#        plt.figure(figsize=(10,10))
#        plt.title('VE - Mean_Curvilinear_Speed - %s' %(analyse_stages[iii]))
#        plt.imshow(polar_grid_lin_mean_speeds_ve, cmap='coolwarm', vmin = -max_mean_curve_speed, vmax=max_mean_curve_speed)
#        
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=3)
#        
#        for ii in range(len(x_disps_ve)):
#    #        plt.plot(reg_centroids[ii,1],
#    #                 reg_centroids[ii,0], 'k.')
#            plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#                                 reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#                                xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#                                          reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#                                          arrowprops={'arrowstyle': '-|>', 'lw': 3, 'ec': 'k'}, va='center', ha='center')
#    #        plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#    #                             reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#    #                            xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#    #                                      reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#    #                                      arrowprops={'width':.3,'headlength':.5, 'headwidth':.5 , 'lw': 2, 'ec': 'k'}, va='center', ha='center')
#        plt.savefig(os.path.join(plotsavefolder, 'mean_curve_lin_speed_VE_radial_polar_%s.svg' %(analyse_stages[iii])), bbox_inches='tight')            
#        plt.show()
#        
#        
#        plt.figure(figsize=(10,10))
#        plt.title('Epi - Mean_Curvilinear_Speed - %s' %(analyse_stages[iii]))
#        plt.imshow(polar_grid_lin_mean_speeds_epi, cmap='coolwarm', vmin = -max_mean_curve_speed, vmax=max_mean_curve_speed)
#        
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=3)
#        
#        for ii in range(len(x_disps_ve)):
#    #        plt.plot(reg_centroids[ii,1],
#    #                 reg_centroids[ii,0], 'k.')
#            plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_epi[ii],
#                                 reg_centroids[ii,0]-.5*scale_arrow*y_disps_epi[ii]), 
#                                xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_epi[ii],
#                                          reg_centroids[ii,0]+.5*scale_arrow*y_disps_epi[ii]),
#                                          arrowprops={'arrowstyle': '-|>', 'lw': 3, 'ec': 'k'}, va='center', ha='center')
#    #        plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#    #                             reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#    #                            xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#    #                                      reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#    #                                      arrowprops={'width':.3,'headlength':.5, 'headwidth':.5 , 'lw': 2, 'ec': 'k'}, va='center', ha='center')
#        plt.savefig(os.path.join(plotsavefolder, 'mean_curve_lin_speed_Epi_radial_polar_%s.svg' %(analyse_stages[iii])), bbox_inches='tight')                
#        plt.show()
#        
#        
#        
#        
#    for iii in range(len(analyse_stages)):
#        
#        total_speed_ve, total_speed_epi = data_stages[iii][3]
#        (x_disps_ve, y_disps_ve), (x_disps_epi, y_disps_epi) = data_stages[iii][4]
#        
#        polar_grid_lin_mean_speeds_ve = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                            total_speed_ve, 
#                                                            polar_grid_region.shape, alpha=0)
#        polar_grid_lin_mean_speeds_epi = heatmap_vals_radial(polar_grid_cont_lines, 
#                                                             total_speed_epi, 
#                                                             polar_grid_region.shape, alpha=0)
#    
#        plt.figure(figsize=(10,10))
#        plt.title('VE - Mean_Anterior_Speed - %s' %(analyse_stages[iii]))
#        plt.imshow(polar_grid_lin_mean_speeds_ve, cmap='coolwarm', vmin = -max_mean_anterior_speed, vmax=max_mean_anterior_speed)
#        
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=3)
#        
#        for ii in range(len(x_disps_ve)):
#    #        plt.plot(reg_centroids[ii,1],
#    #                 reg_centroids[ii,0], 'k.')
#            plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#                                 reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#                                xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#                                          reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#                                          arrowprops={'arrowstyle': '-|>', 'lw': 3, 'ec': 'k'}, va='center', ha='center')
#    #        plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#    #                             reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#    #                            xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#    #                                      reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#    #                                      arrowprops={'width':.3,'headlength':.5, 'headwidth':.5 , 'lw': 2, 'ec': 'k'}, va='center', ha='center')
#        plt.savefig(os.path.join(plotsavefolder, 'mean_anterior_speed_VE_radial_polar_%s.svg' %(analyse_stages[iii])), bbox_inches='tight')            
#        plt.show()
#        
#        
#        plt.figure(figsize=(10,10))
#        plt.title('Epi - Mean_Anterior_Speed - %s' %(analyse_stages[iii]))
#        plt.imshow(polar_grid_lin_mean_speeds_epi, cmap='coolwarm', vmin = -max_mean_anterior_speed, vmax=max_mean_anterior_speed)
#        
#        for cc in polar_grid_cont_lines:
#            plt.plot(cc[:,1], 
#                     cc[:,0], 'k-', lw=3)
#        
#        for ii in range(len(x_disps_ve)):
#    #        plt.plot(reg_centroids[ii,1],
#    #                 reg_centroids[ii,0], 'k.')
#            plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_epi[ii],
#                                 reg_centroids[ii,0]-.5*scale_arrow*y_disps_epi[ii]), 
#                                xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_epi[ii],
#                                          reg_centroids[ii,0]+.5*scale_arrow*y_disps_epi[ii]),
#                                          arrowprops={'arrowstyle': '-|>', 'lw': 3, 'ec': 'k'}, va='center', ha='center')
#    #        plt.annotate('', xy=(reg_centroids[ii,1]+.5*scale_arrow*x_disps_ve[ii],
#    #                             reg_centroids[ii,0]-.5*scale_arrow*y_disps_ve[ii]), 
#    #                            xytext = (reg_centroids[ii,1]-.5*scale_arrow*x_disps_ve[ii],
#    #                                      reg_centroids[ii,0]+.5*scale_arrow*y_disps_ve[ii]),
#    #                                      arrowprops={'width':.3,'headlength':.5, 'headwidth':.5 , 'lw': 2, 'ec': 'k'}, va='center', ha='center')
#        
#        plt.savefig(os.path.join(plotsavefolder, 'mean_anterior_speed_Epi_radial_polar_%s.svg' %(analyse_stages[iii])), bbox_inches='tight')                
#        plt.show()
#        
        
        
#    polar_grid_region = tile_uniform_windows_radial_custom( (640,640), 
#                                                            n_r=4, 
#                                                            n_theta=8, 
#                                                            max_r=640/2./1.2, 
#                                                            mask_distal=None, # leave a set distance
#                                                            center=None, 
#                                                            bound_r=True)
#    
#    from skimage.measure import regionprops
#    
#    rotate_angle = 270 + 360/8./2.
##    from skimage.measure import regionprops
#    rotated_grid = sktform.rotate(polar_grid_region, angle=rotate_angle, preserve_range=True).astype(np.int)
#    
#    regpropgrid = regionprops(rotated_grid)
#    reg_centroids = np.vstack([re.centroid for re in regpropgrid])
#    
#    fig, ax = plt.subplots()
#    ax.imshow(rotated_grid, cmap='coolwarm')
#    ax.plot(reg_centroids[:,1], 
#            reg_centroids[:,0], 'go')
#    ax.quiver(reg_centroids[:,1], 
#              reg_centroids[:,0], 
#              np.sin(all_directions_regions_ve_relative[2]), 
#              np.cos(all_directions_regions_ve_relative[2]))
#    plt.show()
#    
#    fig, ax = plt.subplots()
#    ax.imshow(rotated_grid, cmap='coolwarm')
#    ax.plot(reg_centroids[:,1], 
#            reg_centroids[:,0], 'go')
#    ax.quiver(reg_centroids[:,1], 
#              reg_centroids[:,0], 
#              np.sin(all_directions_regions_epi_relative[2]), 
#              np.cos(all_directions_regions_epi_relative[2]))
#    plt.show()
#    
#    fig, ax = plt.subplots()
#    ax.imshow(rotated_grid, cmap='coolwarm')
#    ax.plot(reg_centroids[:,1], 
#            reg_centroids[:,0], 'go')
#    ax.quiver(reg_centroids[:,1], 
#              reg_centroids[:,0], 
#              np.sin(all_directions_regions_ve_relative[3]), 
#              np.cos(all_directions_regions_ve_relative[3]))
#    plt.show()
#    
#    
#    fig, ax = plt.subplots()
#    plt.title('VE')
#    ax.imshow(rotated_grid, cmap='coolwarm')
#    ax.plot(reg_centroids[:,1], 
#            reg_centroids[:,0], 'go')
#    ax.quiver(reg_centroids[:,1], 
#              reg_centroids[:,0], 
#              np.median(np.sin(all_directions_regions_ve_relative), axis=0), 
#              np.median(np.cos(all_directions_regions_ve_relative), axis=0))
#    plt.show()
#    
#    
#    fig, ax = plt.subplots()
#    plt.title('Epi')
#    ax.imshow(rotated_grid, cmap='coolwarm')
#    ax.plot(reg_centroids[:,1], 
#            reg_centroids[:,0], 'go')
#    ax.quiver(reg_centroids[:,1], 
#              reg_centroids[:,0], 
#              np.median(np.sin(all_directions_regions_epi_relative), axis=0), 
#              np.median(np.cos(all_directions_regions_epi_relative), axis=0))
#    plt.show()
#    
#    
#    fig, ax = plt.subplots()
#    ax.imshow(rotated_grid, cmap='coolwarm')
#    ax.plot(reg_centroids[:,1], 
#            reg_centroids[:,0], 'go')
#    ax.quiver(reg_centroids[:,1], 
#              reg_centroids[:,0], 
#              np.median(np.sin(all_directions_regions_ve_relative) * all_mean_speed_regions_ve * mean_vol_factor * voxel_size/Ts, axis=0), 
##              np.cos(np.median(all_directions_regions_epi_relative, axis=0)))
#              np.median(np.cos(all_directions_regions_ve_relative) * all_mean_speed_regions_ve * mean_vol_factor * voxel_size/Ts, axis=0)) 
#    plt.show()
#    
#    
#    fig, ax = plt.subplots()
#    ax.imshow(rotated_grid, cmap='coolwarm')
#    ax.plot(reg_centroids[:,1], 
#            reg_centroids[:,0], 'go')
#    ax.quiver(reg_centroids[:,1], 
#              reg_centroids[:,0], 
#              np.median(np.sin(all_directions_regions_epi_relative) * all_mean_speed_regions_epi * mean_vol_factor * voxel_size/Ts, axis=0), 
##              np.cos(np.median(all_directions_regions_epi_relative, axis=0)))
#              np.median(np.cos(all_directions_regions_epi_relative) * all_mean_speed_regions_epi * mean_vol_factor * voxel_size/Ts, axis=0)) 
#    plt.show()
#    
#    
#    plt.figure()
#    plt.boxplot(all_directions_regions_epi_relative)
#    plt.hlines(0, 1, 32, linestyles='dashed')
#    plt.show()
#    
#    plt.figure()
#    plt.boxplot(all_directions_regions_ve_relative)
#    plt.hlines(0, 1, 32, linestyles='dashed')
#    plt.show()
#    
    
# =============================================================================
#   Export the statistics table out. 
# =============================================================================
#    spio.savemat(savestatsfile, {'all_ve_angles':all_ve_angles,
#                                 'all_ve_angles_rad_global_ref': all_ve_angles_rad,
#                                 'all_directions_regions_ve': all_directions_regions_ve,
#                                 'all_directions_regions_epi': all_directions_regions_epi,
#                                 'all_directions_regions_ve_relative_mean': all_directions_regions_ve_relative_mean,
#                                 'all_directions_regions_epi_relative_mean': all_directions_regions_epi_relative_mean,
#                                 'all_analysed_embryos':all_analysed_embryos,
#                                 'all_speeds_regions_ve':all_speeds_regions_ve,
#                                 'all_speeds_regions_epi':all_speeds_regions_epi,
#                                 'all_speeds_regions_ve_proj_ve':all_speeds_regions_ve_proj_ve,
#                                 'all_speeds_regions_epi_proj_ve':all_speeds_regions_epi_proj_ve,
#                                 'all_total_speed_regions_ve': all_total_speed_regions_ve,
#                                 'all_total_speed_regions_epi': all_total_speed_regions_epi,
#                                 'all_embryo_shapes':all_embryo_shapes,
#                                 'all_embryo_vol_factors':all_embryo_vol_factors,
#                                 'all_grids':all_grids
#                                 })
    


## =============================================================================
##     Compile all the statistics and save out in the master folder for analysis. 
## =============================================================================
#    all_ve_angles = np.hstack(all_ve_angles)
#    all_analysed_embryos = np.hstack(all_analysed_embryos)
#    all_speeds_regions_ve = np.vstack(all_speeds_regions_ve)
#    all_speeds_regions_epi = np.vstack(all_speeds_regions_epi)
#    all_speeds_regions_ve_proj_ve = np.vstack(all_speeds_regions_ve_proj_ve)
#    all_speeds_regions_epi_proj_ve = np.vstack(all_speeds_regions_epi_proj_ve)
#    all_total_speed_regions_ve = np.vstack(all_total_speed_regions_ve)
#    all_total_speed_regions_epi = np.vstack(all_total_speed_regions_epi)
#    
#    all_embryo_shapes = np.vstack(all_embryo_shapes)
#    
#    all_directions_regions_ve = np.vstack(all_directions_regions_ve)
#    all_directions_regions_epi = np.vstack(all_directions_regions_epi)
#    
#    """
#    conversion to AVE relative angles. 
#    """
#    all_ve_angles_rad = (all_ve_angles + 90)/ 180 * np.pi    
#    all_directions_regions_ve_relative = all_directions_regions_ve - all_ve_angles_rad[:,None]
#    all_directions_regions_epi_relative = all_directions_regions_epi - all_ve_angles_rad[:,None]
#
#
#    # add the components to get a mean:
#    all_directions_regions_ve_relative_mean = [np.mean( all_speeds_regions_ve * np.sin(all_directions_regions_ve_relative), axis=0), 
#                                               np.mean( all_speeds_regions_ve * np.cos(all_directions_regions_ve_relative), axis=0)]
##    all_directions_regions_ve_relative_mean_ang = np.arctan2(all_directions_regions_ve_relative_mean[1], 
##                                                             all_directions_regions_ve_relative_mean[0])
#    all_directions_regions_epi_relative_mean = [np.mean( all_speeds_regions_epi * np.sin(all_directions_regions_epi_relative), axis=0), 
#                                                np.mean( all_speeds_regions_epi * np.cos(all_directions_regions_epi_relative), axis=0)]
#    
#    # save all the statistics 
#    savestatsfile = os.path.join(master_analysis_save_folder, '%s_speed_statistics.mat' %(analyse_stage))
#    spio.savemat(savestatsfile, {'all_ve_angles':all_ve_angles,
#                                 'all_ve_angles_rad_global_ref': all_ve_angles_rad,
#                                 'all_directions_regions_ve': all_directions_regions_ve,
#                                 'all_directions_regions_epi': all_directions_regions_epi,
#                                 'all_directions_regions_ve_relative_mean': all_directions_regions_ve_relative_mean,
#                                 'all_directions_regions_epi_relative_mean': all_directions_regions_epi_relative_mean,
#                                 'all_analysed_embryos':all_analysed_embryos,
#                                 'all_speeds_regions_ve':all_speeds_regions_ve,
#                                 'all_speeds_regions_epi':all_speeds_regions_epi,
#                                 'all_speeds_regions_ve_proj_ve':all_speeds_regions_ve_proj_ve,
#                                 'all_speeds_regions_epi_proj_ve':all_speeds_regions_epi_proj_ve,
#                                 'all_total_speed_regions_ve': all_total_speed_regions_ve,
#                                 'all_total_speed_regions_epi': all_total_speed_regions_epi,
#                                 'all_embryo_shapes':all_embryo_shapes,
#                                 'all_embryo_vol_factors':all_embryo_vol_factors,
#                                 'all_grids':all_grids
#                                 })
#    
#    print(all_speeds_regions_ve.shape)
    
    
    
## =============================================================================
##     Heatmap Analysis of the regions. 
## =============================================================================
#    polar_grid_region = tile_uniform_windows_radial_custom( (640,640), 
#                                                            n_r=4, 
#                                                            n_theta=8, 
#                                                            max_r=640/2./1.2, 
#                                                            mask_distal=None, # leave a set distance
#                                                            center=None, 
#                                                            bound_r=True)
#
#    av_speed_map_ve_region = np.zeros(polar_grid_region.shape, dtype=np.float32)
#    av_speed_map_epi_region = np.zeros(polar_grid_region.shape, dtype=np.float32)
#    av_ve_speed_component_map_polar_region = np.zeros(polar_grid_region.shape, dtype=np.float32)
#    av_epi_speed_component_map_polar_region = np.zeros(polar_grid_region.shape, dtype=np.float32)
#    
#    for kk in range(len(uniq_polar_regions)):
#        
##        av_speed_map_ve_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (all_speeds_regions_ve[::].mean(axis=0))[kk] 
##        av_speed_map_epi_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (all_speeds_regions_epi[::].mean(axis=0))[kk] 
##        
##        av_ve_speed_component_map_polar_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (all_speeds_regions_ve_proj_ve[::].mean(axis=0))[kk] 
##        av_epi_speed_component_map_polar_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (all_speeds_regions_epi_proj_ve[::].mean(axis=0))[kk] 
#
##        av_speed_map_ve_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (all_speeds_regions_ve[::].median(axis=0))[kk] 
##        av_speed_map_epi_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (all_speeds_regions_epi[::].median(axis=0))[kk] 
##        
##        av_ve_speed_component_map_polar_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (all_speeds_regions_ve_proj_ve[::].median(axis=0))[kk] 
##        av_epi_speed_component_map_polar_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (all_speeds_regions_epi_proj_ve[::].median(axis=0))[kk] 
#
#        av_speed_map_ve_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = np.mean(all_speeds_regions_ve[::], axis=0)[kk] 
#        av_speed_map_epi_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = np.mean(all_speeds_regions_epi[::], axis=0)[kk] 
#        
#        av_ve_speed_component_map_polar_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = np.mean(all_speeds_regions_ve_proj_ve, axis=0)[kk] 
#        av_epi_speed_component_map_polar_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = np.mean(all_speeds_regions_epi_proj_ve, axis=0)[kk] 
#
##    max_proj_speed = [np.max(np.abs(av_ve_speed_component_map_polar_region)), 
##                      np.max(np.abs(av_epi_speed_component_map_polar_region))]
##    speed_ranges_proj_ve = [-np.max(max_proj_speed),
##                            np.max(max_proj_speed)]
##    
##    speed_ranges_ve = [0,
##                       np.max([np.max(av_speed_map_ve_region), np.max(av_speed_map_epi_region)])]
#    
#    speed_ranges_proj_ve = [-0.0035, 0.0035]
#    speed_ranges_ve = [0, 0.0035]
#    
#    # choose a heatmap scale and visualize all of these (rotated) 
#    rotate_angle = 270 + 360/8./2.
#    
#    from skimage.measure import regionprops
#    rotated_grid = sktform.rotate(polar_grid_region, angle=rotate_angle, preserve_range=True).astype(np.int)
#    
#    regpropgrid = regionprops(rotated_grid)
#    reg_centroids = np.vstack([re.centroid for re in regpropgrid])
#    
#    
#    fig, ax = plt.subplots(figsize=(10,10))
#    plt.suptitle('Analysis Grid - %s' %(analyse_stage))
##    plt.hlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.vlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
#    plt.quiver(reg_centroids[:,1], reg_centroids[:,0],
#               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[0], 
#               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[1], color='g', scale=.0075, units='xy')
##    plt.quiver(reg_centroids[:,1], reg_centroids[:,0], 
##               50*np.sin(all_directions_regions_ve_relative_mean[0]), -50*np.cos(all_directions_regions_ve_relative_mean[1]))
##    for jj in range(len(reg_centroids)):
##        plt.text(reg_centroids[jj,1], 
##                 reg_centroids[jj,0],
##                 str(jj+1), horizontalalignment='center',
##                 verticalalignment='center', fontsize= 18, color='k', fontname='Liberation Sans')
#    ax.imshow(sktform.rotate(polar_grid_region, angle=rotate_angle, preserve_range=True).astype(np.int), cmap='coolwarm' )
#    plt.axis('off')
#    fig.savefig(os.path.join(master_analysis_save_folder, 'polar_grid_map_anlysis-with_vectors.svg'), dpi=300, bbox_inches='tight')
#    
#    
#    
#    fig, ax = plt.subplots(figsize=(10,10))
#    plt.suptitle('VE Speed - %s' %(analyse_stage))
##    plt.hlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.vlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.quiver(reg_centroids[:,1], reg_centroids[:,0],
##               50*all_directions_regions_ve_relative_mean[0], 50*all_directions_regions_ve_relative_mean[1], color='g')
#    plt.quiver(reg_centroids[:,1], reg_centroids[:,0],
#               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[0], 
#               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[1], color='g', scale=.0075, units='xy')
#
##    for jj in range(len(reg_centroids)):
##        plt.text(reg_centroids[jj,1], 
##                 reg_centroids[jj,0],
##                 str(jj+1), horizontalalignment='center',
##                 verticalalignment='center', fontsize= 18, color='k', fontname='Liberation Sans')
#    ax.imshow(sktform.rotate(av_speed_map_ve_region, angle=rotate_angle), cmap='coolwarm', vmin=speed_ranges_ve[0], vmax=speed_ranges_ve[1])
#    plt.axis('off')
#    fig.savefig(os.path.join(master_analysis_save_folder, 'speed_map_VE_anlysis-with_vectors.svg'), dpi=300, bbox_inches='tight')
#    
#    
#    fig, ax = plt.subplots(figsize=(10,10))
#    plt.suptitle('Epi Speed - %s' %(analyse_stage))
##    plt.hlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.vlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
#    plt.quiver(reg_centroids[:,1], reg_centroids[:,0],
#               np.mean(all_embryo_vol_factors)*all_directions_regions_epi_relative_mean[0], 
#               np.mean(all_embryo_vol_factors)*all_directions_regions_epi_relative_mean[1], color='g', scale=.0075, units='xy')
#
##    for jj in range(len(reg_centroids)):
##        plt.text(reg_centroids[jj,1], 
##                 reg_centroids[jj,0],
##                 str(jj+1), horizontalalignment='center',
##                 verticalalignment='center', fontsize= 18, color='k', fontname='Liberation Sans')
#    ax.imshow(sktform.rotate(av_speed_map_epi_region, angle=rotate_angle), cmap='coolwarm',vmin=speed_ranges_ve[0], vmax=speed_ranges_ve[1])
#    plt.axis('off')
#    fig.savefig(os.path.join(master_analysis_save_folder, 'speed_map_Epi_anlysis-with_vectors.svg'), dpi=300, bbox_inches='tight')
#    
#    
#    fig, ax = plt.subplots(figsize=(10,10))
#    plt.suptitle('VE Directional Velocity - %s' %(analyse_stage))
##    plt.hlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.vlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    for jj in range(len(reg_centroids)):
##        plt.text(reg_centroids[jj,1], 
##                 reg_centroids[jj,0],
##                 str(jj+1), horizontalalignment='center',
##                 verticalalignment='center', fontsize= 18, color='k', fontname='Liberation Sans')
#    plt.quiver(reg_centroids[:,1], reg_centroids[:,0],
#               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[0], 
#               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[1], color='g', scale=.0075, units='xy')
#    ax.imshow(sktform.rotate(av_ve_speed_component_map_polar_region, angle=rotate_angle), cmap='coolwarm', vmin=speed_ranges_proj_ve[0], vmax=speed_ranges_proj_ve[1] )
#    plt.axis('off')
#    fig.savefig(os.path.join(master_analysis_save_folder, 'speed_map_VE-directionality_anlysis-with_vectors.svg'), dpi=300, bbox_inches='tight')
#    
#    
#    fig, ax = plt.subplots(figsize=(10,10))
#    plt.suptitle('Epi Directional Velocity - %s' %(analyse_stage))
##    plt.hlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.vlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    for jj in range(len(reg_centroids)):
##        plt.text(reg_centroids[jj,1], 
##                 reg_centroids[jj,0],
##                 str(jj+1), horizontalalignment='center',
##                 verticalalignment='center', fontsize= 18, color='k', fontname='Liberation Sans')
#    plt.quiver(reg_centroids[:,1], reg_centroids[:,0],
#           np.mean(all_embryo_vol_factors)*all_directions_regions_epi_relative_mean[0], 
#           np.mean(all_embryo_vol_factors)*all_directions_regions_epi_relative_mean[1], color='g', scale=.0075, units='xy')
#    ax.imshow(sktform.rotate(av_epi_speed_component_map_polar_region, angle=rotate_angle), cmap='coolwarm', vmin=speed_ranges_proj_ve[0], vmax=speed_ranges_proj_ve[1])
#    plt.axis('off')
#    fig.savefig(os.path.join(master_analysis_save_folder, 'speed_map_Epi-directionality_anlysis-with_vectors.svg'), dpi=300, bbox_inches='tight')
#    
#                                
                                
                                
                                
                                
                                
                                
## =============================================================================
##     arrange the boxplot into concentric rings based on angle. and distance
## =============================================================================
#    from scipy.stats import sem
#    
#    n_r = 4
#    n_ang = 8
#
#    fig, ax = plt.subplots(nrows=n_r, figsize=(5,10))
#    
#    for rr in range(n_r):
#        
#        start_id = rr*n_ang
#        end_id = (rr+1)*n_ang
#    
##        ax[rr].boxplot([all_speeds_regions_ve_proj_ve[::2,jj] for jj in range(start_id, end_id)])
#        ax[rr].bar(np.arange(n_ang)+0.375, [all_speeds_regions_ve_proj_ve[::2,jj].mean(axis=0) for jj in range(start_id, end_id)], width=0.25, color='g')
#        ax[rr].bar(np.arange(n_ang)+0.625, [all_speeds_regions_epi_proj_ve[::2,jj].mean(axis=0) for jj in range(start_id, end_id)], width=0.25, color='r')
#        ax[rr].errorbar(np.arange(n_ang)+0.375, [all_speeds_regions_ve_proj_ve[::2,jj].mean(axis=0) for jj in range(start_id, end_id)], 
#                          yerr=[sem(all_speeds_regions_ve_proj_ve[::2,jj]) for jj in range(start_id, end_id)], color='g', fmt='none', elinewidth=1, capsize=3, ecolor='k')
#        ax[rr].errorbar(np.arange(n_ang)+0.625, [all_speeds_regions_epi_proj_ve[::2,jj].mean(axis=0) for jj in range(start_id, end_id)], 
#                          yerr=[sem(all_speeds_regions_epi_proj_ve[::2,jj]) for jj in range(start_id, end_id)], color='r', fmt='none', elinewidth=1, capsize=3, ecolor='k')
#        ax[rr].hlines(0, 0, n_ang)
#        ax[rr].set_ylim([1.5*speed_ranges_proj_ve[0], 1.5*speed_ranges_proj_ve[1]])
#        ax[rr].set_xlim([0, n_ang])
#        
#        ax[rr].set_ylabel('Normalised VE Velocity')
#        ax[rr].set_xticks(np.arange(n_ang)+0.5)
#        ax[rr].set_xticklabels(np.arange(start_id+1,end_id+1))
#    fig.savefig(os.path.join(master_analysis_save_folder, 'barplot_analysis_VE-Epi_directional-VE_component.svg'), dpi=300, bbox_inches='tight')
#    plt.show()
#    
#    
#    fig, ax = plt.subplots(nrows=n_r, figsize=(5,10))
#    
#    for rr in range(n_r):
#        
#        start_id = rr*n_ang
#        end_id = (rr+1)*n_ang
#    
##        ax[rr].boxplot([all_speeds_regions_ve_proj_ve[::2,jj] for jj in range(start_id, end_id)])
#        ax[rr].bar(np.arange(n_ang)+0.375, [all_speeds_regions_ve[::2,jj].mean(axis=0) for jj in range(start_id, end_id)], width=0.25, color='g')
#        ax[rr].bar(np.arange(n_ang)+0.625, [all_speeds_regions_epi[::2,jj].mean(axis=0) for jj in range(start_id, end_id)], width=0.25, color='r')
#        ax[rr].errorbar(np.arange(n_ang)+0.375, [all_speeds_regions_ve[::2,jj].mean(axis=0) for jj in range(start_id, end_id)], 
#                          yerr=[sem(all_speeds_regions_ve[::2,jj]) for jj in range(start_id, end_id)], color='g', fmt='none', elinewidth=1, capsize=3, ecolor='k')
#        ax[rr].errorbar(np.arange(n_ang)+0.625, [all_speeds_regions_epi[::2,jj].mean(axis=0) for jj in range(start_id, end_id)], 
#                          yerr=[sem(all_speeds_regions_epi[::2,jj]) for jj in range(start_id, end_id)], color='r', fmt='none', elinewidth=1, capsize=3, ecolor='k')
#        ax[rr].hlines(0, 0, n_ang)
#        ax[rr].set_ylim([1.5*speed_ranges_ve[0], 1.5*speed_ranges_ve[1]])
#        ax[rr].set_xlim([0, n_ang])
#        
#        ax[rr].set_ylabel('Normalised Speed')
#        ax[rr].set_xticks(np.arange(n_ang)+0.5)
#        ax[rr].set_xticklabels(np.arange(start_id+1,end_id+1))
#    fig.savefig(os.path.join(master_analysis_save_folder, 'barplot_analysis_VE-Epi_speed.svg'), dpi=300, bbox_inches='tight')
#    plt.show()
#    
#    
#    fig, ax = plt.subplots(figsize=(5,5))
#    ax.plot(all_speeds_regions_ve.ravel(), all_speeds_regions_epi.ravel(), 'o')
#    ax.set_aspect('equal')
#    plt.show()
#    
#    
#    fig, ax = plt.subplots(figsize=(5,5))
#    ax.plot(all_speeds_regions_ve_proj_ve.ravel(), all_speeds_regions_epi_proj_ve.ravel(), 'o')
#    ax.set_aspect('equal')
#    plt.show()
#    
                        
                        
                        
                        
                        
                        
#    from scipy.stats import pearsonr
#    
#    print(pearsonr(all_speeds_regions_ve.ravel(), all_speeds_regions_epi.ravel()))
#    print(pearsonr(all_speeds_regions_ve_proj_ve.ravel(), all_speeds_regions_epi_proj_ve.ravel()))
#    
    
#    fig, ax = plt.subplots(figsize=(15,5))
#    ax.boxplot([all_speeds_regions_epi_proj_ve[::2,jj] for jj in range(all_speeds_regions_epi_proj_ve.shape[1])])
#    plt.hlines(0, 0, all_speeds_regions_epi_proj_ve.shape[1]+1)
#    plt.show()
#    
#    fig, ax = plt.subplots(figsize=(15,5))
#    ax.boxplot([all_speeds_regions_ve[:,jj] for jj in range(all_speeds_regions_ve.shape[1])])
#    plt.hlines(0, 0, all_speeds_regions_ve.shape[1]+1)
#    plt.show()
#    
#    fig, ax = plt.subplots(figsize=(15,5))
#    ax.boxplot([all_speeds_regions_epi[:,jj] for jj in range(all_speeds_regions_ve.shape[1])])
#    plt.hlines(0, 0, all_speeds_regions_ve.shape[1]+1)
#    plt.show()
    
    
    
    
    
#            # =============================================================================
#            #       Load the relevant VE segmentation from trained classifier    
#            # =============================================================================
#                ve_seg_params_file = os.path.join(savevesegfolder, os.path.split(trackfile)[-1].replace('rect','polar').replace('.mat', '-ve_aligned.mat'))
#                ve_seg_params_obj = spio.loadmat(ve_seg_params_file)                             
#                
##                ve_seg_select_rect = ve_seg_params_obj['ve_rect_select'].ravel() > 0
##                ve_seg_select_rect_valid = ve_seg_params_obj['ve_rect_select_valid'].ravel() > 0
#                
#                ve_seg_select_rect_valid = ve_seg_params_obj['density_rect_pts']
#                ve_seg_select_rect_valid = ve_seg_select_rect_valid >= np.mean(ve_seg_select_rect_valid) + 1 * np.std(ve_seg_select_rect_valid)
#                ve_seg_select_rect = ve_seg_select_rect_valid[meantracks_ve[:,0,0], 
#                                                              meantracks_ve[:,0,1]]
##                # figure out the examples to correlate. 
##                fig, ax = plt.subplots(figsize=(15,15))
##                ax.imshow(vid_ve[0])
##                plot_tracks(meantracks_ve[:,:80], ax=ax, color='r')
##                ax.plot(meantracks_ve[:,0,1],
##                        meantracks_ve[:,0,0], 'w.')
##                for jj in  range(len(meantracks_ve)):
##                    ax.text(meantracks_ve[jj,0,1],
##                            meantracks_ve[jj,0,0], str(jj))
##                plt.show()                
#                meantracks_ve3D = unwrap_params_ve[meantracks_ve[...,0], meantracks_ve[...,1]]
#                meantracks_epi3D = unwrap_params_epi[meantracks_epi[...,0], meantracks_epi[...,1]]
#
##                print(meantracks_ve3D.shape)
#                meantracks_ve_smooth = meantracks_ve3D[:,:].copy()
#                meantracks_epi_smooth = meantracks_epi3D[:,:].copy()
##                meantracks_ve_smooth = meantracks_ve.copy()
##                meantracks_epi_smooth = meantracks_epi.copy()
#                
#                # we should choose this based on the autocorrelation 
##                meantracks_ve_smooth = smooth_track_set(meantracks_ve3D, win_size = smoothwinsize, polyorder = 1)
##                meantracks_epi_smooth = smooth_track_set(meantracks_epi3D, win_size = smoothwinsize, polyorder = 1)
##                meantracks_ve_smooth = smooth_track_set(meantracks_ve, win_size = 3, polyorder = 1)
##                meantracks_epi_smooth = smooth_track_set(meantracks_epi, win_size = 3, polyorder = 1)
#                
#                meantracks_ve_smooth_diff = (meantracks_ve_smooth[:,1:] - meantracks_ve_smooth[:,:-1]).astype(np.float32)
##                meantracks_epi_smooth_diff = (meantracks_epi_smooth[:,1:] - meantracks_epi_smooth[:,:-1]).astype(np.float32)
#                
#                meantracks_ve_smooth_diff[...,0] = meantracks_ve_smooth_diff[...,0]/embryo_im_shape[0]
#                meantracks_ve_smooth_diff[...,1] = meantracks_ve_smooth_diff[...,1]/embryo_im_shape[1]
#                meantracks_ve_smooth_diff[...,2] = meantracks_ve_smooth_diff[...,2]/embryo_im_shape[2]
#                """
#                plot to check the segmentation 
#                """
##                fig, ax = plt.subplots(figsize=(15,15))
##                ax.imshow(vid_ve[0], cmap='gray')
###                ax.plot(meantracks_ve[:,0,1], meantracks_ve[:,0,0], 'wo')
##                ax.plot(meantracks_ve[ve_seg_select_rect_valid][ve_seg_select_rect,0,1], 
##                        meantracks_ve[ve_seg_select_rect_valid][ve_seg_select_rect>0,0,0], 'go')
##                
###                plot_tracks(meantracks_ve[:,:], ax=ax, color='g')
###                plot_tracks(meantracks_epi, ax=ax, color='r')
###                ax.scatter(meantracks_ve[:,0,1],
###                           meantracks_ve[:,0,0], 
###                           c=cross_corr_lags3D, cmap='coolwarm', vmin=-len(vid_ve), vmax=len(vid_ve))
##                plt.show()
#                
#                """
#                Verify the projection. 
#                """
#                fig, ax = plt.subplots(figsize=(15,15))
#                ax.imshow(vid_ve[0], cmap='gray')
#                plot_tracks(meantracks_ve[:,:], ax=ax, color='g')
#                plt.show()
#                
#                
#                """
#                Get the valid region of polar projection  
#                """
#                YY,XX = np.meshgrid(range(vid_ve[0].shape[1]), 
#                                    range(vid_ve[0].shape[0]))
#                dist_centre = np.sqrt((YY-vid_ve[0].shape[0]/2.)**2 + (XX-vid_ve[0].shape[1]/2.)**2)
#                valid_region = dist_centre <= vid_ve[0].shape[1]/2./1.2
#                
#                
#                """
#                Get the predicted VE region. 
#                """
#                all_reg = np.arange(len(meantracks_ve_smooth_diff))
#                all_reg_valid = valid_region[meantracks_ve[:,0,0], 
#                                             meantracks_ve[:,0,1]]
#                
#                ve_reg_id = all_reg[np.logical_and(ve_seg_select_rect, all_reg_valid)]
#                neg_reg_id = all_reg[np.logical_and(np.logical_not(ve_seg_select_rect), all_reg_valid)]
#                
#                fig, ax = plt.subplots()
#                ax.imshow(vid_ve[0],cmap='gray')
#                ax.plot(meantracks_ve[:,0,1], meantracks_ve[:,0,0], 'wo')
#                ax.plot(meantracks_ve[neg_reg_id,0,1], meantracks_ve[neg_reg_id,0,0], 'ro')
#                ax.plot(meantracks_ve[ve_reg_id,0,1], meantracks_ve[ve_reg_id,0,0], 'go')
#                plt.show()
#                
#                ve_diff_select = meantracks_ve_smooth_diff[ve_reg_id]
#                ve_diff_mean = np.mean(ve_diff_select, axis=1) 
#                ve_diff_mean = ve_diff_mean/(np.linalg.norm(ve_diff_mean, axis=-1)[:,None] + 1e-8) 
#                ve_component = np.vstack([ve_diff_select[ll].dot(ve_diff_mean[ll]) for ll in range(len(ve_diff_select))])
#                
#
#                all_speed_neg = meantracks_ve_smooth_diff[neg_reg_id]
#                all_speed_neg_mean = all_speed_neg.mean(axis=1) 
#                all_speed_neg_mean = all_speed_neg_mean/(np.linalg.norm(all_speed_neg_mean, axis=-1)[:,None] + 1e-8) 
#                all_speed_component = np.vstack([all_speed_neg[ll].dot(all_speed_neg_mean[ll]) for ll in range(len(all_speed_neg))])
#
#                
##                plt.figure()
##                plt.plot(np.cumsum(ve_component.mean(axis=0)))
##                plt.plot(np.cumsum(all_speed_component.mean(axis=0)))
##                plt.show()
#                
#
#                diff_curve_component = np.cumsum(ve_component.mean(axis=0)) - np.cumsum(all_speed_component.mean(axis=0))
##                diff_curve_component = np.cumsum(ve_component.mean(axis=0)).copy()
#                diff_curve_component[diff_curve_component<0] = 0
#                change_curve_component = np.diff(diff_curve_component)
#                
#                diff_curves_all.append(diff_curve_component)
#                all_embryos_plot.append(embryo_folder.split('/')[-1])
#                
#                from scipy.signal import savgol_filter
#                
##                plt.figure()
##                plt.plot(diff_curve_component, label='diff component')
##                plt.legend(loc='best')
##                plt.show()
## =============================================================================
##                 offline changepoint detection?
## =============================================================================
#                import bayesian_changepoint_detection.offline_changepoint_detection as offcd
#                from functools import partial
#                from scipy.signal import find_peaks
#                from scipy.interpolate import UnivariateSpline
#                
##                spl = UnivariateSpline(np.arange(len(diff_curve_component)), diff_curve_component, k=1,s=1e-4)
##                spl_diff_component = spl(np.arange(len(diff_curve_component)))
#                
##                spl_diff_component = savgol_filter(diff_curve_component, 11,1)
##                diff_curve_component = savgol_filter(diff_curve_component, 15,1)
#                
#                # switch with an ALS filter?
#                diff_curve_component_ = baseline_als(diff_curve_component, lam=1, p=0.25, niter=10) # use lam=5 for lifeact, what about lam=1?
#                
#                plt.figure()
#                plt.plot(diff_curve_component)
##                plt.plot(spl_diff_component)
#                plt.show()
#                
#                # what about detection by step filter convolving (matched filter)
#                up_step = np.hstack([-1*np.ones(len(diff_curve_component)//2), 1*np.ones(len(diff_curve_component)-len(diff_curve_component)//2)])
##                up_step = np.hstack([-1*np.ones(len(diff_curve_component)), 1*np.ones(len(diff_curve_component)-len(diff_curve_component))])
#                down_step = np.hstack([1*np.ones(len(diff_curve_component)//2), -1*np.ones(len(diff_curve_component)-len(diff_curve_component)//2)])
#                
#                conv_res = np.convolve((diff_curve_component - np.mean(diff_curve_component))/(np.std(diff_curve_component)+1e-8)/len(diff_curve_component), down_step, mode='same')
#                
##                plt.figure()
##                plt.subplot(211)
##                plt.plot(diff_curve_component)
##                plt.subplot(212)
##                plt.plot(conv_res)
##                plt.ylim([-1,1])
##                plt.show()
###                plt.figure()
###                plt.plot(np.convolve(conv_res, ))
##                plt.figure()
##                plt.subplot(211)
##                plt.plot(diff_curve_component)
###                plt.plot(spl_diff_component)
##                plt.subplot(212)
##                plt.plot(np.diff(diff_curve_component))
###                plt.plot(np.diff(spl_diff_component))
##                plt.show()
#                
#                print(len(diff_curves_all))
#                
#                # put in the filtered. 
#                in_signal = np.diff(diff_curve_component_)
##                in_signal =  np.diff(np.cumsum(in_signal)/(np.arange(len(in_signal)) + 1))
#                conv_res_up = np.convolve((in_signal - np.mean(in_signal))/(np.std(in_signal)+1e-8)/len(in_signal), down_step, mode='same')
#                conv_res_down = np.convolve((in_signal - np.mean(in_signal))/(np.std(in_signal)+1e-8)/len(in_signal), up_step, mode='same')
#                
#                peaks_conv = find_peaks(np.abs(conv_res_up), distance=5, height=0.2)[0]
#                break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=1.5e-3) # make this automatic... 
##                break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=2.5e-3)
##                if len(break_grads[-1][1]) > 0:
##                    break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=.5*np.max(break_grads[0])) # 10% the max? 
##                    print(.5*np.max(break_grads[0]))
#                
#                regime_colors = ['r','g','b']
#                
#                plt.figure(figsize=(4,7))
#                plt.subplot(411)
#                plt.plot(np.cumsum(ve_component.mean(axis=0)), label='VE')
#                plt.plot(np.cumsum(all_speed_component.mean(axis=0)), label='Epi')
#                plt.legend()
#
#                plt.subplot(412)
#                plt.plot(diff_curve_component)
#                plt.plot(peaks_conv, diff_curve_component[peaks_conv], 'go')
#                
#                for r_i in range(len(break_grads[-1])):
#                    for rr in break_grads[-1][r_i]:
#                        plt.fill_between(rr[1:], np.min(diff_curve_component), np.max(diff_curve_component), color=regime_colors[r_i], alpha=0.25)
#                
#                plt.subplot(413)
#                plt.plot(in_signal)
#                plt.plot(break_grads[1])
#                plt.hlines(0, 0, len(in_signal))
#                plt.subplot(414)
#                plt.plot(np.abs(conv_res_up))
#                plt.plot(peaks_conv, np.abs(conv_res_up)[peaks_conv], 'go')
##                plt.plot(conv_res_down)
#                plt.ylim([0,1])
#                plt.show()
#                
#                # compute the projected distance speed... 
#                merged_timepts = merge_time_points(np.arange(len(vid_ve)), break_grads[-1])
#                
#                print('predicted time stages')
#                print(merged_timepts)
#                print('=============')
#                pre_all_auto.append(merged_timepts[0])
#                mig_all_auto.append(merged_timepts[1])
#                post_all_auto.append(merged_timepts[2])
#                
#                
#    """
#    check diff curves for all . 
#    """
#    pre_data = []; 
#    for d in pre_all_auto:
#        if len(d) == 0:
#            pre_data.append(np.nan)
#        else:
#            pre_data.append('_'.join([str(p+1) for p in d]))
#    pre_data = np.hstack(pre_data)
#    
#    mig_data = []; 
#    for d in mig_all_auto:
#        if len(d) == 0:
#            mig_data.append(np.nan)
#        else:
#            mig_data.append('_'.join([str(p+1) for p in d]))
#    mig_data = np.hstack(mig_data)
#            
#    post_data = []; 
#    for d in post_all_auto:
#        if len(d) == 0:
#            post_data.append(np.nan)
#        else:
#            post_data.append('_'.join([str(p+1) for p in d]))
#    post_data = np.hstack(post_data)
#    
#    
#    all_table_staging_auto = pd.DataFrame(np.vstack([all_embryos_plot, 
#                                                       pre_data,
#                                                       mig_data,
#                                                       post_data]).T, columns=['Embryo', 'Pre-migration', 'Migration to Boundary', 'Post-Boundary'])
#    
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES.xlsx', index=None)
#    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-LifeAct-smooth-lam1.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG_TTR-smooth-lam1.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-smooth-lam5.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-new_hex-smooth-lam1.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-1_smooth.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-new_hex-4-VE-1.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG_TTR.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES_nothresh.xlsx', index=None)
#
#    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-LifeAct.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-MTMG-TTR.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-MTMG-HEX-new_hex.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-MTMG-HEX.mat'
#    spio.savemat(savematfile, {'embryos': all_embryos_plot, 
#                               'staging_table': all_table_staging_auto, 
#                               'staging_curves': diff_curves_all})


