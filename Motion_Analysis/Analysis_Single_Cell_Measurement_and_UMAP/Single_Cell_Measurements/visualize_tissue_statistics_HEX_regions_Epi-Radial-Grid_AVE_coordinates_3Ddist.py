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

def lookup_3D_mean(pts2D, unwrap_params, force_z=None):
    
    pts3D = unwrap_params[pts2D[:,0].astype(np.int), 
                          pts2D[:,1].astype(np.int)]
       
    pts3D_mean = np.median(pts3D, axis=0)
    if force_z is not None:
        pts3D_mean[0] = force_z
    
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

def geodesic_distance_2D(pt1, pt2, unwrap_params, n_samples = 10, decimate_rate=8):

    # helper function to compute a geodesic line along the 3D surface.
    dist2D = int(np.linalg.norm(pt1-pt2)/float(decimate_rate))
    dist2D = np.minimum(20, dist2D)
    
    line_x = np.linspace(pt1[0], pt2[0], dist2D).astype(np.int)
    line_y = np.linspace(pt1[1], pt2[1], dist2D).astype(np.int)

    line_xyz = np.vstack([line_x,line_y]).T
    line_xyz_3D = unwrap_params[line_xyz[:,0], line_xyz[:,1]] # lift line into 3D 

    geodesic_dist = np.sum(np.linalg.norm(line_xyz_3D[1:] - line_xyz_3D[:-1], axis=-1))

    return line_xyz_3D, geodesic_dist


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

def resample_curve(x,y,s=0, k=1, n_samples=10):
    
    import scipy.interpolate
    
    tck, u = scipy.interpolate.splprep([x,y], s=s, k=k)
    unew = np.linspace(0, 1.00, n_samples)
    out = scipy.interpolate.splev(unew, tck) 
    
    return np.vstack(out).T

def measure_geodesic_distances_centroid_epi(centroid, ref_point, epi_line, unwrap_params):

    # all should be in (y,x) convention.
    epi_line_angle = np.arctan2(epi_line[:,0]-ref_point[0],
                                epi_line[:,1]- ref_point[1])
    centroid_angle = np.arctan2(centroid[0]-ref_point[0], centroid[1] - ref_point[1])

    print(centroid_angle, centroid, ref_point)
    epi_line_select = np.argmin(np.abs(epi_line_angle-centroid_angle))
    epi_line_pt = epi_line[epi_line_select]

    dist_centroid_ref = geodesic_distance_2D(ref_point, centroid, unwrap_params, n_samples = 10)
    dist_centroid_epi = geodesic_distance_2D(centroid, epi_line_pt, unwrap_params, n_samples = 10)
#    dist_centroid_ref = geodesic_distance(ref_point, centroid, unwrap_params, n_samples = 25)
#    dist_centroid_epi = geodesic_distance(centroid, epi_line_pt, unwrap_params, n_samples = 25)
    
    return dist_centroid_ref, dist_centroid_epi, epi_line_pt

def geodesic_distance3D(pt1, pt2, unwrap_params, nbr_model=None, nbr_pts3D=None, n_samples = 10):

    # helper function to compute a geodesic line along the 3D surface.
    from sklearn.neighbors import NearestNeighbors 

    if nbr_model is None:
        nbrs_model = NearestNeighbors(n_neighbors=1, algorithm='auto') 
        pts3D_all = unwrap_params.reshape(-1,unwrap_params.shape[-1])
        nbrs_model.fit(pts3D_all)
    else:
        nbrs_model = nbr_model
        pts3D_all = nbr_pts3D.copy()

    pt1_3D = unwrap_params[int(pt1[0]), int(pt1[1])]
    pt2_3D = unwrap_params[int(pt2[0]), int(pt2[1])]

    dist12_3D = np.linalg.norm(pt1_3D - pt2_3D)

    N = int(np.minimum(n_samples, dist12_3D))
    ndim = unwrap_params.shape[-1]

    lines = []
    for dim in range(ndim):
        lines.append(np.linspace(pt1_3D[dim], pt2_3D[dim], N+1))

    line = np.vstack(lines).T
    neighbor_indices = nbrs_model.kneighbors(line, return_distance=False)
    line3D_surf = pts3D_all[neighbor_indices]; 
    line3D_surf = line3D_surf.mean(axis=1) # this is to remove the singleton dimension due to the kneighbours.

    geodesic_dist = np.sum(np.linalg.norm(line3D_surf[1:] - line3D_surf[:-1], axis=-1))

    return line3D_surf, geodesic_dist

def measure_geodesic_distances_centroid_3D(centroid, ref_point, epi_line, unwrap_params, nbr_model, ref_pts_3D, n_samples=10):

    # all should be in (y,x) convention.
    epi_line_angle = np.arctan2(epi_line[:,0]-ref_point[0],
                                epi_line[:,1]- ref_point[1])
    centroid_angle = np.arctan2(centroid[0]-ref_point[0], centroid[1] - ref_point[1])

    print(centroid_angle, centroid, ref_point)
    epi_line_select = np.argmin(np.abs(epi_line_angle-centroid_angle))
    epi_line_pt = epi_line[epi_line_select]

    # dist_centroid_ref = geodesic_distance_2D(ref_point, centroid, unwrap_params, n_samples = 10)
    # dist_centroid_epi = geodesic_distance_2D(centroid, epi_line_pt, unwrap_params, n_samples = 10)
    dist_centroid_ref = geodesic_distance3D(ref_point, centroid, unwrap_params, nbr_model=nbr_model, nbr_pts3D=ref_pts_3D, n_samples = n_samples)
    dist_centroid_epi = geodesic_distance3D(centroid, epi_line_pt, unwrap_params, nbr_model=nbr_model, nbr_pts3D=ref_pts_3D, n_samples = n_samples)
    
    return dist_centroid_ref, dist_centroid_epi, epi_line_pt

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

    from MOSES.Motion_Analysis import tracks3D_tools as tra3Dtools
    
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
    import skimage.io as skio
    from skimage.measure import find_contours

    """
    set the physical scaling parameters.
    """
    voxel_size = 0.3630 #um
    Ts = 10. # for hex this is 10 mins. 


#    master_analysis_save_folder = '/media/felix/Srinivas4/All_results/Migration_Analysis/Staged_Analysis'
#    fio.mkdir(master_analysis_save_folder)
    
    # master_plot_save_folder = '/media/felix/My Passport/Shankar-2/All_Results/Visualizations/HEX-AVE_classifier_consensus_angles_corrected-Temporal(AVE corrected coordinates, Curvilinear,3Ddist)'
    master_plot_save_folder = '/media/felix/My Passport/Shankar-2/All_Results/Visualizations/HEX-AVE_classifier_consensus_angles_Statistics_Overlay(AVE corrected coordinates)'
    fio.mkdir(master_plot_save_folder)

    # define the locations of some extracted statistics folders. 
    masterdemonsfolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper' # this is the shape changes + growth variation. 
    master_curvature_folder = '/media/felix/My Passport/Shankar-2/All_Results/Volumetric_Statistics_All' 
    mastercellstatsfolder = '/media/felix/My Passport/Shankar-2/All_Results/Single_Cell_Shape_Statistics_All/Data'
    
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Set up a nearest neighbour model 
    from sklearn.neighbors import NearestNeighbors 
    from skimage.segmentation import mark_boundaries
    nbrs_model = NearestNeighbors(n_neighbors=9, algorithm='auto') # how many neighbours to include? 9 = 8-connected?
    
    
    # all_rootfolders = ['/media/felix/Srinivas4/LifeAct', 
    #                    '/media/felix/Srinivas4/MTMG-TTR',
    #                    '/media/felix/Srinivas4/MTMG-HEX',
    #                    '/media/felix/Srinivas4/MTMG-HEX/new_hex']

    all_rootfolders = ['/media/felix/Srinivas4/MTMG-HEX',
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
    
    # all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-LifeAct-smooth-lam5-correct_norot_consensus.xlsx', 
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-TTR-smooth-lam5-correct_norot_consensus.xlsx',
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-HEX-smooth-lam5-correct_norot_consensus.xlsx',
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-new-HEX-smooth-lam5-correct_norot_consensus.xlsx']
    all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-HEX-smooth-lam5-correct_norot_consensus.xlsx',
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
#    all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles/LifeAct_polar_angles_AVE_classifier.-consensus.csv',
#                       '/media/felix/Srinivas4/Data_Info/Migration_Angles/MTMG-TTR_polar_angles_AVE_classifier.-consensus.csv',
#                       '/media/felix/Srinivas4/Data_Info/Migration_Angles/MTMG-HEX_polar_angles_AVE_classifier.-consensus.csv',
#                       '/media/felix/Srinivas4/Data_Info/Migration_Angles/new_hex_polar_angles_AVE_classifier.-consensus.csv']
    
    # all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/LifeAct_polar_angles_AVE_classifier-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-TTR_polar_angles_AVE_classifier-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-HEX_polar_angles_AVE_classifier-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/new_hex_polar_angles_AVE_classifier-consensus.csv']
    all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-HEX_polar_angles_AVE_classifier-consensus.csv',
                       '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/new_hex_polar_angles_AVE_classifier-consensus.csv']
    
    
    angle_tab = pd.concat([pd.read_csv(angle_file) for angle_file in all_angle_files], ignore_index=True)
#    all_stages = staging_tab.columns[1:4]
    
#     analyse_stage = str(all_stages[0]) # Pre-migration stage
# #    analyse_stage = str(all_stages[1]) # Migration stage
# #    analyse_stage = str(all_stages[2]) # Post-migration stage
#     print(analyse_stage)
    
#    master_analysis_save_folder = os.path.join('/media/felix/Srinivas4/All_results/Migration_Analysis/Staged_Analysis-growth_correct', analyse_stage)
    # master_analysis_save_folder = os.path.join('/media/felix/Srinivas4/All_results/Migration_Analysis/Staged_Analysis-growth_correct-Epi_Grid-auto-polar_Paper-test', analyse_stage)
    # fio.mkdir(master_analysis_save_folder)
    
    """
    set the master saveplot folders.
    """

    """
    Iterate over experimental folders to build quadranting space.....
    """
    # all_analysed_embryos = []
    # all_grids = []
    # all_embryo_shapes = []
    # all_embryo_vol_factors = []
    # all_ve_angles = []
    
    # # statistics to export
    # all_directions_regions_ave = []
    # all_directions_regions_hex = []
    # all_centroids_Hex = []
    # all_centroids_Hex_hi = []
    # all_centroids_AVE_classifier = []
    # all_distance_Hex = []   
    # all_distance_Hex_hi = []
    # all_distance_AVE = []
    # all_plot_centres = []

    for embryo_folder in all_embryo_folders[:-1]:
#    for embryo_folder in all_embryo_folders[:1]:
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
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-new')
            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-curvilinear_Tcorrected')

#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s' %(n_spixels, analyse_stage))
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_stage-aligned-correct_uniform_scale-%d_%s' %(n_spixels, analyse_stage))
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s-manual' %(n_spixels, analyse_stage))
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_stage-auto-polar-aligned-Paper-%d_%s' %(n_spixels, analyse_stage))
            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels))
            
#            , 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')

            """
            To Do: switch to the manual extracted tracks. 
            """
            saveboundaryfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow-epi-boundary')

            save_ave_classifier_folder =  os.path.join(embryo_folder, 'MOSES_analysis', 'AVE_classifer_region_corrected (AVE coordinates, Curvilinear, 3Ddist)')
            fio.mkdir(save_ave_classifier_folder)
            save_hex_seg_folder =  os.path.join(embryo_folder, 'MOSES_analysis', 'Hex_region_segmentation (AVE coordinates, Curvilinear, 3Ddist)')
            fio.mkdir(save_hex_seg_folder)

#            if os.path.exists(savetrackfolder) and len(os.listdir(savetrackfolder)) > 0:
                # only tabulate this statistic if it exists. 
#                all_analysed_embryos.append(embryo_folder)
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
                    
                    
                    if 'MTMG-HEX' in embryo_folder:
                        vid_hex = fio.read_multiimg_PIL(hex_unwrap_file)
                

                # =============================================================================
                #       Load the relevant manual staging file from Holly and Matt.        
                # =============================================================================
                    """
                    create the lookup embryo name for the manual staging 
                    """
    #                if proj_condition[1] == '':
    #                    embryo_name = proj_condition[0]
    #                else:
    #                    embryo_name = proj_condition[0]+'_E'+proj_condition[1].split('Emb')[1]
                        
                    """
                    create the lookup embryo name for the auto staging 
                    """
                    embryo_name = os.path.split(embryo_folder)[-1]
                    
                    table_emb_names = np.hstack([str(s) for s in staging_tab['Embryo'].values])
                    select_stage_row = staging_tab.loc[table_emb_names==embryo_name]
                    
                    pre = parse_start_end_times(select_stage_row['Pre-migration'])[0]
                    mig = parse_start_end_times(select_stage_row['Migration to Boundary'])[0]
                    post = parse_start_end_times(select_stage_row['Post-Boundary'])[0]


                # =============================================================================
                #       Load the relevant saved pair track file.               
                # =============================================================================
#                    trackfile = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_%s_' %(n_spixels, analyse_stage) + unwrapped_condition+'.mat')
                    trackfile = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')
                    
                    meantracks_ve = spio.loadmat(trackfile)['meantracks_ve']
                    meantracks_epi = spio.loadmat(trackfile)['meantracks_epi']
                    
#                    proj_condition = spio.loadmat(trackfile)['condition'][0].split('-')
                    proj_condition = unwrapped_condition.split('-')
                    proj_type = proj_condition[-1]
#                    start_end_times = spio.loadmat(trackfile)['start_end'].ravel()
                

                # =============================================================================
                #       Load the relevant optical flow information.                
                # =============================================================================
                    """
                    1. load the flow file for VE and Epi surfaces. 
                    """
                    flowfile_ve = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(ve_unwrap_file)[-1].replace('.tif', '.mat'))
                    flow_ve = spio.loadmat(flowfile_ve)['flow']
                
                    flowfile_epi = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(epi_unwrap_file)[-1].replace('.tif', '.mat'))
                    flow_epi = spio.loadmat(flowfile_epi)['flow']
                    flow_epi_resize = sktform.resize(flow_epi, flow_ve.shape, preserve_range=True) # resize to be the same size as that of the VE for comparative analysis

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
                    
#                    epi_contour_line = epi_boundary_time[start_end_times[0]]
                    epi_contour_line = epi_boundary_time # just want the initial division line of the Epi.
                    
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
                            epi_contour_line[...,0] = (vid_ve[0].shape[1]-1) - epi_contour_line[...,0]   
                            
                            
                            if 'MTMG-HEX' in embryo_folder:
                                vid_hex = vid_hex[:,:,::-1]
                            
                            # apply correction to optflow.
                            flow_ve = flow_ve[:,:,::-1]
                            flow_ve[...,0] = -flow_ve[...,0] #(x,y)
                            flow_epi = flow_epi[:,:,::-1]
                            flow_epi[...,0] = -flow_epi[...,0] #(x,y)
                            flow_epi_resize = flow_epi_resize[:,:,::-1]
                            flow_epi_resize[...,0] = -flow_epi_resize[...,0] #(x,y)

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
                            
                            if 'MTMG-HEX' in embryo_folder:
                                vid_hex = vid_hex[:,::-1]
                            
                            epi_contour_line[...,1] = (vid_ve[0].shape[0]-1) - epi_contour_line[...,1] 

                            # apply correction to optflow.
                            flow_ve = flow_ve[:,::-1]
                            flow_ve[...,1] = -flow_ve[...,1] #(x,y)
                            flow_epi = flow_epi[:,::-1]
                            flow_epi[...,1] = -flow_epi[...,1] #(x,y)
                            flow_epi_resize = flow_epi_resize[:,::-1]
                            flow_epi_resize[...,1] = -flow_epi_resize[...,1] #(x,y)


                # =============================================================================
                #   Create the final polar grid on the inversion corrected embryo (if applicable)             
                # =============================================================================
                
                    # create the initial radial tiling oriented with region 1 = VE direction. 
                    
                    # use this to get the temporal 
                    polar_grid_time  = []
                    polar_grid_time_contour_lines = [] 
                    
                    for tt in np.arange(len(vid_ve)):
                        polar_grid = tile_uniform_windows_radial_guided_line(vid_ve[tt].shape, 
                                                                            n_r=4, 
                                                                            n_theta=8,
                                                                            max_r=vid_ve[0].shape[0]/2./1.2,
                                                                            mid_line = epi_contour_line[tt][:,[0,1]],
                                                                            center=None, 
                                                                            bound_r=True,
                                                                            zero_angle = ve_angle_move.values[0])
                        polar_grid_time.append(polar_grid)
                        
                        polar_grid_contour_line = []
                        uniq_regions = np.setdiff1d(np.unique(polar_grid), 0)
                        
                        for reg in uniq_regions:
                            cnt = find_contours(polar_grid == reg, 0)[0]
                            polar_grid_contour_line.append(cnt)
                            
                        polar_grid_time_contour_lines.append(polar_grid_contour_line)
                        
                    polar_grid_time = np.array(polar_grid_time)
                            
                
#                    uniq_polar_regions = np.unique(polar_grid)[1:]
#                    
#                    # assign tracks to unique regions. 
#                    uniq_polar_spixels = [(polar_grid==r_id)[meantracks_ve[:,0,0], meantracks_ve[:,0,1]] for r_id in uniq_polar_regions]
                    
                         
                    # check.... do we need to invert the epi boundary too !.    
                    
#                        plt.figure(figsize=(10,10))
##                        plt.imshow(vid_ve[0], cmap='gray')
#                        plt.imshow(mark_boundaries(np.uint8(np.dstack([vid_ve[-1], vid_ve[-1], vid_ve[-1]])), 
#                                                   polar_grid), cmap='gray')
##                        plt.imshow(polar_grid, cmap='coolwarm', alpha=0.5)
#                        plt.show()
#                        
#                        plt.figure(figsize=(10,10))
##                        plt.imshow(vid_ve[0], cmap='gray')
#                        plt.imshow(mark_boundaries(np.uint8(np.dstack([vid_ve.max(axis=0), 
#                                                                       vid_ve.max(axis=0), 
#                                                                       vid_ve.max(axis=0)])), 
#                                                   polar_grid), cmap='gray')
##                        plt.imshow(polar_grid, cmap='coolwarm', alpha=0.5)
#                        plt.show()
                    plt.figure(figsize=(10,10))
#                        plt.imshow(vid_ve[0], cmap='gray')
                    plt.imshow(mark_boundaries(np.uint8(np.dstack([vid_ve[0], vid_ve[0], vid_ve[0]])), 
                                               polar_grid), cmap='gray')
                    mean_disps_2D = (meantracks_ve[:,1:] - meantracks_ve[:,:-1]).mean(axis=1)
                    plt.quiver(meantracks_ve[:,0,1], 
                               meantracks_ve[:,0,0], 
                               mean_disps_2D[:,1], 
                               -mean_disps_2D[:,0], color='r')
                    plt.show()
                    
                # =============================================================================
                #   Compute the 3D AVE mean direction given the direction !.                       
                # =============================================================================                        
                    ve_angle_vector = np.hstack([np.sin((90+ve_angle_move)/180. * np.pi), 
                                                 np.cos((90+ve_angle_move)/180. * np.pi)])
    
    
                    ref_2D_pos = np.hstack([int(ve_seg_center[0]), 
                                            int(ve_seg_center[1])])
                    query_2D_pos = np.hstack([int(ve_seg_center[0]+10*ve_angle_vector[1]), 
                                              int(ve_seg_center[1]+10*ve_angle_vector[0])])
    
                    ref_3D_pos = unwrap_params_ve[ref_2D_pos[0], 
                                                  ref_2D_pos[1]]
                    query_ref_3D_pos = unwrap_params_ve[query_2D_pos[0], 
                                                        query_2D_pos[1]]
                    
                    fig, ax = plt.subplots()
#                    ax.imshow(ve_seg_select_rect_valid)
                    ax.imshow(vid_epi_resize[0], cmap='gray')
                    ax.plot(ref_2D_pos[1], 
                            ref_2D_pos[0], 'wo')
                    ax.plot(query_2D_pos[1], 
                            query_2D_pos[0], 'ro')
                    
                    ax.plot([ref_2D_pos[1], ref_2D_pos[1]+ 100*ve_angle_vector[0]], 
                            [ref_2D_pos[0], ref_2D_pos[0]+ 100*ve_angle_vector[1]], 'w--')
                    plt.show()
                    
                    mean_ve_direction_3D = (query_ref_3D_pos - ref_3D_pos)
                    mean_ve_direction_3D = mean_ve_direction_3D/(np.linalg.norm(mean_ve_direction_3D) + 1e-8) # normalise this direction !. 
                    

                    """
                    Primary processing. 
                    """
                    # =============================================================================
                    #     This is only to get the mean average direction right?                 
                    # =============================================================================
                    # find the effective superpixel size. 
                    spixel_size = np.abs(meantracks_ve[1,0,1] - meantracks_ve[2,0,1])
                            
                    if np.isnan(ve_seg_migration_times[0]):
                        # the first time is adequate and should be taken for propagation. 
                        ve_seg_select_rect = ve_seg_select_rect_valid[ve_tracks_full[:,0,0], 
                                                                      ve_tracks_full[:,0,1]] > 0
                                                                      
                        ve_seg_select_rect_svm = ve_seg_select_rect_valid_svm[ve_tracks_full[:,0,0], 
                                                                              ve_tracks_full[:,0,1]] > 0
                    else:
                        start_ve_time = int(ve_seg_migration_times[0]) # these are the manual times. 
                        
                        # retrieve the classification at the VE start point. 
                        ve_seg_select_rect = ve_seg_select_rect_valid[ve_tracks_full[:,start_ve_time,0], 
                                                                      ve_tracks_full[:,start_ve_time,1]] > 0
                                                                      
                        ve_seg_select_rect_svm = ve_seg_select_rect_valid_svm[ve_tracks_full[:,start_ve_time,0], 
                                                                              ve_tracks_full[:,start_ve_time,1]] > 0
                                                                      
#                     recompute the connected points at t=0 for this region so we can apply to the new tracks after filtering? 
                    
#                        ve_seg_tracks = ve_tracks_full[ve_seg_select_rect,start_end_times[0]:start_end_times[1]]
                    ve_seg_tracks = ve_tracks_full[ve_seg_select_rect,:]
                    ve_seg_track_pts = ve_seg_tracks[:,0]
                    
                    
                    """
                    new positions. 
                    """
                    start_end_times = [0, len(vid_ve)] # the full length of the video. !
                        
                    connected_labels = postprocess_polar_labels(ve_tracks_full[:,start_end_times[0]], 
                                                                ve_seg_select_rect, dist_thresh=2*spixel_size)

                    connected_labels_svm = postprocess_polar_labels(ve_tracks_full[:,start_end_times[0]], 
                                                                    ve_seg_select_rect_svm, dist_thresh=2*spixel_size)

                    # now we do the propagation to infer the frame-by-frame result

                    AVE_classifier_vid_directional = []
                    AVE_classifier_vid_directional_SVM = []
                    Hex_segmentation_vid_thresh0 = []
                    Hex_segmentation_vid_threshhi = []

                    # save centroids:
                    AVE_classifier_vid_directional_centroids = []
                    AVE_classifier_vid_directional_SVM_centroids = []
                    Hex_segmentation_vid_thresh0_centroids = []
                    Hex_segmentation_vid_threshhi_centroids = []

                    Epi_line_centroids = []

                    center_AVE_dists = []
                    center_AVE_SVM_dists = []
                    center_Hex_dists = []
                    center_Hex_dists_hi = []

                    AVE_epi_dists = []
                    AVE_SVM_epi_dists = []
                    Hex_epi_dists = []
                    Hex_epi_dists_hi = []
                    
                    Epi_line_pts_AVE = []
                    Epi_line_pts_AVE_SVM = []
                    Epi_line_pts_Hex = []
                    Epi_line_pts_Hex_hi = []
                    
                    Geom_centers = []


                    # =============================================================================
                    #       Build the surface properties and AVE surface direction vectors for projection. 
                    # =============================================================================     
                    surf_nbrs_ve, surf_nbrs_ve_3D_pts = tra3Dtools.learn_nearest_neighbor_surf_embedding_model(unwrap_params_ve, 
                                                                                                                nearest_K=1)

                    # step 2: build the surface AVE directional vectors in a track format. # more complicated due to nan....
                    AVE_surf_dir_ve = tra3Dtools.proj_2D_direction_3D_surface(unwrap_params_ve, 
                                                                                ve_angle_vector[::-1][None,None,:], 
                                                                                pt_array_time_2D = None,
                                                                                K_neighbors=1, 
                                                                                nbr_model=surf_nbrs_ve, 
                                                                                nbr_model_pts=surf_nbrs_ve_3D_pts, 
                                                                                dist_sample=10, 
                                                                                return_dist_model=False,
                                                                                mode='dense')

                    surf_nbrs_epi, surf_nbrs_epi_3D_pts = tra3Dtools.learn_nearest_neighbor_surf_embedding_model(unwrap_params_epi, 
                                                                                                                nearest_K=1)

                    # step 2: build the surface AVE directional vectors in a track format. # more complicated due to nan....
                    AVE_surf_dir_epi = tra3Dtools.proj_2D_direction_3D_surface(unwrap_params_epi, 
                                                                                ve_angle_vector[::-1][None,None,:], 
                                                                                pt_array_time_2D = None,
                                                                                K_neighbors=1, 
                                                                                nbr_model=surf_nbrs_ve, 
                                                                                nbr_model_pts=surf_nbrs_ve_3D_pts, 
                                                                                dist_sample=10, 
                                                                                return_dist_model=False,
                                                                                mode='dense')
                    
                    # =============================================================================
                    #       Read and load up the gaussian curvature images.  
                    # =============================================================================     
                    # load the curvature. 
                    emb_out_save_folder = os.path.join(master_curvature_folder, unwrapped_condition);
                    savematfile_curvature = os.path.join(emb_out_save_folder, 'curvature_dense_'+unwrapped_condition+'.mat')
                    curvature_obj = spio.loadmat(savematfile_curvature)

                    # note .... these are already operating on the inversed.
                    curvature_VE = curvature_obj['curvature_VE']
                    curvature_Epi = curvature_obj['curvature_Epi']
                    print(curvature_VE.shape, curvature_Epi.shape)
                    print('+++++++++++++++++')

#                    # =============================================================================
#                    #       Load the epi-contour time  
#                    # =============================================================================     
#                    savephystatmatfile = os.path.join(mastercellstatsfolder, 'cell_physical_stats_'+unwrapped_condition+'.mat')
#                    cellphysstatobj = spio.loadmat(savephystatmatfile) # load this up. 
#
#                    polar_grid_coarse_time = cellphysstatobj['polar_grids_coarse_contour']
#                    polar_grid_coarse_time_rot_AVE = np.array([np.uint8(sktform.rotate(polar_grid_frame, angle=(90-ve_angle_move), 
#                                                                                       order=0,
#                                                                                       preserve_range=True)) for polar_grid_frame in polar_grid_coarse_time] )
#                    
#                    polar_grid_coarse_time_rot_AVE_contour_lines = extract_contour_lines_polar_grid_time(polar_grid_coarse_time_rot_AVE) #hm... but might be far more accurate to extract before rotation. 

                    """
                    Clipping mask 
                    """
                    clip_mask = np.zeros(vid_ve[0].shape)
                    XX, YY = np.meshgrid(range(clip_mask.shape[1]), range(clip_mask.shape[0]))
                    clip_mask = np.sqrt((XX-clip_mask.shape[1]/2.)**2 + (YY-clip_mask.shape[0]/2.)**2)
                    clip_mask = clip_mask<= vid_ve[0].shape[0]/2./1.2; #clip_mask = np.logical_not(clip_mask)


                    # iterate over the respective stages to build up the results.
                    analyse_stages = ['Pre-migration', 
                                      'Migration to Boundary',
                                      'Post-Boundary']
                    stage_embryo = [pre, mig, post]
                    
                    
                    fig, ax = plt.subplots(figsize=(15,15))
                    ax.imshow(vid_ve[0], cmap='gray')
                    ax.imshow(curvature_VE[0], cmap='coolwarm', vmin=-2.5e-4, vmax=2.5e-4, alpha=.25)
                    plt.show()
                    
                    fig, ax = plt.subplots(figsize=(15,15))
                    ax.imshow(vid_ve[0], cmap='gray')
                    ax.imshow(curvature_Epi[0], cmap='coolwarm', vmin=-5e-4, vmax=5e-4, alpha=.25)
                    plt.show()
                
                
                    # =============================================================================
                    #       Iterate over all the timepoints. and compute the relevant information.  
                    # =============================================================================

                    counter = 0
                    rot_center = np.hstack(vid_ve[0].shape)/2.
                    # Go through the timepoints by going through the stages. 

                    for stage_ii in range(len(stage_embryo))[:]:

                        times = stage_embryo[stage_ii]
                        times = times.ravel()
#                        ~np.isnan(pre.ravel()[0]) and ~np.isnan(mig.ravel()[0]):
                        if ~np.isnan(times[0]):

                            analyse_stage = analyse_stages[stage_ii]
                            print(times)
                            
                            # make the adjustments required for computation. 
                            start, end = times
                            start = start -1 
                            if end == len(vid_ve):
                                end = end 
                            else:
                                end = end - 1
                
                            start = int(start)
                            end = int(end)
                            
                            start_end_times = np.hstack([start, end])


                            """
                            Load the demons computation
                            """
                            demons_savefolder = os.path.join(embryo_folder, 'Demons_analysis', 
                                                                                   'Demons_Temporal_Change_Shape', 
                                                                                    analyse_stage); 
            #                            fio.mkdir(savefolder)
                            new_demonsmastersavefolder = '/media/felix/My Passport/Shankar-2/Demons_remap_vector'
                            old_demonsmastersavefolder = os.path.split(embryo_folder)[0]
                                
                            demons_savefile = os.path.join(demons_savefolder, 'demons_unwrap_params_rev_pts_AVE_vects_'+unwrapped_condition+'.mat')
                            demons_saverootfolder, demons_savesuffix = os.path.split(demons_savefile)
                            demons_newsaverootfolder = demons_saverootfolder.replace(old_demonsmastersavefolder,
                                                                                        new_demonsmastersavefolder);

                            demons_savefile = os.path.join(demons_newsaverootfolder, demons_savesuffix)
                            demons_saveobj = spio.loadmat(demons_savefile)
                            # print('saving ... ', savefile)
                    # spio.savemat(savefile, 
                    #              {'ve_pts': ve_pts_stage.astype(np.float32),
                    #               'epi_pts': epi_pts_stage.astype(np.float32), 
                    #               'start_end_times':start_end_times, 
                    #               've_diff_pts': ve_diff_pts_stage.astype(np.float32), 
                    #               'epi_diff_pts': epi_diff_pts_stage.astype(np.float32), 
                    #               'analyse_stage': analyse_stages[stage_ii], 
                    #               've_dir_vect_AVE':ve_dir_vect_AVE.astype(np.float32),
                    #               've_dir_vect_AVE_perp':ve_dir_vect_AVE_perp.astype(np.float32),
                    #               'epi_dir_vect_AVE':epi_dir_vect_AVE.astype(np.float32),
                    #               'epi_dir_vect_AVE_perp':epi_dir_vect_AVE_perp.astype(np.float32),
                    #               've_angle_vector':ve_angle_vector, 
                    #               've_norm_angle_vector':ve_norm_angle_vector,
                    #               'embryo_inversion':embryo_inversion})
                            print(demons_saveobj['start_end_times'], start_end_times)
                            print('==========')

                            ###################### key computations to get the shape variation ####################
                            demons_save_ve_pts_stage = demons_saveobj['ve_pts']
                            demons_save_epi_pts_stage = demons_saveobj['epi_pts']
                            demons_ve_diff_pts_stage = demons_save_ve_pts_stage[1:] - demons_save_ve_pts_stage[:-1] 
                            demons_epi_diff_pts_stage = demons_save_epi_pts_stage[1:] - demons_save_epi_pts_stage[:-1] 

                            demons_ve_dir_vect_AVE = demons_saveobj['ve_dir_vect_AVE']
                            demons_ve_dir_vect_AVE_perp = demons_saveobj['ve_dir_vect_AVE_perp']
#                            ve_dir_vect_AVE_norm = np.array([np.cross(ve_dir_vect_AVE[tt], 
#                                                                      ve_dir_vect_AVE_perp[tt]) for tt in np.arange(len(ve_dir_vect_AVE_perp))])
                            demons_ve_dir_vect_AVE_norm = np.cross(demons_ve_dir_vect_AVE.reshape(-1,3), 
                                                                    demons_ve_dir_vect_AVE_perp.reshape(-1,3)).reshape(demons_ve_dir_vect_AVE.shape)
#                            ve_dir_vect_AVE_norm = np.array([vv / (np.linalg.norm(vv, axis=-1)[...,None]+1e-8) for vv in ve_dir_vect_AVE_norm])
                            demons_ve_dir_vect_AVE_norm = demons_ve_dir_vect_AVE_norm / (np.linalg.norm(demons_ve_dir_vect_AVE_norm, axis=-1)[...,None]+1e-8) 
                            
                            demons_epi_dir_vect_AVE = demons_saveobj['epi_dir_vect_AVE']
                            demons_epi_dir_vect_AVE_perp = demons_saveobj['epi_dir_vect_AVE_perp']
                            demons_epi_dir_vect_AVE_norm = np.cross(demons_epi_dir_vect_AVE, demons_epi_dir_vect_AVE_perp)
                            demons_epi_dir_vect_AVE_norm = demons_epi_dir_vect_AVE_norm / (np.linalg.norm(demons_epi_dir_vect_AVE_norm, axis=-1)[...,None]+1e-8)

                # =============================================================================
                #       Demons projection onto the local curvilinear vector orientations. 
                # =============================================================================

                            # so the below is the most important to save. 
                            demons_ve_diff_dir_vect_AVE = np.sum(demons_ve_diff_pts_stage * demons_ve_dir_vect_AVE, axis=-1)
                            demons_ve_diff_dir_vect_AVE_perp = np.sum(demons_ve_diff_pts_stage * demons_ve_dir_vect_AVE_perp, axis=-1)
                            demons_ve_diff_dir_vect_AVE_norm = np.sum(demons_ve_diff_pts_stage * demons_ve_dir_vect_AVE_norm, axis=-1)
                            # ve_diff_surface = ve_diff_pts_stage - ve_diff_dir_vect_AVE_norm[...,None] * ve_dir_vect_AVE_norm # minus the component normal to surface.
                            demons_ve_diff_surface = demons_ve_diff_dir_vect_AVE[...,None] * demons_ve_dir_vect_AVE + demons_ve_diff_dir_vect_AVE_perp[...,None]*demons_ve_dir_vect_AVE_perp

                            demons_epi_diff_dir_vect_AVE = np.sum(demons_epi_diff_pts_stage * demons_epi_dir_vect_AVE, axis=-1)
                            demons_epi_diff_dir_vect_AVE_perp = np.sum(demons_epi_diff_pts_stage * demons_epi_dir_vect_AVE_perp, axis=-1)
                            demons_epi_diff_dir_vect_AVE_norm = np.sum(demons_epi_diff_pts_stage * demons_epi_dir_vect_AVE_norm, axis=-1)
                            # epi_diff_surface = epi_diff_pts_stage - epi_diff_dir_vect_AVE_norm[...,None] * epi_dir_vect_AVE_norm
                            demons_epi_diff_surface = demons_epi_diff_dir_vect_AVE[...,None]*demons_epi_dir_vect_AVE + demons_epi_diff_dir_vect_AVE_perp[...,None] * demons_epi_dir_vect_AVE_perp 

                            """
                            Main processing.
                            """
                            cnt = 0
                            for iiii in np.arange(start_end_times[0], start_end_times[1], 1)[:]:

                                # frame_iiii = start_end_times[0]+iiii
                                frame_iiii = iiii
                                # 1. get the hex segmentation to overlay and rotate this into AVE frame of reference. # get the boundary of this. for overlaying. # hi and low?  
                                # segment with 0 as threshold 
                                hex_im = np.uint8(255*rescale_intensity(equalize_hist(vid_hex[frame_iiii])))
                                
                                # parse and smoothen the output a bit more. 
                                hex_im_binary = hex_im > 0 
                                hex_im_binary = skmorph.binary_closing(hex_im_binary, skmorph.disk(3))
                                hex_im_binary = binary_fill_holes(hex_im_binary)
                                hex_im_binary = skmorph.remove_small_objects(hex_im_binary, 250)

                                hex_im_binary_cont = find_contours(hex_im_binary, 0)
                                hex_im_binary_cont_largest = hex_im_binary_cont[np.argmax([len(l) for l in hex_im_binary_cont])]
                                
                                # hex_im_binary_cont_polygon = polygon(hex_im_binary_cont_largest[:,0],
                                #                                      hex_im_binary_cont_largest[:,1],
                                #                                      hex_im.shape)
                                hex_im_binary_cont_largest_rot = rotate_pts(hex_im_binary_cont_largest[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]

                                """
                                Repeating the analysis for Hex segmentation with higher threshold
                                """
        #                        hex_im_binary_hi = hex_im >= np.mean(hex_im) + 2*np.std(hex_im)
                                hex_im_binary_hi = hex_im >= threshold_otsu(hex_im)
                                hex_im_binary_hi = skmorph.binary_closing(hex_im_binary_hi, skmorph.disk(3))
                                hex_im_binary_hi = binary_fill_holes(hex_im_binary_hi)
                                hex_im_binary_hi = skmorph.remove_small_objects(hex_im_binary_hi, 250)
                                hex_im_binary_hi = gaussian(hex_im_binary_hi, sigma=3, preserve_range=True)
                                hex_im_binary_hi = hex_im_binary_hi > 0.1
                                
                                hex_im_binary_cont_hi = find_contours(hex_im_binary_hi, 0)
                                hex_im_binary_cont_largest_hi = hex_im_binary_cont_hi[np.argmax([len(l) for l in hex_im_binary_cont_hi])]
                                
                                # hex_im_binary_cont_polygon_hi = polygon(hex_im_binary_cont_largest_hi[:,0],
                                #                                         hex_im_binary_cont_largest_hi[:,1],
                                #                                         hex_im.shape)

                                # AVE frame -> plot this version. 
                                hex_im_binary_cont_largest_hi_rot = rotate_pts(hex_im_binary_cont_largest_hi[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]


                            # 2. rotate the curvature of VE and Epi into the AVE frame of reference.   

                                curvature_VE_frame_rot = sktform.rotate(curvature_VE[frame_iiii], 
                                                                        angle=(90-ve_angle_move), preserve_range=True, cval=0)
                                curvature_Epi_frame_rot = sktform.rotate(curvature_Epi[frame_iiii], 
                                                                        angle=(90-ve_angle_move), preserve_range=True, cval=0)

                                curvature_VE_frame_rot[clip_mask==0] = np.nan
                                curvature_Epi_frame_rot[clip_mask==0] = np.nan
                                
                                curvature_VE_frame_rot = curvature_VE_frame_rot / voxel_size
                                curvature_Epi_frame_rot = curvature_Epi_frame_rot / voxel_size
                                
                                # save the frame. 
                                plt.figure(figsize=(10,10))
                                # plt.subplot(121)
                                # plt.imshow(vid_ve[frame], cmap='gray')
                                plt.title(str(frame_iiii+1))

                                # hex_im = hex_im / float( hex_im.max() )
                                # vid_frame = sktform.rotate(hex_im ,
                                #                             angle=(90-ve_angle_move), preserve_range=True, cval=0)
                                # vid_frame = vid_frame/float(vid_frame.max())
                                # vid_frame [ clip_mask] = np.nan

                                # overlay_im = np.zeros((vid_frame.shape[0], vid_frame.shape[1], 3))
                                # overlay_im[...,1] = vid_frame
                                plt.imshow(curvature_VE_frame_rot, vmin=-5e-4, vmax=5e-4, cmap='coolwarm')

                                if 'MTMG-HEX' in embryo_folder:

                                    plt.plot(hex_im_binary_cont_largest_rot[:,1],
                                             hex_im_binary_cont_largest_rot[:,0],'g-', lw=3)
#                                    plt.plot(hex_im_binary_centroid_3D[1],
#                                             hex_im_binary_centroid_3D[0],'go')

                                    plt.plot(hex_im_binary_cont_largest_hi_rot[:,1],
                                             hex_im_binary_cont_largest_hi_rot[:,0],'--', lw=3, color='turquoise')
#                                    plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                             hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
                                if frame_iiii < len(polar_grid_time_contour_lines):

                                    polar_grid_lines_frame_ii = polar_grid_time_contour_lines[frame_iiii]
                                    for line in polar_grid_lines_frame_ii:
                                        line = rotate_pts(line[:,::-1], 
                                                          angle=-(90-ve_angle_move), 
                                                          center=(np.hstack(vid_ve[0].shape)/2.))[:,::-1]
                                        plt.plot(line[:,0], line[:,1], 'k', lw=2)

                                plt.axis('off')
                                plt.grid('off')
                                # save the visualisation in a plot. 
                                saveplotfigurefolder = os.path.join(master_plot_save_folder, unwrapped_condition, 'Curvature_VE')
                                fio.mkdir(saveplotfigurefolder)
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                plt.savefig(os.path.join(saveplotfigurefolder, 'curvature_VE_'+unwrapped_condition + '_%s.png' %(str(frame_iiii+1).zfill(3))), bbox_inches='tight')
                                plt.show()


#                                plt.figure(figsize=(10,10))
#                                plt.imshow(vid_epi_resize[0])
#                                plt.plot(epi_contour_line[0][:,0], 
#                                         epi_contour_line[0][:,1], 'k', lw=3)
#                                plt.show

                                plt.figure(figsize=(10,10))
                                # plt.subplot(121)
                                # plt.imshow(vid_ve[frame], cmap='gray')
                                plt.title(str(frame_iiii+1))

                                # hex_im = hex_im / float( hex_im.max() )
#                                vid_frame = sktform.rotate(vid_epi_resize[frame_iiii] ,
#                                                             angle=(90-ve_angle_move), preserve_range=True, cval=0)
#                                 vid_frame = vid_frame/float(vid_frame.max())
                                # vid_frame [ clip_mask] = np.nan

                                # overlay_im = np.zeros((vid_frame.shape[0], vid_frame.shape[1], 3))
                                # overlay_im[...,1] = vid_frame
                                plt.imshow(curvature_Epi_frame_rot, vmin=-5e-4, vmax=5e-4, cmap='coolwarm')
#                                plt.imshow(vid_frame)

                                if 'MTMG-HEX' in embryo_folder:

                                    plt.plot(hex_im_binary_cont_largest_rot[:,1],
                                             hex_im_binary_cont_largest_rot[:,0],'g-', lw=3)
#                                    plt.plot(hex_im_binary_centroid_3D[1],
#                                             hex_im_binary_centroid_3D[0],'go')

                                    plt.plot(hex_im_binary_cont_largest_hi_rot[:,1],
                                             hex_im_binary_cont_largest_hi_rot[:,0],'--', lw=3, color='turquoise')
#                                    plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                             hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
                                if frame_iiii < len(polar_grid_time_contour_lines):
                                    
                                    polar_grid_lines_frame_ii = polar_grid_time_contour_lines[frame_iiii]
                                    for line in polar_grid_lines_frame_ii:
                                        line = rotate_pts(line[:,::-1], 
                                                          angle=-(90-ve_angle_move), 
                                                          center=(np.hstack(vid_ve[0].shape)/2.))[:,::-1]
                                        plt.plot(line[:,1], line[:,0], 'k', lw=2)
                                        
                                plt.axis('off')
                                plt.grid('off')
                                # save the visualisation in a plot. 
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                saveplotfigurefolder = os.path.join(master_plot_save_folder, unwrapped_condition, 'Curvature_Epi')
                                fio.mkdir(saveplotfigurefolder)
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                plt.savefig(os.path.join(saveplotfigurefolder, 'curvature_Epi_'+unwrapped_condition + '_%s.png' %(str(frame_iiii+1).zfill(3))), bbox_inches='tight')
                                plt.show()



                                # 3. produce the MOSES flow. 
                                YY, XX = np.indices(vid_ve.shape[1:]) # create a normal grid. 

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

                                if frame_iiii < len(vid_ve)-1:
                                    #### VE
                                    multiply_factor=5. # this is much needed? 
    
                                    flow_ve_frame = flow_ve[frame_iiii].copy()
                                    # numerical errors? -> # can we get better by using a multiplication factor? 
                                    YY_next_ve = YY + multiply_factor*flow_ve_frame[...,1]; YY_next_ve = np.clip(np.rint(YY_next_ve), 0, flow_ve_frame.shape[0]-1) 
                                    XX_next_ve = XX + multiply_factor*flow_ve_frame[...,0]; XX_next_ve = np.clip(np.rint(XX_next_ve), 0, flow_ve_frame.shape[1]-1) 
                                    
                                    # quite a lot of errors? 
                                    flow_ve_3D_frame =  unwrap_params_ve[YY_next_ve.astype(np.int), XX_next_ve.astype(np.int)] - unwrap_params_ve[YY,XX]; flow_ve_3D_frame = flow_ve_3D_frame.astype(np.float32) 
                                    flow_ve_3D_frame = flow_ve_3D_frame / float(multiply_factor) * (voxel_size / float(Ts)) * temp_tform_scales[frame_iiii+1] # growth correction. 
                                    
                                
                                    # compute the AVE_dir component
                                    flow_ve_3D_frame_AVE_dir = np.sum(flow_ve_3D_frame*AVE_surf_dir_ve, axis=-1)
                                    flow_ve_3D_frame_AVE_dir_rot = sktform.rotate(flow_ve_3D_frame_AVE_dir, 
                                                                                     angle=(90-ve_angle_move), preserve_range=True, cval=0)
                                    flow_ve_3D_frame_AVE_dir_rot[clip_mask==0] = np.nan
                                    
                                else:
                                    flow_ve_3D_frame_AVE_dir_rot[:] = np.nan


                                plt.figure(figsize=(10,10))
                                # plt.subplot(121)
                                # plt.imshow(vid_ve[frame], cmap='gray')
                                plt.title(str(frame_iiii+1))
                                plt.imshow(flow_ve_3D_frame_AVE_dir_rot, vmin=-1, vmax=1, cmap='coolwarm')
#                                plt.imshow(vid_frame)

                                if 'MTMG-HEX' in embryo_folder:

                                    plt.plot(hex_im_binary_cont_largest_rot[:,1],
                                             hex_im_binary_cont_largest_rot[:,0],'g-', lw=3)
#                                    plt.plot(hex_im_binary_centroid_3D[1],
#                                             hex_im_binary_centroid_3D[0],'go')

                                    plt.plot(hex_im_binary_cont_largest_hi_rot[:,1],
                                             hex_im_binary_cont_largest_hi_rot[:,0],'--', lw=3, color='turquoise')
#                                    plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                             hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
                                if frame_iiii < len(polar_grid_time_contour_lines):
                                    
                                    polar_grid_lines_frame_ii = polar_grid_time_contour_lines[frame_iiii]
                                    for line in polar_grid_lines_frame_ii:
                                        line = rotate_pts(line[:,::-1], 
                                                          angle=-(90-ve_angle_move), 
                                                          center=(np.hstack(vid_ve[0].shape)/2.))[:,::-1]
                                        plt.plot(line[:,1], line[:,0], 'k', lw=2)
                                        
                                plt.axis('off')
                                plt.grid('off')
                                # save the visualisation in a plot. 
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                saveplotfigurefolder = os.path.join(master_plot_save_folder, unwrapped_condition, 'Flow_VE_AVE_dir')
                                fio.mkdir(saveplotfigurefolder)
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                plt.savefig(os.path.join(saveplotfigurefolder, 'flow_VE_AVE_dir_'+unwrapped_condition + '_%s.png' %(str(frame_iiii+1).zfill(3))), bbox_inches='tight')
                                plt.show()

                                if frame_iiii < len(vid_ve)-1:
                                    #### Epi
                                    flow_epi_frame = flow_epi_resize[frame_iiii].copy()
        #                            flow_epi_frame = smooth_flow_ds(flow_epi_frame, sigma=3, ds=4)
                                    YY_next_epi = YY + multiply_factor*flow_epi_frame[...,1]; YY_next_epi = np.clip(np.rint(YY_next_epi), 0, flow_epi_frame.shape[0]-1) 
                                    XX_next_epi = XX + multiply_factor*flow_epi_frame[...,0]; XX_next_epi = np.clip(np.rint(XX_next_epi), 0, flow_epi_frame.shape[1]-1)
                                    flow_epi_3D_frame =  unwrap_params_epi[YY_next_epi.astype(np.int), XX_next_epi.astype(np.int)] - unwrap_params_epi[YY,XX]; flow_epi_3D_frame = flow_epi_3D_frame.astype(np.float32)       
                                    flow_epi_3D_frame = flow_epi_3D_frame/float(multiply_factor) * (voxel_size / float(Ts)) * temp_tform_scales[frame_iiii+1]
    
    
                                    # compute the AVE_dir component
                                    flow_epi_3D_frame_AVE_dir = np.sum(flow_epi_3D_frame*AVE_surf_dir_epi, axis=-1)
                                    flow_epi_3D_frame_AVE_dir_rot = sktform.rotate(flow_epi_3D_frame_AVE_dir, 
                                                                                    angle=(90-ve_angle_move), preserve_range=True, cval=0)
                                    flow_epi_3D_frame_AVE_dir_rot[clip_mask==0] = np.nan
                                else:
                                    flow_epi_3D_frame_AVE_dir_rot[:] = np.nan

                                # save the frame # using the same metric bounds as the single cell statistics. 
                                plt.figure(figsize=(10,10))
                                # plt.subplot(121)
                                # plt.imshow(vid_ve[frame], cmap='gray')
                                plt.title(str(frame_iiii+1))
                                plt.imshow(flow_epi_3D_frame_AVE_dir_rot, vmin=-1, vmax=1, cmap='coolwarm')
#                                plt.imshow(vid_frame)

                                if 'MTMG-HEX' in embryo_folder:

                                    plt.plot(hex_im_binary_cont_largest_rot[:,1],
                                             hex_im_binary_cont_largest_rot[:,0],'g-', lw=3)
#                                    plt.plot(hex_im_binary_centroid_3D[1],
#                                             hex_im_binary_centroid_3D[0],'go')

                                    plt.plot(hex_im_binary_cont_largest_hi_rot[:,1],
                                             hex_im_binary_cont_largest_hi_rot[:,0],'--', lw=3, color='turquoise')
#                                    plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                             hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
                                if frame_iiii < len(polar_grid_time_contour_lines):
                                    
                                    polar_grid_lines_frame_ii = polar_grid_time_contour_lines[frame_iiii]
                                    for line in polar_grid_lines_frame_ii:
                                        line = rotate_pts(line[:,::-1], 
                                                          angle=-(90-ve_angle_move), 
                                                          center=(np.hstack(vid_ve[0].shape)/2.))[:,::-1]
                                        plt.plot(line[:,1], line[:,0], 'k', lw=2)
                                        
                                plt.axis('off')
                                plt.grid('off')
                                # save the visualisation in a plot. 
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                saveplotfigurefolder = os.path.join(master_plot_save_folder, unwrapped_condition, 'Flow_Epi_AVE_dir')
                                fio.mkdir(saveplotfigurefolder)
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                plt.savefig(os.path.join(saveplotfigurefolder, 'flow_Epi_AVE_dir_'+unwrapped_condition + '_%s.png' %(str(frame_iiii+1).zfill(3))), bbox_inches='tight')
                                plt.show()



                            # 4. load the demons deformations. 

                                # VE versions 

                                if iiii < len(np.arange(start_end_times[0], start_end_times[1], 1)) - 1: 
                                    """
                                    a) AVE-direction
                                    """
                                    demons_AVE_dir_frame = demons_ve_diff_dir_vect_AVE[cnt] * (voxel_size/float(Ts))
                                    # compute the AVE_dir component
                                    demons_AVE_dir_frame_rot = sktform.rotate(demons_AVE_dir_frame, 
                                                                                angle=(90-ve_angle_move), preserve_range=True, cval=0)
                                    demons_AVE_dir_frame_rot[clip_mask==0] = np.nan

                                    """
                                    b) AVE-perp-direction
                                    """
                                    demons_AVE_perp_dir_frame = demons_ve_diff_dir_vect_AVE_perp[cnt] * (voxel_size/float(Ts))
                                    demons_AVE_perp_dir_frame_rot = sktform.rotate(demons_AVE_perp_dir_frame, 
                                                                                angle=(90-ve_angle_move), preserve_range=True, cval=0)
                                    demons_AVE_perp_dir_frame_rot[clip_mask==0] = np.nan

                                    
                                    """
                                    c) Normal-direction
                                    """
                                    demons_norm_dir_frame = demons_ve_diff_dir_vect_AVE_norm[cnt] * (voxel_size/float(Ts))
                                    demons_norm_dir_frame_rot = sktform.rotate(demons_norm_dir_frame, 
                                                                                angle=(90-ve_angle_move), preserve_range=True, cval=0)
                                    demons_norm_dir_frame_rot[clip_mask==0] = np.nan
                                else:
                                    demons_AVE_dir_frame_rot[:] = np.nan
                                    demons_AVE_perp_dir_frame_rot[:] = np.nan
                                    demons_norm_dir_frame_rot[:]= np.nan


                                plt.figure(figsize=(10,10))
                                # plt.subplot(121)
                                # plt.imshow(vid_ve[frame], cmap='gray')
                                plt.title(str(frame_iiii+1))
                                plt.imshow(demons_AVE_dir_frame_rot, vmin=-.5, vmax=.5, cmap='coolwarm')
#                                plt.imshow(vid_frame)

                                if 'MTMG-HEX' in embryo_folder:

                                    plt.plot(hex_im_binary_cont_largest_rot[:,1],
                                             hex_im_binary_cont_largest_rot[:,0],'g-', lw=3)
#                                    plt.plot(hex_im_binary_centroid_3D[1],
#                                             hex_im_binary_centroid_3D[0],'go')

                                    plt.plot(hex_im_binary_cont_largest_hi_rot[:,1],
                                             hex_im_binary_cont_largest_hi_rot[:,0],'--', lw=3, color='turquoise')
#                                    plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                             hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
                                if frame_iiii < len(polar_grid_time_contour_lines):
                                    
                                    polar_grid_lines_frame_ii = polar_grid_time_contour_lines[frame_iiii]
                                    for line in polar_grid_lines_frame_ii:
                                        line = rotate_pts(line[:,::-1], 
                                                          angle=-(90-ve_angle_move), 
                                                          center=(np.hstack(vid_ve[0].shape)/2.))[:,::-1]
                                        plt.plot(line[:,1], line[:,0], 'k', lw=2)
                                        
                                plt.axis('off')
                                plt.grid('off')
                                # save the visualisation in a plot. 
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                saveplotfigurefolder = os.path.join(master_plot_save_folder, unwrapped_condition, 'Demons_VE_AVE_dir')
                                fio.mkdir(saveplotfigurefolder)
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                plt.savefig(os.path.join(saveplotfigurefolder, 'demons_VE_AVE_dir_'+unwrapped_condition + '_%s.png' %(str(frame_iiii+1).zfill(3))), bbox_inches='tight')
                                plt.show()


                                plt.figure(figsize=(10,10))
                                # plt.subplot(121)
                                # plt.imshow(vid_ve[frame], cmap='gray')
                                plt.title(str(frame_iiii+1))
                                plt.imshow(demons_AVE_perp_dir_frame_rot, vmin=-.5, vmax=.5, cmap='coolwarm')
#                                plt.imshow(vid_frame)

                                if 'MTMG-HEX' in embryo_folder:

                                    plt.plot(hex_im_binary_cont_largest_rot[:,1],
                                             hex_im_binary_cont_largest_rot[:,0],'g-', lw=3)
#                                    plt.plot(hex_im_binary_centroid_3D[1],
#                                             hex_im_binary_centroid_3D[0],'go')

                                    plt.plot(hex_im_binary_cont_largest_hi_rot[:,1],
                                             hex_im_binary_cont_largest_hi_rot[:,0],'--', lw=3, color='turquoise')
#                                    plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                             hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
                                if frame_iiii < len(polar_grid_time_contour_lines):
                                    
                                    polar_grid_lines_frame_ii = polar_grid_time_contour_lines[frame_iiii]
                                    for line in polar_grid_lines_frame_ii:
                                        line = rotate_pts(line[:,::-1], 
                                                          angle=-(90-ve_angle_move), 
                                                          center=(np.hstack(vid_ve[0].shape)/2.))[:,::-1]
                                        plt.plot(line[:,1], line[:,0], 'k', lw=2)
                                        
                                plt.axis('off')
                                plt.grid('off')
                                # save the visualisation in a plot. 
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                saveplotfigurefolder = os.path.join(master_plot_save_folder, unwrapped_condition, 'Demons_VE_AVE_perp_dir')
                                fio.mkdir(saveplotfigurefolder)
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                plt.savefig(os.path.join(saveplotfigurefolder, 'demons_VE_AVE_perp_dir_'+unwrapped_condition + '_%s.png' %(str(frame_iiii+1).zfill(3))), bbox_inches='tight')
                                plt.show()


                                plt.figure(figsize=(10,10))
                                # plt.subplot(121)
                                # plt.imshow(vid_ve[frame], cmap='gray')
                                plt.title(str(frame_iiii+1))
                                plt.imshow(demons_norm_dir_frame_rot, vmin=-.5, vmax=.5, cmap='coolwarm')
#                                plt.imshow(vid_frame)

                                if 'MTMG-HEX' in embryo_folder:

                                    plt.plot(hex_im_binary_cont_largest_rot[:,1],
                                             hex_im_binary_cont_largest_rot[:,0],'g-', lw=3)
#                                    plt.plot(hex_im_binary_centroid_3D[1],
#                                             hex_im_binary_centroid_3D[0],'go')

                                    plt.plot(hex_im_binary_cont_largest_hi_rot[:,1],
                                             hex_im_binary_cont_largest_hi_rot[:,0],'--', lw=3, color='turquoise')
#                                    plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                             hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
                                if frame_iiii < len(polar_grid_time_contour_lines):
                                    
                                    polar_grid_lines_frame_ii = polar_grid_time_contour_lines[frame_iiii]
                                    for line in polar_grid_lines_frame_ii:
                                        line = rotate_pts(line[:,::-1], 
                                                          angle=-(90-ve_angle_move), 
                                                          center=(np.hstack(vid_ve[0].shape)/2.))[:,::-1]
                                        plt.plot(line[:,1], line[:,0], 'k', lw=2)
                                        
                                plt.axis('off')
                                plt.grid('off')
                                # save the visualisation in a plot. 
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                saveplotfigurefolder = os.path.join(master_plot_save_folder, unwrapped_condition, 'Demons_VE_norm_dir')
                                fio.mkdir(saveplotfigurefolder)
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                plt.savefig(os.path.join(saveplotfigurefolder, 'demons_VE_norm_dir_'+unwrapped_condition + '_%s.png' %(str(frame_iiii+1).zfill(3))), bbox_inches='tight')
                                plt.show()
                                    

                                # get the Epi versions. 

                                if iiii < len(np.arange(start_end_times[0], start_end_times[1], 1)) - 1: 
                                    """
                                    a) AVE-direction
                                    """
                                    demons_AVE_dir_frame = demons_epi_diff_dir_vect_AVE[cnt] * (voxel_size/float(Ts))
                                    # compute the AVE_dir component
                                    demons_AVE_dir_frame_rot = sktform.rotate(demons_AVE_dir_frame, 
                                                                                angle=(90-ve_angle_move), preserve_range=True, cval=0)
                                    demons_AVE_dir_frame_rot[clip_mask==0] = np.nan

                                    """
                                    b) AVE-perp-direction
                                    """
                                    demons_AVE_perp_dir_frame = demons_epi_diff_dir_vect_AVE_perp[cnt] * (voxel_size/float(Ts))
                                    demons_AVE_perp_dir_frame_rot = sktform.rotate(demons_AVE_perp_dir_frame, 
                                                                                angle=(90-ve_angle_move), preserve_range=True, cval=0)
                                    demons_AVE_perp_dir_frame_rot[clip_mask==0] = np.nan

                                    
                                    """
                                    c) Normal-direction
                                    """
                                    demons_norm_dir_frame = demons_epi_diff_dir_vect_AVE_norm[cnt] * (voxel_size/float(Ts))
                                    demons_norm_dir_frame_rot = sktform.rotate(demons_norm_dir_frame, 
                                                                                angle=(90-ve_angle_move), preserve_range=True, cval=0)
                                    demons_norm_dir_frame_rot[clip_mask==0] = np.nan
                                else:
                                    demons_AVE_dir_frame_rot[:] = np.nan
                                    demons_AVE_perp_dir_frame_rot[:] = np.nan
                                    demons_norm_dir_frame_rot[:]= np.nan


                                plt.figure(figsize=(10,10))
                                # plt.subplot(121)
                                # plt.imshow(vid_ve[frame], cmap='gray')
                                plt.title(str(frame_iiii+1))
                                plt.imshow(demons_AVE_dir_frame_rot, vmin=-.5, vmax=.5, cmap='coolwarm')
#                                plt.imshow(vid_frame)

                                if 'MTMG-HEX' in embryo_folder:

                                    plt.plot(hex_im_binary_cont_largest_rot[:,1],
                                             hex_im_binary_cont_largest_rot[:,0],'g-', lw=3)
#                                    plt.plot(hex_im_binary_centroid_3D[1],
#                                             hex_im_binary_centroid_3D[0],'go')

                                    plt.plot(hex_im_binary_cont_largest_hi_rot[:,1],
                                             hex_im_binary_cont_largest_hi_rot[:,0],'--', lw=3, color='turquoise')
#                                    plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                             hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
                                if frame_iiii < len(polar_grid_time_contour_lines):
                                    
                                    polar_grid_lines_frame_ii = polar_grid_time_contour_lines[frame_iiii]
                                    for line in polar_grid_lines_frame_ii:
                                        line = rotate_pts(line[:,::-1], 
                                                          angle=-(90-ve_angle_move), 
                                                          center=(np.hstack(vid_ve[0].shape)/2.))[:,::-1]
                                        plt.plot(line[:,1], line[:,0], 'k', lw=2)
                                        
                                plt.axis('off')
                                plt.grid('off')
                                # save the visualisation in a plot. 
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                saveplotfigurefolder = os.path.join(master_plot_save_folder, unwrapped_condition, 'Demons_Epi_AVE_dir')
                                fio.mkdir(saveplotfigurefolder)
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                plt.savefig(os.path.join(saveplotfigurefolder, 'demons_Epi_AVE_dir_'+unwrapped_condition + '_%s.png' %(str(frame_iiii+1).zfill(3))), bbox_inches='tight')
                                plt.show()


                                plt.figure(figsize=(10,10))
                                # plt.subplot(121)
                                # plt.imshow(vid_ve[frame], cmap='gray')
                                plt.title(str(frame_iiii+1))
                                plt.imshow(demons_AVE_perp_dir_frame_rot, vmin=-.5, vmax=.5, cmap='coolwarm')
#                                plt.imshow(vid_frame)

                                if 'MTMG-HEX' in embryo_folder:

                                    plt.plot(hex_im_binary_cont_largest_rot[:,1],
                                             hex_im_binary_cont_largest_rot[:,0],'g-', lw=3)
#                                    plt.plot(hex_im_binary_centroid_3D[1],
#                                             hex_im_binary_centroid_3D[0],'go')

                                    plt.plot(hex_im_binary_cont_largest_hi_rot[:,1],
                                             hex_im_binary_cont_largest_hi_rot[:,0],'--', lw=3, color='turquoise')
#                                    plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                             hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
                                if frame_iiii < len(polar_grid_time_contour_lines):
                                    
                                    polar_grid_lines_frame_ii = polar_grid_time_contour_lines[frame_iiii]
                                    for line in polar_grid_lines_frame_ii:
                                        line = rotate_pts(line[:,::-1], 
                                                          angle=-(90-ve_angle_move), 
                                                          center=(np.hstack(vid_ve[0].shape)/2.))[:,::-1]
                                        plt.plot(line[:,1], line[:,0], 'k', lw=2)
                                        
                                plt.axis('off')
                                plt.grid('off')
                                # save the visualisation in a plot. 
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                saveplotfigurefolder = os.path.join(master_plot_save_folder, unwrapped_condition, 'Demons_Epi_AVE_perp_dir')
                                fio.mkdir(saveplotfigurefolder)
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                plt.savefig(os.path.join(saveplotfigurefolder, 'demons_Epi_AVE_perp_dir_'+unwrapped_condition + '_%s.png' %(str(frame_iiii+1).zfill(3))), bbox_inches='tight')
                                plt.show()


                                plt.figure(figsize=(10,10))
                                # plt.subplot(121)
                                # plt.imshow(vid_ve[frame], cmap='gray')
                                plt.title(str(frame_iiii+1))
                                plt.imshow(demons_norm_dir_frame_rot, vmin=-.5, vmax=.5, cmap='coolwarm')
#                                plt.imshow(vid_frame)

                                if 'MTMG-HEX' in embryo_folder:

                                    plt.plot(hex_im_binary_cont_largest_rot[:,1],
                                             hex_im_binary_cont_largest_rot[:,0],'g-', lw=3)
#                                    plt.plot(hex_im_binary_centroid_3D[1],
#                                             hex_im_binary_centroid_3D[0],'go')

                                    plt.plot(hex_im_binary_cont_largest_hi_rot[:,1],
                                             hex_im_binary_cont_largest_hi_rot[:,0],'--', lw=3, color='turquoise')
#                                    plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                             hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
                                if frame_iiii < len(polar_grid_time_contour_lines):
                                    
                                    polar_grid_lines_frame_ii = polar_grid_time_contour_lines[frame_iiii]
                                    for line in polar_grid_lines_frame_ii:
                                        line = rotate_pts(line[:,::-1], 
                                                          angle=-(90-ve_angle_move), 
                                                          center=(np.hstack(vid_ve[0].shape)/2.))[:,::-1]
                                        plt.plot(line[:,1], line[:,0], 'k', lw=2)
                                        
                                plt.axis('off')
                                plt.grid('off')
                                # save the visualisation in a plot. 
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                saveplotfigurefolder = os.path.join(master_plot_save_folder, unwrapped_condition, 'Demons_Epi_norm_dir')
                                fio.mkdir(saveplotfigurefolder)
                                # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
                                plt.savefig(os.path.join(saveplotfigurefolder, 'demons_Epi_norm_dir_'+unwrapped_condition + '_%s.png' %(str(frame_iiii+1).zfill(3))), bbox_inches='tight')
                                plt.show()


                                cnt += 1
                        
                        
                        
            """
            checking a kymograph
            """
#            curvature_VE_frame_rot = sktform.rotate(curvature_VE[frame_iiii], 
#                                                    angle=(90-ve_angle_move), preserve_range=True, cval=0)
#            curvature_Epi_frame_rot = sktform.rotate(curvature_Epi[frame_iiii], 
#                                                    angle=(90-ve_angle_move), preserve_range=True, cval=0)
#
#            curvature_VE_frame_rot[clip_mask==0] = np.nan
#            curvature_Epi_frame_rot[clip_mask==0] = np.nan
            
            curvature_VE_rot = np.array([sktform.rotate(curvature_VE[tt], 
                                               angle=(90-ve_angle_move), preserve_range=True, cval=0) for tt in np.arange(len(curvature_VE))])
            
            curvature_VE_rot[:,np.logical_not(clip_mask)] = np.nan
            mid_x = curvature_VE_rot.shape[2]//2
            
            fig, ax = plt.subplots(figsize=(15,15))
            ax.imshow(np.nanmean(curvature_VE_rot[:,:,mid_x-25:mid_x+25], axis=-1).T, cmap='coolwarm', vmin=-2e-4, vmax=2e-4)
#            ax.set_aspect('auto')
            plt.show()
            
            
            curvature_Epi_rot = np.array([sktform.rotate(curvature_Epi[tt], 
                                               angle=(90-ve_angle_move), preserve_range=True, cval=0) for tt in np.arange(len(curvature_Epi))])
            
            curvature_Epi_rot[:,np.logical_not(clip_mask)] = np.nan
            mid_x = curvature_Epi_rot.shape[2]//2
            
            fig, ax = plt.subplots(figsize=(15,15))
            ax.imshow(np.nanmean(curvature_Epi_rot[:,:,mid_x-25:mid_x+25], axis=-1).T, cmap='coolwarm', vmin=-2.5e-4, vmax=2.5e-4)
#            ax.set_aspect('auto')
            plt.show()
            
            
#            change_curvature_VE_rot = curvature_VE_rot - curvature_VE_rot[0][None,:]
            change_curvature_VE_rot = curvature_VE_rot - curvature_VE_rot[0][None,:]
            mid_x = change_curvature_VE_rot.shape[2]//2
            
            fig, ax = plt.subplots(figsize=(15,15))
            ax.imshow(np.nanmean(change_curvature_VE_rot[:,:,mid_x-25:mid_x+25], axis=-1).T, cmap='coolwarm', vmin=-1e-4, vmax=1e-4)
            plt.vlines(mig[0], 0, change_curvature_VE_rot.shape[1])
            plt.vlines(mig[1], 0, change_curvature_VE_rot.shape[1])
#            ax.set_aspect('auto')
            plt.show()
            
            
            change_curvature_Epi_rot = curvature_Epi_rot - curvature_Epi_rot[int(mig[0])][None,:]
#            change_curvature_Epi_rot = curvature_Epi_rot[1:] - curvature_Epi_rot[:-1]
            mid_x = curvature_Epi_rot.shape[2]//2
            
            fig, ax = plt.subplots(figsize=(15,15))
            ax.imshow(np.nanmean(change_curvature_Epi_rot[:,:,mid_x-25:mid_x+25], axis=-1).T, cmap='coolwarm', vmin=-1e-4, vmax=1e-4)
            plt.vlines(mig[0], 0, change_curvature_Epi_rot.shape[1])
            plt.vlines(mig[1], 0, change_curvature_Epi_rot.shape[1])
#            ax.set_aspect('auto')
            plt.show()
                        
                        

#                     for frame in range(start_end_times[0], start_end_times[1]):
                        
#                         # # retain only largest connected component as the labelled ids. 
#                         # connected_labels = postprocess_polar_labels(ve_tracks_full[:,start_end_times[0]], 
#                         #                                             ve_seg_select_rect, dist_thresh=2*spixel_size)
#                         rot_center = np.hstack(vid_ve[0].shape)/2.
#                         Geom_centers.append(rot_center)
                        
#                         """
#                         Epi line
#                         """
#                         epi_line = epi_contour_line[frame].copy() # this is (y,x)
#                         # epi_line = epi_line[:,::-1]
#                         epi_poly = polygon(epi_line[:,1],
#                                             epi_line[:,0],
#                                             vid_ve[0].shape)

#                         epi_poly_centroid_3D = lookup_3D_mean(np.vstack([epi_poly[0].ravel(),
#                                                                         epi_poly[1].ravel()]).T, 
#                                                                         unwrap_params_epi, force_z=np.max(unwrap_params_epi[...,0]))
#                         Epi_line_centroids.append(epi_poly_centroid_3D)


#                         # apply it now to the start time of interest
#                         ve_seg_track_pts = ve_tracks_full[connected_labels, frame]

#                         """ create the density mask for filtered """
#                         density_mask = np.zeros(vid_ve[0].shape)
#                         density_mask[ve_seg_track_pts[:,0],
#                                     ve_seg_track_pts[:,1]] = 1
#                         density_mask = gaussian(density_mask, 1*spixel_size)
                        
#                         # .5 for lifeact 
#                         density_mask_pts = density_mask > np.mean(density_mask) + 1.*np.std(density_mask) # produce the mask. 
#                         density_mask_pts = binary_fill_holes(density_mask_pts)

#                         AVE_classifier_vid_directional.append(density_mask_pts)

#                         # additionally compute the centroid and visualise? 
#                         cont_seg_0 = find_contours(density_mask_pts,0)[0]
#                         ave_area_poly = polygon(cont_seg_0[:,0],
#                                                 cont_seg_0[:,1],
#                                                 density_mask_pts.shape)
                        
#                         ave_centroid_3D = lookup_3D_mean(np.vstack([ave_area_poly[0],
#                                                                     ave_area_poly[1]]).T, 
#                                                                     unwrap_params_ve)
# #                        AVE_classifier_vid_directional_centroids.append(ave_centroid_3D)                           

#                         """ create the density mask for nonfiltered SVM """
#                         # ve_seg_track_pts_svm = ve_tracks_full[ve_seg_select_rect_svm,:][:,0]
#                         ve_seg_track_pts_svm = ve_tracks_full[connected_labels_svm,frame]
#                         density_mask = np.zeros(vid_ve[0].shape)
#                         density_mask[ve_seg_track_pts_svm[:,0],
#                                     ve_seg_track_pts_svm[:,1]] = 1
#                         density_mask = gaussian(density_mask, 1*spixel_size)
                        
#                         # .5 for lifeact 
#                         density_mask_pts_SVM = density_mask > np.mean(density_mask) + 1.*np.std(density_mask) # produce the mask. 
#                         density_mask_pts_SVM = binary_fill_holes(density_mask_pts_SVM)
                    
#                         AVE_classifier_vid_directional_SVM.append(density_mask_pts_SVM)

#                         # additionally compute the centroid and visualise? 
#                         cont_seg_ve = find_contours(density_mask_pts_SVM,0)[0]
#                         ave_area_poly = polygon(cont_seg_ve[:,0],
#                                                 cont_seg_ve[:,1],
#                                                 density_mask_pts_SVM.shape)
                        
#                         ave_centroid_3D_SVM = lookup_3D_mean(np.vstack([ave_area_poly[0],
#                                                                         ave_area_poly[1]]).T, 
#                                                                         unwrap_params_ve)
# #                        AVE_classifier_vid_directional_SVM_centroids.append(ave_centroid_3D_SVM) 
# #                    
# #                     cont_seg_svm_0 = find_contours(density_mask_pts_SVM,0)[0]

#                         """
#                         (0,0) to centroid to Epi line distance with correction for distance. 
#                         """
# #                          epi_line_resample = resample_curve(epi_line[:,1], epi_line[:,0], n_samples=720+1)
#                         epi_line_resample = resample_curve(epi_line[:,1], epi_line[:,0], n_samples=720+1)

#                         # dist_AVE_centroid, dist_AVE_centroid_epi, AVE_epi_point = measure_geodesic_distances_centroid_epi(ave_centroid_3D, 
#                         #                                                                                                     rot_center, 
#                         #                                                                                                     epi_line_resample, 
#                         #                                                                                                     unwrap_params_ve)

#                         # dist_AVE_centroid_SVM, dist_AVE_centroid_epi_SVM, AVE_epi_point_SVM = measure_geodesic_distances_centroid_epi(ave_centroid_3D_SVM, 
#                         #                                                                                                     rot_center, 
#                         #                                                                                                     epi_line_resample, 
#                         #                                                                                                     unwrap_params_ve)


#                         # using interpolation in 3D space.
#                         dist_AVE_centroid, dist_AVE_centroid_epi, AVE_epi_point = measure_geodesic_distances_centroid_3D(ave_centroid_3D, 
#                                                                                     rot_center, 
#                                                                                     epi_line_resample, 
#                                                                                     unwrap_params_ve, 
#                                                                                     surf_nbrs_ve, surf_nbrs_ve_3D_pts, 
#                                                                                     n_samples=10)

#                         dist_AVE_centroid_SVM, dist_AVE_centroid_epi_SVM, AVE_epi_point_SVM = measure_geodesic_distances_centroid_3D(ave_centroid_3D_SVM, 
#                                                                                                                         rot_center, 
#                                                                                                                         epi_line_resample, 
#                                                                                                                         unwrap_params_ve, 
#                                                                                                                         surf_nbrs_ve, surf_nbrs_ve_3D_pts, 
#                                                                                                                         n_samples=10)
                        

#                         center_AVE_dists.append(dist_AVE_centroid[1])
#                         center_AVE_SVM_dists.append(dist_AVE_centroid_SVM[1])
                        
#                         AVE_epi_dists.append(dist_AVE_centroid_epi[1])
#                         AVE_SVM_epi_dists.append(dist_AVE_centroid_epi_SVM[1])
                        
# #                        Epi_line_pts_AVE.append(AVE_epi_point)
# #                        Epi_line_pts_AVE_SVM.append(AVE_epi_point_SVM)
                        
#                         # """
#                         # Do the same but now for the hex region when applicable. 
#                         # """
#                         if 'MTMG-HEX' in embryo_folder:

#                             # segment with 0 as threshold 
#                             hex_im = np.uint8(255*rescale_intensity(equalize_hist(vid_hex[frame])))
                            
#                             # parse and smoothen the output a bit more. 
#                             hex_im_binary = hex_im > 0 
#                             hex_im_binary = skmorph.binary_closing(hex_im_binary, skmorph.disk(3))
#                             hex_im_binary = binary_fill_holes(hex_im_binary)
#                             hex_im_binary = skmorph.remove_small_objects(hex_im_binary, 250)

# #                            Hex_segmentation_vid_thresh0.append(hex_im_binary)
                            
#                             hex_im_binary_cont = find_contours(hex_im_binary, 0)
#                             hex_im_binary_cont_largest = hex_im_binary_cont[np.argmax([len(l) for l in hex_im_binary_cont])]
                            
#                             hex_im_binary_cont_polygon = polygon(hex_im_binary_cont_largest[:,0],
#                                                                 hex_im_binary_cont_largest[:,1],
#                                                                 hex_im.shape)
                            
# #                            hex_im_binary_cont_polygon_mask = np.zeros_like(hex_im_binary)
# #                            Hex_segmentation_vid_thresh0.append(hex_im_binary_cont_polygon>0)
#                             Hex_segmentation_vid_thresh0.append(hex_im_binary)
                    
#                             # hex_im_binary_centroid = np.mean(hex_im_binary_cont_largest, axis=0)
#                             hex_im_binary_centroid_3D = lookup_3D_mean(np.vstack([hex_im_binary_cont_polygon[0].ravel(),
#                                                                                 hex_im_binary_cont_polygon[1].ravel()]).T, 
#                                                                     unwrap_params_ve)

#                             # Hex_segmentation_vid_thresh0_centroids.append(hex_im_binary_centroid_3D)

                            
#                             """
#                             Repeating the analysis for Hex segmentation with higher threshold
#                             """
#     #                        hex_im_binary_hi = hex_im >= np.mean(hex_im) + 2*np.std(hex_im)
#                             hex_im_binary_hi = hex_im >= threshold_otsu(hex_im)
#                             hex_im_binary_hi = skmorph.binary_closing(hex_im_binary_hi, skmorph.disk(3))
#                             hex_im_binary_hi = binary_fill_holes(hex_im_binary_hi)
#                             hex_im_binary_hi = skmorph.remove_small_objects(hex_im_binary_hi, 250)
#                             hex_im_binary_hi = gaussian(hex_im_binary_hi, sigma=3, preserve_range=True)
#                             hex_im_binary_hi = hex_im_binary_hi > 0.1
                            
#                             Hex_segmentation_vid_threshhi.append(hex_im_binary_hi)
                            
#                             hex_im_binary_cont_hi = find_contours(hex_im_binary_hi, 0)
#                             hex_im_binary_cont_largest_hi = hex_im_binary_cont_hi[np.argmax([len(l) for l in hex_im_binary_cont_hi])]
                            
#                             hex_im_binary_cont_polygon_hi = polygon(hex_im_binary_cont_largest_hi[:,0],
#                                                                 hex_im_binary_cont_largest_hi[:,1],
#                                                                 hex_im.shape)
                            
#                             # Hex_segmentation_vid_threshhi.append(hex_im_binary_cont_polygon_hi>0)

#                             # hex_im_binary_centroid = np.mean(hex_im_binary_cont_largest, axis=0)
#                             hex_im_binary_centroid_3D_hi = lookup_3D_mean(np.vstack([hex_im_binary_cont_polygon_hi[0].ravel(),
#                                                                                 hex_im_binary_cont_polygon_hi[1].ravel()]).T, 
#                                                                     unwrap_params_ve)

# #                            Hex_segmentation_vid_threshhi_centroids.append(hex_im_binary_centroid_3D_hi)
#                             # dist_Hex_centroid, dist_Hex_centroid_epi, Hex_epi_point = measure_geodesic_distances_centroid_epi(hex_im_binary_centroid_3D, 
#                             #                                                                                                     rot_center, 
#                             #                                                                                                     epi_line_resample, 
#                             #                                                                                                     unwrap_params_ve)
#                             # dist_Hex_centroid_hi, dist_Hex_centroid_epi_hi, Hex_epi_point_hi = measure_geodesic_distances_centroid_epi(hex_im_binary_centroid_3D_hi, 
#                             #                                                                                                             rot_center, 
#                             #                                                                                                             epi_line_resample, 
#                             #                                                                                                             unwrap_params_ve)

#                             dist_Hex_centroid, dist_Hex_centroid_epi, Hex_epi_point = measure_geodesic_distances_centroid_3D(hex_im_binary_centroid_3D, 
#                                                                                                                         rot_center, 
#                                                                                                                         epi_line_resample, 
#                                                                                                                         unwrap_params_ve, 
#                                                                                                                         surf_nbrs_ve, surf_nbrs_ve_3D_pts, 
#                                                                                                                         n_samples=10)

#                             dist_Hex_centroid_hi, dist_Hex_centroid_epi_hi, Hex_epi_point_hi = measure_geodesic_distances_centroid_3D(hex_im_binary_centroid_3D_hi, 
#                                                                                                                         rot_center, 
#                                                                                                                         epi_line_resample, 
#                                                                                                                         unwrap_params_ve, 
#                                                                                                                         surf_nbrs_ve, surf_nbrs_ve_3D_pts, 
#                                                                                                                         n_samples=10)

#                             center_Hex_dists.append(dist_Hex_centroid[1])
#                             center_Hex_dists_hi.append(dist_Hex_centroid_hi[1])

#                             Hex_epi_dists.append(dist_Hex_centroid_epi[1])
#                             Hex_epi_dists_hi.append(dist_Hex_centroid_epi_hi[1])
                            
#                             # Epi_line_pts_Hex.append(Hex_epi_point)
#                             # Epi_line_pts_Hex_hi.append(Hex_epi_point_hi)


#                         clip_mask = np.zeros(vid_ve[frame].shape)
#                         XX, YY = np.meshgrid(range(clip_mask.shape[1]), range(clip_mask.shape[0]))
#                         clip_mask = np.sqrt((XX-clip_mask.shape[1]/2.)**2 + (YY-clip_mask.shape[0]/2.)**2)
#                         clip_mask = clip_mask<= vid_ve[frame].shape[0]/2./1.1; clip_mask = np.logical_not(clip_mask)


#                         # set up drawings.
#                         # rot_center = np.hstack(vid_ve[0].shape)/2.

#                         epi_line = rotate_pts(epi_line[:,:], angle=-(90-ve_angle_move), center=rot_center)[:,:] ####### this is already x,y!
#                         epi_poly_centroid_3D = rotate_pts(epi_poly_centroid_3D[::-1][None,:], angle=-(90-ve_angle_move), center=rot_center).ravel()[::-1]

#                         # epi_line_resample = rotate_pts(epi_line_resample[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]
                        
#                         cont_seg_0 = rotate_pts(cont_seg_0[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]
#                         cont_seg_ve = rotate_pts(cont_seg_ve[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]
#                         ave_centroid_3D = rotate_pts(ave_centroid_3D[::-1][None,:], angle=-(90-ve_angle_move), center=rot_center).ravel()[::-1]
#                         ave_centroid_3D_SVM = rotate_pts(ave_centroid_3D_SVM[::-1][None,:], angle=-(90-ve_angle_move), center=rot_center).ravel()[::-1]
#                         AVE_epi_point = rotate_pts(AVE_epi_point[::-1][None,:], angle=-(90-ve_angle_move), center=rot_center).ravel()[::-1]
#                         AVE_epi_point_SVM = rotate_pts(AVE_epi_point_SVM[::-1][None,:], angle=-(90-ve_angle_move), center=rot_center).ravel()[::-1]
                        
                         
#                         Epi_line_pts_AVE.append(AVE_epi_point)
#                         Epi_line_pts_AVE_SVM.append(AVE_epi_point_SVM)

#                         AVE_classifier_vid_directional_centroids.append(ave_centroid_3D)
#                         AVE_classifier_vid_directional_SVM_centroids.append(ave_centroid_3D_SVM) 
                    

#                         if 'MTMG-HEX' in embryo_folder:

#                             hex_im_binary_cont_largest = rotate_pts(hex_im_binary_cont_largest[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]
#                             hex_im_binary_cont_largest_hi = rotate_pts(hex_im_binary_cont_largest_hi[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]
#                             hex_im_binary_centroid_3D = rotate_pts(hex_im_binary_centroid_3D[::-1][None,:], angle=-(90-ve_angle_move), center=rot_center).ravel()[::-1]
#                             hex_im_binary_centroid_3D_hi = rotate_pts(hex_im_binary_centroid_3D_hi[::-1][None,:], angle=-(90-ve_angle_move), center=rot_center).ravel()[::-1]

#                             Hex_segmentation_vid_thresh0_centroids.append(hex_im_binary_centroid_3D)
#                             Hex_segmentation_vid_threshhi_centroids.append(hex_im_binary_centroid_3D_hi)

#                             # additionally rotate these.
#                             Hex_epi_point = rotate_pts(Hex_epi_point[::-1][None,:], angle=-(90-ve_angle_move), center=rot_center).ravel()[::-1]
#                             Hex_epi_point_hi = rotate_pts(Hex_epi_point_hi[::-1][None,:], angle=-(90-ve_angle_move), center=rot_center).ravel()[::-1]
                        
#                             Epi_line_pts_Hex.append(Hex_epi_point)
#                             Epi_line_pts_Hex_hi.append(Hex_epi_point_hi)

#                         """
#                         visualisation 1: the AVE classifier case without showing Hex channel.
#                         """
#                         plotsavefolder_AVE = os.path.join(master_plot_save_folder, 
#                                                         os.path.split(embryo_folder)[-1],
#                                                         unwrapped_condition,
#                                                         'AVE_classifier')
#                         # fio.mkdir(plotsavefolder_AVE)

#                         plotsavefolder_Hex = os.path.join(master_plot_save_folder, 
#                                                         os.path.split(embryo_folder)[-1],
#                                                         unwrapped_condition,
#                                                         'Hex_segmentation')
#                         # fio.mkdir(plotsavefolder_Hex)
                        

#                         # for visualisation we should rotate to the consensus VE angle. 
#                         plt.figure(figsize=(10,10))
#                         # plt.subplot(121)
#                         # plt.imshow(vid_ve[frame], cmap='gray')
#                         plt.title(str(frame+1))
#                         vid_frame = sktform.rotate(vid_ve[frame], 
#                                                     angle=(90-ve_angle_move), preserve_range=True, cval=0)
#                         vid_frame = vid_frame/float(vid_frame.max())
#                         vid_frame [ clip_mask] = np.nan

#                         plt.imshow( vid_frame, cmap='gray')

#                         # if 'MTMG-HEX' in embryo_folder:
#                         #     hex_im = hex_im / float( hex_im.max() )
#                         #     hex_im[hex_im==0] = np.nan
#                         #     # plt.imshow(hex_im, cmap='Greens', alpha=0.5)
#                         #     plt.imshow(sktform.rotate(hex_im, 
#                         #                             angle=(90-ve_angle_move), preserve_range=True, cval=0), cmap='Greens', alpha=0.5)

#                         plt.plot(cont_seg_0[:,1], cont_seg_0[:,0], 'w-', lw=3); plt.plot(ave_centroid_3D[1], ave_centroid_3D[0], 'wo')
#                         plt.plot(cont_seg_ve[:,1], cont_seg_ve[:,0], 'r--', lw=3); plt.plot(ave_centroid_3D_SVM[1], ave_centroid_3D_SVM[0], 'ro')
#                         plt.plot(epi_line[:,0], epi_line[:,1], 'y-', lw=3) 
#                         plt.plot(epi_poly_centroid_3D[1], epi_poly_centroid_3D[0], 'yo', zorder=100)
#                         # plt.plot(epi_line_resample[:,1], epi_line_resample[:,0], 'w.')

#                         plt.plot([ rot_center[1], AVE_epi_point[1]], 
#                                  [ rot_center[0], AVE_epi_point[0]], 'w-')
#                         # add the centroid of the epi line.
#                         if 'MTMG-HEX' in embryo_folder:

#                             plt.plot(hex_im_binary_cont_largest[:,1],
#                                     hex_im_binary_cont_largest[:,0],'g-', lw=3)
#                             plt.plot(hex_im_binary_centroid_3D[1],
#                                     hex_im_binary_centroid_3D[0],'go')

#                             plt.plot(hex_im_binary_cont_largest_hi[:,1],
#                                     hex_im_binary_cont_largest_hi[:,0],'--', lw=3, color='turquoise')
#                             plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                     hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
#                         plt.axis('off')
#                         plt.grid('off')
#                         # save the visualisation in a plot. 
#                         # plt.savefig(os.path.join(plotsavefolder_AVE, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
#                         plt.show()


#                         if 'MTMG-HEX' in embryo_folder:
#                             # for visualisation we should rotate to the consensus VE angle. 
#                             plt.figure(figsize=(10,10))
#                             # plt.subplot(121)
#                             # plt.imshow(vid_ve[frame], cmap='gray')
#                             plt.title(str(frame+1))
#                             hex_im = hex_im / float( hex_im.max() )
#                             vid_frame = sktform.rotate(hex_im ,
#                                                         angle=(90-ve_angle_move), preserve_range=True, cval=0)
#                             vid_frame = vid_frame/float(vid_frame.max())
#                             vid_frame [ clip_mask] = np.nan

#                             overlay_im = np.zeros((vid_frame.shape[0], vid_frame.shape[1], 3))
#                             overlay_im[...,1] = vid_frame

#                             plt.imshow(overlay_im)

#                             plt.plot(cont_seg_0[:,1], cont_seg_0[:,0], 'w-', lw=3); plt.plot(ave_centroid_3D[1], ave_centroid_3D[0], 'wo')
#                             plt.plot(cont_seg_ve[:,1], cont_seg_ve[:,0], 'r--', lw=3); plt.plot(ave_centroid_3D_SVM[1], ave_centroid_3D_SVM[0], 'ro')
#                             plt.plot(epi_line[:,0], epi_line[:,1], 'y-', lw=3) 
#                             plt.plot(epi_poly_centroid_3D[1], epi_poly_centroid_3D[0], 'yo', zorder=100)
#                             # plt.plot(epi_line_resample[:,1], epi_line_resample[:,0], 'w.')

#                             plt.plot([ rot_center[1], AVE_epi_point[1]], 
#                                     [ rot_center[0], AVE_epi_point[0]], 'w-')
#                             # add the centroid of the epi line.
#                             if 'MTMG-HEX' in embryo_folder:

#                                 plt.plot(hex_im_binary_cont_largest[:,1],
#                                         hex_im_binary_cont_largest[:,0],'g-', lw=3)
#                                 plt.plot(hex_im_binary_centroid_3D[1],
#                                         hex_im_binary_centroid_3D[0],'go')

#                                 plt.plot(hex_im_binary_cont_largest_hi[:,1],
#                                         hex_im_binary_cont_largest_hi[:,0],'--', lw=3, color='turquoise')
#                                 plt.plot(hex_im_binary_centroid_3D_hi[1],
#                                         hex_im_binary_centroid_3D_hi[0],'o', color='turquoise')
#                             plt.axis('off')
#                             plt.grid('off')
#                             # save the visualisation in a plot. 
#                             # plt.savefig(os.path.join(plotsavefolder_Hex, unwrapped_condition + '_%s.png' %(str(frame+1).zfill(3))), bbox_inches='tight')
#                             plt.show()


#                     # """
#                     # CSV folder save
#                     # """
#                     # plotsavefolder_CSV = os.path.join(master_plot_save_folder, 
#                     #                                   'CSV',
#                     #                                     os.path.split(embryo_folder)[-1])
#                     # fio.mkdir(plotsavefolder_CSV)
                        
#                     """
#                     Save arrays for future useage. 
#                     """
#                     Geom_centers = np.vstack(Geom_centers)
                    
#                     AVE_classifier_vid_directional = np.array(AVE_classifier_vid_directional)
#                     AVE_classifier_vid_directional_SVM = np.array(AVE_classifier_vid_directional_SVM)

#                     if 'MTMG-HEX' in embryo_folder:
#                         Hex_segmentation_vid_thresh0 = np.array(Hex_segmentation_vid_thresh0)
#                         Hex_segmentation_vid_threshhi = np.array(Hex_segmentation_vid_threshhi)

#                     AVE_classifier_vid_directional_centroids = np.vstack(AVE_classifier_vid_directional_centroids)
#                     AVE_classifier_vid_directional_SVM_centroids = np.vstack(AVE_classifier_vid_directional_SVM_centroids)
#                     Epi_line_centroids = np.vstack(Epi_line_centroids)

#                     center_AVE_dists = np.hstack(center_AVE_dists)
#                     center_AVE_SVM_dists = np.hstack(center_AVE_SVM_dists)
#                     AVE_epi_dists = np.hstack(AVE_epi_dists)
#                     AVE_SVM_epi_dists = np.hstack(AVE_SVM_epi_dists)
#                     Epi_line_pts_AVE = np.vstack(Epi_line_pts_AVE)
#                     Epi_line_pts_AVE_SVM = np.vstack(Epi_line_pts_AVE_SVM)

#                     if 'MTMG-HEX' in embryo_folder:
#                         Hex_segmentation_vid_thresh0_centroids = np.vstack(Hex_segmentation_vid_thresh0_centroids)
#                         Hex_segmentation_vid_threshhi_centroids = np.vstack(Hex_segmentation_vid_threshhi_centroids)
                    
#                         center_Hex_dists = np.hstack(center_Hex_dists)
#                         center_Hex_dists_hi = np.hstack(center_Hex_dists_hi)
#                         Hex_epi_dists = np.hstack(Hex_epi_dists)
#                         Hex_epi_dists_hi = np.hstack(Hex_epi_dists_hi)
#                         Epi_line_pts_Hex = np.vstack(Epi_line_pts_Hex)
#                         Epi_line_pts_Hex_hi = np.vstack(Epi_line_pts_Hex_hi)

#                     # skio.imsave(os.path.join(save_ave_classifier_folder, 
#                     #                         'AVE_classifier_directional-'+unwrapped_condition+'.tif'),
#                     #                         np.uint8(255*AVE_classifier_vid_directional))
#                     # skio.imsave(os.path.join(save_ave_classifier_folder, 
#                     #                         'AVE_classifier_directional_SVM-'+unwrapped_condition+'.tif'),
#                     #                         np.uint8(255*AVE_classifier_vid_directional_SVM))
                    
#                     # dist_norm_factor = np.ones(len(vid_ve))*(np.prod(embryo_im_shape)**(1./3))


#                     # # save the centroids.
#                     # centroid_table = pd.DataFrame(np.hstack([Geom_centers,
#                     #                                          AVE_classifier_vid_directional_centroids,
#                     #                                         AVE_classifier_vid_directional_SVM_centroids,
#                     #                                         Epi_line_centroids, 
#                     #                                         center_AVE_dists[:,None],
#                     #                                         center_AVE_SVM_dists[:,None],
#                     #                                         AVE_epi_dists[:,None],
#                     #                                         AVE_SVM_epi_dists[:,None],
#                     #                                         Epi_line_pts_AVE,
#                     #                                         Epi_line_pts_AVE_SVM, 
#                     #                                         temp_tform_scales[:,None],
#                     #                                         dist_norm_factor[:,None]]),
#                     #                                         index=None,
#                     #                                         columns=['geom_centre_y', 'geom_centre_x',
#                     #                                                  'directional_y', 'directional_x',
#                     #                                                  'classifier_y', 'classifier_x',
#                     #                                                  'epi_y', 'epi_x',
#                     #                                                  'dist_AVE_directional', 
#                     #                                                  'dist_AVE_SVM', 
#                     #                                                  'epi_dist_AVE_directional', 
#                     #                                                  'epi_dist_AVE_SVM', 
#                     #                                                  'epi_line_AVE_directional_y',
#                     #                                                  'epi_line_AVE_directional_x',
#                     #                                                  'epi_line_AVE_SVM_y',
#                     #                                                  'epi_line_AVE_SVM_x',
#                     #                                                  'growth_correction', 
#                     #                                                  'distance_normalisation_factor'])
#                     # centroid_table.to_csv(os.path.join(save_ave_classifier_folder, 
#                     #                         'AVE_classifier_centroids_and_distances-'+unwrapped_condition+'.csv'), index=None)

#                     # centroid_table.to_csv(os.path.join(plotsavefolder_CSV, 
#                     #                         'AVE_classifier_centroids_and_distances-'+unwrapped_condition+'.csv'), index=None)

#                     # if 'MTMG-HEX' in embryo_folder:
#                     #     skio.imsave(os.path.join(save_hex_seg_folder, 
#                     #                         'Hex_segmentation_thresh0-'+unwrapped_condition+'.tif'),
#                     #                         np.uint8(255*Hex_segmentation_vid_thresh0))
#                     #     skio.imsave(os.path.join(save_hex_seg_folder, 
#                     #                         'Hex_segmentation_threshhi-'+unwrapped_condition+'.tif'),
#                     #                         np.uint8(255*Hex_segmentation_vid_threshhi))
                    
#                     #     centroid_table = pd.DataFrame(
#                     #                                   np.hstack([Geom_centers,
#                     #                                           Hex_segmentation_vid_thresh0_centroids,
#                     #                                         Hex_segmentation_vid_threshhi_centroids,
#                     #                                         Epi_line_centroids,
#                     #                                         center_Hex_dists[:,None],
#                     #                                         center_Hex_dists_hi[:,None],
#                     #                                         Hex_epi_dists[:,None],
#                     #                                         Hex_epi_dists_hi[:,None],
#                     #                                         Epi_line_pts_Hex,
#                     #                                         Epi_line_pts_Hex_hi,
#                     #                                         temp_tform_scales[:,None],
#                     #                                         dist_norm_factor[:,None]]),
#                     #                                         index=None,
#                     #                                         columns=['geom_centre_y', 'geom_centre_x',
#                     #                                                  'directional_y', 'directional_x',
#                     #                                                  'classifier_y', 'classifier_x',
#                     #                                                  'epi_y', 'epi_x',
#                     #                                                  'dist_Hex_thresh0', 
#                     #                                                  'dist_Hex_threshhi', 
#                     #                                                  'epi_dist_Hex_thresh0', 
#                     #                                                  'epi_dist_Hex_threshhi', 
#                     #                                                  'epi_line_Hex_thresh0_y',
#                     #                                                  'epi_line_Hex_thresh0_x',
#                     #                                                  'epi_line_Hex_threshhi_y',
#                     #                                                  'epi_line_Hex_threshhi_x',
#                     #                                                  'growth_correction', 
#                     #                                                  'distance_normalisation_factor'])
#                     #     centroid_table.to_csv(os.path.join(save_hex_seg_folder, 
#                     #                         'Hex_segmentation_centroids_and_distances-'+unwrapped_condition+'.csv'), index=None)

#                     #     # add save to csv folder.
#                     #     centroid_table.to_csv(os.path.join(plotsavefolder_CSV, 
#                     #                                        'Hex_segmentation_centroids_and_distances-'+unwrapped_condition+'.csv'), index=None)







#                     # use this as the proper one. 
#                     ve_seg_select_rect = density_mask_pts[meantracks_ve[:,0,0], 
#                                                           meantracks_ve[:,0,1]]
                    
# #                    fig, ax = plt.subplots(figsize=(15,15))
# #                    plt.imshow(vid_ve[start_end_times[0]], cmap='gray')
# #                    plt.imshow(polar_grid, cmap='coolwarm', alpha=0.5)
# #                    plot_tracks(meantracks_ve[:,:10], ax=ax, color='g')
# #                    plot_tracks(meantracks_epi[:,:10], ax=ax, color='r')
# #                    plot_tracks(meantracks_ve[ve_seg_select_rect,:10], ax=ax, color='w')
# #                    plt.show()
                    
                    
#                     # =============================================================================
#                     #      Rotate all points and masks etc for visualization.               
#                     # =============================================================================
#                     from skimage.measure import find_contours
                    
#                     cont_seg_0 = find_contours(density_mask_pts,0)[0]
#                     cont_seg_ve = find_contours(ve_seg_select_rect_valid,0)[0]
                    
#                     cont_seg_svm_0 = find_contours(density_mask_pts_SVM,0)[0]
#                     cont_seg_svm_ve = find_contours(ve_seg_select_rect_valid_svm,0)[0]
                    
#                     rot_center = np.hstack(vid_ve[0].shape)/2.
                    
#                     # rotate the points. 
#                     rotated_cont_seg_0 = rotate_pts(cont_seg_0[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]
#                     rotated_cont_seg_ve = rotate_pts(cont_seg_ve[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]
#                     rotated_cont_seg_svm_0 = rotate_pts(cont_seg_svm_0[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]
#                     rotated_cont_seg_ve = rotate_pts(cont_seg_svm_ve[:,::-1], angle=-(90-ve_angle_move), center=rot_center)[:,::-1]
# #                    rotated_tracks_epi = rotate_tracks(aligned_in_epi_track, angle=-(90-ve_angle_move), center=rot_center)
        
# #                    rotated_motion_center = rotate_pts(mean_pos[::-1][None,:], angle=-correct_angle, center=rot_center).ravel()[::-1]
#                     rot_density_mask_pts = sktform.rotate(density_mask_pts, angle=(90-ve_angle_move), preserve_range=True)
                    
#                     rotated_cont_seg_0_center = np.mean(rotated_cont_seg_0, axis=0)
                
#                     # =============================================================================
#                     #       Here is the visualisation part.               
#                     # =============================================================================
                    
#                     plotsavefolder = os.path.join(master_plot_save_folder, os.path.split(embryo_folder)[-1])
#                     fio.mkdir(plotsavefolder)
                    
                    
#                     if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
                        
#                         disp_im = np.dstack([vid_ve[0], 
#                                              vid_ve[0], 
#                                              vid_ve[0]])
#                         grid_im = mark_boundaries(disp_im, polar_grid, mode='thick')
#                         grid_im = sktform.rotate(grid_im, angle=(90-ve_angle_move), preserve_range=True)

#                         grid_center = np.hstack([vid_hex[0].shape[0]/2.,
#                                                  vid_hex[0].shape[1]/2.])

#                         """
#                         Additional calculations for distance ->, refine computations to 3D projected means etc. 
#                         """
#                         ave_area_poly = polygon(rotated_cont_seg_0[:,0],
#                                                 rotated_cont_seg_0[:,1],
#                                                 grid_im.shape)
                        
#                         ave_centroid_3D = lookup_3D_mean(np.vstack([ave_area_poly[0],
#                                                                     ave_area_poly[1]]).T, 
#                                                          unwrap_params_ve)
                        

#                         # add geodesic distnace from (0,0)
#                         distline_ave_3D, dist_ave_3D = geodesic_distance(np.hstack([grid_im.shape[0]/2.,grid_im.shape[1]/2.]), 
#                                                                         ave_centroid_3D, 
#                                                                         unwrap_params_ve, n_samples = 10)

#                         disp_ave_3D = distline_ave_3D[-1] - distline_ave_3D[0]
#                         disp_ave_3D = disp_ave_3D / float(np.linalg.norm(disp_ave_3D) + 1e-8)
#                         direction_ave_3D_2D = lookup_3D_to_2D_directions(unwrap_params_ve, grid_center, 
#                                                                         disp_ave_3D)
                    
                        
#                         plt.figure(figsize=(10,10))
# #                        plt.imshow(vid_ve[0], cmap='gray')
#                         plt.imshow(grid_im)
# #                        plt.imshow(polar_grid, cmap='coolwarm', alpha=0.5)
#                         plt.plot(rotated_cont_seg_0[:,1], 
#                                  rotated_cont_seg_0[:,0], 'w-', lw=5)
#                         # plt.plot(rotated_cont_seg_0_center[1],
#                                 #  rotated_cont_seg_0_center[0], 'wo', ms=15)
#                         plt.plot(ave_centroid_3D[1],
#                                  ave_centroid_3D[0], 'wo', ms=15)
#                         plt.plot(rotated_cont_seg_svm_0[:,1], 
#                                  rotated_cont_seg_svm_0[:,0], 'r--', lw=5)

#                         plt.plot(grid_im.shape[1]/2.,
#                                  grid_im.shape[0]/2., 'yo', mec='w', mew=3, ms=15)
#                         plt.axis('off')
#                         plt.grid('off')
#                         plt.savefig(os.path.join(plotsavefolder, 'overlay_seg_' + unwrapped_condition+'.svg'), bbox_inches='tight')
#                         # add saving. 
                        
#                         plt.show()

#                         """
#                         Save a bunch of statistics for quantification.
#                         """
#                         # statistics to export
#                         all_directions_regions_ave.append(direction_ave_3D_2D[0])
#                         all_directions_regions_hex.append(np.hstack([np.nan, np.nan]))
#                         all_centroids_Hex.append([np.nan, np.nan])
#                         all_centroids_Hex_hi.append([np.nan, np.nan])
#                         all_centroids_AVE_classifier.append(ave_centroid_3D)
#                         all_distance_Hex.append(np.nan)   
#                         all_distance_Hex_hi.append(np.nan)
#                         all_distance_AVE.append(dist_ave_3D)

#                         all_analysed_embryos.append(unwrapped_condition)
#                         all_grids.append(polar_grid)
#                         all_embryo_shapes.append(embryo_im_shape)
#                         all_ve_angles.append(float(90-ve_angle_move))
#                         all_plot_centres.append(grid_center)
                            
#                     else:
                        
#                         # use unwrap_params_ve  to get the mean 3D and remap.
#                         grid_center = np.hstack([vid_hex[0].shape[0]/2.,
#                                                  vid_hex[0].shape[1]/2.])
                        
#                         hex_im = np.uint8(255*rescale_intensity(equalize_hist(vid_hex[0])))
#                         hex_im = sktform.rotate(hex_im, angle=(90-ve_angle_move), preserve_range=True)
                        
#                         # parse and smoothen the output a bit more. 
#                         hex_im_binary = hex_im > 0 
#                         hex_im_binary = skmorph.binary_closing(hex_im_binary, skmorph.disk(3))
#                         hex_im_binary = binary_fill_holes(hex_im_binary)
#                         hex_im_binary = skmorph.remove_small_objects(hex_im_binary, 250)
                        
                        
#                         hex_im_binary_cont = find_contours(hex_im_binary, 0)
#                         hex_im_binary_cont_largest = hex_im_binary_cont[np.argmax([len(l) for l in hex_im_binary_cont])]
                        
                        
#                         hex_im_binary_cont_polygon = polygon(hex_im_binary_cont_largest[:,0],
#                                                              hex_im_binary_cont_largest[:,1],
#                                                              hex_im.shape)
                        
#                         # hex_im_binary_centroid = np.mean(hex_im_binary_cont_largest, axis=0)
#                         hex_im_binary_centroid_3D = lookup_3D_mean(np.vstack([hex_im_binary_cont_polygon[0].ravel(),
#                                                                               hex_im_binary_cont_polygon[1].ravel()]).T, 
#                                                                    unwrap_params_ve)

#                         # add geodesic distnace from (0,0)
#                         distline_hex_3D, dist_hex_3D = geodesic_distance(grid_center, 
#                                                                             hex_im_binary_centroid_3D, 
#                                                                             unwrap_params_ve, n_samples = 20)

#                         disp_hex_3D = distline_hex_3D[-1] - distline_hex_3D[0]
#                         disp_hex_3D = disp_hex_3D / float(np.linalg.norm(disp_hex_3D) + 1e-8)
#                         direction_hex_3D_2D = lookup_3D_to_2D_directions(unwrap_params_ve, grid_center, 
#                                                                         disp_hex_3D)
                        
#                         """
#                         Repeating the analysis for Hex segmentation with higher threshold
#                         """
# #                        hex_im_binary_hi = hex_im >= np.mean(hex_im) + 2*np.std(hex_im)
#                         hex_im_binary_hi = hex_im >= threshold_otsu(hex_im)
#                         hex_im_binary_hi = skmorph.binary_closing(hex_im_binary_hi, skmorph.disk(3))
#                         hex_im_binary_hi = binary_fill_holes(hex_im_binary_hi)
#                         hex_im_binary_hi = skmorph.remove_small_objects(hex_im_binary_hi, 250)
#                         hex_im_binary_hi = gaussian(hex_im_binary_hi, sigma=3, preserve_range=True)
#                         hex_im_binary_hi = hex_im_binary_hi > 0.1
                        
#                         hex_im_binary_cont_hi = find_contours(hex_im_binary_hi, 0)
#                         hex_im_binary_cont_largest_hi = hex_im_binary_cont_hi[np.argmax([len(l) for l in hex_im_binary_cont_hi])]
                        
                        
#                         hex_im_binary_cont_polygon_hi = polygon(hex_im_binary_cont_largest_hi[:,0],
#                                                              hex_im_binary_cont_largest_hi[:,1],
#                                                              hex_im.shape)
                        
#                         # hex_im_binary_centroid = np.mean(hex_im_binary_cont_largest, axis=0)
#                         hex_im_binary_centroid_3D_hi = lookup_3D_mean(np.vstack([hex_im_binary_cont_polygon_hi[0].ravel(),
#                                                                               hex_im_binary_cont_polygon_hi[1].ravel()]).T, 
#                                                                    unwrap_params_ve)

#                         # add geodesic distnace from (0,0)
#                         distline_hex_3D_hi, dist_hex_3D_hi = geodesic_distance(grid_center, 
#                                                                             hex_im_binary_centroid_3D_hi, 
#                                                                             unwrap_params_ve, n_samples = 20)

                        
#                         plt.figure(figsize=(10,10))
# #                        plt.imshow(vid_ve[0], cmap='gray')
#                         disp_im = np.dstack([np.zeros_like(vid_hex[0]), 
#                                              np.uint8(255*rescale_intensity(equalize_hist(vid_hex[0]))), 
#                                              np.zeros_like(vid_hex[0])])
#                         grid_im = mark_boundaries(disp_im, polar_grid, mode='thick')
                        
#                         # rotate image to align with AVE direction. 
                        
#                         grid_im = sktform.rotate(grid_im, angle=(90-ve_angle_move), preserve_range=True)
                        
                        
#                         ave_area_poly = polygon(rotated_cont_seg_0[:,0],
#                                                 rotated_cont_seg_0[:,1],
#                                                 hex_im.shape)
                        
#                         ave_centroid_3D = lookup_3D_mean(np.vstack([ave_area_poly[0],
#                                                                     ave_area_poly[1]]).T, 
#                                                          unwrap_params_ve)
                        

#                         # add geodesic distnace from (0,0)
#                         distline_ave_3D, dist_ave_3D = geodesic_distance(np.hstack([grid_im.shape[0]/2.,grid_im.shape[1]/2.]), 
#                                                                         ave_centroid_3D, 
#                                                                         unwrap_params_ve, n_samples = 20)

#                         disp_ave_3D = distline_ave_3D[-1] - distline_ave_3D[0]
#                         disp_ave_3D = disp_ave_3D / float(np.linalg.norm(disp_ave_3D) + 1e-8)
#                         direction_ave_3D_2D = lookup_3D_to_2D_directions(unwrap_params_ve, grid_center, 
#                                                                         disp_ave_3D)

#                         plt.imshow(grid_im)
#                         plt.plot(rotated_cont_seg_0[:,1], 
#                                  rotated_cont_seg_0[:,0], 'w-', lw=5)
#                         # plt.plot(rotated_cont_seg_0_center[1],
#                                 #  rotated_cont_seg_0_center[0], 'wo', ms=15)
#                         plt.plot(ave_centroid_3D[1],
#                                  ave_centroid_3D[0], 'wo', ms=15)
                        
#                         plt.plot(rotated_cont_seg_svm_0[:,1], 
#                                  rotated_cont_seg_svm_0[:,0], 'r--', lw=5)
# #                        plt.plot(rotated_cont_seg_svm_0[:,1], 
# #                                 rotated_cont_seg_ve[:,0], 'r-', lw=3)
# #                        plt.imshow(polar_grid, cmap='coolwarm', alpha=0.5)
#                         # add saving. 
#                         plt.plot(hex_im_binary_cont_largest[:,1],
#                                  hex_im_binary_cont_largest[:,0], 'g-', lw=5)
#                         # plt.plot(hex_im_binary_centroid[1], 
#                                 #  hex_im_binary_centroid[0], 'go', mec='w', ms=15)
#                         plt.plot(hex_im_binary_centroid_3D[1], 
#                                  hex_im_binary_centroid_3D[0], 'go', mew=3, mec='w', ms=15)
                        
#                         plt.plot(hex_im_binary_cont_largest_hi[:,1],
#                                  hex_im_binary_cont_largest_hi[:,0], '--', color='turquoise', lw=5)
#                         # plt.plot(hex_im_binary_centroid[1], 
#                                 #  hex_im_binary_centroid[0], 'go', mec='w', ms=15)
#                         plt.plot(hex_im_binary_centroid_3D_hi[1], 
#                                  hex_im_binary_centroid_3D_hi[0], 'o', color='turquoise', mew=3, mec='w', ms=15)
                        
#                         plt.plot(grid_im.shape[1]/2.,
#                                  grid_im.shape[0]/2., 'yo', mec='w', mew=3, ms=15)
#                         plt.axis('off')
#                         plt.grid('off')
#                         plt.savefig(os.path.join(plotsavefolder, 'overlay_seg_' + unwrapped_condition+'.svg'), bbox_inches='tight')
                        
#                         plt.show()
                
#                         # fig = plt.figure()
#                         # ax = fig.gca(projection='3d')  
#                         # ax.scatter(unwrap_params_ve[::50,
#                         #                             ::50,0], 
#                         #            unwrap_params_ve[::50,
#                         #                             ::50,1],
#                         #            unwrap_params_ve[::50,
#                         #                             ::50,2], color='r')
#                         # ax.plot(distline_ave_3D[:,0], 
#                         #         distline_ave_3D[:,1],
#                         #         distline_ave_3D[:,2], 'g')
#                         # ax.plot(distline_hex_3D[:,0], 
#                         #         distline_hex_3D[:,1],
#                         #         distline_hex_3D[:,2], 'r')
#                         # plt.show()
                        
#                         """
#                         Save a bunch of statistics for quantification.
#                         """
#                         all_analysed_embryos.append(unwrapped_condition)
#                         all_grids.append(polar_grid)
#                         all_embryo_shapes.append(embryo_im_shape)
#                         all_ve_angles.append(float(90-ve_angle_move))
                        
#                         # statistics to export
#                         all_directions_regions_ave.append(direction_ave_3D_2D[0])
#                         all_directions_regions_hex.append(direction_hex_3D_2D[0])
#                         all_centroids_Hex.append(hex_im_binary_centroid_3D)
#                         all_centroids_Hex_hi.append(hex_im_binary_centroid_3D_hi)
#                         all_centroids_AVE_classifier.append(ave_centroid_3D)
#                         all_distance_Hex.append(dist_hex_3D)   
#                         all_distance_Hex_hi.append(dist_hex_3D_hi)   
#                         all_distance_AVE.append(dist_ave_3D)
#                         all_plot_centres.append(grid_center)
                        
# # =============================================================================
# #       project tracks into 3D.
# #=============================================================================

# # compile .csv file 
#     import scipy.io as spio
    
#     matfile = os.path.join(master_plot_save_folder, 
#                            'all_segmentation_hex_ave_statistics.mat')
    
#     spio.savemat(matfile, 
#                  {'embryos':all_analysed_embryos, 
#                   'grids':all_grids,
#                   'vol_shapes':all_embryo_shapes,
#                   've_angles':all_ve_angles,
#                   'all_directions_regions_ave':all_directions_regions_ave, 
#                   'all_directions_regions_hex':all_directions_regions_hex,
#                   'all_centroids_Hex':all_centroids_Hex,
#                   'all_centroids_Hex_hi':all_centroids_Hex_hi,
#                   'all_centroids_AVE':all_centroids_AVE_classifier,
#                   'all_distance_Hex':all_distance_Hex,
#                   'all_distance_Hex_hi':all_distance_Hex_hi,
#                   'all_distance_AVE':all_distance_AVE,
#                   'all_plot_centres':all_plot_centres })
    
    
#     # separately get the centroid statistics.
#     import pandas as pd
    
#     distance_norm_factor = np.vstack(all_embryo_shapes)
#     distance_norm_factor = np.power(np.product(distance_norm_factor, axis=-1), 1./3)
    
#     data = np.hstack([np.hstack(all_analysed_embryos)[:,None], 
#                       np.vstack(all_plot_centres), 
#                       np.vstack(all_centroids_AVE_classifier),
#                       np.vstack(all_centroids_Hex), 
#                       np.vstack(all_centroids_Hex_hi),
#                       np.hstack(all_distance_AVE)[:,None],
#                       np.hstack(all_distance_Hex)[:,None],
#                       np.hstack(all_distance_Hex_hi)[:,None],
#                       distance_norm_factor[:,None]])
    
#     data_columns = ['Embryo', 
#                     'Grid_Centre_y', 
#                     'Grid_Centre_x', 
#                     'AVE_Centre_y',
#                     'AVE_Centre_x',
#                     'Hex_Centre_y',
#                     'Hex_Centre_x', 
#                     'Hex_Hi_Centre_y',
#                     'Hex_Hi_Centre_x',
#                     'Geodesic_Distance_AVE', 
#                     'Geodesic_Distance_Hex',
#                     'Geodesic_Distance_Hex_Hi',
#                     'Distance_Normalisation_Distance']
    
#     res_centroid_table = pd.DataFrame(data, index=None,
#                                       columns=data_columns)
    
#     savecsvfile = os.path.join(master_plot_save_folder, 
#                            'all_segmentation_hex_ave_centroid_statistics.csv')
    
#     res_centroid_table.to_csv(savecsvfile, index=None)










##                    # =============================================================================
##                    #       project tracks into 3D.
##                    # =============================================================================
##                        meantracks_ve_3D = unwrap_params_ve[meantracks_ve[...,0], meantracks_ve[...,1]]
##                        meantracks_epi_3D = unwrap_params_epi[meantracks_epi[...,0], meantracks_epi[...,1]]
##                        
##                        disps_ve_3D = meantracks_ve_3D[:,1:] - meantracks_ve_3D[:,:-1]; disps_ve_3D = disps_ve_3D.astype(np.float32)
##                        disps_epi_3D = meantracks_epi_3D[:,1:] - meantracks_epi_3D[:,:-1]; disps_epi_3D = disps_epi_3D.astype(np.float32)
##                        
###                        norm_disps = True
###                        
###                        # normalize the measurements.... based on the overall size of the embryo? approximated by the image crop size. 
###                        if norm_disps:
###                            disps_ve_3D[...,0] = disps_ve_3D[...,0]/embryo_im_shape[0]
###                            disps_ve_3D[...,1] = disps_ve_3D[...,1]/embryo_im_shape[1]
###                            disps_ve_3D[...,2] = disps_ve_3D[...,2]/embryo_im_shape[2]
###                            
###                            disps_epi_3D[...,0] = disps_epi_3D[...,0]/embryo_im_shape[0]
###                            disps_epi_3D[...,1] = disps_epi_3D[...,1]/embryo_im_shape[1]
###                            disps_epi_3D[...,2] = disps_epi_3D[...,2]/embryo_im_shape[2]
###                        
##                        disps_ve_2D = meantracks_ve[:,1:] - meantracks_ve[:,:-1]; disps_ve_2D = disps_ve_2D.astype(np.float32)
##                        disps_epi_2D = meantracks_epi[:,1:] - meantracks_epi[:,:-1]; disps_epi_2D = disps_epi_2D.astype(np.float32)
##                        
##                        norm_disps = True
##                        vol_factor = np.product(embryo_im_shape) ** (1./3)
##                        
##                        start_ = start_end_times[0]
##                        end_ = start_end_times[1]
##                        
##                        # this is only for the wrongly extracted auto stage? 
##                        if end_ == len(vid_ve):
##                            end_ = end_ - 1
##                        
##                        # growth and volume factor correction 
##                        disps_ve_3D = disps_ve_3D / float(vol_factor) * temp_tform_scales[start_:end_][None,:,None]
##                        disps_epi_3D = disps_epi_3D / float(vol_factor) * temp_tform_scales[start_:end_][None,:,None]
##                        
##                        # this was used for direction ....
##                        disps_ve_2D = disps_ve_2D * temp_tform_scales[start_:end_][None,:,None]
##                        disps_epi_2D = disps_epi_2D * temp_tform_scales[start_:end_][None,:,None]
##                        
##                        
##                        if 'LifeAct' in embryo_folder:
##                            # scale because of the temporal timing. 
##                            print('scale', embryo_folder)
##                            disps_ve_3D = disps_ve_3D*2.
##                            disps_epi_3D = disps_epi_3D*2.
###                            pass
##                        else:
##                            # temporal scaling.
##                            print(embryo_folder)
##                        
##                        all_embryo_vol_factors.append(vol_factor)
##                        
##                        
##                        
##                        # average the 2D only to extract directions. -> we don't even need this? -> we perform lookup....
##                        disps_ve_2D = disps_ve_2D.mean(axis=1)
##                        disps_epi_2D = disps_epi_2D.mean(axis=1) 
##                        
###                        ve_pts_2D_t0 = meantracks_ve[:,0]
###                        epi_pts_2D_t0 = 
##                        
##                    # =============================================================================
##                    #   Extract motion statistics.           
##                    # =============================================================================
##                    
#                        """
#                        compute the movement in the direction of AVE 
#                        """
#                        # note some error here. -> not the inferred ve direction. 
#                        disps_ve_3D_component = np.hstack([(disps_ve_3D[ll].mean(axis=0)).dot(mean_ve_direction_3D) for ll in range(len(disps_ve_3D))])
#                        
##                        mean_epi_direction_3D = (disps_epi_2D[ve_seg_select_rect].mean(axis=1)).mean(axis=0)
##                        mean_epi_direction_3D = mean_epi_direction_3D/(np.linalg.norm(mean_epi_direction_3D) + 1e-8) 
#                        disps_epi_3D_component = np.hstack([(disps_epi_3D[ll].mean(axis=0)).dot(mean_ve_direction_3D) for ll in range(len(disps_epi_3D))])
#                        
#                        """
#                        compute the total speed (distance moved over time)
#                        """
#                        speed_ve_3D = np.linalg.norm(disps_ve_3D, axis=-1)
#                        speed_epi_3D = np.linalg.norm(disps_epi_3D, axis=-1)
#                        
#                        # what sort of parameters other than speed? (directionality?) in 3D? (mean speed calc is buggy)
#                        
#                        """
#                        collection of stats over the polar regions of interest. -> use np.median instead of np.mean to aggregate over region for robustness. 
#                        """
#                        mean_speed_regions_ve = [np.linalg.norm(np.median(np.mean(disps_ve_3D[reg], axis=1), axis=0)) for reg in uniq_polar_spixels]
#                        mean_speed_regions_epi = [np.linalg.norm(np.median(np.mean(disps_epi_3D[reg], axis=1), axis=0)) for reg in uniq_polar_spixels]
##                        mean_speed_regions_ve = [np.mean(np.mean(np.linalg.norm(disps_ve_3D[reg], axis=-1), axis=1)) for reg in uniq_polar_spixels]
##                        mean_speed_regions_epi = [np.mean(np.mean(np.linalg.norm(disps_epi_3D[reg], axis=-1), axis=1)) for reg in uniq_polar_spixels]
#                        
#                        mean_total_speed_regions_ve = [np.median(np.mean(np.linalg.norm(disps_ve_3D[reg], axis=-1), axis=1)) for reg in uniq_polar_spixels]
#                        mean_total_speed_regions_epi = [np.median(np.mean(np.linalg.norm(disps_epi_3D[reg], axis=-1), axis=1)) for reg in uniq_polar_spixels]
#                        
##                        # version 1 code: using just the 2D. 
##                        mean_direction_regions_ve = [np.arctan2(np.mean(disps_ve_2D[reg], axis=0)[...,1], np.mean(disps_ve_2D[reg], axis=0)[...,0]) for reg in uniq_polar_spixels]
##                        mean_direction_regions_epi = [np.arctan2(np.mean(disps_epi_2D[reg], axis=0)[...,1], np.mean(disps_epi_2D[reg], axis=0)[...,0]) for reg in uniq_polar_spixels]
##                        
##                        # test the direction inference. 
##                        mean_disps_3D = np.median(np.mean(disps_ve_3D[uniq_polar_spixels[0]], axis=1), axis=0); mean_disps_3D = mean_disps_3D / float(np.linalg.norm(mean_disps_3D))
##                        mean_pos_2D = np.mean(meantracks_ve[uniq_polar_spixels[0],0], axis=0)
##                        
##                        direction_2D = lookup_3D_to_2D_directions(unwrap_params_ve, 
##                                                                  mean_pos_2D, 
##                                                                  mean_disps_3D)
##                        
##                        plt.figure()
##                        plt.imshow(polar_grid, cmap='coolwarm')
##                        plt.plot(mean_pos_2D[1], 
##                                 mean_pos_2D[0], 'o')
##                        plt.plot(mean_pos_2D[1]+10*direction_2D[0][1], 
##                                 mean_pos_2D[0]+10*direction_2D[0][0], 's')
##                        plt.show()
#                        
#                        # version 2 code: using just the 3D. 
#                        mean_direction_regions_ve = [lookup_3D_to_2D_directions(unwrap_params_ve, 
#                                                                                np.mean(meantracks_ve[reg,0], axis=0), 
#                                                                                np.median(np.mean(disps_ve_3D[reg], axis=1), axis=0), scale=10.)[1] for reg in uniq_polar_spixels]
#                        mean_direction_regions_epi = [lookup_3D_to_2D_directions(unwrap_params_epi, 
#                                                                                 np.mean(meantracks_epi[reg,0], axis=0), 
#                                                                                 np.median(np.mean(disps_epi_3D[reg], axis=1), axis=0), scale=10.)[1] for reg in uniq_polar_spixels]
#                        
#                        # allows us to also save the mean 3D velocities !! -> can be used for better combination? over the embryos? 
#                        
#                        # account for directionality and speed. 
#                        mean_speed_regions_ve_component = [np.median(disps_ve_3D_component[reg]) for reg in uniq_polar_spixels]
#                        mean_speed_regions_epi_component = [np.median(disps_epi_3D_component[reg]) for reg in uniq_polar_spixels]
#                        
#                        
#                        # mapping the speed onto the polar gridding. 
#                        speed_map_ve_3D = np.zeros(polar_grid.shape, dtype=np.float32)
#                        speed_map_epi_3D = np.zeros(polar_grid.shape, dtype=np.float32)
#                        speed_map_ve_3D_component = np.zeros(polar_grid.shape, dtype=np.float32)
#                        speed_map_epi_3D_component = np.zeros(polar_grid.shape, dtype=np.float32)
#                        
#                        pos_X, pos_Y = np.meshgrid(range(polar_grid.shape[1]), range(polar_grid.shape[0]))
#                        map_center_X = [np.mean(pos_X[polar_grid == uniq_polar_regions[kk]]) for kk in range(len(uniq_polar_regions))]
#                        map_center_Y = [np.mean(pos_Y[polar_grid == uniq_polar_regions[kk]]) for kk in range(len(uniq_polar_regions))]
#                        
#                        for kk in range(len(uniq_polar_regions)):
#                            speed_map_ve_3D[polar_grid == uniq_polar_regions[kk]] = mean_speed_regions_ve[kk] 
#                            speed_map_epi_3D[polar_grid == uniq_polar_regions[kk]] = mean_speed_regions_epi[kk] 
#                            
#                            speed_map_ve_3D_component[polar_grid == uniq_polar_regions[kk]] = mean_speed_regions_ve_component[kk] 
#                            speed_map_epi_3D_component[polar_grid == uniq_polar_regions[kk]] = mean_speed_regions_epi_component[kk] 
#                            
#                        fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(15,15))
#                        ax[0,0].imshow(speed_map_ve_3D, cmap='coolwarm')
#                        ax[0,0].quiver(map_center_X, map_center_Y, 
#                                      30*np.sin(mean_direction_regions_ve), -30*np.cos(mean_direction_regions_ve), color='k')
#                        ax[0,0].quiver(map_center_X, map_center_Y, 
#                                      50*np.sin((ve_angle_move+90)/180*np.pi), -50*np.cos((ve_angle_move+90)/180*np.pi), color='r')
#                        
#                        
#                        ax[0,1].imshow(speed_map_epi_3D, cmap='coolwarm')
#                        ax[0,1].quiver(map_center_X, map_center_Y, 
#                                      30*np.sin(mean_direction_regions_epi), -30*np.cos(mean_direction_regions_epi), color='k')
#                        ax[0,1].quiver(map_center_X, map_center_Y, 
#                                      50*np.sin((ve_angle_move+90)/180*np.pi), -50*np.cos((ve_angle_move+90)/180*np.pi), color='r')
#                        ax[1,0].imshow(speed_map_ve_3D_component, cmap='coolwarm', vmin=-5e-3,  vmax=5e-3)
#                        ax[1,1].imshow(speed_map_epi_3D_component, cmap='coolwarm', vmin=-1e-3,  vmax=1e-3)
#                        plt.show()
#                        
##                        mean_directionality_regions = [np.linalg.norm(np.mean(disps_ve_3D[reg], axis=1)) for reg in uniq_polar_spixels] # tabulate this relative to the VE migration direction.
#                        # =============================================================================
#                        #   Save the statistics.                
#                        # =============================================================================
#                        all_ve_angles.append(ve_angle_move)
#                        all_analysed_embryos.append('-'.join(proj_condition))
#                        all_grids.append(polar_grid)
#                        all_speeds_regions_ve.append(mean_speed_regions_ve)
#                        all_speeds_regions_epi.append(mean_speed_regions_epi)
#                        all_speeds_regions_ve_proj_ve.append(mean_speed_regions_ve_component)
#                        all_speeds_regions_epi_proj_ve.append(mean_speed_regions_epi_component)
#                        
#                        all_total_speed_regions_ve.append(mean_total_speed_regions_ve)
#                        all_total_speed_regions_epi.append(mean_total_speed_regions_epi)
#                        
#                        all_embryo_shapes.append(embryo_im_shape)
#                    
#                        all_directions_regions_ve.append(mean_direction_regions_ve)
#                        all_directions_regions_epi.append(mean_direction_regions_epi)
    
#    
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
#    all_directions_regions_ve_relative_mean = [np.median( all_speeds_regions_ve * np.sin(all_directions_regions_ve_relative), axis=0), 
#                                               np.median( all_speeds_regions_ve * np.cos(all_directions_regions_ve_relative), axis=0)]
##    all_directions_regions_ve_relative_mean_ang = np.arctan2(all_directions_regions_ve_relative_mean[1], 
##                                                             all_directions_regions_ve_relative_mean[0])
#    all_directions_regions_epi_relative_mean = [np.median( all_speeds_regions_epi * np.sin(all_directions_regions_epi_relative), axis=0), 
#                                                np.median( all_speeds_regions_epi * np.cos(all_directions_regions_epi_relative), axis=0)]
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
##     Analysis
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
## =============================================================================
##     export a canonical grid. 
## =============================================================================
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
#    plt.suptitle('Radial Grid used for Analysis Grid')
##    plt.hlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.vlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.quiver(reg_centroids[:,1], reg_centroids[:,0],
##               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[0], 
##               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[1], color='g', scale=.0075, units='xy')
##    plt.quiver(reg_centroids[:,1], reg_centroids[:,0], 
##               50*np.sin(all_directions_regions_ve_relative_mean[0]), -50*np.cos(all_directions_regions_ve_relative_mean[1]))
#    for jj in range(len(reg_centroids)):
#        plt.text(reg_centroids[jj,1], 
#                 reg_centroids[jj,0],
#                 str(jj+1), horizontalalignment='center',
#                 verticalalignment='center', fontsize= 18, color='k', fontname='Liberation Sans')
#    ax.imshow(sktform.rotate(polar_grid_region, angle=rotate_angle, preserve_range=True).astype(np.int), cmap='coolwarm' )
#    plt.axis('off')
#    fig.savefig(os.path.join(master_analysis_save_folder, 'polar_grid_map_for_analysis.svg'), dpi=300, bbox_inches='tight')
#        
#        
#        
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


