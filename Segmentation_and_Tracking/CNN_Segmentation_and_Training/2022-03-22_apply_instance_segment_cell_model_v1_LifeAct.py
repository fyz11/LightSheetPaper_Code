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


def tile_uniform_windows_radial_guided_line(imsize, n_r, n_theta, max_r, mid_line, center=None, bound_r=True):
    
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
    theta_bounds = np.linspace(-np.pi, np.pi, n_theta+1)
    
    for ii in range(len(theta_bounds)-1):
        mask_theta = np.logical_and( theta>=theta_bounds[ii], theta <= theta_bounds[ii+1])
        angle_masks_list.append(mask_theta)
        
    """
    construct the final set of masks.  
    """  
    spixels = np.zeros((m,n), dtype=np.int)
    
    counter = 1
    for ii in range(len(all_dist_masks)):
        for jj in range(len(angle_masks_list)):
            mask = np.logical_and(all_dist_masks[ii], angle_masks_list[jj])

            spixels[mask] = counter 
            counter += 1
        
    return spixels 

# from tensorflow import keras 
from keras.layers import Layer
from keras.initializers import Constant
import tensorflow as tf
import keras.backend as K
eps = 1e-7

class customFocal_loss_categorical_directionality(Layer):
    def __init__(self, gamma=2, alpha=0.75, nb_outputs=3, nb_tasks=2, losses=None, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.is_placeholder = True
        self.nb_outputs = nb_outputs
        self.nb_tasks = nb_tasks
        self.loss_types = losses 
        super(customFocal_loss_categorical_directionality, self).__init__(**kwargs)

    def weighted_softmax_focal_loss(self, y_true, y_pred, weights):
        
#        weights = K.exp(-self.log_vars) # create the precision matrix. 
        y_pred *= weights
        
        # the rest is normal softmax
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon) # predictions are clipped. 

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate Focal Loss
        loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy
        # Sum the losses in mini_batch, and should this not be mean over the -1.
        return K.sum(loss, axis=-1) ####### previously over the 1 axis
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        # add different initialisations for different types of losses. 
        for i in range(self.nb_tasks):
            if self.loss_types[i] == 'categorical':
                self.log_vars+=[self.add_weight(name='log_vars'+str(i), shape=(self.nb_outputs,),
                                                      initializer=Constant([0. for ii in range(self.nb_outputs)]), trainable=True)]
            if self.loss_types[i] == 'L1':
                self.log_vars+=[self.add_weight(name='log_vars'+str(i), shape=(1,),
                                                      initializer=Constant(0.), trainable=True)]

        super(customFocal_loss_categorical_directionality, self).build(input_shape)
        
    def multi_loss(self, ys_true, ys_pred):       
        loss = 0
        for ii, (y_true, y_pred, log_var) in enumerate(zip(ys_true, ys_pred, self.log_vars)):
            precision = K.exp(-log_var[0]) # 1/sigma**2 this gives positivity.
            if self.loss_types[ii] == 'categorical':
                print(ii, 'categorical')
#                loss += K.mean(self.weighted_softmax_focal_loss(y_true, y_pred, precision))
                loss += self.weighted_softmax_focal_loss(y_true, y_pred, precision)
            if self.loss_types[ii] == 'L1':
                print(ii, 'L1')
                mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
#                y_pred = K.clip(y_pred, -1 + K.epsilon(), 1. - K.epsilon()) #### clip the predictions.
                # prevent division by 0 .
                masked_mae = K.sum(mask_true* precision * K.abs(y_true - y_pred) + log_var[0], axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
                loss += masked_mae #K.mean(masked_mae)
#                loss += K.mean(K.sum(precision * K.abs((y_true - y_pred))  + log_var[0], -1))
        return K.mean(loss)
            
    def call(self, inputs):
        # parse the list input. 
        ys_true = inputs[:self.nb_tasks]
        ys_pred = inputs[self.nb_tasks:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)


def softmax_3d(class_dim=-1):
    """ 3D extension of softmax, class is last dim"""
    import keras.backend as K
    def activation(x):
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=class_dim, keepdims=True))
            s = K.sum(e, axis=class_dim, keepdims=True)
            return e / s
        else:
            raise Exception('Cannot apply softmax to a tensor that is not 2D or '
                            '3D. Here, ndim=' + str(ndim))
    return activation

class customFocal_loss_mse_directionality(Layer):
    def __init__(self, nb_outputs=3, nb_tasks=2, losses=None, **kwargs):
        self.is_placeholder = True
        self.nb_outputs = nb_outputs
        self.nb_tasks = nb_tasks
        self.loss_types = losses 
        super(customFocal_loss_mse_directionality, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        # add different initialisations for different types of losses. 
        for i in range(self.nb_tasks):
            if self.loss_types[i] == 'L2':
                self.log_vars+=[self.add_weight(name='log_vars'+str(i), shape=(1,),
                                                      initializer=Constant(0.), trainable=True)]
            if self.loss_types[i] == 'L2':
                self.log_vars+=[self.add_weight(name='log_vars'+str(i), shape=(1,),
                                                      initializer=Constant(0.), trainable=True)]
        super(customFocal_loss_mse_directionality, self).build(input_shape)
        
    def multi_loss(self, ys_true, ys_pred):       
        loss = 0
        for ii, (y_true, y_pred, log_var) in enumerate(zip(ys_true, ys_pred, self.log_vars)):
            precision = K.exp(-log_var[0]) # 1/sigma**2 this gives positivity.
            if self.loss_types[ii] == 'L2':
                print(ii, 'L2')
                epsilon = K.epsilon()
                y_pred = K.clip(y_pred, epsilon, 1. - epsilon) # make sure this is in the proper range
#                loss += K.mean(self.weighted_softmax_focal_loss(y_true, y_pred, precision))
                loss += K.sum( precision * K.abs(y_true - y_pred) + log_var[0], axis=-1) 
            if self.loss_types[ii] == 'L1':
                print(ii, 'L1')
                mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
#                y_pred = K.clip(y_pred, -1 + K.epsilon(), 1. - K.epsilon()) #### clip the predictions.
                # prevent division by 0 .
#                masked_mae = K.sum(mask_true* precision * K.abs(y_true - y_pred) + log_var[0], axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
                masked_mae = K.sum(mask_true* precision * K.abs(y_true - y_pred) + log_var[0], axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
                loss += masked_mae #K.mean(masked_mae)
#                loss += K.mean(K.sum(precision * K.abs((y_true - y_pred))  + log_var[0], -1))
        return K.mean(loss)
            
    def call(self, inputs):
        # parse the list input. 
        ys_true = inputs[:self.nb_tasks]
        ys_pred = inputs[self.nb_tasks:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)
    
def build_model(weightsfile, im_size=(512,1024), out_channels=1):
    
    from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Concatenate, UpSampling2D
    from keras.layers import Conv3D, Conv3DTranspose, TimeDistributed
    from attention import Self_Attention2D, UnetGatingSignal, AttnGatingBlock
    from keras.models import Model
    
    input_shape = (3, im_size[0], im_size[1], 1) # requires a 5D input of (samples, time, rows, cols, channels)
    y1_shape = (3, im_size[0], im_size[1], out_channels) # segmentations for each time point. 
    y2_shape = (3, im_size[0], im_size[1], 2) # directionality 

    #c = 16
    input_img = Input(input_shape, name='input')
    y_true_1 = Input(y1_shape, name='y_true_1')
    y_true_2 = Input(y2_shape, name='y_true_2')
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', padding='same')(input_img)
    c1 = Conv3D(filters=44, kernel_size=(3,3,3), activation='linear', padding='same')(x)
    x = TimeDistributed(BatchNormalization())(c1)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
    #x = Conv3D(filters=44, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    c2 = Conv3D(filters=64, kernel_size=(3,3,3), activation='linear', padding='same')(x)
    x = TimeDistributed(BatchNormalization())(c2)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
    #x = Conv3D(filters=64, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    c3 = Conv3D(filters=72, kernel_size=(3,3,3), activation='linear', padding='same')(x)
    x = TimeDistributed(BatchNormalization())(c3)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
    #x = Conv3D(filters=72, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)
    
    x = Conv3D(filters=72, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    c4 = Conv3D(filters=96, kernel_size=(3,3,3), activation='linear', padding='same')(x)
    x = TimeDistributed(BatchNormalization())(c4)
    x = Activation('relu')(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
    #x = Conv3D(filters=96, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)
    
    x = Conv3D(filters=72, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    c5 = Conv3D(filters=96, kernel_size=(3,3,3), activation='linear', padding='same')(x)
    x = TimeDistributed(BatchNormalization())(c5)
    x = Activation('relu')(x)
    #x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
    #x = Conv3D(filters=96, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)
    # upsampling branch.
    # [ attn 1 ] # how to extend this to work with TimeDistributed?
    #gating = TimeDistributed(UnetGatingSignal(x, is_batchnorm=True))
    #attn_1 = TimeDistributed(AttnGatingBlock(c4, gating, 96))
    ###### time distributed is required to create a u-net type architecture.
    x = TimeDistributed(UpSampling2D((2,2)))(x)
    #x = Conv3DTranspose(filters=96, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
    x = Concatenate(axis=-1)([x,c4])
    x = Conv3DTranspose(filters=96, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = Conv3DTranspose(filters=72, kernel_size=(3,3,3), activation='linear', padding='same')(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    
    x = TimeDistributed(UpSampling2D((2,2)))(x)
    #x = Conv3DTranspose(filters=96, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
    x = Concatenate(axis=-1)([x,c3])
    x = Conv3DTranspose(filters=96, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = Conv3DTranspose(filters=72, kernel_size=(3,3,3), activation='linear', padding='same')(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    
    x = TimeDistributed(UpSampling2D((2,2)))(x)
    #x = Conv3DTranspose(filters=72, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
    x = Concatenate(axis=-1)([x,c2])
    x = Conv3DTranspose(filters=72, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), activation='linear', padding='same')(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    
    x = TimeDistributed(UpSampling2D((2,2)))(x)
    #x = Conv3DTranspose(filters=44, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
    x = Concatenate(axis=-1)([x,c1])
    x = Conv3DTranspose(filters=44, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), activation='linear', padding='same')(x)
    x = TimeDistributed(BatchNormalization())(x)
    #x = TimeDistributed(Activation('relu'))(x)
    
    # predict the softmax output at all timepoints.... # should i increase this?
    output1 = TimeDistributed(Activation('relu'))(x)
    #output1 = Conv3D(filters=out_channels, kernel_size=(1,1,1), activation=softmax_3d(-1), padding='same')(output1)
    output1 = Conv3D(filters=out_channels, kernel_size=(1,1,1), activation='relu', padding='same')(output1)
    output2 = Conv3D(filters=16, kernel_size=(3,3,3), activation='linear', padding='same')(x) #### tanh is not working. 
    output2 = Conv3D(filters=16, kernel_size=(3,3,3), activation='linear', padding='same')(output2)
    output2 = Conv3D(filters=2, kernel_size=(1,1,1), activation='linear', padding='same')(output2)  
    
    prediction_model = Model(inputs=[input_img], outputs=[output1, output2]) # classification branch, centroid regression branch.
    
    y_pred1, y_pred2 = prediction_model(input_img) # gives two outputs. 
    ##focus loss model. 
    output = customFocal_loss_mse_directionality(nb_outputs=out_channels,
                                                 nb_tasks=2,
                                                 losses=['L2','L1'])([y_true_1, y_true_2, y_pred1, y_pred2])
    infer_model = Model(inputs=[input_img, y_true_1, y_true_2], outputs=[output])

    #infer_model = Model(inputs=[input_img, y_true], outputs=[output])
#    infer_model.load_weights('cell-seg-unet-sigmoid-Conv3D-256-customFL-weighted-direction-correct-soft-1.h5') # load the learnt weights. 
    infer_model.load_weights(weightsfile) # load the learnt weights. 
    #prediction_model.load_weights('cell-seg-unet-softmax3-Conv3D-256-categorical-FL-weighted.h5') # load the learnt weights. 
    model = prediction_model
    
    return model

def apply_segment_model(img, model, target_shape=(512,1024,3)):
    
    from skimage.exposure import rescale_intensity, equalize_hist
    pad_rows = (target_shape[0] - img.shape[1]) // 2
    pad_cols = (target_shape[1] - img.shape[2]) // 2

    print(pad_rows, pad_cols)
    print(img.transpose(1,2,0).shape)
    in_im = np.zeros(target_shape)
    in_im[pad_rows:pad_rows+img.shape[1], 
          pad_cols:pad_cols+img.shape[2],:] = img.transpose(1,2,0)
    
    in_im = rescale_intensity(in_im/255.).astype(np.float32)
    out = model.predict(in_im.transpose(2,0,1)[None,:][...,None], batch_size=1)
    cells, votes = out # split the two channel outputs. 
    
    cells = np.squeeze(cells); votes = np.squeeze(votes)

    # cut-out the relevant parts. 
    cells = cells[:,pad_rows:pad_rows+img.shape[1], 
                    pad_cols:pad_cols+img.shape[2]]
    cells = np.clip(cells, 0, 1)
    
    votes = votes[:,pad_rows:pad_rows+img.shape[1], 
                    pad_cols:pad_cols+img.shape[2]]
    
    return cells, votes

def apply_segment_model_video(vid, weightfile, network_scale = 16., n_slices=3):
    
    model_rows = int(np.ceil(vid.shape[1]/network_scale) * network_scale)
    model_cols = int(np.ceil(vid.shape[2]/network_scale) * network_scale)
    
    segment_model = build_model(weightfile, im_size=(model_rows, model_cols), out_channels=1)
    n_frames, n_rows, n_cols = vid.shape
    
    # pad the video symmetrically 
    vid = np.pad(vid, [[n_slices-1,n_slices-1],[0,0],[0,0]], mode='reflect') # pad this much to get exactly 3 votes per frame.
    
    out_cells = np.zeros(vid.shape)
    out_votes = np.zeros((vid.shape[0], vid.shape[1], vid.shape[2], 2))
    
    # iterate over the frames 
    for frame in tqdm(range(vid.shape[0]-n_slices)[:]):
#            print('Frame %d' %(frame))
        vid_frame = vid[frame:frame+n_slices] # take out 3
        cells, centroid_votes = apply_segment_model(vid_frame, segment_model, target_shape=(model_rows, model_cols, n_slices))
        
        out_cells[frame:frame+n_slices] += cells
        out_votes[frame:frame+n_slices] += centroid_votes   
        
    out_cells = out_cells / float(n_slices) # to average 
    out_votes = out_votes / float(n_slices) # to average
    out_cells = out_cells[n_slices-1:-(n_slices-1)]
    out_votes = out_votes[n_slices-1:-(n_slices-1)] 
    
    return out_cells, out_votes



def apply_segment_model_video_no_average(vid, weightfile, network_scale = 16., n_slices=3):
    
    model_rows = int(np.ceil(vid.shape[1]/network_scale) * network_scale)
    model_cols = int(np.ceil(vid.shape[2]/network_scale) * network_scale)
    
    segment_model = build_model(weightfile, im_size=(model_rows, model_cols), out_channels=1)
    n_frames, n_rows, n_cols = vid.shape
    
    # pad the video symmetrically 
    vid = np.pad(vid, [[n_slices-1,n_slices-1],[0,0],[0,0]], mode='reflect') # pad this much to get exactly 3 votes per frame.
    
#    out_cells = np.zeros(vid.shape)
    out_cells = np.zeros(np.hstack([vid.shape, n_slices]))
    out_votes = np.zeros((vid.shape[0], vid.shape[1], vid.shape[2], n_slices, 2))
    
    # iterate over the frames 
    for frame in tqdm(range(vid.shape[0]-n_slices)[:]):
#            print('Frame %d' %(frame))
        vid_frame = vid[frame:frame+n_slices] # take out 3
        cells, centroid_votes = apply_segment_model(vid_frame, segment_model, target_shape=(model_rows, model_cols, n_slices))
        
        print(out_cells[frame].shape)
        print(cells.shape)
#        out_cells[frame:frame+n_slices] = cells
#        out_votes[frame:frame+n_slices] = centroid_votes   
        out_cells[frame] = cells.transpose(1,2,0)
        out_votes[frame] = centroid_votes.transpose(1,2,0,3)   
        
#    out_cells = out_cells / float(n_slices) # to average 
#    out_votes = out_votes / float(n_slices) # to average
#    out_cells = out_cells[n_slices-1:-(n_slices-1)]
#    out_votes = out_votes[n_slices-1:-(n_slices-1)] 
    
    return out_cells, out_votes


def mkdir(directory):
    
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def save_multipage_tiff(np_array, savename):
    
    from tifffile import imsave
    import numpy as np 
    
    if np_array.max() < 1.1:
        imsave(savename, np.uint8(255*np_array))
    else:
        imsave(savename, np.uint8(np_array))
    
    return [] 

import numpy as np
def save_multipage_tiff_type(np_array, savename, savetype=np.uint16):
    
    from tifffile import imsave
    import numpy as np 
    
    imsave(savename, savetype(np_array))
    
    return [] 

def largest_connected_component(binary):
    
    from skimage.measure import label   
    from scipy.ndimage.morphology import binary_fill_holes

#    def getLargestCC(segmentation):
    labels = label(binary)
    assert( labels.max() != 0 ) # assume at least 1 CC
    
    if len(np.unique(labels)) == 2:
        largestCC = labels.copy()
        return binary_fill_holes(largestCC)
    else:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return binary_fill_holes(largestCC)


def clean_segmentation_mask(seg_mask, min_size=100):
    
    from skimage.segmentation import relabel_sequential
    from skimage.measure import label, regionprops
    
    regions = np.unique(seg_mask)[:]
    new_mask = np.zeros_like(seg_mask)
    
    for reg in regions: 
        mask = seg_mask == reg
#        mask = largest_connected_component(mask)
        labels = label(mask); labels_reg = regionprops(labels)
        
        if len(np.unique(labels)) > 2:
            mask = labels == np.unique(labels)[1:][np.argmax([re.area for re in labels_reg])]
        
#        if np.sum(mask) >= min_size:
        new_mask[mask] = reg
        
    new_mask = relabel_sequential(new_mask)[0]
    return new_mask


def deep_watershed_regions(vote_field_x, vote_field_y, sigma_spatial=1, niter=15, mask=None, debug_viz=False, thresh_factor=0.5, min_area=200):
    
    import pylab as plt 
    from skimage.measure import label # label connected regions.
    from skimage.segmentation import relabel_sequential
    
    XX, YY = np.meshgrid(range(vote_field_x.shape[1]), 
                         range(vote_field_x.shape[0]))

    for ii in range(niter):
    
#        centres_grid_xx = centres_grid_xx + test_vote_img[test_time,centres_grid_yy.astype(np.int), centres_grid_xx.astype(np.int), 1] # move them according to the displacement grid. 
#        centres_grid_yy = centres_grid_yy + test_vote_img[test_time,centres_grid_yy.astype(np.int), centres_grid_xx.astype(np.int),0]
        XX = XX + vote_field_x[np.clip(np.rint(YY).astype(np.int), 0, vote_field_x.shape[0]-1), 
                               np.clip(np.rint(XX).astype(np.int), 0, vote_field_x.shape[1]-1)] # move them according to the displacement grid. 
        YY = YY + vote_field_y[np.clip(np.rint(YY).astype(np.int), 0, vote_field_x.shape[0]-1), 
                               np.clip(np.rint(XX).astype(np.int), 0, vote_field_x.shape[1]-1)]
        
        # clip the XX, YY points 
        XX = np.clip(XX, 0, vote_field_x.shape[1]-1)
        YY = np.clip(YY, 0, vote_field_x.shape[0]-1)
        
        if debug_viz:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,15))
            ax[0].imshow(vote_field_x)
            ax[0].plot(XX.ravel(), YY.ravel(), 'r.')
            
            ax[1].imshow(vote_field_y)
            ax[1].plot(XX.ravel(), YY.ravel(), 'r.')
            plt.show()
    
    votes_grid_acc = np.zeros(XX.shape)

#    if mask is not None:
#        
##        plt.figure()
##        plt.imshow(mask>0)
##        plt.show()
#        votes_grid_acc[np.rint(YY[mask>0]).astype(np.int), 
#                       np.rint(XX[mask>0]).astype(np.int)] += 1.
#                       
##        plt.figure()
##        plt.imshow(mask)
##        plt.imshow(votes_grid_acc, alpha=0.5)
##        plt.show()
#                   
#    else:
    votes_grid_acc[np.rint(YY).astype(np.int), 
                   np.rint(XX).astype(np.int)] += 1. # add a vote. 
                   
    # smooth to get a density (fast KDE estimation)
    votes_grid_acc = gaussian(votes_grid_acc, sigma=sigma_spatial, preserve_range=True)  

#    """
#    revise more this procedure for identification of cliques. 
#    """
#    votes_grid_peaks = peak_local_max(votes_grid_acc, min_distance=5)

    # threshold and get the connected region labels
    
    if thresh_factor is not None:
        if mask is not None:
            print('mask')
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc[mask]) + thresh_factor*np.std(votes_grid_acc[mask])
        else:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc) + thresh_factor*np.std(votes_grid_acc)
    else:
        print('hello')
        votes_grid_binary = votes_grid_acc > np.mean(votes_grid_acc)
        
#    plt.figure()
#    plt.imshow(votes_grid_binary)
#    plt.show()
    connected_regions = label(votes_grid_binary)
    
    cell_seg_connected = connected_regions[np.rint(YY).astype(np.int), 
                                           np.rint(XX).astype(np.int)]
    
    
#    cell_seg_connected = cell_seg_connected + 1 # leave 0 for background. 
    cell_uniq_regions = np.unique(cell_seg_connected)
    cell_seg_connected_area = np.bincount(cell_seg_connected.flat)[cell_uniq_regions]
    invalid_areas = cell_uniq_regions[cell_seg_connected_area<=min_area]
    
    for invalid in invalid_areas:
        cell_seg_connected[cell_seg_connected==invalid] = 0
        
    if cell_seg_connected.max() > 0:
        cell_seg_connected = relabel_sequential(cell_seg_connected)[0]
#    cell_seg_connected = clean_segmentation_mask(cell_seg_connected)
    
    if mask is not None:
        cell_seg_connected[mask == 0] = 0
    
#    """
#    To do remove excessively small regions. 
#    """
#    for reg in np.unique(cell_seg_connected):
#        area = np.sum(cell_seg_connected == reg)
#        if area <= min_area:
#            cell_seg_connected[cell_seg_connected==reg] = 0
#        
    return cell_seg_connected


if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    import glob 
    import os 
    import scipy.io as spio
    from skimage.io import imsave
    
#    import MOSES.Utility_Functions.file_io as fio
#    from MOSES.Optical_Flow_Tracking.optical_flow import farnebackflow, TVL1Flow, DeepFlow
#    from MOSES.Optical_Flow_Tracking.superpixel_track import compute_vid_opt_flow, compute_grayscale_vid_superpixel_tracks, compute_grayscale_vid_superpixel_tracks_w_optflow
#    from MOSES.Visualisation_Tools.track_plotting import plot_tracks
#    from flow_vis import make_colorwheel, flow_to_color
    
    from skimage.filters import threshold_otsu
    import skimage.transform as sktform
    from tqdm import tqdm 
    import pandas as pd 
    
    from skimage.filters import gaussian
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.io import imread, imsave
    
#    master_analysis_save_folder = '/media/felix/Srinivas4/All_results/Migration_Analysis/Staged_Analysis'
#    fio.mkdir(master_analysis_save_folder)
    
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    
# =============================================================================
#     Load the weightsfile
# =============================================================================
    weightfile = 'cell-seg-unet-Conv3D-full-L1-direction-mae_mae-more_aug-v2-1.h5'
    
    
# =============================================================================
#   Should set up a nearest neighbour model to check out the transformation between polar and planar views.    
# =============================================================================
    # Set up a nearest neighbour model 
    from sklearn.neighbors import NearestNeighbors 
    from skimage.segmentation import mark_boundaries
    nbrs_model = NearestNeighbors(n_neighbors=9, algorithm='auto') # how many neighbours to include? 9 = 8-connected?
    
    
    all_rootfolders = ['/media/felix/Srinivas4/LifeAct']
#    all_rootfolders = ['/media/felix/Srinivas4/MTMG-TTR']


    all_embryo_folders = []
    
    for rootfolder in all_rootfolders[:1]:
    # =============================================================================
    #     1. Load all embryos 
    # =============================================================================
        embryo_folders = np.hstack([os.path.join(rootfolder,folder) for folder in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, folder)) and 'new_hex' not in folder])
        all_embryo_folders.append(embryo_folders)
        
    all_embryo_folders = np.hstack(all_embryo_folders)
    
    
    # =============================================================================
    #   2. iterate over all the embryos and save .
    # =============================================================================
#    for embryo_folder in all_embryo_folders[:] :
#    for embryo_folder in all_embryo_folders[5:6] :
    for embryo_folder in embryo_folders[4:5]:
    
        # find the geodesic-rotmatrix. 
        unwrapped_folder = locate_unwrapped_im_folder(embryo_folder, key='geodesic-rotmatrix')
        
        # shouldn't include the lefty embryo anyway in the analysis
        if len(unwrapped_folder) == 1 and '_LEFTY_' not in unwrapped_folder[0]:
            print('processing: ', unwrapped_folder[0])
            
            # locate the demons folder to obtain the x,y,z dimensions for normalization.         
            embryo_im_folder = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step6_' in name or 'step6tf_' in name or 'step6tf-' in name or 'step6-' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
            embryo_im_folder_files = np.hstack(glob.glob(os.path.join(embryo_im_folder[0], '*.tif')))
            embryo_im_shape = np.hstack(imread(embryo_im_folder_files[0]).shape)
            embryo_im_shape = embryo_im_shape[[1,0,2]]#.transpose([1,0,2])
            
            unwrapped_folder = unwrapped_folder[0]
            unwrapped_params_folder = unwrapped_folder.replace('geodesic-rotmatrix', 
                                                               'unwrap_params_geodesic-rotmatrix')
#            # determine the savefolder. 
#            saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
##            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned')
##            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-5000_VE_seg_directions') # this was the segmentation folder. 
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
##            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s' %(n_spixels, analyse_stage))
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_stage-aligned-correct_uniform_scale-%d_%s' %(n_spixels, analyse_stage))
#            
#            saveboundaryfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow-epi-boundary')

            cellsegfolder = unwrapped_params_folder + '_cell-seg'
            cellsegfolder_in = cellsegfolder#.copy()
            # switch this cellsegfolder to refer to new drive structure. 
            cellsegfolder = cellsegfolder.replace('/media/felix/Srinivas4','/media/felix/My Passport/Shankar-2/Data_Processing')
            mkdir(cellsegfolder)

        # =============================================================================
        #       a) Locate all the unwrapped files.       
        # =============================================================================
            unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))

            if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
            
            else:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_hex_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_hex_embryo(unwrap_im_files, unwrap_meta_files)
                
    
            for ii in tqdm(range(len(paired_unwrap_file_condition))[1:2]):
#            for ii in tqdm(range(len(paired_unwrap_file_condition))[:1]):
                    
                unwrapped_condition = paired_unwrap_file_condition[ii]
                print(unwrapped_condition)
            
                if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
                    ve_unwrap_file, epi_unwrap_file = paired_unwrap_files[ii]
                else:
                    ve_unwrap_file, epi_unwrap_file, hex_unwrap_file = paired_unwrap_files[ii]
            
            
                vid_ve = imread(ve_unwrap_file)
                vid_epi = imread(epi_unwrap_file)
                
                
                if 'MTMG-HEX' in embryo_folder:
                    vid_hex = imread(hex_unwrap_file)
                         
        # =============================================================================
        #       b) Use the trained CNN for enhancement of cells. (auto figure out the padding factor.) 
        # =============================================================================
        
                basename_ve = os.path.split(ve_unwrap_file)[-1]; #print(basename_ve)
                basename_epi = os.path.split(epi_unwrap_file)[-1]; #print(basename_epi)
                
                
                """
                Read in the obtained votes VE and Epi
                """
                vid_ve_seg = imread(os.path.join(cellsegfolder_in, 'prob_mask_img-' +basename_ve))
                vid_ve_votes = spio.loadmat(os.path.join(cellsegfolder_in, 'deep-vote-img_' + basename_ve.replace('.tif', '.mat')))['votes'].astype(np.float32)
                
                vid_epi_seg = imread(os.path.join(cellsegfolder_in, 'prob_mask_img-' +basename_epi))
                vid_epi_votes = spio.loadmat(os.path.join(cellsegfolder_in, 'deep-vote-img_' + basename_epi.replace('.tif', '.mat')))['votes'].astype(np.float32)
                
                            
                vid_ve_seg_cell = []
                vid_epi_seg_cell = []
                
                for frame_no in tqdm(range(len(vid_ve_seg))):
                
                    if unwrapped_condition.split('-')[-1] == 'rect':
                        vid_ve_seg_cell_frame = deep_watershed_regions(vid_ve_votes[frame_no,...,1], 
                                                                       vid_ve_votes[frame_no,...,0], 
                                                                       sigma_spatial=3, niter=30, 
                                                                       mask=None, debug_viz=False, thresh_factor=0.5, min_area=200)
                        vid_epi_seg_cell_frame = deep_watershed_regions(vid_epi_votes[frame_no,...,1], 
                                                                        vid_epi_votes[frame_no,...,0], 
                                                                        sigma_spatial=1, 
                                                                        niter=30, 
                                                                        mask=None, debug_viz=False, thresh_factor=0.5, min_area=100)
                    
                    if unwrapped_condition.split('-')[-1] == 'polar':
                        
                        y_grid, x_grid = np.indices(vid_ve[0].shape)
                        mag = np.sqrt((x_grid-x_grid.mean())**2 + (y_grid-y_grid.mean())**2)
                        polar_mask_ve = mag <= vid_ve[0].shape[1]/2./1.1
                        
                        vid_ve_seg_cell_frame = deep_watershed_regions(vid_ve_votes[frame_no,...,1], 
                                                                       vid_ve_votes[frame_no,...,0], 
                                                                       sigma_spatial=.5, 
                                                                       niter=50, 
                                                                       mask=polar_mask_ve, debug_viz=False, thresh_factor=.25, min_area=50)
                        
                        y_grid, x_grid = np.indices(vid_epi[0].shape)
                        mag = np.sqrt((x_grid-x_grid.mean())**2 + (y_grid-y_grid.mean())**2)
                        polar_mask_epi = mag <= vid_epi[0].shape[1]/2./1.1
                        
                        
                        vid_epi_seg_cell_frame = deep_watershed_regions(vid_epi_votes[frame_no,...,1], 
                                                                        vid_epi_votes[frame_no,...,0], 
                                                                        sigma_spatial=.5, 
                                                                        niter=100, 
                                                                        mask=polar_mask_epi, debug_viz=False, thresh_factor=.25, min_area=1)
                
                
                    vid_ve_seg_cell.append(vid_ve_seg_cell_frame)
                    vid_epi_seg_cell.append(vid_epi_seg_cell_frame)
                    
                vid_ve_seg_cell = np.array(vid_ve_seg_cell)
                vid_epi_seg_cell = np.array(vid_epi_seg_cell)
                """
                output the density map and the votes map. 
                """                
                from skimage.exposure import equalize_hist
                plt.figure(figsize=(15,15)); 
                plt.imshow(mark_boundaries(np.dstack([equalize_hist(vid_epi[15]), 
                                                      equalize_hist(vid_epi[15]), 
                                                      equalize_hist(vid_epi[15])]), 
                                           vid_epi_seg_cell[15]))
                
                plt.figure(figsize=(15,15)); 
                plt.imshow(mark_boundaries(np.dstack([equalize_hist(vid_ve[15]), 
                                                      equalize_hist(vid_ve[15]), 
                                                      equalize_hist(vid_ve[15])]), 
                                           vid_ve_seg_cell[15]))
                
                plt.figure(figsize=(15,15)); 
                plt.imshow(vid_ve[15])
                
                
#                save_multipage_tiff_type(np.uint16(vid_ve_seg_cell), os.path.join(cellsegfolder, 'cell_img-' + basename_ve))
#                save_multipage_tiff_type(np.uint16(vid_epi_seg_cell), os.path.join(cellsegfolder, 'cell_img-' + basename_epi))
                
#                save_multipage_tiff(vid_ve_seg[1:-2-1,...,1], os.path.join(cellsegfolder, 'prob_mask_img-' +basename_ve))
#                spio.savemat(os.path.join(cellsegfolder, 'deep-vote-img_' + basename_ve.replace('.tif', '.mat')), 
#                                                         {'votes': vid_ve_votes[1:-2-1,:,:,1,:].astype(np.float32)})
#                
#                
#                save_multipage_tiff(vid_epi_seg[1:-2-1,...,1], os.path.join(cellsegfolder, 'prob_mask_img-' +basename_epi))
#                spio.savemat(os.path.join(cellsegfolder, 'deep-vote-img_' + basename_epi.replace('.tif', '.mat')), 
#                                                         {'votes': vid_epi_votes[1:-2-1,:,:,1,:].astype(np.float32)})
                

#    """
#    Iterate over experimental folders to build quadranting space.....
#    """
#    all_analysed_embryos = []
#    all_grids = []
#    all_speeds_regions_ve = []
#    all_speeds_regions_epi = []
#    all_speeds_regions_ve_proj_ve = []
#    all_speeds_regions_epi_proj_ve = []
#    all_embryo_shapes = []
#    all_embryo_vol_factors = []
#    all_ve_angles = []
#    
#    all_directions_regions_ve = []
#    all_directions_regions_epi = []
#
#
##    for embryo_folder in embryo_folders[-2:-1]:
#    for embryo_folder in all_embryo_folders[:]:
##    for embryo_folder in embryo_folders[4:5]:
#        
#        # find the geodesic-rotmatrix. 
#        unwrapped_folder = locate_unwrapped_im_folder(embryo_folder, key='geodesic-rotmatrix')
#        
#        # shouldn't include the lefty embryo anyway in the analysis
#        if len(unwrapped_folder) == 1 and '_LEFTY_' not in unwrapped_folder[0]:
#            print('processing: ', unwrapped_folder[0])
#            
#            # locate the demons folder to obtain the x,y,z dimensions for normalization.         
#            embryo_im_folder = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step6_' in name or 'step6tf_' in name or 'step6tf-' in name or 'step6-' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
#            embryo_im_folder_files = np.hstack(glob.glob(os.path.join(embryo_im_folder[0], '*.tif')))
#            embryo_im_shape = np.hstack(fio.read_multiimg_PIL(embryo_im_folder_files[0]).shape)
#            embryo_im_shape = embryo_im_shape[[1,0,2]]#.transpose([1,0,2])
#            
#            unwrapped_folder = unwrapped_folder[0]
#            unwrapped_params_folder = unwrapped_folder.replace('geodesic-rotmatrix', 
#                                                               'unwrap_params_geodesic-rotmatrix')
#            # determine the savefolder. 
#            saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
##            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned')
##            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-5000_VE_seg_directions') # this was the segmentation folder. 
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
##            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s' %(n_spixels, analyse_stage))
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_stage-aligned-correct_uniform_scale-%d_%s' %(n_spixels, analyse_stage))
#            
#            saveboundaryfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow-epi-boundary')
#            
#            if os.path.exists(savetrackfolder) and len(os.listdir(savetrackfolder)) > 0:
#                # only tabulate this statistic if it exists. 
##                all_analysed_embryos.append(embryo_folder)
#            
#                unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))
#    
#                if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
#                    unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
#                    paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
#                
#                else:
#                    unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_hex_unwrapped(ff)) for ff in unwrap_im_files])
#                    paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_hex_embryo(unwrap_im_files, unwrap_meta_files)
#                
#                # iterate over the pairs 
#                for ii in tqdm(range(len(paired_unwrap_file_condition))[:1]):
#                    
#                    unwrapped_condition = paired_unwrap_file_condition[ii]
#                    print(unwrapped_condition)
#                    
#                    # =============================================================================
#                    #   get scaling transforms from registration           
#                    # =============================================================================
#                    """
#                    get the temporal transforms and extract the scaling parameters. 
#                    """
#                    temp_tform_files = glob.glob(os.path.join(embryo_folder, '*temp_reg_tforms*.mat'))
#                    assert(len(temp_tform_files) > 0)
#                    
#                    temp_tform_scales = np.vstack([parse_temporal_scaling(tformfile) for tformfile in temp_tform_files])
#                    temp_tform_scales = np.prod(temp_tform_scales, axis=0)
#                    
#                    # =============================================================================
#                    #   Specific analyses.  
#                    # =============================================================================
#                    # only analyse if the prereq condition is met 
#                    if unwrapped_condition.split('-')[-1] == 'polar':
#                        
#                        if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
#                            ve_unwrap_file, epi_unwrap_file = paired_unwrap_files[ii]
#                        else:
#                            ve_unwrap_file, epi_unwrap_file, hex_unwrap_file = paired_unwrap_files[ii]
#                                
#                    # =============================================================================
#                    #        Load the relevant videos.          
#                    # =============================================================================
#                        vid_ve = fio.read_multiimg_PIL(ve_unwrap_file)
#                        vid_epi = fio.read_multiimg_PIL(epi_unwrap_file)
#                        vid_epi_resize = sktform.resize(vid_epi, vid_ve.shape, preserve_range=True)
#                        
#                        
#                        if 'MTMG-HEX' in embryo_folder:
#                            vid_hex = fio.read_multiimg_PIL(hex_unwrap_file)
#                            
#                    # =============================================================================
#                    #       Load the relevant saved pair track file.               
#                    # =============================================================================
#                        trackfile = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_%s_' %(n_spixels, analyse_stage) + unwrapped_condition+'.mat')
#                        
#                        meantracks_ve = spio.loadmat(trackfile)['meantracks_ve']
#                        meantracks_epi = spio.loadmat(trackfile)['meantracks_epi']
#                        
#                        proj_condition = spio.loadmat(trackfile)['condition'][0].split('-')
#                        proj_type = proj_condition[-1]
#                        start_end_times = spio.loadmat(trackfile)['start_end'].ravel()
#        
##                        if proj_type == 'polar':
##                        all_analysed_embryos.append(embryo_folder)
#                    # =============================================================================
#                    #       Load the relevant directionality info     
#                    # =============================================================================
#                        if proj_type == 'polar':
#                            ve_angle_move = angle_tab.loc[angle_tab['polar_conditions'].values == '-'.join(proj_condition)]['correct_angles_vertical']
#                        if proj_type == 'rect':
#                            ve_angle_move = angle_tab.loc[angle_tab['rect_conditions'].values == '-'.join(proj_condition)]['correct_angles_vertical']
#        
#        
#                        ve_angle_move = 90 - ve_angle_move # to get back to the rectangular angle coordinate? 
#                        print('ve movement direction: ', ve_angle_move)
#        
#                    # =============================================================================
#                    #       Produce the required gridding (polar for now, rotated to the angle of motion ) - regular quadranting. 
#                    #       --- create the more irregular type of gridding. 
#                    # =============================================================================
#                    
#                        # grab the relevant boundary file. 
#                        epi_boundary_file = os.path.join(saveboundaryfolder,unwrapped_condition,'inferred_epi_boundary-'+unwrapped_condition+'.mat')
#                        epi_boundary_time = spio.loadmat(epi_boundary_file)['contour_line']
#                        
#                        epi_contour_line = epi_boundary_time[start_end_times[0]]
#                        
#                        
#                        polar_grid = tile_uniform_windows_radial_guided_line(vid_ve[0].shape, 
#                                                                             n_r=4, 
#                                                                             n_theta=8,
#                                                                             max_r=vid_ve[0].shape[0]/2./1.2,
#                                                                             mid_line = epi_contour_line[:,[0,1]],
#                                                                             center=None, 
#                                                                             bound_r=True)
##                        polar_grid = tile_uniform_windows_radial_custom(vid_ve[0].shape, 
##                                                                        n_r=4, 
##                                                                        n_theta=8, 
##                                                                        max_r=vid_ve[0].shape[0]/2./1.2, 
##                                                                        mask_distal=None, # leave a set distance
##                                                                        center=None, 
##                                                                        bound_r=True)
##                        from skimage.segmentation import mark_boundaries
#                        plt.figure(figsize=(10,10))
##                        plt.imshow(vid_ve[0], cmap='gray')
#                        plt.imshow(mark_boundaries(np.uint8(np.dstack([vid_epi_resize[0], vid_epi_resize[0], vid_epi_resize[0]])), 
#                                                   polar_grid), cmap='gray')
##                        plt.imshow(polar_grid, cmap='coolwarm', alpha=0.5)
#                        plt.show()
#                        
#                        
#                        polar_grid = realign_grid_center_polar(polar_grid, 
#                                                               n_angles=8, 
#                                                               center_dist_r=0, 
#                                                               center_angle=ve_angle_move)
#                        uniq_polar_regions = np.unique(polar_grid)[1:]
#                        uniq_polar_spixels = [(polar_grid==r_id)[meantracks_ve[:,0,0], meantracks_ve[:,0,1]] for r_id in uniq_polar_regions]
#                        
#                    # =============================================================================
#                    #       Load the unwrapping 3D geometry coordinates 
#                    # =============================================================================
#                
#                        # load the associated unwrap param files depending on polar/rect coordinates transform. 
#                        unwrap_param_file_ve = ve_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
#                        unwrap_params_ve = spio.loadmat(unwrap_param_file_ve); unwrap_params_ve = unwrap_params_ve['ref_map_'+proj_type+'_xyz']
#                        
#                        unwrap_param_file_epi = epi_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
#                        unwrap_params_epi = spio.loadmat(unwrap_param_file_epi); unwrap_params_epi = unwrap_params_epi['ref_map_'+proj_type+'_xyz']
#                        unwrap_params_epi = sktform.resize(unwrap_params_epi, unwrap_params_ve.shape, preserve_range=True)
#                        
#                        
#                        if proj_type == 'rect':
#                            unwrap_params_ve = unwrap_params_ve[::-1] # reverse ....
#                            unwrap_params_epi = unwrap_params_epi[::-1]
#                        
##                        plt.figure()
##                        plt.imshow(unwrap_params_ve[...,0])
##                        plt.show()
#                    # =============================================================================
#                    #       Load the relevant VE segmentation from trained classifier    
#                    # =============================================================================
#                        trackfile_full = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels), 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')
##                        trackfile_full = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels), 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')
#                        
##                        ve_seg_params_file = os.path.join(savevesegfolder, os.path.split(trackfile_full)[-1].replace('rect','polar').replace('.mat', '-ve_aligned.mat'))
##                        ve_seg_params_obj = spio.loadmat(ve_seg_params_file)                             
#                        
#                        ve_tracks_full = spio.loadmat(trackfile_full)['meantracks_ve']
##                        ve_seg_params_file = os.path.join(savevesegfolder, os.path.split(trackfile_full)[-1].replace('rect','polar').replace('.mat', '-ve_aligned.mat'))
##                        ve_seg_params_obj = spio.loadmat(ve_seg_params_file)                
#                        ve_seg_params_file = 'deepflow-meantracks-1000_' + os.path.split(unwrap_param_file_ve)[-1].replace('_unwrap_params-geodesic.mat',  '_polar-ve_aligned.mat')
#                        ve_seg_params_file = os.path.join(savevesegfolder, ve_seg_params_file)
#                        ve_seg_params_obj = spio.loadmat(ve_seg_params_file)                             
#                
#                
#                        # load the polar mask and the times at which this was calculated to infer the positions at earlier and later times.  
#                        ve_seg_select_rect_valid = ve_seg_params_obj['density_rect_pts_mask']>0 # get the polar associated mask. 
#                        ve_seg_migration_times = ve_seg_params_obj['migration_stage_times'][0]
#                        
#                        embryo_inversion = ve_seg_params_obj['embryo_inversion'].ravel()[0] > 0 
#                        
#        #                ve_seg_select_rect = ve_seg_params_obj['ve_rect_select'].ravel() > 0
#        #                ve_seg_select_rect_valid = ve_seg_params_obj['ve_rect_select_valid'].ravel() > 0
##                        ve_seg_select_rect_valid = ve_seg_params_obj['density_rect_pts']
##                        ve_seg_select_rect_valid = ve_seg_select_rect_valid >= np.mean(ve_seg_select_rect_valid) + 1 * np.std(ve_seg_select_rect_valid)
##                        ve_seg_select_rect = ve_seg_select_rect_valid[meantracks_ve[:,0,0], 
##                                                                      meantracks_ve[:,0,1]]
#                            
#                    # =============================================================================
#                    #    Embryo Inversion                      
#                    # =============================================================================
#                        if embryo_inversion:
#                            if proj_type == 'rect':
#                                unwrap_params_ve = unwrap_params_ve[:,::-1] # reverse x direction 
#                                unwrap_params_epi = unwrap_params_epi[:,::-1]
#                                
#                                ve_tracks_full[...,1] = vid_ve[0].shape[1]-1 - ve_tracks_full[...,1] # reverse the x direction 
#                                meantracks_ve[...,1] = vid_ve[0].shape[0]-1 - meantracks_ve[...,1]
#                                meantracks_epi[...,1] = vid_epi_resize[0].shape[1]-1 - meantracks_epi[...,1]
#                                
#                                vid_ve = vid_ve[:,:,::-1]
#                                vid_epi = vid_epi[:,:,::-1]
#                                vid_epi_resize = vid_epi_resize[:,:,::-1]
#                            else:
#                                unwrap_params_ve = unwrap_params_ve[::-1] # reverse y direction
#                                unwrap_params_epi = unwrap_params_epi[::-1]
#                                
#                                ve_tracks_full[...,0] = vid_ve[0].shape[0]-1 - ve_tracks_full[...,0] # switch the y direction. 
#                                meantracks_ve[...,0] = vid_ve[0].shape[0]-1 - meantracks_ve[...,0]
#                                meantracks_epi[...,0] = vid_epi_resize[0].shape[0]-1 - meantracks_epi[...,0]
#                                
#                                vid_ve = vid_ve[:,::-1]
#                                vid_epi = vid_epi[:,::-1]
#                                vid_epi_resize = vid_epi_resize[:,::-1]
#                                
#                        
#                    # =============================================================================
#                    #     Compute the VE region                    
#                    # =============================================================================
#                        # find the effective superpixel size. 
#                        spixel_size = np.abs(meantracks_ve[1,0,1] - meantracks_ve[2,0,1])
#                                
#                        if np.isnan(ve_seg_migration_times[0]):
#                            # the first time is adequate and should be taken for propagation. 
#                            ve_seg_select_rect = ve_seg_select_rect_valid[ve_tracks_full[:,0,0], 
#                                                                          ve_tracks_full[:,0,1]] > 0
#                        else:
#                            start_ve_time = int(ve_seg_migration_times[0])
#                            
#                            # retrieve the classification at the VE start point. 
#                            ve_seg_select_rect = ve_seg_select_rect_valid[ve_tracks_full[:,start_ve_time,0], 
#                                                                          ve_tracks_full[:,start_ve_time,1]] > 0
#                        
##                        ve_seg_tracks = ve_tracks_full[ve_seg_select_rect,start_end_times[0]:start_end_times[1]]
#                        ve_seg_tracks = ve_tracks_full[ve_seg_select_rect,:]
#                        ve_seg_track_pts = ve_seg_tracks[:,0]
#                            
#                        # retain only largest connected component as the labelled ids. 
#                        connected_labels = postprocess_polar_labels(ve_tracks_full[:,start_end_times[0]], 
#                                                                    ve_seg_select_rect, dist_thresh=2*spixel_size)
#                        
#                        # apply it now to the start time of interest
#                        ve_seg_track_pts = ve_tracks_full[connected_labels, start_end_times[0]]
#
#                        density_mask = np.zeros(vid_ve[0].shape)
#                        density_mask[ve_seg_track_pts[:,0],
#                                     ve_seg_track_pts[:,1]] = 1
#                        density_mask = gaussian(density_mask, 1*spixel_size)
#                        
#                        # .5 for lifeact 
#                        density_mask_pts = density_mask > np.mean(density_mask) + 1.*np.std(density_mask) # produce the mask. 
#                        density_mask_pts = binary_fill_holes(density_mask_pts)
#                        
#                        
#                        # use this as the proper one. 
#                        ve_seg_select_rect = density_mask_pts[meantracks_ve[:,0,0], 
#                                                              meantracks_ve[:,0,1]]
#                        
#                        fig, ax = plt.subplots(figsize=(15,15))
#                        plt.imshow(vid_ve[start_end_times[0]], cmap='gray')
#                        plt.imshow(polar_grid, cmap='coolwarm', alpha=0.5)
#                        plot_tracks(meantracks_ve[:,:10], ax=ax, color='g')
#                        plot_tracks(meantracks_epi[:,:10], ax=ax, color='r')
#                        plot_tracks(meantracks_ve[ve_seg_select_rect,:10], ax=ax, color='w')
#                        plt.show()
#                
#                        
#                    # =============================================================================
#                    #       project tracks into 3D.
#                    # =============================================================================
#                        meantracks_ve_3D = unwrap_params_ve[meantracks_ve[...,0], meantracks_ve[...,1]]
#                        meantracks_epi_3D = unwrap_params_epi[meantracks_epi[...,0], meantracks_epi[...,1]]
#                        
#                        disps_ve_3D = meantracks_ve_3D[:,1:] - meantracks_ve_3D[:,:-1]; disps_ve_3D = disps_ve_3D.astype(np.float32)
#                        disps_epi_3D = meantracks_epi_3D[:,1:] - meantracks_epi_3D[:,:-1]; disps_epi_3D = disps_epi_3D.astype(np.float32)
#                        
##                        norm_disps = True
##                        
##                        # normalize the measurements.... based on the overall size of the embryo? approximated by the image crop size. 
##                        if norm_disps:
##                            disps_ve_3D[...,0] = disps_ve_3D[...,0]/embryo_im_shape[0]
##                            disps_ve_3D[...,1] = disps_ve_3D[...,1]/embryo_im_shape[1]
##                            disps_ve_3D[...,2] = disps_ve_3D[...,2]/embryo_im_shape[2]
##                            
##                            disps_epi_3D[...,0] = disps_epi_3D[...,0]/embryo_im_shape[0]
##                            disps_epi_3D[...,1] = disps_epi_3D[...,1]/embryo_im_shape[1]
##                            disps_epi_3D[...,2] = disps_epi_3D[...,2]/embryo_im_shape[2]
##                        
#                        
#                        disps_ve_2D = meantracks_ve[:,1:] - meantracks_ve[:,:-1]; disps_ve_2D = disps_ve_2D.astype(np.float32)
#                        disps_epi_2D = meantracks_epi[:,1:] - meantracks_epi[:,:-1]; disps_epi_2D = disps_epi_2D.astype(np.float32)
#                        
#                        norm_disps = True
#                        vol_factor = np.product(embryo_im_shape) ** (1./3)
#                        
#                        start_ = start_end_times[0]
#                        end_ = start_end_times[1]
#                        
#                        if end_ == len(vid_ve):
#                            end_ = end_ - 1
#                        
#                        # growth and volume factor correction 
#                        disps_ve_3D = disps_ve_3D / float(vol_factor) * temp_tform_scales[start_:end_][None,:,None]
#                        disps_epi_3D = disps_epi_3D / float(vol_factor) * temp_tform_scales[start_:end_][None,:,None]
#                        disps_ve_2D = disps_ve_2D * temp_tform_scales[start_:end_][None,:,None]
#                        disps_epi_2D = disps_epi_2D * temp_tform_scales[start_:end_][None,:,None]
#                        
#                        
#                        if 'LifeAct' in embryo_folder:
#                            print(embryo_folder)
##                            pass
#                        else:
#                            # temporal scaling.
#                            print('scale', embryo_folder)
#                            # scale because of the temporal timing. 
#                            disps_ve_3D = disps_ve_3D/2.
#                            disps_epi_3D = disps_epi_3D/2.
#                            disps_ve_2D = disps_ve_2D/2.
#                            disps_epi_2D = disps_epi_2D/2.
#                        
#                        all_embryo_vol_factors.append(vol_factor)
#                        
#                        
#                        # projecting the speed onto this ? 
#                        mean_ve_direction_3D = (disps_ve_3D[ve_seg_select_rect].mean(axis=1)).mean(axis=0)
#                        mean_ve_direction_3D = mean_ve_direction_3D/(np.linalg.norm(mean_ve_direction_3D) + 1e-8)  # normalise the vectors. 
#                        
#                        
#                        # average the 2D only to extract directions. 
#                        disps_ve_2D = disps_ve_2D.mean(axis=1)
#                        disps_epi_2D = disps_epi_2D.mean(axis=1) 
#                        
#                        # note some error here. -> not the inferred ve direction. 
#                        disps_ve_3D_component = np.hstack([(disps_ve_3D[ll].mean(axis=0)).dot(mean_ve_direction_3D) for ll in range(len(disps_ve_3D))])
#                        
##                        mean_epi_direction_3D = (disps_epi_2D[ve_seg_select_rect].mean(axis=1)).mean(axis=0)
##                        mean_epi_direction_3D = mean_epi_direction_3D/(np.linalg.norm(mean_epi_direction_3D) + 1e-8) 
#                        disps_epi_3D_component = np.hstack([(disps_epi_3D[ll].mean(axis=0)).dot(mean_ve_direction_3D) for ll in range(len(disps_epi_3D))])
#                        
#                        
##                        speed_ve_3D = np.linalg.norm(disps_ve_3D, axis=-1)
##                        speed_epi_3D = np.linalg.norm(disps_epi_3D, axis=-1)
#                        
#                        
#                        # what sort of parameters other than speed? (directionality?) in 3D? (mean speed calc is buggy)
#                        mean_speed_regions_ve = [np.linalg.norm(np.mean(np.mean(disps_ve_3D[reg], axis=1), axis=0)) for reg in uniq_polar_spixels]
#                        mean_speed_regions_epi = [np.linalg.norm(np.mean(np.mean(disps_epi_3D[reg], axis=1), axis=0)) for reg in uniq_polar_spixels]
##                        mean_speed_regions_ve = [np.mean(np.mean(np.linalg.norm(disps_ve_3D[reg], axis=-1), axis=1)) for reg in uniq_polar_spixels]
##                        mean_speed_regions_epi = [np.mean(np.mean(np.linalg.norm(disps_epi_3D[reg], axis=-1), axis=1)) for reg in uniq_polar_spixels]
#                        
#                        mean_direction_regions_ve = [np.arctan2(np.mean(disps_ve_2D[reg], axis=0)[...,1], np.mean(disps_ve_2D[reg], axis=0)[...,0]) for reg in uniq_polar_spixels]
#                        mean_direction_regions_epi = [np.arctan2(np.mean(disps_epi_2D[reg], axis=0)[...,1], np.mean(disps_epi_2D[reg], axis=0)[...,0]) for reg in uniq_polar_spixels]
#                        
#                        # account for directionality and speed. 
#                        mean_speed_regions_ve_component = [np.mean(disps_ve_3D_component[reg]) for reg in uniq_polar_spixels]
#                        mean_speed_regions_epi_component = [np.mean(disps_epi_3D_component[reg]) for reg in uniq_polar_spixels]
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
#                        all_embryo_shapes.append(embryo_im_shape)
#                    
#                        all_directions_regions_ve.append(mean_direction_regions_ve)
#                        all_directions_regions_epi.append(mean_direction_regions_epi)
#    
#    
