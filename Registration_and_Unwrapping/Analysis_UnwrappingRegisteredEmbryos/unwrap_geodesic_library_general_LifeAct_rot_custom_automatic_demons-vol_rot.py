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

def parse_condition_fnames_lat_unwrapped(vidfile):
    
    import os 
    import re 
    
    fname = os.path.split(vidfile)[-1]
    
    """ get the embryo number """
    emb_no = re.findall('Lat\d+', fname)

    if emb_no:
        emb_no = emb_no[0].split('Lat')[1]
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


def test_unique_break_segments(gradients, break_points, p_thresh=0.05):
    
    from scipy.stats import ttest_ind
    
    # add the start and end stretch and parse out the unique segments. 
    breaks = np.hstack([0, break_points, len(gradients)+1])
    break_gradients_list = []
    uniq_break_gradients = []
    uniq_breaks = []
    
    for ii in range(len(breaks)-1):
        
        break_start = breaks[ii]
        break_end = breaks[ii+1]
        
        break_grad = gradients[break_start:break_end]
        break_gradients_list.append(break_grad)
        
        if ii == 0:
            uniq_break_gradients.append([break_grad])
            uniq_breaks.append([[ii, breaks[ii], breaks[ii+1]]])
        else:
            # test if is unique. 
            last_gradient_set = np.hstack(uniq_break_gradients[-1])
            pval = ttest_ind(break_grad, last_gradient_set)[-1]
            
            print(pval)
            if pval <= p_thresh:
                # statistically different.
                uniq_break_gradients.append([break_grad])
                uniq_breaks.append([[ii, breaks[ii], breaks[ii+1]]])
            else:
                uniq_break_gradients[-1].append(break_grad)
                uniq_breaks[-1].append([ii, breaks[ii], breaks[ii+1]])
                

    return break_gradients_list, uniq_break_gradients, uniq_breaks


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
    
def remove_global_component_tracks_robust(tracks, mask_track=None, return_tforms=False):
    
    from skimage import transform as sktf
    from skimage.measure import ransac
        
    disps_frames = []
    tform_frames = []
    
    for ii in range(tracks.shape[1]-1):
    
#        model = sktf.EuclideanTransform()
        if mask_track is None:
#            tform_frame = sktf.estimate_transform('euclidean', XY[mask>0,:], XY[mask>0,:] + optflow[ii, mask>0, :])
            tform_frame, _ = ransac((tracks[:,ii], tracks[:,ii+1]), sktf.EuclideanTransform, min_samples=12,
                               residual_threshold=2., max_trials=100)
        else:
#            tform_frame = sktf.estimate_transform('euclidean', XY.reshape(-1,2), XY.reshape(-1,2) + optflow[ii].reshape(-1,2))
            tform_frame, _ = ransac((tracks[mask_track,ii], tracks[mask_track,ii+1]), sktf.EuclideanTransform, min_samples=12,
                               residual_threshold=2., max_trials=100)
        
#        estimated_tform = sktf.EuclideanTransform(translation=tform_frame.translation) # must require additionally the displacement. 
        pred_pts = sktf.matrix_transform(tracks[:,ii], tform_frame.params)
        
#        if ii ==0:
#            print(tform_frame.params)
        disp_frame_global = (pred_pts - tracks[:,ii]).copy()
        disp_frame = (tracks[:,ii+1] - tracks[:,ii]) - disp_frame_global
        disps_frames.append(disp_frame)
        
        if return_tforms:
            tform_frames.append(tform_frame.params)

    disps_frames = np.array(disps_frames)
    disps_frames = disps_frames.transpose(1,0,2)
    
    if return_tforms:
        tform_frames = np.array(tform_frames)   
        
        return disps_frames, tform_frames
    else:    
        return disps_frames
    
    
def compute_3D_displacements(meantracks, unwrap_3d_coords, correct_global=True, mask_track=None):
    
    if correct_global:
        meantrack_pos_t = meantracks[:,:-1].copy()
        meantrack_disp_t = remove_global_component_tracks_robust(meantracks, mask_track=mask_track, return_tforms=False)
        meantrack_pos_t_1 = (np.rint(meantrack_pos_t + meantrack_disp_t)).astype(np.int)
        meantrack_pos_t_1[...,0] = np.clip(meantrack_pos_t_1[...,0], 0, np.max(meantracks[...,0]))
        meantrack_pos_t_1[...,1] = np.clip(meantrack_pos_t_1[...,1], 0, np.max(meantracks[...,1]))
    else:
        meantrack_pos_t = meantracks[:,:-1].copy()
        meantrack_pos_t_1 = meantracks[:,1:].copy()
    
    meantrack_pos_t_3D = unwrap_3d_coords[meantrack_pos_t[...,0], meantrack_pos_t[...,1]].astype(np.float32)
    meantrack_pos_t_1_3D = unwrap_3d_coords[meantrack_pos_t_1[...,0], meantrack_pos_t_1[...,1]].astype(np.float32)
    
    meantrack_disps_3D = (meantrack_pos_t_1_3D - meantrack_pos_t_3D).astype(np.float32)
    
    return meantrack_disps_3D
    

def parse_temporal_scaling(tformfile):
    
    import scipy.io as spio 
    from transforms3d.affines import decompose44
    
    obj = spio.loadmat(tformfile)
    scales = np.hstack([1./decompose44(matrix)[2][0] for matrix in obj['tforms']])
    
    return scales    

def find_time_string(name):
    
    import re
    
    time_string = re.findall(r't\d+', name)
    time_string = time_string[0]
#    if len(time_string):
    time = int(time_string.split('t')[1])
        
    return time


def findPointNormals(points, nNeighbours, viewPoint=[0,0,0], dirLargest=True):
    
    """
    construct kNN and estimate normals from the local PCA
    
    reference: https://uk.mathworks.com/matlabcentral/fileexchange/48111-find-3d-normals-and-curvature
    """
    # construct kNN object to look for nearest neighbours. 
    from sklearn.neighbors import NearestNeighbors
    
    neigh = NearestNeighbors(n_neighbors=nNeighbours+1)
    neigh.fit(points)
    
    nn_inds = neigh.kneighbors(points, return_distance=False) # also get the distance, the distance is used for cov computation.
    nn_inds = nn_inds[:,1:] # remove self
    
    # find difference in position from neighbouring points (#technically this should be relative to the centroid of the patch!)
    # refine this computation. to take into account the central points. 
#    p = points[:,None,:] - points[nn_inds]
    p = points[nn_inds] - (points[nn_inds].mean(axis=1))[:,None,:]
    
    # compute covariance
    C = np.zeros((len(points), 6))
    C[:,0] = np.sum(p[:,:,0]*p[:,:,0], axis=1)
    C[:,1] = np.sum(p[:,:,0]*p[:,:,1], axis=1)
    C[:,2] = np.sum(p[:,:,0]*p[:,:,2], axis=1)
    C[:,3] = np.sum(p[:,:,1]*p[:,:,1], axis=1)
    C[:,4] = np.sum(p[:,:,1]*p[:,:,2], axis=1)
    C[:,5] = np.sum(p[:,:,2]*p[:,:,2], axis=1)
    C = C / float(nNeighbours)
    
    # normals and curvature calculation 
    normals = np.zeros(points.shape)
    curvature = np.zeros((len(points)))
    
    for i in range(len(points))[:]:
        
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
    points = points - np.array(viewPoint).ravel()[None,:]; # this is outward facing

    if dirLargest:
        idx = np.argmax(np.abs(normals), axis=1)
        dir = normals[np.arange(len(idx)),idx]*points[np.arange(len(idx)),idx] < 0;
    else:
        dir = np.sum(normals*points,axis=1) < 0;
    
    normals[dir,:] = -normals[dir,:];
    
    return normals, curvature

def preprocess_unwrap_pts(pts3D, im_array, rot_angle_3d=None, transpose_axis=None, clip_pts_to_array=True):
    
    import Registration.registration_new as reg
    import Geometry.geometry as geom
    
    pts = pts3D.copy()
    
    if transpose_axis is not None:
        pts = pts[...,transpose_axis]
        
    if rot_angle_3d is not None:
        
        rot_matrix = np.eye(3)
        if rot_angle_3d != 0:
            rot_matrix = geom.get_rotation_y(-rot_angle_3d/180.*np.pi)[:3,:3]
            
        pts = reg.warp_3D_transforms_xyz_similarity_pts(pts, 
                                                        translation=[0,0,0], 
                                                        rotation=rot_matrix, 
                                                        zoom=[1,1,1], 
                                                        shear=[0,0,0], 
                                                        center_tform=True, 
                                                        im_center = np.hstack(im_array.shape)/2.,
#                                                        im_center = pts.reshape(-1,pts.shape[-1]).mean(axis=0),
                                                        direction='B') # for pts is reverse of image. 

    if clip_pts_to_array==True:
        # clip the values to the array.
        pts[...,0] = np.clip(pts[...,0], 0, im_array.shape[0]-1)
        pts[...,1] = np.clip(pts[...,1], 0, im_array.shape[1]-1)
        pts[...,2] = np.clip(pts[...,2], 0, im_array.shape[2]-1)
    
    return pts



def warp_3D_transforms_xyz_similarity_pts(pts3D, translation=[0,0,0], 
                                          rotation=np.eye(3), 
                                          zoom=[1,1,1], shear=[0,0,0], 
                                          center_tform=True, 
                                          im_center=None, 
                                          direction='F'):

    from transforms3d.affines import compose 
    
    # compose the 4 x 4 homogeneous matrix. 
    tmatrix = compose(translation, rotation, zoom, shear)

    if center_tform:
        # im_center = np.array(im.shape)//2
        tmatrix[:-1,-1] = tmatrix[:-1,-1] + np.array(im_center)
        decenter = np.eye(4); decenter[:-1,-1] = -np.array(im_center)
        tmatrix = tmatrix.dot(decenter)

#    if direction == 'F':
#        print(tmatrix)
#    if direction == 'B':
#        print(np.linalg.inv(tmatrix))
        
    # first make homogeneous coordinates.
    xyz = np.vstack([(pts3D[...,0]).ravel().astype(np.float32), 
                      (pts3D[...,1]).ravel().astype(np.float32), 
                      (pts3D[...,2]).ravel().astype(np.float32),
                      np.ones(len(pts3D[...,0].ravel()), dtype=np.float32)])
    
    if direction == 'F':
        xyz_ = tmatrix.dot(xyz)
    if direction == 'B':
        xyz_ = (np.linalg.inv(tmatrix)).dot(xyz)
    
    pts3D_warp = (xyz_[:3].T).reshape(pts3D.shape)

    return pts3D_warp

def get_rotation_y(theta):
    
    R_z = np.zeros((4,4))
    R_z[-1:] = np.array([0,0,0,1])
    
    R_z[:-1,:-1] = np.array([[np.cos(theta), 0, np.sin(theta)],
                             [0, 1, 0], 
                             [-np.sin(theta), 0, np.cos(theta)]])
    
    return R_z

def preprocess_unwrap_pts_(pts3D, ref_pt, rot_angle_3d=None, transpose_axis=None):
    
    # import Registration.registration_new as reg
    # import Geometry.geometry as geom
    
    pts = pts3D.copy()
    
    if transpose_axis is not None:
        pts = pts[...,transpose_axis]
        ref_pt = ref_pt[transpose_axis]
        
    if rot_angle_3d is not None:
        
        rot_matrix = np.eye(3)
        if rot_angle_3d != 0:
            rot_matrix = get_rotation_y(-rot_angle_3d/180.*np.pi)[:3,:3]
            
        pts = warp_3D_transforms_xyz_similarity_pts(pts, 
                                                    translation=[0,0,0], 
                                                    rotation=rot_matrix, 
                                                    zoom=[1,1,1], 
                                                    shear=[0,0,0], 
                                                    center_tform=True, 
                                                    im_center = ref_pt,
#                                                        im_center = pts.reshape(-1,pts.shape[-1]).mean(axis=0),
                                                    direction='B') # for pts is reverse of image. 
    # # clip the values to the array.
    # pts[...,0] = np.clip(pts[...,0], 0, im_array.shape[0]-1)
    # pts[...,1] = np.clip(pts[...,1], 0, im_array.shape[1]-1)
    # pts[...,2] = np.clip(pts[...,2], 0, im_array.shape[2]-1)
    
    return pts

def find_time_string_MTMG(name):
    
#    print(name)
    if '_' in name:
        cands = name.split('_')
    elif '-' in name:
        cands = name.split('-')
    else:
        print('error in naming')

    time = []
    
    for c in cands:
        try:
#            if len(c) == 2:
            c = int(c)
            time = c
        except:
            continue
        
    return time


def isotropic_scale_pts(pts, scale):
    
    # isotropic scaling with respect to the centroid. (doesn't matter)
    pts_ = pts.reshape(-1,pts.shape[-1])
    centroid = np.nanmean(pts_[pts_[:,0]>0], axis=0)
    
    pts_diff = pts_ - centroid[None,:]
    pts_ = centroid[None,:] + pts_diff*scale
    pts_[pts_[:,0]==0] = 0
    
    pts_ = pts_.reshape(pts.shape)
    
    return pts_
    

def map_intensity_interp3(query_pts, grid_shape, I_ref):
    
    # interpolate instead of discretising to avoid artifact.
    from scipy.interpolate import RegularGridInterpolator
    
    #ZZ,XX,YY = np.indices(im_array.shape)
    spl_3 = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                     np.arange(grid_shape[1]), 
                                     np.arange(grid_shape[2])), 
                                     I_ref, method='linear', bounds_error=False, fill_value=0)
    
#    I_query = np.uint8(spl_3((query_pts[...,0], 
#                              query_pts[...,1],
#                              query_pts[...,2])))
    I_query = np.uint8(spl_3((query_pts[...,0], 
                                query_pts[...,1],
                                query_pts[...,2])))

    return I_query




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
    from mpl_toolkits.mplot3d import Axes3D 
    
#    from MOSES.Motion_Analysis import tracks3D_tools as tra3Dtools
    
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Set up a nearest neighbour model 
    from sklearn.neighbors import NearestNeighbors 
    import Geometry.meshtools as meshtools
    import Geometry.geometry as geom
    
    from skimage.exposure import equalize_hist
    import Registration.registration_new as reg
    import skimage.io as skio 
    import Utility_Functions.file_io as fio
    
    nbrs_model = NearestNeighbors(n_neighbors=9, algorithm='auto') # how many neighbours to include? 9 = 8-connected?
    
    
#    mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper'
    # mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper-Full-Rec'
    # mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Depth_Unwrap-NonDemons'

    # Create a new folder to save the rotated volumes out. 
    mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Depth_Unwrap-Demons_Vol'
    fio.mkdir(mastersavefolder)
    
    """
    start of algorithm
    """
    rootfolder = '/media/felix/Srinivas4/LifeAct'
#    rootfolder = '/media/felix/Srinivas4/MTMG-TTR' # not complete L910-270., L945, 
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX' # L 927, L930 -> nothing.....? 
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
    # rootfolder = '/media/felix/My Passport/Shankar-2/All_Embryos/Truncullin'
#    rootfolder = '/media/felix/My Passport/Shankar-2/All_Embryos/DMSO_Ctrl'
# =============================================================================
#     1. Load all embryos 
# =============================================================================
    embryo_folders = np.hstack([os.path.join(rootfolder,folder) for folder in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, folder)) and 'new_hex' not in folder])
    
    
    """
    load the staging file. 
    """
    manual_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/manual_staging_holly_matt.xlsx'
    manual_staging_tab_Matt_Holly = pd.read_excel(manual_staging_file)
    
# =============================================================================
#     1b. Load the staging file 
# =============================================================================
    # all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-LifeAct-smooth-lam5-correct_norot_consensus.xlsx', 
       #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-TTR-smooth-lam5-correct_norot_consensus.xlsx',
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-HEX-smooth-lam5-correct_norot_consensus.xlsx',
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-new-HEX-smooth-lam5-correct_norot_consensus.xlsx']
    all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-LifeAct-smooth-lam5-correct_norot_consensus.xlsx'] 
#    all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-new-HEX-smooth-lam5-correct_norot_consensus.xlsx'] 
    
    manual_staging_tab = pd.concat([pd.read_excel(staging_file) for staging_file in all_staging_files], ignore_index=True)
    all_stages = manual_staging_tab.columns[1:4]


# =============================================================================
#     1c. Load the angle file 
# =============================================================================
    # all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/LifeAct_polar_angles_AVE_classifier-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-TTR_polar_angles_AVE_classifier-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-HEX_polar_angles_AVE_classifier-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/new_hex_polar_angles_AVE_classifier-consensus.csv']
    all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/LifeAct_polar_angles_AVE_classifier-consensus.csv']
#    all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/new_hex_polar_angles_AVE_classifier-consensus.csv']
    angle_tab = pd.concat([pd.read_csv(angle_file) for angle_file in all_angle_files], ignore_index=True)
    
    
    
# =============================================================================
#     2. find valid embryo folders i.e. has a geodesic-rotmatrix. 
# =============================================================================
    n_spixels = 5000 # set the number of pixels. 
    smoothwinsize = 3
#    n_spixels = 5000
    
    # manual_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/manual_staging_holly_matt.xlsx'
    # manual_staging_tab = pd.read_excel(manual_staging_file)
    
    """
    set which experiment folder. 
    """
    all_embryos_plot = []
    diff_curves_all = []
    pre_all = []
    mig_all = []
    post_all = []
    
    pre_all_auto = []
    mig_all_auto = []
    post_all_auto = []
    
#    for embryo_folder in embryo_folders[-2:-1]:
#    for embryo_folder in embryo_folders[:]:
#    for embryo_folder in embryo_folders[5:6]:
#    for embryo_folder in embryo_folders[2:]: #L491
#    for embryo_folder in embryo_folders[-1:]: # 1 doesn't seem to have the particular output folder? ????
#    for embryo_folder in embryo_folders[3:4]: # L491 has been done. # L864 needs separate tilt correction too. 
    for embryo_folder in embryo_folders[1:]: # run through all of them#    for embryo_folder in embryo_folders[3:4]:
        
        # find the geodesic-rotmatrix. 
        unwrapped_folder = locate_unwrapped_im_folder(embryo_folder, key='geodesic-rotmatrix')
        
        saveanalysisfolder = os.path.join(mastersavefolder, os.path.split(embryo_folder)[-1]);
        fio.mkdir(saveanalysisfolder)
        
        
        if len(unwrapped_folder) == 1:
            print('processing: ', unwrapped_folder[0])
            
            """
            Locate the Demons folder and get the image. 
            """
            # locate the demons folder to obtain the x,y,z dimensions for normalization.         
#            embryo_im_folder = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step6_' in name or 'step6tf_' in name or 'step6tf-' in name or 'step6-' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
            
#            # specifically want the hex + mtmg separately. 
#            embryo_folder_alt = embryo_folder.replace(rootfolder, '/media/felix/My Passport/Srinivas4/Hex')
#            embryo_im_folder = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step5_' in name or 'step5tf_' in name or 'step5tf-' in name or 'step5-' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
            embryo_im_folder = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step6_' in name or 'step6tf_' in name or 'step6tf-' in name or 'step6-' in name) and os.path.isdir(os.path.join(embryo_folder, name))])

            # check how many there are.
            if len(embryo_im_folder) > 1:
                embryo_im_folder = [f for f in embryo_im_folder if '_tc' in os.path.split(f)[-1]]


#            embryo_im_folder = np.hstack([os.path.join(embryo_folder_alt, name) for name in os.listdir(embryo_folder_alt) if ('step5_' in name or 'step5tf_' in name or 'step5tf-' in name or 'step5-' in name) and os.path.isdir(os.path.join(embryo_folder_alt, name))])
#            embryo_im_folder_hex = [f for f in embryo_im_folder if 'hex' in os.path.split(f)[-1]]#[0]
#            embryo_im_folder_mtmg = [f for f in embryo_im_folder if 'mtmg' in os.path.split(f)[-1]]#[0]
##            
##            embryo_im_folder_hex = embryo_im_folder_hex[0].replace( '/media/felix/My Passport/Srinivas4/Hex/L928_Emb2')
##            embryo_im_folder_hex = [f for f in embryo_im_folder.replace(rootfolder, '/media/felix/My Passport/Srinivas4/Hex') if 'hex' in os.path.split(f)[-1]]#[0]
##            embryo_im_folder_mtmg = [f for f in embryo_im_folder.replace(rootfolder, '/media/felix/My Passport/Srinivas4/Hex') if 'mtmg' in os.path.split(f)[-1]]
            embryo_im_folder_files = np.hstack(glob.glob(os.path.join(embryo_im_folder[0], '*.tif')))
            sort_order = np.argsort(embryo_im_folder_files)
            embryo_im_folder_files = embryo_im_folder_files[sort_order]
            
#            embryo_im_folder_files_hex = np.hstack(glob.glob(os.path.join(embryo_im_folder_hex[0], '*.tif')))
#            sort_order = np.argsort(embryo_im_folder_files_hex)
#            embryo_im_folder_files_hex = embryo_im_folder_files_hex[sort_order]
#            
#            embryo_im_folder_files_mtmg = np.hstack(glob.glob(os.path.join(embryo_im_folder_mtmg[0], '*.tif')))
#            embryo_im_folder_files_mtmg = embryo_im_folder_files_mtmg[sort_order]

            print(embryo_im_folder_files)
            print('=====')

            im_array = fio.read_multiimg_PIL(embryo_im_folder_files[0]) # this needs transposition ....? 
            embryo_im_shape = np.hstack(im_array.shape)
            embryo_im_shape = embryo_im_shape[[1,0,2]]#.transpose([1,0,2])
            
#            """
#            Load the temporal  registered result in order to debug the transformations. 
#            """
#            """
#            for LifeAct
#            """
#            embryo_im_folder_reg = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step5' in name or 'step4' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
#
##            embryo_im_folder_reg = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step5' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
#            embryo_im_folder_reg = np.sort(embryo_im_folder_reg)[::-1]
#            
#            """
#            for LifeAct
#            """
##            embryo_im_folder_reg = embryo_im_folder_reg[0]
##            embryo_im_folder_reg = [folder for folder in embryo_im_folder_reg if 'mtmg' in os.path.split(folder)[-1] or 've' in os.path.split(folder)[-1]][0] 
#            """
#            for MTMG
#            """
##            embryo_im_folder_reg = [folder for folder in embryo_im_folder_reg if 'mtmg' in os.path.split(folder)[-1] or 've' in os.path.split(folder)[-1]][0] 
#            """
#            for Hex
#            """
#            embryo_im_folder_reg = [folder for folder in embryo_im_folder_reg if 'mtmg' in os.path.split(folder)[-1] or 've' in os.path.split(folder)[-1]][0] 
##
##            
#            embryo_im_files_time_reg_files = np.hstack(glob.glob(os.path.join(embryo_im_folder_reg, '*.tif')))
##            frame_nos = np.hstack([find_time_string_MTMG(os.path.split(f)[-1].split('.tif')[0]) for f in embryo_im_files_time_reg_files])
##            frame_nos = np.hstack([find_time_string(os.path.split(f)[-1].split('.tif')[0]) for f in embryo_im_files_time_reg_files])
##            embryo_im_files_time_reg_files = embryo_im_files_time_reg_files[np.argsort(frame_nos)]
#            embryo_im_files_time_reg_files = np.sort(embryo_im_files_time_reg_files)
#            embryo_im_folder_reg = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('demons' in name or 'step6' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
#            embryo_im_folder_reg = [folder for folder in embryo_im_folder_reg if 'mtmg' in os.path.split(folder)[-1] or 've' in os.path.split(folder)[-1]][0] 
#            embryo_im_folder_reg = [folder for folder in embryo_im_folder_reg if 'mtmg' in os.path.split(folder)[-1] or 've' in os.path.split(folder)[-1]][0] 
#            embryo_im_folder_reg = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('demons' in name or 'step5' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
#             embryo_im_folder_reg = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if (('step5' in name) or ('step4' in name)) and os.path.isdir(os.path.join(embryo_folder, name))])
# #            embryo_im_folder_reg = [folder for folder in embryo_im_folder_reg if 'mtmg' in os.path.split(folder)[-1] or 've' in os.path.split(folder)[-1]][0] 
            
# #            embryo_im_folder_reg = [folder for folder in embryo_im_folder_reg if 've' in os.path.split(folder)[-1] or 'mtmg' in os.path.split(folder)[-1]]
#             embryo_im_folder_reg = [folder for folder in embryo_im_folder_reg if 've' in os.path.split(folder)[-1] or 'hex' in os.path.split(folder)[-1]]
#             embryo_im_folder_reg = embryo_im_folder_reg[-1]
            
            
#             # for L932 need to go one deeper. 
#             embryo_im_folder_reg = np.hstack([os.path.join(embryo_im_folder_reg, name) for name in os.listdir(embryo_im_folder_reg) if (('step5' in name) or ('step4' in name)) and os.path.isdir(os.path.join(embryo_im_folder_reg, name))])
#             embryo_im_folder_reg = [folder for folder in embryo_im_folder_reg if 've' in os.path.split(folder)[-1] or 'hex' in os.path.split(folder)[-1]]
#             embryo_im_folder_reg = embryo_im_folder_reg[-1]
#             embryo_im_files_time_reg_files = np.hstack(glob.glob(os.path.join(embryo_im_folder_reg, '*.tif')))
            
# #             need to parse the time from this. 
#             embryo_file_times = np.hstack([find_time_string_MTMG(os.path.split(ff)[-1].split('.tif')[0]) for ff in embryo_im_files_time_reg_files])
# #            embryo_file_times = np.hstack([find_time_string(os.path.split(ff)[-1].split('.tif')[0]) for ff in embryo_im_files_time_reg_files])
# #            embryo_im_folder_files = np.sort(embryo_im_files_time_reg_files)
#             embryo_im_folder_files = embryo_im_files_time_reg_files[np.argsort(embryo_file_times)]
            
            
            unwrapped_folder = unwrapped_folder[0]
            unwrapped_params_folder = unwrapped_folder.replace('geodesic-rotmatrix', 
                                                               'unwrap_params_geodesic-rotmatrix')
            # determine the savefolder. 
            saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned')
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-5000_VE_seg_directions')
            # savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels))
            
            unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))
            
            # =============================================================================
            #   get scaling transforms from registration           
            # =============================================================================
            """
            get the temporal transforms and extract the scaling parameters. 
            """
            temp_tform_files = glob.glob(os.path.join(embryo_folder, '*temp_reg_tforms*.mat'))
            print(len(temp_tform_files))
            
            assert(len(temp_tform_files) > 0)
            
            # temp_tform_scales = np.vstack([parse_temporal_scaling(tformfile) for tformfile in temp_tform_files])
            # temp_tform_scales = np.prod(temp_tform_scales, axis=0)
            temp_tform_scales = []
            for tformfile in temp_tform_files:
                if 'rev' in os.path.split(tformfile)[-1]:
                    temp_tform_scales.append(parse_temporal_scaling(tformfile)[::-1])
                else:
                    temp_tform_scales.append(parse_temporal_scaling(tformfile))
#            temp_tform_scales = np.vstack([parse_temporal_scaling(tformfile) for tformfile in temp_tform_files])
            temp_tform_scales = np.prod(temp_tform_scales, axis=0) # this line is just in case there were multiple transformations. 
                    
            plt.figure()
            plt.plot(temp_tform_scales)
            plt.show()

            if 'MTMG-TTR' in rootfolder or 'LifeAct' in rootfolder or 'DMSO' in rootfolder:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
            else:
                if 'Truncullin' in rootfolder:
                    unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lat_unwrapped(ff)) for ff in unwrap_im_files])
                    paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
                else:
                    unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_hex_unwrapped(ff)) for ff in unwrap_im_files])
                    paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_hex_embryo(unwrap_im_files, unwrap_meta_files)
            

            # iterate over the pairs # just do for one angle!. 
            for ii in tqdm(range(len(paired_unwrap_file_condition))[:1]): # only do it for the first angle to save time. 
#            for ii in tqdm(range(len(paired_unwrap_file_condition))[:1]):
                
                unwrapped_condition = paired_unwrap_file_condition[ii]
                print(unwrapped_condition)
                
                if 'MTMG-TTR' in rootfolder or 'LifeAct' in rootfolder or 'Truncullin' in rootfolder or 'DMSO' in rootfolder:
                    ve_unwrap_file, epi_unwrap_file = paired_unwrap_files[ii]
                else:
                    ve_unwrap_file, epi_unwrap_file, hex_unwrap_file = paired_unwrap_files[ii]
                        
            # =============================================================================
            #        Load the relevant videos.          
            # =============================================================================
                vid_ve = fio.read_multiimg_PIL(ve_unwrap_file)
                vid_epi = fio.read_multiimg_PIL(epi_unwrap_file)
                vid_epi_resize = sktform.resize(vid_epi, vid_ve.shape, preserve_range=True)
                
                vid_size = vid_ve[0].shape
                
                if 'MTMG-HEX' in rootfolder:
                    vid_hex = fio.read_multiimg_PIL(hex_unwrap_file)
                    
            # =============================================================================
            #       Load the relevant saved pair track file.               
            # =============================================================================
                trackfile = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')
                
                meantracks_ve = spio.loadmat(trackfile)['meantracks_ve']
                meantracks_epi = spio.loadmat(trackfile)['meantracks_epi']
                proj_condition = spio.loadmat(trackfile)['condition'][0].split('-')
                proj_type = proj_condition[-1]
                  
                
                if proj_type =='polar':
#                # =============================================================================
#                #       Load the relevant manual staging file from Holly and Matt.        
#                # =============================================================================
#                    """
#                    create the lookup embryo name
#                    """
#                    embryo_name = proj_condition[0] + '_' + proj_condition[1]
#                    print(embryo_name)
#
#                    # if proj_condition[1] == '':
#                    #     embryo_name = proj_condition[0]
#                    # else:
#                    #     embryo_name = proj_condition[0]+'_E'+proj_condition[1].split('Emb')[1]
#                    
#                    # table_emb_names = np.hstack([str(s) for s in manual_staging_tab['Embryo'].values])
#                    # select_stage_row = manual_staging_tab.loc[table_emb_names==embryo_name]
#                    
#                    # pre = parse_start_end_times(select_stage_row['Pre-migration'])
#                    # mig = parse_start_end_times(select_stage_row['Migration to Boundary'])
#                    # post = parse_start_end_times(select_stage_row['Post-Boundary'])
#                    
#                    # pre_all.append(pre)
#                    # mig_all.append(mig)
#                    # post_all.append(post)
#                    # print(pre, mig, post)#
                # =============================================================================
                #       Load the relevant manual staging file from Holly and Matt.        
                # =============================================================================
#                    """
#                    create the lookup embryo name
#                    """
#                    embryo_name = proj_condition[0] + '_' + proj_condition[1]
#                    print(embryo_name)

#                    if proj_condition[1] == '':
#                        embryo_name = proj_condition[0]
#                    else:
#                        embryo_name = proj_condition[0]+'_E'+proj_condition[1].split('Emb')[1]
                    embryo_name = os.path.split(embryo_folder)[-1]
                    
                    table_emb_names = np.hstack([str(s) for s in manual_staging_tab['Embryo'].values])
                    select_stage_row = manual_staging_tab.loc[table_emb_names==embryo_name]
                    
                    pre = parse_start_end_times(select_stage_row['Pre-migration'])
                    mig = parse_start_end_times(select_stage_row['Migration to Boundary'])
                    post = parse_start_end_times(select_stage_row['Post-Boundary'])
                    
                    
                # =============================================================================
                #     Assess the need to invert the embryos ... the correction angle was derived after correcting for the embryo inversion.         
                # =============================================================================
                    # get the inversion status from the manual
                    if proj_condition[1] == '':
                        manual_embryo_name = proj_condition[0]
                    else:
                        manual_embryo_name = proj_condition[0]+'_E'+proj_condition[1].split('Emb')[1]
                        
                    table_emb_names_Matt_Holly = np.hstack([str(s) for s in manual_staging_tab_Matt_Holly['Embryo'].values])
                    select_stage_row_Matt_Holly = manual_staging_tab_Matt_Holly.loc[table_emb_names_Matt_Holly==manual_embryo_name]
                
                    embryo_inversion = select_stage_row_Matt_Holly['Inverted'].values[0] > 0 
                    
                    print('embryo_inversion: ;', embryo_inversion)
                    
                # =============================================================================
                #      Get the rotation angle.               
                # =============================================================================
                    # specify the angle of rotation and the timepoint. 
                    rot_angle = int(proj_condition[-2]) # this is the second value in the unwrapped condition. 

                    print('Rotation angle: ', rot_angle)
                    print('---------------')
                    
                    
                # =============================================================================
                #       Rotate the volume to align with the unwrapping coordinates.           
                # =============================================================================
                    """
                    doesn't work well for some reason with the expected rotational transformations, have no clue what Holly has done to mess this up. 
                    """
                    if rot_angle  != 0:
                        print('loading rotation angles')
                        rottformfile = os.path.join(embryo_folder, 'step6_%s_rotation_tforms.mat' %(str(rot_angle).zfill(3)))
                        print(rottformfile)
                        rottform = spio.loadmat(rottformfile)['tforms'][0]
#                        rottform = np.linalg.inv(rottform)
                        print(rottform)
                        
#                        unwrap_params_ve = reg.warp_3D_xyz_tmatrix_pts(unwrap_params_ve[...,[1,0,2]], rottform)
#                        unwrap_params_epi = reg.warp_3D_xyz_tmatrix_pts(unwrap_params_epi[...,[1,0,2]], rottform)
                        
                    else:
                        rottform = np.eye(4)
                        
                    print('====')
                    print(rottform)
                    print('====')
#                        unwrap_params_ve = unwrap_params_ve[...,[1,0,2]]
#                        unwrap_params_epi = unwrap_params_epi[...,[1,0,2]]
                    
                    
                # =============================================================================
                #       Load the 3D demons mapping file coordinates.        
                # =============================================================================
                # load the associated unwrap param files depending on polar/rect coordinates transform. 
                    unwrap_param_file_ve = ve_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
                    unwrap_params_ve = spio.loadmat(unwrap_param_file_ve); unwrap_params_ve = unwrap_params_ve['ref_map_%s_xyz' %(proj_type)]
    #                unwrap_params_ve_polar = spio.loadmat(unwrap_param_file_ve); unwrap_params_ve_polar = unwrap_params_ve_polar['ref_map_'+'polar'+'_xyz']
                    
                    if proj_type == 'rect':
                        unwrap_params_ve = unwrap_params_ve[::-1] # reverse ....
    #                    unwrap_params_epi = unwrap_params_epi[::-1]
                        
                    unwrap_param_file_epi = epi_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
                    unwrap_params_epi = spio.loadmat(unwrap_param_file_epi); unwrap_params_epi = unwrap_params_epi['ref_map_%s_xyz' %(proj_type)]
    #                unwrap_params_epi_polar = spio.loadmat(unwrap_param_file_ve); unwrap_params_epi_polar = unwrap_params_epi_polar['ref_map_'+'polar'+'_xyz']
                    
                    unwrap_params_epi = sktform.resize(unwrap_params_epi, unwrap_params_ve.shape, preserve_range=True) # make sure to resize. 


                    # smooth both of theses. 
                    plt.figure()
                    plt.imshow(unwrap_params_ve[...,0])
                    
                    plt.figure()
                    plt.imshow(unwrap_params_ve[...,1])
                    
                    plt.figure()
                    plt.imshow(unwrap_params_ve[...,2])
                    
                    plt.show()
                    
                # =============================================================================
                #       Load the AVE movement angle              
                # =============================================================================
#                    if proj_type == 'polar':
#                        ve_angle_move = angle_tab.loc[angle_tab['polar_conditions'].values == '-'.join(proj_condition)]['correct_angles_vertical']
#                    if proj_type == 'rect':
#                        ve_angle_move = angle_tab.loc[angle_tab['rect_conditions'].values == '-'.join(proj_condition)]['correct_angles_vertical']
                    if proj_type == 'polar':
                        ve_angle_move = angle_tab.loc[angle_tab['Embryo'].values == '-'.join(proj_condition)]['Angle_consenus']
                    if proj_type == 'rect':
                        ve_angle_move = angle_tab.loc[angle_tab['Embryo'].values == '-'.join(proj_condition)]['Angle_consenus']
                        
#                    ve_angle = ve_angle_move.values[0]
                    ve_angle_move = ve_angle_move.values[0] # this is given as the correction angle anyway. 
                    # ve_angle_move = 90 - ve_angle_move.values[0] # to get back to the rectangular angle coordinate? 
                    # ve_angle = ve_angle_move.values[0]
                    print('ve movement direction: ', ve_angle_move)
                
                # # =============================================================================
                # #    Establish the mapping between polar and planar unwrappings and save this. (redundant now)              
                # # =============================================================================
                #     nbrs_model_3D = NearestNeighbors(n_neighbors=1, algorithm='auto')
                #     nbrs_model_3D.fit(unwrap_params_ve_polar.reshape(-1,3))
                    
                #     neighbours_rect_polar_id = nbrs_model_3D.kneighbors(unwrap_params_ve.reshape(-1,3))
                #     val_dist_img = neighbours_rect_polar_id[1].reshape(unwrap_params_ve.shape[:2])
                #     val_dist_img_binary = neighbours_rect_polar_id[0].reshape(unwrap_params_ve.shape[:2]) <= 5
                #     val_dist_ijs = np.unravel_index(val_dist_img,unwrap_params_ve_polar.shape[:2] )
                    
                #     # these are the mapping parameters. 
                #     mapped_ij = [val_dist_ijs[0], 
                #                  val_dist_ijs[1]]
                
                
                # =============================================================================
                #   Make absolute sure to rotate the unwrap_params into original image coordinates.   
                #   We should get the actual rotation matrices used? -> I have no idea why the translation is so wrong!. 
                # =============================================================================
#                    unwrap_params_ve = preprocess_unwrap_pts(unwrap_params_ve, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2]) 
#                    unwrap_params_epi = preprocess_unwrap_pts(unwrap_params_epi, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2])
                
                
#                    # option 1: isotropic scale the points (with respect to the centre of the VE volume. )
                    unwrap_params_ve_2 = isotropic_scale_pts(unwrap_params_ve, scale=1.25)
                    proj_dir_ve_epi = unwrap_params_ve - unwrap_params_ve_2
                    
#                    min_dist = np.nanmin(np.linalg.norm(proj_dir_ve_epi, axis=-1))
                
##                    # option 2: projecting based the proportional alignment of epi and VE.  # least distortion. 
#                    proj_dir_ve_epi = unwrap_params_epi - unwrap_params_ve
                    
                    # need to smooth this... ?
                    
                    
                    proj_dir_ve_epi = proj_dir_ve_epi / (np.linalg.norm(proj_dir_ve_epi, axis=-1)[...,None] + 1e-8) * 1.5
#                    proj_dir_ve_epi = proj_dir_ve_epi / (np.nanmean(np.linalg.norm(proj_dir_ve_epi, axis=-1)) + 1e-8) # this is the min? 
#                    proj_dir_ve_epi = proj_dir_ve_epi / (50 + 1e-8) 
                    # we just want the mean 
                    
            
                    
#                 # =============================================================================
#                 #   Load the demons displacements at the point of the embryo. corresponding to unwrap_params coordinates.        
#                 # =============================================================================
                
#                     try:
#                         demons_rev_folder = os.path.join(embryo_folder, 'unwrapped', 'geodesic_VE_Epi_matched-resize-rotmatrix_demons_revmap', 
#                                                          os.path.split(unwrap_param_file_ve)[-1].split('_unwrap_params')[0] + '_unwrap_align_params_geodesic')
#                         demons_rev_warp_files = np.hstack(glob.glob(os.path.join(demons_rev_folder, '*.mat')))
#                     except:
#                         try:
#                             demons_rev_folder = os.path.join(embryo_folder, 'unwrapped', 'geodesic_VE_Epi_matched-resize-rotmatrix_demons_revmap', 
#                                                              os.path.split(unwrap_param_file_ve)[-1].split('-unwrap_params')[0] + '_unwrap_align_params_geodesic')
                   
#                             demons_rev_warp_files = np.hstack(glob.glob(os.path.join(demons_rev_folder, '*.mat')))
#                         except:
# #                            demons_rev_folder = os.path.join(embryo_folder, 'unwrapped', 'geodesic_VE_Epi_matched-resize-rotmatrix_demons_revmap', 
# #                                                             os.path.split(unwrap_param_file_ve)[-1].split('-unwrap_params_')[0] + '_unwrap_align_params_geodesic')
                   
# #                            demons_rev_warp_files = np.hstack(glob.glob(os.path.join(demons_rev_folder, '*.mat')))
#                             demons_rev_folder = os.path.join(embryo_folder, 'unwrapped', 'geodesic_VE_Epi_matched-resize-rotmatrix_demons_revmap', 
#                                                              os.path.split(unwrap_param_file_ve)[-1].split('_unwrap_params-')[0] + '_unwrap_params-geodesic')
                   
#                             demons_rev_warp_files = np.hstack(glob.glob(os.path.join(demons_rev_folder, '*.mat')))
                            
#                     demons_times = np.hstack([find_time_string(os.path.split(ff)[-1]) for ff in demons_rev_warp_files])
#                     demons_rev_warp_files = demons_rev_warp_files[np.argsort(demons_times)]
    
                # =============================================================================
                #    Iterate over the demon files put back the global scaling. 
                # =============================================================================  
                    
                    n_frames = len(embryo_im_folder_files)
#                    # all_demons_ve_pts = [] # remain to be corrected for with growth.
#                    # all_demons_epi_pts = []
#                    
#                    # all_dir_r_ve = []
#                    # all_dir_r_epi = []
#                    # all_dir_theta_ve = []
#                    # all_dir_theta_epi = []
#                    # all_dir_normal_ve = []
#                    # all_dir_normal_epi = []
#
#                    # want to take 20 um.... either side...
#                    voxel_size = 0.3630 #um
#                    depth = 30 #um # this is the maximum but we don't need to cut quite so much out.... 
#                    max_depth_pixels = np.rint(depth/voxel_size)
#                    min_disp_depth = -max_depth_pixels//2
#                    max_disp_depth = max_depth_pixels
#                    depth_range = np.linspace(min_disp_depth, max_disp_depth, int(max_disp_depth-min_disp_depth) + 1)
#                    # construct the lookup coordinate array. and save this directly. 
#                    unwrap_params_ve_depth = []
#                    
#                    for dist_ii in depth_range:
#                        
#                        xyz = unwrap_params_ve + dist_ii*proj_dir_ve_epi
#                        unwrap_params_ve_depth.append(xyz)
#                        
#                    unwrap_params_ve_depth = np.array(unwrap_params_ve_depth)
#
#                    # save this now to a new folder .... and a different unwrap_params name. 
                    saveprojfolder = os.path.join(saveanalysisfolder, unwrapped_condition) 
                    fio.mkdir(saveprojfolder)

                    saveprojfolder_vol = os.path.join(saveprojfolder, 'Volume')
                    fio.mkdir(saveprojfolder_vol)

#                    saveprojfile_unwrap_depth = unwrap_param_file_ve.replace(unwrapped_params_folder, saveprojfolder)
#                    print(saveprojfile_unwrap_depth)

#                    spio.savemat(saveprojfile_unwrap_depth, 
#                                    {'unwrap_param_file_ve':unwrap_param_file_ve,
#                                     'unwrap_param_file_epi':unwrap_param_file_epi, 
#                                     'voxel_size':voxel_size,
#                                     'depth':depth, 
#                                     'max_depth_pixels': max_depth_pixels, 
#                                     'depth_range': depth_range, 
#                                     'unwrap_params_ve_depth':unwrap_params_ve_depth.astype(np.float32)})
#                    # 'demons_remapped_pts-'+unwrapped_condition+'.mat')
#                    
##                    saveprojfolder_hex = os.path.join(saveprojfolder, 'HEX'); fio.mkdir(saveprojfolder_hex)
##                    saveprojfolder_mtmg = os.path.join(saveprojfolder, 'MTMG'); fio.mkdir(saveprojfolder_mtmg)
#                   center = np.hstack(unwrap_params_ve.shape[:2])//2
                    # rotate the unwrap_params rotate.
                    # =============================================================================
                    #       AVE rotate.....                          
                    # =============================================================================
#                        hex_depth_vid_rot_AVE = np.array([np.array([sktform.rotate(vv, 
#                                                				angle=(90-ve_angle_move), 
#                                                				preserve_range=True, cval=0) for vv in hex_depth_vid[tt]]) for tt in np.arange(len(hex_depth_vid))])
                    if embryo_inversion==True:
                            
                        # im_array_ve_rot = im_array_ve_rot[:,:,::-1]
                        if proj_type == 'rect':
                            unwrap_params_ve = unwrap_params_ve[:,::-1] # reverse x direction 
                            unwrap_params_epi = unwrap_params_epi[:,::-1]
                        else:
                            unwrap_params_ve = unwrap_params_ve[::-1] # reverse y direction
                            unwrap_params_epi = unwrap_params_epi[::-1]
                        
                        # properly invert the 3D. 
                        unwrap_params_ve[...,2] = embryo_im_shape[2] - unwrap_params_ve[...,2]
                        unwrap_params_epi[...,2] = embryo_im_shape[2] - unwrap_params_epi[...,2]



                    # was minus
#                    angle_rot = -(90-ve_angle_move) # will this work? # make this anticlockwise? 
                    angle_rot = ve_angle_move
                    rot_matrix_rot = geom.get_rotation_x(angle_rot/180.*np.pi)[:3,:3] # which is the correct way to rotate? 
                    
                    center = np.hstack(unwrap_params_ve.shape[:2])//2
                    ref_pt_3D_center = unwrap_params_ve[center[0], center[1]]
                    
                    
                    
#                    # i think the problem might still be this ? bit... -> this rotation is centered on the reference point....
#                    unwrap_params_ve_AVE_rot = np.concatenate([sktform.rotate(unwrap_params_ve[...,ch], 
#                                                                              angle_rot, preserve_range=True)[...,None] for ch in range(unwrap_params_ve.shape[-1])], axis=-1)
#                    unwrap_params_epi_AVE_rot = np.concatenate([sktform.rotate(unwrap_params_ve[...,ch], 
#                                                                              angle_rot, preserve_range=True)[...,None] for ch in range(unwrap_params_ve.shape[-1])], axis=-1)
#                    

                    unwrap_params_ve_AVE_rot = preprocess_unwrap_pts_(unwrap_params_ve, 
                                                                        ref_pt=ref_pt_3D_center, # change this to the image center. 
#                                                                        ref_pt=np.hstack(embryo_im_shape/2.),
                                                                        rot_angle_3d=angle_rot, 
                                                                        transpose_axis=[1,0,2])[...,[1,0,2]]

                    unwrap_params_epi_AVE_rot = preprocess_unwrap_pts_(unwrap_params_epi, 
#                                                                        ref_pt=np.hstack(embryo_im_shape/2.),
                                                                        ref_pt=ref_pt_3D_center,
                                                                        rot_angle_3d=angle_rot, 
                                                                        transpose_axis=[1,0,2])[...,[1,0,2]]
                    
                    #     do we still need to rotate  in 2D??????? 
                    # rotate in 2D!!!! [essential to get the coorect cross section when we later index] -> annnnnnd this is in the same direction ass rotation.!
                    
                    # i think the problem might still be this ? bit... -> this rotation is centered on the reference point....
                    unwrap_params_ve_AVE_rot = np.concatenate([sktform.rotate(unwrap_params_ve_AVE_rot[...,ch], 
                                                                              angle_rot, preserve_range=True)[...,None] for ch in range(unwrap_params_ve_AVE_rot.shape[-1])], axis=-1)
                    unwrap_params_epi_AVE_rot = np.concatenate([sktform.rotate(unwrap_params_epi_AVE_rot[...,ch], 
                                                                              angle_rot, preserve_range=True)[...,None] for ch in range(unwrap_params_ve_AVE_rot.shape[-1])], axis=-1)
                    
                # =============================================================================
                #       Save out the images and unwrap params                      
                # =============================================================================    
                    # save out. 
                    # mastersavefolder

                    saveprojfile_unwrap_ve = unwrap_param_file_ve.replace(unwrapped_params_folder, saveprojfolder)
                    # saveprojfile_unwrap_epi = unwrap_param_file_epi.replace(unwrapped_params_folder, saveprojfolder)
                    print(saveprojfile_unwrap_ve)

        
                    spio.savemat(saveprojfile_unwrap_ve, 
                               {'unwrap_param_file_ve':unwrap_param_file_ve,
                                'unwrap_param_file_epi':unwrap_param_file_epi, 
#                                'voxel_size':voxel_size,
                                've_angle': ve_angle_move,
                                'unwrap_params_ve_AVE': unwrap_params_ve_AVE_rot.astype(np.float32),
                                'unwrap_params_epi_AVE': unwrap_params_epi_AVE_rot.astype(np.float32)})
#    
###                    print(len(temp_tform_scales), len(demons_rev_warp_files))
    
#                    from skimage.exposure import equalize_hist
                    for tt in tqdm(range(n_frames)[:]):
                        
#                        im_array = fio.read_multiimg_PIL(embryo_im_files_time_reg_files[tt])
                        
#                        demons_dxyz_ve_pt = unwrap_params_ve.copy() # show we have the correct shape!. 
#                        demons_dxyz_epi_pt = unwrap_params_epi.copy()
#                        
#                        # # properly put back the scales. 
#                        # demons_dxyz_ve_pt = isotropic_scale_pts(demons_dxyz_ve + unwrap_params_ve,  temp_tform_scales[tt])
#                        # demons_dxyz_epi_pt = isotropic_scale_pts(demons_dxyz_epi + unwrap_params_epi, temp_tform_scales[tt])
#                        
#                        print(demons_dxyz_ve_pt.max(), demons_dxyz_ve_pt.min())
##                        print( (demons_dxyz_ve * temp_tform_scales[tt] )).max()
#                        
##                        demons_dxyz_ve_pt = unwrap_params_ve#[...,[1,0,21]]
##                        demons_dxyz_epi_pt = unwrap_params_epi#[...,[1,0,2]]
#                        
##                        if tt == 0:
##                        print(embryo_im_files_time_reg_files[tt])
##                        im_array = fio.read_multiimg_PIL(embryo_im_files_time_reg_files[tt])
                        
                        print(embryo_im_folder_files[tt])
#                        im_array_hex = fio.read_multiimg_PIL(embryo_im_folder_files_hex[tt]) # read this. 
#                        im_array_mtmg = fio.read_multiimg_PIL(embryo_im_folder_files_mtmg[tt]) # read this. 
                        im_array_ve = fio.read_multiimg_PIL(embryo_im_folder_files[tt])
                        
                        # =============================================================================
                        #   Rotation of the volume.                    
                        # =============================================================================
#                        if rot_angle != 0: 
#                            rot_matrix = geom.get_rotation_y(rot_angle/180.*np.pi)[:3,:3]
#                        else:
#                            rot_matrix = np.eye(3)
                
#                t1 = time.time()
                        if rot_angle != 0: 
                            im_array_ve_rot = reg.warp_3D_transforms_xyz_similarity(im_array_ve, 
                                                                            rotation = rottform[:3,:3],
                                                                            zoom = np.hstack([1,1,1]),
                                                                            center = None, 
                                                                            center_tform=True,
                                                                            direction='F',
                                                                            pad=None)
                            
                        else:
                            im_array_ve_rot = im_array_ve.copy()
#                        im_array_ve_rot = im_array_ve.transpose((1,0,2))
#                            im_array = apply_affine_tform(im_array, rottform)
                        im_array_ve_rot = im_array_ve_rot.transpose(1,0,2) # this is now .... more in the line of our current. cross-section. 

    
                    # =============================================================================
                    #       Embryo inversion.                     
                    # =============================================================================
                        if embryo_inversion==True:
                            
                            print('inverting embryo....')
                            im_array_ve_rot = im_array_ve_rot[:,:,::-1]
                            
                            # if proj_type == 'rect':
                            #     unwrap_params_ve = unwrap_params_ve[:,::-1] # reverse x direction 
                            #     unwrap_params_epi = unwrap_params_epi[:,::-1]
                            # else:
                            #     unwrap_params_ve = unwrap_params_ve[::-1] # reverse y direction
                            #     unwrap_params_epi = unwrap_params_epi[::-1]
                            
                            # # properly invert the 3D. 
                            # unwrap_params_ve[...,2] = embryo_im_shape[2] - unwrap_params_ve[...,2]
                            # unwrap_params_epi[...,2] = embryo_im_shape[2] - unwrap_params_epi[...,2]
                            

#                    # =============================================================================
#                    #       AVE rotate.....                          
#                    # =============================================================================
##                        hex_depth_vid_rot_AVE = np.array([np.array([sktform.rotate(vv, 
##                                                				angle=(90-ve_angle_move), 
##                                                				preserve_range=True, cval=0) for vv in hex_depth_vid[tt]]) for tt in np.arange(len(hex_depth_vid))])
#                        # was minus
#                        angle_rot = -(90-ve_angle_move) # will this work? # make this anticlockwise? 
#                        rot_matrix_rot = geom.get_rotation_x(angle_rot/180.*np.pi)[:3,:3] # which is the correct way to rotate? 
                        im_array_ve_rot_AVE = reg.warp_3D_transforms_xyz_similarity(im_array_ve_rot, 
                                                                                    rotation = rot_matrix_rot[:3,:3],
                                                                                    zoom = np.hstack([1,1,1]),
                                                                                    center = ref_pt_3D_center,#ref_pt_3D_center, # don't need to transpose now. 
                                                                                    center_tform=True,
                                                                                    direction='F',
                                                                                    pad=None)
                        
#                    # 'demons_remapped_pts-'+unwrapped_condition+'.mat')
#                    
##                    saveprojfolder_hex = os.path.join(saveprojfolder, 'HEX'); fio.mkdir(saveprojfolder_hex)
##                    saveprojfolder_mtmg = os.path.join(saveprojfolder, 'MTMG'); fio.mkdir(saveprojfolder_mtmg)
                    # =============================================================================
                    #       Also rotate the pts to double check.                      
                    # =============================================================================

                        # check the cross-section. 
#                        select_z_vis = im_array_hex_rot.shape[2]//2
                        select_z_vis = int(ref_pt_3D_center[2])
                        select_z_vis_pts = np.logical_and(unwrap_params_ve[...,2] >select_z_vis-1, 
                                                          unwrap_params_ve[...,2] <select_z_vis+1)
                        
                        
                        plt.figure()
                        plt.imshow(im_array_ve_rot[im_array_ve_rot.shape[0]//4*3:].max(axis=0),cmap='gray')
                        plt.vlines(int(ref_pt_3D_center[2]), 0, im_array_ve_rot.shape[1]-1, color='y')
                        plt.hlines(int(ref_pt_3D_center[1]), 0, im_array_ve_rot.shape[2],color='y')
                        plt.show()
                        
                        plt.figure()
                        plt.imshow(im_array_ve_rot_AVE[im_array_ve_rot.shape[0]//4*3:,:,:].max(axis=0),cmap='gray')
                        plt.vlines(int(ref_pt_3D_center[2]), 0, im_array_ve_rot.shape[1]-1, color='y')
                        plt.hlines(int(ref_pt_3D_center[1]), 0, im_array_ve_rot.shape[2],color='y')
                        plt.show()
                        
                        
                        plt.figure(figsize=(5,5))
#                        plt.subplot(121)
                        plt.imshow(equalize_hist(im_array_ve_rot[:,:,select_z_vis]),cmap='gray')
                        plt.plot(unwrap_params_ve[select_z_vis_pts,1], 
                                 unwrap_params_ve[select_z_vis_pts,0], 'r.', ms=1)
                        plt.show()
                        
                        # check the cross-section. -> we need to take now the y-cut coordinate. 
                        select_z_vis = int(np.rint(unwrap_params_ve[center[0], center[1]][1]))
                        select_z_vis_pts = np.logical_and(unwrap_params_ve_AVE_rot[...,1] >select_z_vis-1, 
                                                          unwrap_params_ve_AVE_rot[...,1] <select_z_vis+1)
                        
                        # if we select the following we should have the mid-cut... 
                        mid_cut = center[1]
                        select_z_vis = int(np.rint(unwrap_params_ve[center[0], center[1]][1]))
                        select_z_vis_pts = np.logical_and(unwrap_params_ve_AVE_rot[...,1] >select_z_vis-1, 
                                                          unwrap_params_ve_AVE_rot[...,1] <select_z_vis+1)
                        
                        plt.figure()
                        plt.imshow(select_z_vis_pts)
                        plt.show()
                        
                        plt.figure(figsize=(5,5))
                        plt.title(str(select_z_vis))
                        plt.imshow(equalize_hist(im_array_ve_rot_AVE[:,select_z_vis]),cmap='gray')
                        plt.plot(unwrap_params_ve_AVE_rot[select_z_vis_pts,2], 
                                 unwrap_params_ve_AVE_rot[select_z_vis_pts,0], 'r.', ms=1)
                        plt.plot(unwrap_params_epi_AVE_rot[select_z_vis_pts,2], 
                                 unwrap_params_epi_AVE_rot[select_z_vis_pts,0], 'g.', ms=1)
                        plt.plot(unwrap_params_ve_AVE_rot[:,mid_cut,2], 
                                 unwrap_params_ve_AVE_rot[:,mid_cut,0], 'y', lw=2) 
                        plt.show()
                        
#                        # i think we might have to do a 3D interpolation to slice through the mid-cut points. 
#                        
                        hexfile_save = (embryo_im_folder_files[tt]).replace(embryo_im_folder[0], saveprojfolder_vol)
                        print(tt, hexfile_save)
#
                        skio.imsave(hexfile_save, im_array_ve_rot_AVE.transpose(1,0,2)) # switch the cut around. so we can just cut straight.... 
#                        
                        
                        # check the other selections. 
                        unwrap_ve_rot = im_array_ve_rot[unwrap_params_ve[...,0].astype(np.int), 
                                                         unwrap_params_ve[...,1].astype(np.int),
                                                         unwrap_params_ve[...,2].astype(np.int)-1]
                        
                        unwrap_params_ve_AVE_rot[...,0] = np.clip(unwrap_params_ve_AVE_rot[...,0], 0, im_array_ve_rot.shape[0]-1)
                        unwrap_params_ve_AVE_rot[...,1] = np.clip(unwrap_params_ve_AVE_rot[...,1], 0, im_array_ve_rot.shape[1]-1)
                        unwrap_params_ve_AVE_rot[...,2] = np.clip(unwrap_params_ve_AVE_rot[...,2], 0, im_array_ve_rot.shape[2]-1)
                        
                        unwrap_params_ve_AVE_rot[np.isnan(unwrap_params_ve_AVE_rot)] = 0
                        unwrap_ve_rot_AVE = im_array_ve_rot_AVE[unwrap_params_ve_AVE_rot[...,0].astype(np.int), 
                                                                 unwrap_params_ve_AVE_rot[...,1].astype(np.int),
                                                                 unwrap_params_ve_AVE_rot[...,2].astype(np.int)]
                        
                        
                        plt.figure(figsize=(10,10))
                        plt.imshow(unwrap_ve_rot, cmap='gray')
                        plt.figure(figsize=(10,10))
                        plt.imshow(unwrap_ve_rot_AVE, cmap='gray')
                        plt.show()
                        
                        
                        
                        
                        
#                        unwrap_params_ve = preprocess_unwrap_pts(unwrap_params_ve, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2]) 
#                    unwrap_params_epi = preprocess_unwrap_pts(unwrap_params_epi, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2])
#                        if rot_angle != 0: 
#                    rot_matrix = geom.get_rotation_y(rot_angle/180.*np.pi)[:3,:3]
#                else:
#                    rot_matrix = np.eye(3)
#                
##                t1 = time.time()
#                vid_demons_temp_rev = reg.warp_3D_transforms_xyz_similarity(vid_demons_rev, 
#                                                                rotation = rot_matrix,
#                                                                zoom = Z,
#                                                                center_tform=True,
#                                                                direction='F',
#                                                                pad=None)
                        
                        
                        
#                        # =============================================================================
#                        #   Intensity lookup                 
#                        # =============================================================================
#                        unwrap_hex_tt = map_intensity_interp3(unwrap_params_ve_depth.reshape(-1,3), 
#                                                              im_array_ve.shape, 
#                                                              im_array_ve)
#                        unwrap_hex_tt = unwrap_hex_tt.reshape(unwrap_params_ve_depth.shape[:-1])
#                        
##                        unwrap_mtmg_tt = map_intensity_interp3(unwrap_params_ve_depth.reshape(-1,3), 
##                                                              im_array_mtmg.shape, 
##                                                              im_array_mtmg)
##                        unwrap_mtmg_tt = unwrap_mtmg_tt.reshape(unwrap_params_ve_depth.shape[:-1])
##                        
##                        import skimage.io as skio 
#                            
##                        # save the unwraps out. 
##                        hexfile_save = (embryo_im_folder_files[tt]).replace(embryo_im_folder[0], saveprojfolder)
###                        mtmgfile_save = (embryo_im_folder_files_mtmg[tt]).replace(embryo_im_folder_mtmg[0], saveprojfolder_mtmg)
##
##                        print(tt, hexfile_save)
##
##                        skio.imsave(hexfile_save, unwrap_hex_tt)
###                        skio.imsave(mtmgfile_save, unwrap_mtmg_tt)
##                        
##                         # check the unwrapped at the middle. 
##                         plt.figure(figsize=(15,15))
##                         plt.imshow(unwrap_mtmg_tt[unwrap_hex_tt.shape[0]//2], cmap='gray')
##                         plt.show()
#                        # as a check . 
#                        plt.figure(figsize=(8,8))
#                        plt.imshow(np.mean(unwrap_hex_tt[:,unwrap_hex_tt.shape[1]//2-10:unwrap_hex_tt.shape[1]//2+10], axis=1), cmap='gray')
#                        plt.show()
#                        
#                        # plt.figure(figsize=(15,15))
#                        # plt.imshow(unwrap_hex_tt[:,unwrap_hex_tt.shape[1]//2], cmap='gray')
#                        # plt.show()
#                        
##                         plt.figure(figsize=(15,15))
##                         plt.imshow(unwrap_mtmg_tt[:,:,unwrap_hex_tt.shape[1]//2//2], cmap='gray')
##                         plt.hlines(len(unwrap_params_ve_depth)//2, 0, unwrap_params_ve_depth.shape[1], color='w')
##                         plt.show()
##                         plt.figure(figsize=(15,15))
##                         plt.imshow(unwrap_hex_tt[:,:,unwrap_hex_tt.shape[1]//2], cmap='gray')
##                         plt.hlines(len(unwrap_params_ve_depth)//2, 0, unwrap_params_ve_depth.shape[1], color='w')
##                         plt.show()
#                        
##                         fig, ax = plt.subplots(figsize=(15,15))
##                         for iii in np.arange(0,len(unwrap_params_ve_depth),4):
##                             ax.plot(unwrap_params_ve_depth[iii][unwrap_hex_tt.shape[1]//2,:,0], 
##                                     unwrap_params_ve_depth[iii][unwrap_hex_tt.shape[1]//2,:,1], '-')
##                         ax.set_aspect(1)
##                         plt.show()
#                        
#
##                         demons_dxyz_ve_pt = unwrap_params_ve.copy() # show we have the correct shape!. 
##                         demons_dxyz_epi_pt = unwrap_params_epi.copy()
## #                        im_array = fio.read_multiimg_PIL(embryo_im_folder_files[-1])
# #                            im_array = fio.read_multiimg_PIL(np.sort(embryo_im_folder_files)[tt])
#                         select_z_vis = im_array.shape[0]//2
#                        
#                         select_z_vis_pts = np.logical_and(demons_dxyz_ve_pt.reshape(-1,3)[...,0] >select_z_vis-1, 
#                                                           demons_dxyz_ve_pt.reshape(-1,3)[...,0] <select_z_vis+1)
#                        
#                         plt.figure(figsize=(5,5))
#                         plt.imshow(equalize_hist(im_array_mtmg[select_z_vis]),cmap='gray')
#                         plt.plot(demons_dxyz_ve_pt.reshape(-1,3)[select_z_vis_pts,2], 
#                                  demons_dxyz_ve_pt.reshape(-1,3)[select_z_vis_pts,1], 'r.')
#                         plt.show()
#                        
#                        
#                         select_z_vis = im_array.shape[1]*2//3
#                        
#                         select_z_vis_pts = np.logical_and(demons_dxyz_ve_pt[...,1] >select_z_vis-1, 
#                                                           demons_dxyz_ve_pt[...,1] <select_z_vis+1)
#                        
#                         plt.figure(figsize=(5,5))
#                         plt.imshow(equalize_hist(im_array_mtmg[:,select_z_vis]),cmap='gray')
#                         plt.plot(demons_dxyz_ve_pt[select_z_vis_pts,2], 
#                                  demons_dxyz_ve_pt[select_z_vis_pts,0], 'r.')
#                         plt.show()
 #                        
 #                        
 #                        select_z_vis = im_array.shape[2]//2
 #                        
 #                        select_z_vis_pts = np.logical_and(demons_dxyz_ve_pt[...,2] >select_z_vis-1, 
 #                                                          demons_dxyz_ve_pt[...,2] <select_z_vis+1)
 #                        
 #                        plt.figure(figsize=(5,5))
 #                        plt.imshow(equalize_hist(im_array[:,:,select_z_vis]),cmap='gray')
 #                        plt.plot(demons_dxyz_ve_pt[select_z_vis_pts,1], 
 #                                 demons_dxyz_ve_pt[select_z_vis_pts,0], 'r.')
 #                        plt.show()
                            
#                            
## # #                        demons_dxyz_ve_pt = demons_dxyz_ve * temp_tform_scales[tt] + unwrap_params_ve
## # #                        demons_dxyz_epi_pt = demons_dxyz_epi * temp_tform_scales[tt] + unwrap_params_epi
#                        
## #                         demons_dxyz_ve_pt = isotropic_scale_pts(demons_dxyz_ve + unwrap_params_ve,  temp_tform_scales[tt])
## #                         demons_dxyz_epi_pt = isotropic_scale_pts(demons_dxyz_epi + unwrap_params_epi, temp_tform_scales[tt])
#                        
## #                         plt.figure()
## #                         plt.subplot(131)
## #                         plt.imshow(demons_dxyz_ve_pt[...,0])
## #                         plt.subplot(132)
## #                         plt.imshow(demons_dxyz_ve_pt[...,1])
## #                         plt.subplot(133)
## #                         plt.imshow(demons_dxyz_ve_pt[...,2])
## #                         plt.show()
#
## #                         """
## #                         reverse rotation back to unwrapped parameters.
## #                         """
## #                         demons_dxyz_ve_pt = preprocess_unwrap_pts(demons_dxyz_ve_pt, im_array, rot_angle_3d=-rot_angle, transpose_axis=[0,1,2], clip_pts_to_array=False)[...,[1,0,2]] 
## #                         demons_dxyz_epi_pt = preprocess_unwrap_pts(demons_dxyz_epi_pt, im_array, rot_angle_3d=-rot_angle, transpose_axis=[0,1,2], clip_pts_to_array=False)[...,[1,0,2]] 
## # #                
##                         """
##                         Should absolutely check.... 
##                         """
## #                        plt.figure()
## #                        plt.subplot(131)
## #                        plt.imshow(demons_dxyz_ve_pt[...,0])
## #                        plt.subplot(132)
## #                        plt.imshow(demons_dxyz_ve_pt[...,1])
## #                        plt.subplot(133)
## #                        plt.imshow(demons_dxyz_ve_pt[...,2])
## #                        plt.show()
#                        
#                        
## # #                        print('----')
## #                         all_demons_ve_pts.append(demons_dxyz_ve_pt)
## #                         all_demons_epi_pts.append(demons_dxyz_epi_pt)
#                        
## #                         # with the new points we build new tangent vectors ( most meaningful would be to set this up according to AVE direction?)
#                        
## #                         # in any case we can do it agnositically for radial etc vectors. 
## #                         """
## #                         Build the tangent and also normal components using the latest updated script. 
## #                         """
## #                         direction_r_3D_ve, direction_theta_3D_ve, direction_normal_3D_ve = meshtools.build_full_polar_tangent_and_normal_unit_vectors(demons_dxyz_ve_pt,                                                           
## #                                                                                                                                                 r_dist=10, 
## #                                                                                                                                                 theta_dist=5, 
## #                                                                                                                                                 smooth=True, 
## #                                                                                                                                                 smooth_theta_dist=10, 
## #                                                                                                                                                 smooth_radial_dist=20, 
## #                                                                                                                                                 smooth_tip_sigma=15, 
## #                                                                                                                                                 smooth_tip_radius=35) # also apply smoothing of 3 degrees.
#                    
## #                         direction_r_3D_epi, direction_theta_3D_epi, direction_normal_3D_epi = meshtools.build_full_polar_tangent_and_normal_unit_vectors(demons_dxyz_epi_pt,                                                           
## #                                                                                                                                                 r_dist=10, 
## #                                                                                                                                                 theta_dist=5, 
## #                                                                                                                                                 smooth=True, 
## #                                                                                                                                                 smooth_theta_dist=10, 
## #                                                                                                                                                 smooth_radial_dist=20, 
## #                                                                                                                                                 smooth_tip_sigma=15, 
## #                                                                                                                                                 smooth_tip_radius=35) # also apply smoothing of 3 degrees.
#                    
## #                         all_dir_r_ve.append(direction_r_3D_ve)
## #                         all_dir_r_epi.append(direction_r_3D_epi)
## #                         all_dir_theta_ve.append(direction_theta_3D_ve)
## #                         all_dir_theta_epi.append(direction_theta_3D_epi)
## #                         all_dir_normal_ve.append(direction_normal_3D_ve)
## #                         all_dir_normal_epi.append(direction_normal_3D_epi)
#    
#                        
## #                     all_demons_ve_pts = np.array(all_demons_ve_pts)
## #                     all_demons_epi_pts = np.array(all_demons_epi_pts)
#                    
## #                     all_dir_r_ve = np.array(all_dir_r_ve)
## #                     all_dir_r_epi = np.array(all_dir_r_epi)
## #                     all_dir_theta_ve = np.array(all_dir_theta_ve)
## #                     all_dir_theta_epi = np.array(all_dir_theta_epi)
## #                     all_dir_normal_ve = np.array(all_dir_normal_ve)
## #                     all_dir_normal_epi = np.array(all_dir_normal_epi)
    
#     #             # =============================================================================
#     #             #    Save out for analysis. 
#     #             # ============================================================================= 
                
#     #                 saveprojfile = os.path.join(saveanalysisfolder, 'demons_remapped_pts-'+unwrapped_condition+'.mat')
#     # #                spio.savemat(saveprojfile, {'ve_pts': all_demons_ve_pts,
#     # #                                            'epi_pts': all_demons_epi_pts, 
#     # #                                            'surf_normal_vect_ref': surface_normal_norm, 
#     # #                                            'surf_y_vect_ref': surface_y_norm,
#     # #                                            'surf_radial_vect_ref': surface_theta_norm,
#     # #                                            'demons_files': demons_rev_warp_files,
#     # #                                            'demons_times': demons_times,
#     # #                                            'rect_polar_mapping': np.dstack(mapped_ij)})
        
#     #                 spio.savemat(saveprojfile, {'ve_pts': all_demons_ve_pts.astype(np.float32),
#     #                                             'epi_pts': all_demons_epi_pts.astype(np.float32), 
#     #                                             'surf_normal_ve':all_dir_normal_ve.astype(np.float32),
#     #                                             'surf_radial_ve':all_dir_r_ve.astype(np.float32),
#     #                                             'surf_theta_ve':all_dir_theta_ve.astype(np.float32),
#     #                                             'surf_normal_epi':all_dir_normal_epi.astype(np.float32),
#     #                                             'surf_radial_epi':all_dir_r_epi.astype(np.float32),
#     #                                             'surf_theta_epi':all_dir_theta_epi.astype(np.float32),
#     #                                             'demons_files': demons_rev_warp_files,
#     #                                             'demons_times': demons_times}, do_compression=True)
            
            
# #                fig = plt.figure(figsize=(15,15))
# #                ax = fig.add_subplot(111, projection='3d')
# #                ax.scatter(all_demons_ve_pts[0,::20, ::20,2],  
# #                           all_demons_ve_pts[0,::20, ::20,1], 
# #                           all_demons_ve_pts[0,::20, ::20,0])
# #                ax.scatter(all_demons_ve_pts[20,::20, ::20,2],  
# #                           all_demons_ve_pts[20,::20, ::20,1], 
# #                           all_demons_ve_pts[20,::20, ::20,0])
# #                ax.scatter(all_demons_ve_pts[100,::20, ::20,2],  
# #                           all_demons_ve_pts[100,::20, ::20,1], 
# #                           all_demons_ve_pts[100,::20, ::20,0])
# #                plt.show()
# #            
                
# #                # give indication of where in the embryo is the greatest change with respect to the start geometry?
# #                all_demons_ve_mag = np.zeros(unwrap_params_ve_polar.shape[:2]); 
# #                all_demons_ve_mag[mapped_ij[0],
# #                                  mapped_ij[1]] = np.linalg.norm(all_demons_ve[:24], axis=-1).mean(axis=0)
# #                
# #                
# #                all_demons_epi_mag = np.zeros(unwrap_params_ve_polar.shape[:2]); 
# #                all_demons_epi_mag[mapped_ij[0],
# #                                  mapped_ij[1]] = sktform.resize(np.linalg.norm(all_demons_epi[:24], axis=-1).mean(axis=0), 
# #                                                                  all_demons_ve.shape[1:-1], 
# #                                                                  preserve_range=True)
# #                
# #                plt.figure(figsize=(15,15))
# #                plt.subplot(131)
# #                plt.imshow(all_demons_ve_mag, cmap='coolwarm', vmin=0, vmax=30)
# #                plt.subplot(132)
# #                plt.imshow(all_demons_epi_mag, cmap='coolwarm', vmin=0, vmax=30)
# #                plt.subplot(133)
# #                plt.imshow(all_demons_epi_mag, cmap='gray')
# #                plt.imshow(all_demons_ve_mag, cmap='coolwarm', vmin=0, vmax=30, alpha=0.5)
# #                plt.show()
#                    
#        # =============================================================================
#        #   Mapping vs the components.      
#        # =============================================================================
#                all_demons_ve_component_norm = np.array([ np.sum( v*surface_normal_norm, axis=-1) for v in  all_demons_ve ])
#                all_demons_ve_component_y = np.array([ np.sum( v*surface_y_norm, axis=-1) for v in  all_demons_ve ])
#                all_demons_ve_component_theta = np.array([ np.sum( v*surface_theta_norm, axis=-1) for v in  all_demons_ve ])
#                
#                all_demons_ve_mag = np.zeros(unwrap_params_ve_polar.shape[:2]); 
#                all_demons_ve_mag[mapped_ij[0],
#                                  mapped_ij[1]] = all_demons_ve_component_norm[50:].mean(axis=0)
#                
#                all_demons_epi_resize = sktform.resize(all_demons_epi, all_demons_ve.shape[:], preserve_range=True)
#                
#                all_demons_epi_component_norm = np.array([ np.sum( v*surface_normal_norm, axis=-1) for v in  all_demons_epi_resize ])
#                all_demons_epi_component_y = np.array([ np.sum( v*surface_y_norm, axis=-1) for v in  all_demons_epi_resize ])
#                all_demons_epi_component_theta = np.array([ np.sum( v*surface_theta_norm, axis=-1) for v in  all_demons_epi_resize ])
#                
#                all_demons_epi_mag = np.zeros(unwrap_params_ve_polar.shape[:2]); 
#                all_demons_epi_mag[mapped_ij[0],
#                                  mapped_ij[1]] = all_demons_epi_component_norm[50:].mean(axis=0)
#                
#                
#                fig, ax = plt.subplots(figsize=(15,15))
#                plt.imshow(vid_ve[0])
#                plot_tracks(meantracks_ve[:,:50], ax = ax, color='r')
#                plt.show()
                
                
                
                
                
                
                
                
#                fig, ax = plt.subplots()
#                plt.imshow(vid_ve[0])
#                plot_tracks(meantracks_ve, ax=ax, color='g')
#                plt.show()


#                    demons_dxyz_ve_2 = spio.loadmat(demons_rev_warp_files[tt+1])['ve_rect_dxyz'][...,[1,0,2]]
#                    
#                    demons_dxyz_ve_move = demons_dxyz_ve_2 - demons_dxyz_ve_1
#                    #
#                    surface_pts = demons_dxyz_ve_1 + unwrap_params_ve
#                    surface_normal, _  = findPointNormals(surface_pts.reshape(-1,3), 
#                                                          nNeighbours=35, 
#                                                          viewPoint=np.mean(surface_pts.reshape(-1,3), axis=0), 
#                                                          dirLargest=True)
#                    surface_normal = surface_normal.reshape(unwrap_params_ve.shape)
#                    
#                    # normalise surface_normals
#                    surface_normal = surface_normal / np.linalg.norm(surface_normal, axis=-1)[...,None] # normalise
#                    surface_y = np.gradient(surface_pts, axis=0) ; surface_y = surface_y / np.linalg.norm(surface_y, axis=-1)[...,None]
#                    surface_theta = np.gradient(surface_pts, axis=1) ; surface_theta = surface_theta / np.linalg.norm(surface_theta, axis=-1)[...,None]
#                    
#                    
#                    
#                    fig = plt.figure(figsize=(15,15))
#                    ax = fig.add_subplot(111, projection='3d')
#                    ax.quiver(unwrap_params_ve[::20, ::20,2],  
#                              unwrap_params_ve[::20, ::20,1], 
#                              unwrap_params_ve[::20, ::20,0], 
#                              surface_normal[::20,::20,2], 
#                              surface_normal[::20,::20,1], 
#                              surface_normal[::20,::20,0], length=20., normalize=True)
#                    plt.show()
#                    
#                    fig = plt.figure(figsize=(15,15))
#                    ax = fig.add_subplot(111, projection='3d')
#                    ax.quiver(unwrap_params_ve[::20, ::20,2],  
#                              unwrap_params_ve[::20, ::20,1], 
#                              unwrap_params_ve[::20, ::20,0], 
#                              surface_theta[::20,::20,2], 
#                              surface_theta[::20,::20,1], 
#                              surface_theta[::20,::20,0], length=20., normalize=True)
#                    plt.show()
#
#                    
#                    
#                    all_demons_disps_normal.append(np.nansum(demons_dxyz_ve_move * surface_normal, axis=-1))
#                    all_demons_disps_theta.append(np.nansum(demons_dxyz_ve_move * surface_theta, axis=-1))
#                    all_demons_disps_y.append(np.nansum(demons_dxyz_ve_move * surface_y, axis=-1))
#                    
#                    
##                all_demons_disps = np.array(all_demons_disps)
##                all_demons_disps_time = all_demons_disps[1:] - all_demons_disps[:-1]
##                all_demons_disps_time_mag = np.linalg.norm(all_demons_disps_time, axis=-1)
#                all_demons_disps_theta = np.array(all_demons_disps_theta)
#                all_demons_disps_y = np.array(all_demons_disps_y)
#                
#                
#                fig, ax = plt.subplots()
#                ax.imshow(all_demons_disps_normal_z.T, cmap='coolwarm', vmin=-2, vmax=2)
#                ax.set_aspect('auto')
#                plt.show()
#                
#                
#                # map the rectangular values into polar.
#                polar_demons_disps_normal = np.zeros((len(all_demons_disps_normal),unwrap_params_ve_polar.shape[0], unwrap_params_ve_polar.shape[1]))
#                polar_demons_disps_normal[:, mapped_ij[0], mapped_ij[1]] = all_demons_disps_normal
#                
#                
#                polar_demons_disps_theta = np.zeros((len(all_demons_disps_theta),unwrap_params_ve_polar.shape[0], unwrap_params_ve_polar.shape[1]))
#                polar_demons_disps_theta[:, mapped_ij[0], mapped_ij[1]] = all_demons_disps_theta
#                
#                polar_demons_disps_y = np.zeros((len(all_demons_disps_normal),unwrap_params_ve_polar.shape[0], unwrap_params_ve_polar.shape[1]))
#                polar_demons_disps_y[:, mapped_ij[0], mapped_ij[1]] = all_demons_disps_y
#                
##                
#                
#                fig, ax = plt.subplots(figsize=(15,15))
#                ax.imshow(vid_ve[0], cmap='gray')
#                plot_tracks(meantracks_ve[:,24:104], ax=ax, color='g')
#                ax.set_aspect('auto')
#                plt.show()
#                
#                
#                
#                polar_surface_theta = np.zeros(unwrap_params_ve_polar.shape)
#                polar_surface_theta[mapped_ij[0], mapped_ij[1]] = surface_theta
                
#                normals_ve_polar = np.zeros(unwrap_params_ve_polar.shape)
#                normals_ve_polar[mapped_ij[0],
#                                 mapped_ij[1]] = normals_ve
#                
##                track_label_rect_polar_ids_i = val_dist_ijs[0][track_2D_int_rect[:,0,0], 
##                                                        track_2D_int_rect[:,0,1]]  
##                track_label_rect_polar_ids_j = val_dist_ijs[1][track_2D_int_rect[:,0,0], 
##                                                                track_2D_int_rect[:,0,1]] 
#                
#                
#                if proj_type == 'rect':
#                    unwrap_params_ve = unwrap_params_ve[::-1] # reverse ....
##                    unwrap_params_epi = unwrap_params_epi[::-1]
#                    
#                    plt.figure()
#                    plt.imshow(unwrap_params_ve[...,0])
#                    plt.show()
##                    
#                    
##                 compute surface normals at the unwrapping xyz points. [ need a lot of nearest neighbours for a smooth normals.] 
#                normals_ve, curvature_ve = findPointNormals(unwrap_params_ve.reshape(-1,3), nNeighbours=35, viewPoint=np.mean(unwrap_params_ve.reshape(-1,3), axis=0), dirLargest=True)
#                
#                normals_ve = normals_ve.reshape(unwrap_params_ve.shape)
#                
#                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
#                ax[0].imshow(normals_ve[...,0])
#                ax[1].imshow(normals_ve[...,1])
#                ax[2].imshow(normals_ve[...,2])
#                plt.show()
#                
#                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#                
##                fig = plt.figure(figsize=(15,15))
##                ax = fig.gca(projection='3d')
##                
##                ax.quiver(unwrap_params_ve[::10,::10,0].ravel(), 
##                          unwrap_params_ve[::10,::10,1].ravel(),
##                          unwrap_params_ve[::10,::10,2].ravel(), 
##                          normals_ve[::10,::10,0].ravel(), 
##                          normals_ve[::10,::10,1].ravel(), 
##                          normals_ve[::10,::10,2].ravel(), 
##                          length=10, normalize=False)
##                
##                plt.show()
                
                
#                diff_xyz_col = np.gradient(unwrap_params_ve, axis=1)
#                diff_xyz_row = np.gradient(unwrap_params_ve, axis=0)
#                diff_xyz_normal = np.cross(diff_xyz_col, diff_xyz_row)
#                
#                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
#                ax[0].imshow(diff_xyz_normal[...,0], vmin=-1, vmax=1)
#                ax[1].imshow(diff_xyz_normal[...,1], vmin=-1, vmax=1)
#                ax[2].imshow(diff_xyz_normal[...,2], vmin=-1, vmax=1)
#                plt.show()
#                
#                
#                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#                
#                fig = plt.figure(figsize=(15,15))
#                ax = fig.gca(projection='3d')
#                
#                ax.quiver(unwrap_params_ve[::10,::10,0].ravel(), 
#                          unwrap_params_ve[::10,::10,1].ravel(),
#                          unwrap_params_ve[::10,::10,2].ravel(), 
#                          diff_xyz_normal[::10,::10,0].ravel(), 
#                          diff_xyz_normal[::10,::10,1].ravel(), 
#                          diff_xyz_normal[::10,::10,2].ravel(), 
#                          length=10, normalize=True)
#                
#                plt.show()
##               

##                L491_ve_000_unwrap_align_params_geodesic
#            # =============================================================================
#            #       Load the relevant VE segmentation from trained classifier    
#            # =============================================================================
##                ve_seg_params_file = os.path.join(savevesegfolder, os.path.split(trackfile)[-1].replace('rect','polar').replace('.mat', '-ve_aligned.mat'))
##                ve_seg_params_file = ve_seg_params_file.replace('-5000_', '-1000_')
#                
#                ve_seg_params_file = 'deepflow-meantracks-1000_' + os.path.split(unwrap_param_file_ve)[-1].replace('_unwrap_params-geodesic.mat',  '_polar-ve_aligned.mat')
#                ve_seg_params_file = os.path.join(savevesegfolder, ve_seg_params_file)
#                ve_seg_params_obj = spio.loadmat(ve_seg_params_file)                             
#                
##                ve_seg_select_rect = ve_seg_params_obj['ve_rect_select'].ravel() > 0
##                ve_seg_select_rect_valid = ve_seg_params_obj['ve_rect_select_valid'].ravel() > 0
#                
##                ve_seg_select_rect_valid = ve_seg_params_obj['density_rect_pts']
##                ve_seg_select_rect_valid = ve_seg_select_rect_valid >= np.mean(ve_seg_select_rect_valid) + 1 * np.std(ve_seg_select_rect_valid)
#                ve_seg_select_rect_valid = ve_seg_params_obj['density_rect_pts_mask'] # get the polar associated mask. 
#                ve_seg_migration_times = ve_seg_params_obj['migration_stage_times'][0]
#                
#                embryo_inversion = ve_seg_params_obj['embryo_inversion'].ravel()[0] > 0 
#                
#            # =============================================================================
#            #   Check for embryo inversion -> all params etc have to be inverted accordingly.                
#            # =============================================================================
#                if embryo_inversion:
#                    if proj_type == 'rect':
#                        unwrap_params_ve = unwrap_params_ve[:,::-1] # reverse x direction 
#                        unwrap_params_epi = unwrap_params_epi[:,::-1]
#                        
#                        meantracks_ve[...,1] = vid_size[1]-1 - meantracks_ve[...,1] # reverse the x direction 
#                        meantracks_epi[...,1] = vid_epi[0].shape[1]-1 - meantracks_epi[...,1]
#                        
#                        vid_ve = vid_ve[:,:,::-1]
#                        vid_epi = vid_epi[:,:,::-1]
#                        vid_epi_resize = vid_epi_resize[:,:,::-1]
#                        
#                        
#                    else:
#                        unwrap_params_ve = unwrap_params_ve[::-1] # reverse y direction
#                        unwrap_params_epi = unwrap_params_epi[::-1]
#                        
#                        meantracks_ve[...,0] = vid_size[0]-1 - meantracks_ve[...,0] # switch the y direction. 
#                        meantracks_epi[...,0] = vid_epi[0].shape[0]-1 - meantracks_epi[...,0]
#                        
#                        vid_ve = vid_ve[:,::-1]
#                        vid_epi = vid_epi[:,::-1]
#                        vid_epi_resize = vid_epi_resize[:,::-1]
#                        
#                if np.isnan(ve_seg_migration_times[0]):
#                    ve_seg_select_rect = ve_seg_select_rect_valid[meantracks_ve[:,0,0], 
#                                                                  meantracks_ve[:,0,1]] > 0
#                else:
#                    ve_seg_select_rect = ve_seg_select_rect_valid[meantracks_ve[:,int(ve_seg_migration_times[0]),0], 
#                                                                  meantracks_ve[:,int(ve_seg_migration_times[0]),1]] > 0
#                
#                
#                """
#                Verify the projection. 
#                """
#                fig, ax = plt.subplots(figsize=(15,15))
#                ax.imshow(vid_ve[0], cmap='gray')
#                plot_tracks(meantracks_ve[:,:10], ax=ax, color='g')
#                ax.plot(meantracks_ve[ve_seg_select_rect, 0, 1], 
#                        meantracks_ve[ve_seg_select_rect, 0, 0], 'ro')
#                plt.show()
#                
##                ve_seg_select_rect = ve_seg_select_rect_valid[meantracks_ve[:,0,0], 
##                                                              meantracks_ve[:,0,1]]
#                """
#                Get the valid region of polar projection  
#                """
#                YY,XX = np.meshgrid(range(vid_ve[0].shape[1]), 
#                                    range(vid_ve[0].shape[0]))
#                dist_centre = np.sqrt((YY-vid_ve[0].shape[0]/2.)**2 + (XX-vid_ve[0].shape[1]/2.)**2)
#                valid_region = dist_centre <= vid_ve[0].shape[1]/2./1.2
#                
#                """
#                Get the predicted VE region. 
#                """
#                all_reg = np.arange(len(meantracks_ve))
#                all_reg_valid = valid_region[meantracks_ve[:,0,0], 
#                                             meantracks_ve[:,0,1]]
#                
##                """
##                Compute the 3D displacements of the tracks. 
##                """
###                # =============================================================================
###                # Computation A: -> direct displacement computation (no removal of global components)                 
###                # =============================================================================
###                
###                meantracks_ve_smooth_diff = (meantracks_ve_smooth[:,1:] - meantracks_ve_smooth[:,:-1]).astype(np.float32)
####                meantracks_epi_smooth_diff = (meantracks_epi_smooth[:,1:] - meantracks_epi_smooth[:,:-1]).astype(np.float32)
###                
###                meantracks_ve_smooth_diff[...,0] = meantracks_ve_smooth_diff[...,0]/embryo_im_shape[0]
###                meantracks_ve_smooth_diff[...,1] = meantracks_ve_smooth_diff[...,1]/embryo_im_shape[1]
###                meantracks_ve_smooth_diff[...,2] = meantracks_ve_smooth_diff[...,2]/embryo_im_shape[2]
##                
##                # =============================================================================
##                # Computation B: ->                  
##                # =============================================================================
##                meantracks_ve_smooth_diff = compute_3D_displacements(meantracks_ve, unwrap_params_ve, correct_global=False, mask_track=all_reg_valid)
##                
##                # account for the growth correction !
##                meantracks_ve_smooth_diff = meantracks_ve_smooth_diff * temp_tform_scales[None,1:,None]
##                
##                vol_factor = np.product(embryo_im_shape) ** (1./3) # take the cube root
##                meantracks_ve_smooth_diff = meantracks_ve_smooth_diff/vol_factor # normalise by the volume factor. 
##                
###                meantracks_ve_smooth_diff[...,0] = meantracks_ve_smooth_diff[...,0]/embryo_im_shape[0]
###                meantracks_ve_smooth_diff[...,1] = meantracks_ve_smooth_diff[...,1]/embryo_im_shape[1]
###                meantracks_ve_smooth_diff[...,2] = meantracks_ve_smooth_diff[...,2]/embryo_im_shape[2]
##                
##                """
##                plot to check the segmentation 
##                """
##                                                
##                ve_reg_id = all_reg[np.logical_and(ve_seg_select_rect, all_reg_valid)]
##                neg_reg_id = all_reg[np.logical_and(np.logical_not(ve_seg_select_rect), all_reg_valid)]
##                
##                fig, ax = plt.subplots()
##                ax.imshow(vid_ve[0],cmap='gray')
##                ax.plot(meantracks_ve[:,0,1], meantracks_ve[:,0,0], 'wo')
##                ax.plot(meantracks_ve[neg_reg_id,0,1], meantracks_ve[neg_reg_id,0,0], 'ro')
##                ax.plot(meantracks_ve[ve_reg_id,0,1], meantracks_ve[ve_reg_id,0,0], 'go')
##                plt.show()
##                
##                ve_diff_select = meantracks_ve_smooth_diff[ve_reg_id]
##                ve_diff_mean = np.mean(ve_diff_select, axis=1) # check the median ?
##                ve_diff_mean = ve_diff_mean/(np.linalg.norm(ve_diff_mean, axis=-1)[:,None] + 1e-8) 
##                ve_component = np.vstack([ve_diff_select[ll].dot(ve_diff_mean[ll]) for ll in range(len(ve_diff_select))])
##                
##
##                all_speed_neg = meantracks_ve_smooth_diff[neg_reg_id]
##                all_speed_neg_mean = np.mean(all_speed_neg, axis=1) # check the median ?
##                all_speed_neg_mean = all_speed_neg_mean/(np.linalg.norm(all_speed_neg_mean, axis=-1)[:,None] + 1e-8) 
##                all_speed_component = np.vstack([all_speed_neg[ll].dot(all_speed_neg_mean[ll]) for ll in range(len(all_speed_neg))])
##
##                
##                diff_curve_component = np.cumsum(ve_component.mean(axis=0)) - np.cumsum(all_speed_component.mean(axis=0))
###                diff_curve_component = np.cumsum(ve_component.mean(axis=0)).copy()
###                diff_curve_component[diff_curve_component<0] = 0
##                change_curve_component = np.diff(diff_curve_component)
##                
##                diff_curves_all.append(diff_curve_component)
##                all_embryos_plot.append(embryo_folder.split('/')[-1])
##                
##
### =============================================================================
###                 offline changepoint detection?
### =============================================================================
##                from scipy.signal import savgol_filter
##                import bayesian_changepoint_detection.offline_changepoint_detection as offcd
##                from functools import partial
##                from scipy.signal import find_peaks
##                from scipy.interpolate import UnivariateSpline
##                
###                spl = UnivariateSpline(np.arange(len(diff_curve_component)), diff_curve_component, k=1,s=1e-4)
###                spl_diff_component = spl(np.arange(len(diff_curve_component)))
##                
###                spl_diff_component = savgol_filter(diff_curve_component, 11,1)
###                diff_curve_component = savgol_filter(diff_curve_component, 15,1)
##                
##                # switch with an ALS filter?
###                diff_curve_component_ = baseline_als(diff_curve_component, lam=2.5, p=0.5, niter=10) # TTR?
##                if 'LifeAct' in os.path.split(rootfolder)[-1]: 
##                    print('LifeAct')
##                    diff_curve_component_ = baseline_als(diff_curve_component, lam=5, p=0.5, niter=10)
##                else:
##                    print('non-LifeAct')
##                    diff_curve_component_ = baseline_als(diff_curve_component, lam=10, p=0.5, niter=10)
###                diff_curve_component_ = baseline_als(diff_curve_component, lam=5, p=0.5, niter=10) # use lam=5 for lifeact, what about lam=1?
##                
##                plt.figure()
##                plt.plot(diff_curve_component)
###                plt.plot(spl_diff_component)
##                plt.show()
##                
##                # what about detection by step filter convolving (matched filter)
##                up_step = np.hstack([-1*np.ones(len(diff_curve_component)//2), 1*np.ones(len(diff_curve_component)-len(diff_curve_component)//2)])
###                up_step = np.hstack([-1*np.ones(len(diff_curve_component)), 1*np.ones(len(diff_curve_component)-len(diff_curve_component))])
##                down_step = np.hstack([1*np.ones(len(diff_curve_component)//2), -1*np.ones(len(diff_curve_component)-len(diff_curve_component)//2)])
##                
##                conv_res = np.convolve((diff_curve_component - np.mean(diff_curve_component))/(np.std(diff_curve_component)+1e-8)/len(diff_curve_component), down_step, mode='same')
##                
##                
##                print(len(diff_curves_all))           
##                # put in the filtered. 
##                in_signal = np.diff(diff_curve_component_)
###                in_signal_norm = 
###                in_signal =  np.diff(np.cumsum(in_signal)/(np.arange(len(in_signal)) + 1))
##                conv_res_up = np.convolve((in_signal - np.mean(in_signal))/(np.std(in_signal)+1e-8)/len(in_signal), down_step, mode='same')
##                conv_res_down = np.convolve((in_signal - np.mean(in_signal))/(np.std(in_signal)+1e-8)/len(in_signal), up_step, mode='same')
##                
###                conv_res_up = np.convolve(in_signal_norm, down_step, mode='same')
###                conv_res_down = np.convolve(in_signal_norm, up_step, mode='same')
##                
###                peaks_conv = find_peaks(np.abs(conv_res_up), distance=5, height=0.15, prominence=0.05)[0]
##                peaks_conv = find_peaks(np.abs(conv_res_up), distance=5, height=0.1)[0]
###                peaks_conv = find_peaks(np.abs(conv_res_up), distance=2.5, height=0.2)[0]
###                break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=1.25e-3) # make this automatic... 
##                
##                if 'LifeAct' in os.path.split(rootfolder)[-1]: 
##                    print('LifeAct')
##                    break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=1.5e-3)
##                else:
##                    print('non-LifeAct')
##                    break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=3e-3) # make this automatic... 
##
##
###                zzz = test_unique_break_segments(in_signal, peaks_conv, p_thresh=0.05)
###                zz = test_unique_break_segments(in_signal, peaks_conv)
###                print(test_unique_break_segments(in_signal, peaks_conv))
###                break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=3e-3)
##                break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=2.5e-3)
##                if len(break_grads[-1][1]) > 0:
##                    break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=.5*np.max(break_grads[0])) # 10% the max? 
##                    print(.5*np.max(break_grads[0]))
#                
#                regime_colors = ['r','g','b']
#                
#                # plot the human dictated separation lines. 
#                
#                plt.figure(figsize=(4,7))
#                plt.subplot(411)
#                plt.plot(np.cumsum(ve_component.mean(axis=0)), label='AVE')
#                plt.plot(np.cumsum(all_speed_component.mean(axis=0)), label='Non-AVE')
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
#                if ~np.isnan(pre.ravel()[0]):
#                    plt.vlines( pre.ravel()[1]-1, 0, np.max(diff_curve_component), color='r', linestyles='dashed')
#                
#                if ~np.isnan(mig.ravel()[0]):
#                    plt.vlines( mig.ravel()[1]-1, 0, np.max(diff_curve_component), color='g', linestyles='dashed')
#                
#                if ~np.isnan(post.ravel()[0]):
#                    plt.vlines( post.ravel()[1]-1, 0, np.max(diff_curve_component), color='b', linestyles='dashed')
#                
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
#                plt.savefig('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/MOSES_staging_plot_%s.svg' %(os.path.split(ve_unwrap_file)[-1].split('.tif')[0]), 
#                            dpi=300, bbox_inches='tight')
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
#    all_table_staging_auto = pd.DataFrame(np.vstack([all_embryos_plot, 
#                                                       pre_data,
#                                                       mig_data,
#                                                       post_data]).T, columns=['Embryo', 'Pre-migration', 'Migration to Boundary', 'Post-Boundary'])
#    
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-LifeAct-smooth-lam1.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_MOSES-LifeAct-smooth-lam10-correct_rot.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_MOSES-MTMG-TTR-smooth-lam10-correct_rot.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_MOSES-MTMG-HEX-smooth-lam10-correct_rot.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_MOSES-MTMG-new-HEX-smooth-lam10-correct_norot.xlsx', index=None)
#    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-%s-smooth-lam10-correct_norot.xlsx' %(os.path.split(rootfolder)[-1]), index=None)
#
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES/auto_staging_MOSES-MTMG-TTR-smooth-lam10-correct_norot.xlsx', index=None)
#
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG_TTR-smooth-lam1.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-smooth-lam5.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-new_hex-smooth-lam1.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-1_smooth.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-new_hex-4-VE-1.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG_TTR.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES_nothresh.xlsx', index=None)
#
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_curves_MOSES-LifeAct-correct_rot.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_curves_MOSES-MTMG-TTR-correct_rot.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_curves_MOSES-MTMG-HEX-correct_rot.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_curves_MOSES-MTMG-new-HEX-correct_norot.mat'
#    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_curves_MOSES-%s-correct_norot.mat' %(os.path.split(rootfolder)[-1])
#
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES/auto_staging_curves_MOSES-MTMG-TTR-correct_norot.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-LifeAct.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-MTMG-TTR.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-MTMG-HEX-new_hex.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-MTMG-HEX.mat'
#    spio.savemat(savematfile, {'embryos': all_embryos_plot, 
#                               'staging_table': all_table_staging_auto, 
#                               'staging_curves': diff_curves_all})
#














#    import bayesian_changepoint_detection.offline_changepoint_detection as offcd
#    from functools import partial
#    from scipy.signal import find_peaks
#    
#    data = np.diff(diff_curve_component)[:,None]
#    Q, P, Pcp = offcd.offline_changepoint_detection(data, 
#                                                    partial(offcd.const_prior, 
#                                                            l=(len(data)+1)), 
#                                                            offcd.fullcov_obs_log_likelihood, 
#                                                            truncate=-100)
#    
#    peak_pos = find_peaks(np.exp(Pcp).sum(0), height=0.01, distance=10)
#                
#    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[5, 4])
##        ax = fig.add_subplot(2, 1, 1)
#    ax[0].plot(diff_curve_component[:])
#    ax[0].plot(peak_pos[0], diff_curve_component[peak_pos[0]], 'ro')
##        ax = fig.add_subplot(2, 1, 2)
#    ax[1].plot(np.exp(Pcp).sum(0))
#    ax[1].plot(peak_pos[0], np.exp(Pcp).sum(0)[peak_pos[0]], 'ro')
#    plt.show()
    
    
#    diff_curves_all = []
#    pre_all = []
#    mig_all = []
#    post_all = []
#    
#    
#    fig, ax = plt.subplots(figsize=(15,15))
#    
#    for kk in range(len(diff_curves_all)):
#        ax.plot(diff_curves_all[kk], label=embryo_folders[kk].split('/')[-1])
#        
#    plt.legend(loc='best')
#    plt.show()
                
                
###                # try incorporating neighbour features.
#                nbr_inds_ve = nbrs_model.fit(meantracks_ve[:,0]).kneighbors(meantracks_ve[:,0], return_distance=False)
##                nbr_inds_epi = nbrs_model.fit(meantracks_epi[:,0]).kneighbors(meantracks_epi[:,0], return_distance=False)
###                
#                
##                meantracks_ve_smooth_diff_nbr = meantracks_ve_smooth_diff.copy()
##                meantracks_epi_smooth_diff_nbr = meantracks_epi_smooth_diff.copy()
#                meantracks_ve_smooth_diff_nbr = meantracks_ve_smooth_diff[nbr_inds_ve].transpose(0,2,1,3)
#                meantracks_epi_smooth_diff_nbr = meantracks_epi_smooth_diff[nbr_inds_ve].transpose(0,2,1,3)
#                
#                meantracks_ve_smooth_diff_nbr = meantracks_ve_smooth_diff_nbr.reshape(meantracks_ve_smooth_diff_nbr.shape[0],
#                                                                                      meantracks_ve_smooth_diff_nbr.shape[1], -1)
#                meantracks_epi_smooth_diff_nbr = meantracks_epi_smooth_diff_nbr.reshape(meantracks_epi_smooth_diff_nbr.shape[0],
#                                                                                      meantracks_epi_smooth_diff_nbr.shape[1], -1)
#                
#                cross_corr_lags3D, cross_corr_3D_vals = cross_corr_tracksets(meantracks_ve_smooth_diff_nbr, 
#                                                                             meantracks_epi_smooth_diff_nbr, 
#                                                                             return_curves=False, normalize=True, mode='full')
##                test_ve = np.vstack(cross_corr_3D_vals)
##                test_ve = test_ve[:,test_ve.shape[1]//2:]
##                
##                from scipy.signal import find_peaks
##                
##                min_pos = [find_peaks(s.max()-s)[0][0] for s in test_ve]
##                print(np.mean(min_pos))
##                
##                
##                cross_corr_lags3D, cross_corr_3D_vals = cross_corr_tracksets_multivariate(meantracks_epi_smooth_diff_nbr, 
##                                                                                          meantracks_epi_smooth_diff_nbr, 
##                                                                                          return_curves=True, normalize=True, mode='full')
##                test_ve = np.vstack(cross_corr_3D_vals)
##                test_ve = test_ve[:,test_ve.shape[1]//2:]
##                
##                from scipy.signal import find_peaks
##                
##                min_pos = [find_peaks(s.max()-s)[0][0] for s in test_ve]
##                print(np.mean(min_pos))
#                
##                cross_corr_3D_vals = np.vstack(cross_corr_3D_vals)
##                cross_corr_3D_vals = cross_corr_3D_vals/float(meantracks_ve_smooth_diff.shape[-1]) # for however many dimensions.
#                
#                cross_corr_lags3D = np.hstack(cross_corr_lags3D)
#                cross_corr_3D_vals = np.hstack(cross_corr_3D_vals)
#                cross_corr_lags3D[np.abs(cross_corr_3D_vals) < 1e-8] = 0 
#                
#                fig, ax = plt.subplots(figsize=(15,15))
#                ax.imshow(vid_ve[0], cmap='gray')
##                plot_tracks(meantracks_ve[:,:30], ax=ax, color='g')
##                plot_tracks(meantracks_epi[:,:30], ax=ax, color='r')
#                ax.scatter(meantracks_ve[:,0,1],
#                           meantracks_ve[:,0,0], 
#                           c=cross_corr_3D_vals, cmap='coolwarm', vmin=-1, vmax=1)
#                plt.show()
#                
#                fig, ax = plt.subplots(figsize=(15,15))
#                ax.imshow(vid_ve[0], cmap='gray')
##                plot_tracks(meantracks_ve, ax=ax, color='g')
##                plot_tracks(meantracks_epi, ax=ax, color='r')
#                ax.scatter(meantracks_ve[:,0,1],
#                           meantracks_ve[:,0,0], 
#                           c=cross_corr_lags3D, cmap='coolwarm', vmin=-len(vid_ve), vmax=len(vid_ve))
#                
#                plt.show()
#                
#                
#                fig, ax = plt.subplots(figsize=(15,15))
#                ax.imshow(vid_ve[0], cmap='gray')
#                plot_tracks(meantracks_ve[:,:], ax=ax, color='g')
##                plot_tracks(meantracks_epi, ax=ax, color='r')
##                ax.scatter(meantracks_ve[:,0,1],
##                           meantracks_ve[:,0,0], 
##                           c=cross_corr_lags3D, cmap='coolwarm', vmin=-len(vid_ve), vmax=len(vid_ve))
#                plt.show()
#                
#                fig, ax = plt.subplots(figsize=(15,15))
#                ax.imshow(vid_epi_resize[0], cmap='gray')
##                plot_tracks(meantracks_ve, ax=ax, color='g')
#                plot_tracks(meantracks_epi[:,:], ax=ax, color='r')
##                ax.scatter(meantracks_ve[:,0,1],
##                           meantracks_ve[:,0,0], 
##                           c=cross_corr_lags3D, cmap='coolwarm', vmin=-len(vid_ve), vmax=len(vid_ve))
#                
#                plt.show()
#                
#            # =============================================================================
#            #       Save out the results.                
#            # =============================================================================
#                # save out the correlation values + the lags obtained. 
#                save_corr_folder = savetrackfolder + '_corr-ve-epi_smooth'
#                fio.mkdir(save_corr_folder)
#                
#                savecorrfile = trackfile.replace(savetrackfolder, save_corr_folder).replace('.mat', '_corr-ve-epi.mat')
#                
#                spio.savemat(savecorrfile, {'trackfile': trackfile,
#                                            'xcorr_3D_vals': cross_corr_3D_vals, 
#                                            'xcorr_3D_lags': cross_corr_lags3D, 
#                                            'unwrapfile_ve': unwrap_param_file_ve, 
#                                            'unwrapfile_epi': unwrap_param_file_epi,
#                                            'smoothwinsize': smoothwinsize})
                
                
#                for frame in range(meantracks_ve.shape[1]):
#                    
#                    fig, ax = plt.subplots(figsize=(7,7))
#                    ax.imshow(vid_ve[frame])
#                    ax.plot(meantracks_ve[:,frame,1], 
#                            meantracks_ve[:,frame,0], 'go')
#                    ax.plot(meantracks_epi[:,frame,1], 
#                            meantracks_epi[:,frame,0], 'ro')
#                    plt.show()

#                test_region = 758
#                
#                mean_direction_ve_test = np.mean(meantracks_ve[test_region,1:] - meantracks_ve[test_region,:-1], axis=0)
#                mean_direction_epi_test = np.mean(meantracks_epi[test_region,1:] - meantracks_epi[test_region,:-1], axis=0)
#                
#                track_ve_test = meantracks_ve[test_region].copy()
#                track_epi_test = meantracks_epi[test_region].copy()
#                
#                # smooth these curves. 
#                from scipy.signal import savgol_filter
#                
#                track_ve_test[...,0] = savgol_filter(track_ve_test[...,0], window_length=3, polyorder=1)
#                track_ve_test[...,1] = savgol_filter(track_ve_test[...,1], window_length=3, polyorder=1)
#                
#                track_epi_test[...,0] = savgol_filter(track_epi_test[...,0], window_length=3, polyorder=1)
#                track_epi_test[...,1] = savgol_filter(track_epi_test[...,1], window_length=3, polyorder=1)
#                
#                
#                fig, ax = plt.subplots(figsize=(20,20))
##                ax.imshow(vid_ve[0])
#                plot_tracks(meantracks_ve[test_region,:][None,:], ax=ax, color='r')
#                ax.plot(meantracks_ve[test_region,0,1],
#                        meantracks_ve[test_region,0,0], 'ko')
#                ax.plot([meantracks_ve[test_region,0,1], meantracks_ve[test_region,0,1] + 20*mean_direction_ve_test[1]],
#                        [meantracks_ve[test_region,0,0], meantracks_ve[test_region,0,0] + 20*mean_direction_ve_test[0]], 'r--')
#                
#                plot_tracks(meantracks_epi[test_region,:][None,:], ax=ax, color='g')
#                plot_tracks(track_epi_test[None,:], ax=ax, color='g', lw=3, alpha=0.5)
#                plot_tracks(track_ve_test[None,:], ax=ax, color='r', lw=3)
#                
#                ax.plot(meantracks_epi[test_region,0,1],
#                        meantracks_epi[test_region,0,0], 'ko')
#                ax.plot([meantracks_epi[test_region,0,1], meantracks_epi[test_region,0,1] + 20*mean_direction_epi_test[1]],
#                        [meantracks_epi[test_region,0,0], meantracks_epi[test_region,0,0] + 20*mean_direction_epi_test[0]], 'g--')
#                
#                plt.show()
               
                
                
#                # embedding? 
#                import skccm as ccm
#                
#                lag = 5
#                embed = 3
#                e1 = ccm.Embed(meantracks_ve[test_region,1:,0] - meantracks_ve[test_region,:-1,0])
##                e1y = ccm.Embed(meantracks_ve[test_region,:,1])
#                e2 = ccm.Embed(meantracks_epi[test_region,1:,0] - meantracks_epi[test_region,:-1,0])
#                X1 = e1.embed_vectors_1d(lag,embed)
#                X2 = e2.embed_vectors_1d(lag,embed)
#                
#                plt.figure()
#                plt.plot(X1[:,0], X1[:,1], 'o')
#                plt.figure()
#                plt.plot(X2[:,0], X2[:,1], 'o')
#                plt.show()
#                
#                
#                from skccm.utilities import train_test_split
#
#                #split the embedded time series
#                x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)
#                
#                CCM = ccm.CCM() #initiate the class
#                
#                #library lengths to test
#                len_tr = len(x1tr)
#                lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')
#                
#                #test causation
#                CCM.fit(x1tr,x2tr)
#                x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)
#                
#                sc1,sc2 = CCM.score()
#                
#                plt.figure()
#                plt.plot(lib_lens, sc1, label='x1')
#                plt.plot(lib_lens, sc2, label='x2')
#                plt.ylim([-1,1])
#                plt.legend()
#                plt.show()
#                
##                meantracks_ve_3D_diff = (meantracks_ve[:,:] - meantracks_ve[:,0][:,None]).astype(np.float32)
##                meantracks_epi_3D_diff = (meantracks_epi[:,:] - meantracks_epi[:,0][:,None]).astype(np.float32)
#                meantracks_ve_3D_diff = (meantracks_ve[:,1:] - meantracks_ve[:,:-1]).astype(np.float32)
#                meantracks_epi_3D_diff = (meantracks_epi[:,1:] - meantracks_epi[:,:-1]).astype(np.float32)
#
#                corrs = cross_corr_tracksets(meantracks_ve_3D_diff[test_region:test_region+1], 
#                                             meantracks_epi_3D_diff[test_region:test_region+1], 
#                                             return_curves=True, normalize=True, mode='full')
#                
#                plt.figure()
#                plt.plot(corrs[-1][0])
#                plt.hlines(0, 0, len(corrs[-1][0]))
#                plt.vlines(len(corrs[-1][0])//2, -1,1)
#                plt.show()
                
#                corrs = cross_corr_tracksets( (track_ve_test[1:]-track_ve_test[:-1])[None,:], 
#                                              (track_epi_test[1:]-track_epi_test[:-1])[None,:], 
#                                             return_curves=True, normalize=True, mode='full')
#                print(corrs[0])
#                plt.figure()
#                plt.plot(corrs[-1][0])
#                plt.hlines(0, 0, len(corrs[-1][0]))
#                plt.vlines(len(corrs[-1][0])//2, -1,1)
#                plt.show()
#                
#                corrs = cross_corr_tracksets2D( (track_ve_test[1:]-track_ve_test[:-1])[None,:], 
#                                                (track_epi_test[1:]-track_epi_test[:-1])[None,:], 
#                                             return_curves=True, normalize=True, mode='full')
#                print(corrs[0])
#                plt.figure()
#                plt.plot(corrs[-1][0])
#                plt.hlines(0, 0, len(corrs[-1][0]))
#                plt.vlines(len(corrs[-1][0])//2, -1,1)
#                plt.show()
#                
#                
#                
#                fig, ax = plt.subplots(figsize=(5,5))
##                ax.imshow(vid_ve[0])
##                plot_tracks(meantracks_ve[test_region,:][None,:], ax=ax, color='r')
#                ax.plot(meantracks_ve[test_region,0,1],
#                        meantracks_ve[test_region,0,0], 'ko')
##                ax.plot([meantracks_ve[test_region,0,1], meantracks_ve[test_region,0,1] + 20*mean_direction_ve_test[1]],
##                        [meantracks_ve[test_region,0,0], meantracks_ve[test_region,0,0] + 20*mean_direction_ve_test[0]], 'r--')
##                plot_tracks(meantracks_epi[test_region,:][None,:], ax=ax, color='g')
#                plot_tracks(track_epi_test[None,:], ax=ax, color='r', lw=3, alpha=0.5)
#                plot_tracks(track_ve_test[None,63:], ax=ax, color='g', lw=3)
#                ax.plot(meantracks_epi[test_region,0,1],
#                        meantracks_epi[test_region,0,0], 'ko')
##                ax.plot([meantracks_epi[test_region,0,1], meantracks_epi[test_region,0,1] + 20*mean_direction_epi_test[1]],
##                        [meantracks_epi[test_region,0,0], meantracks_epi[test_region,0,0] + 20*mean_direction_epi_test[0]], 'g--')
#                
#                plt.show()
#                
#                
                
                
                
                
#                
##                corrs = cross_corr_tracksets( (track_ve_test[1:]-track_ve_test[:-1])[None,:], 
##                                              (track_ve_test[1:]-track_ve_test[:-1])[None,:], 
##                                             return_curves=True, normalize=True, mode='full')
#                corrs = cross_corr_tracksets( (track_ve_test[None,:]), 
#                                              (track_epi_test[None,:]), 
#                                             return_curves=True, normalize=True, mode='full')
#                
#                print(corrs[0])
#                plt.figure()
#                plt.plot(corrs[-1][0])
#                plt.hlines(0, 0, len(corrs[-1][0]))
#                plt.vlines(len(corrs[-1][0])//2, -1,1)
#                plt.show()
                
                
                    
                    
#                # compare the correlation in 2D and in 3D ?
#                meantracks_ve_3D = unwrap_params_ve[meantracks_ve[...,0], meantracks_ve[...,1]]
#                meantracks_epi_3D = unwrap_params_epi[meantracks_epi[...,0], meantracks_epi[...,1]]
#                
##                meantracks_ve_3D_diff = (meantracks_ve_3D[:,1:] - meantracks_ve_3D[:,:-1]).astype(np.float32)
##                meantracks_epi_3D_diff = (meantracks_epi_3D[:,1:] - meantracks_epi_3D[:,:-1]).astype(np.float32)
#                
#                meantracks_ve_3D_diff = (meantracks_ve_3D[:,:] - meantracks_ve_3D[:,0][:,None]).astype(np.float32)
#                meantracks_epi_3D_diff = (meantracks_epi_3D[:,:] - meantracks_epi_3D[:,0][:,None]).astype(np.float32)
#                
##                meantracks_ve_3D_diff = (meantracks_ve[:,1:] - meantracks_ve[:,0][:,None]).astype(np.float32)
##                meantracks_epi_3D_diff = (meantracks_epi[:,1:] - meantracks_epi[:,0][:,None]).astype(np.float32)
#                cross_corr_lags3D, cross_corr_3D_vals = cross_corr_tracksets(meantracks_ve_3D_diff, 
#                                                                             meantracks_epi_3D_diff, 
#                                                                             return_curves=False, normalize=True, mode='full')
                
                
                
                
                
                
                
#                cross_corr_3D_vals = np.vstack(cross_corr_3D_vals)
#                
#                cross_corr_lags3D = np.hstack(cross_corr_lags3D)
#                cross_corr_3D_vals = np.hstack(cross_corr_3D_vals)
#                cross_corr_lags3D[np.abs(cross_corr_3D_vals) < 1e-8] = 0 
#                
#                fig, ax = plt.subplots(figsize=(10,10))
#                ax.imshow(vid_ve[0], cmap='gray')
#                ax.scatter(meantracks_ve[:,0,1],
#                           meantracks_ve[:,0,0], 
#                           c=cross_corr_3D_vals, cmap='coolwarm', vmin=-1, vmax=1)
#                plt.show()
#                
#                fig, ax = plt.subplots(figsize=(10,10))
#                ax.imshow(vid_ve[0], cmap='gray')
##                plot_tracks(meantracks_ve, ax=ax, color='g')
##                plot_tracks(meantracks_epi, ax=ax, color='r')
#                ax.scatter(meantracks_ve[:,0,1],
#                           meantracks_ve[:,0,0], 
#                           c=cross_corr_lags3D, cmap='coolwarm', vmin=-len(vid_ve), vmax=len(vid_ve))
#                
#                plt.show()
#                
#                
#                fig, ax = plt.subplots(figsize=(10,10))
#                ax.imshow(vid_ve[0], cmap='gray')
#                plot_tracks(meantracks_ve, ax=ax, color='g')
##                plot_tracks(meantracks_epi, ax=ax, color='r')
##                plot_tracks(meantracks_hex, ax=ax, color='b')
#                plt.show()
#                
#                fig, ax = plt.subplots(figsize=(10,10))
#                ax.imshow(vid_ve[0], cmap='gray')
##                plot_tracks(meantracks_ve, ax=ax, color='g')
#                plot_tracks(meantracks_epi, ax=ax, color='r')
##                plot_tracks(meantracks_hex, ax=ax, color='b')
#                plt.show()
#                

    
