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


#def tile_uniform_windows_radial_guided_line(imsize, n_r, n_theta, max_r, mid_line, center=None, bound_r=True):
#    
#    from skimage.segmentation import mark_boundaries
#    m, n = imsize
#    
#    XX, YY = np.meshgrid(range(n), range(m))
#    
#    if center is None:
#        center = np.hstack([m/2., n/2.])
#
#    r2 = (XX - center[1])**2  + (YY - center[0])**2
#    r = np.sqrt(r2)
#    theta = np.arctan2(YY-center[0],XX-center[1])
#    
#    if max_r is None:
#        if bound_r: 
#            max_r = np.minimum(np.abs(np.max(XX-center[1])),
#                               np.abs(np.max(YY-center[0])))
#        else:
#            max_r = np.maximum(np.max(np.abs(XX-center[1])),
#                               np.max(np.abs(YY-center[0])))
#
#    """
#    construct contour lines that bipartition the space. 
#    """
#    mid_line_polar = np.vstack(cart2pol(mid_line[:,0] - center[1], mid_line[:,1] - center[0])).T
#    mid_line_central = np.vstack(pol2cart(mid_line_polar[:,0], mid_line_polar[:,1])).T
#    
#    # derive lower and upper boundary lines -> make sure to 
#    contour_r_lower_polar = np.array([np.linspace(0, l, n_r//2+1) for l in mid_line_polar[:,0]]).T
#    contour_r_upper_polar = np.array([np.linspace(l, max_r, n_r//2+1) for l in mid_line_polar[:,0]]).T
#    
#    contour_r_lower_lines = [np.vstack(pol2cart(l, mid_line_polar[:,1])).T for l in contour_r_lower_polar][:-1]
#    contour_r_upper_lines = [np.vstack(pol2cart(l, mid_line_polar[:,1])).T for l in contour_r_upper_polar][1:]
#    
#    all_dist_lines = contour_r_lower_lines + [mid_line_central] + contour_r_upper_lines
#    all_dist_lines = [np.vstack([ll[:,0] + center[1], ll[:,1] + center[0]]).T  for ll in all_dist_lines]
#    all_dist_masks = [draw_polygon_mask(ll, imsize) for ll in all_dist_lines]
#    # now construct binary masks. 
#    all_dist_masks = [np.logical_xor(all_dist_masks[ii+1], all_dist_masks[ii]) for ii in range(len(all_dist_masks)-1)] # ascending distance. 
#    
#    """
#    construct angle masks to partition the angle space. 
#    """
#    # 2. partition the angular direction
#    angle_masks_list = [] # list of the binary masks in ascending angles. 
#    theta_bounds = np.linspace(-np.pi, np.pi, n_theta+1)
#    
#    for ii in range(len(theta_bounds)-1):
#        mask_theta = np.logical_and( theta>=theta_bounds[ii], theta <= theta_bounds[ii+1])
#        angle_masks_list.append(mask_theta)
#        
#    """
#    construct the final set of masks.  
#    """  
#    spixels = np.zeros((m,n), dtype=np.int)
#    
#    counter = 1
#    for ii in range(len(all_dist_masks)):
#        for jj in range(len(angle_masks_list)):
#            mask = np.logical_and(all_dist_masks[ii], angle_masks_list[jj])
#
#            spixels[mask] = counter 
#            counter += 1
#        
#    return spixels
    


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


def robust_fit_ellipse(pts_2D):
    
    from skimage.measure import EllipseModel, ransac


#    ellipse = EllipseModel()
#    ellipse.estimate(xy)
    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(pts_2D, 
                                   EllipseModel, 
                                   min_samples=4,
                                   residual_threshold=1, max_trials=100)
    
    return model_robust
    
def delaunay_region_pts(pts, shape):
    
    from scipy.spatial import Delaunay
    from skimage.draw import polygon
    
    mask = np.zeros(shape)
    
    tri = Delaunay(pts)
    
    for tri_pts in tri.simplices:
        
        tri_pts = pts[tri_pts]
        
        rr, cc = polygon(tri_pts[:,0], tri_pts[:,1])
        mask[rr,cc] = 1
        
    return mask>0

def estimate_ellipse_2D_binary(pts_2D):
    
    from skimage.measure import regionprops
    
    min_x = int(np.min(pts_2D[:,0]))
    max_x = int(np.max(pts_2D[:,0]))
    min_y = int(np.min(pts_2D[:,1]))
    max_y = int(np.max(pts_2D[:,1]))
    
    m = max_y - min_y + 1
    n = max_x - min_x + 1
#    blank = np.zeros((m,n))
    
    pts_2D_ = pts_2D.copy()
    pts_2D_[:,0] = pts_2D_[:,0] - min_x
    pts_2D_[:,1] = pts_2D_[:,1] - min_y
    
    binary = delaunay_region_pts(pts_2D_[:,::-1], (m,n))
    binary_prop = regionprops((binary*1).astype(np.int))[0]
    
#    plt.figure()
#    plt.imshow(binary)
#    plt.show()
    
    centroid = np.hstack(binary_prop.centroid)
    major_len = binary_prop.major_axis_length / 2.
    minor_len = binary_prop.minor_axis_length / 2.
    orientation  = binary_prop.orientation
    
    # reproject these back into the original 'space'
    centroid = centroid[::-1] + np.hstack([min_x, min_y])
    
    
    return np.hstack([centroid[::-1], major_len, minor_len, orientation])


def fit_ellipse_3D(pts3D):
    
    from sklearn.decomposition import PCA
    
    pts = pts3D- pts3D.mean(axis=0)[None,:]
#    cov = np.cov(pts.T)
#    evalues, evecs = np.linalg.eigh(cov)
#    
#    sort_evalues = np.argsort(np.abs(evalues))
#    evecs = evecs[:,sort_evalues]
#    evalues = np.sqrt(evalues[sort_evalues])
    
    # we need to do the PCA to project. 
    pca_model = PCA(n_components=2, whiten=False)
    pts_2D = pca_model.fit_transform(pts)
    pts_vec_3D = pca_model.components_
    
    # fit the ellipse to the projected 2D to refine the mean + direction vector
#                                """
#                                method 1: SVD on points directly. 
#                                """
#                                ell_model_2D = robust_fit_ellipse(pts_2D)
#                                ell_model_2D_params = ell_model_2D.params
#                                
#                                x_c, y_c, a, b, theta = ell_model_2D_params
    
    """
    method 2: compute parameters from the image.
    """
    ell_binary_2D = estimate_ellipse_2D_binary(pts_2D)
    
    x_c, y_c, a, b, theta = ell_binary_2D
    center_c = np.hstack([x_c, y_c])
    dir_c = center_c + np.hstack([a*np.cos(theta), a*np.sin(theta)])
    minor_c = center_c + np.hstack([b*np.sin(theta), -b*np.cos(theta)])
    
    """
    reproject back into 3D to get the proper angle. 
    """
    mean_centre_3D = pca_model.inverse_transform(center_c[None,:])[0] + pts3D.mean(axis=0)
    dir_centre_3D = pca_model.inverse_transform(dir_c[None,:])[0] + pts3D.mean(axis=0)
    dir_minor_centre_3D = pca_model.inverse_transform(minor_c[None,:])[0] + pts3D.mean(axis=0)
    
#    major_direction_3D_ve = dir_centre_3D - embryo_ve_centres_3D[frame] # 3D - 3D. 
    return a, b, dir_centre_3D


#def fit_ellipse_3D_time(pts3D):
#    
#    from sklearn.decomposition import PCA
#    
#    pts = pts3D - pts3D.mean(axis=1)[:, None,:]
#    
#    """
#    PCA in 3D to solve the plane. 
#    """
#    pts_ = pts.transpose(0,2,1) - pts.transpose(0,2,1).mean(axis=2)[:,:,None]
#    cov =  np.matmul(pts_, pts_.transpose(0,2,1)) / (pts_.shape[-1]-1)
#
#    evalues, evecs = np.linalg.eigh(cov)
##    
##    sort_evalues = np.argsort(np.abs(evalues))
##    evecs = evecs[:,sort_evalues]
##    evalues = np.sqrt(evalues[sort_evalues])
#    
#    # we need to do the PCA to project. 
#    pca_model = PCA(n_components=2, whiten=False)
#    pts_2D = pca_model.fit_transform(pts)
#    pts_vec_3D = pca_model.components_
#    
#    # fit the ellipse to the projected 2D to refine the mean + direction vector
##                                """
##                                method 1: SVD on points directly. 
##                                """
##                                ell_model_2D = robust_fit_ellipse(pts_2D)
##                                ell_model_2D_params = ell_model_2D.params
##                                
##                                x_c, y_c, a, b, theta = ell_model_2D_params
#    
#    """
#    method 2: compute parameters from the image.
#    """
#    ell_binary_2D = estimate_ellipse_2D_binary(pts_2D)
#    
#    x_c, y_c, a, b, theta = ell_binary_2D
#    center_c = np.hstack([x_c, y_c])
#    dir_c = center_c + np.hstack([a*np.cos(theta), a*np.sin(theta)])
#    minor_c = center_c + np.hstack([b*np.sin(theta), -b*np.cos(theta)])
#    
#    """
#    reproject back into 3D to get the proper angle. 
#    """
#    mean_centre_3D = pca_model.inverse_transform(center_c[None,:])[0] + pts3D.mean(axis=0)
#    dir_centre_3D = pca_model.inverse_transform(dir_c[None,:])[0] + pts3D.mean(axis=0)
#    dir_minor_centre_3D = pca_model.inverse_transform(minor_c[None,:])[0] + pts3D.mean(axis=0)
#    
##    major_direction_3D_ve = dir_centre_3D - embryo_ve_centres_3D[frame] # 3D - 3D. 
#    return a, b, dir_centre_3D

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

def poly_perimeter(poly):

    N  = len(poly)

    dist = 0

    for i in range(N-1):
        vi1 = poly[i]
        vi2 = poly[i+1]

        vi1_vi2_dist = np.linalg.norm(vi1-vi2)
        dist += vi1_vi2_dist

    return dist


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

def get_reg_boundary_mask(yx_pts, shape):

    from skimage.draw import polygon

    mask = np.zeros(shape)
    rr,cc = polygon(yx_pts[:,0], yx_pts[:,1], shape=shape)
    mask[rr,cc] = 1

    return mask > 0 


"""
Custom script
"""
def fit_ellipse_3D_to_cells(pts3D):
    
    from sklearn.decomposition import PCA
    
    pts = pts3D- pts3D.mean(axis=0)[None,:]
    cov = np.cov(pts.T)
    evalues, evecs = np.linalg.eigh(cov)
   
    sort_evalues = np.argsort(np.abs(evalues))
    evecs = evecs[:,sort_evalues]
    evalues = np.sqrt(evalues[sort_evalues]) # starts from smallest. 

    return evalues
    
#     # we need to do the PCA to project. 
#     pca_model = PCA(n_components=2, whiten=False)
#     pts_2D = pca_model.fit_transform(pts)
#     pts_vec_3D = pca_model.components_
    
#     # fit the ellipse to the projected 2D to refine the mean + direction vector
# #                                """
# #                                method 1: SVD on points directly. 
# #                                """
# #                                ell_model_2D = robust_fit_ellipse(pts_2D)
# #                                ell_model_2D_params = ell_model_2D.params
# #                                
# #                                x_c, y_c, a, b, theta = ell_model_2D_params
    
#     """
#     method 2: compute parameters from the image.
#     """
#     ell_binary_2D = estimate_ellipse_2D_binary(pts_2D)
    
#     x_c, y_c, a, b, theta = ell_binary_2D
#     center_c = np.hstack([x_c, y_c])
#     dir_c = center_c + np.hstack([a*np.cos(theta), a*np.sin(theta)])
#     minor_c = center_c + np.hstack([b*np.sin(theta), -b*np.cos(theta)])
    
#     """
#     reproject back into 3D to get the proper angle. 
#     """
#     mean_centre_3D = pca_model.inverse_transform(center_c[None,:])[0] + pts3D.mean(axis=0)
#     dir_centre_3D = pca_model.inverse_transform(dir_c[None,:])[0] + pts3D.mean(axis=0)
#     dir_minor_centre_3D = pca_model.inverse_transform(minor_c[None,:])[0] + pts3D.mean(axis=0)
    
# #    major_direction_3D_ve = dir_centre_3D - embryo_ve_centres_3D[frame] # 3D - 3D. 
#     return a, b, dir_centre_3D


def compute_cell_area(cell_img, unwrap_params, invalid_mask, unwrap_params_normals=None):
    
    from skimage.measure import find_contours
    
    uniq_cells = np.setdiff1d(np.unique(cell_img), 0 )

    cell_areas_3D = []
    cell_perimeters_3D = []
    cell_centroids_2D = []
    cell_major_minor_lengths = []

    for cc in uniq_cells:

        reg_boundary = find_contours(cell_img == cc, 0)
        reg_boundary_lens = [len(rr) for rr in reg_boundary]
        reg_boundary = reg_boundary[np.argmax(reg_boundary_lens)] # only one boundary
        
        # map the entire area into 3d and take the centroid.
        reg_area_mask = get_reg_boundary_mask(reg_boundary, cell_img.shape)
        reg_area_3D_pts = unwrap_params[reg_area_mask>0]
        
        centroid = reg_area_3D_pts.mean(axis=0) # this is (x,y,z)
        
        # we would need to remap to 2D to get the equivalent 2D. 
        centroid_2D_ind = np.argmin(np.linalg.norm(unwrap_params - centroid[None,None,:], axis=-1))
        centroid_2D_ind_y, centroid_2D_ind_x = np.unravel_index([centroid_2D_ind], cell_img.shape)
        centroid_2D_ind_y = centroid_2D_ind_y[0]
        centroid_2D_ind_x = centroid_2D_ind_x[0]

        # use the centroid to 'lookup' the polygon normal
        surf_normal = unwrap_params_normals[int(centroid_2D_ind_y), int(centroid_2D_ind_x)]

        # this is should be unit length. 
#        print(np.linalg.norm(surf_normal))

        # map the closed boundaries only into 3D. 
        reg_boundary_3D = unwrap_params[reg_boundary[:,0].astype(np.int), 
                                        reg_boundary[:,1].astype(np.int)]
        
        reg_boundary_3D_invalid = invalid_mask[reg_boundary[:,0].astype(np.int), 
                                               reg_boundary[:,1].astype(np.int)]   
    
        reg_boundary_3D_refine = reg_boundary_3D[reg_boundary_3D_invalid==0]
    
        # ensure the boundary is closed. 
        if len(reg_boundary_3D_refine) < len(reg_boundary_3D):
            reg_boundary_3D_refine = np.vstack([reg_boundary_3D_refine, reg_boundary_3D_refine[0]])
    
    #                # for debugging.
    #                fig = plt.figure()
    #                ax = fig.add_subplot(111, projection='3d')
    ##                
    #                ax.plot(reg_boundary_3D_refine[:,0], 
    #                           reg_boundary_3D_refine[:,1], 
    #                           reg_boundary_3D_refine[:,2], c='r')
    #                ax.scatter(unwrap_params[::20, ::20, 0].ravel(), 
    #                           unwrap_params[::20, ::20, 1].ravel(),
    #                           unwrap_params[::20, ::20, 2].ravel(), c='k', alpha=0.5)
    ##                plt.show()
        # compute the polygon area. 
        surf_area = poly_area(reg_boundary_3D_refine, normal=surf_normal)
        surf_perimeter = poly_perimeter(reg_boundary_3D_refine)
        
        # estimate the major and minor axis lengths. 
        reg_evalues_3D_pts = fit_ellipse_3D_to_cells(reg_area_3D_pts) # use the entire regions's pts. 
        # print(reg_evalues_3D_pts)
        major_length = reg_evalues_3D_pts[-1]
        minor_length = reg_evalues_3D_pts[-2]

#        print(surf_area, surf_perimeter)
        cell_areas_3D.append(surf_area)
        cell_perimeters_3D.append(surf_perimeter)
        cell_centroids_2D.append(np.hstack([centroid_2D_ind_y, centroid_2D_ind_x]))
        cell_major_minor_lengths.append(np.hstack([major_length, minor_length]))

    cell_areas_3D = np.hstack(cell_areas_3D)
    cell_perimeters_3D = np.hstack(cell_perimeters_3D)
    cell_centroids_2D = np.vstack(cell_centroids_2D)
    cell_major_minor_lengths = np.vstack(cell_major_minor_lengths)
    
#    fig, ax = plt.subplots()
#    ax.imshow(cell_img)
#    ax.plot(cell_centroids_2D[:,1], 
#            cell_centroids_2D[:,0], 'w.')
#    plt.show()

    return uniq_cells, (cell_areas_3D, cell_perimeters_3D, cell_major_minor_lengths)


def assign_stats_to_tracks(cell_tracks_ids, cell_ids_list, cell_stats_list):

    n_tracks, n_time = cell_tracks_ids.shape
    n_stats = len(cell_stats_list)

    cell_track_stats_out = np.zeros((n_tracks, n_time, n_stats))
    cell_track_stats_out[:] = np.nan
    
#    print(cell_track_stats_out.shape)

    for track_ii in np.arange(n_tracks):
        for time_ii in np.arange(n_time):

            track_id_time = cell_tracks_ids[track_ii, time_ii]
#            print(track_id_time)

            if track_id_time > 0: 
                cell_id_list = cell_ids_list[time_ii]
                cell_id_select = np.arange(len(cell_id_list))[cell_id_list==track_id_time]
#                print(cell_id_select)
                cell_id_select = cell_id_select[0]
                
                for stat_ii in np.arange(n_stats):
                    cell_track_stats_out[track_ii, time_ii, stat_ii] = cell_stats_list[stat_ii][time_ii][cell_id_select]

    return cell_track_stats_out


#def # add a quad heatmap function
def heatmap_vals_cell_metrics(cells, vals):
    
    # this is just for one image. 
    shape = cells.shape
    uniq_cells = np.setdiff1d(np.unique(cells), 0)
    
    n = len(uniq_cells)
    
    assert(len(vals)==n)
    
#    heatmap = np.ones((shape[0], shape[1], 4)) * alpha # this is a color mappable.... 
    heatmap = np.zeros(shape)
    heatmap[:] = np.nan
    
    for ii in range(n):
        cell_id = uniq_cells[ii]
        cell_mask = cells == cell_id
        heatmap[cell_mask] = vals[ii]
    
    return heatmap

def lookup_2D_region_centroids_in_3D(labelled, unwrap_params):

    uniq_regions = np.setdiff1d(np.unique(labelled), 0)

    centroids_2D = []
    centroids_3D = []
    
    for re in uniq_regions:
        
        binary = labelled == re # 2D area mask 
        binary_3D = unwrap_params[binary,:].copy() # equivalent 3D mask
        
        binary_centre_3D = np.nanmean(binary_3D, axis=0)

        # map back to 2D. 
        binary_centre_3D_dists = unwrap_params - binary_centre_3D[None,None,:]
        binary_centre_3D_dists_min = np.argmin(np.linalg.norm(binary_centre_3D_dists, axis=-1))
        index_i, index_j = np.unravel_index(binary_centre_3D_dists_min, 
                                            unwrap_params.shape[:-1]) 
        # print('enforcing', index_i, index_j)
        binary_centre_2D = np.hstack([index_i, index_j])

        centroids_2D.append(binary_centre_2D)
        centroids_3D.append(binary_centre_3D)
        

    centroids_2D = np.vstack(centroids_2D)
    centroids_3D = np.vstack(centroids_3D)
    
    return centroids_2D, centroids_3D


def cell_tracks_mother_daughter_ids( cell_tracks_ids, div_times, mother_ids, daugh1_ids, daugh2_ids, max_time):

    cell_track_select = np.arange(len(cell_tracks_ids))
    cell_track_lineage = []

    for ii in range(len(div_times)):

        div_ii = int(div_times[ii])
        mother_ids_ii = int(mother_ids[ii])
        daugh1_ids_ii = int(daugh1_ids[ii])
        daugh2_ids_ii = int(daugh2_ids[ii])

        # find the corresponding track ids. 
        tra_mother = cell_track_select[cell_tracks_ids[:,div_ii]==mother_ids_ii]
        tra_daugh1 = cell_track_select[cell_tracks_ids[:,div_ii+1]==daugh1_ids_ii]
        tra_daugh2 = cell_track_select[cell_tracks_ids[:,div_ii+1]==daugh2_ids_ii]

        if div_ii < max_time - 2:
#        print(div_ii, mother_ids_ii, daugh1_ids_ii, daugh2_ids_ii) # should be able to detect this. 
#        print(ii, tra_mother, tra_daugh1, tra_daugh2)
#        print('====')
            # cell_track_lineage.append([tra_mother[0], tra_daugh1[0], tra_daugh2[0]])
            entry = []
            if len(tra_mother)>0:
                entry.append(tra_mother[0])
            if len(tra_daugh1)>0:
                entry.append(tra_daugh1[0])
            if len(tra_daugh2)>0:
                entry.append(tra_daugh2[0])
            
            cell_track_lineage.append(entry)
        else:
            cell_track_lineage.append([])

    return cell_track_lineage

def lookup_cell_to_tra_neighbors(cell_neighbors, cell_tracks_cell_ids, cell_times):

    # convert the previous cell neighbor ids to track ids, difference being the track id would remain consistent for the same cell. 
    n_cells = len(cell_neighbors)
    n_time = len(cell_times)

    tra_neighbors = []

    for c_ii in range(n_cells):
        tra_neighbors_c_ii = []

        for tt in range(n_time):

            neigh = cell_neighbors[c_ii][tt]

            if len(neigh) == 0:
                tra_neighbors_c_ii.append([])
            else:
                if len(neigh.shape) > 1:
                    neigh = neigh[0]
                time = cell_times[tt]
                tra_neigh = []

                for nn in neigh:
                    nn_tra_id = np.arange(len(cell_tracks_cell_ids))[cell_tracks_cell_ids[:,time] == nn]
                    tra_neigh.append(nn_tra_id)

                tra_neigh = np.hstack(tra_neigh); tra_neigh = np.unique(tra_neigh)
                tra_neighbors_c_ii.append(tra_neigh)

        tra_neighbors.append(tra_neighbors_c_ii)

    return tra_neighbors

def map_stats_to_tracks(cell_tracks_ids, 
                        cell_ids_list, 
                        cell_stats_list,
                        times):
    
    """
    cell_tracks_ids : tracks given as an array of cell ids with cell_id = 0 being the background 
    cell_ids_list : list of all available cell_ids 
    cell_stats_list : this the particular stat we are trying to map onto the track. 
    """

    n_tracks, n_time = cell_tracks_ids.shape
#    n_stats = len(cell_stats_list)

    cell_track_stats_out = np.zeros((n_tracks, n_time))
    cell_track_stats_out[:] = np.nan # initialise with nan to prevent it appearing in statistics. 
    
    for track_ii in np.arange(n_tracks):
        for time_ii in times: # evaluate at the specific given times. 

            track_id_time = cell_tracks_ids[track_ii, time_ii]

            if track_id_time > 0: # if not background 
                cell_id_list = np.squeeze(cell_ids_list[time_ii]) # load up all available cell ids at this timepoint. 
                cell_id_select = np.arange(len(cell_id_list))[cell_id_list==track_id_time] # try to match up.
#                print(cell_id_select)
                if len(cell_id_select) > 0:
                    cell_id_select = cell_id_select[0]
#                    print(cell_id_select)
#                    for stat_ii in np.arange(n_stats):
                    cell_track_stats_out[track_ii, time_ii] = np.squeeze(cell_stats_list[time_ii])[cell_id_select]

    return cell_track_stats_out


def map_stat_vects_to_tracks(cell_tracks_ids, 
                            cell_ids_list, 
                            cell_stats_list,
                            times,
                            ndim=None):
    
    
    n_tracks, n_time = cell_tracks_ids.shape
#    n_stats = len(cell_stats_list)

    if ndim is None:
        cell_track_stats_out = []
    else:
        cell_track_stats_out = np.zeros((n_tracks, n_time, ndim))
        cell_track_stats_out[:] = np.nan # initialise with nan to prevent it appearing in statistics. 
    
    for track_ii in np.arange(n_tracks):
        
        cell_track_stats_out_ii = [] # have to compile it before appending to the master. 
        
        for time_ii in times: # evaluate at the specific given times. 

            track_id_time = cell_tracks_ids[track_ii, time_ii]

            if track_id_time > 0: # if not background 
                cell_id_list = np.squeeze(cell_ids_list[time_ii]) # load up all available cell ids at this timepoint. 
                cell_id_select = np.arange(len(cell_id_list))[cell_id_list==track_id_time] # try to match up.
#                print(cell_id_select)
                if len(cell_id_select) > 0:
                    cell_id_select = cell_id_select[0]
#                    print(cell_id_select)
#                    for stat_ii in np.arange(n_stats):
                    stat = np.squeeze(cell_stats_list[time_ii])[cell_id_select]
#                    print(stat)
                    if ndim is None:
                        cell_track_stats_out_ii.append(stat)
                    else:
                        # append the out array! 
                        cell_track_stats_out[track_ii, time_ii, :] = np.hstack(stat)
                else:
                    if ndim is None:
                        cell_track_stats_out_ii.append([]) # append a blank. 
                    else:
#                        cell_track_stats_out_ii.append([])
                        pass
            else:
                # still return something!. 
                if ndim is None:
                    cell_track_stats_out_ii.append([]) # append a blank. 
                else:
#                    cell_track_stats_out_ii.append([])
                    pass
        if ndim is None:
            cell_track_stats_out.append(cell_track_stats_out_ii)
        else:
            pass

    return cell_track_stats_out
    

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


def assign_cell_track_quad_id(cell_track_yx_2D, 
                              polar_grids_time, debug_viz=False): 
    
    import numpy as np 
    
    cell_track_quads = np.zeros(len(cell_track_yx_2D), dtype=np.int)
    
    for track_ii in range(len(cell_track_yx_2D)):
        
        cell_track_ii_yx = cell_track_yx_2D[track_ii]
        tra_start = np.arange(len(cell_track_ii_yx))[~np.isnan(cell_track_ii_yx[:,0])]
        
        if len(tra_start) > 0:
            pos_start = cell_track_ii_yx[tra_start[0]]
            quad_id = polar_grids_time[tra_start[0], int(pos_start[0]), int(pos_start[1])]
            
            if debug_viz:
                plt.figure(figsize=(10,10))
                plt.imshow(polar_grids_time[tra_start[0]])
                plt.plot(pos_start[1],
                         pos_start[0], 'o')
                plt.show()
            
            cell_track_quads[track_ii] = quad_id
            
    return cell_track_quads


def count_cell_neighbors_time(cell_neighbors_time_list):

    import numpy as np 

    n_cells = len(cell_neighbors_time_list)
    n_time = len(cell_neighbors_time_list[0]) # number of timepoints to return

    out = np.zeros((n_cells, n_time))
    out[:] = np.nan # pre-initialisation

    for cell_ii in range(n_cells):
        for tt in range(n_time):
            neigh = cell_neighbors_time_list[cell_ii][tt]
            if len(neigh) > 0:
                out[cell_ii, tt] = len(np.squeeze(neigh))

    return out

def count_cell_neighbors_changes_time(cell_neighbors_time_list): 

    import numpy as np 

    n_cells = len(cell_neighbors_time_list)
    n_time = len(cell_neighbors_time_list[0]) # number of timepoints to return

    out = np.zeros((n_cells, n_time-1))
    out[:] = np.nan # pre-initialisation

    for cell_ii in range(n_cells):
        for tt in range(n_time-1):
            # since we are doing comparison
            neigh_tt = cell_neighbors_time_list[cell_ii][tt]
            neigh_tt1 = cell_neighbors_time_list[cell_ii][tt+1]

            # only if both are valid do we do this computation.
            if len(neigh_tt) > 0 and len(neigh_tt1) > 0:
                neigh_tt = np.squeeze(neigh_tt)
                neigh_tt1 = np.squeeze(neigh_tt1)
#                print(neigh_tt, neigh_tt1)
                num_ids_total = len(neigh_tt) + len(neigh_tt1)
                num_ids_same = len(np.intersect1d(neigh_tt, neigh_tt1))
                num_ids_union = num_ids_total - num_ids_same
                out[cell_ii, tt] = num_ids_union - num_ids_same

    return out


def lookup_3D_to_2D_directions(unwrap_params, ref_pos_2D, disp_3D, scale=10., use_3D=False):
#    
    # if use_3D then assume inputs are 3D. 
    
    disp_3D_ = disp_3D / float(np.linalg.norm(disp_3D) + 1e-8) # MUST normalise
    
    if use_3D:
        ref_pos_3D = ref_pos_2D.copy()
        
        # push this to a surface point.
        min_dists = np.linalg.norm(unwrap_params - ref_pos_3D[None,None,:], axis=-1)
        min_pos = np.argmin(min_dists);
        pos_2D = np.unravel_index(min_pos, unwrap_params.shape[:-1])
        ref_pos_3D = unwrap_params[pos_2D[0], pos_2D[1]].ravel() 
#        print(ref_pos_3D).shape
#        print(ref_pos_3D)
#        print(ref_pos_2D)
    else:
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
    if use_3D:
        direction_2D = np.hstack(new_pos_2D) - np.hstack(pos_2D)
    else:
        direction_2D = np.hstack(new_pos_2D) - np.hstack(ref_pos_2D)
    return direction_2D, np.arctan2(direction_2D[1], direction_2D[0])


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
    from mpl_toolkits.mplot3d import Axes3D 
    from MOSES.Motion_Analysis import tracks3D_tools as tra3Dtools
    import Geometry.meshtools as meshtools
    from sklearn.decomposition import PCA
    import warnings
    
    import skimage.io as skio
    
    
    warnings.filterwarnings("ignore")
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Set up a nearest neighbour model 
    from sklearn.neighbors import NearestNeighbors 
    nbrs_model = NearestNeighbors(n_neighbors=9, algorithm='auto') # how many neighbours to include? 9 = 8-connected?
    
    """
    Check this path ... 
    """
#    mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis' # this contains the precomputed reversed demons points? 
#    mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis-Paper/Components_3D_Analysis'
#    fio.mkdir(mastersavefolder)
    # mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper'
#    mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper-Full-Rec' # this is the full reconstruction. 
    mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper-Demons-Rec' # this is the full reconstruction. 
    
    masterannotfolder = '/media/felix/My Passport/Shankar-2/All_Manual_Annotations/all_cell_annotations' # for fetching all associated information regarding masks + cell tracks + cell divisions etc .... 
    mastercellstatsfolder = '/media/felix/My Passport/Shankar-2/All_Results/Single_Cell_Shape_Statistics_All/Data'

    """
    define some save folders ? 
    """


    """
    start of algorithm
    """
    # all_rootfolders = ['/media/felix/Srinivas4/LifeAct', 
    #                    '/media/felix/Srinivas4/MTMG-TTR',
    #                    '/media/felix/Srinivas4/MTMG-HEX',
    #                    '/media/felix/Srinivas4/MTMG-HEX/new_hex']
    all_rootfolders = ['/media/felix/Srinivas4/LifeAct']
#    rootfolder = '/media/felix/Srinivas4/LifeAct'
#    rootfolder = '/media/felix/Srinivas4/MTMG-TTR'
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX'
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
    
# =============================================================================
#     1. Load all embryos 
# =============================================================================
    all_embryo_folders = []
    
    for rootfolder in all_rootfolders:
        embryo_folders = np.hstack([os.path.join(rootfolder,folder) for folder in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, folder)) and 'new_hex' not in folder])
        all_embryo_folders.append(embryo_folders)
        
    all_embryo_folders = np.hstack(all_embryo_folders)
    
    
# =============================================================================
#     2. find valid embryo folders i.e. has a geodesic-rotmatrix. 
# =============================================================================
    n_spixels = 5000 # set the number of pixels. 
    smoothwinsize = 3
#    n_spixels = 5000
    
    manual_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/manual_staging_holly_matt.xlsx'
    manual_staging_tab_Matt_Holly = pd.read_excel(manual_staging_file)
    
    # give the autostaging files. 
#    manual_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-LifeAct-smooth-lam10-correct_norot.xlsx' 
#    manual_staging_tab = pd.read_excel(manual_staging_file)
    
    """
    Load alll. 
    """
    # all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-LifeAct-smooth-lam5-correct_norot_consensus.xlsx', 
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-TTR-smooth-lam5-correct_norot_consensus.xlsx',
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-HEX-smooth-lam5-correct_norot_consensus.xlsx',
    #                      '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-new-HEX-smooth-lam5-correct_norot_consensus.xlsx']
    all_staging_files = ['/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-LifeAct-smooth-lam5-correct_norot_consensus.xlsx'] 
    
    manual_staging_tab = pd.concat([pd.read_excel(staging_file) for staging_file in all_staging_files], ignore_index=True)
    all_stages = manual_staging_tab.columns[1:4]

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
    
    
    VE_analysis_mapping = []
    Epi_analysis_mapping = []
    
    VE_analysis_mapping_grid_pre = []
    VE_analysis_mapping_grid_mig  = []
    VE_analysis_mapping_grid_post  = []
    
    Epi_analysis_mapping_grid_pre = []
    Epi_analysis_mapping_grid_mig = []
    Epi_analysis_mapping_grid_post = []
#    
#    all_angle_files = ['/media/felix/Srinivas4/correct_angle-global-1000_LifeAct.csv']
##                       '/media/felix/Srinivas4/correct_angle-global-1000_MTMG-TTR.csv',
##                       '/media/felix/Srinivas4/correct_angle-global-1000_MTMG-HEX.csv',
##                       '/media/felix/Srinivas4/correct_angle-global-1000_new_hex.csv']
#    
#    angle_tab = pd.concat([pd.read_csv(angle_file) for angle_file in all_angle_files], ignore_index=True)
    
    # all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/LifeAct_polar_angles_AVE_classifier-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-TTR_polar_angles_AVE_classifier-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/MTMG-HEX_polar_angles_AVE_classifier-consensus.csv',
    #                    '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/new_hex_polar_angles_AVE_classifier-consensus.csv']
    all_angle_files = ['/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean/LifeAct_polar_angles_AVE_classifier-consensus.csv']
    angle_tab = pd.concat([pd.read_csv(angle_file) for angle_file in all_angle_files], ignore_index=True)
#  
    
#    for embryo_folder in embryo_folders[-2:-1]:
#    for embryo_folder in all_embryo_folders[:1]:
    for embryo_folder in all_embryo_folders[2:]: #giving the L491. 
#    for embryo_folder in embryo_folders[3:4]:
        # find the geodesic-rotmatrix. 
        unwrapped_folder = locate_unwrapped_im_folder(embryo_folder, key='geodesic-rotmatrix')
        
        saveanalysisfolder = os.path.join(mastersavefolder, os.path.split(embryo_folder)[-1]);
#        fio.mkdir(saveanalysisfolder)
        
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
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-5000_VE_seg_directions')
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-curvilinear_Tcorrected')
            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels)) 
            saveboundaryfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow-epi-boundary')
            
            unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))
        
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
            
            
            if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
            
            else:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_hex_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_hex_embryo(unwrap_im_files, unwrap_meta_files)

            
            # iterate over the pairs 
            for ii in tqdm(range(len(paired_unwrap_file_condition))[:]):
                
                
                """
                Only process if polar. 
                """
                unwrapped_condition = paired_unwrap_file_condition[ii]
                print(unwrapped_condition)
                
                if unwrapped_condition.split('-')[-1] == 'polar':
                    print(unwrapped_condition)
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
                    
                    vid_size = vid_ve[0].shape
                    
                    if 'MTMG-HEX' in embryo_folder:
                        vid_hex = fio.read_multiimg_PIL(hex_unwrap_file)
                        
                # =============================================================================
                #       Load the relevant saved pair track file.               
                # =============================================================================
                    trackfile = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')
                    
                    meantracks_ve = spio.loadmat(trackfile)['meantracks_ve']
                    meantracks_epi = spio.loadmat(trackfile)['meantracks_epi']
                    proj_condition = spio.loadmat(trackfile)['condition'][0].split('-')
                    proj_type = proj_condition[-1]
                    
                
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
                    
                    table_emb_names = np.hstack([str(s) for s in manual_staging_tab['Embryo'].values])
                    select_stage_row = manual_staging_tab.loc[table_emb_names==embryo_name]
                    
                    pre = parse_start_end_times(select_stage_row['Pre-migration'])[0]
                    mig = parse_start_end_times(select_stage_row['Migration to Boundary'])[0]
                    post = parse_start_end_times(select_stage_row['Post-Boundary'])[0]
                    
                    pre_all.append(pre)
                    mig_all.append(mig)
                    post_all.append(post)
                    print(pre, mig, post)
                    
                    
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
                #       Load the 3D demons mapping file coordinates.        
                # =============================================================================
                    # load the associated unwrap param files depending on polar/rect coordinates transform. 
                    unwrap_param_file_ve = ve_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
                    unwrap_params_ve = spio.loadmat(unwrap_param_file_ve); unwrap_params_ve = unwrap_params_ve['ref_map_%s_xyz' %(proj_type)]
#                    unwrap_params_ve_polar = spio.loadmat(unwrap_param_file_ve); unwrap_params_ve_polar = unwrap_params_ve_polar['ref_map_'+'polar'+'_xyz']
                    
                    if proj_type == 'rect':
                        unwrap_params_ve = unwrap_params_ve[::-1] # reverse ....
    #                    unwrap_params_epi = unwrap_params_epi[::-1]
                        
                    unwrap_param_file_epi = epi_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
                    unwrap_params_epi = spio.loadmat(unwrap_param_file_epi); unwrap_params_epi = unwrap_params_epi['ref_map_%s_xyz' %(proj_type)]
#                    unwrap_params_epi_polar = spio.loadmat(unwrap_param_file_ve); unwrap_params_epi_polar = unwrap_params_epi_polar['ref_map_'+'polar'+'_xyz']
                    
                    
                # =============================================================================
                #       Load the remapped points.                 
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
                    ve_angle_move = 90 - ve_angle_move.values[0] # to get back to the rectangular angle coordinate? 
                    # ve_angle = ve_angle_move.values[0]
                    print('ve movement direction: ', ve_angle_move)
                    
                    # the consenus angle is used for rotation/?
                    
                # =============================================================================
                #       Load the computed statistics of the real geometry       
                # =============================================================================
     
                    saveprojfile = os.path.join(saveanalysisfolder, 'demons_remapped_pts-'+unwrapped_condition+'.mat')
                    print(saveprojfile)
                    stats_obj = spio.loadmat(saveprojfile)
                    
                    
                    """
                    Get the required statistics, 
                    the remapped pts is the actual (x,y,z) in the same reference of polar. 
                    """
                    demons_times = stats_obj['demons_times']
                    
                    ve_pts = stats_obj['ve_pts']
                    epi_pts = stats_obj['epi_pts']
                    epi_pts = sktform.resize(epi_pts, ve_pts.shape, preserve_range=True)
                    
#                    ve_angle_vector = savedirectionobj['ve_angle_vector'].ravel()
                    # these are the respective in the r + normal directions. 
                    surf_radial_ve = stats_obj['surf_radial_ve']
                    surf_theta_ve = stats_obj['surf_theta_ve']
                    surf_normal_ve = stats_obj['surf_normal_ve']
                    
                    surf_radial_epi = stats_obj['surf_radial_epi']
                    surf_theta_epi = stats_obj['surf_theta_epi']
                    surf_normal_epi = stats_obj['surf_normal_epi'] 
                
                
                    test_time = 0 
                    fig = plt.figure(figsize=(10,10))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_aspect('equal')

                    # we need to revert these properly !.     
                    ax.scatter(ve_pts[test_time, ::40 ,::40, 0], 
                               ve_pts[test_time,::40 ,::40, 2],
                               ve_pts[test_time,::40 ,::40, 1], c='g')
                    
                    # plot the quiver arrows. 
                    ax.quiver(ve_pts[test_time,::40 ,::40, 0], 
                               ve_pts[test_time,::40 ,::40, 2],
                               ve_pts[test_time,::40 ,::40, 1], 
                               surf_radial_ve[test_time,::40, ::40, 0],
                               surf_radial_ve[test_time,::40, ::40, 2], 
                               surf_radial_ve[test_time,::40, ::40, 1], length=50., 
                               normalize=True, color='r')
                    
                    ax.quiver(ve_pts[test_time,::40 ,::40, 0], 
                               ve_pts[test_time,::40 ,::40, 2],
                               ve_pts[test_time,::40 ,::40, 1], 
                               surf_theta_ve[test_time,::40, ::40, 0],
                               surf_theta_ve[test_time,::40, ::40, 2], 
                               surf_theta_ve[test_time,::40, ::40, 1], length=50., 
                               normalize=True, color='b')
                    
                    tra3Dtools.set_axes_equal(ax)
                    plt.show()
                # =============================================================================
                #         Load the revelant boundary contour line.        
                # =============================================================================
                    
                    saveboundaryfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow-epi-boundary')
                        
                    epi_boundary_file = os.path.join(saveboundaryfolder,unwrapped_condition,'inferred_epi_boundary-'+unwrapped_condition+'.mat')
                    epi_boundary_time = spio.loadmat(epi_boundary_file)['contour_line']
                    
#                    epi_contour_line_pre = epi_boundary_time[0]
#                    epi_contour_line_mig = epi_boundary_time[mig_tp_pt]
                
                
                # =============================================================================
                #         Attempt to load the corresponding cell tracks.         
                # =============================================================================
                    celltracks_folder = os.path.join(masterannotfolder, unwrapped_condition)
                    
                    if os.path.exists(celltracks_folder):

                        #######
                        # use keyword search to locate the tracks of relevance. 
                        #######
                        # celltracks_file = os.path.join(celltracks_folder, 
                                                       # 'cell_tracks_w_ids_Holly_'+ unwrapped_condition +'_distal_corrected_nn_tracking-full-dist30.mat') # this file will include the cell ids. # and the centroid information.
                        celltracks_file = glob.glob(os.path.join(celltracks_folder, '*.mat')) # the equivalent .csv is for Jason's program. 
                        celltracks_file = [ff for ff in celltracks_file if 'cell_tracks_' in os.path.split(ff)[-1]]

                        # cell img segmentation 
                        cell_file = glob.glob(os.path.join(celltracks_folder, '*.tif'))
                        cell_file = [ff for ff in cell_file if 'cells_' in os.path.split(ff)[-1]]

                        # also get the cell division file. 
                        csvfiles = glob.glob(os.path.join(celltracks_folder, '*.csv'))
                        divfile = [ csvfile for csvfile in csvfiles if 'div_polar' in os.path.split(csvfile)[-1]]

                        # needs both files. 
                        if len(cell_file) > 0 and len(celltracks_file) > 0: 
                            # proceed only if we have this segmentation mask file. 
                            cell_file = cell_file[0] 
                            celltracks_file = celltracks_file[0]
                            celldivfile = divfile[0]

                            """
                            read in the data. so we can do embryo inversion on the data for final quantification. 
                            """
                            celltracks_ids_obj = spio.loadmat(celltracks_file) # this is unaffected by inversion.... 
                            cell_seg_img = skio.imread(cell_file)
                            cell_div_table = pd.read_csv(celldivfile) # load all the tabular data to help us determine the mother-daughter tracks.  
                            
                            # note the ids should not need to be 'invereted in any manner long as the underlying cell images change.
                            
                            try:
                                cell_tracks_cell_ids = celltracks_ids_obj['cell_tracks_ids']; # this is in terms of the superpixels? 
                            except:
                                cell_tracks_cell_ids = celltracks_ids_obj['cell_track_cell_ids_time']
                            cell_tracks_cell_xy = celltracks_ids_obj['cell_tracks_xy']
                        
    #                        embryo_inversion = True -> testing only. 
                        # =============================================================================
                        #       apply the embryo inversion to the inputs...               
                        # =============================================================================
                            if embryo_inversion == True:
                                if proj_type == 'rect':
                                    unwrap_params_ve = unwrap_params_ve[:,::-1] # reverse x direction 
                                    unwrap_params_epi = unwrap_params_epi[:,::-1]
        
                                    # properly invert the 3D. 
                                    unwrap_params_ve[...,2] = embryo_im_shape[2] - unwrap_params_ve[...,2]
                                    unwrap_params_epi[...,2] = embryo_im_shape[2] - unwrap_params_epi[...,2]
                                    
                                    ve_pts = ve_pts[:,:,::-1]; ve_pts[...,2] = embryo_im_shape[2] - ve_pts[...,2] # this is temporal. 
                                    epi_pts = epi_pts[:,:,::-1]; epi_pts[...,2] = embryo_im_shape[2] - epi_pts[...,2]
                                    
                                    
        #                            # i don't think this needs changing.... since this is really definedd on the image? -> worst case is recompute. 
        #                            surf_radial_ve = surf_radial_ve['surf_radial_ve']
        #                            surf_theta_ve = surf_theta_ve['surf_theta_ve']
        #                            surf_normal_ve = surf_normal_ve['surf_normal_ve']
        #                            
        #                            surf_radial_epi = surf_radial_epi['surf_radial_epi']
        #                            surf_theta_epi = surf_theta_epi['surf_theta_epi']
        #                            surf_normal_epi = surf_normal_epi['surf_normal_epi'] 
                                    
                                    vid_ve = vid_ve[:,:,::-1]
                                    vid_epi = vid_epi[:,:,::-1]
                                    vid_epi_resize = vid_epi_resize[:,:,::-1]
        
                                    # check.... do we need to invert the epi boundary too !. 
                                    epi_boundary_time[...,0] = (vid_ve[0].shape[1]-1) - epi_boundary_time[...,0] # correct x-coordinate.  
                                    
                                    # apply corrections. 
                                    cell_seg_img = cell_seg_img[:,:,::-1] # flipping the x-axis.
                                    cell_tracks_cell_xy[...,0] = (vid_ve[0].shape[1]-1) - cell_tracks_cell_xy[...,0]  # flip the x coordinate. 
                                    
                                    # apply corrections to the xy of the mother, daughter. 
                                    cell_div_table['mum_x'] = (vid_ve[0].shape[1]-1) - cell_div_table['mum_x']
                                    cell_div_table['daug1_x'] = (vid_ve[0].shape[1]-1) - cell_div_table['daug1_x']
                                    cell_div_table['daug2_x'] = (vid_ve[0].shape[1]-1) - cell_div_table['daug2_x']
                                    

                                    if 'MTMG-HEX' in embryo_folder:
                                        vid_hex = vid_hex[:,:,::-1]
          
                                else:
        
                                    unwrap_params_ve = unwrap_params_ve[::-1] # reverse y direction
                                    unwrap_params_epi = unwrap_params_epi[::-1]
                                    
                                    # properly invert the 3D. 
                                    unwrap_params_ve[...,2] = embryo_im_shape[2] - unwrap_params_ve[...,2]
                                    unwrap_params_epi[...,2] = embryo_im_shape[2] - unwrap_params_epi[...,2]
        
                                    # invert the coordinates
                                    ve_pts = ve_pts[:,::-1]; ve_pts[...,2] = embryo_im_shape[2] - ve_pts[...,2] # this is temporal. 
                                    epi_pts = epi_pts[:,::-1]; epi_pts[...,2] = embryo_im_shape[2] - epi_pts[...,2]
                                    
                                    vid_ve = vid_ve[:,::-1]
                                    vid_epi = vid_epi[:,::-1]
                                    vid_epi_resize = vid_epi_resize[:,::-1]
        
                                    epi_boundary_time[...,1] = (vid_ve[0].shape[0]-1) - epi_boundary_time[...,1] # correct the y-coordinate 
                                    
                                    cell_seg_img = cell_seg_img[:,::-1] # flipping the y-axis.
                                    cell_tracks_cell_xy[...,1] = (vid_ve[0].shape[0]-1) - cell_tracks_cell_xy[...,1]  # flip the y coordinate. 
                                    
                                    cell_div_table['mum_y'] = (vid_ve[0].shape[0]-1) - cell_div_table['mum_y']
                                    cell_div_table['daug1_y'] = (vid_ve[0].shape[0]-1) - cell_div_table['daug1_y']
                                    cell_div_table['daug2_y'] = (vid_ve[0].shape[0]-1) - cell_div_table['daug2_y']

                                    if 'MTMG-HEX' in embryo_folder:
                                        vid_hex = vid_hex[:,::-1]
 
                            """
                            plot check the tracks have been parsed and inverted correctly. ?  
                            """
                            cell_tracks_cell_xy[cell_tracks_cell_xy<.5] = np.nan # this is implemented as (0,0) is given for nan in tracks data. 
                            
                            print(cell_tracks_cell_xy.shape)

                            # visualise the tracks as a check.... 
                            fig, ax = plt.subplots(figsize=(10,10))
                            ax.imshow(vid_ve[0], cmap='gray')
                            for cc_track in cell_tracks_cell_xy:
                                ax.plot(cc_track[...,0], 
                                        cc_track[...,1], '-')
                            plt.show()

                            # don't need this below. 
                        #     # =============================================================================
                        #     #     Create the find to coarse ids for spatial averaging and for transforming the fine grid to the coarse grid for statistics.  
                        #     # =============================================================================
                        #     supersample_theta = 4 # 8 angle divisions
                        #     supersample_r = 4  # 4 distance divisions. 

                        #     theta_spacing = int(supersample_theta)+1
                        #     r_spacing = int(supersample_r)
                            
                        #     polar_grid_fine_2_coarse_ids = [] 
                            
                        #     for jj in range(4):
                        #         for ii in range(8):
                                    
                        #             start_ind = jj*r_spacing*theta_spacing*8
                        #             # this builds the inner ring. 
                        #             ind = np.mod(np.arange(-(theta_spacing//2), (theta_spacing//2)+1) + ii*theta_spacing, theta_spacing*8)
                                    
                        #             ind = np.hstack([ind+kk*theta_spacing*8 for kk in range(r_spacing)])
                        # #            print(ind)
                        #             all_quad_ind = ind + start_ind

                        #             polar_grid_fine_2_coarse_ids.append(all_quad_ind)
                                    
                        #     # we use this to index into the array to get averages for each respective region.
                        #     polar_grid_fine_2_coarse_ids = np.array(polar_grid_fine_2_coarse_ids) 


                            # =============================================================================
                            #   Use the cell divisions + cell_tracks_cell_ids to build up the mother_daughter track associations. 
                            # =============================================================================
                            
                            # use the first gen to build up any 2nd gen relationships amongst the tracks.
                            div_times =  cell_div_table['t_stamp'].values
                            div_mother_xy = np.vstack([ cell_div_table['mum_x'].values,
                                                        cell_div_table['mum_y'].values]).T
                            div_daugh1_xy = np.vstack([ cell_div_table['daug1_x'].values,
                                                        cell_div_table['daug1_y'].values]).T
                            div_daugh2_xy = np.vstack([ cell_div_table['daug2_x'].values,
                                                        cell_div_table['daug2_y'].values]).T

                            # to build this we lookup the cell ids from the cell mask, absolutely none of these should have 0 for an id which is reserved for background. 
                            mother_ids = cell_seg_img[div_times.astype(np.int), 
                                                      div_mother_xy[:,1].astype(np.int),
                                                      div_mother_xy[:,0].astype(np.int)] 
                            cell_1_ids = cell_seg_img[div_times.astype(np.int)+1, 
                                                      div_daugh1_xy[:,1].astype(np.int),
                                                      div_daugh1_xy[:,0].astype(np.int)]                             
                            cell_2_ids = cell_seg_img[div_times.astype(np.int)+1, 
                                                      div_daugh2_xy[:,1].astype(np.int),
                                                      div_daugh2_xy[:,0].astype(np.int)] 


                            # search for mother, daughter pairings from the tracks. ?
                            mother_daughter_tra_id_pairings = cell_tracks_mother_daughter_ids( cell_tracks_cell_ids, 
                                                                                               div_times, 
                                                                                               mother_ids, 
                                                                                               cell_1_ids, 
                                                                                               cell_2_ids,
                                                                                               max_time = len(vid_ve))

                            # refine the connected divtables etc 
                            keep_pairings = np.hstack([len(pp)==3 for pp in mother_daughter_tra_id_pairings])
                            mother_daughter_tra_id_pairings = [mother_daughter_tra_id_pairings[iii] for iii in range(len(mother_daughter_tra_id_pairings)) if keep_pairings[iii]==True]
                                                    
                            # so the order of this is mother, daugh1, daugh2 
                            # what we want to do is find lineage trees? # cliques?
                            # strategy = clique? then reorganise? into tree?  
                            
                            # =============================================================================
                            #       Load up the pre-computed single cell statistics. 
                            # =============================================================================     
                            savesinglestatsmatfile = os.path.join(mastercellstatsfolder, 'cell_stats_'+unwrapped_condition+'.mat')
                            singlestatsobj = spio.loadmat(savesinglestatsmatfile) # load this up. 

                            # load all the possible saved statistics of interest so we can ascribe them to a track and have them available as arrays in the same format as the tracks. :D
                            cell_times = np.squeeze(singlestatsobj['time'])
                            cell_stages = np.squeeze(singlestatsobj['stage']); cell_stages = np.hstack([ss.strip() for ss in cell_stages])
                            cell_ve_angle = singlestatsobj['ve_angle'].ravel()[0]

                            cell_ids = np.squeeze(singlestatsobj['cell_ids'])
                            cell_areas = np.squeeze(singlestatsobj['cell_areas'])
                            cell_perims = np.squeeze(singlestatsobj['cell_perims'])
                            cell_shape_index = np.squeeze(singlestatsobj['cell_shape_index'])
                            cell_eccentricity = np.squeeze(singlestatsobj['cell_eccentricity'])
                            
                            # note these are not normalised!. 
                            cell_major_axis_2D = np.squeeze(singlestatsobj['cell_major_axis_2D']) # this is used to compute an orientation (already has inversion, need rotation correction!)
                            cell_minor_axis_2D = np.squeeze(singlestatsobj['cell_minor_axis_2D'])
                            cell_centroids_2D = np.squeeze(singlestatsobj['cell_centroids_2D']) # for plotting again. 
                            cell_centroids_3D = np.squeeze(singlestatsobj['cell_centroids_3D']) # this is using the full pullback which we don't need per say (instead if we want the equivalent of MOSES, we do the unwrap_params.)
                            polar_grids_coarse = singlestatsobj['polar_grids_coarse'] # this is the key grid for averaging everything over. 
                            
                            # this tallies the cell ids that are neighbors. -> for tracks we need to convert this... to track ids!
                            cell_neighbors = np.squeeze(singlestatsobj['cell_neighbors']) # this is just counting how many neighbors. and for figuring out neighbor exchange over the cell track. 
            
                            # =============================================================================
                            #       a) Get the dynamic tracks on the unwrap_params surface. -> used for computing the instantaneous velocities?  
                            #           -> remove the inconsistency of the centroid ? -> # keep a fixed point on the surface fixed. 
                            # =============================================================================                   
                            # load the single shape statistics file that has been computed. 
                            savetrackstatsmatfile = os.path.join(mastercellstatsfolder, 'cell_track_and_polargrid_stats_w_lineage_and_density_'+unwrapped_condition+'.mat')
                            savetrackstatsmatobj = spio.loadmat(savetrackstatsmatfile)

                            cell_track_centroids_2D = savetrackstatsmatobj['cell_track_centroids_2D']
                            cell_track_centroids_3D_real = savetrackstatsmatobj['cell_track_centroids_3D']
                            cell_track_quad_ids_stage = savetrackstatsmatobj['cell_track_quad_ids_stage']
                            analyse_stages = savetrackstatsmatobj['analyse_stages']; analyse_stages = np.hstack([s.strip() for s in analyse_stages])
                            analyse_time_segs = savetrackstatsmatobj['analyse_time_segs']
                            if analyse_time_segs.shape[0] == len(analyse_stages):
                                pass
                            else:
                                analyse_time_segs = analyse_time_segs[0]


                            """
                            Setup the 3D centroids positions so we can derive proper velocities from them, at all time points for the tracks. 
                            Construct the proper 3D centroid positions
                            """

                            # do this for both the centroids 3D (single cells) and for the tracks? 
                            distal_center = (np.hstack(ve_pts[0].shape[:2])//2).astype(np.int)
                            distal_3D_pts_ve = ve_pts[:,distal_center[0], distal_center[1]].copy()

                            # i think this is still the correct normalization. 
                            cell_centroids_3D_rel = np.array([cell_centroids_3D[tt] - distal_3D_pts_ve[tt][None,:] for tt in np.arange(len(cell_centroids_3D))]) # -> this is to make sure all the distal point = 0. -> check this!. 
                            cell_track_centroids_3D_real_rel = cell_track_centroids_3D_real - distal_3D_pts_ve[None,:cell_track_centroids_3D_real.shape[1]]

                            # this is good. 
#                            ve_pts_rel = ve_pts - distal_3D_pts_ve[:,None,None,:]
                            ve_pts_rel = ve_pts #- distal_3D_pts_ve[:,None,None,:]
#                             # this is for the movement analysis. 
#                             cell_track_centroids_3D_ref = np.zeros((cell_track_centroids_2D.shape[0], cell_track_centroids_2D.shape[1], 3))
#                             cell_track_centroids_3D_ref[:] = np.nan; non_nan_select = ~np.isnan(cell_track_centroids_2D[...,0])
#                             cell_track_centroids_3D_ref[non_nan_select] = unwrap_params_ve[cell_track_centroids_2D[non_nan_select][...,0].astype(np.int),
#                                                                                            cell_track_centroids_2D[non_nan_select][...,1].astype(np.int)] 

                            # =============================================================================
                            #       b) Prepare the prerequisites to do the movement analysis..... 
                            # =============================================================================                   

                            # create a surf_nbrs lookup model for querying the surface.
                            # construct the ve_angle_vector in real space to help get the local AVE direction 
                            ve_angle_vector = np.hstack([np.sin((90+ve_angle_move)/180. * np.pi), 
                                                          np.cos((90+ve_angle_move)/180. * np.pi)])        
                            print('AVE angle vector 2D', ve_angle_vector)


                            # would have to create this ... for every time point which might be very very slow? compared to just iteratively? 

                            AVE_surf_dir_ve_time = []

#                            for tt in tqdm(np.arange(cell_track_centroids_3D_real_rel.shape[1])):
                            for tt in tqdm(np.arange(len(cell_centroids_2D))):

                                # step 1: build reusable neighbor model. # only for the VE
                                surf_nbrs_ve, surf_nbrs_ve_3D_pts = tra3Dtools.learn_nearest_neighbor_surf_embedding_model(ve_pts_rel[tt], 
                                                                                                                               nearest_K=1)

                                """
                                we need to compute the AVE direction with respect to the surface at every time point using the ve_pts at this point. 
                                """
                                # step 2: build the surface AVE directional vectors in a track format. # more complicated due to nan....
                                AVE_surf_dir_ve = tra3Dtools.proj_2D_direction_3D_surface(ve_pts_rel[tt], 
                                           
                                                                                          ve_angle_vector[::-1][None,None,:], 
                                                                                            pt_array_time_2D = cell_track_centroids_2D[:,None,tt],
                                                                                            K_neighbors=1, 
                                                                                            nbr_model=surf_nbrs_ve, 
                                                                                            nbr_model_pts=surf_nbrs_ve_3D_pts, 
                                                                                            dist_sample=10, 
                                                                                            return_dist_model=False,
                                                                                            mode='sparse')

                                AVE_surf_dir_ve_time.append(AVE_surf_dir_ve[:,0]) # remove the size 1 dimension. 

                            AVE_surf_dir_ve_time = np.array(AVE_surf_dir_ve_time)
                            
                            
                            """
                            Construct the saving which will be very simple, just the 3D positions and the computed AVE surf dir ve .... with the real coordinates. 
                            """
                            vol_factor = np.product(embryo_im_shape) ** (1./3)
                            
                            savespeedstatsmatfile = os.path.join(mastercellstatsfolder, 'cell_track_velocity_stats_demons_xyz_'+unwrapped_condition+'.mat')
                            
                            spio.savemat(savespeedstatsmatfile, 
                                         {'cell_times':cell_times,
                                          'cell_centroids_2D': cell_centroids_2D, 
                                          'cell_centroids_3D': cell_centroids_3D,
                                          'cell_centroids_3D_rel': cell_centroids_3D_rel,
                                          'distal_center': distal_center,
                                          'distal_3D_pts_ve':distal_3D_pts_ve,
                                          'cell_track_centroids_3D_real_rel': cell_track_centroids_3D_real_rel,
                                          'temp_tform_scales': temp_tform_scales,
                                          'AVE_surf_dir_ve_time_3D': AVE_surf_dir_ve_time,
#                                          'cell_track_3D_xyz_ref': cell_track_centroids_3D_ref,
#                                          'cell_track_AVE_dir_3D': AVE_surf_dir_ve,
                                          'analyse_stages':analyse_stages,
                                          'analyse_time_segs':analyse_time_segs,
                                          'cell_track_quad_ids_stage':cell_track_quad_ids_stage,
                                          've_angle':ve_angle_move, 
                                          've_angle_vector_2D':ve_angle_vector,
                                          'vol_factor':vol_factor,
                                          'embryo_shape': embryo_im_shape}) 
#                                          'cell_track_ids_stage': analysed_cell_tracks_all, 
#                                          'disps_ve_3D_all_norm_stage': disps_ve_3D_all, 
#                                          'AVE_dir_ve_3D_stage': anterior_dirs_3D_all, 
#                                          'region_ids': uniq_polar_regions, 
#                                          'region_lin_velocity': lin_velocity_regions_ve_all, 
#                                          'region_mean_curve_velocity': curvilinear_velocity_regions_ve_all, 
#                                          'region_mean_AVE_velocity': disps_ve_3D_component_reg_all, 
#                                          'region_mean_lin_speed': mean_lin_speed_regions_ve_all, 
#                                          'region_mean_curve_speed': mean_curvilinear_speed_regions_ve_all,
#                                          'region_mean_total_speed': mean_total_speed_regions_ve_all, 
#                                          'region_mean_dir_3D': mean_direction_regions_ve_3D_all, 
#                                          'region_mean_dir_2D': mean_direction_regions_ve_2D_all, 
#                                          'region_mean_AVE_speed': mean_speed_regions_ve_AVE_dir_all})

#                             # =============================================================================
#                             #       c) Main movement analysis scripts. (cell speed along the track!)
#                             # =============================================================================      

#                             vol_factor = np.product(embryo_im_shape) ** (1./3)

#                             # save all the outputs here ready to be saved finally. 

#                             # do want to also save the temporal information here.
#                             analysed_cell_tracks_all = [] # give the ids of all tracks we analyse in this stage. 
#                             disps_ve_3D_all = [] # give the temporal displacements here?
#                             anterior_dirs_3D_all = []

#                             uniq_polar_regions = np.setdiff1d(np.unique(polar_grids_coarse[0]), 0)

#                             lin_velocity_regions_ve_all = []
#                             curvilinear_velocity_regions_ve_all = []
#                             disps_ve_3D_component_reg_all = []

#                             mean_lin_speed_regions_ve_all = []
#                             mean_curvilinear_speed_regions_ve_all = []
#                             mean_total_speed_regions_ve_all = []
#                             mean_direction_regions_ve_3D_all = []
#                             mean_direction_regions_ve_2D_all = []
#                             mean_speed_regions_ve_AVE_dir_all = []

#                             for stage_ii in np.arange(len(analyse_stages))[:]:

#                                 if len(analyse_time_segs[stage_ii].shape) > 1:
#                                     analyse_time_stage_ii = analyse_time_segs[stage_ii][0] # get the flattened times. 
#                                 else:
#                                     analyse_time_stage_ii = analyse_time_segs[stage_ii]
#                                 cell_track_centroids_3D_ref_stage_ii = cell_track_centroids_3D_ref[:,analyse_time_stage_ii]

#                                 polar_grid_stage_ii = polar_grids_coarse[analyse_time_stage_ii[0]]
#                                 """
#                                 limit analysis to select tracks -> tracks that are in this stage.  
#                                 """
#                                 cell_track_quad_stage_ii = cell_track_quad_ids_stage[stage_ii] # this is assigned to the track based on its 'starting time' in the current stage, and we vary the grid over time. 
#                                 select_cells = cell_track_quad_stage_ii > 0  # this select_cells currently includes all cells which appear later on the dynamic MOSES grid -> NO. 

#                                 select_cells_start_stage = cell_tracks_cell_ids[:,analyse_time_stage_ii[0]] > 0 # i.e. enforce valid cell must start at first time point of the stage!. 
#                                 select_cells = np.logical_and(select_cells, select_cells_start_stage) # then we are only taking statistic of tracks that start in this time.? 

#                                 analysed_cell_tracks_all.append(select_cells) # just give this as boolean. 

#                                 # for directionality computation + linear speed directions in 3D. 
#                                 disps_ve_3D = cell_track_centroids_3D_ref_stage_ii[:,1:] - cell_track_centroids_3D_ref_stage_ii[:,:-1]; 
#                                 disps_ve_3D = disps_ve_3D.astype(np.float32)
#                                 disps_ve_3D = disps_ve_3D[select_cells]
                                
#                                 # normalisation of scale.
#                                 disps_ve_3D = disps_ve_3D / float(vol_factor) * temp_tform_scales[analyse_time_stage_ii[0]+1:analyse_time_stage_ii[-1]+1][None,:,None] # normalised ... 
    

#                                 disps_ve_3D_all.append(disps_ve_3D)

#                                 # get the anterior directions for this stage. 
#                                 anterior_directions_3D_tracks_ve = AVE_surf_dir_ve[:,analyse_time_stage_ii].copy()
#                                 anterior_directions_3D_tracks_ve = anterior_directions_3D_tracks_ve[select_cells]

#                                 anterior_dirs_3D_all.append(anterior_directions_3D_tracks_ve)
                                
#                                 """
#                                 get just the anterior components
#                                 """
#                                 # project to anterior direction over time. 
#                                 disps_ve_3D_component = np.nansum(disps_ve_3D * anterior_directions_3D_tracks_ve[:,:-1], axis=-1)
                            
#                                 print(disps_ve_3D_component.shape)

#                                 """
#                                 compute the curvilinear movement in the mean track direction, since this requires a time segment needs to be done over the given time segment.... hm.... 
#                                 """
#                                 # check this handles np.nans. => seems to work? 
#                                 polar_ve_tracks_3D_diff, (polar_ve_surf_dist, polar_ve_surf_lines), (polar_ve_tracks_3D_diff_mean, polar_ve_tracks_3D_diff_mean_curvilinear) = tra3Dtools.compute_curvilinear_3D_disps_tracks(cell_track_centroids_2D[select_cells][:,analyse_time_stage_ii], 
#                                                                                                                                                                                                                               unwrap_params_ve, 
#                                                                                                                                                                                                                               n_samples=20, 
#                                                                                                                                                                                                                               nearestK=1, 
#                                                                                                                                                                                                                               temporal_weights=temp_tform_scales[analyse_time_stage_ii[0]+1:analyse_time_stage_ii[-1]+1],
#                                                                                                                                                                                                                               cell_tracks_mode=True)

#                                    #######
#                                    # Collect all the statistics. 
#                                    #######
#                                 """
#                                 Metric 1: compute the total speed (distance moved over time, this is curvilinear due to small displacement assumption)
#                                 """
#                                 # primary statistics comparison over the grid regions. 
#                                 select_cells_quad = cell_track_quad_stage_ii[select_cells] # we use this to tabulate all the statistics over, very much like we do with the MOSES tracks. 

#                                 ### a) linear speed of interest. 
#                                 lin_velocity_regions_ve = [disps_ve_3D[select_cells_quad==reg] for reg in uniq_polar_regions] # bin the measurements into the different regions of the grid, preserving time. 
#                                 lin_velocity_regions_ve_all.append(lin_velocity_regions_ve) # temporal variation. 

#                                 # get the statistics over the regions of interest.  
#                                 mean_lin_speed_regions_ve = []
#                                 for speed in lin_velocity_regions_ve:
#                                     if len(speed)>0:
#                                         mean_lin_speed_regions_ve.append(np.linalg.norm(np.nanmedian(np.nanmean(speed, axis=1), axis=0)))
#                                     else:
#                                         mean_lin_speed_regions_ve.append(np.nan)

#                                 mean_lin_speed_regions_ve = np.hstack(mean_lin_speed_regions_ve)


#                                 ### b) curvilinear speed of interest.
#                                 curvilinear_velocity_regions_ve = [polar_ve_tracks_3D_diff_mean_curvilinear[select_cells_quad==reg] for reg in uniq_polar_regions] # this is not by time. 
#                                 curvilinear_velocity_regions_ve_all.append(curvilinear_velocity_regions_ve)

#                                 mean_curvilinear_speed_regions_ve = []

#                                 for speed in curvilinear_velocity_regions_ve:
#                                     if len(speed)>0:
#                                         mean_curvilinear_speed_regions_ve.append(np.nanmedian(np.linalg.norm(speed, axis=-1))/float(vol_factor))
#                                     else:
#                                         mean_curvilinear_speed_regions_ve.append(np.nan)

#                                 mean_curvilinear_speed_regions_ve = np.hstack(mean_curvilinear_speed_regions_ve)


#                                 ### c) total speed of interest.
#                                 mean_total_speed_regions_ve = [] 

#                                 for speed in lin_velocity_regions_ve:
#                                     if len(speed)>0:
#                                         mean_total_speed_regions_ve.append(np.nanmedian(np.nanmean(np.linalg.norm(speed, axis=-1), axis=1)))
#                                     else:
#                                         mean_total_speed_regions_ve.append(np.nan)

#                                 mean_total_speed_regions_ve = np.hstack(mean_total_speed_regions_ve)


#                                 ### d) the mean direction as mapped to 2D. ( averaging in 3D then reprojecting in 2D.)
#                                 mean_direction_regions_ve_3D = []
#                                 mean_direction_regions_ve_2D = []

#                                 for speed_ii, speed in enumerate(lin_velocity_regions_ve):
#                                     if len(speed)>0:
#                                         dir3D = np.nanmedian(np.nanmean(speed, axis=1), axis=0)
#                                         mean_direction_regions_ve_3D.append(dir3D)
#                                         dir2D = lookup_3D_to_2D_directions(unwrap_params_ve, 
#                                                                             np.nanmean(unwrap_params_ve[polar_grid_stage_ii==uniq_polar_regions[speed_ii]], axis=0), 
#                                                                             dir3D, 
#                                                                             scale=10., use_3D=True)[1]
#                                         mean_direction_regions_ve_2D.append(dir2D)
#                                     else:
#                                         mean_direction_regions_ve_3D.append(np.hstack([np.nan, np.nan, np.nan]))

#                                 mean_direction_regions_ve_3D = np.array(mean_direction_regions_ve_3D)
#                                 mean_direction_regions_ve_2D = np.array(mean_direction_regions_ve_2D)



#                                 ### e) the mean anterior speed as mapped to 2D. ( averaging in 3D then reprojecting in 2D.)
#                                 disps_ve_3D_component_reg = [disps_ve_3D_component[select_cells_quad==reg] for reg in uniq_polar_regions]
#                                 disps_ve_3D_component_reg_all.append(disps_ve_3D_component_reg)

#                                 mean_speed_regions_ve_AVE_dir = []    

#                                 for speed_ii, speed in enumerate(disps_ve_3D_component_reg):
#                                     if len(speed)>0:
#                                         mean_speed_regions_ve_AVE_dir.append(np.nanmedian(speed))
#                                     else:
#                                         mean_speed_regions_ve_AVE_dir.append(np.nan)
                                        
#                                 mean_speed_regions_ve_AVE_dir = np.hstack(mean_speed_regions_ve_AVE_dir)



#                                 # =============================================================================
#                                 #      Try to do some sort of plot as a check? for the direction inference and plotting of heatmap to check all is ok.                         
#                                 # =============================================================================
#                                 from skimage.measure import regionprops
#                                 mean_pos_2D = np.vstack([re.centroid for re in regionprops(polar_grid_stage_ii)])
#                                 curvilinear_speed_map_ve = np.zeros(polar_grid_stage_ii.shape, dtype=np.float32)
#                                 for reg_ii, reg in enumerate(uniq_polar_regions):
#                                     curvilinear_speed_map_ve[polar_grid_stage_ii==reg] = mean_curvilinear_speed_regions_ve[reg_ii]

#                                 plt.figure()
#                                 plt.title(analyse_stages[stage_ii])
#                                 # plt.imshow(polar_grid_stage_ii, cmap='coolwarm')
#                                 plt.imshow(curvilinear_speed_map_ve, cmap='coolwarm', vmin=-np.max(mean_curvilinear_speed_regions_ve), vmax=np.max(mean_curvilinear_speed_regions_ve))
#                                 plt.plot(mean_pos_2D[:,1], 
#                                          mean_pos_2D[:,0], 'o')
#                                 plt.quiver(mean_pos_2D[:,1], 
#                                            mean_pos_2D[:,0], 
#                                            30*np.sin(mean_direction_regions_ve_2D), 
#                                            -30*np.cos(mean_direction_regions_ve_2D), color='k')
#                                 plt.show()



#                                 mean_lin_speed_regions_ve_all.append(mean_lin_speed_regions_ve)
#                                 mean_curvilinear_speed_regions_ve_all.append(mean_curvilinear_speed_regions_ve)
#                                 mean_total_speed_regions_ve_all.append(mean_total_speed_regions_ve)
#                                 mean_direction_regions_ve_3D_all.append(mean_direction_regions_ve_3D)
#                                 mean_direction_regions_ve_2D_all.append(mean_direction_regions_ve_2D)
#                                 mean_speed_regions_ve_AVE_dir_all.append(mean_speed_regions_ve_AVE_dir)
                        
# # #                         # version 2 code: using just the 3D. 
                        
# # #                         mean_direction_regions_ve_3D = [np.median(np.mean(disps_ve_3D[reg], axis=1), axis=0) for reg in uniq_polar_spixels]
# # #                         mean_direction_regions_epi_3D = [np.median(np.mean(disps_epi_3D[reg], axis=1), axis=0) for reg in uniq_polar_spixels]
                        
# # #                         all_directions_regions_ve_3D.append(mean_direction_regions_ve_3D)
# # #                         all_directions_regions_epi_3D.append(mean_direction_regions_epi_3D)


#                             """
#                             save everything into a separate .matfile for the embryo for easy analysis and plotting later, when we can concatenate over everything. 
#                             """
#                             savespeedstatsmatfile = os.path.join(mastercellstatsfolder, 'cell_track_velocity_stats_'+unwrapped_condition+'.mat')
                            
#                             spio.savemat(savespeedstatsmatfile, 
#                                         {'cell_times':cell_times,
#                                          'temp_tform_scales': temp_tform_scales,
#                                          'cell_track_3D_xyz_ref': cell_track_centroids_3D_ref,
#                                          'cell_track_AVE_dir_3D': AVE_surf_dir_ve,
#                                          'analyse_stages':analyse_stages,
#                                          'analyse_time_segs':analyse_time_segs,
#                                          'cell_track_quad_ids_stage':cell_track_quad_ids_stage,
#                                          've_angle':ve_angle_move, 
#                                          've_angle_vector_2D':ve_angle_vector,
#                                          'vol_factor':vol_factor,
#                                          'embryo_shape': embryo_im_shape, 
#                                          'cell_track_ids_stage': analysed_cell_tracks_all, 
#                                          'disps_ve_3D_all_norm_stage': disps_ve_3D_all, 
#                                          'AVE_dir_ve_3D_stage': anterior_dirs_3D_all, 
#                                          'region_ids': uniq_polar_regions, 
#                                          'region_lin_velocity': lin_velocity_regions_ve_all, 
#                                          'region_mean_curve_velocity': curvilinear_velocity_regions_ve_all, 
#                                          'region_mean_AVE_velocity': disps_ve_3D_component_reg_all, 
#                                          'region_mean_lin_speed': mean_lin_speed_regions_ve_all, 
#                                          'region_mean_curve_speed': mean_curvilinear_speed_regions_ve_all,
#                                          'region_mean_total_speed': mean_total_speed_regions_ve_all, 
#                                          'region_mean_dir_3D': mean_direction_regions_ve_3D_all, 
#                                          'region_mean_dir_2D': mean_direction_regions_ve_2D_all, 
#                                          'region_mean_AVE_speed': mean_speed_regions_ve_AVE_dir_all})
                            
                            

#                             cell_track_cell_neighbors = map_stat_vects_to_tracks(cell_tracks_cell_ids, 
#                                                                                 cell_ids, 
#                                                                                 cell_neighbors,
#                                                                                 cell_times)
#                             # quantification has to be done on this.... the track ids!. 
#                             cell_track_track_neighbors = lookup_cell_to_tra_neighbors(cell_track_cell_neighbors, 
#                                                                                       cell_tracks_cell_ids,
#                                                                                       cell_times)
#                             # compute the orientation relative to the ve !. 
                            
#                             # =============================================================================
#                             #   Visual checking (check orientation + ve_angle.      -> to prove that all the parameters are good. !                      
#                             # ============================================================================= 
                            
#                             # rotate the vid at time 0 
#                             cell_seg_img_rot0 = np.uint16(sktform.rotate(cell_seg_img[0], angle=(90-ve_angle_move), preserve_range=True))
                            
# #                            # rotate the inferred 2D points. 
# #                            cell_centroids_2D_rot = rotate_pts(cell_track_centroids_2D[:,0,::-1],
# #                                                               angle=-(90-ve_angle_move), 
# #                                                               center=[vid_ve[0].shape[1]/2.,vid_ve[0].shape[0]/2.])[:,::-1]
# #                            
# #                            plt.figure(figsize=(10,10))
# #                            plt.imshow(cell_seg_img_rot0)
# #                            plt.plot(cell_centroids_2D_rot[:,1], 
# #                                     cell_centroids_2D_rot[:,0], 
# #                                     'r.')
# #                            plt.show()
                            
                            
#                             # create rotated points of the 2D. ? 
#                             cell_track_centroids_2D_rot = np.array([rotate_pts(cc_centroids_2D[:,::-1],
#                                                                                angle=-(90-ve_angle_move), 
#                                                                                center=[vid_ve[0].shape[1]/2.,
#                                                                                        vid_ve[0].shape[0]/2.])[:,::-1] for cc_centroids_2D in cell_track_centroids_2D])
                            
#                             # display the centroids of the first timepoint!. 
#                             plt.figure(figsize=(10,10))
#                             plt.imshow(cell_seg_img_rot0)
#                             plt.plot(cell_track_centroids_2D_rot[:,0,1], 
#                                      cell_track_centroids_2D_rot[:,0,0], 
#                                      'r.')
#                             plt.show()
#                             # =============================================================================
#                             #       Set the analysis settings for the polar grid. 
#                             # =============================================================================    
                            
#                             # analyse_stages = ['Pre-migration', 
#                             #                   'Migration to Boundary',
#                             #                   'Post-Boundary']
#                             # stage_embryo = [pre, mig, post]

#                             # only for two stages is ok. 
#                             analyse_stages = ['Pre-migration', 
#                                               'Migration to Boundary']
# #                            stage_embryo = [pre, mig]
                            
#                             analyse_time_segs = [cell_times[cell_stages==stage] for stage in analyse_stages] # get the time segments for doing the averaging. (of the dynamic grid.) -> without tracking would be the same as static analysis!. 


#                             # For each track we now assign them to grid ids! based on their starting position in each stage based on their centroids 2D. 
#                             # if either they do not exist or whatever then their id will be 0 ! and excluded for analysis. 
                            
#                             # we only need to use the cell centroids to assign.
#                             track_quad_ids_stage = [assign_cell_track_quad_id(cell_track_centroids_2D[:,times], 
#                                                                               polar_grids_coarse[times], debug_viz=False) for times in analyse_time_segs[:]]
                            
                            
#                             # iterate over stage.
#                             uniq_polar_regions = np.setdiff1d(np.unique(polar_grids_coarse[0]), 0)
                            
#                             polar_grid_stats_area = []
#                             polar_grid_stats_perims = []
#                             polar_grid_stats_shape_index =[]
#                             polar_grid_stats_eccentricity = []
#                             polar_grid_stats_eccentricity_major_axis_2D = []
                            
#                             polar_grid_stats_diff_area = []
#                             polar_grid_stats_diff_perims = []
#                             polar_grid_stats_diff_shape_index =[]
#                             polar_grid_stats_diff_eccentricity = []
                            
#                             # build stats on the neighbors too..... -> now this is much harder... 
#                             polar_grid_stats_num_neighbors = []
#                             polar_grid_stats_num_neighbor_switches = []
                            
#                             for stage_ii in range(len(track_quad_ids_stage)):
                                
#                                 track_quad_ids_stage_ii = track_quad_ids_stage[stage_ii]
                                
#                                 polar_grid_stats_area_stage_ii = []
#                                 polar_grid_stats_perims_stage_ii = []
#                                 polar_grid_stats_shape_index_stage_ii =[]
#                                 polar_grid_stats_eccentricity_stage_ii = []
#                                 polar_grid_stats_major_axis_2D_stage_ii = []

#                                 polar_grid_stats_diff_area_stage_ii = []
#                                 polar_grid_stats_diff_perims_stage_ii = []
#                                 polar_grid_stats_diff_shape_index_stage_ii =[]
#                                 polar_grid_stats_diff_eccentricity_stage_ii = []

#                                 polar_grid_stats_num_neighbors_stage_ii = []
#                                 polar_grid_stats_num_neighbor_switches_stage_ii = []

#                                 # iterate over the unique_polar_regions and get the tracks we are interesting in.
#                                 for reg_ii in uniq_polar_regions:
#                                     select_tracks = track_quad_ids_stage_ii == reg_ii
                                    
#                                     if np.sum(select_tracks)>0:
#                                         # there are tracks that fulfill the criteria then, 
#                                         areas_time = cell_track_areas[select_tracks][:, analyse_time_segs[stage_ii]] # look up only the relevant. 
#                                         perims_time = cell_track_perims[select_tracks][:, analyse_time_segs[stage_ii]]
#                                         shape_index_time = cell_track_shape_index[select_tracks][:, analyse_time_segs[stage_ii]]
#                                         eccentricity_time = cell_track_eccentricity[select_tracks][:, analyse_time_segs[stage_ii]]
#                                         major_axis_2D_time = cell_track_major_axis_2D[select_tracks][:, analyse_time_segs[stage_ii]] # don't need the minor axis.
                                        
#                                         cell_neighbors_time = [cell_track_track_neighbors[nn] for nn in np.arange(len(select_tracks)) if select_tracks[nn]==True] # i.e. add to this if selected. 
#                                         cell_neighbors_time = [[cell_neighbors_time[cc][jjj] for jjj in analyse_time_segs[stage_ii]] for cc in range(len(cell_neighbors_time))] 
#                                         cell_neighbor_nums_time = count_cell_neighbors_time(cell_neighbors_time)
#                                         cell_neighbor_changes_time = count_cell_neighbors_changes_time(cell_neighbors_time) # this will have one less timepoint. 

#                                         # one for orientation
# #                                        orientation_time = 
#                                         # take the average over time. 
#                                         mean_areas_time = np.nanmedian(np.nanmean(areas_time, axis=1)) # over time then ensemble. 
#                                         mean_perims_time = np.nanmedian(np.nanmean(perims_time, axis=1))
#                                         mean_shape_index_time = np.nanmedian(np.nanmean(shape_index_time, axis=1))
#                                         mean_eccentricity_time = np.nanmedian(np.nanmean(eccentricity_time, axis=1))
#                                         mean_major_axis_2D_time = np.nanmedian(np.nanmean((eccentricity_time-1.)[...,None]*major_axis_2D_time, axis=1), axis=0)
                                        
#                                         mean_cell_neighbor_nums_time = np.nanmedian(np.nanmean(cell_neighbor_nums_time, axis=1))
#                                         mean_cell_neighbor_changes_time = np.nanmedian(np.nanmean(cell_neighbor_changes_time, axis=1))

#                                         polar_grid_stats_num_neighbors_stage_ii.append(mean_cell_neighbor_nums_time)
#                                         polar_grid_stats_num_neighbor_switches_stage_ii.append(mean_cell_neighbor_changes_time)

#                                         # also consider the rate of change of the same parameters? 
#                                         diff_areas_time = areas_time[:,1:] - areas_time[:,:-1]
#                                         diff_perims_time = perims_time[:,1:] - perims_time[:,:-1]
#                                         diff_shape_index_time = shape_index_time[:,1:] - shape_index_time[:,:-1]
#                                         diff_eccentricity_time = eccentricity_time[:,1:] - eccentricity_time[:,:-1]

#                                         mean_diff_areas_time = np.nanmedian(np.nanmean(diff_areas_time, axis=1)) # over time then ensemble. 
#                                         mean_diff_perims_time = np.nanmedian(np.nanmean(diff_perims_time, axis=1))
#                                         mean_diff_shape_index_time = np.nanmedian(np.nanmean(diff_shape_index_time, axis=1))
#                                         mean_diff_eccentricity_time = np.nanmedian(np.nanmean(diff_eccentricity_time, axis=1))
                                        
#                                         polar_grid_stats_area_stage_ii.append(mean_areas_time)
#                                         polar_grid_stats_perims_stage_ii.append(mean_perims_time)
#                                         polar_grid_stats_shape_index_stage_ii.append(mean_shape_index_time)
#                                         polar_grid_stats_eccentricity_stage_ii.append(mean_eccentricity_time)
#                                         polar_grid_stats_major_axis_2D_stage_ii.append(mean_major_axis_2D_time)

#                                         polar_grid_stats_diff_area_stage_ii.append(mean_diff_areas_time)
#                                         polar_grid_stats_diff_perims_stage_ii.append(mean_diff_perims_time)
#                                         polar_grid_stats_diff_shape_index_stage_ii.append(mean_diff_shape_index_time)
#                                         polar_grid_stats_diff_eccentricity_stage_ii.append(mean_diff_eccentricity_time)
                                    
#                                     else:
#                                         # append np.nan
#                                         polar_grid_stats_area_stage_ii.append(np.nan)
#                                         polar_grid_stats_perims_stage_ii.append(np.nan)
#                                         polar_grid_stats_shape_index_stage_ii.append(np.nan)
#                                         polar_grid_stats_eccentricity_stage_ii.append(np.nan)
#                                         polar_grid_stats_major_axis_2D_stage_ii.append(np.hstack([np.nan, np.nan]))

#                                         polar_grid_stats_diff_area_stage_ii.append(np.nan)
#                                         polar_grid_stats_diff_perims_stage_ii.append(np.nan)
#                                         polar_grid_stats_diff_shape_index_stage_ii.append(np.nan)
#                                         polar_grid_stats_diff_eccentricity_stage_ii.append(np.nan)

#                                         polar_grid_stats_num_neighbors_stage_ii.append(np.nan)
#                                         polar_grid_stats_num_neighbor_switches_stage_ii.append(np.nan)


#                                 polar_grid_stats_area.append(polar_grid_stats_area_stage_ii)
#                                 polar_grid_stats_perims.append(polar_grid_stats_perims_stage_ii)
#                                 polar_grid_stats_shape_index.append(polar_grid_stats_shape_index_stage_ii)
#                                 polar_grid_stats_eccentricity.append(polar_grid_stats_eccentricity_stage_ii)
#                                 polar_grid_stats_eccentricity_major_axis_2D.append(polar_grid_stats_major_axis_2D_stage_ii)

#                                 polar_grid_stats_diff_area.append(polar_grid_stats_diff_area_stage_ii)
#                                 polar_grid_stats_diff_perims.append(polar_grid_stats_diff_perims_stage_ii)
#                                 polar_grid_stats_diff_shape_index.append(polar_grid_stats_diff_shape_index_stage_ii)
#                                 polar_grid_stats_diff_eccentricity.append(polar_grid_stats_diff_eccentricity_stage_ii)

#                                 polar_grid_stats_num_neighbors.append(polar_grid_stats_num_neighbors_stage_ii)
#                                 polar_grid_stats_num_neighbor_switches.append(polar_grid_stats_num_neighbor_switches_stage_ii)


#                             # produce the grid based average statistics
#                             polar_grid_stats_area = np.array(polar_grid_stats_area)
#                             polar_grid_stats_perims = np.array(polar_grid_stats_perims)
#                             polar_grid_stats_shape_index = np.array(polar_grid_stats_shape_index)
#                             polar_grid_stats_eccentricity = np.array(polar_grid_stats_eccentricity)
#                             polar_grid_stats_eccentricity_major_axis_2D = np.array(polar_grid_stats_eccentricity_major_axis_2D) # this is still in image coordinates ( we need to compute the angles relative to the AVE direction!. )
                            
#                             # compute this to give an orientation angle relative to the ave ... 
#                             polar_grid_stats_eccentricity_angle_abs = np.arctan2(polar_grid_stats_eccentricity_major_axis_2D[...,1], 
#                                                                                  polar_grid_stats_eccentricity_major_axis_2D[...,0]) # since this is in yx convention.
#                             polar_grid_stats_eccentricity_angle_abs_rel = (ve_angle_move + 90)/180*np.pi - polar_grid_stats_eccentricity_angle_abs
#                             polar_grid_stats_eccentricity_angle_abs_rel = np.mod(polar_grid_stats_eccentricity_angle_abs_rel+np.pi, 2*np.pi) - np.pi
                            
#                             polar_grid_stats_diff_area = np.array(polar_grid_stats_diff_area)
#                             polar_grid_stats_diff_perims = np.array(polar_grid_stats_diff_perims)
#                             polar_grid_stats_diff_shape_index = np.array(polar_grid_stats_diff_shape_index)
#                             polar_grid_stats_diff_eccentricity = np.array(polar_grid_stats_diff_eccentricity)
                            
#                             # now this all looks better ..... 
#                             polar_grid_stats_num_neighbors = np.array(polar_grid_stats_num_neighbors)
#                             polar_grid_stats_num_neighbor_switches = np.array(polar_grid_stats_num_neighbor_switches) # still looks wrong!. 


#                             # now we can finally save all of this into its own .mat file ready for analysis and plotting. 
#                             # =============================================================================
#                             #          Save the array of collected stats for the single cell tracks.                   
#                             # =============================================================================
#                             savetrackstatsmatfile = os.path.join(mastercellstatsfolder, 'cell_track_and_polargrid_stats_'+unwrapped_condition+'.mat')
                           

#                             # may also want to take the opportunity to compute the movements? (save for separate analysis)
#                             # easiest for now to save as an array for computation. 
#                             spio.savemat(savetrackstatsmatfile, 
#                                             {'cell_times': cell_times,
#                                              'cell_stages': cell_stages,
#                                              'cell_ve_angle': cell_ve_angle, 
#                                              'singlestatsfile':savesinglestatsmatfile, 
#                                              'cell_track_areas': cell_track_areas,
#                                               'cell_track_perims': cell_track_perims,
#                                               'cell_track_shape_index': cell_track_shape_index,
#                                               'cell_track_eccentricity': cell_track_eccentricity,
#                                               'cell_track_major_axis_2D': cell_track_major_axis_2D,
#                                               'cell_track_minor_axis_2D': cell_track_minor_axis_2D,
#                                               'cell_track_centroids_2D': cell_track_centroids_2D,
#                                               'cell_track_centroids_3D': cell_track_centroids_3D,
#                                               'cell_track_cell_neighbors': cell_track_cell_neighbors,
#                                               'cell_track_track_neighbors': cell_track_track_neighbors,
#                                               'cell_track_centroids_2D_rot': cell_track_centroids_2D_rot,
#                                               'analyse_stages': analyse_stages,
#                                               'analyse_time_segs': analyse_time_segs,
#                                               'cell_track_quad_ids_stage': track_quad_ids_stage,
#                                               'uniq_polar_regions': uniq_polar_regions,
#                                               'polar_grid_stats_area': polar_grid_stats_area, 
#                                               'polar_grid_stats_perims': polar_grid_stats_perims,
#                                               'polar_grid_stats_shape_index': polar_grid_stats_shape_index,
#                                               'polar_grid_stats_eccentricity': polar_grid_stats_eccentricity, 
#                                               'polar_grid_stats_eccentricity_major_axis_2D': polar_grid_stats_eccentricity_major_axis_2D,
#                                               'polar_grid_stats_eccentricity_angle_abs_rel': polar_grid_stats_eccentricity_angle_abs_rel,
#                                               'polar_grid_stats_diff_area': polar_grid_stats_diff_area,
#                                               'polar_grid_stats_diff_perims': polar_grid_stats_diff_perims,
#                                               'polar_grid_stats_diff_shape_index': polar_grid_stats_diff_shape_index,
#                                               'polar_grid_stats_diff_eccentricity': polar_grid_stats_diff_eccentricit,
#                                               'polar_grid_stats_num_neighbors': polar_grid_stats_num_neighbors,
#                                               'polar_grid_stats_num_neighbor_switches': polar_grid_stats_num_neighbor_switches})



