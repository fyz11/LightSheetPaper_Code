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

def polar_grid_mask_coarse_from_fine(squares, polar_grid_fine_2_coarse_ids, polar_grid_ids, shape):

    from skimage.draw import polygon

    polar_grid_mask = np.zeros(shape, dtype=np.int)

    # n_regions x n_squares. - polar_grid_fine_2_coarse_ids
    # n_time x n_regions x n_points x n_dim
    for region_ii, fine_2_coarse_ids in enumerate(polar_grid_fine_2_coarse_ids):
        squares_grid = squares[fine_2_coarse_ids]

        for square in squares_grid:
            # draw the polygon
            rr,cc = polygon(square[:,1], 
                            square[:,0], 
                            shape=shape)
            polar_grid_mask[rr,cc] = polar_grid_ids[region_ii]

    return polar_grid_mask

def count_cells_mesh(grid_squares_2D, 
                     centroids_2D,
                     shape):

    from skimage.draw import polygon

    polar_grid_mask = np.zeros(shape, dtype=np.int)
    
    # 2D centroids are in x,y coordinates. 

    for square_ii, square in enumerate(grid_squares_2D):

        # draw the polygon
        rr,cc = polygon(square[:,1], 
                        square[:,0], 
                        shape=shape)
        polar_grid_mask[rr,cc] = square_ii+1

    uniq_polar_grid_ids = np.setdiff1d(np.unique(polar_grid_mask), 0)
    centroids_2D_ids = polar_grid_mask[centroids_2D[:,1].astype(np.int), 
                                       centroids_2D[:,0].astype(np.int)]

    counts_polar_grid_ids = np.hstack([np.nansum(centroids_2D_ids==iid) for iid in uniq_polar_grid_ids])

    return counts_polar_grid_ids


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

# wrapper function for create a polar fine grid from unwrap_params. 
def polar_grid_boundary_line_square_pts(shape, n_r, n_theta, 
                                        max_r, 
                                        mid_line, 
                                        center=None,
                                        bound_r=True,
                                        zero_angle = 0, #.values[0], 
                                        return_grid_pts=True, 
                                        ctrl_pt_resolution=360): 

    # prop a finer grid which we will use for obtaining canonical PCA deformation maps and computing average areal changes.
    polar_grid_fine, polar_grid_fine_ctrl_pts = tile_uniform_windows_radial_guided_line(shape, 
                                                                                        n_r=n_r, 
                                                                                        n_theta=n_theta,
                                                                                        max_r=max_r,
                                                                                        mid_line = mid_line,
                                                                                        center=center, 
                                                                                        bound_r=bound_r,
                                                                                        zero_angle = zero_angle, #.values[0], 
                                                                                        return_grid_pts=return_grid_pts)
    grid_center = np.hstack([shape[1]/2.,
                             shape[0]/2.])

    # # assign tracks to unique regions. 
    # uniq_polar_spixels = [(polar_grid==r_id)[meantracks_ve[:,0,0], meantracks_ve[:,0,1]] for r_id in uniq_polar_regions]
    polar_grid_fine_coordinates = infer_control_pts_xy(polar_grid_fine_ctrl_pts[0],
                                                        polar_grid_fine_ctrl_pts[1], 
                                                        ref_pt=grid_center, s=0, 
                                                        resolution=ctrl_pt_resolution)

    polar_grid_fine_coordinates = np.squeeze(polar_grid_fine_coordinates)
    
    #create periodic boundary conditions.
    polar_grid_fine_coordinates = np.hstack([polar_grid_fine_coordinates, 
                                            polar_grid_fine_coordinates[:,0][:,None]])

    grid_center_vect = np.array([grid_center]*polar_grid_fine_coordinates.shape[1])
    polar_grid_fine_coordinates = np.vstack([grid_center_vect[None,:],
                                            polar_grid_fine_coordinates])

    # additionally save polar_grid_fine_coordinates.
    """
    parse the grid squares point
    """
    # polar_grid_squares = parse_grid_squares_from_pts(polar_grid_coordinates)
    polar_grid_fine_squares = parse_grid_squares_from_pts(polar_grid_fine_coordinates)

    return polar_grid_fine_coordinates, polar_grid_fine_squares

def plot_polar_grid_boundaries(ax, coords, coarse_r, coarse_theta, rot_angle, rot_center, color='w', lw=3):

    coords_rot = rotate_pts(coords.reshape(-1,coords.shape[-1]), 
                            angle=rot_angle,
                            center=rot_center)
    coords_rot = coords_rot.reshape(coords.shape)
    
    for kk in coarse_r:
        kk = int(kk)
        ax.plot(coords_rot[kk,:,0], 
                coords_rot[kk,:,1], '-', color=color, lw=lw)

    for kk in coarse_line_index_theta:
        kk = int(kk)
        ax.plot(coords_rot[:,kk,0], 
                coords_rot[:,kk,1], '-', color=color, lw=lw)

    return []

def get_3D_centroids(unwrap_params, polar_grid):

    uniq_regs = np.setdiff1d(np.unique(polar_grid), 0) 
    centroids_2D = []
    centroids_3D = []

    for reg in uniq_regs:

        mask = polar_grid == reg
        mask_centre_3D = np.nanmean(unwrap_params[mask>0], axis=0)
        
        # if enforce_surface:
        mask_centre_3D_dists = unwrap_params - mask_centre_3D[None,None,:]
        mask_centre_3D_dists_min = np.argmin(np.linalg.norm(mask_centre_3D_dists, axis=-1))
        index_i, index_j = np.unravel_index(mask_centre_3D_dists_min, 
                                            unwrap_params.shape[:-1]) 
        # print('enforcing', index_i, index_j)
        mask_centre_2D = np.hstack([index_i, index_j])
        mask_centre_3D = unwrap_params[index_i, index_j]
            
        centroids_2D.append(mask_centre_2D)
        centroids_3D.append(mask_centre_3D)

    centroids_2D = np.array(centroids_2D)
    centroids_3D = np.array(centroids_3D)

    # additionally return the cell id. 
    return uniq_regs, np.array(centroids_2D), np.array(centroids_3D)

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


    # double check is this the correct one?
#    mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper'
    mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper-Full-Rec' # this is the full reconstruction and has been recomputed. 
    masterannotfolder = '/media/felix/My Passport/Shankar-2/All_Manual_Annotations/all_cell_annotations' # there will only be one of these therefore we should name these accordingly. 
    

    # specify the absolute save folder for usage. 
    mastersaveoutfolder = '/media/felix/My Passport/Shankar-2/All_Results/Cell_Divisions_Angles_Analysis'
    fio.mkdir(mastersaveoutfolder)

    mastersaveoutfolder_VE_surf_areas = '/media/felix/My Passport/Shankar-2/All_Results/VE_Surface_Areas_Rec'
    fio.mkdir(mastersaveoutfolder_VE_surf_areas)

    """
    start of algorithm
    """
    # all_rootfolders = ['/media/felix/Srinivas4/LifeAct', 
                    #    '/media/felix/Srinivas4/MTMG-TTR',
                    #    '/media/felix/Srinivas4/MTMG-HEX',
                    #    '/media/felix/Srinivas4/MTMG-HEX/new_hex']
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
    all_stages = manual_staging_tab.columns[1:4] # all 3 migration phases. 

    """
    set up all the save statistics for each stage. 
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
    for embryo_folder in all_embryo_folders[2:3]: # test just on the first embryo.  
    # for embryo_folder in all_embryo_folders[4:5]: #giving the L491. 
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
            # saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
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
            
            # this should be fine. 
            temp_tform_scales = np.vstack([parse_temporal_scaling(tformfile) for tformfile in temp_tform_files])
            temp_tform_scales = np.prod(temp_tform_scales, axis=0)
            
            plt.figure()
            plt.plot(temp_tform_scales)
            plt.show()
            
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
#                    ve_angle_move = ve_angle_move.values[0] # use consensus 
                    ve_angle_move = 90 - ve_angle_move.values[0] # to get back to the rectangular angle coordinate? 
                    # ve_angle = ve_angle_move.values[0]
                    print('ve movement direction: ', ve_angle_move)
                    
                    # the consenus angle is used for rotation/?
                # =============================================================================
                #       Load the computed statistics.      
                # =============================================================================
     
                    # get the demons_remapped_pts (this needs to be the true geometric points for us to get the respective surface area.)
                    saveprojfile = os.path.join(saveanalysisfolder, 'demons_remapped_pts-'+unwrapped_condition+'.mat')
                    print(saveprojfile)
                    stats_obj = spio.loadmat(saveprojfile)
                    
                    """
                    Get the required statistics, 
                    the remapped pts is the actual (x,y,z) in the same reference of polar. 
                    """
                    demons_times = stats_obj['demons_times']
                    
                    # this bit is the most useful for getting the points specific for the stage. 
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
                #         Attempt to load the corresponding cell divisions. (later on this is placeholder for cell tracks.)         
                # =============================================================================
                    celltracks_folder = os.path.join(masterannotfolder, unwrapped_condition)
                    
                    if os.path.exists(celltracks_folder):

                        """
                        attempt to load the cell div file (by detecting .csv file) through name.  -> must be loaded a-priori so we can apply all the necessary transformations. 
                        """
                        csvfiles = glob.glob(os.path.join(celltracks_folder, '*.csv'))
                        
                        divfile = [ csvfile for csvfile in csvfiles if 'div_polar' in os.path.split(csvfile)[-1]]
                        divfile = divfile[0]
                        
                        # read the annotated cell division table. 
                        celldivtable = pd.read_csv(divfile) # get the whole table. 


                        cellimgfiles = glob.glob(os.path.join(celltracks_folder, '*.tif'))
                        cellimgfiles = [ff for ff in cellimgfiles if 'cells_' in os.path.split(ff)[-1]]
                        
                        if len(cellimgfiles) > 0:

                            cellimgfile = cellimgfiles[0]
                            print(cellimgfile)
                            print('=========')

                            cell_img_ve = skio.imread(cellimgfile)
                            # then we continue. 
    #                        celltracks_file = os.path.join(celltracks_folder, 
    #                                                       'cell_tracks_w_ids_Holly_'+ unwrapped_condition +'_distal_corrected_nn_tracking-full-dist30.mat') # this file will include the cell ids. # and the centroid information.
    #                        cell_file = os.path.join(celltracks_folder, 
    #                                                 'cells_' + unwrapped_condition + '_2nd_edit_merge.tif')
    #                        
    #                        """
    #                        read in the data. so we can do embryo inversion. 
    #                        """
    #                        celltracks_ids_obj = spio.loadmat(celltracks_file) # this is unaffected by inversion.... 
    #                        cell_seg_img = skio.imread(cell_file)
    #                        
    #                        # note the ids should not need to be 'invereted in any manner long as the underlying cell images change.
    #                        cell_tracks_cell_ids = celltracks_ids_obj['cell_tracks_ids']; 
    #                        cell_tracks_cell_xy = celltracks_ids_obj['cell_tracks_xy']
                        
                        
    #                        embryo_inversion = True -> testing only. 
                        # =============================================================================
                        #       apply the embryo inversion               
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

                                    cell_img_ve = cell_img_ve[:,:,::-1] # flip the x-direction.
        
                                    # check.... do we need to invert the epi boundary too !. 
                                    epi_boundary_time[...,0] = (vid_ve[0].shape[1]-1) - epi_boundary_time[...,0] # correct x-coordinate.  
                                    
                                    
    #                                cell_seg_img = cell_seg_img[:,:,::-1] # flipping the x-axis.
    #                                cell_tracks_cell_xy[...,0] = (vid_ve[0].shape[1]-1) - cell_tracks_cell_xy[...,0]  # flip the x coordinate. 
                                    
                                    # flip the x-direction.
                                    celldivtable['mum_x'] = (vid_ve[0].shape[1]-1) - celldivtable['mum_x']
                                    celldivtable['daug1_x'] = (vid_ve[0].shape[1]-1) - celldivtable['daug1_x']
                                    celldivtable['daug2_x'] = (vid_ve[0].shape[1]-1) - celldivtable['daug2_x']
                                    
                                    
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

                                    cell_img_ve = cell_img_ve[:,::-1] # flip the y-direction. 
        
                                    epi_boundary_time[...,1] = (vid_ve[0].shape[0]-1) - epi_boundary_time[...,1] # correct the y-coordinate 
                                    
    #                                cell_seg_img = cell_seg_img[:,::-1] # flipping the y-axis.
    #                                cell_tracks_cell_xy[...,1] = (vid_ve[0].shape[0]-1) - cell_tracks_cell_xy[...,1]  # flip the y coordinate. 
                                    
                                    # flip the y-direction, updating the table directly. 
                                    celldivtable['mum_y'] = (vid_ve[0].shape[0]-1) - celldivtable['mum_y']
                                    celldivtable['daug1_y'] = (vid_ve[0].shape[0]-1) - celldivtable['daug1_y']
                                    celldivtable['daug2_y'] = (vid_ve[0].shape[0]-1) - celldivtable['daug2_y']
                                    
                                    if 'MTMG-HEX' in embryo_folder:
                                        vid_hex = vid_hex[:,::-1]
                                

                            """
                            plot and check the divisions on the plots ( seems ok ?)
                            """
                            fig, ax = plt.subplots(figsize=(10,10))
                            ax.imshow(vid_ve[-1], cmap='gray')
                            
                            for row_ii in range(len(celldivtable)):
                                celldivrow = celldivtable.iloc[row_ii]
                                ax.plot([celldivrow['daug1_x'], celldivrow['daug2_x']], 
                                        [celldivrow['daug1_y'], celldivrow['daug2_y']], 'o-')
                            plt.show()
                            
                            """
                            Rotate the coordinates to compute the alignments. (can also do this by image coordinates too...)
                                => more as a check
                            """
                            # mother pts
                            mother_xy = np.vstack([celldivtable['mum_x'].values, 
                                                   celldivtable['mum_y'].values]).T
                            daugh1_xy = np.vstack([celldivtable['daug1_x'].values, 
                                                   celldivtable['daug1_y'].values]).T
                            daugh2_xy = np.vstack([celldivtable['daug2_x'].values, 
                                                   celldivtable['daug2_y'].values]).T

                            # img_rot = np.uint8(sktform.rotate(vid_ve[0], angle=(90-ve_angle_move), preserve_range=True))
                            
                            # # rotate the points ( this is for debugging only. )
                            # mother_xy_rot = rotate_pts(mother_xy, angle=-(90-ve_angle_move), center=[vid_ve[0].shape[1]/2.,vid_ve[0].shape[0]/2.])
                            # daugh1_xy_rot = rotate_pts(daugh1_xy, angle=-(90-ve_angle_move), center=[vid_ve[0].shape[1]/2.,vid_ve[0].shape[0]/2.])
                            # daugh2_xy_rot = rotate_pts(daugh2_xy, angle=-(90-ve_angle_move), center=[vid_ve[0].shape[1]/2.,vid_ve[0].shape[0]/2.])

                            # fig, ax = plt.subplots(figsize=(10,10))
                            # ax.imshow(img_rot, cmap='gray')
                            # for iiii in range(len(daugh1_xy_rot)):
                            #     ax.plot([daugh1_xy_rot[iiii,0], daugh2_xy_rot[iiii,0]],
                            #             [daugh1_xy_rot[iiii,1], daugh2_xy_rot[iiii,1]], 'o-')
                            # plt.show()
                            
                            """
                            re-compute which of currently labelled 1 or 2 is '1' or '2' -> idea here is just to continue with the use of 'distal' as a setting point. 
                            """
                            img_center = np.hstack(vid_ve[0].shape[:2]) / 2. # y,x
                            distal_1_dist = np.linalg.norm(daugh1_xy - img_center[::-1][None,:], axis=-1)
                            distal_2_dist = np.linalg.norm(daugh2_xy - img_center[::-1][None,:], axis=-1)
                            
                            daugh1_xy_new = []
                            daugh2_xy_new = []
                            # reallocate daughter 1 and daughter 2.
                            
                            for daughter_iii in range(len(distal_1_dist)):
                                
                                min_id = np.argmin([distal_1_dist[daughter_iii], 
                                                    distal_2_dist[daughter_iii]])
                                if min_id == 0:
                                    daugh1_xy_new.append(daugh1_xy[daughter_iii])
                                    daugh2_xy_new.append(daugh2_xy[daughter_iii])
                                if min_id == 1:
                                    # do a switch. 
                                    daugh1_xy_new.append(daugh2_xy[daughter_iii])
                                    daugh2_xy_new.append(daugh1_xy[daughter_iii])
                            
                            daugh1_xy_new = np.vstack(daugh1_xy_new)
                            daugh2_xy_new = np.vstack(daugh2_xy_new)
                            
                            # plot check the reversal makes sense,
                            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
                            ax[0].set_title('before sort')
                            ax[0].imshow(vid_ve[0], cmap='gray')
                            for iiii in range(len(daugh1_xy_new)):
                                ax[0].plot([daugh1_xy[iiii,0], daugh2_xy[iiii,0]],
                                           [daugh1_xy[iiii,1], daugh2_xy[iiii,1]], 'o-')
                            ax[0].plot(daugh1_xy[:,0], daugh1_xy[:,1], 'go')
                            ax[0].plot(daugh2_xy[:,0], daugh2_xy[:,1], 'ro')
                            
                            # plot check the reversal makes sense,
                            # fig, ax = plt.subplots(figsize=(10,10))
                            ax[1].set_title('after sort')
                            ax[1].imshow(vid_ve[0], cmap='gray')
                            for iiii in range(len(daugh1_xy_new)):
                                ax[1].plot([daugh1_xy_new[iiii,0], daugh2_xy_new[iiii,0]],
                                           [daugh1_xy_new[iiii,1], daugh2_xy_new[iiii,1]], 'o-')
                            ax[1].plot(daugh1_xy_new[:,0], daugh1_xy_new[:,1], 'go')
                            ax[1].plot(daugh2_xy_new[:,0], daugh2_xy_new[:,1], 'ro')
                            plt.show()
                            
                            
    # #                        """
    # #                        Compute the relative angles with the vertical (with the rotated coordinates.)
    # #                        """
    # #                        vertical_vector = np.hstack([0,-1]) # in image coordinate format. 
    # #                        vertical_angle = -np.pi/2.
    # #                        
    # #                        daugh_angles_disp_vector = daugh1_xy_rot - daugh2_xy_rot
    # #                        daugh_angles_angle = np.arctan2(daugh_angles_disp_vector[:,1], 
    # #                                                        daugh_angles_disp_vector[:,0]) # the absolute angles.
    # ##                        daugh_angles_rel = np.arccos(np.sum(daugh_angles_disp_vector * vertical_vector[None,:], axis=-1)/(np.linalg.norm(daugh_angles_disp_vector,axis=-1) + 1e-8))
    # #                        
    # #                        # not in range of -np.pi/pi. 
    # #                        daugh_angles_angle_rel = daugh_angles_angle - vertical_angle
    # #                        daugh_angles_angle_rel_ = np.mod(daugh_angles_angle_rel+np.pi, 2*np.pi) - np.pi
    #                         """
    #                         computing the relative angles without rotation. (should be identical to the above :D, but without rotation, hence the above is now grayed out.)
    #                         """
    #                         daugh_angles_disp_vector_abs = daugh1_xy_new - daugh2_xy_new
    #                         # to get this consistent we should normalise this by taking the points closer to the distal etc... 
                            
    #                         daugh_angles_angle_abs = np.arctan2(daugh_angles_disp_vector_abs[:,0], daugh_angles_disp_vector_abs[:,1])
    #                         daugh_angles_angle_abs_rel = (ve_angle_move + 90)/180*np.pi - daugh_angles_angle_abs
    #                         daugh_angles_angle_abs_rel = np.mod(daugh_angles_angle_abs_rel+np.pi, 2*np.pi) - np.pi


                            """
                            Update the cell table directly with the new computed statistics, regarding the daughters. 
                            """
                            celldivtable['daug1_x'] = daugh1_xy_new[:,0]        
                            celldivtable['daug1_y'] = daugh1_xy_new[:,1]
                            celldivtable['daug2_x'] = daugh2_xy_new[:,0] 
                            celldivtable['daug2_y'] = daugh2_xy_new[:,1]
                            celldivtable['embryo_invert'] = embryo_inversion # this is for plotting purposes.  
                            # celldivtable['daug_div_angles_rel_AVE_dir'] = daugh_angles_angle_abs_rel # add this extra column. 
                            """
                            Create new columns to fill in information on quad ids, areas for static and dynamic grids.  
                            """
                            celldivtable['daug_div_angles_rel_AVE_dir'] = np.nan # initialise the cell division angle column  
                            celldivtable['Stage'] = np.nan # what stage of the migration is this in. 
                            celldivtable['Static_Quad_ID'] = np.nan # what is the quad id, this sits within .
                            celldivtable['Static_Quad_Cell_No'] = np.nan # what is the associated total cell number of the quad it sits in ....(based on cell centroid) 
                            celldivtable['Dynamic_Quad_ID'] = np.nan
                            celldivtable['Dynamic_Quad_Cell_No'] = np.nan

                            # =============================================================================
                            #     Create the find to coarse ids for spatial averaging.  
                            # =============================================================================
                            supersample_theta = 4 # 8 angle divisions
                            supersample_r = 4  # 4 distance divisions. 

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
                                    
                            # we use this to index into the array to get averages for each respective region.
                            polar_grid_fine_2_coarse_ids = np.array(polar_grid_fine_2_coarse_ids) 

                            """
                            create containers to save all of the area computations for each region for the embryo.
                            """
                            embryo_cell_nos_all_time = []
                            embryo_stages_all_time = []
                            embryo_cell_ids_all_time = []
                            embryo_cell_centroids_2D_all_time = [] # this is the centroids mapped back. 
                            embryo_cell_centroids_3D_all_time = [] # 

                            embryo_polar_grid_fine_cell_nos_3D_dynamic = []
                            embryo_polar_grid_fine_cell_nos_3D_static = []

                            embryo_polar_grid_coarse_cell_nos_3D_dynamic = []
                            embryo_polar_grid_coarse_cell_nos_3D_static = []

                            embryo_polar_grid_fine_cell_nos_3D_epi_contour = []
                            embryo_polar_grid_coarse_cell_nos_3D_epi_contour = []

                        # =============================================================================
                        #       Iterate over the different stages, get the times, get the deformation changes. 
                        # =============================================================================     
                            analyse_stages = ['Pre-migration', 
                                              'Migration to Boundary',
                                              'Post-Boundary']
                            stage_embryo = [pre, mig, post]
                        
                                        
                            for stage_ii in range(len(stage_embryo))[:]:
                                times = stage_embryo[stage_ii]
                                times = times.ravel()
        #                        ~np.isnan(pre.ravel()[0]) and ~np.isnan(mig.ravel()[0]):
                                if ~np.isnan(times[0]) and times[0] < len(cell_img_ve):
                                    
                                    analyse_stage = analyse_stages[stage_ii]
                                    print(times)
                                    # parse the migration times.                                 
                                    start, end = times
                                    start = start -1 
                                    if end == len(vid_ve):
                                        end = end 
                                    else:
                                        end = end - 1
                        
                                    start = int(start)
                                    end = int(end)
                                    
                                    start_end_times = np.hstack([start, end])
                                    
                        # =============================================================================
                        #       extract the statistics over time. and space.              
                        # =============================================================================
                        
                                    # first get the true 3D points in this region. # used to compute the area.  
                                    ve_pts_stage = ve_pts[start:end].copy()
    #                                epi_pts_stage = epi_pts[start:end].copy()
                                    
                                    # use the points to define the invalid polar region for assessement. investigate this .... might not hold now once the scaling is put back.  
                                    invalid_mask = ve_pts_stage[...,0] < 50
                                    
    #                                YY,XX = np.indices(vid_ve[0].shape)
    #                                dist_r = np.sqrt( (YY - vid_ve[0].shape[0]/2.)**2 + (XX - vid_ve[1].shape[0]/2.)**2)
    #                                r_max_2D = int(np.hstack(vid_ve[0].shape)[0]/2./1.2)
                                    
                                    # ve_cell_ids_stage = []
                                    # ve_cell_areas_stage = []
                                    # ve_cell_perims_stage = []
                                    # # ve_cell_shape_factor_stage = []
                                    # ve_cell_major_lens_stage = []
                                    # ve_cell_minor_lens_stage = []
                        # =============================================================================
                        #       Load the predefined square grids (propagated by optical flow)
                        # =============================================================================
                                    savedefmapfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'deformation_maps', 'motion_only', analyse_stage)
                                    # fio.mkdir(savedefmapfolder)

                                    savedefmapfile = os.path.join(savedefmapfolder, 'deformation_quadrants_'+unwrapped_condition+'.mat')
                                    # spio.savemat(savedefmapfile, {'pts_ve': pts_ve_all, 
                                    #                               'squares_ve':squares_ve_all,
                                    #                               'areas_ve':areas_ve_all, 
                                    #                               'pts_epi': pts_epi_all, 
                                    #                               'squares_epi':squares_epi_all, 
                                    #                               'areas_epi':areas_epi_all, 
                                    #                               've_angle':ve_angle_move, 
                                    #                               #.values[0],
                                    #                               'polar_grid': polar_grid,
                                    #                               'polar_grid_fine':polar_grid_fine,
                                    #                               'supersample_radii': supersample_r, 
                                    #                               'supersample_theta': supersample_theta, 
                                    #                               'analyse_stage': analyse_stage, 
                                    #                               'polar_grid_fine_coordinates':polar_grid_fine_coordinates, 
                                    #                               'coarse_line_index_r':coarse_line_index_r,
                                    #                               'coarse_line_index_theta':coarse_line_index_theta,
                                    #                               'epi_line_time':epi_contour_line,
                                    #                               'start_end_time':start_end_times})
                                    """
                                    do a check of the times.
                                    """
                                    print(analyse_stage, start_end_times, spio.loadmat(savedefmapfile)['start_end_time']) # ascertain these are the same. 
                                    print('=====')
                                    polar_grid = spio.loadmat(savedefmapfile)['polar_grid']
                                    uniq_polar_regions = np.setdiff1d(np.unique(polar_grid), 0)

                                    polar_grid_fine_coordinates = spio.loadmat(savedefmapfile)['polar_grid_fine_coordinates']
                                    coarse_line_index_r = spio.loadmat(savedefmapfile)['coarse_line_index_r'].ravel()
                                    coarse_line_index_theta = spio.loadmat(savedefmapfile)['coarse_line_index_theta'].ravel()

                                    squares_ve_stage = spio.loadmat(savedefmapfile)['squares_ve'] # this should be in line. # this is in (x,y) format? 

                                    ### check that the coarse to fine scheme works well. 
                                    # polar_grid_rec = np.zeros_like(polar_grid)
                                
                                    # from skimage.draw import polygon

                                    # # n_regions x n_squares. - polar_grid_fine_2_coarse_ids
                                    # # n_time x n_regions x n_points x n_dim
                                    # for region_ii, fine_2_coarse_ids in enumerate(polar_grid_fine_2_coarse_ids):
                                    #     squares_grid = squares_ve_stage[0, fine_2_coarse_ids]

                                    #     for square in squares_grid:
                                    #         # draw the polygon
                                    #         rr,cc = polygon(square[:,1], 
                                    #                         square[:,0], 
                                    #                         shape=polar_grid.shape)
                                    #         polar_grid_rec[rr,cc] = region_ii+1

                                    # # test1: plot this and test the coarse gridding scheme.
                                    # fig, ax = plt.subplots(figsize=(15,15))
                                    # ax.imshow(polar_grid, cmap='coolwarm')

                                    # fig, ax = plt.subplots(figsize=(15,15))
                                    # ax.imshow(polar_grid_rec, cmap='coolwarm')
                                    # plt.show()

                                    """
                                    check the propagation of the squares over time. 
                                    """

                                    # """
                                    # Plot all the division angles in this time period, rotated correctly. with the grid overlaid for this time period. 
                                    # """
                                    # celldivplotfolder = os.path.join(mastersaveoutfolder,
                                    #                                  'plots_cell_annots'); # to distinguish from the areas. 
                                    # fio.mkdir(celldivplotfolder)

                                    # # find cell divisions in this stage... 
                                    # select_div_stage = np.logical_and(celldivtable['t_stamp'].values >= int(start_end_times[0]),
                                    #                                   celldivtable['t_stamp'].values < int(start_end_times[1]))


                                    # fig, ax = plt.subplots(figsize=(8,8))
                                    # ax.set_title('cell-division-angles_'+analyse_stage)

                                    # img_rot = np.uint8(sktform.rotate(vid_ve[int(start_end_times[0])], 
                                    #                                   angle=(90-ve_angle_move), 
                                    #                                   preserve_range=True))
                                    # ax.imshow(img_rot, cmap='gray')


                                    # if np.sum(select_div_stage) > 0: 

                                    #     # Plot all the divisions in this stage. 
                                    #     mother_xy = np.vstack([celldivtable['mum_x'], celldivtable['mum_y']]).T
                                    #     daugh1_xy = np.vstack([celldivtable['daug1_x'], celldivtable['daug1_y']]).T
                                    #     daugh2_xy = np.vstack([celldivtable['daug2_x'], celldivtable['daug2_y']]).T

                                    #     mother_xy = mother_xy[select_div_stage]
                                    #     daugh1_xy = daugh1_xy[select_div_stage]
                                    #     daugh2_xy = daugh2_xy[select_div_stage]

                                    #     # rotate the points ( this is for debugging only. )
                                    #     mother_xy_rot = rotate_pts(mother_xy, angle=-(90-ve_angle_move), center=[vid_ve[0].shape[1]/2.,vid_ve[0].shape[0]/2.])
                                    #     daugh1_xy_rot = rotate_pts(daugh1_xy, angle=-(90-ve_angle_move), center=[vid_ve[0].shape[1]/2.,vid_ve[0].shape[0]/2.])
                                    #     daugh2_xy_rot = rotate_pts(daugh2_xy, angle=-(90-ve_angle_move), center=[vid_ve[0].shape[1]/2.,vid_ve[0].shape[0]/2.])

                                    #     for iijj in range(len(daugh1_xy_rot)):
                                    #         ax.plot([daugh1_xy_rot[iijj,0], daugh2_xy_rot[iijj,0]],
                                    #                 [daugh1_xy_rot[iijj,1], daugh2_xy_rot[iijj,1]], '-', color='g', lw=2)
                                    #         ax.plot(mother_xy_rot[iijj,0],
                                    #                 mother_xy_rot[iijj,1], 'go') # draw the centers. 

                                    # # Overlay the gridding. 
                                    # plot_polar_grid_boundaries(ax, 
                                    #                            polar_grid_fine_coordinates,
                                    #                            coarse_line_index_r,
                                    #                            coarse_line_index_theta, 
                                    #                            rot_angle = -(90-ve_angle_move),
                                    #                            rot_center= [vid_ve[0].shape[1]/2.,vid_ve[0].shape[0]/2.])
                                                               
                                    # ax.axis('off')
                                    # ax.grid('off')
                                    # fig.savefig(os.path.join(celldivplotfolder, unwrapped_condition+'-%s.svg' %(analyse_stage)), 
                                    #             dpi=300, bbox_inches='tight')
                                    # plt.show()

                        # =============================================================================
                        #       Main meat of the computation. (for the area) of each quadrant in the region. (we want to save this.) 
                        # =============================================================================

                                    # # based on these we can compute the areas for the coarse scheme afterwards # these will be saved separately in csvs. 
                                    # polar_grid_cell_no_times = []
                                    # polar_grid_stage_times = []

                                    # polar_grid_cell_ids_times = []
                                    # polar_grid_centroids_2D_times = []
                                    # polar_grid_centroids_3D_times = []

                                    # polar_grid_fine_cell_no_3D_dynamic = []
                                    # polar_grid_fine_cell_no_3D_static = []

                                    # polar_grid_coarse_cell_no_3D_dynamic = []
                                    # polar_grid_coarse_cell_no_3D_static = []

                                    # iterating over the timepoints. to compute the area statistics. for the quadranting scheme. 
                                    iter_counter = 0

                                    for time_ii_stage in np.arange(start,end)[:]:

                                        # update the time 
                                        embryo_cell_nos_all_time.append(time_ii_stage)
                                        # update the staging information to avoid having to operate with many different files. 
                                        embryo_stages_all_time.append(analyse_stage)

                                        """
                                        Load the VE surface points at this timepoint.  
                                        """
                                        # this is the absolute time. 
                                        ve_pts_time_ii = ve_pts_stage[iter_counter].copy()

                                        """
                                        Get the cells at this time frame 
                                        """
#                                        if time_ii_stage >= len(cell_img_ve):
                                        cells_frame = cell_img_ve[time_ii_stage].copy()

                                        # lookup centroids of the cells, used to refine the cell division angles.  
                                        cell_ids, cell_centroids_2D, cell_centroids_3D = get_3D_centroids(ve_pts_time_ii, 
                                                                                                          cells_frame)
                                        cell_centroids_2D = cell_centroids_2D[:,::-1] # to get (x,y) convention 

                                        embryo_cell_ids_all_time.append(cell_ids)
                                        embryo_cell_centroids_2D_all_time.append(cell_centroids_2D)
                                        embryo_cell_centroids_3D_all_time.append(cell_centroids_3D)

                                        """
                                        compute the dynamic ver.
                                        """
                                        squares_time_ii = squares_ve_stage[iter_counter].copy() # get the square grid. 

                                        cell_nos_fine_dynamic = count_cells_mesh(squares_time_ii, 
                                                                                 cell_centroids_2D,
                                                                                 vid_ve[0].shape)
                                        cell_nos_coarse_dynamic = np.array([ np.nansum(cell_nos_fine_dynamic[ids]) for ids in polar_grid_fine_2_coarse_ids])

                                        embryo_polar_grid_fine_cell_nos_3D_dynamic.append(cell_nos_fine_dynamic)
                                        embryo_polar_grid_coarse_cell_nos_3D_dynamic.append(cell_nos_coarse_dynamic)

                                        """
                                        compute the static ver.
                                        """
                                        cell_nos_fine_static = count_cells_mesh(squares_ve_stage[0], 
                                                                                cell_centroids_2D,
                                                                                vid_ve[0].shape) # always use the first time point gridding. 
                                        cell_nos_coarse_static = np.array([ np.nansum(cell_nos_fine_static[ids]) for ids in polar_grid_fine_2_coarse_ids])

                                        # append the areas. 
                                        embryo_polar_grid_fine_cell_nos_3D_static.append(cell_nos_fine_static)
                                        embryo_polar_grid_coarse_cell_nos_3D_static.append(cell_nos_coarse_static)


                                        """
                                        compute the static ver with the epi_contour at this time point. 
                                        """
                                        # by definition this is the same between epi and ve!. # define this grid for the current timepoint ( dynamic adjustment !)
                                        _, squares_fine_epi_contour = polar_grid_boundary_line_square_pts(shape=vid_ve[0].shape, 
                                                                                                            n_r=supersample_r*4, 
                                                                                                            n_theta=(supersample_theta+1)*8,
                                                                                                            max_r=vid_ve[0].shape[0]/2./1.2, # always the same. 
                                                                                                            mid_line = epi_boundary_time[time_ii_stage,:],
                                                                                                            center=None, 
                                                                                                            bound_r=True,
                                                                                                            zero_angle = ve_angle_move, 
                                                                                                            ctrl_pt_resolution=360)

                                        plt.figure(figsize=(10,10))
                                        plt.imshow(vid_ve[0])
#                                        for 
                                        plt.plot(squares_fine_epi_contour[:,:,0].ravel(),
                                                 squares_fine_epi_contour[:,:,1].ravel(), 'r.')
                                        plt.show()
                                        # # i) areas.
                                        # surf_area_ve_pts_fine_epi_contour = area_3D_mesh_fast(squares_fine_epi_contour, 
                                        #                                                         ve_pts_time_ii) # always use the first time point gridding. 
                                        # surf_area_epi_pts_fine_epi_contour = area_3D_mesh_fast(squares_fine_epi_contour, 
                                        #                                                         epi_pts_time_ii) # always use the first time point gridding. 
                                        cell_nos_fine_epi_contour_static = count_cells_mesh(squares_fine_epi_contour, 
                                                                                cell_centroids_2D,
                                                                                vid_ve[0].shape) # always use the first time point gridding. 
                                        cell_nos_coarse_epi_contour_static = np.array([ np.nansum(cell_nos_fine_epi_contour_static[ids]) for ids in polar_grid_fine_2_coarse_ids])

                                        embryo_polar_grid_fine_cell_nos_3D_epi_contour.append(cell_nos_fine_epi_contour_static)
                                        embryo_polar_grid_coarse_cell_nos_3D_epi_contour.append(cell_nos_coarse_epi_contour_static)


                                        """
                                        mark the dynamic quadrants any divisions belong to. 
                                        """
                                        # detect if any divisions occur at this timepoint. 
                                        TP = time_ii_stage
                                        div_select_TP = celldivtable['t_stamp'].values == TP # need to double check this. ? 

                                        if np.sum(div_select_TP) > 0:

                                            # wait.... this is a parallel update? => double check. !
                                            # if a cell division has been detected .... 

                                            # mother
                                            mother_xy = np.vstack([celldivtable.loc[div_select_TP]['mum_x'].values, 
                                                                   celldivtable.loc[div_select_TP]['mum_y'].values]).T
                                            mother_xy_int = mother_xy.astype(np.int) # get the integer position of the list!. 

                                            # daughter 1
                                            daugh1_xy = np.vstack([celldivtable.loc[div_select_TP]['daug1_x'].values, 
                                                                   celldivtable.loc[div_select_TP]['daug1_y'].values]).T
                                            daugh1_xy_int = daugh1_xy.astype(np.int) # get the integer position

                                            # daughter 2
                                            daugh2_xy = np.vstack([celldivtable.loc[div_select_TP]['daug2_x'].values, 
                                                                   celldivtable.loc[div_select_TP]['daug2_y'].values]).T
                                            daugh2_xy_int = daugh2_xy.astype(np.int) # get the integer position

                                            # get the mother cell id! in order to get the refined xy position.  
                                            mother_cell_ids = cells_frame[mother_xy_int[:,1], 
                                                                          mother_xy_int[:,0]]
                                            # write a loop for this!. 
                                            mother_xy = np.vstack([cell_centroids_2D[cell_ids==mother_cell_id] for mother_cell_id in mother_cell_ids] )


                                            # get the cell id of daughters.
                                            cells_frame_plus_1 = cell_img_ve[time_ii_stage+1].copy()
                                            daugh1_cell_ids = cells_frame_plus_1[daugh1_xy_int[:,1], 
                                                                                 daugh1_xy_int[:,0]]
                                            daugh2_cell_ids = cells_frame_plus_1[daugh2_xy_int[:,1], 
                                                                                 daugh2_xy_int[:,0]]

                                            cell_ids_plus_1, cell_centroids_2D_plus_1, cell_centroids_3D_plus_1 = get_3D_centroids(ve_pts[time_ii_stage+1], 
                                                                                                                                   cells_frame_plus_1)
                                            
                                            daugh1_xy = np.vstack([cell_centroids_2D_plus_1[cell_ids_plus_1==daugh1_cell_id] for daugh1_cell_id in daugh1_cell_ids])
                                            daugh2_xy = np.vstack([cell_centroids_2D_plus_1[cell_ids_plus_1==daugh2_cell_id] for daugh2_cell_id in daugh2_cell_ids])


                                            """
                                            update the cell table. with the new divisions. 
                                            """
                                            celldivtable.loc[div_select_TP]['mum_x'] = mother_xy[:,0]
                                            celldivtable.loc[div_select_TP]['mum_y'] = mother_xy[:,1]
                                            celldivtable.loc[div_select_TP]['daug1_x'] = daugh1_xy[:,0]
                                            celldivtable.loc[div_select_TP]['daug1_y'] = daugh1_xy[:,1]
                                            celldivtable.loc[div_select_TP]['daug2_x'] = daugh2_xy[:,0]
                                            celldivtable.loc[div_select_TP]['daug2_y'] = daugh1_xy[:,1]
                                            
                                            mother_xy_int = mother_xy.astype(np.int) # update this. 

                                            # if a cell division is detected at this timepoint update the relevant statistics. 

                                            # update the stage
                                            celldivtable.loc[div_select_TP, 'Stage'] = analyse_stage

                                            # compute the polar grid masks for this time from the squares. 
                                            polar_grid_time_ii = polar_grid_mask_coarse_from_fine(squares_ve_stage[iter_counter], 
                                                                                                  polar_grid_fine_2_coarse_ids, 
                                                                                                  polar_grid_ids=uniq_polar_regions, 
                                                                                                  shape=polar_grid.shape)

                                            mother_xy_dynamic_quad_id = polar_grid_time_ii[mother_xy_int[:,1],mother_xy_int[:,0]]
                                            mother_xy_dynamic_quad_cell_no = cell_nos_coarse_dynamic[mother_xy_dynamic_quad_id-1]
                                            
                                            celldivtable.loc[div_select_TP, 'Dynamic_Quad_ID'] = mother_xy_dynamic_quad_id # what is the quad id, this sits within .
                                            celldivtable.loc[div_select_TP, 'Dynamic_Quad_Cell_No'] = mother_xy_dynamic_quad_cell_no

                                            """
                                            mark the static quadrants any divisions belong to. 
                                            """
                                            # technically don't need this but just for consistency. (same as polar_grid.)
                                            polar_grid_time_00 = polar_grid_mask_coarse_from_fine(squares_ve_stage[0], 
                                                                                                  polar_grid_fine_2_coarse_ids, 
                                                                                                  polar_grid_ids=uniq_polar_regions, 
                                                                                                  shape=polar_grid.shape)

                                            mother_xy_static_quad_id = polar_grid_time_00[mother_xy_int[:,1],mother_xy_int[:,0]]
                                            mother_xy_static_quad_cell_no = cell_nos_coarse_static[mother_xy_static_quad_id-1]

                                            celldivtable.loc[div_select_TP, 'Static_Quad_ID'] = mother_xy_static_quad_id # what is the quad id, this sits within .
                                            celldivtable.loc[div_select_TP, 'Static_Quad_Cell_No'] = mother_xy_static_quad_cell_no


                                            # compute the division angles using the revised centroids. 

                                            """
                                            computing the relative angles without rotation. (should be identical to the above :D, but without rotation, hence the above is now grayed out.)
                                            """
                                            daugh_angles_disp_vector_abs = daugh1_xy - daugh2_xy
                                            # to get this consistent we should normalise this by taking the points closer to the distal etc... 
                                            
                                            daugh_angles_angle_abs = np.arctan2(daugh_angles_disp_vector_abs[:,0], daugh_angles_disp_vector_abs[:,1])
                                            daugh_angles_angle_abs_rel = (ve_angle_move + 90)/180*np.pi - daugh_angles_angle_abs
                                            daugh_angles_angle_abs_rel = np.mod(daugh_angles_angle_abs_rel+np.pi, 2*np.pi) - np.pi

                                            celldivtable.loc[div_select_TP, 'daug_div_angles_rel_AVE_dir'] = daugh_angles_angle_abs_rel # initialise the cell division angle column  

                                        iter_counter += 1

                                    # polar_grid_cell_no_times = np.hstack(polar_grid_cell_no_times)
                                    # polar_grid_stage_times = np.hstack(polar_grid_stage_times)

                                    # polar_grid_cell_ids_times = np.hstack(polar_grid_cell_ids_times)

                                    # polar_grid_fine_areas_3D_dynamic = np.array(polar_grid_fine_areas_3D_dynamic)
                                    # polar_grid_fine_areas_3D_static = np.array(polar_grid_fine_areas_3D_static)

                                    # polar_grid_coarse_areas_3D_dynamic = np.array(polar_grid_coarse_areas_3D_dynamic)
                                    # polar_grid_coarse_areas_3D_static = np.array(polar_grid_coarse_areas_3D_static)


                                    # # append to global ????  
                                    # embryo_areas_all_time.append(polar_grid_area_times)
                            
                                    # embryo_polar_grid_fine_areas_3D_dynamic.append(polar_grid_fine_areas_3D_dynamic)
                                    # embryo_polar_grid_fine_areas_3D_static.append(polar_grid_fine_areas_3D_static)

                                    # embryo_polar_grid_coarse_areas_3D_dynamic.append(polar_grid_coarse_areas_3D_dynamic)
                                    # embryo_polar_grid_coarse_areas_3D_static.append(polar_grid_coarse_areas_3D_static)

                                    # polar_grid_cell_no_times = []
                                    # polar_grid_stage_times = []

                                    # polar_grid_cell_ids_times = []
                                    # polar_grid_centroids_2D_times = []
                                    # polar_grid_centroids_3D_times = []

                                    # polar_grid_fine_cell_no_3D_dynamic = []
                                    # polar_grid_fine_cell_no_3D_static = []

                                    # polar_grid_coarse_cell_no_3D_dynamic = []
                                    # polar_grid_coarse_cell_no_3D_static = []


                                    
                            # compile this into csvtable for saving. 
                            embryo_cell_nos_all_time = np.hstack(embryo_cell_nos_all_time)
                            embryo_stages_all_time = np.hstack(embryo_stages_all_time)

                            embryo_cell_ids_all_time = np.array(embryo_cell_ids_all_time)
                            embryo_cell_centroids_2D_all_time = np.array(embryo_cell_centroids_2D_all_time) # this is the centroids mapped back. 
                            embryo_cell_centroids_3D_all_time = np.array(embryo_cell_centroids_3D_all_time) # 

                            embryo_polar_grid_fine_cell_nos_3D_dynamic = np.vstack(embryo_polar_grid_fine_cell_nos_3D_dynamic)
                            embryo_polar_grid_fine_cell_nos_3D_static = np.vstack(embryo_polar_grid_fine_cell_nos_3D_static)

                            embryo_polar_grid_coarse_cell_nos_3D_dynamic = np.vstack(embryo_polar_grid_coarse_cell_nos_3D_dynamic)
                            embryo_polar_grid_coarse_cell_nos_3D_static = np.vstack(embryo_polar_grid_coarse_cell_nos_3D_static)

                            embryo_polar_grid_fine_cell_nos_3D_epi_contour = np.vstack(embryo_polar_grid_fine_cell_nos_3D_epi_contour)
                            embryo_polar_grid_coarse_cell_nos_3D_epi_contour = np.vstack(embryo_polar_grid_coarse_cell_nos_3D_epi_contour)


                            cols_fine = np.hstack(['Quad_%s' %(str(quad_ii+1)).zfill(3) for quad_ii in range(embryo_polar_grid_fine_cell_nos_3D_dynamic.shape[1])])
                            cols_coarse = np.hstack(['Quad_%s' %(str(quad_ii+1)).zfill(3) for quad_ii in range(embryo_polar_grid_coarse_cell_nos_3D_dynamic.shape[1])])


                            """
                            Compute the four tables, which we can save to.
                            """
                            embryo_polar_grid_fine_cell_nos_3D_dynamic_table = pd.DataFrame(np.hstack([embryo_cell_nos_all_time[:,None], 
                                                                                                        embryo_stages_all_time[:,None],
                                                                                                        embryo_polar_grid_fine_cell_nos_3D_dynamic]), 
                                                                                         index=None, 
                                                                                         columns=np.hstack(['Frame', 'Stage', cols_fine]))

                            embryo_polar_grid_fine_cell_nos_3D_static_table = pd.DataFrame(np.hstack([embryo_cell_nos_all_time[:,None], 
                                                                                                      embryo_stages_all_time[:,None],
                                                                                                      embryo_polar_grid_fine_cell_nos_3D_static]), 
                                                                                         index=None, 
                                                                                         columns=np.hstack(['Frame', 'Stage', cols_fine]))


                            embryo_polar_grid_coarse_cell_nos_3D_dynamic_table = pd.DataFrame(np.hstack([embryo_cell_nos_all_time[:,None], 
                                                                                                         embryo_stages_all_time[:,None],
                                                                                                         embryo_polar_grid_coarse_cell_nos_3D_dynamic]), 
                                                                                         index=None, 
                                                                                         columns=np.hstack(['Frame', 'Stage', cols_coarse]))


                            embryo_polar_grid_coarse_cell_nos_3D_static_table = pd.DataFrame(np.hstack([embryo_cell_nos_all_time[:,None], 
                                                                                                        embryo_stages_all_time[:,None],
                                                                                                        embryo_polar_grid_coarse_cell_nos_3D_static]), 
                                                                                         index=None, 
                                                                                         columns=np.hstack(['Frame', 'Stage', cols_coarse]))

     

                            embryo_polar_grid_fine_cell_nos_3D_epi_contour_table = pd.DataFrame(np.hstack([embryo_cell_nos_all_time[:,None], 
                                                                                                        embryo_stages_all_time[:,None],
                                                                                                        embryo_polar_grid_fine_cell_nos_3D_epi_contour]), 
                                                                                         index=None, 
                                                                                         columns=np.hstack(['Frame', 'Stage', cols_fine]))

                            embryo_polar_grid_coarse_cell_nos_3D_epi_contour_table = pd.DataFrame(np.hstack([embryo_cell_nos_all_time[:,None], 
                                                                                                        embryo_stages_all_time[:,None],
                                                                                                        embryo_polar_grid_coarse_cell_nos_3D_epi_contour]), 
                                                                                         index=None, 
                                                                                         columns=np.hstack(['Frame', 'Stage', cols_coarse]))


                            """
                            Save out all the generated files. 
                            """
                            savedivtablefilepath = os.path.join(mastersaveoutfolder, 
                                                               'divtable_angles_from_cell_annot_%s.csv' %(unwrapped_condition))
                            celldivtable.to_csv(savedivtablefilepath, index=None)

     
                            mastersaveoutfolder_VE_surf_areas_embryo = os.path.join(mastersaveoutfolder_VE_surf_areas, unwrapped_condition)
                            fio.mkdir(mastersaveoutfolder_VE_surf_areas_embryo)
                           
                           
                            # save the areas. 
                            embryo_polar_grid_fine_cell_nos_3D_dynamic_table.to_csv(os.path.join(mastersaveoutfolder_VE_surf_areas_embryo, 
                                                                                             'dynamic_quad_cell_no_fine-%s.csv' %(unwrapped_condition)), index=None)
                            embryo_polar_grid_fine_cell_nos_3D_static_table.to_csv(os.path.join(mastersaveoutfolder_VE_surf_areas_embryo, 
                                                                                             'static_quad_cell_no_fine-%s.csv' %(unwrapped_condition)), index=None)
                            embryo_polar_grid_coarse_cell_nos_3D_dynamic_table.to_csv(os.path.join(mastersaveoutfolder_VE_surf_areas_embryo, 
                                                                                             'dynamic_quad_cell_no_coarse-%s.csv' %(unwrapped_condition)), index=None)
                            embryo_polar_grid_coarse_cell_nos_3D_static_table.to_csv(os.path.join(mastersaveoutfolder_VE_surf_areas_embryo, 
                                                                                             'static_quad_cell_no_coarse-%s.csv' %(unwrapped_condition)), index=None)

                            # a static ver. with epi-contour marking the partitioning for every timepoint. 
                            embryo_polar_grid_fine_cell_nos_3D_epi_contour_table.to_csv(os.path.join(mastersaveoutfolder_VE_surf_areas_embryo, 
                                                                                             'static_epi-contour_quad_cell_no_fine-%s.csv' %(unwrapped_condition)), index=None)
                            embryo_polar_grid_coarse_cell_nos_3D_epi_contour_table.to_csv(os.path.join(mastersaveoutfolder_VE_surf_areas_embryo, 
                                                                                             'static_epi-contour_quad_cell_no_coarse-%s.csv' %(unwrapped_condition)), index=None)
                            

                            save_cell_centroids_file = os.path.join(mastersaveoutfolder_VE_surf_areas_embryo, 
                                                                    'cell_centroids_VE_info-%s.mat' %(unwrapped_condition))
                            spio.savemat(save_cell_centroids_file, 
                                            {'condition': unwrapped_condition, 
                                             've_angle': ve_angle_move, 
                                             'times': embryo_cell_nos_all_time, 
                                             'stages':embryo_stages_all_time, 
                                             'cell_ids': embryo_cell_ids_all_time,
                                             'centroids_2D': embryo_cell_centroids_2D_all_time, 
                                             'centroids_3D': embryo_cell_centroids_3D_all_time, 
                                             'embryo_inversion':embryo_inversion})

    ######################### rest is not required ############################