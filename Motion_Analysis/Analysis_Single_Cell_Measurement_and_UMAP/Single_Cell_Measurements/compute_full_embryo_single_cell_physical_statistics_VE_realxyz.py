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
def fit_ellipse_3D_to_cells(pts3D, return_axes=True):
    
    # Local axes. 
    from sklearn.decomposition import PCA
    
    pts = pts3D- pts3D.mean(axis=0)[None,:]
    cov = np.cov(pts.T)
    evalues, evecs = np.linalg.eigh(cov)
   
    sort_evalues = np.argsort(np.abs(evalues))
    evecs = evecs[:,sort_evalues]
    evalues = np.sqrt(evalues[sort_evalues]) # starts from smallest. 

    if return_axes:
        return evecs, evalues
    else:
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


def compute_cell_props(cell_img, unwrap_params, invalid_mask, unwrap_params_normals=None):
    
    from skimage.measure import find_contours
    
    uniq_cells = np.setdiff1d(np.unique(cell_img), 0 )

    cell_areas_3D = []
    cell_perimeters_3D = []
    cell_centroids_2D = []
    cell_major_minor_lengths = []
    cell_major_minor_axes = [] 

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
        reg_evecs_3D_pts, reg_evalues_3D_pts = fit_ellipse_3D_to_cells(reg_area_3D_pts) # use the entire regions's pts. 
        # print(reg_evalues_3D_pts)
        major_length = reg_evalues_3D_pts[-1]
        minor_length = reg_evalues_3D_pts[-2]

        # check these ! ( should be fine )
        major_axis = reg_evecs_3D_pts[:,-1] # check this. 
        minor_axis = reg_evecs_3D_pts[:,-2] 

#        print(surf_area, surf_perimeter)
        cell_areas_3D.append(surf_area)
        cell_perimeters_3D.append(surf_perimeter)
        cell_centroids_2D.append(np.hstack([centroid_2D_ind_y, centroid_2D_ind_x]))
        cell_major_minor_lengths.append(np.hstack([major_length, minor_length]))
        cell_major_minor_axes.append(np.vstack([major_axis,
                                                minor_axis]))

    cell_areas_3D = np.hstack(cell_areas_3D)
    cell_perimeters_3D = np.hstack(cell_perimeters_3D)
    cell_centroids_2D = np.vstack(cell_centroids_2D)
    cell_major_minor_lengths = np.vstack(cell_major_minor_lengths)
    cell_major_minor_axes = np.array(cell_major_minor_axes)
    
#    fig, ax = plt.subplots()
#    ax.imshow(cell_img)
#    ax.plot(cell_centroids_2D[:,1], 
#            cell_centroids_2D[:,0], 'w.')
#    plt.show()

    return uniq_cells, (cell_areas_3D, cell_perimeters_3D, cell_major_minor_lengths, cell_major_minor_axes)


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


def pullback_3D_directions(pos3D, dir3D, unwrap_params, disp_dist=10):
    
#    """
#    pos3D is the query position on the surface. 
#    dir3d = normalised.
#        - returns in (y,x) coordinate conventions 
#    """# this will find the equivalent 2D directional vector on the unwrap_params. 
    dir_2D = []
    
    for pos_ii, pos_3D_ii in enumerate(pos3D):
        
        pt1 = pos_3D_ii - disp_dist/2. * dir3D[pos_ii]
        pt2 = pos_3D_ii + disp_dist/2. * dir3D[pos_ii]

        # project these 2 points onto the unwrap. 

        # map back to 2D point 1  
        binary_centre_3D_dists_1 = unwrap_params - pt1[None,None,:]
        binary_centre_3D_dists_min_1 = np.argmin(np.linalg.norm(binary_centre_3D_dists_1, axis=-1))
        index_i_1, index_j_1 = np.unravel_index(binary_centre_3D_dists_min_1, 
                                            unwrap_params.shape[:-1]) 
        pt1_2D = np.hstack([index_i_1, index_j_1])

        # map back to 2D point 2  
        binary_centre_3D_dists_2 = unwrap_params - pt2[None,None,:]
        binary_centre_3D_dists_min_2 = np.argmin(np.linalg.norm(binary_centre_3D_dists_2, axis=-1))
        index_i_2, index_j_2 = np.unravel_index(binary_centre_3D_dists_min_2, 
                                                unwrap_params.shape[:-1]) 
        pt2_2D = np.hstack([index_i_2, index_j_2])

        dir2D_ii = pt2_2D - pt1_2D
        dir_2D.append(dir2D_ii)

    dir_2D = np.vstack(dir_2D)
    
    return dir_2D


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


def get_neighbor_cell_ids(cell_img, 
                          cell_ids, 
                          dilate_size=1, 
                          dilate_elem=None, 
                          min_count=2):

    import skimage.morphology as skmorph

    cell_neighbors = []

    for cell_id in cell_ids:

        mask = cell_img == cell_id 
        # any other tricks other than dilation? 
        if dilate_elem is not None:
            mask = skmorph.binary_dilation(mask, dilate_elem(dilate_size))
        else:
            mask = skmorph.binary_dilation(mask, skmorph.square(dilate_size))
        
        mask_cell_ids = cell_img[mask>0]
        mask_ids = np.setdiff1d(np.unique(mask_cell_ids), [0,cell_id]) 
#        cell_id_frequency = np.hstack([np.sum(mask_cell_ids==mask_id) for mask_id in mask_ids])
#        mask_ids = mask_ids[cell_id_frequency>min_count]

        if len(mask_ids) > 0:
            cell_id_frequency = np.hstack([np.sum(mask_cell_ids==mask_id) for mask_id in mask_ids])
            mask_ids = mask_ids[cell_id_frequency>min_count]
            if len(mask_ids) > 0:
                cell_neighbors.append(np.hstack(mask_ids))
            else:
                cell_neighbors.append([])
        else:
            cell_neighbors.append([])

    return cell_neighbors

# as a checking function. 
def get_segmentation_mask(cell_img, 
                          cell_ids):
    
    out = np.zeros(cell_img.shape)
    
    for cc in cell_ids:
        out[cell_img==cc] = cc
        
    return out
    

def grow_polar_regions_nearest(labels, unwrap_params, dilate=5, metric='manhattan', debug=False):
    
    from scipy.ndimage.morphology import binary_fill_holes
    from sklearn.neighbors import NearestNeighbors
    """
    script will grow the cell regions so that there is 0 distance between each cell region. 
    """
    bg = labels>0
    bg = skmorph.binary_closing(bg, skmorph.disk(dilate))
    valid_mask = binary_fill_holes(bg) # this needs to be a circle... 
    
    uncovered_mask = np.logical_and(valid_mask, labels==0)
    covered_mask = np.logical_and(valid_mask, labels > 0)
    
    if debug: 
        plt.figure()
        plt.title('uncovered')
        plt.imshow(uncovered_mask)
        plt.show()
        
        plt.figure()
        plt.title('covered')
        plt.imshow(covered_mask)
        plt.show()
        
    YY, XX = np.indices(labels.shape)
    
    uncovered_coords = np.vstack([YY[uncovered_mask], 
                                  XX[uncovered_mask]]).T
        
    covered_coords = np.vstack([YY[covered_mask], 
                                XX[covered_mask]]).T
    
    uncovered_coords_3D = unwrap_params[uncovered_coords[:,0], 
                                        uncovered_coords[:,1]]
    covered_coords_3D = unwrap_params[covered_coords[:,0], 
                                      covered_coords[:,1]]
    
    """
    build a kNN tree for assigning the remainder. 
    """
    neigh = NearestNeighbors(n_neighbors=1, metric=metric)
    neigh.fit(covered_coords_3D)

    nbr_indices = neigh.kneighbors(uncovered_coords_3D, return_distance=False)
    nbr_indices = nbr_indices.ravel()
    
    """
    look up what labels should be assigned
    """
    labels_new = labels.copy()
    
    nbr_pts = covered_coords[nbr_indices]
    
    labels_new[uncovered_coords[:,0], 
               uncovered_coords[:,1]] = labels[nbr_pts[:,0], nbr_pts[:,1]].copy()
    
    return labels_new

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


def polar_grid_boundary_line_square_pts(shape, n_r, n_theta, 
                                        max_r, 
                                        mid_line, 
                                        center=None,
                                        bound_r=True,
                                        zero_angle = 0, #.values[0], 
                                        return_grid_pts=True, 
                                        return_mask=True,
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

    if return_mask:
        return polar_grid_fine, polar_grid_fine_coordinates, polar_grid_fine_squares
    else:
        return polar_grid_fine_coordinates, polar_grid_fine_squares


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
    import skimage.morphology as skmorph


    """
    Purpose of this main data measurement script = 
        1) area
        2) perimeter
        3) shape index
        4) major / minor / eccentricity of all cells. 
        5) eccentricity orientation of all cells. 

        in addition.                 
        2) determine all cells neighbour cell ids in polar coordinates over all time. -> investigate a check for this !. 
        3) 
    """
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

    mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper-Demons-Rec' # this is the demons only reconstruction. 
    # mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper-Full-Rec' # this is the full reconstruction for the surface from the binary
    masterannotfolder = '/media/felix/My Passport/Shankar-2/All_Manual_Annotations/all_cell_annotations' # there will only be one of these therefore we should name these accordingly. 
    

    """
    Set up a master save folder to dump all the statistics to 
    """
    mastercellstatsfolder = '/media/felix/My Passport/Shankar-2/All_Results/Single_Cell_Shape_Statistics_All/Data'
    fio.mkdir(mastercellstatsfolder)

    """
    define some save folders. 
    """
    # master_curvature_folder = '/media/felix/My Passport/Shankar-2/All_Results/Volumetric_Statistics_All' 
    master_curvature_folder = '/media/felix/My Passport/Shankar-2/All_Results/Volumetric_Statistics_All-2' 

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
    for embryo_folder in all_embryo_folders[:]: #giving the L491. 
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
                    
#                    pre_all.append(pre)
#                    mig_all.append(mig)
#                    post_all.append(post)
#                    print(pre, mig, post)
                    
                    
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
                    unwrap_params_epi = sktform.resize(unwrap_params_epi, unwrap_params_ve.shape, preserve_range=True) # proportional alignment
                         
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
                #       Load the inverted coordinate statistics. (real?)      
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
                    
#                    # remove the distal point from this 
#                    # do this for both the centroids 3D (single cells) and for the tracks? 
#                    distal_center = (np.hstack(ve_pts[0].shape[:2])//2).astype(np.int)
#                    distal_3D_pts_ve = ve_pts[:,distal_center[0], distal_center[1]].copy()
#                    
#                    ve_pts = ve_pts - distal_3D_pts_ve[:,None,None,:]
#                    epi_pts = epi_pts - distal_3D_pts_ve[:,None,None,:] # this is probably the right way.... 
                    
                    
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

                        # do all of this separately? 
                        # #######
                        # # use keyword search to locate the tracks of relevance. 
                        # #######
                        # # celltracks_file = os.path.join(celltracks_folder, 
                        #                                # 'cell_tracks_w_ids_Holly_'+ unwrapped_condition +'_distal_corrected_nn_tracking-full-dist30.mat') # this file will include the cell ids. # and the centroid information.
                        # celltracks_file = glob.glob(os.path.join(celltracks_folder, '*.mat')) # the equivalent .csv is for Jason's program. 
                        # celltracks_file = [ff for ff in celltracks_file if 'cell_tracks_' in os.path.split(ff)[-1]]

                        # locate the cells file 
                        cell_file = glob.glob(os.path.join(celltracks_folder, '*.tif'))
                        cell_file = [ff for ff in cell_file if 'cells_' in os.path.split(ff)[-1]]

                        # needs both files. 
#                        if len(cell_file) > 0 and len(celltracks_file) > 0: 
                        if len(cell_file) > 0:
                            # proceed only if we have this segmentation mask file. 
                            cell_file = cell_file[0] 
#                            celltracks_file = celltracks_file[0]

                            """
                            read in the data. so we can do embryo inversion on the data for final quantification. 
                            """
                            # celltracks_ids_obj = spio.loadmat(celltracks_file) # this is unaffected by inversion.... 
                            cell_seg_img = skio.imread(cell_file) # this is a uint16 object. 
                            # # note the ids should not need to be 'invereted in any manner long as the underlying cell images change.
                            # cell_tracks_cell_ids = celltracks_ids_obj['cell_tracks_ids']; 
                            # cell_tracks_cell_xy = celltracks_ids_obj['cell_tracks_xy']
                        
    #                        embryo_inversion = True -> testing only. 
                        # =============================================================================
                        #       apply the embryo inversion to the inputs...               
                        # =============================================================================
                            if embryo_inversion == True:

                                # apply the necessary corrections !. 
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
                                    
                                    
                                    cell_seg_img = cell_seg_img[:,:,::-1] # flipping the x-axis.
                                    # cell_tracks_cell_xy[...,0] = (vid_ve[0].shape[1]-1) - cell_tracks_cell_xy[...,0]  # flip the x coordinate. 
                                    
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
        
                                    # invert the coordinates
                                    ve_pts = ve_pts[:,::-1]; ve_pts[...,2] = embryo_im_shape[2] - ve_pts[...,2] # this is temporal. 
                                    epi_pts = epi_pts[:,::-1]; epi_pts[...,2] = embryo_im_shape[2] - epi_pts[...,2]
                                    
                                    vid_ve = vid_ve[:,::-1]
                                    vid_epi = vid_epi[:,::-1]
                                    vid_epi_resize = vid_epi_resize[:,::-1]
        
                                    epi_boundary_time[...,1] = (vid_ve[0].shape[0]-1) - epi_boundary_time[...,1] # correct the y-coordinate 
                                    
                                    cell_seg_img = cell_seg_img[:,::-1] # flipping the y-axis.
                                    # cell_tracks_cell_xy[...,1] = (vid_ve[0].shape[0]-1) - cell_tracks_cell_xy[...,1]  # flip the y coordinate.                                     
                                    
                                    if 'MTMG-HEX' in embryo_folder:
                                        vid_hex = vid_hex[:,::-1]

                                    # apply correction to optflow.
                                    flow_ve = flow_ve[:,::-1]
                                    flow_ve[...,1] = -flow_ve[...,1] #(x,y)
                                    flow_epi = flow_epi[:,::-1]
                                    flow_epi[...,1] = -flow_epi[...,1] #(x,y)
                                    flow_epi_resize = flow_epi_resize[:,::-1]
                                    flow_epi_resize[...,1] = -flow_epi_resize[...,1] #(x,y)
                                
                            # Create the grid to go from fine to coarse grid ( this is just to reuse information. )
                            # =============================================================================
                            #     Create the find to coarse ids for spatial averaging and for transforming the fine grid to the coarse grid for statistics.  
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



                            # =============================================================================
                            #       Read and load up the gaussian curvature images.  
                            # =============================================================================     
                            # directly save this. 
                            emb_out_save_folder = os.path.join(master_curvature_folder, unwrapped_condition);
                            savematfile_curvature = os.path.join(emb_out_save_folder, 'curvature_dense_'+unwrapped_condition+'.mat')
                            curvature_obj = spio.loadmat(savematfile_curvature)

                            # note .... these are already operating on the inversed.
                            curvature_VE = curvature_obj['curvature_VE']
                            curvature_Epi = curvature_obj['curvature_Epi']

                            print(curvature_VE.shape, curvature_Epi.shape)
                            print('+++++++++++++++++')
                            # spio.savemat(savematfile_curvature, {'times': embryo_areas_all_time,
                            #                                      'stages': embryo_areas_all_time_stages,
                            #                                      've_angle': ve_angle_move, 
                            #                                      'curvature_VE': curvature_VE_embryo,
                            #                                      'curvature_Epi': curvature_Epi_embryo})
    
                            # =============================================================================
                            #       Set up the saves for individual cell area computation of all cells. How to handle cell area  
                            # =============================================================================     
    
    
                            # set up the saves for the embryo ... for the whole time -> then we only need to have one save file? -> allowing more flexibility ?
    
                            # this is here as a check all timepoints that we want have been incorporated ... ssss
                            embryo_all_time = [] # give the full individual cell areas over time. 
                            embryo_all_cell_ids = []
                            embryo_all_stage = []

                            polar_grids_static_coarse = [] # this is recomputed based entirely on the AVE migration angle + the epi contour at every time point!. 
    

                            all_flow_VE_3D = []
                            all_flow_Epi_3D = []

                            # these are the stuff already been computed ..... 
                            embryo_all_cell_VE_flow = []
                            embryo_all_cell_Epi_flow = []

                            # embryo_all_cell_demons_flow_VE_AVE_dir = []
                            # embryo_all_cell_demons_flow_VE_AVE_perp_dir = []
                            # embryo_all_cell_demons_flow_VE_AVE_norm_dir = []
                            # embryo_all_cell_demons_flow_VE_AVE_surf_dir = []

                            # embryo_all_cell_demons_flow_Epi_AVE_dir = []
                            # embryo_all_cell_demons_flow_Epi_AVE_perp_dir = []
                            # embryo_all_cell_demons_flow_Epi_AVE_norm_dir = []
                            # embryo_all_cell_demons_flow_Epi_AVE_surf_dir = []

                            # embryo_all_cell_curvature_VE = []
                            # embryo_all_cell_curvature_Epi = []

                            analyse_stages = ['Pre-migration', 
                                              'Migration to Boundary']
                            stage_embryo = [pre, mig]
                        
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
                                    
                        # =============================================================================
                        #       extract the statistics over time. and space.              
                        # =============================================================================
                        
                                    # first get the 3D points. 
                                    ve_pts_stage = ve_pts[start:end].copy()
                                    epi_pts_stage = epi_pts[start:end].copy()
                                    
                                    # use the points to define the invalid polar region for assessement. .... 
                                    invalid_mask = ve_pts_stage[...,0] < 50 # do we need this ? i think so? for the shape/perimeter computations? 
                                    
    #                                YY,XX = np.indices(vid_ve[0].shape)
    #                                dist_r = np.sqrt( (YY - vid_ve[0].shape[0]/2.)**2 + (XX - vid_ve[1].shape[0]/2.)**2)
    #                                r_max_2D = int(np.hstack(vid_ve[0].shape)[0]/2./1.2)
                                    
                                    
                        # =============================================================================
                        #       Load the predefined square grids (propagated by optical flow that gives the moving grid scheme)
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
                                    coarse_line_index_r = spio.loadmat(savedefmapfile)['coarse_line_index_r']
                                    coarse_line_index_theta = spio.loadmat(savedefmapfile)['coarse_line_index_theta']
    
                                    squares_ve_stage = spio.loadmat(savedefmapfile)['squares_ve'] # this should be in line. # this is in (x,y) format? 
    
        #                     # =============================================================================
        #                     #       Load previously prepared Demons points. 
        #                     # =============================================================================

        #                             demons_savefolder = os.path.join(embryo_folder, 'Demons_analysis', 
        #                                                                            'Demons_Temporal_Change_Shape', 
        #                                                                             analyse_stage); 
        #     #                            fio.mkdir(savefolder)
        #                             new_demonsmastersavefolder = '/media/felix/My Passport/Shankar-2/Demons_remap_vector'
        #                             old_demonsmastersavefolder = os.path.split(embryo_folder)[0]
                                        
        #                             demons_savefile = os.path.join(demons_savefolder, 'demons_unwrap_params_rev_pts_AVE_vects_'+unwrapped_condition+'.mat')
        #                             demons_saverootfolder, demons_savesuffix = os.path.split(demons_savefile)
        #                             demons_newsaverootfolder = demons_saverootfolder.replace(old_demonsmastersavefolder,
        #                                                                                         new_demonsmastersavefolder);

        #                             demons_savefile = os.path.join(demons_newsaverootfolder, demons_savesuffix)
        #                             demons_saveobj = spio.loadmat(demons_savefile)
        #                             # print('saving ... ', savefile)
        #                     # spio.savemat(savefile, 
        #                     #              {'ve_pts': ve_pts_stage.astype(np.float32),
        #                     #               'epi_pts': epi_pts_stage.astype(np.float32), 
        #                     #               'start_end_times':start_end_times, 
        #                     #               've_diff_pts': ve_diff_pts_stage.astype(np.float32), 
        #                     #               'epi_diff_pts': epi_diff_pts_stage.astype(np.float32), 
        #                     #               'analyse_stage': analyse_stages[stage_ii], 
        #                     #               've_dir_vect_AVE':ve_dir_vect_AVE.astype(np.float32),
        #                     #               've_dir_vect_AVE_perp':ve_dir_vect_AVE_perp.astype(np.float32),
        #                     #               'epi_dir_vect_AVE':epi_dir_vect_AVE.astype(np.float32),
        #                     #               'epi_dir_vect_AVE_perp':epi_dir_vect_AVE_perp.astype(np.float32),
        #                     #               've_angle_vector':ve_angle_vector, 
        #                     #               've_norm_angle_vector':ve_norm_angle_vector,
        #                     #               'embryo_inversion':embryo_inversion})
        #                             print(demons_saveobj['start_end_times'], start_end_times)
        #                             print('==========')


        #                             ###################### key computations to get the shape variation ####################
        #                             demons_save_ve_pts_stage = demons_saveobj['ve_pts']
        #                             demons_save_epi_pts_stage = demons_saveobj['epi_pts']
        #                             demons_ve_diff_pts_stage = demons_save_ve_pts_stage[1:] - demons_save_ve_pts_stage[:-1] 
        #                             demons_epi_diff_pts_stage = demons_save_epi_pts_stage[1:] - demons_save_epi_pts_stage[:-1] 

        #                             demons_ve_dir_vect_AVE = demons_saveobj['ve_dir_vect_AVE']
        #                             demons_ve_dir_vect_AVE_perp = demons_saveobj['ve_dir_vect_AVE_perp']
        # #                            ve_dir_vect_AVE_norm = np.array([np.cross(ve_dir_vect_AVE[tt], 
        # #                                                                      ve_dir_vect_AVE_perp[tt]) for tt in np.arange(len(ve_dir_vect_AVE_perp))])
        #                             demons_ve_dir_vect_AVE_norm = np.cross(demons_ve_dir_vect_AVE.reshape(-1,3), 
        #                                                                     demons_ve_dir_vect_AVE_perp.reshape(-1,3)).reshape(demons_ve_dir_vect_AVE.shape)
        # #                            ve_dir_vect_AVE_norm = np.array([vv / (np.linalg.norm(vv, axis=-1)[...,None]+1e-8) for vv in ve_dir_vect_AVE_norm])
        #                             demons_ve_dir_vect_AVE_norm = demons_ve_dir_vect_AVE_norm / (np.linalg.norm(demons_ve_dir_vect_AVE_norm, axis=-1)[...,None]+1e-8) 
                                    
        #                             demons_epi_dir_vect_AVE = demons_saveobj['epi_dir_vect_AVE']
        #                             demons_epi_dir_vect_AVE_perp = demons_saveobj['epi_dir_vect_AVE_perp']
        #                             demons_epi_dir_vect_AVE_norm = np.cross(demons_epi_dir_vect_AVE, demons_epi_dir_vect_AVE_perp)
        #                             demons_epi_dir_vect_AVE_norm = demons_epi_dir_vect_AVE_norm / (np.linalg.norm(demons_epi_dir_vect_AVE_norm, axis=-1)[...,None]+1e-8)

        #                 # =============================================================================
        #                 #       Demons projection onto the local curvilinear vector orientations. 
        #                 # =============================================================================

        #                             # so the below is the most important to save. 
        #                             demons_ve_diff_dir_vect_AVE = np.sum(demons_ve_diff_pts_stage * demons_ve_dir_vect_AVE, axis=-1)
        #                             demons_ve_diff_dir_vect_AVE_perp = np.sum(demons_ve_diff_pts_stage * demons_ve_dir_vect_AVE_perp, axis=-1)
        #                             demons_ve_diff_dir_vect_AVE_norm = np.sum(demons_ve_diff_pts_stage * demons_ve_dir_vect_AVE_norm, axis=-1)
        #                             # ve_diff_surface = ve_diff_pts_stage - ve_diff_dir_vect_AVE_norm[...,None] * ve_dir_vect_AVE_norm # minus the component normal to surface.
        #                             demons_ve_diff_surface = demons_ve_diff_dir_vect_AVE[...,None] * demons_ve_dir_vect_AVE + demons_ve_diff_dir_vect_AVE_perp[...,None]*demons_ve_dir_vect_AVE_perp

        #                             demons_epi_diff_dir_vect_AVE = np.sum(demons_epi_diff_pts_stage * demons_epi_dir_vect_AVE, axis=-1)
        #                             demons_epi_diff_dir_vect_AVE_perp = np.sum(demons_epi_diff_pts_stage * demons_epi_dir_vect_AVE_perp, axis=-1)
        #                             demons_epi_diff_dir_vect_AVE_norm = np.sum(demons_epi_diff_pts_stage * demons_epi_dir_vect_AVE_norm, axis=-1)
        #                             # epi_diff_surface = epi_diff_pts_stage - epi_diff_dir_vect_AVE_norm[...,None] * epi_dir_vect_AVE_norm
        #                             demons_epi_diff_surface = demons_epi_diff_dir_vect_AVE[...,None]*demons_epi_dir_vect_AVE + demons_epi_diff_dir_vect_AVE_perp[...,None] * demons_epi_dir_vect_AVE_perp 

                        # =============================================================================
                        #       Doing the statistics for the individual cells. 
                        # =============================================================================
                                    # # these are the saved statistics for each cell. 
                                    # ve_cell_ids_stage = []
                                    # ve_cell_areas_stage = []
                                    # ve_cell_perims_stage = []
                                    # # not computed here, check below. 
                                    # # ve_jamming_cell_shape_factor_stage = [] # ? https://jcs.biologists.org/content/129/18/3375 , perimeter/root(area) = 3.81 is the critical point.
                                    # ve_cell_major_lens_stage = []
                                    # ve_cell_minor_lens_stage = []
    
    
                                    # # geometric properties of the ve_cells -> technically we have this saved? 
                                    # ve_cell_centroids_2D_stage = []
                                    # ve_cell_centroids_3D_stage = []
    
                                    # polar_grids_dynamic_stage = []
    
                                    # iterating over the timepoints. to compute various statistics.

                                    cnt=0 
                                    # max_diff_cnt = len(demons_ve_diff_dir_vect_AVE)

                                    for time_ii_stage in np.arange(start,end)[:]:
                                        
                                        # time_ii_stage is absolute time !!!

                                        # append the time. 
                                        embryo_all_time.append(time_ii_stage)
                                        embryo_all_stage.append(analyse_stage)

                                        """
                                        1. Compute the polar grid partition. 
                                        """
                                        polar_grid_time_ii, grid_pts_coarse_contour, squares_coarse_epi_contour = polar_grid_boundary_line_square_pts(shape=vid_ve[0].shape, 
                                                                                                                                                        n_r=4, 
                                                                                                                                                        n_theta=8,
                                                                                                                                                        max_r=vid_ve[0].shape[0]/2./1.2, # always the same. 
                                                                                                                                                        mid_line = epi_boundary_time[time_ii_stage,:],
                                                                                                                                                        center=None, 
                                                                                                                                                        bound_r=True,
                                                                                                                                                        zero_angle = ve_angle_move, 
                                                                                                                                                        ctrl_pt_resolution=360, 
                                                                                                                                                        return_mask=True)

                                        polar_grids_static_coarse.append(polar_grid_time_ii)
    
                                        """
                                        Fetch the unique cell ids in this frame. 
                                        """
                                        #### now we get the average displacements for each 'cell' in the cell mask :D
                                        cell_seg_img_frame = cell_seg_img[time_ii_stage].copy()
                                        uniq_cells_frame = np.setdiff1d(np.unique(cell_seg_img_frame), 0 )
                                        embryo_all_cell_ids.append(uniq_cells_frame)

                                        """
                                        2. get the 3D remapped optflow at the timepoint over all the cells.  
                                        """
                                        # get the actual values of the 3D flow and save this separate to saving just the cells? 
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
                                        
                                        # flow_ve_3D = []
                                        # flow_epi_3D_resize = []

                #                        multiply factor = 2
#                                        multiply_factor=2. # 1 is definitiely too small .... 
                                        multiply_factor=5. # this is much needed? 

                                        # print(multiply_factor)
                                        # for frame_no in np.arange(len(flow_ve))[:]:
                                        flow_ve_frame = flow_ve[time_ii_stage].copy()
            #                            flow_ve_frame = smooth_flow_ds(flow_ve_frame, sigma=3, ds=4)
            #                            multiply_factor=2 
                                        # numerical errors? -> # can we get better by using a multiplication factor? 
                                        YY_next_ve = YY + multiply_factor*flow_ve_frame[...,1]; YY_next_ve = np.clip(np.rint(YY_next_ve), 0, flow_ve_frame.shape[0]-1) 
                                        XX_next_ve = XX + multiply_factor*flow_ve_frame[...,0]; XX_next_ve = np.clip(np.rint(XX_next_ve), 0, flow_ve_frame.shape[1]-1) 
                                        
                                        # quite a lot of errors? 
                                        flow_ve_3D_frame =  ve_pts[time_ii_stage+1][YY_next_ve.astype(np.int), XX_next_ve.astype(np.int)] - ve_pts[time_ii_stage][YY,XX]; 
                                        flow_ve_3D_frame = flow_ve_3D_frame.astype(np.float32) 
                                        flow_ve_3D_frame = flow_ve_3D_frame / float(multiply_factor)
                                        # flow_ve_3D.append(flow_ve_3D_frame)
                                        
                                        flow_epi_frame = flow_epi_resize[time_ii_stage].copy()
            #                            flow_epi_frame = smooth_flow_ds(flow_epi_frame, sigma=3, ds=4)
                                        YY_next_epi = YY + multiply_factor*flow_epi_frame[...,1]; YY_next_epi = np.clip(np.rint(YY_next_epi), 0, flow_epi_frame.shape[0]-1) 
                                        XX_next_epi = XX + multiply_factor*flow_epi_frame[...,0]; XX_next_epi = np.clip(np.rint(XX_next_epi), 0, flow_epi_frame.shape[1]-1)
                                        flow_epi_3D_frame =  epi_pts[time_ii_stage+1][YY_next_epi.astype(np.int), XX_next_epi.astype(np.int)] - epi_pts[time_ii_stage][YY,XX]; 
                                        flow_epi_3D_frame = flow_epi_3D_frame.astype(np.float32)       
                                        flow_epi_3D_frame = flow_epi_3D_frame/float(multiply_factor) # is the multiply factor still needed? 
                                        # flow_epi_3D_resize.append(flow_epi_3D_frame)


#                                        # check the pullback.
#                                        #                            # look this back up in 2D? # bit noisy but not that bad? 
#                                        centroid_2D, centroid_2D_prinvec, centroid_2D_secvec = lookup_3D_to_2D_loc_and_directions(ve_pts[time_ii_stage], 
#                                                                                                                                  ve_pts[time_ii_stage].reshape(-1,3), 
#                                                                                                                                  flow_ve_3D_frame.reshape(-1,3), 
#                                                                                                                                  flow_ve_3D_frame.reshape(-1,3),
#                                                                                                                                  scale=10.)
#                                        centroid_2D_prinvec = centroid_2D_prinvec.reshape(flow_ve_frame.shape)
#                                        # sampling = 10
#                                        # fig, ax = plt.subplots(figsize=(15,15))
#                                        # ax.imshow(vid_ve[start_], cmap='gray')
#                                        # ax.quiver(XX[::sampling,::sampling], 
#                                        #          YY[::sampling,::sampling],
#                                        #          flow_ve_frame[::sampling,::sampling,0],
#                                        #          -flow_ve_frame[::sampling,::sampling,1], color='r', scale=50)
#                                        # ax.quiver(XX[::sampling,::sampling], 
#                                        #          YY[::sampling,::sampling],
#                                        #          centroid_2D_prinvec[::sampling,::sampling,1],
#                                        #          -centroid_2D_prinvec[::sampling,::sampling,0], color='g', scale=50)
#                                        # plt.show()
#
#                                        """
#                                        Can we check this by pullback?
#                                        """
#                                        YYY,XXX = np.indices(flow_ve_3D_frame.shape[:2])
#
#                                        plt.figure(figsize=(20,20))
#                                        plt.subplot(121)
#                                        plt.imshow(vid_ve[time_ii_stage], cmap='gray')
#                                        plt.quiver(XXX[::20,::20],
#                                                   YYY[::20,::20],
#                                                   flow_ve_frame[::20,::20,0],
#                                                   -flow_ve_frame[::20,::20,1], color='r')
#                                        plt.subplot(122)
#                                        plt.imshow(vid_ve[time_ii_stage], cmap='gray')
#                                        plt.quiver(XXX[::20,::20],
#                                                   YYY[::20,::20],
#                                                   centroid_2D_prinvec[::20,::20,1],
#                                                   -centroid_2D_prinvec[::20,::20,0], color='r')
#                                        plt.show()
##
##
#                                        centroid_2D, centroid_2D_prinvec, centroid_2D_secvec = lookup_3D_to_2D_loc_and_directions(epi_pts[time_ii_stage], 
#                                                                                                                                  epi_pts[time_ii_stage].reshape(-1,3), 
#                                                                                                                                  flow_epi_3D_frame.reshape(-1,3), 
#                                                                                                                                  flow_epi_3D_frame.reshape(-1,3),
#                                                                                                                                  scale=10.)
#                                        centroid_2D_prinvec = centroid_2D_prinvec.reshape(flow_ve_frame.shape)
#
#                                        plt.figure(figsize=(20,20))
#                                        plt.subplot(121)
#                                        plt.imshow(vid_epi_resize[time_ii_stage], cmap='gray')
#                                        plt.quiver(XXX[::20,::20],
#                                                   YYY[::20,::20],
#                                                   flow_epi_frame[::20,::20,0],
#                                                   -flow_epi_frame[::20,::20,1], color='r')
#                                        plt.subplot(122)
#                                        plt.imshow(vid_epi_resize[time_ii_stage], cmap='gray')
#                                        plt.quiver(XXX[::20,::20],
#                                                   YYY[::20,::20],
#                                                   centroid_2D_prinvec[::20,::20,1],
#                                                   -centroid_2D_prinvec[::20,::20,0], color='r')
#                                        plt.show()

                                        all_flow_VE_3D.append(flow_ve_3D_frame)
                                        all_flow_Epi_3D.append(flow_epi_3D_frame)


                                        #### now we get the average displacements for each 'cell' in the cell mask :D
                                        cell_ve_flow_frame_3D = np.array([np.nanmean(flow_ve_3D_frame[cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])
                                        cell_epi_flow_frame_3D = np.array([np.nanmean(flow_epi_3D_frame[cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])
                                        
                                        # these are the stuff already been computed ..... 
                                        embryo_all_cell_VE_flow.append(cell_ve_flow_frame_3D)
                                        embryo_all_cell_Epi_flow.append(cell_epi_flow_frame_3D)


                                        # """
                                        # 3. get the Demons diff at the timepoint over all the cells. -> we don't need to care about the cell normals per say -> these are given in the tracks. 
                                        # """

                                        # if cnt < max_diff_cnt:

                                        #     cell_demons_flow_VE_AVE_dir = np.array([np.nanmean(demons_ve_diff_dir_vect_AVE[cnt][cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])
                                        #     cell_demons_flow_VE_AVE_perp_dir = np.array([np.nanmean(demons_ve_diff_dir_vect_AVE_perp[cnt][cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])
                                        #     cell_demons_flow_VE_AVE_norm_dir = np.array([np.nanmean(demons_ve_diff_dir_vect_AVE_norm[cnt][cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])
                                        #     cell_demons_flow_VE_AVE_surf_dir = np.array([np.nanmean(demons_ve_diff_surface[cnt][cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])

                                        #     cell_demons_flow_Epi_AVE_dir = np.array([np.nanmean(demons_epi_diff_dir_vect_AVE[cnt][cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])
                                        #     cell_demons_flow_Epi_AVE_perp_dir = np.array([np.nanmean(demons_epi_diff_dir_vect_AVE_perp[cnt][cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])
                                        #     cell_demons_flow_Epi_AVE_norm_dir = np.array([np.nanmean(demons_epi_diff_dir_vect_AVE_norm[cnt][cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])
                                        #     cell_demons_flow_Epi_AVE_surf_dir = np.array([np.nanmean(demons_epi_diff_surface[cnt][cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])

                                        #     # unlike for the flow above ... we only take this for the stage we have defined over .... 
                                        #     embryo_all_cell_demons_flow_VE_AVE_dir.append(cell_demons_flow_VE_AVE_dir)
                                        #     embryo_all_cell_demons_flow_VE_AVE_perp_dir.append(cell_demons_flow_VE_AVE_perp_dir)
                                        #     embryo_all_cell_demons_flow_VE_AVE_norm_dir.append(cell_demons_flow_VE_AVE_norm_dir)
                                        #     embryo_all_cell_demons_flow_VE_AVE_surf_dir.append(cell_demons_flow_VE_AVE_surf_dir)

                                        #     embryo_all_cell_demons_flow_Epi_AVE_dir.append(cell_demons_flow_Epi_AVE_dir)
                                        #     embryo_all_cell_demons_flow_Epi_AVE_perp_dir.append(cell_demons_flow_Epi_AVE_perp_dir)
                                        #     embryo_all_cell_demons_flow_Epi_AVE_norm_dir.append(cell_demons_flow_Epi_AVE_norm_dir)
                                        #     embryo_all_cell_demons_flow_Epi_AVE_surf_dir.append(cell_demons_flow_Epi_AVE_surf_dir)

                                        """
                                        4. get the curvature at the timepoint over all the cells. 
                                        """
                                        # curvature data operates in global time.!
                                        curvature_VE = curvature_obj['curvature_VE']
                                        curvature_Epi = curvature_obj['curvature_Epi']

                                        print(curvature_VE.shape, curvature_Epi.shape)
                                        print('+++++++++++++++++')
                                        # curvature_VE_time_ii = curvature_VE[time_ii_stage].copy()
                                        # curvature_Epi_time_ii = curvature_Epi[time_ii_stage].copy()

                                        # cell_ve_curvature_3D = np.array([np.nanmean(curvature_VE_time_ii[cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])
                                        # cell_epi_curvature_3D = np.array([np.nanmean(curvature_Epi_time_ii[cell_seg_img_frame==cell_id_frame], axis=0) for cell_id_frame in uniq_cells_frame])
                                        
                                        # # # these are the stuff already been computed ..... 
                                        # # embryo_all_cell_VE_flow.append(cell_ve_flow_frame_3D)
                                        # # embryo_all_cell_Epi_flow.append(cell_epi_flow_frame_3D)
                                        # embryo_all_cell_curvature_VE.append(cell_ve_curvature_3D)
                                        # embryo_all_cell_curvature_Epi.append(cell_epi_curvature_3D)

                                        cnt += 1 # update the counter accordingly. 


                            embryo_all_time = np.array(embryo_all_time) # give the full individual cell areas over time. 
                            embryo_all_cell_ids = np.array(embryo_all_cell_ids)
                            embryo_all_stage = np.array(embryo_all_stage)

                            polar_grids_static_coarse = np.array(polar_grids_static_coarse) # this is recomputed based entirely on the AVE migration angle + the epi contour at every time point!. 
    
                            all_flow_VE_3D = np.array(all_flow_VE_3D)
                            all_flow_Epi_3D = np.array(all_flow_Epi_3D)

                            # these are the stuff already been computed ..... 
                            embryo_all_cell_VE_flow = np.array(embryo_all_cell_VE_flow)
                            embryo_all_cell_Epi_flow = np.array(embryo_all_cell_Epi_flow)

                            # embryo_all_cell_demons_flow_VE_AVE_dir = np.array(embryo_all_cell_demons_flow_VE_AVE_dir)
                            # embryo_all_cell_demons_flow_VE_AVE_perp_dir = np.array(embryo_all_cell_demons_flow_VE_AVE_perp_dir)
                            # embryo_all_cell_demons_flow_VE_AVE_norm_dir = np.array(embryo_all_cell_demons_flow_VE_AVE_norm_dir)
                            # embryo_all_cell_demons_flow_VE_AVE_surf_dir = np.array(embryo_all_cell_demons_flow_VE_AVE_surf_dir)

                            # embryo_all_cell_demons_flow_Epi_AVE_dir = np.array(embryo_all_cell_demons_flow_Epi_AVE_dir)
                            # embryo_all_cell_demons_flow_Epi_AVE_perp_dir = np.array(embryo_all_cell_demons_flow_Epi_AVE_perp_dir)
                            # embryo_all_cell_demons_flow_Epi_AVE_norm_dir = np.array(embryo_all_cell_demons_flow_Epi_AVE_norm_dir)
                            # embryo_all_cell_demons_flow_Epi_AVE_surf_dir = np.array(embryo_all_cell_demons_flow_Epi_AVE_surf_dir)

                            # embryo_all_cell_curvature_VE = np.array(embryo_all_cell_curvature_VE)
                            # embryo_all_cell_curvature_Epi = np.array(embryo_all_cell_curvature_Epi)
                            
                            # analysis will follow after all exporting of statistics... 
                            savestatsmatfile = os.path.join(mastercellstatsfolder, 'cell_ve-epi-flow_3Ddemons-correct_'+unwrapped_condition+'.mat')
   
                            spio.savemat(savestatsmatfile, 
                                        {'embryo': unwrapped_condition, 
                                         'embryo_inversion':embryo_inversion,
                                         'cellimgfile': cell_file, 
                                         've_angle': ve_angle_move, 
                                         'time': np.hstack(embryo_all_time), 
                                         'stage': np.hstack(embryo_all_stage),
                                         'cell_ids': embryo_all_cell_ids, 
                                         'polar_grids_coarse_contour': np.array(polar_grids_static_coarse).astype(np.int),
                                         'VE_flow_3D': embryo_all_cell_VE_flow,
                                         'Epi_flow_3D': embryo_all_cell_Epi_flow,
                                         'cell_flow_VE_3D': embryo_all_cell_VE_flow,
                                         'cell_flow_Epi_3D': embryo_all_cell_Epi_flow,
                                         'epi_contour_line': epi_boundary_time
                                         # 'cell_demons_flow_VE_AVE_dir': embryo_all_cell_demons_flow_VE_AVE_dir,
                                         # 'cell_demons_flow_VE_AVE_perp_dir': embryo_all_cell_demons_flow_VE_AVE_perp_dir,
                                         # 'cell_demons_flow_VE_AVE_norm_dir': embryo_all_cell_demons_flow_VE_AVE_norm_dir,
                                         # 'cell_demons_flow_VE_surface': embryo_all_cell_demons_flow_VE_AVE_surf_dir, 
                                         # 'cell_demons_flow_Epi_AVE_dir': embryo_all_cell_demons_flow_Epi_AVE_dir,
                                         # 'cell_demons_flow_Epi_AVE_perp_dir': embryo_all_cell_demons_flow_Epi_AVE_perp_dir, 
                                         # 'cell_demons_flow_Epi_AVE_norm_dir': embryo_all_cell_demons_flow_Epi_AVE_norm_dir,
                                         # 'cell_demons_flow_Epi_surface': embryo_all_cell_demons_flow_Epi_AVE_surf_dir,
                                         # 'cell_curvature_VE': embryo_all_cell_curvature_VE,
                                         # 'cell_curvature_Epi': embryo_all_cell_curvature_Epi
                                         })
   
   
#    fig, ax = plt.subplots(figsize=(15,15))
#    ax.imshow(curvature_VE[:,330,:].T, cmap='coolwarm', vmin=-2.5e-4, vmax=2.5e-4)
#    ax.set_aspect('auto')
#    plt.show()

    