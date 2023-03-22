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


def poly_perimeter_proj(poly, normal):

    # normal is unit length. 
    N  = len(poly)

    dist = 0

    for i in range(N-1):
        vi1 = poly[i]
        vi2 = poly[i+1]
	
	vi12 = vi2-vi1
	vi12_proj = vi12 - np.dot(vi12, normal)*normal
        vi1_vi2_dist = np.linalg.norm(vi12_proj)
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
    cell_perimeters_3D_proj = []
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
        surf_area = poly_area(reg_boundary_3D_refine, normal=surf_normal) # we just use the one 
        surf_perimeter = poly_perimeter(reg_boundary_3D_refine)
	surf_perimeter_proj = poly_perimeter_proj(reg_boundary_3D_refine, normal=surf_normal)
        
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
	cell_perimeters_3D_proj.append(surf_perimeter_proj)
        cell_centroids_2D.append(np.hstack([centroid_2D_ind_y, centroid_2D_ind_x]))
        cell_major_minor_lengths.append(np.hstack([major_length, minor_length]))
        cell_major_minor_axes.append(np.vstack([major_axis,
                                                minor_axis]))

    cell_areas_3D = np.hstack(cell_areas_3D)
    cell_perimeters_3D = np.hstack(cell_perimeters_3D)
    cell_perimeters_3D_proj = np.hstack(cell_perimeters_3D_proj)
    cell_centroids_2D = np.vstack(cell_centroids_2D)
    cell_major_minor_lengths = np.vstack(cell_major_minor_lengths)
    cell_major_minor_axes = np.array(cell_major_minor_axes)
    
#    fig, ax = plt.subplots()
#    ax.imshow(cell_img)
#    ax.plot(cell_centroids_2D[:,1], 
#            cell_centroids_2D[:,0], 'w.')
#    plt.show()

    return uniq_cells, (cell_areas_3D, cell_perimeters_3D, cell_perimeters_3D_proj, cell_major_minor_lengths, cell_major_minor_axes)


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
    mastersavefolder = '/media/felix/My Passport/Shankar-2/All_Results/Demons_Analysis_Paper-Full-Rec' # this is the full reconstruction for the surface from the binary
    masterannotfolder = '/media/felix/My Passport/Shankar-2/All_Manual_Annotations/all_cell_annotations' # there will only be one of these therefore we should name these accordingly. 
    
    
    """
    Set up a master save folder to dump all the statistics to 
    """
    mastercellstatsfolder = '/media/felix/My Passport/Shankar-2/All_Results/Single_Cell_Shape_Statistics_All/Data'
    fio.mkdir(mastercellstatsfolder)

    """
    define some save folders. 
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
                #       Load the computed statistics.      
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
                            #       Set up the saves for individual cell area computation of all cells. How to handle cell area  
                            # =============================================================================     
    
    
                            # set up the saves for the embryo ... for the whole time -> then we only need to have one save file? -> allowing more flexibility ?
    
                            # this is here as a check all timepoints that we want have been incorporated ... ssss
                            embryo_all_time = [] # give the full individual cell areas over time. 
                            embryo_all_cell_ids = []
                            embryo_all_stage = []
    
                            embryo_all_areas = []
                            embryo_all_perims = []
                            embryo_all_perims_proj = []
                            embryo_all_shape_index = [] # ? https://jcs.biologists.org/content/129/18/3375 , perimeter/root(area) = 3.81 is the critical point.
                            embryo_all_shape_index_proj = []
                            
                            embryo_all_major_lens = []
                            embryo_all_minor_lens = []
                            embryo_all_eccentricity = []
    
                            # just to show the orientation of the cell in the sheet. 
                            embryo_all_major_axis_3D = []
                            embryo_all_minor_axis_3D = []
                            embryo_all_major_axis_2D_polar = [] 
                            embryo_all_minor_axis_2D_polar = []  
    
                            embryo_all_centroids_2D = []
                            embryo_all_centroids_3D = []
    
                            embryo_all_cell_neighbor_list_ids = []
                            polar_grids_dynamic_coarse = [] # save the inferred coarse grids?                       
    
                            embryo_all_cell_thickness = []
    
                            # =============================================================================
                            #       Iterate over the different stages, get the times, get the deformation changes. 
                            # =============================================================================     
                            # analyse_stages = ['Pre-migration', 
                            #                   'Migration to Boundary',
                            #                   'Post-Boundary']
                            # stage_embryo = [pre, mig, post]
    
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
                                    for time_ii_stage in np.arange(start,end)[:]:
                                        
                                        # append the time. 
                                        embryo_all_time.append(time_ii_stage)
                                        embryo_all_stage.append(analyse_stage)
    
                                        # the iterations seem quite fast. 
                                        print('computing time, %d' %(time_ii_stage))
    
                                        # compute the surface normals necessary to get the individual cell area. 
                                        direction_r_3D_ve, direction_theta_3D_ve, direction_normal_3D_ve = meshtools.build_full_polar_tangent_and_normal_unit_vectors(ve_pts_stage[time_ii_stage-start],                                                           
                                                                                                                                                                r_dist=10, 
                                                                                                                                                                theta_dist=5, 
                                                                                                                                                                smooth=True, 
                                                                                                                                                                smooth_theta_dist=10, 
                                                                                                                                                                smooth_radial_dist=20, 
                                                                                                                                                                smooth_tip_sigma=15, 
                                                                                                                                                                smooth_tip_radius=35) # also apply smoothing of 3 degrees.
                                        
                                        # compute for each cell id their respective cell areas. / perimeters, using valid coordinates. for each time point. 
                                        # this computes for all cells in the image 
                                        cell_ids_frame, (cell_areas_frame, cell_perims_frame, cell_perims_frame_proj, cell_major_minor_lens, cell_major_minor_axes) = compute_cell_props(cell_seg_img[time_ii_stage], 
                                                                                                                                                                 ve_pts_stage[time_ii_stage-start], 
                                                                                                                                                                 invalid_mask=invalid_mask[time_ii_stage-start],
                                                                                                                                                                 unwrap_params_normals=direction_normal_3D_ve)
    
                                        cell_shape_index_frame = cell_perims_frame / (np.sqrt(cell_areas_frame +1e-8))
                                        cell_shape_index_frame_proj = cell_perims_frame_proj / (np.sqrt(cell_areas_frame +1e-8)) # this should be more accurate from the way we have measured this. 
                                    
                                    
                                        thickness_ve_epi = ve_pts_stage[time_ii_stage-start] - epi_pts_stage[time_ii_stage-start]
                                        thickness_ve_epi = np.linalg.norm(thickness_ve_epi, axis=-1)
                                        cell_thickness_frame = np.hstack([np.nanmean(thickness_ve_epi[cell_seg_img[time_ii_stage]==cell_id]) for cell_id in cell_ids_frame])
                                        embryo_all_cell_thickness.append(cell_thickness_frame)
                                    
                                        embryo_all_cell_ids.append(cell_ids_frame)
                                        embryo_all_areas.append(cell_areas_frame)
                                        embryo_all_perims.append(cell_perims_frame)
                                        embryo_all_perims_proj.append(cell_perims_frame_proj) # doesn't make much difference. 
                                        embryo_all_shape_index.append( cell_shape_index_frame )
                                        embryo_all_shape_index_proj.append( cell_shape_index_frame_proj )
    
                                        embryo_all_major_lens.append(cell_major_minor_lens[:,0])
                                        embryo_all_minor_lens.append(cell_major_minor_lens[:,1])
    
                                        # need to project these directions ! -> see below !. 
                                        embryo_all_major_axis_3D.append(cell_major_minor_axes[:,0]) # these should all be unit normals (with some orientation / direction)
                                        embryo_all_minor_axis_3D.append(cell_major_minor_axes[:,1])
                                        
                                        embryo_all_eccentricity.append(cell_major_minor_lens[:,0]/(cell_major_minor_lens[:,1]+.1)) # give the major minor length ratio as a measure of eccentricity. 
    
    
                                        """
                                        here we get the centroids_3D for the cell area as well as map it back to the 2D, given the 3D surface points.
                                        """
                                        cells_centroids_2D, cells_centroids_3D = lookup_2D_region_centroids_in_3D(cell_seg_img[time_ii_stage], 
                                                                                                                  ve_pts_stage[time_ii_stage-start])
    
                                        embryo_all_centroids_2D.append(cells_centroids_2D) # note these can vary over time. 
                                        embryo_all_centroids_3D.append(cells_centroids_3D) 
                                        
                                        
    #                                    # check the 3D axis. 
    #                                    fig = plt.figure(figsize=(10,10))
    #                                    ax = fig.add_subplot(111, projection='3d')
    #                                    ax.set_aspect('equal')
    #                
    #                                    # we need to revert these properly !.     
    #                                    ax.scatter(cells_centroids_3D[:,1], 
    #                                               cells_centroids_3D[:,2],
    #                                               cells_centroids_3D[:,0], c='k')
    #                                    
    #                                    # plot the quiver arrows. 
    #                                    ax.quiver(cells_centroids_3D[:,1], 
    #                                              cells_centroids_3D[:,2],
    #                                              cells_centroids_3D[:,0], 
    #                                              cell_major_minor_axes[:,0][:, 1],
    #                                              cell_major_minor_axes[:,0][:, 2], 
    #                                              cell_major_minor_axes[:,0][:, 0], length=50., 
    #                                               normalize=True, color='r')
    #                                    
    #                                    ax.quiver(cells_centroids_3D[:,1], 
    #                                              cells_centroids_3D[:,2],
    #                                              cells_centroids_3D[:,0], 
    #                                              cell_major_minor_axes[:,1][:, 1],
    #                                              cell_major_minor_axes[:,1][:, 2], 
    #                                              cell_major_minor_axes[:,1][:, 0], length=50., 
    #                                               normalize=True, color='g')
    #                                    
    #                                    tra3Dtools.set_axes_equal(ax)
    #                                    plt.show()  
                                        
                                        
                                        """
                                        lookup the major minor axis directions back in the expected polar view. 
                                        """  
                                        major_axis_2D = pullback_3D_directions(cells_centroids_3D, cell_major_minor_axes[:,0], ve_pts_stage[time_ii_stage-start], disp_dist=10)
                                        minor_axis_2D = pullback_3D_directions(cells_centroids_3D, cell_major_minor_axes[:,1], ve_pts_stage[time_ii_stage-start], disp_dist=10)
    
                                        embryo_all_major_axis_2D_polar.append(major_axis_2D)
                                        embryo_all_minor_axis_2D_polar.append(minor_axis_2D)
                                        
                                        
                                        
                                        # optional checkplot of the inferred 2D angles. using quiver plot. 
                                        
                                        fig, ax = plt.subplots(figsize=(15,15))
                                        ax.imshow(cell_seg_img[time_ii_stage])
                                        ax.quiver(cells_centroids_2D[:,1], 
                                                  cells_centroids_2D[:,0], 
                                                  major_axis_2D[:,1], 
                                                  -major_axis_2D[:,0])
#                                        ax.quiver(cells_centroids_2D[:,1], 
#                                                  cells_centroids_2D[:,0], 
#                                                  minor_axis_2D[:,1], 
#                                                  -minor_axis_2D[:,0], color='r')
                                        plt.show()
                                        
                                        """
                                        determine neighboring for all valid cells through area overlap. # check for validity? (how much to dilate?)  
                                        """
                                        
                                        # settings for L491? -> not sure how well this is for the very thing cases?
                                        
                                        # might be best to iterate this depending on the particular cell images dilation / closeness. 
                                        
                                        # still missing some? 
                                        
                                        # first do optional dealiasing? 
                                        cell_seg_img_frame_dealias = grow_polar_regions_nearest(cell_seg_img[time_ii_stage], 
                                                                                                ve_pts_stage[time_ii_stage-start], 
                                                                                                dilate=5, metric='manhattan', debug=False)
                                        
                                        # think this works now. 
                                        cell_neighbor_list_ids = get_neighbor_cell_ids(cell_seg_img_frame_dealias,
                                                                                       cell_ids_frame, 
                                                                                       dilate_size=3, 
                                                                                       dilate_elem=skmorph.square,
                                                                                       min_count=5) # check all other cells that are neighbors. to get a handle on the neighbor exchange.  
    
    #                                    test__ = np.random.randint(len(cell_neighbor_list_ids))
    #                                    
    #                                    plt.figure()
    #                                    plt.imshow(cell_seg_img[time_ii_stage] == cell_ids_frame[test__])
    #                                    plt.show()
    #                                    
    #                                    plt.figure()
    #                                    plt.imshow(get_segmentation_mask(cell_seg_img[time_ii_stage], 
    #                                                                     cell_neighbor_list_ids[test__]))
    #                                    plt.imshow(cell_seg_img[time_ii_stage] == cell_ids_frame[test__], alpha=.5)
    #                                    plt.show()
                                        
                                        embryo_all_cell_neighbor_list_ids.append(cell_neighbor_list_ids)
                                        
                                        """
                                        here we compute the dynamic polar grid for doing the computations in this regimen. 
                                        """
                                        polar_grid_time_ii = polar_grid_mask_coarse_from_fine(squares_ve_stage[time_ii_stage-start], 
                                                                                              polar_grid_fine_2_coarse_ids, 
                                                                                              polar_grid_ids=uniq_polar_regions, 
                                                                                              shape=polar_grid.shape)
                                        polar_grids_dynamic_coarse.append(polar_grid_time_ii)
    
    
                            # analysis will follow after all exporting of statistics... 
                            savestatsmatfile = os.path.join(mastercellstatsfolder, 'cell_stats_'+unwrapped_condition+'.mat')
    
                            spio.savemat(savestatsmatfile, 
                                        {'embryo': unwrapped_condition, 
                                         'embryo_inversion':embryo_inversion,
                                         'cellimgfile': cell_file, 
                                         've_angle': ve_angle_move, 
                                         'time': np.hstack(embryo_all_time), 
                                         'stage': np.hstack(embryo_all_stage),
                                         'cell_ids': embryo_all_cell_ids, 
                                         'cell_areas': embryo_all_areas, 
                                          'cell_perims': embryo_all_perims,
                                          'cell_perims_proj': embryo_all_perims_proj, 
                                          'cell_shape_index': embryo_all_shape_index, 
                                          'cell_shape_index_proj': embryo_all_shape_index_proj, 
                                          'cell_thickness': embryo_all_cell_thickness, 
                                          'cell_major_length': embryo_all_major_lens,
                                          'cell_minor_length': embryo_all_minor_lens,
                                          'cell_eccentricity': embryo_all_eccentricity,
                                          'cell_major_axis_2D': embryo_all_major_axis_2D_polar, 
                                          'cell_minor_axis_2D': embryo_all_minor_axis_2D_polar,
                                          'cell_major_axis_3D': embryo_all_major_axis_3D,
                                          'cell_minor_axis_3D': embryo_all_minor_axis_3D,
                                          'cell_centroids_2D': embryo_all_centroids_2D,
                                          'cell_centroids_3D': embryo_all_centroids_3D,
                                          'polar_grids_coarse': np.array(polar_grids_dynamic_coarse).astype(np.int), 
                                          'cell_neighbors': embryo_all_cell_neighbor_list_ids})

#                            """
#                            now we first plot and then check out the ellipsity ? -> is there a way to compute this fit accurately? 
#                            
#                            how to best do this ?
#                            -> iterate densely by s or by z? ?
#                            by s is the only way to ensure registered comparison?
#                            
#                            # do it by distance then.
#                            """
#                            YY,XX = np.indices(vid_ve[0].shape)
#                            dist_r = np.sqrt( (YY - vid_ve[0].shape[0]/2.)**2 + (XX - vid_ve[1].shape[0]/2.)**2)
#                            
#                            r_max_2D = int(np.hstack(vid_ve[0].shape)[0]/2./1.2)
#                            s_dist_range = np.linspace(0, r_max_2D, r_max_2D+1)
#                            ds = s_dist_range[1] - s_dist_range[0] # for radial quadrant it might be best to do it according to epi? 
#                            
#                            """
#                            iterate over each possible r and get the possible pts to fit.? 
#                            is 1 too fine? 
#                            """
#                            valid_ve_pts_3D = ve_pts_stage[:,dist_r<r_max_2D]
#                            valid_epi_pts_3D = epi_pts_stage[:,dist_r<r_max_2D]
#                            
#                            embryo_ve_centres_3D = valid_ve_pts_3D.mean(axis=1)
#                            embryo_epi_centres_3D = valid_epi_pts_3D.mean(axis=1)
#                            
#                            
#                            # ok this can be parallelised 
#                            ve_eccentricity = []
#                            epi_eccentricity = []
#                            ve_eccentricity_image = np.zeros(np.hstack([len(ve_pts_stage), vid_ve[0].shape])); ve_eccentricity_image[:] = np.nan
#                            epi_eccentricity_image = np.zeros(np.hstack([len(epi_pts_stage), vid_ve[0].shape])); epi_eccentricity_image[:] = np.nan
#                            
#                            for dd in tqdm(range(len(s_dist_range)-1)): 
#                                
#                                select_s = np.logical_and(dist_r>=s_dist_range[dd]-.5*ds, 
#                                                          dist_r<=s_dist_range[dd]+.5*ds)
#                                
#                                select_pts_s_ve = ve_pts_stage[:,select_s].copy() # this is temporal 
#                                select_pts_s_epi = epi_pts_stage[:,select_s].copy() 
#                                
#                                """
#                                attempt 3D ellipse fitting and orientation # first we project. 
#                                """
#                                ve_eccentricity_dd = []
#                                epi_eccentricity_dd = []
#                                
#                                if len(select_pts_s_ve) > 0:
#                                
#                                    if select_pts_s_ve.shape[1] > 3: 
#                                        
#                                        # should be parallelisable. 
#                                        for frame in range(len(select_pts_s_ve)):
#                                            select_pts_s_ve_frame = select_pts_s_ve[frame]
#                                            
#                                            """
#                                            fit ellipse 3D 
#                                            """
#                                            major_len_ve, minor_len_ve, major_pt_3D_ve = fit_ellipse_3D(select_pts_s_ve_frame)
#                                            
#                                            """
#                                            get the orientation.
#                                            """
#                                            dir_pt_3D_ve = major_pt_3D_ve -  embryo_ve_centres_3D[frame]  
#                                            orientation_ve = np.arctan2(dir_pt_3D_ve[2],  dir_pt_3D_ve[1]) # in the same format -> how does this compare now to that of the unwrap_ve_angle?
#                                            
#                                            ve_eccentricity_dd.append([major_len_ve, minor_len_ve, major_len_ve/(minor_len_ve+1e-8), orientation_ve])
#                                    
#                                            ve_eccentricity_image[frame,select_s] = major_len_ve/(minor_len_ve+1e-8)
#                                    else:
#                                        ve_eccentricity_dd = np.zeros(( len(ve_pts_stage), 4))
#                                        ve_eccentricity_dd[:] = np.nan
#                                else:
#                                    ve_eccentricity_dd = np.zeros(( len(ve_pts_stage), 4))
#                                    ve_eccentricity_dd[:] = np.nan
#                                        
#                                if len(select_pts_s_epi) > 0:
#                                    if select_pts_s_epi.shape[1] > 3: 
#                                        for frame in range(len(select_pts_s_epi)):
#                                            select_pts_s_epi_frame = select_pts_s_epi[frame]
#                                            
#                                            """
#                                            fit ellipse 3D 
#                                            """
#                                            major_len_epi, minor_len_epi, major_pt_3D_epi = fit_ellipse_3D(select_pts_s_epi_frame)
#                                            
#                                            """
#                                            get the orientation.
#                                            """
#                                            dir_pt_3D_epi = major_pt_3D_epi -  embryo_epi_centres_3D[frame]  
#                                            orientation_epi = np.arctan2(dir_pt_3D_epi[2],  dir_pt_3D_epi[1]) # in the same format -> how does this compare now to that of the unwrap_ve_angle?
#                                            
#                                            epi_eccentricity_dd.append([major_len_epi, minor_len_epi, major_len_epi/(minor_len_epi+1e-8), orientation_epi])
#                                            
#                                            epi_eccentricity_image[frame,select_s] = major_len_epi/(minor_len_epi+1e-8)
#                                    else: 
#                                        epi_eccentricity_dd = np.zeros(( len(epi_pts_stage), 4))
#                                        epi_eccentricity_dd[:] = np.nan
#                                else: 
#                                    epi_eccentricity_dd = np.zeros(( len(epi_pts_stage), 4))
#                                    epi_eccentricity_dd[:] = np.nan
#                                    
#                                        
#                                ve_eccentricity.append(ve_eccentricity_dd)
#                                epi_eccentricity.append(epi_eccentricity_dd)
#                                    
#                            ve_eccentricity = np.array(ve_eccentricity)
#                            epi_eccentricity = np.array(epi_eccentricity)
#                                
#                            ve_eccentricity = ve_eccentricity.transpose(1,0,2)
#                            epi_eccentricity = epi_eccentricity.transpose(1,0,2)
#                            
#                            
#                            
#                            """
#                            set up the 'static' vector and check with the orientation vector. 
#                            """
#                            # what to build it for all corners / points of the squares over time.
#                            ve_angle_vector = np.hstack([np.sin((90+ve_angle)/180. * np.pi), 
#                                                         np.cos((90+ve_angle)/180. * np.pi)])
#            
#                            ve_norm_angle_vector = np.hstack([-ve_angle_vector[1], 
#                                                              ve_angle_vector[0]])
#        
#        
#                            ecc_axis_angle_ve = ve_eccentricity[0,ve_eccentricity.shape[1]//2,-1]
#                            ecc_axis_angle_ve_vector = np.hstack([np.sin(ecc_axis_angle_ve), 
#                                                                  np.cos(ecc_axis_angle_ve)])
#                            
#                            
#                            """
#                            Do the saving. 
#                            """
#                            # we can leave the polar grid? 
#                            # save all the outputs for future computation.
#                            
#                            newmastersavefolder = '/media/felix/My Passport/Shankar-2/Demons_remap_eccentricity_analysis'
#                            oldmastersavefolder = os.path.split(embryo_folder)[0]
#                            
#                            saverootfolder, savesuffix = os.path.split(savefile)
#                            newsaverootfolder = saverootfolder.replace(oldmastersavefolder,
#                                                                        newmastersavefolder);
#                            fio.mkdir(newsaverootfolder)
#                            
#                            savefile = os.path.join(newsaverootfolder, savesuffix)
#                            
#                            print('saving ... ', savefile)
#                            spio.savemat(savefile, 
#                                         {'ve_pts': ve_pts_stage.astype(np.float32),
#                                          'epi_pts': epi_pts_stage.astype(np.float32), 
#                                          'start_end_times':start_end_times, 
#                                          've_eccentricity': ve_eccentricity.astype(np.float32), 
#                                          'epi_eccentricity': epi_eccentricity.astype(np.float32), 
#                                          'analyse_stage': analyse_stages[stage_ii], 
#                                          've_angle_vector': ve_angle_vector, 
#                                          've_norm_angle_vector': ve_norm_angle_vector,
#                                          've_angle': (90+ve_angle)/180. * np.pi, # this is to have it in the same coordinate system. 
#                                          'embryo_inversion':embryo_inversion,
#                                          'embryo_folder':embryo_folder})
#                            
#                            
#                            fig, ax = plt.subplots()
#                            ax.imshow(ve_eccentricity[...,2], cmap='coolwarm', vmin=.8, vmax=1.2)
#                            ax.set_aspect('auto')
#                            
#                            fig, ax = plt.subplots()
#                            ax.imshow(epi_eccentricity[...,2], cmap='coolwarm', vmin=.8, vmax=1.2)
#                            ax.set_aspect('auto')
#                            
#                            fig, ax = plt.subplots()
#                            ax.plot(ve_eccentricity[...,3], 
#                                    ve_eccentricity[...,2],'o')
#                            plt.ylim([.8,1.2])
#                            plt.show()
#                            
#                            """
#                            set up the 'static' vector and check with the orientation vector. 
#                            """
##                            # what to build it for all corners / points of the squares over time.
##                            ve_angle_vector = np.hstack([np.sin((90+ve_angle)/180. * np.pi), 
##                                                         np.cos((90+ve_angle)/180. * np.pi)])
##            
##                            ve_norm_angle_vector = np.hstack([-ve_angle_vector[1], 
##                                                              ve_angle_vector[0]])
##        
##        
##                            ecc_axis_angle_ve = ve_eccentricity[0,len(ve_eccentricity)//2,-1]
##                            ecc_axis_angle_ve_vector = np.hstack([np.sin(ecc_axis_angle_ve), 
##                                                                  np.cos(ecc_axis_angle_ve)])
#                        
#                        
#                            # checking by projecting the cross secton. 
#                        
#                            fig, ax = plt.subplots()
#                            select_s_test = np.logical_and(dist_r>=s_dist_range[len(s_dist_range)//2]-.5*ds, 
#                                                           dist_r<=s_dist_range[len(s_dist_range)//2]+.5*ds)
#                            pts_select = ve_pts_stage[0, select_s_test]
#                            ax.plot(pts_select[...,2], 
#                                    pts_select[...,1], 'o' )
#                            ax.quiver(pts_select[...,2].mean(), 
#                                      pts_select[...,1].mean(),
#                                      ecc_axis_angle_ve_vector[0], 
#                                      ecc_axis_angle_ve_vector[1], color='k')
#                            ax.set_aspect(1)
#                            plt.show()
#                        
#                            
#                            print('AVE angle vector 2D', ve_angle_vector)
#                            
#                            im_center = np.hstack([vid_ve[0].shape[1]/2., vid_ve[0].shape[0]/2.])
#                            fig, ax = plt.subplots(figsize=(10,10))
#                            plt.imshow(vid_ve.max(axis=0), cmap='gray')
#                            ax.quiver(im_center[1], 
#                                      im_center[0],
#                                      ve_angle_vector[0], 
#                                      -ve_angle_vector[1], color='r')
#                            ax.quiver(im_center[1], 
#                                      im_center[0],
#                                      ve_norm_angle_vector[0], 
#                                      -ve_norm_angle_vector[1], color='g')
#                            ax.quiver(im_center[1], 
#                                      im_center[0],
#                                      ecc_axis_angle_ve_vector[0], 
#                                      -ecc_axis_angle_ve_vector[1], color='k')
#                            plt.show()

#
#                        
#                            """
#                            how fast is this do we need to go to subsampling? 
#                            """
#                            ve_dir_vect_AVE = []
#                            ve_dir_vect_AVE_perp = []
#                            epi_dir_vect_AVE = []
#                            epi_dir_vect_AVE_perp = []
#                            
#                            for frame_ii in tqdm(np.arange(nframes)[:]):
#                            
#                                # step 1: build neighbor model at every single time.... . 
#                                surf_nbrs_ve, surf_nbrs_ve_3D_pts = tra3Dtools.learn_nearest_neighbor_surf_embedding_model(ve_pts_stage[frame_ii], 
#                                                                                                                       nearest_K=1)
#                                surf_nbrs_epi, surf_nbrs_epi_3D_pts = tra3Dtools.learn_nearest_neighbor_surf_embedding_model(epi_pts_stage[frame_ii], 
#                                                                                                                       nearest_K=1)
#                    
#                                # step 2: construct the expected 3D directions, projecting the 2D onto surface for both VE and Epi 'sparse is better?
#                                # NOTE: ve_angle_vector is (x,y) we want (y,x)     
#                                # this is already reshaped? 
#                                AVE_surf_dir_ve_full = tra3Dtools.proj_2D_direction_3D_surface(ve_pts_stage[frame_ii], 
#                                                                                          ve_angle_vector[::-1][None,None,:], #(x,y)
#                                                                                          pt_array_time_2D = None,#(x,y)
#                                                                                          K_neighbors=1, 
#                                                                                          nbr_model=surf_nbrs_ve, 
#                                                                                          nbr_model_pts=surf_nbrs_ve_3D_pts, 
#                                                                                          dist_sample=10, 
#                                                                                          return_dist_model=False,
#                                                                                          mode='dense',
#                                                                                          smooth=True,
#                                                                                          r_dist=10, 
#                                                                                            smooth_theta_dist=20, 
#                                                                                            smooth_radial_dist=20, 
#                                                                                            smooth_tip_sigma=None, 
#                                                                                            smooth_tip_radius=35)
#                            
#                                AVE_surf_dir_ve_norm_full = tra3Dtools.proj_2D_direction_3D_surface(ve_pts_stage[frame_ii], 
#                                                                                  ve_norm_angle_vector[::-1][None,None,:], 
#                                                                                  pt_array_time_2D = None,
#                                                                                  K_neighbors=1, 
#                                                                                  nbr_model=surf_nbrs_ve, 
#                                                                                  nbr_model_pts=surf_nbrs_ve_3D_pts, 
#                                                                                  dist_sample=10, 
#                                                                                  return_dist_model=False,
#                                                                                  mode='dense',
#                                                                                    smooth=True,
#                                                                                    r_dist=10, 
#                                                                                    smooth_theta_dist=20, 
#                                                                                    smooth_radial_dist=20, 
#                                                                                    smooth_tip_sigma=None, 
#                                                                                    smooth_tip_radius=35)
#
#                                """
#                                Construct for Epi
#                                """                                        
#                                AVE_surf_dir_epi_full = tra3Dtools.proj_2D_direction_3D_surface(epi_pts_stage[frame_ii], 
#                                                                                          ve_angle_vector[::-1][None,None,:], 
#                                                                                          pt_array_time_2D = None,
#                                                                                          K_neighbors=1, 
#                                                                                          nbr_model=surf_nbrs_epi, 
#                                                                                          nbr_model_pts=surf_nbrs_epi_3D_pts, 
#                                                                                          dist_sample=10, 
#                                                                                          return_dist_model=False,
#                                                                                          mode='dense',
#                                                                                            smooth=True,
#                                                                                            r_dist=10, 
#                                                                                            smooth_theta_dist=20, 
#                                                                                            smooth_radial_dist=20, 
#                                                                                            smooth_tip_sigma=None, 
#                                                                                            smooth_tip_radius=35)
#                                
#        #                      
#                                AVE_surf_dir_epi_norm_full = tra3Dtools.proj_2D_direction_3D_surface(epi_pts_stage[frame_ii], 
#                                                                                              ve_norm_angle_vector[::-1][None,None,:], 
#                                                                                              pt_array_time_2D = None,
#                                                                                              K_neighbors=1, 
#                                                                                              nbr_model=surf_nbrs_epi, 
#                                                                                              nbr_model_pts=surf_nbrs_epi_3D_pts, 
#                                                                                              dist_sample=10, 
#                                                                                              return_dist_model=False,
#                                                                                              mode='dense',
#                                                                                            smooth=True,
#                                                                                            r_dist=10, 
#                                                                                            smooth_theta_dist=20, 
#                                                                                            smooth_radial_dist=20, 
#                                                                                            smooth_tip_sigma=None, 
#                                                                                            smooth_tip_radius=35)
                                
                                
#                                plt.figure()
#                                plt.subplot(131)
#                                plt.imshow(AVE_surf_dir_ve_full[...,0])
#                                plt.subplot(132)
#                                plt.imshow(AVE_surf_dir_ve_full[...,1])
#                                plt.subplot(133)
#                                plt.imshow(AVE_surf_dir_ve_full[...,2])
#                                plt.show()
#                                
#                                plt.figure()
#                                plt.subplot(131)
#                                plt.imshow(AVE_surf_dir_ve_norm_full[...,0])
#                                plt.subplot(132)
#                                plt.imshow(AVE_surf_dir_ve_norm_full[...,1])
#                                plt.subplot(133)
#                                plt.imshow(AVE_surf_dir_ve_norm_full[...,2])
#                                plt.show()
#                                print(AVE_surf_dir_epi_norm_full.shape)
#                                
#                                
#                                fig = plt.figure(figsize=(15,15))
#                                ax = fig.add_subplot(111, projection='3d')
#                                ax.set_aspect('equal')
#            
#                                # we need to revert these properly !.     
#                                ax.scatter(ve_pts_stage[frame_ii, ::40 ,::40, 0], 
#                                           ve_pts_stage[frame_ii,::40 ,::40, 2],
#                                           ve_pts_stage[frame_ii,::40 ,::40, 1], c='g')
#                                
#                                # plot the quiver arrows. 
#                                ax.quiver(ve_pts_stage[frame_ii,::40 ,::40, 0], 
#                                           ve_pts_stage[frame_ii,::40 ,::40, 2],
#                                           ve_pts_stage[frame_ii,::40 ,::40, 1], 
#                                           AVE_surf_dir_ve_full[::40, ::40, 0],
#                                           AVE_surf_dir_ve_full[::40, ::40, 2], 
#                                           AVE_surf_dir_ve_full[::40, ::40, 1], length=50., 
#                                           normalize=True, color='r')
#                                
#                                ax.quiver(ve_pts_stage[test_time,::40 ,::40, 0], 
#                                           ve_pts_stage[test_time,::40 ,::40, 2],
#                                           ve_pts_stage[test_time,::40 ,::40, 1], 
#                                           AVE_surf_dir_ve_norm_full[::40, ::40, 0],
#                                           AVE_surf_dir_ve_norm_full[::40, ::40, 2], 
#                                           AVE_surf_dir_ve_norm_full[::40, ::40, 1], length=50., 
#                                           normalize=True, color='k')
#                                
#                                tra3Dtools.set_axes_equal(ax)
#                                plt.show()  
#                                
#                                ve_dir_vect_AVE.append(AVE_surf_dir_ve_full)
#                                ve_dir_vect_AVE_perp.append(AVE_surf_dir_ve_norm_full)
#                                epi_dir_vect_AVE.append(AVE_surf_dir_epi_full)
#                                epi_dir_vect_AVE_perp.append(AVE_surf_dir_epi_norm_full)
#                                
#                            ve_dir_vect_AVE = np.array(ve_dir_vect_AVE)
#                            ve_dir_vect_AVE_perp = np.array(ve_dir_vect_AVE_perp)
#                            epi_dir_vect_AVE = np.array(epi_dir_vect_AVE)
#                            epi_dir_vect_AVE_perp = np.array(epi_dir_vect_AVE_perp)
#                            
#                            
#                            # we can leave the polar grid? 
#                            # save all the outputs for future computation.
#                            
#                            newmastersavefolder = '/media/felix/My Passport/Shankar-2/Demons_remap_vector'
#                            oldmastersavefolder = os.path.split(embryo_folder)[0]
#                            
#                            saverootfolder, savesuffix = os.path.split(savefile)
#                            newsaverootfolder = saverootfolder.replace(oldmastersavefolder,
#                                                                           newmastersavefolder);
#                            fio.mkdir(newsaverootfolder)
#                            
#                            savefile = os.path.join(newsaverootfolder, savesuffix)
#                            
#                            print('saving ... ', savefile)
#                            spio.savemat(savefile, 
#                                         {'ve_pts': ve_pts_stage.astype(np.float32),
#                                          'epi_pts': epi_pts_stage.astype(np.float32), 
#                                          'start_end_times':start_end_times, 
#                                          've_diff_pts': ve_diff_pts_stage.astype(np.float32), 
#                                          'epi_diff_pts': epi_diff_pts_stage.astype(np.float32), 
#                                          'analyse_stage': analyse_stages[stage_ii], 
#                                          've_dir_vect_AVE':ve_dir_vect_AVE.astype(np.float32),
#                                          've_dir_vect_AVE_perp':ve_dir_vect_AVE_perp.astype(np.float32),
#                                          'epi_dir_vect_AVE':epi_dir_vect_AVE.astype(np.float32),
#                                          'epi_dir_vect_AVE_perp':epi_dir_vect_AVE_perp.astype(np.float32),
#                                          've_angle_vector':ve_angle_vector, 
#                                          've_norm_angle_vector':ve_norm_angle_vector,
#                                          'embryo_inversion':embryo_inversion})
    
    
    
    
    
#                            test_time = 10
                            
#                            ax.quiver(ve_pts[test_time,::40 ,::40, 0], 
#                                       ve_pts[test_time,::40 ,::40, 2],
#                                       ve_pts[test_time,::40 ,::40, 1], 
#                                       surf_theta_ve[test_time,::40, ::40, 0],
#                                       surf_theta_ve[test_time,::40, ::40, 2], 
#                                       surf_theta_ve[test_time,::40, ::40, 1], length=50., 
#                                       normalize=True, color='b')
#                            
#                            ax.quiver(ve_pts[test_time,::40 ,::40, 0], 
#                                       ve_pts[test_time,::40 ,::40, 2],
#                                       ve_pts[test_time,::40 ,::40, 1], 
#                                       direction_r_3D_ve[::40, ::40, 0],
#                                       direction_r_3D_ve[::40, ::40, 2], 
#                                       direction_r_3D_ve[::40, ::40, 1], length=50., 
#                                       normalize=True, color='purple')
#                            
#                            ax.quiver(ve_pts[test_time,::40 ,::40, 0], 
#                                       ve_pts[test_time,::40 ,::40, 2],
#                                       ve_pts[test_time,::40 ,::40, 1], 
#                                       direction_theta_3D_ve[::40, ::40, 0],
#                                       direction_theta_3D_ve[::40, ::40, 2], 
#                                       direction_theta_3D_ve[::40, ::40, 1], length=50., 
#                                       normalize=True, color='k')
#                            
#                            
#                            tra3Dtools.set_axes_equal(ax)
#                            plt.show()  

                            
                            
#                    # this is fixed. and was computed in rect view...... -> then a remapping was done to polar -> inefficient.
#                    
#                    """
#                    This is now fixed such that we can project the component at every timepoint. 
#                    """
#                    surf_radial_vect_ref = stats_obj['surf_radial_vect_ref']
#                    surf_y_vect_ref = stats_obj['surf_y_vect_ref']
#                    surf_normal_vect_ref = stats_obj['surf_normal_vect_ref']
#
#                    mig_tp_pt = int(mig.ravel()[0]) - 1; 
##                    mig_tp_pt = len(vid_ve) // 2
#                    mig_end_tp_pt = int(mig.ravel()[1]) - 1
#                    
###                    ve_ref = ve_pts[mig_tp_pt]
###                    epi_ref = epi_pts[mig_tp_pt]
##                    ve_ref = ve_pts[mig_tp_pt]
##                    epi_ref = epi_pts[mig_tp_pt]
##                    
##                    ve_pts_diff = ve_pts - ve_ref[None,:] # displacement relative to the reference shape. 
##                    epi_pts_diff = epi_pts - epi_ref[None,:]
#                    ve_pts_diff = np.concatenate([ np.zeros_like(ve_pts[0])[None,:], ve_pts[1:] - ve_pts[:-1]])
#                    epi_pts_diff = np.concatenate([ np.zeros_like(epi_pts[0])[None,:], epi_pts[1:] - epi_pts[:-1]])
#                    
##                    ve_pts_diff = np.sum(ve_pts_diff * surf_normal_vect_ref[None,:], axis=-1 )
##                    epi_pts_diff = np.sum(epi_pts_diff * surf_normal_vect_ref[None,:], axis=-1)
#                    ve_pts_diff = np.sum(ve_pts_diff * surf_radial_vect_ref[None,:], axis=-1 )
#                    epi_pts_diff = np.sum(epi_pts_diff * surf_radial_vect_ref[None,:], axis=-1)
##                    ve_pts_diff = np.sum(ve_pts_diff * surf_normal_vect_ref[None,:], axis=-1 )
##                    epi_pts_diff = np.sum(epi_pts_diff * surf_normal_vect_ref[None,:], axis=-1)
#                    
##                    ve_pre_mag, ve_mig_mag = [np.nanmean(np.linalg.norm(ve_pts_diff[:mig_tp_pt], axis=-1), axis=0), 
##                                              np.nanmean(np.linalg.norm(ve_pts_diff[mig_tp_pt:mig_end_tp_pt], axis=-1), axis=0)] 
#                    
##                    epi_pre_mag, epi_mig_mag = [np.nanmean(np.linalg.norm(epi_pts_diff[:mig_tp_pt], axis=-1), axis=0), 
##                                                np.nanmean(np.linalg.norm(epi_pts_diff[mig_tp_pt:mig_end_tp_pt], axis=-1), axis=0)] 
#                    
#                    ve_pre_mag, ve_mig_mag = [np.nanmean(ve_pts_diff[:mig_tp_pt], axis=0), 
#                                              np.nanmean(ve_pts_diff[mig_tp_pt:mig_end_tp_pt], axis=0)] 
#                    
#                    epi_pre_mag, epi_mig_mag = [np.nanmean(epi_pts_diff[:mig_tp_pt], axis=0), 
#                                                np.nanmean(epi_pts_diff[mig_tp_pt:mig_end_tp_pt], axis=0)] 
#                    
#                    ve_pre_mag_polar = np.zeros(unwrap_params_ve_polar.shape[:2])
#                    ve_mig_mag_polar = np.zeros(unwrap_params_ve_polar.shape[:2])
#                    epi_pre_mag_polar = np.zeros(unwrap_params_ve_polar.shape[:2])
#                    epi_mig_mag_polar = np.zeros(unwrap_params_ve_polar.shape[:2])
#                    
#                    
#                    ve_pre_mag_polar[mapped_ij[...,0],
#                                     mapped_ij[...,1]] = ve_pre_mag
#                    ve_mig_mag_polar[mapped_ij[...,0],
#                                     mapped_ij[...,1]] = ve_mig_mag
#                    epi_pre_mag_polar[mapped_ij[...,0],
#                                      mapped_ij[...,1]] = sktform.resize(epi_pre_mag, ve_pre_mag.shape, preserve_range=True)
#                    epi_mig_mag_polar[mapped_ij[...,0],
#                                      mapped_ij[...,1]] = sktform.resize(epi_mig_mag, ve_pre_mag.shape, preserve_range=True)
#                    
#                    if embryo_inversion:
#                        ve_pre_mag_polar = ve_pre_mag_polar[::-1]
#                        ve_mig_mag_polar = ve_mig_mag_polar[::-1]
#                        epi_pre_mag_polar = epi_pre_mag_polar[::-1]
#                        epi_mig_mag_polar = epi_mig_mag_polar[::-1]
#                        
#                        
#                    norm_vol_factor = np.prod(embryo_im_shape) **(1/3.)
#                        
#                    ve_pre_mag_polar = ve_pre_mag_polar / norm_vol_factor
#                    ve_mig_mag_polar = ve_mig_mag_polar / norm_vol_factor
#                    epi_pre_mag_polar = epi_pre_mag_polar / norm_vol_factor
#                    epi_mig_mag_polar = epi_mig_mag_polar / norm_vol_factor
#                    
#                    
#                    # =============================================================================
#                    #       Load and create the radial grid for mapping the statistics.         
#                    # =============================================================================
#                    saveboundaryfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow-epi-boundary')
#                        
#                    epi_boundary_file = os.path.join(saveboundaryfolder,unwrapped_condition,'inferred_epi_boundary-'+unwrapped_condition+'.mat')
#                    epi_boundary_time = spio.loadmat(epi_boundary_file)['contour_line']
#                    
#                    epi_contour_line_pre = epi_boundary_time[0]
#                    epi_contour_line_mig = epi_boundary_time[mig_tp_pt]
#                    
#                    
#                    polar_grid_pre = tile_uniform_windows_radial_guided_line(ve_pre_mag_polar.shape, 
#                                                                             n_r=4, 
#                                                                             n_theta=8,
#                                                                             max_r=vid_ve[0].shape[0]/2./1.2,
#                                                                             mid_line = epi_contour_line_pre[:,[0,1]],
#                                                                             center=None, 
#                                                                             bound_r=True)
#                    
#                    uniq_polar_regions = np.unique(polar_grid_pre)[1:] 
#                    polar_grid_pre = realign_grid_center_polar(polar_grid_pre, 
#                                                               n_angles=8, 
#                                                               center_dist_r=0, 
#                                                               center_angle=ve_angle_move)
#                    
#                    
#                    polar_grid_mig = tile_uniform_windows_radial_guided_line(ve_pre_mag_polar.shape, 
#                                                                             n_r=4, 
#                                                                             n_theta=8,
#                                                                             max_r=vid_ve[0].shape[0]/2./1.2,
#                                                                             mid_line = epi_contour_line_mig[:,[0,1]],
#                                                                             center=None, 
#                                                                             bound_r=True)
#                    polar_grid_mig = realign_grid_center_polar(polar_grid_mig, 
#                                                               n_angles=8, 
#                                                               center_dist_r=0, 
#                                                               center_angle=ve_angle_move) 
#                    
#                    
#                    
#                    plt.figure()
#                    plt.imshow(polar_grid_pre)
#                    plt.show()
#                        
#                    VE_analysis_mapping_grid_pre_emb = np.hstack([np.mean(ve_pre_mag_polar[polar_grid_pre==spixel_reg]) for spixel_reg in uniq_polar_regions])
#                    VE_analysis_mapping_grid_mig_emb = np.hstack([np.mean(ve_mig_mag_polar[polar_grid_mig==spixel_reg]) for spixel_reg in uniq_polar_regions])
#                    Epi_analysis_mapping_grid_pre_emb = np.hstack([np.mean(epi_pre_mag_polar[polar_grid_pre==spixel_reg]) for spixel_reg in uniq_polar_regions])
#                    Epi_analysis_mapping_grid_mig_emb = np.hstack([np.mean(epi_mig_mag_polar[polar_grid_mig==spixel_reg]) for spixel_reg in uniq_polar_regions])
#                    
#                    
##                    VE_analysis_mapping.append([sktform.rotate(ve_pre_mag_polar, angle=ve_angle_move, preserve_range=True), 
##                                                sktform.rotate(ve_mig_mag_polar, angle=ve_angle_move, preserve_range=True)])
##                    Epi_analysis_mapping.append([sktform.rotate(epi_pre_mag_polar, angle=ve_angle_move, preserve_range=True), 
##                                                 sktform.rotate(epi_mig_mag_polar, angle=ve_angle_move, preserve_range=True)])
##                
#                    VE_analysis_mapping_grid_pre.append(VE_analysis_mapping_grid_pre_emb)
#                    VE_analysis_mapping_grid_mig.append(VE_analysis_mapping_grid_mig_emb)
#                    Epi_analysis_mapping_grid_pre.append(Epi_analysis_mapping_grid_pre_emb)
#                    Epi_analysis_mapping_grid_mig.append(Epi_analysis_mapping_grid_mig_emb)
#                    
#                    
#    VE_analysis_mapping_grid_pre = np.vstack(VE_analysis_mapping_grid_pre) 
#    VE_analysis_mapping_grid_mig = np.vstack(VE_analysis_mapping_grid_mig)     
#    Epi_analysis_mapping_grid_pre = np.vstack(Epi_analysis_mapping_grid_pre)       
#    Epi_analysis_mapping_grid_mig = np.vstack(Epi_analysis_mapping_grid_mig)  
#    
#    
#    
#    
## =============================================================================
##     Visualization on a radial grid basis.
## =============================================================================
#    polar_grid_region = tile_uniform_windows_radial_custom( (640,640), 
#                                                            n_r=4, 
#                                                            n_theta=8, 
#                                                            max_r=640/2./1.2, 
#                                                            mask_distal=None, # leave a set distance
#                                                            center=None, 
#                                                            bound_r=True)
#
#    VE_analysis_mapping_grid_pre_region = np.zeros(polar_grid_region.shape, dtype=np.float32)
#    VE_analysis_mapping_grid_mig_region = np.zeros(polar_grid_region.shape, dtype=np.float32)
#    Epi_analysis_mapping_grid_pre_region = np.zeros(polar_grid_region.shape, dtype=np.float32)
#    Epi_analysis_mapping_grid_mig_region = np.zeros(polar_grid_region.shape, dtype=np.float32)
#    
#    
#    
#    for kk in range(len(uniq_polar_regions)):
#        
#        VE_analysis_mapping_grid_pre_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (VE_analysis_mapping_grid_pre[::].mean(axis=0))[kk] 
#        VE_analysis_mapping_grid_mig_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (VE_analysis_mapping_grid_mig[::].mean(axis=0))[kk] 
#        
#        Epi_analysis_mapping_grid_pre_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (Epi_analysis_mapping_grid_pre[::].mean(axis=0))[kk] 
#        Epi_analysis_mapping_grid_mig_region[polar_grid_region == np.unique(polar_grid_region)[1:][kk]] = (Epi_analysis_mapping_grid_mig[::].mean(axis=0))[kk] 
#
#    
#    # choose a heatmap scale and visualize all of these (rotated) 
#    rotate_angle = 270 + 360/8./2.
#    
#    
#    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
##    plt.suptitle('Analysis Grid - %s' %(analyse_stage))
##    plt.hlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.vlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.quiver(reg_centroids[:,1], reg_centroids[:,0],
##               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[0], 
##               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[1], color='g', scale=.0075, units='xy')
##    plt.quiver(reg_centroids[:,1], reg_centroids[:,0], 
##               50*np.sin(all_directions_regions_ve_relative_mean[0]), -50*np.cos(all_directions_regions_ve_relative_mean[1]))
##    for jj in range(len(reg_centroids)):
##        plt.text(reg_centroids[jj,1], 
##                 reg_centroids[jj,0],
##                 str(jj+1), horizontalalignment='center',
##                 verticalalignment='center', fontsize= 18, color='k', fontname='Liberation Sans')
#    ax[0,0].imshow(sktform.rotate(VE_analysis_mapping_grid_pre_region, angle=rotate_angle, preserve_range=True).astype(np.float32), cmap='coolwarm', vmin=0, vmax=0.006)
#    ax[0,1].imshow(sktform.rotate(VE_analysis_mapping_grid_mig_region, angle=rotate_angle, preserve_range=True).astype(np.float32), cmap='coolwarm', vmin=0, vmax=0.006 )
#    ax[1,0].imshow(sktform.rotate(Epi_analysis_mapping_grid_pre_region, angle=rotate_angle, preserve_range=True).astype(np.float32), cmap='coolwarm', vmin=0, vmax=0.006 )
#    ax[1,1].imshow(sktform.rotate(Epi_analysis_mapping_grid_mig_region, angle=rotate_angle, preserve_range=True).astype(np.float32), cmap='coolwarm', vmin=0, vmax=0.006 )
#    plt.axis('off')
#    
#    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
##    plt.suptitle('Analysis Grid - %s' %(analyse_stage))
##    plt.hlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.vlines(polar_grid_region.shape[0]/2., 0, polar_grid_region.shape[0], lw=2)
##    plt.quiver(reg_centroids[:,1], reg_centroids[:,0],
##               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[0], 
##               np.mean(all_embryo_vol_factors)*all_directions_regions_ve_relative_mean[1], color='g', scale=.0075, units='xy')
##    plt.quiver(reg_centroids[:,1], reg_centroids[:,0], 
##               50*np.sin(all_directions_regions_ve_relative_mean[0]), -50*np.cos(all_directions_regions_ve_relative_mean[1]))
##    for jj in range(len(reg_centroids)):
##        plt.text(reg_centroids[jj,1], 
##                 reg_centroids[jj,0],
##                 str(jj+1), horizontalalignment='center',
##                 verticalalignment='center', fontsize= 18, color='k', fontname='Liberation Sans')
#    ax[0,0].imshow(sktform.rotate(VE_analysis_mapping_grid_pre_region, angle=rotate_angle, preserve_range=True).astype(np.float32), cmap='coolwarm', vmin=-5e-4, vmax=5e-4)
#    ax[0,1].imshow(sktform.rotate(VE_analysis_mapping_grid_mig_region, angle=rotate_angle, preserve_range=True).astype(np.float32), cmap='coolwarm', vmin=-5e-4, vmax=5e-4)
#    ax[1,0].imshow(sktform.rotate(Epi_analysis_mapping_grid_pre_region, angle=rotate_angle, preserve_range=True).astype(np.float32), cmap='coolwarm',vmin=-5e-4, vmax=5e-4 )
#    ax[1,1].imshow(sktform.rotate(Epi_analysis_mapping_grid_mig_region, angle=rotate_angle, preserve_range=True).astype(np.float32), cmap='coolwarm', vmin=-5e-4, vmax=5e-4 )
#    plt.axis('off')
    
    
#    VE_shapes = np.vstack([ve[0].shape for ve in VE_analysis_mapping])
#    VE_analysis_pre_maps_average = np.mean(np.dstack([sktform.resize(ve[0], (512,512), preserve_range=True) for ve in VE_analysis_mapping]), axis=-1)
#    VE_analysis_mig_maps_average = np.mean(np.dstack([sktform.resize(ve[1], (512,512), preserve_range=True) for ve in VE_analysis_mapping]), axis=-1)
#    Epi_analysis_pre_maps_average = np.mean(np.dstack([sktform.resize(epi[0], (512,512), preserve_range=True) for epi in Epi_analysis_mapping]), axis=-1)
#    Epi_analysis_mig_maps_average = np.mean(np.dstack([sktform.resize(epi[1], (512,512), preserve_range=True) for epi in Epi_analysis_mapping]), axis=-1)
#
#
#    plt.figure(figsize=(15,15))
#    plt.suptitle('VE')
#    plt.subplot(121)
#    plt.imshow(VE_analysis_pre_maps_average, cmap='coolwarm', vmin=0, vmax=15)
#    plt.subplot(122)
#    plt.imshow(VE_analysis_mig_maps_average, cmap='coolwarm', vmin=0, vmax=15)
#    
#    
#    plt.figure(figsize=(15,15))
#    plt.suptitle('Epi')
#    plt.subplot(121)
#    plt.imshow(Epi_analysis_pre_maps_average, cmap='coolwarm', vmin=0, vmax=15)
#    plt.subplot(122)
#    plt.imshow(Epi_analysis_mig_maps_average, cmap='coolwarm', vmin=0, vmax=15)
#        
        
#                ve_pts_polar = 
#                epi_pts_polar = 
                
#                spio.savemat(saveprojfile, {'ve_pts': all_demons_ve_pts,
#                                            'epi_pts': all_demons_epi_pts, 
#                                            'surf_normal_vect_ref': surface_normal_norm, 
#                                            'surf_y_vect_ref': surface_y_norm,
#                                            'surf_radial_vect_ref': surface_theta_norm,
#                                            'demons_files': demons_rev_warp_files,
#                                            'demons_times': demons_times,
#                                            'rect_polar_mapping': np.dstack(mapped_ij)})ss
            
            
            
            
#                fig = plt.figure(figsize=(15,15))
#                ax = fig.add_subplot(111, projection='3d')
#                ax.scatter(all_demons_ve_pts[0,::20, ::20,2],  
#                           all_demons_ve_pts[0,::20, ::20,1], 
#                           all_demons_ve_pts[0,::20, ::20,0])
#                ax.scatter(all_demons_ve_pts[20,::20, ::20,2],  
#                           all_demons_ve_pts[20,::20, ::20,1], 
#                           all_demons_ve_pts[20,::20, ::20,0])
#                ax.scatter(all_demons_ve_pts[100,::20, ::20,2],  
#                           all_demons_ve_pts[100,::20, ::20,1], 
#                           all_demons_ve_pts[100,::20, ::20,0])
#                plt.show()
#            
                
#                # give indication of where in the embryo is the greatest change with respect to the start geometry?
#                all_demons_ve_mag = np.zeros(unwrap_params_ve_polar.shape[:2]); 
#                all_demons_ve_mag[mapped_ij[0],
#                                  mapped_ij[1]] = np.linalg.norm(all_demons_ve[:24], axis=-1).mean(axis=0)
#                
#                
#                all_demons_epi_mag = np.zeros(unwrap_params_ve_polar.shape[:2]); 
#                all_demons_epi_mag[mapped_ij[0],
#                                  mapped_ij[1]] = sktform.resize(np.linalg.norm(all_demons_epi[:24], axis=-1).mean(axis=0), 
#                                                                  all_demons_ve.shape[1:-1], 
#                                                                  preserve_range=True)
#                
#                plt.figure(figsize=(15,15))
#                plt.subplot(131)
#                plt.imshow(all_demons_ve_mag, cmap='coolwarm', vmin=0, vmax=30)
#                plt.subplot(132)
#                plt.imshow(all_demons_epi_mag, cmap='coolwarm', vmin=0, vmax=30)
#                plt.subplot(133)
#                plt.imshow(all_demons_epi_mag, cmap='gray')
#                plt.imshow(all_demons_ve_mag, cmap='coolwarm', vmin=0, vmax=30, alpha=0.5)
#                plt.show()
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

    
