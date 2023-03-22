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



def test_VE_piecewise_gradient_refined(gradients, break_points, thresh_grad_buffer=0.5, min_frames_stage=2):
    
    # add the start and end stretch 
    breaks = np.hstack([0, break_points, len(gradients)+1])
    break_gradients = np.zeros(len(breaks)-1)
    break_gradient_signal = np.zeros(len(gradients))
    
    for ii in range(len(break_gradients)):
        
        break_start = breaks[ii]
        break_end = breaks[ii+1]
        
        break_grad = gradients[break_start:break_end]
        # add the abs? 
        break_gradients[ii] = np.mean(break_grad)
        break_gradient_signal[break_start:break_end] = np.mean(break_grad)
        
    grad_nonzero = break_gradients >= thresh_grad_buffer # must be positive for it to be migrating 
    
    pre_breaks = []
    migrate_breaks = []
    post_breaks = []
    
    
    if np.sum(grad_nonzero) > 0:
#        print('migration exist at ', thresh_grad_buffer)
        
        # initialise the arrays. 
        # detect the first such region as the core AVE region
        max_region_id = np.arange(len(grad_nonzero))[grad_nonzero > 0 ][0]
#        max_region_id = np.argmax(break_gradients)
#        print(break_grad)
        migrate_breaks.append([max_region_id, breaks[max_region_id], breaks[max_region_id+1], break_gradients[max_region_id]])
        
        null_region_ids = np.arange(len(grad_nonzero))[grad_nonzero == 0 ]
        
        if len(null_region_ids) > 0:
            if null_region_ids[0] < max_region_id:
                # pre-migration exist before migration!
                pre_breaks.append([null_region_ids[0], breaks[null_region_ids[0]], breaks[null_region_ids[0]+1], break_gradients[null_region_ids[0]]])            
            
            
        # now we iterate over all the break gradients -> check if they have been assigned and if not which one they are closer to.... 
        for ii in range(len(break_gradients)):
            
            if len(post_breaks) > 0: 
                post_breaks.append([ii, breaks[ii], breaks[ii+1], break_gradients[ii]])
            # we can now assign the remainder cases !. 
            # check has this been assigned?
            else:
                assigned = False
                for b in pre_breaks:
    #                print(b)
                    if ii==b[0]:
                        assigned = True
                        break
                
                if assigned == False:
    #                print(b)
                    for b in migrate_breaks:
    #                    print(ii, b)
                        if ii==b[0]:
                            assigned = True
                            break
                
    #            print(assigned)
                if assigned == False:
                    break_grad = break_gradients[ii]
                    
    #                if grad_nonzero[ii] == False:
                    if len(pre_breaks) > 0:
                        diff_pre = break_grad - np.mean([b[-1] for b in pre_breaks]); diff_pre = np.abs(diff_pre)
                    else:
#                        if np.sum(grad_nonzero == 0) > 0:
#                            diff_pre = break_grad - np.min(break_gradients); diff_pre = np.abs(diff_pre) # comparison to 0 if no pre-migration phase. 
#                        else:
                        diff_pre = np.abs(break_grad - 0) # comparison to 0 if no pre-migration phase. 
                    diff_mig = break_grad - np.mean([b[-1] for b in migrate_breaks]); diff_mig = np.abs(diff_mig)
                    
                    diffs = [diff_pre, diff_mig]
                    diff_min_id = np.argmin(diffs)
                    if diff_min_id == 0: 
                        
                        if len(pre_breaks) > 0: 
                            if ii - pre_breaks[-1][0] == 1:
                                # directly after
                                pre_breaks.append([ii, breaks[ii], breaks[ii+1], break_gradients[ii]])
                            else:
                                if ii > migrate_breaks[-1][0]:
                                    # more than directly after -> discontinuous. 
                                    post_breaks.append([ii, breaks[ii], breaks[ii+1], break_gradients[ii]])
                                else:
                                    migrate_breaks.append([ii, breaks[ii], breaks[ii+1], break_gradients[ii]])
                                    migrate_breaks = sorted(migrate_breaks, key=lambda x:x[0])
                        else:
                            post_breaks.append([ii, breaks[ii], breaks[ii+1], break_gradients[ii]])
                        
                    if diff_min_id == 1:
                        if np.abs(ii - migrate_breaks[-1][0]) == 1 :
                            migrate_breaks.append([ii, breaks[ii], breaks[ii+1], break_gradients[ii]])
                            migrate_breaks = sorted(migrate_breaks, key=lambda x:x[0])
                        else:
                            if ii < migrate_breaks[0][0] :
                                migrate_breaks.append([ii, breaks[ii], breaks[ii+1], break_gradients[ii]])
                                migrate_breaks = sorted(migrate_breaks, key=lambda x:x[0])
                            
#    #                else:
#                    print(ii, diff_pre, diff_mig)
#                    # which is closer? 
                    
    else:
        # everything can be regarded as pre=migration. 
        for ii in range(len(break_gradients)):
            pre_breaks.append([ii, breaks[ii], breaks[ii+1], break_gradients[ii]])
        
        
    ##### final check of assignments -> if a region is say less than 2 frames -> not useful! -> can't even get speed. 
    
    all_breaks = [pre_breaks, migrate_breaks, post_breaks]
#    all_breaks_times = []
#    
#    for b in all_breaks:
#        if len(b) > 0:
#            all_breaks_times.append(b[-1][2] + 1 - b[0][1])
#        else:
#            all_breaks_times.append(0)
#        
#    all_breaks_times = np.hstack(all_breaks_times)
#    
#    invalid_regions = np.arange(len(all_breaks_times))[np.logical_and(all_breaks_times>0, all_breaks_times<=min_frames_stage)]
#        
#    if len(invalid_regions) > 0:
#        for inval in invalid_regions:
#            
#            # try the successive region..... 
#            next_region = inval + 1
#            
#            if next_region not in invalid_regions:
#                if next_region < len(all_breaks_times):
#                    all_breaks[next_region] += all_breaks[inval]
#                    all_breaks[next_region] = sorted(all_breaks[next_region], key=lambda x:x[0])
#                    
#            # delete the invalid region 
#            all_breaks[inval] = []
#            
#    print('==== break times ====')
#    print(all_breaks_times)
#    print('---------------------------')
        
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
            out_times.append(per[0][1:-1])
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


def densify_pts(pts_all, pts_mask, spixel_size, imshape, thresh_factor=1.5, smooth_sigma=1., sigma_thresh=1.):
    
    from skimage.filters import gaussian
    from scipy.ndimage.morphology import binary_fill_holes
    
    pts_mask_connected = postprocess_polar_labels(pts_all, 
                                                  pts_mask, 
                                                  dist_thresh=thresh_factor*spixel_size)
    
    val_pts = pts_all[pts_mask_connected>0]
        
    density_val_pts = np.zeros(imshape)
    density_val_pts[val_pts[:,0],
                    val_pts[:,1]] = 1
    density_pts = gaussian(density_val_pts, smooth_sigma*spixel_size)
    density_pts_mask = density_pts > np.mean(density_pts) + sigma_thresh*np.std(density_pts) # produce the mask. 
    density_pts_mask = binary_fill_holes(density_pts_mask)
    
    return density_pts_mask

    
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
    
    # import the 3D tools package.
    from MOSES.Motion_Analysis import tracks3D_tools as tra3Dtools

    
    from skimage.filters import threshold_otsu
    import skimage.transform as sktform
    from tqdm import tqdm 
    import pandas as pd 
    from skimage.measure import find_contours
    import skimage.morphology as skmorph
    from mpl_toolkits.mplot3d import Axes3D
    
#    saveallplotsfolder = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Paper'
#    saveallplotsfolder = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Rect-Paper'
    saveallplotsfolder = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Rect-Curvilinear-Paper'
    fio.mkdir(saveallplotsfolder)
    
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Set up a nearest neighbour model 
    from sklearn.neighbors import NearestNeighbors 
    nbrs_model = NearestNeighbors(n_neighbors=9, algorithm='auto') # how many neighbours to include? 9 = 8-connected?
    
    
    rootfolder = '/media/felix/Srinivas4/LifeAct'
#    rootfolder = '/media/felix/Srinivas4/MTMG-TTR'
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX'
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
# =============================================================================
#     1. Load all embryos 
# =============================================================================
    embryo_folders = np.hstack([os.path.join(rootfolder,folder) for folder in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, folder)) and 'new_hex' not in folder])
    
# =============================================================================
#     2. find valid embryo folders i.e. has a geodesic-rotmatrix. 
# =============================================================================
    n_spixels = 5000 # set the number of pixels. 
    smoothwinsize = 3
#    n_spixels = 5000
    
    manual_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/manual_staging_holly_matt.xlsx'
    manual_staging_tab = pd.read_excel(manual_staging_file)
    
    """
    set which experiment folder. 
    """
    all_embryos_plot = []
    diff_curves_all = []
    break_grads_all = []
    break_pts_all = []
    pre_all = []
    mig_all = []
    post_all = []
    
    pre_all_auto = []
    mig_all_auto = []
    post_all_auto = []
    
#    for embryo_folder in embryo_folders[-2:-1]:
    for embryo_folder in embryo_folders[:]:
#    for embryo_folder in embryo_folders[3:4]:
        
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
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned')
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-5000_VE_seg_directions')
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-new')
            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-curvilinear_Tcorrected')
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-5000_VE_seg_directions')
            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels))
            
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
            temp_tform_scales = np.prod(temp_tform_scales, axis=0) # this line is just in case there were multiple transformations. 
            
            
            if 'MTMG-TTR' in rootfolder or 'LifeAct' in rootfolder:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
            
            else:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_hex_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_hex_embryo(unwrap_im_files, unwrap_meta_files)
            
            # iterate over the pairs # and do it for all views!. 
            for ii in tqdm(range(len(paired_unwrap_file_condition))[:]):
                
                unwrapped_condition = paired_unwrap_file_condition[ii]
                
                if unwrapped_condition.split('-')[-1] == 'rect':
                
                    if 'MTMG-TTR' in rootfolder or 'LifeAct' in rootfolder:
                        ve_unwrap_file, epi_unwrap_file = paired_unwrap_files[ii]
                    else:
                        ve_unwrap_file, epi_unwrap_file, hex_unwrap_file = paired_unwrap_files[ii]
                            
                # =============================================================================
                #        Load the relevant videos.          
                # =============================================================================
                    vid_ve = fio.read_multiimg_PIL(ve_unwrap_file)
#                    vid_epi = fio.read_multiimg_PIL(epi_unwrap_file)
#                    vid_epi_resize = sktform.resize(vid_epi, vid_ve.shape, preserve_range=True)
                    
                    vid_size = vid_ve[0].shape
                    
                    if 'MTMG-HEX' in rootfolder:
                        vid_hex = fio.read_multiimg_PIL(hex_unwrap_file)
                        
                # =============================================================================
                #       Load the relevant saved pair track file.               
                # =============================================================================
                    trackfile = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels) + unwrapped_condition+'.mat')
                    
                    meantracks_ve = spio.loadmat(trackfile)['meantracks_ve']
#                    meantracks_epi = spio.loadmat(trackfile)['meantracks_epi']
                    proj_condition = spio.loadmat(trackfile)['condition'][0].split('-')
                    proj_type = proj_condition[-1]
                      
                    
                # =============================================================================
                #       Load the relevant manual staging file from Holly and Matt.        
                # =============================================================================
                    """
                    create the lookup embryo name
                    """
                    if proj_condition[1] == '':
                        embryo_name = proj_condition[0]
                    else:
                        embryo_name = proj_condition[0]+'_E'+proj_condition[1].split('Emb')[1]
                    
                    table_emb_names = np.hstack([str(s) for s in manual_staging_tab['Embryo'].values])
                    select_stage_row = manual_staging_tab.loc[table_emb_names==embryo_name]
                    
                    pre = parse_start_end_times(select_stage_row['Pre-migration'])
                    mig = parse_start_end_times(select_stage_row['Migration to Boundary'])
                    post = parse_start_end_times(select_stage_row['Post-Boundary'])
                    
                    pre_all.append(pre)
                    mig_all.append(mig)
                    post_all.append(post)
                    print(pre, mig, post)
                    
                # =============================================================================
                #       Load the 3D demons mapping file coordinates.        
                # =============================================================================
                # load the associated unwrap param files depending on polar/rect coordinates transform. 
                    unwrap_param_file_ve = ve_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
                    unwrap_params_ve = spio.loadmat(unwrap_param_file_ve); unwrap_params_ve = unwrap_params_ve['ref_map_'+proj_type+'_xyz']
                    
#                    unwrap_param_file_epi = epi_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
#                    unwrap_params_epi = spio.loadmat(unwrap_param_file_epi); unwrap_params_epi = unwrap_params_epi['ref_map_'+proj_type+'_xyz']
#                    unwrap_params_epi = sktform.resize(unwrap_params_epi, unwrap_params_ve.shape, preserve_range=True)
                    
                    
                    if proj_type == 'rect':
                        unwrap_params_ve = unwrap_params_ve[::-1] # reverse ....
#                        unwrap_params_epi = unwrap_params_epi[::-1]
                        
                        plt.figure()
                        plt.imshow(unwrap_params_ve[...,0])
                        plt.show()
                        
                # =============================================================================
                #       Load the relevant VE segmentation from trained classifier    
                # =============================================================================
    #                ve_seg_params_file = os.path.join(savevesegfolder, os.path.split(trackfile)[-1].replace('rect','polar').replace('.mat', '-ve_aligned.mat'))
    #                ve_seg_params_file = ve_seg_params_file.replace('-5000_', '-1000_')
                    
                    ve_seg_params_file = 'deepflow-meantracks-1000_' + os.path.split(unwrap_param_file_ve)[-1].replace('_unwrap_params-geodesic.mat',  '_polar-ve_aligned.mat')
#                    ve_seg_params_file = 'deepflow-meantracks-5000_' + unwrapped_condition.split('-rect')[0] + '-polar-ve_aligned.mat'
                    ve_seg_params_file = os.path.join(savevesegfolder, ve_seg_params_file)
                    ve_seg_params_obj = spio.loadmat(ve_seg_params_file)                             
#                    
                    
#                    ve_seg_select_rect = ve_seg_params_obj['ve_rect_select'].ravel() > 0
    #                ve_seg_select_rect = ve_seg_params_obj['ve_rect_select'].ravel() > 0
    #                ve_seg_select_rect_valid = ve_seg_params_obj['ve_rect_select_valid'].ravel() > 0
                    
    #                ve_seg_select_rect_valid = ve_seg_params_obj['density_rect_pts']
    #                ve_seg_select_rect_valid = ve_seg_select_rect_valid >= np.mean(ve_seg_select_rect_valid) + 1 * np.std(ve_seg_select_rect_valid)
                    ve_seg_select_rect_valid = ve_seg_params_obj['density_polar_pts_mask'] # get the polar associated mask. 
                    ve_seg_mask_cont = find_contours(ve_seg_select_rect_valid>0, 0)[0]
                    ve_seg_migration_times = ve_seg_params_obj['migration_stage_times'][0]
#                                    
                    embryo_inversion = ve_seg_params_obj['embryo_inversion'].ravel()[0] > 0 
                    print(embryo_inversion)
                    
                    
                # =============================================================================
                #   Check for embryo inversion -> all params etc have to be inverted accordingly.                
                # =============================================================================
                    if embryo_inversion:
                        if proj_type == 'rect':
                            # fix the correct inversion 
                            unwrap_params_ve = unwrap_params_ve[:,::-1] # reverse x direction 
#                            unwrap_params_epi = unwrap_params_epi[:,::-1]
                            unwrap_params_ve[...,2] = embryo_im_shape[2] - unwrap_params_ve[...,2]
#                            unwrap_params_epi[...,2] = embryo_im_shape[2] - unwrap_params_epi[...,2]
                            
                            # 2D inversions.
                            meantracks_ve[...,1] = (vid_size[1]-1) - meantracks_ve[...,1] # reverse the x direction 
#                            meantracks_epi[...,1] = (vid_epi[0].shape[1]-1) - meantracks_epi[...,1]
                            
                            vid_ve = vid_ve[:,:,::-1]
#                            vid_epi = vid_epi[:,:,::-1]
#                            vid_epi_resize = vid_epi_resize[:,:,::-1]
                        else:
                            unwrap_params_ve = unwrap_params_ve[::-1] # reverse y direction
#                            unwrap_params_epi = unwrap_params_epi[::-1] # polar or rect should refer to the same set of 3D coordinates. 
                            unwrap_params_ve[...,2] = embryo_im_shape[2] - unwrap_params_ve[...,2]
#                            unwrap_params_epi[...,2] = embryo_im_shape[2] - unwrap_params_epi[...,2]
                            
                            meantracks_ve[...,0] = (vid_size[0]-1) - meantracks_ve[...,0] # switch the y direction. 
#                            meantracks_epi[...,0] = (vid_epi[0].shape[0]-1) - meantracks_epi[...,0]
                            
                            vid_ve = vid_ve[:,::-1]
#                            vid_epi = vid_epi[:,::-1]
#                            vid_epi_resize = vid_epi_resize[:,::-1]
                            
#                    from skimage.filters import gaussian
#                    unwrap_params_ve = np.dstack([gaussian(unwrap_params_ve[...,jj], sigma=3, preserve_range=True) for jj in range(unwrap_params_ve.shape[-1])])
                    
                    # clearly define the invalid regions of the mask for rect.
                    invalid_unwrap_params_ve = unwrap_params_ve[...,0] == 0 
                    invalid_unwrap_params_ve = skmorph.binary_dilation(invalid_unwrap_params_ve, skmorph.square(51))
#                    ve_seg_select_rect_valid = np.logical_and(ve_seg_select_rect_valid, invalid_unwrap_params_ve == 0)
                            
                    if np.isnan(ve_seg_migration_times[0]):
                        ve_seg_select_rect = ve_seg_select_rect_valid[meantracks_ve[:,0,0], 
                                                                      meantracks_ve[:,0,1]] > 0
                    else:
                        ve_seg_select_rect = ve_seg_select_rect_valid[meantracks_ve[:,int(ve_seg_migration_times[0]),0], 
                                                                      meantracks_ve[:,int(ve_seg_migration_times[0]),1]] > 0
#                    
##                    ve_seg_select_rect = ve_seg_params_obj['ve_rect_select'].ravel()
#                    ve_seg_select_rect = ve_seg_select_rect_valid[meantracks_ve[:,0,0], 
#                                                                  meantracks_ve[:,0,1]] > 0
                    
                    plt.figure()
                    plt.imshow(vid_ve[0])
                    plt.plot(meantracks_ve[ve_seg_select_rect>0,0,1], 
                               meantracks_ve[ve_seg_select_rect>0,0,0], 'go')
                    plt.show()
                    
                    
                    """
                    Verify the projection. -> producing an image to check the segmented region from the algorithm (also good for the paper)
                    """
                    
                    # plot 1: showing the position at the start of migration and mask if available. 
                    fig, ax = plt.subplots(figsize=(15,15))
                    plt.title('Migration Start + classified track points locations')
                    if np.isnan(ve_seg_migration_times[0]):
                        ax.imshow(vid_ve[0], cmap='gray')
                        seg_pts_time = meantracks_ve[ve_seg_select_rect, 0].copy()
    #                    seg_pts_mask = densify_pts()
                        ax.plot(seg_pts_time[:,1], seg_pts_time[:,0], 'wo')
                        ax.plot(ve_seg_mask_cont[:,1], 
                                ve_seg_mask_cont[:,0], 'r--', lw=5)
    #                    plt.show()
                    else:
                        ax.imshow(vid_ve[int(ve_seg_migration_times[0])], cmap='gray')
                        seg_pts_time = meantracks_ve[ve_seg_select_rect, int(ve_seg_migration_times[0])].copy()
    #                    seg_pts_mask = densify_pts()
                        ax.plot(seg_pts_time[:,1], seg_pts_time[:,0], 'wo')
                        ax.plot(ve_seg_mask_cont[:,1], 
                                ve_seg_mask_cont[:,0], 'r--', lw=5)
                        
                    # save the plots. 
                    fig.savefig(os.path.join(saveallplotsfolder, 
                                             'AVE_seg_TP0_or_Mig-Start_MOSES_%s.svg' %(os.path.split(ve_unwrap_file)[-1].split('.tif')[0])), 
                                bbox_inches='tight')
                    plt.show()
#                        
                        
                    fig, ax = plt.subplots(figsize=(15,15))
                    plt.title('Migration Start + inferred start ocations of AVE cells')
                    ax.imshow(vid_ve[0], cmap='gray')
                    seg_pts_time = meantracks_ve[ve_seg_select_rect, 0].copy()
                    meantracks_ve_spixel_size = np.abs(meantracks_ve[1,0,1] - meantracks_ve[0,0,1])
                    seg_pts_mask_time = densify_pts(meantracks_ve[:,0], 
                                                    ve_seg_select_rect, 
                                                    meantracks_ve_spixel_size, 
                                                    vid_ve[0].shape, thresh_factor=1.5, smooth_sigma =1., sigma_thresh=1.)
                    seg_pts_mask_time_cont = find_contours(seg_pts_mask_time,0)[0]
                    ax.plot(seg_pts_time[:,1], seg_pts_time[:,0], 'wo')
                    ax.plot(seg_pts_mask_time_cont[:,1], 
                            seg_pts_mask_time_cont[:,0], 'r--', lw=5)
                    fig.savefig(os.path.join(saveallplotsfolder, 
                                             'Inferred_AVE_seg_TP0-MOSES_%s.svg' %(os.path.split(ve_unwrap_file)[-1].split('.tif')[0])), 
                                bbox_inches='tight')
                    plt.show()
                    
    #                ve_seg_select_rect = ve_seg_select_rect_valid[meantracks_ve[:,0,0], 
    #                                                              meantracks_ve[:,0,1]]
#                    """
#                    Get the valid region of polar projection  
#                    """
#                    YY,XX = np.meshgrid(range(vid_ve[0].shape[1]), 
#                                        range(vid_ve[0].shape[0]))
#                    dist_centre = np.sqrt((YY-vid_ve[0].shape[0]/2.)**2 + (XX-vid_ve[0].shape[1]/2.)**2)
#                    valid_region = dist_centre <= vid_ve[0].shape[1]/2./1.2
                    
                    valid_region = invalid_unwrap_params_ve==0
#                    valid_region = np.ones(invalid_unwrap_params_ve.shape)
                    
                    """
                    Get the predicted VE region. 
                    """
                    all_reg = np.arange(len(meantracks_ve))
                    all_reg_valid = valid_region[meantracks_ve[:,0,0], 
                                                 meantracks_ve[:,0,1]] > 0 
                    
                    """
                    Compute the 3D displacements of the tracks. 
                    """
    #                # =============================================================================
    #                # Computation A: -> direct displacement computation (no removal of global components)                 
    #                # =============================================================================
    #                
    #                meantracks_ve_smooth_diff = (meantracks_ve_smooth[:,1:] - meantracks_ve_smooth[:,:-1]).astype(np.float32)
    ##                meantracks_epi_smooth_diff = (meantracks_epi_smooth[:,1:] - meantracks_epi_smooth[:,:-1]).astype(np.float32)
    #                
    #                meantracks_ve_smooth_diff[...,0] = meantracks_ve_smooth_diff[...,0]/embryo_im_shape[0]
    #                meantracks_ve_smooth_diff[...,1] = meantracks_ve_smooth_diff[...,1]/embryo_im_shape[1]
    #                meantracks_ve_smooth_diff[...,2] = meantracks_ve_smooth_diff[...,2]/embryo_im_shape[2]
                    
                    # =============================================================================
                    # Computation B: ->  DO NOT correct for the global bias , Do apply a single volumetric scaling.                 
                    # =============================================================================
                    meantracks_ve_smooth_diff = compute_3D_displacements(meantracks_ve, 
                                                                         unwrap_params_ve, 
                                                                         correct_global=False, 
                                                                         mask_track=invalid_unwrap_params_ve==0)
                    
                    
                    # so far it just sees like 
                    meantracks_ve_smooth_diff = meantracks_ve_smooth_diff  * temp_tform_scales[None,1:,None] # apply the growth correction. 
                    
                    vol_scale_factor = np.product(embryo_im_shape) **(1/3.)
                    
                    meantracks_ve_smooth_diff[...,0] = meantracks_ve_smooth_diff[...,0]/vol_scale_factor
                    meantracks_ve_smooth_diff[...,1] = meantracks_ve_smooth_diff[...,1]/vol_scale_factor
                    meantracks_ve_smooth_diff[...,2] = meantracks_ve_smooth_diff[...,2]/vol_scale_factor
                    
                    
                    mean_disps_spixels_regions = np.linalg.norm(meantracks_ve_smooth_diff.mean(axis=1), axis=1)
                    
                    """
                    plot to check the segmentation  (debugging only)
                    """
                    ve_reg_id = all_reg[np.logical_and(ve_seg_select_rect, all_reg_valid)]
                    
                    # how do we choose this? 
                    neg_reg_id = all_reg[np.logical_and(np.logical_not(ve_seg_select_rect), all_reg_valid)]
                    
                    fig, ax = plt.subplots(figsize=(15,15))
                    ax.imshow(vid_ve[0],cmap='gray')
#                    ax.plot(meantracks_ve[:,0,1], meantracks_ve[:,0,0], 'wo')
                    ax.plot(meantracks_ve[neg_reg_id,0,1], meantracks_ve[neg_reg_id,0,0], 'ro')
                    ax.plot(meantracks_ve[ve_reg_id,0,1], meantracks_ve[ve_reg_id,0,0], 'go')
                    plt.show()
                    
                    """
                    Build the unwrap_params nearest neighbor model
                    """
                    surf_nbrs_ve, surf_nbrs_ve_3D_pts = tra3Dtools.learn_nearest_neighbor_surf_embedding_model(unwrap_params_ve, 
                                                                                                               nearest_K=1)
                    
#                    fig, ax = plt.subplots(figsize=(15,15))
#                    ax.imshow(vid_ve[0],cmap='gray')
##                    ax.plot(meantracks_ve[:,0,1], meantracks_ve[:,0,0], 'wo')
##                    ax.plot(meantracks_ve[neg_reg_id,0,1], meantracks_ve[neg_reg_id,0,0], 'ro')
#                    plot_tracks(meantracks_ve[ve_reg_id], ax=ax, color='r')
#                    plt.show()
#                    
                    fig, ax = plt.subplots(figsize=(15,15))
                    ax.imshow(vid_ve[0],cmap='gray')
#                    ax.plot(meantracks_ve[:,0,1], meantracks_ve[:,0,0], 'wo')
#                    ax.plot(meantracks_ve[neg_reg_id,0,1], meantracks_ve[neg_reg_id,0,0], 'ro')
                    plot_tracks(meantracks_ve, ax=ax, color='r')
                    plt.show()
#                    
                    # presence of very large outlier!. 
                    ve_diff_select = meantracks_ve_smooth_diff[ve_reg_id] # this is strongly affected by extremal? 
                    ve_diff_mean = np.mean(ve_diff_select, axis=1)  
#                    ve_diff_mean = np.median(ve_diff_select, axis=1)  
                    ve_diff_mean = ve_diff_mean/(np.linalg.norm(ve_diff_mean, axis=-1)[:,None] + 1e-8) 
                    
                    # build the tangent vectors in the mean direction.
                    ve_diff_mean_tangent_vects = tra3Dtools.construct_surface_tangent_vectors(meantracks_ve[ve_reg_id], 
                                                                                              unwrap_params_ve, 
                                                                                              ve_diff_mean[:,None,:], 
                                                                                              K_neighbors=1, 
                                                                                              nbr_model=surf_nbrs_ve, 
                                                                                              pts3D_all=surf_nbrs_ve_3D_pts, 
                                                                                              dist_sample=10, 
                                                                                              return_dist_model=False)
                    
                    # method 1) this is just the linear version
#                    ve_component = np.vstack([ve_diff_select[ll].dot(ve_diff_mean[ll]) for ll in range(len(ve_diff_select))]) # n_superpixels x time
                    ve_component = np.sum(ve_diff_select*ve_diff_mean_tangent_vects[:,:-1], axis=-1)
                    
    
                    all_speed_neg = meantracks_ve_smooth_diff[neg_reg_id]
#                    all_speed_neg_mean = all_speed_neg.mean(axis=1) 
                    all_speed_neg_mean = np.mean(all_speed_neg, axis=1) 
                    all_speed_neg_mean = all_speed_neg_mean/(np.linalg.norm(all_speed_neg_mean, axis=-1)[:,None] + 1e-8) 
                    
                    # build tangent vectors.
                    all_neg_mean_tangent_vects = tra3Dtools.construct_surface_tangent_vectors(meantracks_ve[neg_reg_id], 
                                                                                              unwrap_params_ve, 
                                                                                              all_speed_neg_mean[:,None,:], 
                                                                                              K_neighbors=1, 
                                                                                              nbr_model=surf_nbrs_ve, 
                                                                                              pts3D_all=surf_nbrs_ve_3D_pts, 
                                                                                              dist_sample=10, 
                                                                                              return_dist_model=False)
                    
#                    all_speed_component = np.vstack([all_speed_neg[ll].dot(all_speed_neg_mean[ll]) for ll in range(len(all_speed_neg))]) # n_superpixels x time
                    all_speed_component = np.sum(all_speed_neg*all_neg_mean_tangent_vects[:,:-1], axis=-1)

#                    plt.figure()
#                    plt.plot(ve_component.mean(axis=0))
#                    plt.plot(all_speed_component.mean(axis=0))
#                    plt.show()
    
                    """
                    Speed correction for LifeAct timing. 
                    """
                    if 'LifeAct' in rootfolder:
                        ve_component = ve_component * 2
                        all_speed_component = all_speed_component * 2 
                        
                    
#                    diff_curve_component = np.cumsum(ve_component.mean(axis=0)) - np.cumsum(all_speed_component.mean(axis=0))
                    # mean is very much not robust!. 
                    diff_curve_component = np.cumsum(np.median(ve_component,axis=0)) - np.cumsum(np.median(all_speed_component,axis=0))
    #                diff_curve_component = np.cumsum(ve_component.mean(axis=0)).copy()
#                    diff_curve_component[diff_curve_component<0] = 0
                    change_curve_component = np.diff(diff_curve_component)
                    
                    diff_curves_all.append(diff_curve_component)
#                    all_embryos_plot.append(embryo_folder.split('/')[-1])
                    all_embryos_plot.append(unwrapped_condition)
                    
    
    # =============================================================================
    #                 offline changepoint detection?
    # =============================================================================
                    from scipy.signal import savgol_filter
                    import bayesian_changepoint_detection.offline_changepoint_detection as offcd
                    from functools import partial
                    from scipy.signal import find_peaks
                    from scipy.interpolate import UnivariateSpline
                    
    #                spl = UnivariateSpline(np.arange(len(diff_curve_component)), diff_curve_component, k=1,s=1e-4)
    #                spl_diff_component = spl(np.arange(len(diff_curve_component)))
                    
    #                spl_diff_component = savgol_filter(diff_curve_component, 11,1)
    #                diff_curve_component = savgol_filter(diff_curve_component, 15,1)
                    
                    # switch with an ALS filter?
    #                diff_curve_component_ = baseline_als(diff_curve_component, lam=2.5, p=0.5, niter=10) # TTR?
                    diff_curve_component_ = baseline_als(diff_curve_component, lam=5, p=0.5, niter=10)
    #                diff_curve_component_ = baseline_als(diff_curve_component, lam=5, p=0.5, niter=10) # use lam=5 for lifeact, what about lam=1?
                    
                    plt.figure()
                    plt.plot(diff_curve_component)
    #                plt.plot(spl_diff_component)
                    plt.show()
                    
                    # what about detection by step filter convolving (matched filter)
                    up_step = np.hstack([-1*np.ones(len(diff_curve_component)//2), 1*np.ones(len(diff_curve_component)-len(diff_curve_component)//2)])
    #                up_step = np.hstack([-1*np.ones(len(diff_curve_component)), 1*np.ones(len(diff_curve_component)-len(diff_curve_component))])
                    down_step = np.hstack([1*np.ones(len(diff_curve_component)//2), -1*np.ones(len(diff_curve_component)-len(diff_curve_component)//2)])
                    
                    conv_res = np.convolve((diff_curve_component - np.mean(diff_curve_component))/(np.std(diff_curve_component)+1e-8)/len(diff_curve_component), down_step, mode='same')
                    
                    
                    print(len(diff_curves_all))           
                    # put in the filtered. 
                    in_signal = np.diff(diff_curve_component_)
    #                in_signal =  np.diff(np.cumsum(in_signal)/(np.arange(len(in_signal)) + 1))
                    conv_res_up = np.convolve((in_signal - np.mean(in_signal))/(np.std(in_signal)+1e-8)/len(in_signal), down_step, mode='same')
                    conv_res_down = np.convolve((in_signal - np.mean(in_signal))/(np.std(in_signal)+1e-8)/len(in_signal), up_step, mode='same')
                    
    #                peaks_conv = find_peaks(np.abs(conv_res_up), distance=5, height=0.15, prominence=0.05)[0]
                    peaks_conv = find_peaks(np.abs(conv_res_up), distance=5, height=0.15)[0]
                    peaks_conv = np.hstack([p for p in peaks_conv if p>2 and p < len(diff_curve_component_)-2])
                    
    #                peaks_conv = find_peaks(np.abs(conv_res_up), distance=2.5, height=0.2)[0]
    #                
    #                break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=1.25e-3) # make this automatic... 
                    
                    # try the single threshold? 
#                    break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=4.5e-3) # make this automatic... 
#                    break_grads = test_VE_piecewise_gradient_refined(in_signal, peaks_conv, thresh_grad_buffer=4.5e-3)
                    break_grads = test_VE_piecewise_gradient_refined(in_signal, peaks_conv, thresh_grad_buffer=2.5e-3)
#                    break_grads = test_VE_piecewise_gradient_refined(in_signal, peaks_conv, thresh_grad_buffer=3e-3)
#                    break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=2.7e-3) # make this automatic... 
                    
                    # save out all the candidates. 
                    break_grads_all.append(break_grads[0])
                    break_pts_all.append(break_grads[-1])
    #                zzz = test_unique_break_segments(in_signal, peaks_conv, p_thresh=0.05)
    #                zz = test_unique_break_segments(in_signal, peaks_conv)
    #                print(test_unique_break_segments(in_signal, peaks_conv))
    #                break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=3e-3)
    #                break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=2.5e-3)
    #                if len(break_grads[-1][1]) > 0:
    #                    break_grads = test_VE_piecewise_gradient(in_signal, peaks_conv, thresh_grad_buffer=.5*np.max(break_grads[0])) # 10% the max? 
    #                    print(.5*np.max(break_grads[0]))
                    
                    regime_colors = ['r','g','b']
                    
                    plt.figure(figsize=(4,7))
                    plt.subplot(411)
#                    plt.plot(np.cumsum(ve_component.mean(axis=0)), label='AVE')
#                    plt.plot(np.cumsum(all_speed_component.mean(axis=0)), label='Non-AVE')
                    plt.plot(np.cumsum(np.median(ve_component, axis=0)), label='AVE')
                    plt.plot(np.cumsum(np.median(all_speed_component,axis=0)), label='Non-AVE')
                    plt.legend()
    
                    plt.subplot(412)
                    plt.plot(diff_curve_component)
                    plt.plot(peaks_conv, diff_curve_component[peaks_conv], 'go')
                    
                    for r_i in range(len(break_grads[-1])):
                        for rr in break_grads[-1][r_i]:
                            plt.fill_between(rr[1:-1], np.min(diff_curve_component), np.max(diff_curve_component), color=regime_colors[r_i], alpha=0.25)
                    
                    plt.subplot(413)
                    plt.plot(in_signal)
                    plt.plot(break_grads[1])
                    plt.hlines(0, 0, len(in_signal))
                    plt.subplot(414)
                    plt.plot(np.abs(conv_res_up))
                    plt.plot(peaks_conv, np.abs(conv_res_up)[peaks_conv], 'go')
    #                plt.plot(conv_res_down)
                    plt.ylim([0,1])
                    plt.savefig(os.path.join(saveallplotsfolder, 'MOSES_staging_plot_%s.svg' %(os.path.split(ve_unwrap_file)[-1].split('.tif')[0])), 
                                dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    
                    
                    # plot again prettified for publication. 
                    if 'LifeAct' in rootfolder:
                        Ts = 5./60
                    else:
                        Ts = 10./60
                        
                    Ts_curve = 10./60
                    
                    xplot = np.arange(len(conv_res_up)) * Ts
                    min_shade = -0.1
                    max_shade = 1.1
                    
                    plt.figure(figsize=(4,7))
                    plt.subplot(411)
                    plt.plot(xplot, np.cumsum(np.median(ve_component, axis=0)), color='g', lw=2, label='AVE')
                    plt.plot(xplot, np.cumsum(np.median(all_speed_component,axis=0)), color='k', lw=2, label='Non-AVE')
                    plt.tick_params(length=5, right=True, labelbottom=False)
                    plt.yticks(fontsize=12, fontname='Liberation Sans')
                    plt.ylim([-0.1, 1.1])
                    plt.xlim([0*Ts, (ve_component.shape[1]-1)*Ts])
                    plt.ylabel('$d_{direct}$')
                    plt.legend()
    
                    plt.subplot(412)
                    plt.plot(xplot, diff_curve_component, color='g', lw=2, label='AVE-nonAVE diff')
                    plt.plot(xplot[peaks_conv], diff_curve_component[peaks_conv], 'ko')
                    
                    for r_i in range(len(break_grads[-1])):
                        for rr in break_grads[-1][r_i]:
                            plt.fill_between([xplot[rr[1]], xplot[np.minimum(len(xplot)-1, rr[2])]], 
                                              min_shade, max_shade, color=regime_colors[r_i], alpha=0.25)
                    plt.xlim([0*Ts, (ve_component.shape[1]-1)*Ts])
                    plt.ylim([-0.1, 1.1])
                    plt.tick_params(length=5, right=True, labelbottom=False)
                    plt.ylabel('$d^{AVE}_{direct} - d^{nonAVE}_{direct}$')
                    plt.legend()
                    
                    plt.subplot(413)
                    plt.plot(xplot[:-1], in_signal/Ts_curve, 'g-', label='$\Delta$diff')
                    plt.plot(xplot[:-1], break_grads[1]/Ts_curve, 'k-', label='Mean $\Delta$diff')
                    plt.hlines(0, 0, len(in_signal)*Ts, linestyles='dashed', alpha=0.5)
                    plt.hlines(2.5e-3/Ts_curve, 0, len(in_signal)*Ts, colors='r', linestyles='dashed', alpha=0.5)
                    for r_i in range(len(break_grads[-1])):
                        for rr in break_grads[-1][r_i]:
                            plt.vlines(xplot[rr[1]], -0.02, 0.04, 
                                       colors=regime_colors[r_i], 
                                       alpha=.5, 
                                       linestyles='dashed')
                    
                    plt.xlim([0*Ts, (ve_component.shape[1]-1)*Ts])
                    plt.tick_params(length=5, right=True, labelbottom=False)
                    plt.ylim([-0.1, 0.2])
                    plt.ylabel('$d(\Delta d_{direct}) /d t$')
                    plt.legend()
                    
                    plt.subplot(414)
                    plt.plot(xplot, np.abs(conv_res_up), 'k', lw=1)
                    plt.plot(xplot[peaks_conv], np.abs(conv_res_up)[peaks_conv], 'ko')
    #                plt.plot(conv_res_down)
                    for r_i in range(len(break_grads[-1])):
                        for rr in break_grads[-1][r_i]:
                            plt.vlines(xplot[rr[1]], 0, 1, 
                                       colors=regime_colors[r_i], 
                                       alpha=.5, 
                                       linestyles='dashed')
                    plt.ylim([0,1])
                    plt.tick_params(length=5, right=True)
                    plt.xlim([0*Ts, (ve_component.shape[1]-1)*Ts])
                    plt.xlabel('Time [h]', fontname='Liberation Sans')
                    plt.ylabel('Breakpoint Prob.', fontname='Liberation Sans')
                    plt.savefig(os.path.join(saveallplotsfolder, 'MOSES_staging_plot_%s-um.svg' %(os.path.split(ve_unwrap_file)[-1].split('.tif')[0])), 
                                dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    # compute the projected distance speed... # do we need to be cleverer? 
                    merged_timepts = merge_time_points(np.arange(len(vid_ve)), break_grads[-1])
#                    merged_timepts = [m[:-1] for m in merged_timepts]
                    
                    
                    print('predicted time stages')
                    print(merged_timepts)
                    print('=============')
                    pre_all_auto.append(merged_timepts[0])
                    mig_all_auto.append(merged_timepts[1])
                    post_all_auto.append(merged_timepts[2])
                
                
    """
    check diff curves for all . 
    """
    pre_data = []; 
    for d in pre_all_auto:
        if len(d) == 0:
            pre_data.append(np.nan)
        else:
            pre_data.append('_'.join([str(p+1) for p in d]))
    pre_data = np.hstack(pre_data)
    
    mig_data = []; 
    for d in mig_all_auto:
        if len(d) == 0:
            mig_data.append(np.nan)
        else:
            mig_data.append('_'.join([str(p+1) for p in d]))
    mig_data = np.hstack(mig_data)
            
    post_data = []; 
    for d in post_all_auto:
        if len(d) == 0:
            post_data.append(np.nan)
        else:
            post_data.append('_'.join([str(p+1) for p in d]))
    post_data = np.hstack(post_data)
    
    all_table_staging_auto = pd.DataFrame(np.vstack([all_embryos_plot, 
                                                       pre_data,
                                                       mig_data,
                                                       post_data]).T, columns=['Embryo', 'Pre-migration', 'Migration to Boundary', 'Post-Boundary'])
    
#    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES.xlsx', index=None)
#    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-LifeAct-smooth-lam1.xlsx', index=None)
                    
    if 'LifeAct' in rootfolder:
        all_table_staging_auto.to_excel(os.path.join(saveallplotsfolder, 'auto_staging_MOSES-LifeAct-smooth-lam5-correct_norot.xlsx'), index=None)
        savematfile = os.path.join(saveallplotsfolder, 'auto_staging_curves_MOSES-LifeAct-correct_norot.mat')
        spio.savemat(savematfile, {'embryos': all_embryos_plot, 
                                   'staging_table': all_table_staging_auto, 
                                   'staging_curves': diff_curves_all,
                                   'break_grads_all':break_grads_all, 
                                   'break_pts_all':break_pts_all})
    
    if 'MTMG-TTR' in rootfolder:
        all_table_staging_auto.to_excel(os.path.join(saveallplotsfolder, 'auto_staging_MOSES-MTMG-TTR-smooth-lam5-correct_norot.xlsx'), index=None)
        savematfile = os.path.join(saveallplotsfolder, 'auto_staging_curves_MOSES-MTMG-TTR-correct_norot.mat')
        spio.savemat(savematfile, {'embryos': all_embryos_plot, 
                                   'staging_table': all_table_staging_auto, 
                                   'staging_curves': diff_curves_all,
                                   'break_grads_all':break_grads_all, 
                                   'break_pts_all':break_pts_all})
    
    if 'MTMG-HEX' in rootfolder and 'new_hex' not in rootfolder:
        all_table_staging_auto.to_excel(os.path.join(saveallplotsfolder, 'auto_staging_MOSES-MTMG-HEX-smooth-lam5-correct_norot.xlsx'), index=None)
        savematfile = os.path.join(saveallplotsfolder, 'auto_staging_curves_MOSES-MTMG-HEX-correct_norot.mat')
        spio.savemat(savematfile, {'embryos': all_embryos_plot, 
                                   'staging_table': all_table_staging_auto, 
                                   'staging_curves': diff_curves_all,
                                   'break_grads_all':break_grads_all, 
                                   'break_pts_all':break_pts_all})
    
    if 'MTMG-HEX' in rootfolder and 'new_hex' in rootfolder:
        all_table_staging_auto.to_excel(os.path.join(saveallplotsfolder, 'auto_staging_MOSES-MTMG-new-HEX-smooth-lam5-correct_norot.xlsx'), index=None)
        savematfile = os.path.join(saveallplotsfolder, 'auto_staging_curves_MOSES-MTMG-new-HEX-correct_norot.mat')
        spio.savemat(savematfile, {'embryos': all_embryos_plot, 
                                   'staging_table': all_table_staging_auto, 
                                   'staging_curves': diff_curves_all,
                                   'break_grads_all':break_grads_all, 
                                   'break_pts_all':break_pts_all
                                   })
    

#    
# =============================================================================
#     investigate global threshold
# =============================================================================
    break_grads_all_flat = np.hstack(break_grads_all)
    from skimage.filters import threshold_otsu
    
    print(np.mean(break_grads_all_flat))
    print(threshold_otsu(break_grads_all_flat))
    
    

##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_MOSES-MTMG-TTR-smooth-lam10-correct_rot.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_MOSES-MTMG-HEX-smooth-lam10-correct_rot.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_MOSES-MTMG-new-HEX-smooth-lam10-correct_norot.xlsx', index=None)
##    all_table_staging_auto.to_excel('/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Paper/auto_staging_MOSES-MTMG-new-HEX-smooth-lam10-correct_norot.xlsx', index=None)
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
#    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Paper/auto_staging_curves_MOSES-LifeAct-correct_norot.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_curves_MOSES-MTMG-TTR-correct_rot.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_curves_MOSES-MTMG-HEX-correct_rot.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_rot/auto_staging_curves_MOSES-MTMG-new-HEX-correct_norot.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Paper/auto_staging_curves_MOSES-MTMG-new-HEX-correct_norot.mat'
#
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES/auto_staging_curves_MOSES-MTMG-TTR-correct_norot.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-LifeAct.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-MTMG-TTR.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-MTMG-HEX-new_hex.mat'
##    savematfile = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_curves_MOSES-MTMG-HEX.mat'
#    spio.savemat(savematfile, {'embryos': all_embryos_plot, 
#                               'staging_table': all_table_staging_auto, 
#                               'staging_curves': diff_curves_all})


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

    
