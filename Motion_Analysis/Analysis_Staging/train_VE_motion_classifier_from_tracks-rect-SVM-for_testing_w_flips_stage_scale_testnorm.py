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
            
    return tissue, emb_no, ang, proj_type

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


def clean_hex_mask(binary):
    
    from skimage.morphology import binary_closing, disk
    from skimage.measure import regionprops, label
    from scipy.ndimage.morphology import binary_fill_holes
    
    binary_ = binary_closing(binary, disk(5))
    labelled = label(binary_>0)
    region_properties = regionprops(labelled)
    
    uniq_labels = np.unique(labelled)[1:]
    reg_areas = np.hstack([re.area for re in region_properties])
    
    reg_id_largest_area = uniq_labels[np.argmax(reg_areas)]
    binary_mask = labelled == reg_id_largest_area
    binary_mask = binary_fill_holes(binary_mask)
    
    return binary_mask
    
def find_time_string(name):
    
    import re
    
    time_string = re.findall(r't\d+', name)
    time_string = time_string[0]
#    if len(time_string):
    time = int(time_string.split('t')[1])
        
    return time
    
def construct_recover_xyz_tracks(meantracks_ve, rev_unwrapparam_folder, proj_type, tissue_type, key='.mat'):
    
    import glob 
    import scipy.io as spio
    
    # sort the frame numbers.
    unwrapfiles = np.hstack(glob.glob(os.path.join(rev_unwrapparam_folder, '*'+key)))
    frame_nos = np.hstack([find_time_string(f) for f in unwrapfiles])
    unwrapfiles = unwrapfiles[np.argsort(frame_nos)]
    
#    print(unwrapfiles)
    # iterate over the files. 
    meantracks_new = []
    
    for ii in range(len(unwrapfiles)):
        retrieve_key = tissue_type+'_xyz_'+proj_type+'_rev'
#        print(retrieve_key)
        unwrap_xyz = spio.loadmat(unwrapfiles[ii])[retrieve_key]
            
#        if ii == 0:
#            plt.figure()
#            plt.imshow(unwrap_xyz[...,0])
#            plt.figure()
#            plt.imshow(unwrap_xyz[...,1])
#            plt.figure()
#            plt.imshow(unwrap_xyz[...,2])
#            plt.show()
#        print()
        unwrap_xyz = unwrap_xyz[...,[1,0,2]]
        
        if proj_type == 'rect':
            unwrap_xyz = unwrap_xyz[::-1]
#        print(ii)
#        print(ii, meantracks_ve[:,ii,0].max(), meantracks_ve[:,ii,1].max())
        track_xyz = unwrap_xyz[meantracks_ve[:,ii,0],
                               meantracks_ve[:,ii,1],:]
        
        meantracks_new.append(track_xyz[:,None,:])
        
        
    meantracks_new = np.concatenate(meantracks_new, axis = 1)
    
    return meantracks_new

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


def parse_temporal_scaling(tformfile):
    
    import scipy.io as spio 
    from transforms3d.affines import decompose44
    
    obj = spio.loadmat(tformfile)
    scales = np.hstack([1./decompose44(matrix)[2][0] for matrix in obj['tforms']])
    
    return scales
    
# =============================================================================
# =============================================================================
# # Create a bunch of utility functions for training. 
# =============================================================================
# =============================================================================
def load_vid_and_tracks_folders_htmg(embryo_folders, proj_type, unwrap_folder_key='geodesic-rotmatrix', rev_tracks='demons', staging_table=None, staging_stage=None):
    
    # embryo metadata
    all_embryos = []
    
    # hex associated masks. 
    all_hex_ve_masks = [] 
    all_hex_ve_track_select = [] 
    
    # tracks.
    all_meantracks_ve_2D = []
    all_meantracks_ve_3D = []
    all_meantracks_ve_select = [] 
    
    # shapes
    all_2d_shapes = []
    all_3d_shapes = []
    all_vol_scales = []
    

    # load up all the video and associated tracks for training. 
    for embryo_folder in embryo_folders[:]:
        
        # find the geodesic-rotmatrix. 
        unwrapped_folder = locate_unwrapped_im_folder(embryo_folder, key=unwrap_folder_key)
        
        # get one of these to get the size of the volume. 
        embryo_im_folder = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step6_' in name or 'step6tf_' in name or 'step6tf-' in name or 'step6-' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
        embryo_im_folder_files = np.hstack(glob.glob(os.path.join(embryo_im_folder[0], '*.tif')))
        embryo_im_shape = np.hstack(fio.read_multiimg_PIL(embryo_im_folder_files[0]).shape)
        embryo_im_shape = embryo_im_shape[[1,0,2]]#.transpose([1,0,2])
#        print(embryo_im_shape)
#        all_3d_shapes.append(np.hstack(embryo_im_shape))
        
        if len(unwrapped_folder) == 1:
            print('processing: ', unwrapped_folder[0])
            
            unwrapped_folder = unwrapped_folder[0]
            unwrapped_params_folder = unwrapped_folder.replace(unwrap_folder_key, 
                                                               'unwrap_params_geodesic-rotmatrix')
            
            
            """
            get the temporal transforms and extract the scaling parameters. 
            """
            temp_tform_files = glob.glob(os.path.join(embryo_folder, '*temp_reg_tforms*.mat'))
            assert(len(temp_tform_files) > 0)
            
            temp_tform_scales = np.vstack([parse_temporal_scaling(tformfile) for tformfile in temp_tform_files])
            temp_tform_scales = np.prod(temp_tform_scales, axis=0)
            
            """
            Set the folders and load up the unwrapped files
            """
            # determine the savefolder. 
#            saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
            saverevparamfolder = os.path.join(embryo_folder, 'unwrapped', 'geodesic_VE_Epi_matched-resize-rotmatrix_revmap')

            
            unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))
            unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
            """
            pair up the unwrapped files such that the same unwrap files belong, one for ve, epi and hex. 
            """
            paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_hex_embryo(unwrap_im_files, unwrap_meta_files)
            
            
            for ii in tqdm(range(len(paired_unwrap_file_condition))[:]):
                
                file_condition = paired_unwrap_file_condition[ii]
                unwrap_file_set = paired_unwrap_files[ii] # in the order of 'VE', 'Epi', 'Hex'
                
                if file_condition.split('-')[-1] == proj_type:
                    
                # =============================================================================
                #   Parse out the staging information.            
                # =============================================================================
                    proj_condition = file_condition.split('-')
                    
                    # get whether we need to invert. 
                    if 'Emb' not in proj_condition[1]:
                        embryo_name = proj_condition[0]
                    else:
                        embryo_name = proj_condition[0]+'_E'+proj_condition[1].split('Emb')[1]
                    
                    table_emb_names = np.hstack([str(s) for s in staging_table['Embryo'].values])
                    select_stage_row = manual_staging_tab.loc[table_emb_names==embryo_name]
                    
                    # get the embryo inversion 
                    embryo_inversion = select_stage_row['Inverted'].values[0] > 0 
                    stage_start_end = np.hstack(parse_start_end_times(select_stage_row[staging_stage].values)[0]).astype(np.int)-1
                    
                    print(stage_start_end)
                # =============================================================================
                #   Load the tracks            
                # =============================================================================
                    all_embryos.append(file_condition)
                    
                    """
                    1. read the hex video in to get the VE mask. 
                    """
                    # load the hex image file
                    hex_im_file = unwrap_file_set[-1]
                    hex_vid = fio.read_multiimg_PIL(hex_im_file)
                    hex_vid = hex_vid[stage_start_end[0]:stage_start_end[1]+1]
                    
                    hex_vid_max = hex_vid.max(axis=0)
#                    hex_vid_max = hex_vid[0]
#                    hex_mask_0 = hex_vid[0] >= np.mean(hex_vid[0])
#                    hex_mask_0 = hex_vid_max >= threshold_otsu(hex_vid_max)
                    hex_mask_0 = hex_vid_max >= np.mean(hex_vid_max)
                    
                    if embryo_inversion:
                        print('inverting')
                        plt.figure()
                        plt.imshow(hex_mask_0)
                        plt.show()
                        if proj_type == 'polar':
                            hex_mask_0 = hex_mask_0[::-1] # flip y 
                        if proj_type == 'rect':
                            hex_mask_0 = hex_mask_0[:,::-1] # flip x
                            
                        plt.figure()
                        plt.imshow(hex_mask_0)
                        plt.show()
                    
#                    hex_mask_0 = clean_hex_mask(hex_mask_0)
                    all_hex_ve_masks.append(hex_mask_0)
                    
                    ### give the shape info
                    all_2d_shapes.append(np.hstack(hex_mask_0.shape))
                    all_3d_shapes.append(np.hstack(embryo_im_shape))
                    
                    """
                    2. read in the VE trackfile
                    """
                    ve_im_file = unwrap_file_set[0]
                    
                    ve_vid = fio.read_multiimg_PIL(ve_im_file)
                    ve_vid_mask = ve_vid[stage_start_end[0]] < 1
                    
                    if embryo_inversion:
                        if proj_type == 'polar':
                            ve_vid_mask = ve_vid_mask[::-1]
                        if proj_type == 'rect':
                            ve_vid_mask = ve_vid_mask[:,::-1]
                    
#                    staging_stage
#                    ve_track_file = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels)+os.path.split(ve_im_file)[-1].replace('.tif', '.mat'))
                    savetrackfolder_ve_stage = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-1000_%s-manual' %(staging_stage))
#                    ve_track_file = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_%s_' %(n_spixels,staging_stage)+os.path.split(ve_im_file)[-1].replace('.tif', '.mat'))
                    ve_track_file = os.path.join(savetrackfolder_ve_stage, 'deepflow-meantracks-%d_%s_' %(n_spixels,staging_stage)+file_condition+'.mat')
                    
#                    meantracks_ve = spio.loadmat(ve_track_file)['meantracks']; #all_meantracks_ve_2D.append(meantracks_ve)
                    meantracks_ve = spio.loadmat(ve_track_file)['meantracks_ve']
                    start_end = spio.loadmat(ve_track_file)['start_end'].ravel()
                    print(start_end)
                    print(meantracks_ve.shape, start_end[1]-start_end[0] + 1)
                    print(temp_tform_scales[start_end[0]:start_end[1]].shape)
                    all_vol_scales.append(temp_tform_scales[start_end[0]:start_end[1]+1])
                    
#                    meantracks_ve = meantracks_ve[:,stage_start_end[0]:stage_start_end[1]+1]
#                    meantracks_ve[...,0] = np.clip(meantracks_ve[...,0], 0, ve_vid.shape[1]-1)
#                    meantracks_ve[...,1] = np.clip(meantracks_ve[...,1], 0, ve_vid.shape[2]-1)
#                    print(meantracks_ve[...,1].max(), meantracks_ve[...,1].min())
                    if embryo_inversion:
                        # flip the tracks according to polar or rect. 
                        if proj_type == 'polar':
                            meantracks_ve[...,0] = ve_vid_mask.shape[0]-1 - meantracks_ve[...,0]
                        if proj_type == 'rect': 
                            meantracks_ve[...,1] = ve_vid_mask.shape[1]-1 - meantracks_ve[...,1]
                            
#                    meantracks_ve = meantracks_ve.astype(np.float32)
                    all_hex_ve_track_select.append(hex_mask_0[meantracks_ve[:,0,0],
                                                              meantracks_ve[:,0,1]])
                    
                    meantracks_ve_filt = ve_vid_mask[meantracks_ve[:,0,0], 
                                                     meantracks_ve[:,0,1]]
#                    meantracks_ve[meantracks_ve_filt] = meantracks_ve[meantracks_ve_filt,0][:,None]
                    all_meantracks_ve_select.append(meantracks_ve_filt)
                    
                    """
                    4. read in the reversed mapped parameter folders. 
                    """
#                    rev_unwrapparam_folder = os.path.join(saverevparamfolder, 'L%s_ve_%s_unwrap_align_params_geodesic' %(file_condition.split('-')[0], file_condition.split('-')[1]))
                    # search for the correct angle. 
                    rev_unwrapparam_folders = os.listdir(saverevparamfolder)
                    rev_unwrapparam_folder = [os.path.join(saverevparamfolder, folder) for folder in rev_unwrapparam_folders if file_condition.split('-')[2] in folder][0]
##                    rev_unwrapparam_folder = os.path.join(saverevparamfolder, 'L%s_ve_%s_unwrap_align_params_geodesic' %(file_condition.split('-')[0], file_condition.split('-')[1]))
##                    all_xyz_position_folders.append(rev_unwrapparam_folder)
#                    print(rev_unwrapparam_folder)
#                    print(file_condition)
                    if rev_tracks=='original':
#                        print(meantracks_ve.shape)
#                        print(rev_tracks)
                        # use the projected xyz files and use the meantracks to get back to the true positions. 
                        meantracks_ve_3D = construct_recover_xyz_tracks(meantracks_ve, rev_unwrapparam_folder,
                                                                         proj_type=proj_type, 
                                                                         tissue_type='ve', key='.mat')
                        # we should normalize? 
                        meantracks_ve_3D_filt = meantracks_ve_3D.copy()#([meantracks_ve_filt]
                        meantracks_ve_3D_filt = meantracks_ve_3D_filt.astype(np.float32)
                        # normalise the readings.
                        for kk in range(meantracks_ve_3D_filt.shape[-1]):
                            meantracks_ve_3D_filt[...,kk] = meantracks_ve_3D_filt[...,kk] / float(embryo_im_shape[kk]) 
                        all_meantracks_ve_3D.append(meantracks_ve_3D_filt)
                    if rev_tracks == 'demons':
                        unwrap_param_file = ve_im_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
                        unwrap_params_ve = spio.loadmat(unwrap_param_file)
                        unwrap_params_ve = unwrap_params_ve['ref_map_'+proj_type+'_xyz']
                        
                        if proj_type == 'rect':
                            unwrap_params_ve = unwrap_params_ve[::-1]
                            
                        if embryo_inversion: 
                            if proj_type == 'rect':
                                unwrap_params_ve = unwrap_params_ve[:,::-1]
                            if proj_type == 'polar':
                                unwrap_params_ve = unwrap_params_ve[::-1]
                        
                        
                        print(meantracks_ve[...,1].max())
                        meantracks_ve_3D = unwrap_params_ve[meantracks_ve[...,0],
                                                            meantracks_ve[...,1]]
                        
                        # we should normalize? 
                        meantracks_ve_3D_filt = meantracks_ve_3D.copy()#[meantracks_ve_filt]
                        meantracks_ve_3D_filt = meantracks_ve_3D_filt.astype(np.float32)
                        meantracks_ve_3D_filt = meantracks_ve_3D_filt / float( (np.product(embryo_im_shape))**(1./3))
#                         normalise the readings.
#                        for kk in range(meantracks_ve_3D_filt.shape[-1]):
#                            meantracks_ve_3D_filt[...,kk] = meantracks_ve_3D_filt[...,kk] / float(embryo_im_shape[kk]) 
                        all_meantracks_ve_3D.append(meantracks_ve_3D_filt)
                    
                    # normalize the 2d. 
                    meantracks_ve = meantracks_ve.astype(np.float32)
#                    meantracks_ve[...,0] = meantracks_ve[...,0] / float(hex_mask_0.shape[0])
#                    meantracks_ve[...,1] = meantracks_ve[...,1] / float(hex_mask_0.shape[1])
                    meantracks_ve = meantracks_ve/(float(np.sqrt(np.product(hex_mask_0.shape))))
#                    all_meantracks_ve_2D_select.append(meantracks_ve_filt)
                    all_meantracks_ve_2D.append(meantracks_ve)
                    
                    if rev_tracks == 'none':
                        all_meantracks_ve_3D.append(meantracks_ve)
#                    all_meantracks_ve_select.append(meantracks_ve_filt)
                    
    return all_embryos, (all_hex_ve_masks, all_hex_ve_track_select), (all_meantracks_ve_2D, all_meantracks_ve_3D, all_meantracks_ve_select), (np.vstack(all_2d_shapes), np.vstack(all_3d_shapes), all_vol_scales)
    

def construct_track_classifier_velocity_feats( tracks, label_track_masks, im_shapes, growth_correction=None, track_2D=None, label_masks=None, tracks_select=None, neighbor_model=None, debug_viz=True):
    
#    from MOSES.Visualisation_Tools.track_plotting import plot_tracks
    # get the number of tracks. 
    N = len(tracks)
    
    # storing the features.
    all_X = []
    all_Y = []
    
    for ii in range(N):
        
        # 1. get the raw tracks and label information provided. 
        track = tracks[ii]
        if tracks_select is not None:
            track_select = np.logical_not(tracks_select[ii])
        else:
            track_select = np.ones(len(track), dtype=np.bool)
        
        label = label_track_masks[ii]
        
        # 2. generate the velocity features. 
        X_feats = track[track_select]
        # compute velocity
        X_feats = (X_feats[:,1:] - X_feats[:,:-1]).astype(np.float32)
        
        if growth_correction is not None:
            # this wo
            print(ii, len(growth_correction[ii]))
            X_feats = X_feats*growth_correction[ii][None,1:,None] # original only time but needs to apply to all. 
        
        # mean velocity
        X_feat_mean = X_feats.mean(axis=1)

        # 3. optionally construct the neighbourhood features
        if neighbor_model is not None:
            neighbor_model.fit(track[track_select, 0]) # use the initial topology 
            nbr_inds = neighbor_model.kneighbors(track[track_select,0], return_distance=False)
            
            X_feat_mean_ = X_feat_mean[nbr_inds]
            # reshape into vectors. 
            X_feat_mean_ = X_feat_mean_.reshape(-1,nbr_inds.shape[1] * X_feats.shape[-1])
            Y_label = label[track_select]
            
            all_X.append(X_feat_mean_)
        else:
            
            all_X.append(X_feat_mean)
            
        if debug_viz:
            
            track_2D_vis = track_2D[ii][track_select]
            disps_track_2D_vis = (track_2D_vis[:,1:] - track_2D_vis[:,:-1]).astype(np.float32)
            mean_disps_2D_vis = np.mean(disps_track_2D_vis, axis=1)
            
            fig, ax = plt.subplots(figsize=(10,10))
            ax.imshow(label_masks[ii])
            ax.plot(track_2D_vis[np.logical_not(label[track_select]),0,1]*im_shapes[ii][1], 
                    track_2D_vis[np.logical_not(label[track_select]),0,0]*im_shapes[ii][0], 'w.')
            ax.plot(track_2D_vis[label[track_select],0,1]*im_shapes[ii][1], 
                    track_2D_vis[label[track_select],0,0]*im_shapes[ii][0], 'go')
            ax.quiver(track_2D_vis[:,0,1]*im_shapes[ii][1], 
                      track_2D_vis[:,0,0]*im_shapes[ii][0], 
                      mean_disps_2D_vis[:,1]*im_shapes[ii][1],
                     -mean_disps_2D_vis[:,0]*im_shapes[ii][0], color = 'r')
#            plot_tracks(track_2D_vis, ax=ax, color='r')
            plt.show()

        all_Y.append(Y_label)
        
    all_X = np.vstack(all_X)
    all_Y = np.hstack(all_Y)
        
    return all_X, all_Y


def train_VE_classifier_SVM( X, Y, svm_kernel='rbf', svm_C=1., degree=3, gamma='scale', class_weight='balanced'):
    
    from sklearn.svm import SVC 
    
    svm_clf = SVC(C=svm_C, kernel=svm_kernel,
                  degree = degree,
                  gamma=gamma, 
                  coef0=0.0, 
                  shrinking=True,
                  probability=False, 
                  tol=0.001,
                  cache_size=200,
                  class_weight=class_weight,
                  verbose=False, max_iter=-1, 
                  decision_function_shape='ovr', 
                  random_state=0)
    
    svm_clf.fit(X, Y)
    
    return svm_clf


def train_VE_classifier_RF( X, Y, svm_kernel='rbf', svm_C=1., degree=3, gamma='scale', class_weight='balanced'):
    
#    from sklearn.svm import SVC 
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    
#    svm_clf = RandomForestClassifier(n_estimators=100, 
#                                     criterion='entropy', 
#                                     class_weight='balanced', random_state=0)
    
    svm_clf = QuadraticDiscriminantAnalysis()
    
    svm_clf.fit(X, Y)
    
    return svm_clf


def predict_VE_classifier_SVM( X, clf ):
    
    y_predict = clf.predict(X)
    
    return y_predict
    

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
    from tqdm import tqdm 
    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import KFold
    from sklearn.metrics import adjusted_rand_score, accuracy_score
    from sklearn.model_selection import RepeatedKFold
    
    import pandas as pd 
    
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # #     1. Get the training Data for the different projections (polar vs rectangular)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
    
    rootfolder_1 = '/media/felix/Srinivas4/MTMG-HEX'
    embryo_folders_1 = np.hstack([os.path.join(rootfolder_1,folder) for folder in os.listdir(rootfolder_1) if os.path.isdir(os.path.join(rootfolder_1, folder)) and 'new_hex' not in folder])
        
    rootfolder_2 = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
    embryo_folders_2 = np.hstack([os.path.join(rootfolder_2,folder) for folder in os.listdir(rootfolder_2) if os.path.isdir(os.path.join(rootfolder_2, folder)) and 'new_hex' not in folder])
          
# =============================================================================
#   Load the staging information with all the various inversions 
# =============================================================================

    manual_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/manual_staging_holly_matt.xlsx'
    manual_staging_tab = pd.read_excel(manual_staging_file)

    ve_migration_stage = manual_staging_tab['Migration to Boundary'].values
    ve_migration_stage_times = parse_start_end_times(ve_migration_stage)
    ve_migration_stage_times_valid = ~np.isnan(ve_migration_stage_times[:,0])
    
    hex_staging_tab = manual_staging_tab.loc[np.logical_and(manual_staging_tab['Line'].values=='HTMG',ve_migration_stage_times_valid)]
    hex_staging_embryos = np.hstack([str(s).replace('E','Emb') for s in hex_staging_tab['Embryo'].values])


# =============================================================================
#     Take only embryo folders which have defined migration phase. 
# =============================================================================
    all_embryo_folders = np.hstack([embryo_folders_1,
                                    embryo_folders_2])
    
    
    # check the embryo folders have a migration phase.
    folder_check = []
    for ii in range(len(all_embryo_folders)):
        basename = os.path.split(all_embryo_folders[ii])[-1]
        basename = basename.split('_TP')[0].split('L')[1]
        if basename in hex_staging_embryos:
            folder_check.append(all_embryo_folders[ii])
    folder_check = np.hstack(folder_check) 
    all_embryo_folders = folder_check
    
    
    staging_phase = 'Migration to Boundary'
# =============================================================================
#     Train-testing procedure to create the K-fold
# =============================================================================    
    
    kf = RepeatedKFold(n_splits=2, n_repeats = 10, random_state = 0)
    kf.get_n_splits(all_embryo_folders)
    
    
    all_global_train_scores = []
    all_global_test_scores = []
    
    all_hex_train_scores = []
    all_hex_test_scores = []
    
    
    trained_models = []
    trained_splits = []
    train_test_files = []

    for train_index, test_index in kf.split(all_embryo_folders):

        print(train_index)
        print(test_index)
        print('----')
        trained_splits.append([train_index,test_index])
        
        train_embryo_folders = all_embryo_folders[train_index]
#    # =============================================================================
#    # =============================================================================
#        train_rootfolder = '/media/felix/Srinivas4/MTMG-HEX'
#    #    rootfolder = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
#    # =============================================================================
#    #     1. Load all the valid embryo folders. 
#    # =============================================================================
#        train_embryo_folders = np.hstack([os.path.join(train_rootfolder,folder) for folder in os.listdir(train_rootfolder) if os.path.isdir(os.path.join(train_rootfolder, folder)) and 'new_hex' not in folder])
#        
    # =============================================================================
    #     2. find valid embryo folders i.e. has a geodesic-rotmatrix. 
    # =============================================================================
        n_spixels = 1000 # set the number of pixels. 
        proj_type = 'rect'
        
        
        all_embryos_train, all_hex_masks_train, all_meantracks_train, all_shapes_train = load_vid_and_tracks_folders_htmg(train_embryo_folders, 
                                                                                                                          proj_type, 
                                                                                                                          unwrap_folder_key='geodesic-rotmatrix', 
                                                                                                                          rev_tracks='none',
                                                                                                                          staging_table=manual_staging_tab,
                                                                                                                          staging_stage=staging_phase)
        
        # separate out all the data meta-information. 
        all_2d_shapes_train, all_3d_shapes_train, all_growth_factor_train = all_shapes_train
        all_hex_ve_masks_train, all_hex_ve_track_select_train = all_hex_masks_train
        all_meantracks_ve_2D_train, all_meantracks_ve_train, all_meantracks_ve_select_train = all_meantracks_train
        
        """
        Train the classifier
        """
        nbrs_model = NearestNeighbors(n_neighbors=9, algorithm='auto')
        
        train_X, train_Y = construct_track_classifier_velocity_feats(all_meantracks_ve_train, all_hex_ve_track_select_train, 
                                                                     im_shapes=all_2d_shapes_train, 
                                                                     growth_correction = all_growth_factor_train, 
                                                                     label_masks=all_hex_ve_masks_train, 
                                                                     track_2D = all_meantracks_ve_2D_train,
                                                                     tracks_select=all_meantracks_ve_select_train, 
                                                                     neighbor_model=nbrs_model, 
                                                                     debug_viz=False)
        
    # =============================================================================
    #     3. Test embryo folder
    # =============================================================================
        
        test_embryo_folders = all_embryo_folders[test_index]
#        test_rootfolder = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
#        test_embryo_folders = np.hstack([os.path.join(test_rootfolder,folder) for folder in os.listdir(test_rootfolder) if os.path.isdir(os.path.join(test_rootfolder, folder)) and 'new_hex' not in folder])
#        
        all_embryos_test, all_hex_masks_test, all_meantracks_test, all_shapes_test = load_vid_and_tracks_folders_htmg(test_embryo_folders, 
                                                                                                                      proj_type, 
                                                                                                                      unwrap_folder_key='geodesic-rotmatrix', 
                                                                                                                      rev_tracks='none',
                                                                                                                      staging_table=manual_staging_tab,
                                                                                                                      staging_stage=staging_phase)
        
        # separate out all the data meta-information. 
        all_2d_shapes_test, all_3d_shapes_test, all_growth_factor_test = all_shapes_test
        all_hex_ve_masks_test, all_hex_ve_track_select_test = all_hex_masks_test
        all_meantracks_ve_2D_test, all_meantracks_ve_test, all_meantracks_ve_select_test = all_meantracks_test
        
        """
        Train the classifier
        """
        nbrs_model = NearestNeighbors(n_neighbors=9, algorithm='auto')
        
        test_X, test_Y = construct_track_classifier_velocity_feats(all_meantracks_ve_test, all_hex_ve_track_select_test, 
                                                                   im_shapes=all_2d_shapes_test, 
                                                                   growth_correction = all_growth_factor_test,
                                                                   label_masks=all_hex_ve_masks_test, 
                                                                   track_2D = all_meantracks_ve_2D_test,
                                                                   tracks_select=all_meantracks_ve_select_test, 
                                                                   neighbor_model=nbrs_model, 
                                                                   debug_viz=False)
        
        train_test_files.append([all_embryos_train, all_embryos_test])
        
    # =============================================================================
    #     4. Train-Test using the 5-4 embryos.
    # =============================================================================   
        
#        SVM_clf_polar = train_VE_classifier_SVM( train_X, train_Y, 
#                                                svm_kernel='rbf', 
#                                                svm_C=1., 
#                                                degree=3, gamma='scale', class_weight='balanced')
#        
#        train_VE_classifier_RF
        
        SVM_clf_polar = train_VE_classifier_RF( train_X, train_Y, 
                                                svm_kernel='rbf', 
                                                svm_C=1., 
                                                degree=3, gamma='scale', class_weight='balanced')
        
        trained_models.append(SVM_clf_polar)
        
        # Compute the test statistics. 
        predict_train = SVM_clf_polar.predict(train_X)
        predict_test = SVM_clf_polar.predict(test_X)
        
        global_train_score = accuracy_score(train_Y, predict_train, normalize=True)
        global_test_score = accuracy_score(test_Y, predict_test, normalize=True)
        global_train_randi = adjusted_rand_score(train_Y, predict_train)
        global_test_randi = adjusted_rand_score(test_Y, predict_test)
        
        print(global_train_randi, global_test_randi)
        
        N_train_Y_GT = np.sum(train_Y)
        N_train_Y_Pred = np.sum(predict_train)
        N_train_Y_Match = np.sum(np.logical_and(train_Y, predict_train))
        Precision_train = N_train_Y_Match / float(N_train_Y_Pred)
        Recall_train = N_train_Y_Match / float(N_train_Y_GT)
        Fscore_train = 2*(Precision_train*Recall_train) / float(Precision_train + Recall_train + 1e-12)
        
        
        N_test_Y_GT = np.sum(test_Y)
        N_test_Y_Pred = np.sum(predict_test)
        N_test_Y_Match = np.sum(np.logical_and(test_Y, predict_test))
        Precision_test = N_test_Y_Match / float(N_test_Y_Pred)
        Recall_test = N_test_Y_Match / float(N_test_Y_GT)
        Fscore_test = 2*(Precision_test*Recall_test) / float(Precision_test + Recall_test + 1e-12)
        
        
#        print('Training accuracy: ', accuracy_score(train_Y, predict_train, normalize=True))
#        print('Testing accuracy: ', accuracy_score(test_Y, predict_test, normalize=True))
#        
#        print('+++++++++++++')
#        print('Training Statistics ======' )
#        print ('# GT labels train: ', np.sum(train_Y))
#        print ('# Pred labels train: ', np.sum(predict_train))
#        print ('# Matched labels train: ', accuracy_score(train_Y[train_Y==1], predict_train[train_Y==1], normalize=False))
#        print('precision: %.3f, recall: %.3f train' %(accuracy_score(train_Y[train_Y==1], predict_train[train_Y==1], normalize=False)/ float(np.sum(predict_train)), accuracy_score(train_Y[train_Y==1], predict_train[train_Y==1], normalize=False)/ float(np.sum(train_Y)))) 
#        
#        
#        # also evaluate the predictions only in positive regions. 
#        print('+++++++++++++')
#        print('Testing Statistics ======' )
#        print ('# GT labels test: ', np.sum(test_Y))
#        print ('# Pred labels test: ', np.sum(predict_test))
#        print ('# Matched labels test: ', accuracy_score(test_Y[test_Y==1], predict_test[test_Y==1], normalize=False))
#        print('precision: %.3f, recall: %.3f test' %(accuracy_score(test_Y[test_Y==1], predict_test[test_Y==1], normalize=False)/ float(np.sum(predict_test)), 
#                                                     accuracy_score(test_Y[test_Y==1], predict_test[test_Y==1], normalize=False)/ float(np.sum(test_Y)))) 
        
        all_global_train_scores.append([global_train_score, global_train_randi])
        all_global_test_scores.append([global_test_score, global_test_randi])
        
        all_hex_train_scores.append([Precision_train, Recall_train, Fscore_train])
        all_hex_test_scores.append([Precision_test, Recall_test, Fscore_test])
        

    all_global_train_scores = np.vstack(all_global_train_scores)
    all_global_test_scores = np.vstack(all_global_test_scores)
    
    all_hex_train_scores = np.vstack(all_hex_train_scores)
    all_hex_test_scores = np.vstack(all_hex_test_scores)
    
    
    # save as a .csv file 
    import pandas as pd 
    
    all_score_tab = pd.DataFrame(np.hstack([all_global_train_scores, 
                                            all_global_test_scores, 
                                            all_hex_train_scores,
                                            all_hex_test_scores]), columns=['Global_Train_Accuracy', 
                                                                            'Global_Train_Randi',
                                                                            'Global_Test_Accuracy',
                                                                            'Global_Test_Randi',
                                                                            'HEX_Train_Precision', 
                                                                            'HEX_Train_Recall', 
                                                                            'HEX_Train_Fscore', 
                                                                            'HEX_Test_Precision', 
                                                                            'HEX_Test_Recall',
                                                                            'HEX_Test_Fscore'])

#    
##    all_score_tab.to_csv('SVM-poly_classifier_train-test_scores_2-fold.csv', index=None)
##    all_score_tab.to_csv('SVM-rbf_classifier_train-test_scores_2-fold.csv', index=None)
##    all_score_tab.to_csv('SVM-rbf_classifier_train-test_scores_2-fold-original3D.csv', index=None)
##    all_score_tab.to_csv('SVM-poly_classifier_train-test_scores_2-fold-none-rect.csv', index=None)
##    all_score_tab.to_csv('SVM-rbf_classifier_train-test_scores_2-fold-none-rect.csv', index=None)
##    all_score_tab.to_csv('SVM-rbf_classifier_train-test_scores_2-fold-demons-rect.csv', index=None)
#    all_score_tab.to_csv('SVM-rbf_classifier_train-test_scores_2-fold-original-rect.csv', index=None)
#       
#    
    print('global_train: ', np.mean(all_global_train_scores, axis=0), np.std(all_global_train_scores, axis=0))
    print('global_test: ', np.mean(all_global_test_scores, axis=0), np.std(all_global_test_scores, axis=0))
    
    print('hex_train: ', np.mean(all_hex_train_scores, axis=0), np.std(all_hex_train_scores, axis=0))
    print('hex_test: ', np.mean(all_hex_test_scores, axis=0), np.std(all_hex_test_scores, axis=0))
    
    
    
    # save the scores of inidividual models. 
#    from sklearn.externals import joblib 
#    
##    savemodelfolder = 'Hex-trained-VE_rect_SVM_classifier/SVM';
#    savemodelfolder = 'Hex-trained-VE_rect_SVM_classifier-flips-scale-Migration_Phase/SVM';
#    fio.mkdir(savemodelfolder)
#    
#    all_score_tab.to_csv(os.path.join(savemodelfolder, 'SVM-rbf_classifier_train-test_scores_2-fold-none-rect.csv'), index=None)
##    all_score_tab.to_csv(os.path.join(savemodelfolder, 'SVM-rbf_classifier_train-test_scores_2-fold-demons-rect.csv'), index=None)
#    
#    
#    for kk in range(len(trained_splits)):
#        joblib.dump(trained_models[kk], os.path.join(savemodelfolder, 'SVM_rect-rbf_classifier_%s.joblib' %(str(kk).zfill(3))))
#        
#    spio.savemat(os.path.join(savemodelfolder, 'SVM_rect-rbf_classifier-train_test-splits.mat'), {'train-test_splits': trained_splits})
#        
        
        
        
    
    
    
    
    
    
#    # check the hex region scores.     
#    all_hex_score_distributions = [all_hex_train_scores[:,1], 
#                                   all_hex_test_scores[:,1]]
#    
#    
#    plt.figure()
#    plt.boxplot(all_hex_score_distributions)
#    plt.show()
    
    
# =============================================================================
#     Save the models and training parameters. 
# =============================================================================
    
    
    
    
    
        
## =============================================================================
##     
## =============================================================================
##    fig, ax = plt.subplots()
##    ax.plot()
#    
##    for ii in range(N-2,N,1):
#    for ii in range(N):
#        
#        track = all_meantracks_ve[ii]
#        track_2D = all_meantracks_ve_2D[ii]
#        track_select =  np.logical_not(all_meantracks_ve_select[ii])
#        
#        hex_mask = all_hex_ve_masks[ii]
#        hex_select = all_hex_ve_track_select[ii]
#        
#        X_feats = track[track_select]
##        X_feats = (X_feats[:,1:] - X_feats[:,0][:,None]).astype(np.float32)
#        X_feats = (X_feats[:,1:] - X_feats[:,:-1]).astype(np.float32)
#        X_feat_mean = X_feats.mean(axis=1)
#        
## =============================================================================
##         neighbours
## =============================================================================
#        nbrs_model.fit(track[track_select,0])
#        nbr_inds = nbrs_model.kneighbors(track[track_select,0], return_distance=False)
#        
#        X_feat_mean_ = X_feat_mean[nbr_inds]
#        X_feat_mean_ = X_feat_mean_.reshape(-1,nbr_inds.shape[1] * X_feats.shape[-1])
#        
#        
#        Y_track = hex_select[track_select]
#        
#        predict_track = svm_clf.predict(X_feat_mean_)
#       
#        fig, ax = plt.subplots(figsize=(5,5))
#        ax.imshow(hex_mask)
#        ax.plot(track_2D[np.logical_not(hex_select),0,1]*all_2d_shapes[ii][1], 
#                track_2D[np.logical_not(hex_select),0,0]*all_2d_shapes[ii][0], 'w.')
#        ax.plot(track_2D[hex_select,0,1]*all_2d_shapes[ii][1], 
#                track_2D[hex_select,0,0]*all_2d_shapes[ii][0], 'g.')
#        ax.plot(track_2D[track_select][predict_track][:,0,1]*all_2d_shapes[ii][1], 
#                track_2D[track_select][predict_track][:,0,0]*all_2d_shapes[ii][0], 'ro')
##        ax.quiver(track_2D[:,0,1], 
##                  track_2D[:,0,0], 
##                  X_feat_mean[:,1],
##                  -X_feat_mean[:,0], color = 'r')
#        plt.show()
#    
#    
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                    

