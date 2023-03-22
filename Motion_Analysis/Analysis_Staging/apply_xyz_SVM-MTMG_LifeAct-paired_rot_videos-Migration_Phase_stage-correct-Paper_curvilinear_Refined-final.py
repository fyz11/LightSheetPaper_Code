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
#        print(ii)
#        print(ii, meantracks_ve[:,ii,0].max(), meantracks_ve[:,ii,1].max())
        if proj_type == 'rect':
            plt.figure()
            plt.imshow(unwrap_xyz[...,0])
            plt.show()
            track_xyz = unwrap_xyz[::-1][meantracks_ve[:,ii,0],
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



# =============================================================================
# =============================================================================
# # Create a bunch of utility functions for training. 
# =============================================================================
# =============================================================================
def load_vid_and_tracks_folders_mtmg_lifeact(embryo_folders, proj_type, unwrap_folder_key='geodesic-rotmatrix', rev_tracks='demons', ret_imgs=False):
    
    # embryo metadata
    all_embryos = []
    
#    # hex associated masks. 
#    all_hex_ve_masks = [] 
#    all_hex_ve_track_select = [] 
    
    # tracks.
    all_meantracks_ve_2D = []
    all_meantracks_ve_3D = []
    all_meantracks_ve_select = [] 
    
    # shapes
    all_2d_shapes = []
    all_3d_shapes = []
    
    # snapshots
    all_2d_frames = []
    

    # load up all the video and associated tracks for training. 
    for embryo_folder in embryo_folders[:]:
        
        print(embryo_folder)
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
            Set the folders and load up the unwrapped files
            """
            # determine the savefolder. 
#            saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
            saverevparamfolder = os.path.join(embryo_folder, 'unwrapped', 'geodesic_VE_Epi_matched-resize-rotmatrix_revmap')

            
            unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))
            unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
            """
            pair up the unwrapped files such that the same unwrap files belong, one for ve, epi and hex. 
            """
            paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
            
            
            for ii in tqdm(range(len(paired_unwrap_file_condition))[:]):
                
                file_condition = paired_unwrap_file_condition[ii]
                unwrap_file_set = paired_unwrap_files[ii] # in the order of 'VE', 'Epi', 'Hex'
                
                if file_condition.split('-')[-1] == proj_type:
                    
                    all_embryos.append(file_condition)
                    
#                    """
#                    1. read the hex video in to get the VE mask. 
#                    """
#                    # load the hex image file
#                    hex_im_file = unwrap_file_set[-1]
#                    hex_vid = fio.read_multiimg_PIL(hex_im_file)
#                    
#                    hex_vid_max = hex_vid.max(axis=0)
##                    hex_mask_0 = hex_vid[0] >= np.mean(hex_vid[0])
##                    hex_mask_0 = hex_vid_max >= threshold_otsu(hex_vid_max)
#                    hex_mask_0 = hex_vid_max >= np.mean(hex_vid_max)
##                    hex_mask_0 = clean_hex_mask(hex_mask_0)
#                    all_hex_ve_masks.append(hex_mask_0)
#                    
#                    
#                    ### give the shape info
##                    all_2d_shapes.append(np.hstack(hex_mask_0.shape))
                    all_3d_shapes.append(np.hstack(embryo_im_shape))
                    
                    """
                    2. read in the VE trackfile
                    """
                    ve_im_file = unwrap_file_set[0]
                    
                    ve_vid = fio.read_multiimg_PIL(ve_im_file); all_2d_frames.append(ve_vid[0])
                    ve_vid_mask = ve_vid[0] < 1
                    
                    all_2d_shapes.append(np.hstack(ve_vid_mask.shape))
                    
                    ve_track_file = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels)+os.path.split(ve_im_file)[-1].replace('.tif', '.mat'))
#                
                    meantracks_ve = spio.loadmat(ve_track_file)['meantracks']; #all_meantracks_ve_2D.append(meantracks_ve)
#                    meantracks_ve[...,0] = np.clip(meantracks_ve[...,0], 0, ve_vid.shape[1]-1)
#                    meantracks_ve[...,1] = np.clip(meantracks_ve[...,1], 0, ve_vid.shape[2]-1)
                    
#                    meantracks_ve = meantracks_ve.astype(np.float32)
#                    all_hex_ve_track_select.append(hex_mask_0[meantracks_ve[:,0,0],
#                                                              meantracks_ve[:,0,1]])
                    
                    meantracks_ve_filt = ve_vid_mask[meantracks_ve[:,0,0], 
                                                     meantracks_ve[:,0,1]]
                    
                    assert(len(meantracks_ve) == len(meantracks_ve_filt))
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
                        
                        meantracks_ve_3D = unwrap_params_ve[meantracks_ve[...,0],
                                                            meantracks_ve[...,1]]
                        
                        # we should normalize? 
                        meantracks_ve_3D_filt = meantracks_ve_3D.copy()#[meantracks_ve_filt]
                        meantracks_ve_3D_filt = meantracks_ve_3D_filt.astype(np.float32)
                        # normalise the readings.
                        for kk in range(meantracks_ve_3D_filt.shape[-1]):
                            meantracks_ve_3D_filt[...,kk] = meantracks_ve_3D_filt[...,kk] / float(embryo_im_shape[kk]) 
                        all_meantracks_ve_3D.append(meantracks_ve_3D_filt)
                    
                    # normalize the 2d. 
                    meantracks_ve = meantracks_ve.astype(np.float32)
                    meantracks_ve[...,0] = meantracks_ve[...,0] / float(ve_vid_mask.shape[0])
                    meantracks_ve[...,1] = meantracks_ve[...,1] / float(ve_vid_mask.shape[1])
#                    all_meantracks_ve_2D_select.append(meantracks_ve_filt)
                    all_meantracks_ve_2D.append(meantracks_ve)
                    
                    if rev_tracks == 'none':
                        all_meantracks_ve_3D.append(meantracks_ve)
#                    all_meantracks_ve_select.append(meantracks_ve_filt)
                    
    if ret_imgs == True:
        return all_embryos, all_2d_frames, (all_meantracks_ve_2D, all_meantracks_ve_3D, all_meantracks_ve_select), (np.vstack(all_2d_shapes), np.vstack(all_3d_shapes))
    else:
        return all_embryos, (all_meantracks_ve_2D, all_meantracks_ve_3D, all_meantracks_ve_select), (np.vstack(all_2d_shapes), np.vstack(all_3d_shapes))
 

def parse_temporal_scaling(tformfile):
    
    import scipy.io as spio 
    from transforms3d.affines import decompose44
    
    obj = spio.loadmat(tformfile)
    scales = np.hstack([1./decompose44(matrix)[2][0] for matrix in obj['tforms']])
    
    return scales
    
    
def load_vid_and_tracks_folders_mtmg_lifeact_w_unwrap(embryo_folders, proj_type, unwrap_folder_key='geodesic-rotmatrix', rev_tracks='demons', ret_imgs=False, ret_unwrap_params=False, n_spixels=1000):
    
    print(n_spixels)
    # embryo metadata
    all_embryos = []
    all_embryos_track_folders = []
    
#    # hex associated masks. 
#    all_hex_ve_masks = [] 
#    all_hex_ve_track_select = [] 
    
    # tracks.
    all_meantracks_ve_2D = []
    all_meantracks_ve_3D = []
    all_meantracks_ve_select = [] 
    
    # shapes
    all_2d_shapes = []
    all_3d_shapes = []
    all_vol_scales = []
    
    # snapshots
    all_2d_frames = []
    
    # all_unwrap_params = []
    all_unwrap_params = []

    # load up all the video and associated tracks for training. 
    for embryo_folder in embryo_folders[:]:
        
        print(embryo_folder)
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
            if n_spixels == 1000:
                savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
            else:
                savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels))
                
            all_embryos_track_folders
            saverevparamfolder = os.path.join(embryo_folder, 'unwrapped', 'geodesic_VE_Epi_matched-resize-rotmatrix_revmap')

            
            unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))
            unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
            """
            pair up the unwrapped files such that the same unwrap files belong, one for ve, epi and hex. 
            """
            paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
            
            
            for ii in tqdm(range(len(paired_unwrap_file_condition))[:]):
                
                file_condition = paired_unwrap_file_condition[ii]
                unwrap_file_set = paired_unwrap_files[ii] # in the order of 'VE', 'Epi', 'Hex'
                
                if file_condition.split('-')[-1] == proj_type:
                    
                    all_embryos.append(file_condition)
#                    all_embryos_track_folders.append(paired_unwrap_files[ii])

#                    ### give the shape info
##                    all_2d_shapes.append(np.hstack(hex_mask_0.shape))
                    all_3d_shapes.append(np.hstack(embryo_im_shape))
                    all_vol_scales.append(temp_tform_scales)
                    
                    """
                    2. read in the VE trackfile
                    """
                    ve_im_file = unwrap_file_set[0]
                    epi_im_file = unwrap_file_set[1]
                    
                    ve_vid = fio.read_multiimg_PIL(ve_im_file); all_2d_frames.append(ve_vid[0])
                    ve_vid_mask = ve_vid[0] < 1
                    
                    all_2d_shapes.append(np.hstack(ve_vid_mask.shape))
                    
                    if n_spixels == 1000:
                        ve_track_file = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels)+os.path.split(ve_im_file)[-1].replace('.tif', '.mat'))
                        epi_track_file = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels)+os.path.split(epi_im_file)[-1].replace('.tif', '.mat'))
                        meantracks_ve = spio.loadmat(ve_track_file)['meantracks']; #all_meantracks_ve_2D.append(meantracks_ve)
#                    ve_track_file = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels)+os.path.split(ve_im_file)[-1].replace('.tif', '.mat').replace('_unwrap_params_', '-'))
                    else:
                        ve_track_file = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels) + file_condition+'.mat')
                        epi_track_file = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels) + file_condition+'.mat')
#                    meantracks_ve = spio.loadmat(ve_track_file)['meantracks']; #all_meantracks_ve_2D.append(meantracks_ve)
                        meantracks_ve = spio.loadmat(ve_track_file)['meantracks_ve'];
#                    meantracks_ve[...,0] = np.clip(meantracks_ve[...,0], 0, ve_vid.shape[1]-1)
#                    meantracks_ve[...,1] = np.clip(meantracks_ve[...,1], 0, ve_vid.shape[2]-1)
                    
#                    meantracks_ve = meantracks_ve.astype(np.float32)
#                    all_hex_ve_track_select.append(hex_mask_0[meantracks_ve[:,0,0],
#                                                              meantracks_ve[:,0,1]])
                    all_embryos_track_folders.append(np.hstack([ve_track_file, epi_track_file]))
                    
                    
                    meantracks_ve_filt = ve_vid_mask[meantracks_ve[:,0,0], 
                                                     meantracks_ve[:,0,1]]
                    
                    assert(len(meantracks_ve) == len(meantracks_ve_filt))
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
                    if ret_unwrap_params:
                        unwrap_param_file = ve_im_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
                        unwrap_params_ve = spio.loadmat(unwrap_param_file)
                        unwrap_params_ve = unwrap_params_ve['ref_map_'+proj_type+'_xyz']
                        all_unwrap_params.append(unwrap_params_ve)
                    
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
                        
                        meantracks_ve_3D = unwrap_params_ve[meantracks_ve[...,0],
                                                            meantracks_ve[...,1]]
                        
                        # we should normalize? 
                        meantracks_ve_3D_filt = meantracks_ve_3D.copy()#[meantracks_ve_filt]
                        meantracks_ve_3D_filt = meantracks_ve_3D_filt.astype(np.float32)
                        # normalise the readings.
                        for kk in range(meantracks_ve_3D_filt.shape[-1]):
                            meantracks_ve_3D_filt[...,kk] = meantracks_ve_3D_filt[...,kk] / float(embryo_im_shape[kk]) 
                        all_meantracks_ve_3D.append(meantracks_ve_3D_filt)
                    
                    # normalize the 2d. 
                    meantracks_ve = meantracks_ve.astype(np.float32)
                    meantracks_ve[...,0] = meantracks_ve[...,0] / float(ve_vid_mask.shape[0])
                    meantracks_ve[...,1] = meantracks_ve[...,1] / float(ve_vid_mask.shape[1])
#                    meantracks_ve[...,0] = meantracks_ve[...,0] / float(np.sqrt(np.product(ve_vid_mask.shape)))
#                    meantracks_ve[...,1] = meantracks_ve[...,1] / float(np.sqrt(np.product(ve_vid_mask.shape)))
#                    all_meantracks_ve_2D_select.append(meantracks_ve_filt)
                    all_meantracks_ve_2D.append(meantracks_ve)
                    
                    if rev_tracks == 'none':
                        all_meantracks_ve_3D.append(meantracks_ve)
#                    all_meantracks_ve_select.append(meantracks_ve_filt)
                    
                
    if ret_imgs == True and ret_unwrap_params == True:
        return all_embryos, all_embryos_track_folders, all_2d_frames, all_unwrap_params, (all_meantracks_ve_2D, all_meantracks_ve_3D, all_meantracks_ve_select), (np.vstack(all_2d_shapes), np.vstack(all_3d_shapes), all_vol_scales)
    if ret_imgs == True and ret_unwrap_params == False:
        return all_embryos, all_embryos_track_folders, all_2d_frames, (all_meantracks_ve_2D, all_meantracks_ve_3D, all_meantracks_ve_select), (np.vstack(all_2d_shapes), np.vstack(all_3d_shapes), all_vol_scales)
    if ret_imgs == False and ret_unwrap_params == True:
        return all_embryos, all_embryos_track_folders, all_unwrap_params, (all_meantracks_ve_2D, all_meantracks_ve_3D, all_meantracks_ve_select), (np.vstack(all_2d_shapes), np.vstack(all_3d_shapes), all_vol_scales) 
    if ret_imgs == False and ret_unwrap_params == False:
        return all_embryos, all_embryos_track_folders, (all_meantracks_ve_2D, all_meantracks_ve_3D, all_meantracks_ve_select), (np.vstack(all_2d_shapes), np.vstack(all_3d_shapes), all_vol_scales)
    

def load_vid_and_tracks_folders_htmg(embryo_folders, proj_type, unwrap_folder_key='geodesic-rotmatrix', rev_tracks='demons'):
    
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
    

    # load up all the video and associated tracks for training. 
    for embryo_folder in embryo_folders[:]:
        
        # find the geodesic-rotmatrix. 
        unwrapped_folder = locate_unwrapped_im_folder(embryo_folder, key=unwrap_folder_key)
        
        # get one of these to get the size of the volume. 
        embryo_im_folder = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step6_' in name or 'step6tf_' in name or 'step6tf-' in name or 'step6-' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
        embryo_im_folder_files = np.hstack(glob.glob(os.path.join(embryo_im_folder[0], '*.tif')))
        embryo_im_shape = np.hstack(fio.read_multiimg_PIL(embryo_im_folder_files[0]).shape)
        embryo_im_shape = embryo_im_shape[[1,0,2]]#.transpose([1,0,2])

        
        if len(unwrapped_folder) == 1:
            print('processing: ', unwrapped_folder[0])
            
            unwrapped_folder = unwrapped_folder[0]
            unwrapped_params_folder = unwrapped_folder.replace(unwrap_folder_key, 
                                                               'unwrap_params_geodesic-rotmatrix')
            
            """
            Set the folders and load up the unwrapped files
            """
            # determine the savefolder. 
#            saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
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
                    
                    all_embryos.append(file_condition)
                    
                    """
                    1. read the hex video in to get the VE mask. 
                    """
                    # load the hex image file
                    hex_im_file = unwrap_file_set[-1]
                    hex_vid = fio.read_multiimg_PIL(hex_im_file)
                    
                    hex_vid_max = hex_vid.max(axis=0)
#                    hex_mask_0 = hex_vid[0] >= np.mean(hex_vid[0])
#                    hex_mask_0 = hex_vid_max >= threshold_otsu(hex_vid_max)
                    hex_mask_0 = hex_vid_max >= np.mean(hex_vid_max)
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
                    ve_vid_mask = ve_vid[0] < 1
                    
                    ve_track_file = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels)+os.path.split(ve_im_file)[-1].replace('.tif', '.mat'))
#                
                    meantracks_ve = spio.loadmat(ve_track_file)['meantracks']; #all_meantracks_ve_2D.append(meantracks_ve)
#                    meantracks_ve[...,0] = np.clip(meantracks_ve[...,0], 0, ve_vid.shape[1]-1)
#                    meantracks_ve[...,1] = np.clip(meantracks_ve[...,1], 0, ve_vid.shape[2]-1)
                    
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
                            meantracks_ve_3D = unwrap_params_ve[::-1][meantracks_ve[...,0],
                                                                      meantracks_ve[...,1]]
                        
                        # we should normalize? 
                        meantracks_ve_3D_filt = meantracks_ve_3D.copy()#[meantracks_ve_filt]
                        meantracks_ve_3D_filt = meantracks_ve_3D_filt.astype(np.float32)
                        # normalise the readings.
                        for kk in range(meantracks_ve_3D_filt.shape[-1]):
                            meantracks_ve_3D_filt[...,kk] = meantracks_ve_3D_filt[...,kk] / float(embryo_im_shape[kk]) 
                        all_meantracks_ve_3D.append(meantracks_ve_3D_filt)
                    
                    # normalize the 2d. 
                    meantracks_ve = meantracks_ve.astype(np.float32)
                    meantracks_ve[...,0] = meantracks_ve[...,0] / float(hex_mask_0.shape[0])
                    meantracks_ve[...,1] = meantracks_ve[...,1] / float(hex_mask_0.shape[1])
#                    meantracks_ve[...,0] = meantracks_ve[...,0] / float(np.sqrt(np.product(hex_mask_0.shape)))
#                    meantracks_ve[...,1] = meantracks_ve[...,1] / float(np.sqrt(np.product(hex_mask_0.shape)))
#                    all_meantracks_ve_2D_select.append(meantracks_ve_filt)
                    all_meantracks_ve_2D.append(meantracks_ve)
                    
                    if rev_tracks == 'none':
                        all_meantracks_ve_3D.append(meantracks_ve)
#                    all_meantracks_ve_select.append(meantracks_ve_filt)
                    
    return all_embryos, (all_hex_ve_masks, all_hex_ve_track_select), (all_meantracks_ve_2D, all_meantracks_ve_3D, all_meantracks_ve_select), (np.vstack(all_2d_shapes), np.vstack(all_3d_shapes))



def construct_track_classifier_velocity_feats( tracks, label_track_masks, im_shapes, growth_correction=None, track_2D=None, label_masks=None, tracks_select=None, neighbor_model=None, debug_viz=True, stack_feats=True):
    
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
            print('growth_correction', ii, len(growth_correction[ii]))
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
        
    if stack_feats == True:
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


def predict_VE_classifier_SVM( X, clf ):
    
    y_predict = clf.predict(X)
    
    return y_predict


def predict_VE_classifier_SVM_ensemble(X, clf_list, thresh):
    
    y_predicts = np.vstack([clf.predict(X) for clf in clf_list])
    av_y_predicts = np.mean(y_predicts, axis=0)

    return av_y_predicts > thresh    
    


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


def convex_hull_pts(points):
    
    from scipy.spatial import ConvexHull
    
    hull = ConvexHull(points)
    
    return hull 


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


""" get a more accurate representation of the AVE migration direction using robust ? """
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
        
        if ii ==0:
            print(tform_frame.params)
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
    
    
"""
Additional code to do postfiltering based on directional persistence.
"""
def compute_track_linearity(tracks):
    
    coefs_ = []
    mean_speeds = []
    directional_distances = []
    
    for tra in tracks:
        disp_tra = tra[1:] - tra[:-1]
        
        lin_speed = np.linalg.norm(np.nanmean(disp_tra, axis=0))
        total_speed = np.nanmean(np.linalg.norm(disp_tra,axis=1))
        
        coefs_.append(lin_speed / float(total_speed+1e-8))
        mean_speeds.append(lin_speed)
        directional_distances.append(np.nanmean(np.hstack([np.dot(d, np.nanmean(disp_tra, axis=0)) for d in disp_tra])))
        
    return np.hstack(coefs_), np.hstack(mean_speeds), np.hstack(directional_distances)


#def get_largest_connected_component( pts, dist_thresh):
#
#    from sklearn.metrics.pairwise import pairwise_distances
#    import networkx as nx
#    A = pairwise_distances(pts) <= dist_thresh
# 
#    G = nx.from_numpy_matrix(A)
#    largest_cc = max(nx.connected_components(G), key=len)
#    largest_cc = np.hstack(list(largest_cc))
#                    
#    return largest_cc
#
#
#def postprocess_polar_labels(pos0, labels0, dist_thresh):
#    
#    select = np.arange(len(labels0))[labels0>0]
#    pos_pt_ids = get_largest_connected_component( pos0[select], dist_thresh)
#    
#    labels_new = np.zeros_like(labels0)
#    labels_new[select[pos_pt_ids]] = 1
#    
#    return labels_new

def lookup_3D_to_2D_directions(unwrap_params, ref_pos_2D, ref_pos_3D, disp_3D, scale=None):
#    
    ref_pos_3D = unwrap_params[int(ref_pos_2D[0]), 
                               int(ref_pos_2D[1])]
    
    new_pos_3D = ref_pos_3D + 10*disp_3D
    
    min_dists = np.linalg.norm(unwrap_params - new_pos_3D[None,None,:], axis=-1) # require the surface normal ? 
    
#    plt.figure()
#    plt.imshow(min_dists, cmap='coolwarm')
#    plt.plot
#    plt.show()
    min_pos = np.argmin(min_dists)
    new_pos_2D = np.unravel_index(min_pos, unwrap_params.shape[:-1]) # returns as the index coords.
    
#    print(new_pos_2D)
#    
    plt.figure()
    plt.imshow(min_dists, cmap='coolwarm')
    plt.plot(ref_pos_2D[1], ref_pos_2D[0], 'ko')
    plt.plot(new_pos_2D[1], new_pos_2D[0], 'go')
    plt.show()
    
#    print(new_pos_2D)
    direction_2D = new_pos_2D - ref_pos_2D
    return direction_2D

    

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

    # add additional module to handle 3D curved nuances? or input this into the geometry package?
    import MOSES.Motion_Analysis.tracks3D_tools as tra3Dtools # additional module to heavy lift the 3D geometry stuff.
    
    from skimage.filters import threshold_otsu
    from tqdm import tqdm 
    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import KFold
    from sklearn.metrics import adjusted_rand_score, accuracy_score
    from sklearn.model_selection import RepeatedKFold
    from sklearn.externals import joblib
    
    import pandas as pd 
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial import convex_hull_plot_2d
    from skimage.filters import gaussian
    import skimage.transform as sktform
    import skimage.morphology as skmorph
    
    from scipy.ndimage.morphology import binary_fill_holes
    from mpl_toolkits.mplot3d import Axes3D
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # #     1. Get the training Data for the different projections (polar vs rectangular)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

    nbrs_model = NearestNeighbors(n_neighbors=9, algorithm='auto')
#    savetracksfolderkey = 'tracks_all_time-aligned-5000'
#    newsavetracksfolderkey = 'tracks_all_time-aligned-5000_VE_seg_directions-Migration_Stage'
    savetracksfolderkey = 'tracks_all_time'
    # savetracksfolderkey = 'tracks_all_time-aligned-5000'
    newsavetracksfolderkey = 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-curvilinear_Tcorrected_3Dangles'
    # newsavetracksfolderkey = 'tracks_all_time-aligned-5000_VE_seg_directions-Migration_Stage-correct-Final-meanvector'
#    nbrs_model = NearestNeighbors(n_neighbors=15, algorithm='auto')


    rootfolder = '/media/felix/Srinivas4/LifeAct'
#    rootfolder = '/media/felix/Srinivas4/MTMG-TTR'
#    rootfolder = '/media/felix/Srinivas4/MTMG-TTR'
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX'
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
    embryo_folders = np.hstack([os.path.join(rootfolder,folder) for folder in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, folder)) and 'new_hex' not in folder and 'L949_TP1' not in folder])
        
    
    clf_folder_polar = '/media/felix/Elements/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_polar_SVM_classifier/SVM'
#    clf_folder_polar = 'media/felix/Elements1/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_polar_SVM_demons-classifier-corrected-5000/SVM';
#    clf_folder_polar = '/media/felix/Elements1/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_polar_SVM_demons-classifier-corrected-5000/SVM'
#    clf_folder_rect = '/media/felix/Elements1/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_rect_SVM_classifier/SVM'
#    clf_folder_rect = '/media/felix/Elements1/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_rect_SVM_classifier-flips/SVM'
#    clf_folder_rect = '/media/felix/Elements1/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_rect_SVM_classifier-flips-Migration_Phase/SVM'
    clf_folder_rect = '/media/felix/Elements/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_rect_SVM_classifier-flips-scale-Migration_Phase/SVM'
#    clf_folder_rect = '/media/felix/Elements1/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_rect_SVM_classifier-flips-scale-optflowglobal-Migration_Phase/SVM';
#    clf_folder_rect = '/media/felix/Elements1/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_rect_SVM_classifier-flips-scale-global-Migration_Phase/SVM';
#    clf_folder_rect = '/media/felix/Elements1/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_rect_SVM_classifier-corrected-5000/SVM'
#    clf_folder_rect = '/media/felix/Elements1/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_rect_SVM_demons-classifier-corrected-5000-Rbf/SVM'
#    clf_folder_rect = '/media/felix/Elements1/Shankar LightSheet/Github_Scripts/MOSES_analysis/Hex-trained-VE_rect_SVM_demons-classifier-corrected-5000-Rbf-neighbors_15/SVM'

# =============================================================================
#   Load the staging information with all the various inversions 
# =============================================================================

#    manual_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/manual_staging_holly_matt.xlsx'
#    manual_staging_tab = pd.read_excel(manual_staging_file)
# =============================================================================
#   Load the staging information with all the various inversions 
# =============================================================================
    manual_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/manual_staging_holly_matt.xlsx'
    manual_staging_tab = pd.read_excel(manual_staging_file)

    staging_phase = 'Migration to Boundary'
    
    ve_migration_stage = manual_staging_tab[staging_phase].values
    ve_migration_stage_times = parse_start_end_times(ve_migration_stage)
    ve_migration_stage_times_valid = ~np.isnan(ve_migration_stage_times[:,0])
    
#    hex_staging_tab = manual_staging_tab.loc[np.logical_and(manual_staging_tab['Line'].values=='LA',ve_migration_stage_times_valid)]
#    hex_staging_tab = manual_staging_tab.loc[np.logical_and(manual_staging_tab['Line'].values=='TTR_MTMG',ve_migration_stage_times_valid)]
    hex_staging_tab = manual_staging_tab.loc[np.logical_and(manual_staging_tab['Line'].values=='HTMG',ve_migration_stage_times_valid)]
    
    hex_staging_embryos = np.hstack([str(s).replace('E','Emb') for s in hex_staging_tab['Embryo'].values])


## =============================================================================
##     Take only embryo folders which have defined migration phase. 
## =============================================================================
#    all_embryo_folders = embryo_folders
#    
#    # check the embryo folders have a migration phase.
#    folder_check = []
#    for ii in range(len(all_embryo_folders)):
#        basename = os.path.split(all_embryo_folders[ii])[-1]
#        if '_TP' in basename:
#            basename = basename.split('_TP')[0].split('L')[1]
#        elif 'LEFTY' in basename:
#            basename = basename.split('_LEFTY')[0].split('L')[1]
#        else:
#            basename = basename.split('_')[0].split('L')[1]
#            
#        print(basename)
#        if basename in hex_staging_embryos:
#            folder_check.append(all_embryo_folders[ii])
#    folder_check = np.hstack(folder_check) 
#    all_embryo_folders = folder_check
    """
    what happens if no migration phase? 
    """
    
# =============================================================================
#   Load first all the polar features. 
# =============================================================================
    # n_spixels = 5000 # set the number of pixels. 
    n_spixels = 1000 # 1000 gave better classification and segmentation intuitively 
#    proj_type = 'polar'

    # load all the tracks
    all_embryos_polar, all_embryos_polar_savefolders, all_ve_vid_frame0_polar, all_unwrap_params_ve_polar, all_meantracks_polar, all_shapes_polar = load_vid_and_tracks_folders_mtmg_lifeact_w_unwrap(embryo_folders, 
                                                                                                                                                                       proj_type='polar', 
                                                                                                                                                                       unwrap_folder_key='geodesic-rotmatrix', 
                                                                                                                                                                       rev_tracks='demons',
                                                                                                                                                                       ret_imgs=True,
                                                                                                                                                                       ret_unwrap_params=True,
                                                                                                                                                                       n_spixels=n_spixels)
    
    all_embryos_rect, all_embryos_rect_savefolders, all_ve_vid_frame0_rect, all_unwrap_params_ve_rect, all_meantracks_rect, all_shapes_rect = load_vid_and_tracks_folders_mtmg_lifeact_w_unwrap(embryo_folders, 
                                                                                                                                                                  proj_type='rect', 
                                                                                                                                                                  unwrap_folder_key='geodesic-rotmatrix', 
                                                                                                                                                                  rev_tracks='none',
                                                                                                                                                                  ret_imgs=True,
                                                                                                                                                                  ret_unwrap_params=True,
                                                                                                                                                                  n_spixels=n_spixels)

    # parse out the returned track information 
    all_2d_shapes_polar, all_3d_shapes_polar, all_tform_scales_polar = all_shapes_polar
    all_meantracks_ve_2D_polar, all_meantracks_ve_polar, all_meantracks_ve_select_polar = all_meantracks_polar
    
    all_2d_shapes_rect, all_3d_shapes_rect, all_tform_scales_rect = all_shapes_rect
    all_meantracks_ve_2D_rect, all_meantracks_ve_rect, all_meantracks_ve_select_rect = all_meantracks_rect
    

# =============================================================================
#     construct the features for classification from the rectangular planar only. 
# =============================================================================
    all_X_rect, all_Y_rect = construct_track_classifier_velocity_feats(all_meantracks_ve_rect, [np.ones(len(ve)) for ve in all_meantracks_ve_2D_rect], 
                                                                                                   im_shapes=all_2d_shapes_rect, 
                                                                                                   growth_correction = all_tform_scales_rect,
                                                                                                   label_masks=np.ones(len(all_meantracks_ve_2D_rect)), 
                                                                                                   track_2D = all_meantracks_ve_2D_rect,
                                                                                                   tracks_select=None, 
                                                                                                   neighbor_model=NearestNeighbors(n_neighbors=9, algorithm='auto'), 
                                                                                                   debug_viz=False,
                                                                                                   stack_feats=False)
    
    # load the trained classifier + the results 
#    training_results_csv_polar = pd.read_csv(os.path.join(clf_folder_polar, 'SVM-rbf_classifier_train-test_scores_2-fold-demons-polar.csv' ))
#    training_results_csv_polar = pd.read_csv(os.path.join(clf_folder_polar, 'SVM-rbf_classifier_train-test_scores_2-fold-demons-polar.csv' ))
    training_results_csv_polar = pd.read_csv(os.path.join(clf_folder_polar, 'SVM-rbf_classifier_train-test_scores_2-fold-demons-polar.csv' ))
    trained_classifier_files_polar = np.sort(glob.glob(os.path.join(clf_folder_polar, '*.joblib'))) # get the trained classifier files 
    
#    training_results_csv_rect = pd.read_csv(os.path.join(clf_folder_rect, 'SVM-rbf_classifier_train-test_scores_2-fold-none-rect.csv' ))
#    training_results_csv_rect = pd.read_csv(os.path.join(clf_folder_rect, 'SVM-rbf_classifier_train-test_scores_2-fold-none-rect.csv' ))
    training_results_csv_rect = pd.read_csv(os.path.join(clf_folder_rect, 'SVM-rbf_classifier_train-test_scores_2-fold-none-rect.csv'))
    trained_classifier_files_rect = np.sort(glob.glob(os.path.join(clf_folder_rect, '*.joblib'))) # get the trained classifier files 
    
    
    track_clf_ensemble_polar = [joblib.load(clf_file) for clf_file in trained_classifier_files_polar]
    track_clf_ensemble_rect = [joblib.load(clf_file) for clf_file in trained_classifier_files_rect]
    
    
#    save_predict_clf_track_ve = os.path.join('VE_classifier_paper_figures/VE_classifier', rootfolder.split('/')[-1])
#    fio.mkdir(save_predict_clf_track_ve)
    
    all_correct_angles = []
    
    # iterate over the tracks and apply the learnt classifier.
#    for track_i in range(len(all_meantracks_ve_2D_rect))[:1]:
#    for track_i in range(len(all_meantracks_ve_2D_rect))[5:6]:
#    for track_i in range(len(all_meantracks_ve_2D_rect))[:1]:
        
    for track_i in range(len(all_meantracks_ve_2D_rect))[8:9]:
#    for track_i in range(len(all_meantracks_ve_2D_rect))[:]:
        embryo_polar = all_embryos_polar[track_i]
        embryo_rect = all_embryos_rect[track_i]
        
        embryo_polar_shape_i = all_3d_shapes_polar[track_i] # longest axis on the first dimension of the array. 
        embryo_rect_shape_i = all_3d_shapes_rect[track_i]
        
        spatial_scale_factor2D = np.sqrt(np.product(embryo_rect_shape_i))
        
        print(all_embryos_polar[track_i])
        proj_condition = embryo_polar.split('-')
        
        # get whether we need to invert. 
        if 'Emb' not in proj_condition[1]:
            embryo_name = proj_condition[0]
        else:
            embryo_name = proj_condition[0]+'_E'+proj_condition[1].split('Emb')[1]
        
        
        # load info regarding the staging. 
        table_emb_names = np.hstack([str(s) for s in manual_staging_tab['Embryo'].values])
        select_stage_row = manual_staging_tab.loc[table_emb_names==embryo_name]
        migration_stage = parse_start_end_times(select_stage_row[staging_phase])[0]
        
        migration_exist = False    
        if ~np.isnan(migration_stage[0]):
            migration_exist = True
        
        embryo_inversion = select_stage_row['Inverted'].values[0] > 0 

        embryo_polar_im = all_ve_vid_frame0_polar[track_i]
        embryo_rect_im = all_ve_vid_frame0_rect[track_i]
        
        if embryo_inversion:
            # invert the original images. 
            embryo_polar_im = embryo_polar_im[::-1]
            embryo_rect_im = embryo_rect_im[:,::-1]
        
        #### double checked and it looks ok. 
        unwrap_xyz_polar = all_unwrap_params_ve_polar[track_i]
        unwrap_xyz_rect = all_unwrap_params_ve_rect[track_i][::-1] # switch it to be the same orientation as the image! 
        
        if embryo_inversion:
            
            # invert the 3D ...... 
            
            # These are the unwrap parameters in 3D! -> cannot be inverted just using flipping of the 2D!.                    
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
#            ax.scatter(unwrap_xyz_rect[::40 ,::40, 0], 
#                       unwrap_xyz_rect[::40 ,::40, 2],
#                       unwrap_xyz_rect[::40 ,::40, 1], c='r')
            ax.scatter(unwrap_xyz_polar[::40 ,::40, 0], 
                       unwrap_xyz_polar[::40 ,::40, 2],
                       unwrap_xyz_polar[::40 ,::40, 1], c='r')
            
            # we need to revert these properly !. 
            unwrap_xyz_polar = unwrap_xyz_polar[::-1]
            unwrap_xyz_polar[...,2] = embryo_polar_shape_i[2] - unwrap_xyz_polar[...,2]
            
            unwrap_xyz_rect = unwrap_xyz_rect[:,::-1]
            unwrap_xyz_rect[...,2] = embryo_rect_shape_i[2] - unwrap_xyz_rect[...,2]
        
#            ax.scatter(unwrap_xyz_rect[::40 ,::40, 0], 
#                       unwrap_xyz_rect[::40 ,::40, 2],
#                       unwrap_xyz_rect[::40 ,::40, 1], c='g')
            ax.scatter(unwrap_xyz_polar[::40 ,::40, 0], 
                       unwrap_xyz_polar[::40 ,::40, 2],
                       unwrap_xyz_polar[::40 ,::40, 1], c='g')
            
            plt.show()  
            
#            # mark the invalid pixels -> do this only later. ?
#            unwrap_xyz_rect[unwrap_xyz_rect[...,0] < 1] = np.nan
#            unwrap_xyz_polar[unwrap_xyz_polar[...,0] < 1] = np.nan
        
# =============================================================================
#         nearest neighbour model to match rect + polar unwrapping parameters. 
# =============================================================================
        # use the nearest neighbours model 
        nbrs_model_3D = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nbrs_model_3D.fit(unwrap_xyz_polar.reshape(-1,3))
        
        neighbours_rect_polar_id = nbrs_model_3D.kneighbors(unwrap_xyz_rect.reshape(-1,3))
        val_dist_img = neighbours_rect_polar_id[1].reshape(unwrap_xyz_rect.shape[:2])
        val_dist_img_binary = neighbours_rect_polar_id[0].reshape(unwrap_xyz_rect.shape[:2]) <= 5
        val_dist_ijs = np.unravel_index(val_dist_img,unwrap_xyz_polar.shape[:2] )
        
        nbrs_model_3D = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nbrs_model_3D.fit(unwrap_xyz_rect.reshape(-1,3))
        neighbours_polar_rect_id = nbrs_model_3D.kneighbors(unwrap_xyz_polar.reshape(-1,3))
        val_dist_img_rect = neighbours_polar_rect_id[1].reshape(unwrap_xyz_polar.shape[:2])
        val_dist_img_binary_rect = neighbours_polar_rect_id[0].reshape(unwrap_xyz_polar.shape[:2]) <= 5
        val_dist_ijs_rect = np.unravel_index(val_dist_img_rect, unwrap_xyz_rect.shape[:2] )
        
#        # just double check the unravel_index. 
#        val_dist = neighbours_rect_polar_id[0] < 5
#        mapped_polar_ids = np.zeros(unwrap_xyz_polar.shape[:2])
#        mapped_polar_ids[val_dist_ijs[0], 
#                         val_dist_ijs[1]] = 1
#                            
#        mapped_polar_ids_test = np.zeros(unwrap_xyz_polar.shape[:2])           
#        mapped_polar_ids_test[val_dist_ijs[0][:,0], 
#                              val_dist_ijs[1][:,1]] = 1

        """
        plot the polar coordinates. 
        """
        imshape = all_2d_shapes_polar[track_i]
        
        growth_correction = all_tform_scales_rect[track_i]
# =============================================================================
#         If migration state exists load information regarding the tracks in the specific stage. 
# =============================================================================
        
        if migration_exist:
            print('migration exist')
            savetrackfolder_ve_stage = os.path.join(all_embryos_polar_savefolders[track_i][0].split('/MOSES_analysis')[0], 'MOSES_analysis', 'tracks_all_time-aligned-1000_%s-manual' %(staging_phase))
            # savetrackfolder_ve_stage = os.path.join(all_embryos_polar_savefolders[track_i][0].split('/MOSES_analysis')[0], 'MOSES_analysis', 'tracks_all_time-aligned-5000_%s-manual' %(staging_phase))
            ve_track_file_stage_polar = os.path.join(savetrackfolder_ve_stage, 'deepflow-meantracks-%d_%s_' %(n_spixels,staging_phase)+embryo_polar+'.mat')
            meantracks_ve_stage_polar = spio.loadmat(ve_track_file_stage_polar)['meantracks_ve']
            
            ve_track_file_stage_rect = os.path.join(savetrackfolder_ve_stage, 'deepflow-meantracks-%d_%s_' %(n_spixels,staging_phase)+embryo_rect+'.mat')
            meantracks_ve_stage_rect = spio.loadmat(ve_track_file_stage_rect)['meantracks_ve']
            
            stage_start_end = spio.loadmat(ve_track_file_stage_rect)['start_end'].ravel()
            
#            meantracks_ve_disps = remove_global_component_tracks_robust(meantracks_ve, mask_track=None, return_tforms=False)
#            meantracks_ve = np.concatenate([meantracks_ve[:,0][:,None], np.cumsum(meantracks_ve_disps, axis=1)+meantracks_ve[:,0][:,None]], axis=1)
            track_2D = meantracks_ve_stage_polar.astype(np.float32)
            track_2D[...,0] = track_2D[...,0] / float(all_2d_shapes_polar[track_i][0])
            track_2D[...,1] = track_2D[...,1] / float(all_2d_shapes_polar[track_i][1])
#            track_2D[...,0] = track_2D[...,0] / float(np.sqrt(np.product(all_2d_shapes_polar[track_i])))
#            track_2D[...,1] = track_2D[...,1] / float(np.sqrt(np.product(all_2d_shapes_polar[track_i])))
#            spatial_scale_factor2D
            
            track_2D_int_rect = meantracks_ve_stage_rect.astype(np.float32)
            track_2D_int_rect[...,0] = track_2D_int_rect[...,0] / float(all_2d_shapes_rect[track_i][0])
            track_2D_int_rect[...,1] = track_2D_int_rect[...,1] / float(all_2d_shapes_rect[track_i][1])
#            track_2D_int_rect[...,0] = track_2D_int_rect[...,0] / float(np.sqrt(np.product(all_2d_shapes_rect[track_i])))
#            track_2D_int_rect[...,1] = track_2D_int_rect[...,1] / float(np.sqrt(np.product(all_2d_shapes_rect[track_i])))
            
            growth_correction = growth_correction[stage_start_end[0]:stage_start_end[1]+1]
            
        else:
            # if the migration phase exists we use this to derive the orientation. 
            track_2D = all_meantracks_ve_2D_polar[track_i].copy()
            track_2D_int_rect = all_meantracks_ve_2D_rect[track_i].copy()
        
        
        # adjust the 2D tracks. 
        if embryo_inversion:
            track_2D[...,0] = 1 - track_2D[...,0] # invert the y-axis of the polar.  
            track_2D_int_rect[...,1] = 1 - track_2D_int_rect[...,1] # invert x-axis 
            
            # must also invert the polar? 
        
        track_2D_disps = (track_2D[:,1:] - track_2D[:,:-1]).astype(np.float32)
        track_2D_mean_disps = track_2D_disps.mean(axis=1)
        
        
#        if migration_exist:
            # construct the track feats to put into the new classifier. 
        X_feats = track_2D_int_rect.copy()
#            X_feats = meantracks_ve_stage_rect.copy()
#            # compute velocity
#            meantracks_ve_stage_rect_disps = remove_global_component_tracks_robust(X_feats, mask_track=None, 
#                                                                                   return_tforms=False)
#            meantracks_ve_stage_rect = np.concatenate([X_feats[:,0][:,None], 
#                                                       np.cumsum(meantracks_ve_stage_rect_disps, axis=1)+X_feats[:,0][:,None]], axis=1)
#            meantracks_ve_stage_rect[...,0] = meantracks_ve_stage_rect[...,0] / float(all_2d_shapes_rect[track_i][0])
#            meantracks_ve_stage_rect[...,1] = meantracks_ve_stage_rect[...,1] / float(all_2d_shapes_rect[track_i][1])
        
#            if 'LifeAct' in rootfolder:
        X_feats = (X_feats[:,1:] - X_feats[:,:-1]).astype(np.float32) * growth_correction[None,1:,None]
#            X_feats = (meantracks_ve_stage_rect[:,1:] - meantracks_ve_stage_rect[:,:-1]).astype(np.float32) * growth_correction[None,1:,None]
        X_feat_mean = X_feats.mean(axis=1)
        
        if 'LifeAct' in rootfolder:
            # multiplied by 2 the speeds. 
            X_feat_mean = X_feat_mean * 2. # since half the sampling time.

        X_feat_neighbor_model = NearestNeighbors(n_neighbors=9, algorithm='auto')
        # 3. optionally construct the neighbourhood features
        X_feat_neighbor_model.fit(track_2D_int_rect[:, 0]) # use the initial topology 
        nbr_inds = X_feat_neighbor_model.kneighbors(track_2D_int_rect[:,0], return_distance=False)
            
        X_feat_mean_ = X_feat_mean[nbr_inds]
        # reshape into vectors. 
        track_feats_rect = X_feat_mean_.reshape(-1,nbr_inds.shape[1] * X_feats.shape[-1])
                
#        else:       
#    #        # select the classifier. 
#    #        track_feats_polar = all_X_polar[track_i]# [all_meantracks_ve_select[track_i]>0]
#            track_feats_rect = all_X_rect[track_i]
###        track_label = track_clf.predict(track_feats)
##        track_label_polar = predict_VE_classifier_SVM_ensemble(track_feats_polar, 
##                                                               track_clf_ensemble_polar, 
##                                                               thresh=0.8)


        # might want to verify this..... ? what is good consensus ? 
        track_label_rect = predict_VE_classifier_SVM_ensemble(track_feats_rect, 
                                                               track_clf_ensemble_rect, 
                                                               thresh=0.85)
        
# =============================================================================
# =============================================================================
# #         Set up the savefolder! for the debugging. 
# =============================================================================
# =============================================================================
        aligned_meantrack_file_in = all_embryos_polar_savefolders[track_i][0]
        aligned_meantrack_file_out = aligned_meantrack_file_in.replace(savetracksfolderkey, newsavetracksfolderkey).replace('.mat', '-ve_aligned.mat')
        saveoutfolder = os.path.split(aligned_meantrack_file_out)[0]
        fio.mkdir(saveoutfolder)
        
# =============================================================================
#         Plot the output of these predictions for the paper. 
# =============================================================================
        
        fig, ax = plt.subplots(figsize=(15,15))
        plot_tracks(track_2D, ax=ax, color='r')
        plt.show()
        
        track_2D_int_polar = track_2D.copy()
        track_2D_int_polar[...,0] = track_2D_int_polar[...,0]*all_2d_shapes_polar[track_i][0]
        track_2D_int_polar[...,1] = track_2D_int_polar[...,1]*all_2d_shapes_polar[track_i][1]
#        track_2D_int_polar[...,0] = track_2D_int_polar[...,0]*np.sqrt(np.product(all_2d_shapes_polar[track_i]))
#        track_2D_int_polar[...,1] = track_2D_int_polar[...,1]*np.sqrt(np.product(all_2d_shapes_polar[track_i]))
        track_2D_int_polar = track_2D_int_polar.astype(np.int)


        fig, ax = plt.subplots(figsize=(15,15))
        plot_tracks(track_2D_int_polar, ax=ax, color='r')
        plt.show()

        """
        convert to the proper sizes. 
        """
#        if embryo_inversion:
#            track_2D_int_rect[...,1] = 1 - track_2D_int_rect[...,1] # invert x. 
        track_2D_int_rect[...,0] = track_2D_int_rect[...,0]*all_2d_shapes_rect[track_i][0]
        track_2D_int_rect[...,1] = track_2D_int_rect[...,1]*all_2d_shapes_rect[track_i][1]
#        track_2D_int_rect[...,0] = track_2D_int_rect[...,0]*np.sqrt(np.product(all_2d_shapes_rect[track_i]))#[0]
#        track_2D_int_rect[...,1] = track_2D_int_rect[...,1]*np.sqrt(np.product(all_2d_shapes_rect[track_i]))
        track_2D_int_rect = track_2D_int_rect.astype(np.int)
        
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(embryo_rect_im, cmap='gray')
        plot_tracks(track_2D_int_rect, ax=ax, color='r')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(embryo_polar_im, cmap='gray')
        plot_tracks(track_2D_int_polar, ax=ax, color='r')
        plt.show()
        
        
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('SVM rect prediction')
        ax.imshow(embryo_rect_im, cmap='gray')
        ax.plot(track_2D_int_rect[track_label_rect>0,0,1], 
                track_2D_int_rect[track_label_rect>0,0,0], 'w.')
#        fig.savefig(os.path.join(saveoutfolder, embryo_polar+'_initial_SVM_rect_points.svg'), bbox_inches='tight')
        plt.show()
        

##        track_2D_polar_pos = all_unwrap_params_ve_polar[track_i][track_2D_int_polar[:,0,0], track_2D_int_polar[:,0,1]]
##        track_2D_rect_pos = all_unwrap_params_ve_rect[track_i][track_2D_int_rect[:,0,0], track_2D_int_rect[:,0,1]]
        
        """
        Map the rect SVM predictions into Polar to filter. 
        """
        track_label_rect_polar_ids_i = val_dist_ijs[0][track_2D_int_rect[:,0,0], 
                                                        track_2D_int_rect[:,0,1]]  
        track_label_rect_polar_ids_j = val_dist_ijs[1][track_2D_int_rect[:,0,0], 
                                                        track_2D_int_rect[:,0,1]]  
        
        track_label_rect_valid = val_dist_img_binary[track_2D_int_rect[:,0,0], 
                                                     track_2D_int_rect[:,0,1]]
        
        track_label_rect_polar_ids_i = track_label_rect_polar_ids_i[track_label_rect_valid]
        track_label_rect_polar_ids_j = track_label_rect_polar_ids_j[track_label_rect_valid]
        track_label_rect = track_label_rect[track_label_rect_valid]
#        
#        # postprocess the results. 
        spixel_size = np.abs(track_2D[1,0,1] - track_2D[2,0,1])
#        track_label_polar = postprocess_polar_labels(track_2D[:,0], track_label_polar, dist_thresh=1.5*spixel_size)
        track_label_rect = postprocess_polar_labels(np.vstack([track_label_rect_polar_ids_i,
                                                               track_label_rect_polar_ids_j]).T, 
                                                               track_label_rect, dist_thresh=1.5*spixel_size*all_2d_shapes_polar[track_i][0])
#        track_label_rect = postprocess_polar_labels(np.vstack([track_label_rect_polar_ids_i,
#                                                               track_label_rect_polar_ids_j]).T, 
#                                                               track_label_rect, 
#                                                               dist_thresh=1.5*spixel_size*np.sqrt(np.product(all_2d_shapes_polar[track_i])))
        
        
        rect_polar_pts = (np.vstack([track_label_rect_polar_ids_i,
                                     track_label_rect_polar_ids_j]).T)[track_label_rect>0]
        convexhull_rect = convex_hull_pts(rect_polar_pts)

            
        density_rect_pts = np.zeros(all_ve_vid_frame0_polar[track_i].shape)
        density_rect_pts[track_label_rect_polar_ids_i[track_label_rect>0],
                         track_label_rect_polar_ids_j[track_label_rect>0]] = 1
        density_rect_pts = gaussian(density_rect_pts, 1*spixel_size*all_2d_shapes_polar[track_i][0])
        
# =============================================================================
#         Derive the polar associated mask. 
# =============================================================================
        density_rect_pts_mask = density_rect_pts > np.mean(density_rect_pts) + 1.*np.std(density_rect_pts) # produce the mask. 
        density_rect_pts_mask = binary_fill_holes(density_rect_pts_mask)
        
        # this gets the selected polar spixels that are inferred as VE. 
        polar_ve_select = density_rect_pts_mask[track_2D_int_polar[:,0,0], 
                                                track_2D_int_polar[:,0,1]]
        
        
        
        # save the original SVM predicted Region. 
        density_rect_pts_mask_SVM = density_rect_pts_mask.copy()
        
        spixel_size_rect = np.abs(track_2D_int_rect[1,0,1] - track_2D_int_rect[2,0,1])
# =============================================================================
#         Reverse the density mask from polar now to rect to refine this!. 
# =============================================================================
        track_label_polar_rect_ids_i = val_dist_ijs_rect[0][density_rect_pts_mask>0]  
        track_label_polar_rect_ids_j = val_dist_ijs_rect[1][density_rect_pts_mask>0]
        
        density_polar_pts = np.zeros(all_ve_vid_frame0_rect[track_i].shape)
        density_polar_pts[track_label_polar_rect_ids_i, 
                          track_label_polar_rect_ids_j] = 1
        density_polar_pts = gaussian(density_polar_pts, .5*spixel_size_rect) # use 2 here to get as smooth as possible 
        density_polar_pts_mask = density_polar_pts > .1*np.mean(density_polar_pts)
        density_polar_pts_mask = skmorph.binary_closing(density_polar_pts_mask, skmorph.disk(2*spixel_size_rect))
        density_polar_pts_mask = binary_fill_holes(density_polar_pts_mask)
        
        density_polar_pts_mask_SVM = density_polar_pts_mask.copy()
        
# =============================================================================
#       # refine this selection using the notions of track persistence that has been developed. 
# =============================================================================
        polar_ave_pred_tracks = track_2D_int_polar[polar_ve_select] # no growth correction ( still need to decide on directionality ... )
        polar_ave_tracks_lin, polar_ave_tracks_mean_speed, polar_ave_tracks_persistence = compute_track_linearity(polar_ave_pred_tracks)
        
        
        # try with the mean speeds of the 3D equivalent -> how much difference does this make?
        # including growth correction?
        if migration_exist:
            polar_ave_tracks_3D_diff, (polar_ave_surf_dist, polar_ave_surf_lines), (polar_ave_tracks_3D_diff_mean, polar_ave_tracks_3D_diff_mean_curvilinear) = tra3Dtools.compute_curvilinear_3D_disps_tracks(polar_ave_pred_tracks, 
                                                                                                                                                                                       unwrap_xyz_polar, 
                                                                                                                                                                                       n_samples=20, nearestK=1, 
                                                                                                                                                                                       temporal_weights=growth_correction[1:])
        else:
            polar_ave_tracks_3D_diff, (polar_ave_surf_dist, polar_ave_surf_lines), (polar_ave_tracks_3D_diff_mean, polar_ave_tracks_3D_diff_mean_curvilinear) = tra3Dtools.compute_curvilinear_3D_disps_tracks(polar_ave_pred_tracks, 
                                                                                                                                                                                       unwrap_xyz_polar, 
                                                                                                                                                                                       n_samples=20, nearestK=1, 
                                                                                                                                                                                       temporal_weights=all_tform_scales_polar[track_i][1:])
        
#        polar_ave_tracks_mean_speed = np.linalg.norm(polar_ave_tracks_3D_diff_mean, axis=-1)
        # for this example seems to combine the best of both.?
        polar_ave_tracks_mean_speed = np.linalg.norm(polar_ave_tracks_3D_diff_mean_curvilinear, axis=-1)

        # speed compensation for different temporal samplings
        if 'LifeAct' in rootfolder:
            polar_ave_tracks_mean_speed = polar_ave_tracks_mean_speed *2.
        
        
        
#        threshold = 0.5
#        from skimage.filters import threshold_otsu
        polar_ave_pred_select_persistence = polar_ave_tracks_mean_speed>=np.mean(polar_ave_tracks_mean_speed) + 1.*np.std(polar_ave_tracks_mean_speed)
#        polar_ave_pred_select_persistence = polar_ave_tracks_mean_speed >= threshold_otsu(polar_ave_tracks_mean_speed)
#        polar_ave_pred_select_persistence = polar_ave_tracks_mean_speed >= 1.2
#        polar_ave_pred_select_persistence = polar_ave_tracks_persistence>=np.mean(polar_ave_tracks_persistence) + .5*np.std(polar_ave_tracks_persistence)
        
        polar_ave_pred_select_persistence = postprocess_polar_labels(polar_ave_pred_tracks[:,0], 
                                                                     polar_ave_pred_select_persistence, 
                                                                     dist_thresh=1.5*np.abs(track_2D_int_polar[1,0,1] - track_2D_int_polar[2,0,1]))

        
        # create a new density_rect_pts_mask and 
        density_rect_pts_mask_refine = np.zeros(all_ve_vid_frame0_polar[track_i].shape)
        density_rect_pts_mask_refine[polar_ave_pred_tracks[polar_ave_pred_select_persistence>0, 0, 0],
                                     polar_ave_pred_tracks[polar_ave_pred_select_persistence>0, 0, 1]] = 1
        density_rect_pts_mask_refine = gaussian(density_rect_pts_mask_refine, 1*np.abs(track_2D_int_polar[1,0,1] - track_2D_int_polar[2,0,1]))
        
        
        density_rect_pts_mask = density_rect_pts_mask_refine > np.mean(density_rect_pts_mask_refine) + 2.*np.std(density_rect_pts_mask_refine) # produce the mask. 
        density_rect_pts_mask = binary_fill_holes(density_rect_pts_mask)
        
        # this gets the selected polar spixels that are inferred as VE. 
        polar_ve_select = density_rect_pts_mask[track_2D_int_polar[:,0,0], 
                                                track_2D_int_polar[:,0,1]]
        
        density_rect_pts_mask_color = np.zeros((density_rect_pts_mask.shape[0], 
                                                density_rect_pts_mask.shape[1], 
                                                4))
        density_rect_pts_mask_color[...,1] = density_rect_pts_mask.copy()
        density_rect_pts_mask_color[...,-1] = 0
        density_rect_pts_mask_color[density_rect_pts_mask>0,-1] = 0.25
        
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('Initial SVM consensus prediction + track persistence')
        ax.imshow(embryo_polar_im, cmap='gray')
#        ax.imshow(density_rect_pts_mask, cmap='Greens', alpha=0.5)
        ax.imshow(density_rect_pts_mask_color)
#        plot_tracks(polar_ave_pred_tracks, ax=ax, color='r')
        ax.scatter(polar_ave_pred_tracks[:,0,1], 
                   polar_ave_pred_tracks[:,0,0], c=polar_ave_tracks_mean_speed, cmap='coolwarm')
#        plt.colorbar()
        fig.savefig(os.path.join(saveoutfolder, embryo_polar+'_initial_SVM_polar_points_and_persistence.svg'), bbox_inches='tight')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10,10))
#        ax.imshow(density_rect_pts_mask)
        ax.imshow(embryo_polar_im, cmap='gray')
#        plot_tracks(polar_ave_pred_tracks, ax=ax, color='r')
        ax.plot(polar_ave_pred_tracks[polar_ave_pred_select_persistence,0,1], 
                   polar_ave_pred_tracks[polar_ave_pred_select_persistence,0,0], 
                   'go', ms=10)
#        plt.colorbar()
        fig.savefig(os.path.join(saveoutfolder, embryo_polar+'_filtered_SVM_polar_points.svg'), bbox_inches='tight')
        plt.show()
        
## =============================================================================
##         Reverse derive and refine the rectangular associated mask. 
## =============================================================================
#        # look up what track 2D rect pos map to 
#        track_label_polar_rect_ids_i = val_dist_ijs_rect[0][track_2D_int_polar[:,0,0], 
#                                                            track_2D_int_polar[:,0,1]]  
#        track_label_polar_rect_ids_j = val_dist_ijs_rect[1][track_2D_int_polar[:,0,0], 
#                                                            track_2D_int_polar[:,0,1]]  
#        
#        track_label_polar_valid = val_dist_img_binary_rect[track_2D_int_polar[:,0,0], 
#                                                           track_2D_int_polar[:,0,1]]
#        
#        track_label_polar_rect_ids_i = track_label_polar_rect_ids_i[track_label_polar_valid]
#        track_label_polar_rect_ids_j = track_label_polar_rect_ids_j[track_label_polar_valid]
#        track_label_polar = polar_ve_select[track_label_polar_valid]
##        
#        polar_rect_pts = (np.vstack([track_label_polar_rect_ids_i,
#                                     track_label_polar_rect_ids_j]).T)[track_label_polar>0]
#        
        spixel_size_rect = np.abs(track_2D_int_rect[1,0,1] - track_2D_int_rect[2,0,1])
#        
#        density_polar_pts = np.zeros(all_ve_vid_frame0_rect[track_i].shape)
#        density_polar_pts[polar_rect_pts[:,0], 
#                          polar_rect_pts[:,1]] = 1
#        # augment with previous points 
#        density_polar_pts[track_2D_int_rect[track_label_rect_valid][track_label_rect>0,0,0],
#                          track_2D_int_rect[track_label_rect_valid][track_label_rect>0,0,1]] = 1
#        density_polar_pts = gaussian(density_polar_pts, .5*spixel_size_rect) # use 2 here to get as smooth as possible 
#        density_polar_pts_mask = density_polar_pts > .25*np.mean(density_polar_pts)
#        density_polar_pts_mask = skmorph.binary_closing(density_polar_pts_mask, skmorph.disk(2*spixel_size_rect))
#        density_polar_pts_mask = binary_fill_holes(density_polar_pts_mask)
        
# =============================================================================
#         Reverse the density mask from polar now to rect to refine this!. 
# =============================================================================
        track_label_polar_rect_ids_i = val_dist_ijs_rect[0][density_rect_pts_mask>0]  
        track_label_polar_rect_ids_j = val_dist_ijs_rect[1][density_rect_pts_mask>0]
        
        density_polar_pts = np.zeros(all_ve_vid_frame0_rect[track_i].shape)
        density_polar_pts[track_label_polar_rect_ids_i, 
                          track_label_polar_rect_ids_j] = 1
        density_polar_pts = gaussian(density_polar_pts, .5*spixel_size_rect) # use 2 here to get as smooth as possible 
        density_polar_pts_mask = density_polar_pts > .1*np.mean(density_polar_pts)
        density_polar_pts_mask = skmorph.binary_closing(density_polar_pts_mask, skmorph.disk(2*spixel_size_rect))
        density_polar_pts_mask = binary_fill_holes(density_polar_pts_mask)
        
        
        density_polar_pts_mask_color = np.zeros((density_polar_pts_mask.shape[0], 
                                                 density_polar_pts_mask.shape[1], 4))
        density_polar_pts_mask_color[ ... ,1 ] = density_polar_pts_mask
        density_polar_pts_mask_color[density_polar_pts_mask>0,-1] = 0.25
        
        
        track_2D_mean_disps_rect = np.mean(track_2D_int_rect[:,1:] - track_2D_int_rect[:,:-1], axis=1)
        """
        plot the predicted AVE tracks onto the rect 
        """
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('Filtered_polar_pred_remap_rect')
        ax.imshow(embryo_rect_im, cmap='gray')
#        ax.plot(track_2D_int_rect[:,0,1], 
#                track_2D_int_rect[:,0,0],'ro')
        ax.plot(track_2D_int_rect[track_label_rect_valid][track_label_rect>0,0,1], 
                track_2D_int_rect[track_label_rect_valid][track_label_rect>0,0,0], 'w.')
#        ax.plot(polar_rect_pts[:,1], 
#                polar_rect_pts[:,0], 'go')
        ax.imshow(density_polar_pts_mask_color)
#        ax.plot(np.hstack([rect_polar_pts[convexhull_rect.vertices,1], rect_polar_pts[convexhull_rect.vertices[0],1]]), 
#                np.hstack([rect_polar_pts[convexhull_rect.vertices,0], rect_polar_pts[convexhull_rect.vertices[0],0]]), 'w-', lw=2)
#        ax.imshow(density_rect_pts>np.mean(density_rect_pts) + .5*np.std(density_rect_pts), cmap='Reds', alpha=0.5)
        ax.quiver(track_2D_int_rect[:,0,1], 
                  track_2D_int_rect[:,0,0], 
                  track_2D_mean_disps_rect[:,1],
                  -track_2D_mean_disps_rect[:,0], color = 'r')
        fig.savefig(os.path.join(saveoutfolder, embryo_polar+'_filtered_SVM_remap-rect.svg'), bbox_inches='tight')
        plt.show()
        
        
        """
        plot the mapped VE tracks onto the polar 
        """
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('SVM rect prediction mapped to polar')
        ax.imshow(all_ve_vid_frame0_polar[track_i], cmap='gray')
        ax.plot(track_label_rect_polar_ids_j[track_label_rect>0], 
                track_label_rect_polar_ids_i[track_label_rect>0], 'w.')
        
        ax.plot(np.hstack([rect_polar_pts[convexhull_rect.vertices,1], rect_polar_pts[convexhull_rect.vertices[0],1]]), 
                np.hstack([rect_polar_pts[convexhull_rect.vertices,0], rect_polar_pts[convexhull_rect.vertices[0],0]]), 'w-', lw=2)
        
        dense_rect_mask_SVM = density_rect_pts>np.mean(density_rect_pts) + 1.*np.std(density_rect_pts)
        dense_rect_mask_SVM_color = np.zeros((dense_rect_mask_SVM.shape[0], 
                                              dense_rect_mask_SVM.shape[1],4))
        dense_rect_mask_SVM_color[...,-1] = 0.25
        dense_rect_mask_SVM_color[...,1] = dense_rect_mask_SVM
        
#        ax.imshow(density_rect_pts>np.mean(density_rect_pts) + 1.*np.std(density_rect_pts), cmap='Reds', alpha=0.5)
        ax.imshow(dense_rect_mask_SVM_color)
        ax.quiver(track_2D[:,0,1]*all_2d_shapes_polar[track_i][1], 
                  track_2D[:,0,0]*all_2d_shapes_polar[track_i][0], 
                  track_2D_mean_disps[:,1]*all_2d_shapes_polar[track_i][1],
                  -track_2D_mean_disps[:,0]*all_2d_shapes_polar[track_i][0], color = 'r')
        fig.savefig(os.path.join(saveoutfolder, embryo_polar+'_raw_SVM_map-polar.svg'), bbox_inches='tight')
        plt.show()
        
#        """
#        plot the inferred VE polar tracks. 
#        """
#        polar_mask = np.ones((all_ve_vid_frame0_polar[track_i].shape))
#        XX, YY = np.meshgrid(range(polar_mask.shape[0]), 
#                             range(polar_mask.shape[1]))
#        polar_mask = np.sqrt((XX-polar_mask.shape[0]/2.)**2 + 
#                             (YY-polar_mask.shape[1]/2.)**2) <= polar_mask.shape[0] /2. / 1.2
#        
#        
#        # this is already the component.
#        track_2D_int_polar_global_disps = remove_global_component_tracks_robust(track_2D_int_polar, 
#                                                                                mask_track=polar_mask[track_2D_int_polar[:,0,0],track_2D_int_polar[:,0,1]], 
#                                                                                return_tforms=False)
#        
#        # this gets the selected polar spixels that are inferred as VE. 
#        polar_ve_select = density_rect_pts_mask[track_2D_int_polar[:,0,0], 
#                                                track_2D_int_polar[:,0,1]]
#        
#        track_2D_int_polar = np.concatenate([ track_2D_int_polar[:,0][:,None], 
#                                              np.cumsum(track_2D_int_polar_global_disps, axis=1)  + track_2D_int_polar[:,0][:,None]], axis=1)
#        track_2D_int_polar = track_2D_int_polar.astype(np.int)
#        
#        fig, ax = plt.subplots(figsize=(15,15))
#        ax.imshow(all_ve_vid_frame0_polar[track_i], cmap='gray')
#        plot_tracks(track_2D_int_polar, ax=ax, color='r')
#        plt.show()
        
        
        """
        get the inferred movement direction using 3D data? + using the refined 5000 spixels track data? 
        """
        n_spixels_directionality = 5000 

        if migration_exist:
            savetrackfolder_ve_stage = os.path.join(all_embryos_polar_savefolders[track_i][0].split('/MOSES_analysis')[0], 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s-manual' %(n_spixels_directionality, staging_phase))
            ve_track_file_stage_polar = os.path.join(savetrackfolder_ve_stage, 'deepflow-meantracks-%d_%s_' %(n_spixels_directionality, staging_phase)+embryo_polar+'.mat')
            
            aligned_in_ve_track = spio.loadmat(ve_track_file_stage_polar)['meantracks_ve']
            aligned_in_epi_track = spio.loadmat(ve_track_file_stage_polar)['meantracks_epi']
        else:
            savetrackfolder_ve = os.path.join(all_embryos_polar_savefolders[track_i][0].split('/MOSES_analysis')[0], 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels_directionality))
            ve_track_file_stage_polar = os.path.join(savetrackfolder_ve, 'deepflow-meantracks-%d_' %(n_spixels_directionality)+embryo_polar+'.mat')

            try:
                aligned_in_ve_track = spio.loadmat(ve_track_file_stage_polar)['meantracks']
            except:
                aligned_in_ve_track = spio.loadmat(ve_track_file_stage_polar)['meantracks_ve']
#            aligned_in_epi_track = spio.loadmat(aligned_meantrack_file_in)['meantracks_epi']
        
        # invert the tracks if need be... 
        if embryo_inversion:
            # invert the tracks 
            aligned_in_ve_track[...,0] = embryo_polar_im.shape[0] - aligned_in_ve_track[...,0]
            aligned_in_epi_track[...,0] = embryo_polar_im.shape[0] - aligned_in_epi_track[...,0]
        

        # apply the 5000.
        polar_ve_select_SVM = density_rect_pts_mask[aligned_in_ve_track[:,0,0], 
                                                    aligned_in_ve_track[:,0,1]]

        mean_pos = np.mean(aligned_in_ve_track[polar_ve_select_SVM,0], axis=0)
        
        print('computing directionality with n_superpixels: ', len(aligned_in_ve_track))
        fig, ax = plt.subplots(figsize=(15,15))
        ax.imshow(density_rect_pts_mask)
        ax.plot(aligned_in_ve_track[polar_ve_select_SVM,0,1], 
                aligned_in_ve_track[polar_ve_select_SVM,0,0], 'g.')
        plt.show()
        
        
##        if migration_exist:
##            mean_ve_polar_directions = ( track_2D_int_polar[polar_ve_select_SVM,1:] - track_2D_int_polar[polar_ve_select_SVM,:-1] ).astype(np.float32) * growth_correction[None,1:,None]
##        else:
##            mean_ve_polar_directions = ( track_2D_int_polar[polar_ve_select_SVM,1:] - track_2D_int_polar[polar_ve_select_SVM,:-1] ).astype(np.float32) * all_tform_scales_polar[track_i][None,1:,None]
#        if migration_exist:
#            mean_ve_polar_directions = ( aligned_in_ve_track[polar_ve_select_SVM,1:] - aligned_in_ve_track[polar_ve_select_SVM,:-1] ).astype(np.float32) * growth_correction[None,1:,None]
#        else:
#            mean_ve_polar_directions = ( aligned_in_ve_track[polar_ve_select_SVM,1:] - aligned_in_ve_track[polar_ve_select_SVM,:-1] ).astype(np.float32) * all_tform_scales_polar[track_i][None,1:,None]
#
##        mean_ve_polar_directions = np.mean(mean_ve_polar_directions[:,:len(mean_ve_polar_directions)//2], axis=1)
#        mean_ve_polar_directions = np.mean(mean_ve_polar_directions[:,:], axis=1)
#        mean_ve_polar_directions_reg = np.median(mean_ve_polar_directions, axis=0)
#        
#        angle_vertical = np.arctan2(mean_ve_polar_directions_reg[1],mean_ve_polar_directions_reg[0])
#        print(180-angle_vertical/np.pi*180)
        
        """
        The inferred direction should be found in 3D and then mapped to 2D?   
        """
        # compute the 3D displacements according to the unwrap parameters. 
        track_3D_polar_AVE = unwrap_xyz_polar[aligned_in_ve_track[polar_ve_select_SVM, : ,0], 
                                              aligned_in_ve_track[polar_ve_select_SVM, : ,1]]
        mean_pos_3D = np.mean(track_3D_polar_AVE[:,0], axis=0)
        disps_3D_polar_AVE = track_3D_polar_AVE[:,1:] - track_3D_polar_AVE[:,:-1]; # note this is strictly the linear version.
        
        # put back the growth correction !
        if migration_exist:
            disps_3D_polar_AVE = disps_3D_polar_AVE.astype(np.float32) * growth_correction[None,1:,None]
        else:
            disps_3D_polar_AVE = disps_3D_polar_AVE.astype(np.float32) * all_tform_scales_polar[track_i][None,1:,None]
        
        mean_ve_polar_directions_3D = np.mean(disps_3D_polar_AVE[:,:], axis=1)
#        mean_ve_polar_directions_3D = np.mean(disps_3D_polar_AVE[:,:]/(np.linalg.norm(disps_3D_polar_AVE, axis=-1) + 1e-8 )[...,None], axis=1)
        # lets normalise?
#        mean_ve_polar_directions_3D = mean_ve_polar_directions_3D/(np.linalg.norm(mean_ve_polar_directions_3D, axis=-1) + 1e-8)[:,None]

        mean_ve_polar_directions_reg_3D = np.median(mean_ve_polar_directions_3D, axis=0) # use the mean?
#        mean_ve_polar_directions_reg_3D = np.mean(mean_ve_polar_directions_3D, axis=0) # use the mean?
        
        
        print('mean 3D direction: ', mean_ve_polar_directions_reg_3D)
        print( np.arctan2(mean_ve_polar_directions_reg_3D[2], mean_ve_polar_directions_reg_3D[1]))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for tra in track_3D_polar_AVE:
            ax.plot(tra[..., 0], 
                    tra[..., 2],
                    tra[..., 1], c='r')
        
        # we need to revert these properly !.     
        ax.scatter(unwrap_xyz_polar[::40 ,::40, 0], 
                   unwrap_xyz_polar[::40 ,::40, 2],
                   unwrap_xyz_polar[::40 ,::40, 1], c='g')
        
        ax.scatter(mean_pos_3D[0], 
                   mean_pos_3D[2], 
                   mean_pos_3D[1], c = 'k')
        ax.plot([mean_pos_3D[0], mean_pos_3D[0]+10*mean_ve_polar_directions_reg_3D[0]], 
                [mean_pos_3D[2], mean_pos_3D[2]+10*mean_ve_polar_directions_reg_3D[2]], 
                [mean_pos_3D[1], mean_pos_3D[1]+10*mean_ve_polar_directions_reg_3D[1]], c='b')
        
        plt.show()  
        
        # projection into 2D ! using the mean_pos ....
        ref_pos_2D = np.hstack(all_ve_vid_frame0_polar[track_i].shape) //2
        
        # Is this overcomplicating? -> no i think it is ok? -> if we parametrised as (z, theta), then the 'theta' = directionality!.
        mean_ve_polar_directions_reg_3D_proj2D = lookup_3D_to_2D_directions(unwrap_xyz_polar, 
                                                                            mean_pos, 
                                                                            mean_pos_3D, 
                                                                            mean_ve_polar_directions_reg_3D,
                                                                            scale=None) # use a scale? 
        
#        # apply a minimum magnitude
#        if np.linalg.norm(mean_ve_polar_directions_reg_3D) < 0.85:
#            mean_ve_polar_directions_reg_3D_proj2D[0] = 0 # zero the y component. 
        
        print('3D projected angle: ',mean_ve_polar_directions_reg_3D_proj2D)
        
        angle_vertical = np.arctan2(mean_ve_polar_directions_reg_3D_proj2D[1], 
                                    mean_ve_polar_directions_reg_3D_proj2D[0])
        print(180-angle_vertical/np.pi*180)
        
        mean_ve_polar_directions_reg = mean_ve_polar_directions_reg_3D_proj2D
        mean_ve_polar_directions_reg = mean_ve_polar_directions_reg / float(np.linalg.norm(mean_ve_polar_directions_reg) + 1e-8)
        
        plt.figure()
        plt.imshow(embryo_polar_im, cmap='gray')
        polar_center = mean_pos
        plt.plot(polar_center[1], polar_center[0], 'wo', ms=15)
        plt.plot([polar_center[1], polar_center[1]+100*mean_ve_polar_directions_reg[1]],
                [polar_center[0], polar_center[0]+100*mean_ve_polar_directions_reg[0]],'w--', lw=5)
        plt.plot([polar_center[1], polar_center[1]+10*mean_ve_polar_directions_reg_3D_proj2D[1]],
                [polar_center[0], polar_center[0]+10*mean_ve_polar_directions_reg_3D_proj2D[0]],'r--', lw=5)
        plt.show()
        
        """
        Plot the average 2D directions.... 
        """
        track_2D_mean_disps = track_2D_int_polar[:,1:] - track_2D_int_polar[:,:-1]
        track_2D_mean_disps = track_2D_disps.mean(axis=1) 
        
  
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(embryo_polar_im, cmap='gray')
#        ax.plot(track_label_rect_polar_ids_j[track_label_rect>0], 
#                track_label_rect_polar_ids_i[track_label_rect>0], 'w.')
        ax.plot(track_2D[polar_ve_select,0,1]*all_2d_shapes_polar[track_i][1], 
                track_2D[polar_ve_select,0,0]*all_2d_shapes_polar[track_i][0], 'go')
        
#        polar_center = np.hstack(all_ve_vid_frame0_polar[track_i].shape)/2.
        polar_center = mean_pos
        ax.plot(polar_center[1], polar_center[0], 'wo', ms=15)
        ax.plot([polar_center[1], polar_center[1]+100*mean_ve_polar_directions_reg[1]],
                [polar_center[0], polar_center[0]+100*mean_ve_polar_directions_reg[0]],'w--', lw=5)
#        ax.plot(np.hstack([rect_polar_pts[convexhull_rect.vertices,1], rect_polar_pts[convexhull_rect.vertices[0],1]]), 
#                np.hstack([rect_polar_pts[convexhull_rect.vertices,0], rect_polar_pts[convexhull_rect.vertices[0],0]]), 'w-', lw=2)
        ax.imshow(density_rect_pts>np.mean(density_rect_pts) + 1.*np.std(density_rect_pts), cmap='Reds', alpha=0.5)
        ax.quiver(track_2D[:,0,1]*all_2d_shapes_polar[track_i][1], 
                  track_2D[:,0,0]*all_2d_shapes_polar[track_i][0], 
                  track_2D_mean_disps[:,1]*all_2d_shapes_polar[track_i][1],
                  -track_2D_mean_disps[:,0]*all_2d_shapes_polar[track_i][0], color = 'r', scale=20)
        fig.savefig(os.path.join(saveoutfolder, embryo_polar+'_SVM_polar_w_direction3D.svg'), bbox_inches='tight')
        plt.show()
        
        
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(embryo_polar_im, cmap='gray')
#        ax.plot(track_label_rect_polar_ids_j[track_label_rect>0], 
#                track_label_rect_polar_ids_i[track_label_rect>0], 'w.')
        ax.plot(track_2D[polar_ve_select,0,1]*all_2d_shapes_polar[track_i][1], 
                track_2D[polar_ve_select,0,0]*all_2d_shapes_polar[track_i][0], 'go')
        
#        polar_center = np.hstack(all_ve_vid_frame0_polar[track_i].shape)/2.
        polar_center = mean_pos
        ax.plot(polar_center[1], polar_center[0], 'wo', ms=15)
        ax.plot([polar_center[1], polar_center[1]+100*mean_ve_polar_directions_reg[1]],
                [polar_center[0], polar_center[0]+100*mean_ve_polar_directions_reg[0]],'w--', lw=5)
#        ax.plot(np.hstack([rect_polar_pts[convexhull_rect.vertices,1], rect_polar_pts[convexhull_rect.vertices[0],1]]), 
#                np.hstack([rect_polar_pts[convexhull_rect.vertices,0], rect_polar_pts[convexhull_rect.vertices[0],0]]), 'w-', lw=2)
        ax.imshow(density_rect_pts_mask>0, cmap='Reds', alpha=0.5)
        ax.quiver(track_2D[:,0,1]*all_2d_shapes_polar[track_i][1], 
                  track_2D[:,0,0]*all_2d_shapes_polar[track_i][0], 
                  track_2D_mean_disps[:,1]*all_2d_shapes_polar[track_i][1],
                  -track_2D_mean_disps[:,0]*all_2d_shapes_polar[track_i][0], color = 'r', scale=20)
        fig.savefig(os.path.join(saveoutfolder, embryo_polar+'_SVM_polar_persistent_filter_w_direction3D.svg'), bbox_inches='tight')
        plt.show()
        
        
        """
        now we need to compute and save the correct statistics to transform by .... 
        """
        # this is the aligned meantracksfile
        aligned_meantrack_file_in = all_embryos_polar_savefolders[track_i][0]
        aligned_meantrack_file_out = aligned_meantrack_file_in.replace(savetracksfolderkey, newsavetracksfolderkey).replace('.mat', '-ve_aligned.mat')
        saveoutfolder = os.path.split(aligned_meantrack_file_out)[0]
        fio.mkdir(saveoutfolder)
        
#        print(saveoutfolder)
#        aligned_meantrack_file_out = aligned_meantrack_file_out.replace('-5000','-1000')
# =============================================================================
#         reparse the aligned track we use for plotting purposes. 
# =============================================================================
        if migration_exist:
            savetrackfolder_ve_stage = os.path.join(all_embryos_polar_savefolders[track_i][0].split('/MOSES_analysis')[0], 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s-manual' %(n_spixels, staging_phase))
            ve_track_file_stage_polar = os.path.join(savetrackfolder_ve_stage, 'deepflow-meantracks-%d_%s_' %(n_spixels,staging_phase)+embryo_polar+'.mat')
            
            aligned_in_ve_track = spio.loadmat(ve_track_file_stage_polar)['meantracks_ve']
            aligned_in_epi_track = spio.loadmat(ve_track_file_stage_polar)['meantracks_epi']
        else:
#            aligned_in_ve_track = spio.loadmat(aligned_meantrack_file_in)['meantracks_ve']
            try:
                aligned_in_ve_track = spio.loadmat(aligned_meantrack_file_in)['meantracks']
            except:
                aligned_in_ve_track = spio.loadmat(aligned_meantrack_file_in)['meantracks_ve']
#            aligned_in_epi_track = spio.loadmat(aligned_meantrack_file_in)['meantracks_epi']
        
        # invert the tracks if need be... 
        if embryo_inversion:
            aligned_in_ve_track[...,0] = embryo_polar_im.shape[0] - aligned_in_ve_track[...,0]
            aligned_in_epi_track[...,0] = embryo_polar_im.shape[0] - aligned_in_epi_track[...,0]
        
#        
#        fig, ax = plt.subplots(figsize=(15,15))
#        ax.imshow(embryo_polar_im, cmap='gray', alpha=0.25)
#        plot_tracks(aligned_in_ve_track[:,:30], ax=ax, color='r')
#        plt.show()
#        
        rot_center = np.hstack(embryo_polar_im.shape)/2. 
        correct_angle = 180-angle_vertical/np.pi*180
        
        # get the rotated version of the tracks. 
        rotated_tracks_ve = rotate_tracks(aligned_in_ve_track, angle=-correct_angle, center=rot_center)
        rotated_tracks_epi = rotate_tracks(aligned_in_epi_track, angle=-correct_angle, center=rot_center)
        
        rotated_motion_center = rotate_pts(mean_pos[::-1][None,:], angle=-correct_angle, center=rot_center).ravel()[::-1]
        rotated_mean_vector = rotate_pts(mean_ve_polar_directions_reg[::-1][None,:], angle=-correct_angle, center=[0,0]).ravel()[::-1]
        
        #. 0. save the non rotated density mask. 
        rotate_density_image = sktform.rotate(density_rect_pts_mask, angle=correct_angle, preserve_range=True)
        rotate_initial_image = sktform.rotate(embryo_polar_im, angle=correct_angle, preserve_range=True)
        
        # 1. rotation angle + the rotation (motion center) which is required to set up the radial segments. 
#        from scipy.ndimage.measurements import center_of_mass
#        COM_VE = np.hstack(list(center_of_mass(rotate_density_image>0, labels=rotate_density_image>0)))
        
        all_correct_angles.append(correct_angle)
        
        """
        save out the inferred parameters which is used for analysis (VE mask, VE motion angle, center + ve_classified points)
        """
        print(aligned_meantrack_file_out)
        print(len(track_label_rect_valid) , len(track_2D_int_rect))
        assert(len(track_label_rect_valid) == len(track_2D_int_rect))
        spio.savemat(aligned_meantrack_file_out, {#'meantracks_ve':rotated_tracks_ve.astype(np.float), 
                                                  #'meantracks_epi':rotated_tracks_epi.astype(np.float), 
                                                  've_angle': angle_vertical, 
                                                  've_center': mean_pos,
                                                  've_vector': mean_ve_polar_directions_reg,
                                                  'correct_angle': correct_angle,
                                                  'density_rect_pts': density_rect_pts,
                                                  'density_polar_pts': density_rect_pts,
                                                  'density_rect_pts_mask_SVM': density_rect_pts_mask_SVM,
                                                  'density_polar_pts_mask_SVM': density_polar_pts_mask_SVM,
                                                  'density_rect_pts_mask': density_rect_pts_mask,
                                                  'density_polar_pts_mask': density_polar_pts_mask,
                                                  'embryo_inversion': embryo_inversion,
                                                  'migration_exist':migration_exist,
                                                  'migration_stage_times': migration_stage, 
                                                  'embryo_polar': embryo_polar,
                                                  'embryo_rect': embryo_rect,
#                                                  've_rect_select': track_label_rect,
#                                                  've_rect_select_valid': track_label_rect_valid,
#                                                  'rect_polar_yx': np.vstack([track_label_rect_polar_ids_i, 
#                                                                              track_label_rect_polar_ids_j]).T,
                                                  'track_file': aligned_meantrack_file_in
                                                  })
        
        # 2. compute the associated rotated tracks  based on the image statistics. 
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(rotate_initial_image, cmap='gray')
#        ax.imshow(rotate_density_image, cmap='Reds', alpha=0.5) # this needs to be blended? 
        rotate_density_image_cmap = np.zeros((rotate_density_image.shape[0],
                                              rotate_density_image.shape[1], 
                                              4))
        rotate_density_image_cmap[...,1] = rotate_density_image
        rotate_density_image_cmap[np.logical_not(rotate_density_image),-1] = 0
        rotate_density_image_cmap[rotate_density_image>0,-1] = .25
        ax.imshow(rotate_density_image_cmap) # this needs to be blended? 
        
#        plot_tracks(rotated_tracks_ve[:,:50,::-1], ax=ax, color='r')
        rot_track_vect = np.mean(rotated_tracks_ve[:,1:] -rotated_tracks_ve[:,:-1], axis=1)
        ax.plot(rotated_motion_center[1], rotated_motion_center[0], 'wo', ms=15)
#        ax.plot(COM_VE[1], COM_VE[0], 'go', ms=5,zorder=1e6)
        ax.plot(rot_center[1], rot_center[0], 'ko')
        ax.plot([rotated_motion_center[1], rotated_motion_center[1]+100*rotated_mean_vector[1]],
                [rotated_motion_center[0], rotated_motion_center[0]+100*rotated_mean_vector[0]],'w--', lw=5)
        
        ax.quiver(rotated_tracks_ve[:,0,1], 
                  rotated_tracks_ve[:,0,0], 
                  rot_track_vect[:,1],
                  -rot_track_vect[:,0], color = 'r')
        if embryo_inversion:
            fig.savefig(aligned_meantrack_file_out.replace('.mat', '-inverted.svg'), dpi=300, bbox_inches='tight')
        else:
            fig.savefig(aligned_meantrack_file_out.replace('.mat', '.svg'), dpi=300, bbox_inches='tight')
        plt.show()
        
        
        
        
        
        
        
        
        
        
#        fig, ax = plt.subplots(figsize=(10,10))
#        ax.imshow(rotate_initial_image, cmap='gray')
#        ax.imshow(rotate_density_image, cmap='Reds', alpha=0.5)
##        plot_tracks(rotated_tracks_ve[polar_ve_select,:50,:], ax=ax, color='r')
#        plot_tracks(rotated_tracks_ve[:,50:,:], ax=ax, color='r')
#        
#        rot_track_vect = np.mean(rotated_tracks_ve[:,1:] -rotated_tracks_ve[:,:-1], axis=1)
#        ax.plot(rotated_motion_center[1], rotated_motion_center[0], 'wo', ms=15)
##        ax.plot(COM_VE[1], COM_VE[0], 'go', ms=5,zorder=1e6)
#        ax.plot([rotated_motion_center[1], rotated_motion_center[1]+100*rotated_mean_vector[1]],
#                [rotated_motion_center[0], rotated_motion_center[0]+100*rotated_mean_vector[0]],'w--', lw=5)
#        
##        ax.quiver(rotated_tracks_ve[:,0,1], 
##                  rotated_tracks_ve[:,0,0], 
##                  rot_track_vect[:,1],
##                  -rot_track_vect[:,0], color = 'r')
##        fig.savefig(aligned_meantrack_file_out.replace('.mat', '-inverted.svg'), dpi=300, bbox_inches='tight')
#        plt.show()
#        
#        
#    """
#    Also save out the rotation correction angle? 
#    """
#    all_correct_angles = np.hstack(all_correct_angles)
#    all_embryos_rect = np.hstack(all_embryos_rect)
#    all_embryos_polar = np.hstack(all_embryos_polar)
#    
#    import pandas as pd 
#    
#    correct_angles_table = pd.DataFrame(np.hstack([all_embryos_rect[:,None], 
#                                                   all_embryos_polar[:,None],
#                                                   all_correct_angles[:,None]]), index=None, columns=['rect_conditions', 'polar_conditions', 'correct_angles_vertical'])
#    
#    save_angle_file = '/media/felix/Srinivas4/correct_angle-global-1000_'+os.path.split(rootfolder)[-1] + '.csv'
#    correct_angles_table.to_csv(save_angle_file, index=None)
#    
    


    
    