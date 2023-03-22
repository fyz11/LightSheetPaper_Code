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
    print(uniq_vid_conds)
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


def lookup_3D_to_2D_pos(unwrap_params, ref_pos_3D):

    min_dists = np.linalg.norm(unwrap_params - ref_pos_3D[None,None,:], axis=-1) # require the surface normal ? 
    
    min_pos = np.argmin(min_dists)
    new_pos_2D = np.unravel_index(min_pos, unwrap_params.shape[:-1]) # returns as the index coords.

    return new_pos_2D    


def construct_radial_theta_vectors_rect(unwrap_params, smooth_theta_window=10, smooth_dist=None):
    
    from scipy.ndimage import convolve
    radial_vects = np.gradient(unwrap_params, axis=0) # these are the distances.
    theta_vects = np.gradient(unwrap_params, axis=1) 
    
    # smoothen using periodic boundary conditions
    smooth_window = int(smooth_theta_window/(360./unwrap_params.shape[1])) # this allows smoothing specified with angles.
    smoother = np.ones(smooth_window)/float(smooth_window)
    smooth_x = smoother[None,:,None].copy()
    smooth_y = smoother[:,None,None].copy()
    
    radial_vects = convolve(radial_vects, weights=smooth_y, mode='mirror')
    theta_vects = convolve(theta_vects, weights=smooth_x, mode='mirror')
    
    if smooth_dist is not None:
        smoother = np.ones(smooth_dist)/float(smooth_dist)
        smooth_x = smoother[None,:,None].copy()
        smooth_y = smoother[:,None,None].copy()
        
        radial_vects = convolve(radial_vects, weights=smooth_y, mode='reflect')
        theta_vects = convolve(theta_vects, weights=smooth_x, mode='reflect')
    
    # final unit vector normalisation.
    radial_vects = radial_vects / (np.linalg.norm(radial_vects, axis=-1)[...,None] + 1e-8)
    theta_vects = theta_vects / (np.linalg.norm(theta_vects, axis=-1)[...,None] + 1e-8)
    
    return radial_vects, theta_vects


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    return []

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
    
    import MOSES.Motion_Analysis.tracks3D_tools as tra3Dtools
    
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
    from skimage.measure import find_contours
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # #     1. Get the training Data for the different projections (polar vs rectangular)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
    
#    all_save_plots_folder = '/media/felix/Srinivas4/Data_Info/Migration_Angles'
#    all_save_plots_folder = '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear_curved_mean'
    all_save_plots_folder = '/media/felix/Srinivas4/Data_Info/Migration_Angles-curvilinear'
    
    fio.mkdir(all_save_plots_folder)
    

    nbrs_model = NearestNeighbors(n_neighbors=9, algorithm='auto')
#    savetracksfolderkey = 'tracks_all_time-aligned-5000'
#    newsavetracksfolderkey = 'tracks_all_time-aligned-5000_VE_seg_directions-Migration_Stage'
#    savetracksfolderkey = 'tracks_all_time'
    savetracksfolderkey = 'tracks_all_time-aligned-5000'
#    newsavetracksfolderkey = 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-new'
    # newsavetracksfolderkey = 'tracks_all_time-aligned-5000_VE_seg_directions-Migration_Stage-correct-Final-meanvector'
#    nbrs_model = NearestNeighbors(n_neighbors=15, algorithm='auto')
    newsavetracksfolderkey = 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-curvilinear_Tcorrected'


#    rootfolder = '/media/felix/Srinivas4/LifeAct' # done 
#    rootfolder = '/media/felix/Srinivas4/MTMG-TTR'
#    rootfolder = '/media/felix/Srinivas4/MTMG-TTR'
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX' # done
    rootfolder = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
    embryo_folders = np.hstack([os.path.join(rootfolder,folder) for folder in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, folder)) and 'new_hex' not in folder and 'L949_TP1' not in folder])
    

# =============================================================================
#   Load the staging information with all the various inversion information. ! 
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

    
# =============================================================================
#   Load first all the polar features. 
# =============================================================================
    n_spixels = 5000
    
    all_correct_angles_polar = []; all_embryos_polar = [];
    all_correct_angles_rect = []; all_embryos_rect = []
    
    all_ve_direction_displacements_polar = [];
    all_ve_direction_displacements_rect = [];
    
    # iterate over the tracks and apply the learnt classifier.
#    for track_i in range(len(all_meantracks_ve_2D_rect))[:1]:
#    for track_i in range(len(all_meantracks_ve_2D_rect))[5:6]:
#    for track_i in range(len(all_meantracks_ve_2D_rect))[:1]:
    for ii in range(len(embryo_folders))[:]:
        
        embryo_folder = embryo_folders[ii]
        
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
#            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-new')
            savevesegfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-1000_VE_seg_directions-Migration_Stage-correct-Final-meanvector-curvilinear_Tcorrected'
)
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s' %(n_spixels, analyse_stage))
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_stage-aligned-correct_uniform_scale-%d_%s' %(n_spixels, staging_phase))
            savefulltrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels))
#            savefulltrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned')
            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d_%s-manual' %(n_spixels, staging_phase))
            
            saveboundaryfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow-epi-boundary')
            
#            if os.path.exists(savetrackfolder) and len(os.listdir(savetrackfolder)) > 0:
            # only tabulate this statistic if it exists. 
#                all_analysed_embryos.append(embryo_folder)
        
            unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))

            if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
            
            else:
#                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_hex_unwrapped(ff)) for ff in unwrap_im_files])
#                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_hex_embryo(unwrap_im_files, unwrap_meta_files)
                
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
            
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
#                    if unwrapped_condition.split('-')[-1] == 'polar':
                proj_type = unwrapped_condition.split('-')[-1]
                    
                if 'MTMG-TTR' in embryo_folder or 'LifeAct' in embryo_folder:
                    ve_unwrap_file, epi_unwrap_file = paired_unwrap_files[ii]
                else:
#                    ve_unwrap_file, epi_unwrap_file, hex_unwrap_file = paired_unwrap_files[ii]
                    ve_unwrap_file, epi_unwrap_file = paired_unwrap_files[ii]
                    
            # =============================================================================
            #       Load the relevant inversion and staging information                         
            # =============================================================================
            
                proj_condition = unwrapped_condition.split('-')
                
                # get whether we need to invert. 
                if 'Emb' not in proj_condition[1]:
                    embryo_name = proj_condition[0]
                else:
                    embryo_name = proj_condition[0]+'_E'+proj_condition[1].split('Emb')[1]
                    
                    
                # load info regarding the staging. 
                table_emb_names = np.hstack([str(s) for s in manual_staging_tab['Embryo'].values])
                select_stage_row = manual_staging_tab.loc[table_emb_names==embryo_name]
                migration_stage = parse_start_end_times(select_stage_row[staging_phase])[0]# need to subtract a 1!
                
                migration_exist = False    
                if ~np.isnan(migration_stage[0]):
                    migration_exist = True
                    
                embryo_inversion = select_stage_row['Inverted'].values[0] > 0 
                            
                # =============================================================================
                #        Load the relevant videos.          
                # =============================================================================
                vid_ve = fio.read_multiimg_PIL(ve_unwrap_file)
#                vid_epi = fio.read_multiimg_PIL(epi_unwrap_file)
#                vid_epi_resize = sktform.resize(vid_epi, vid_ve.shape, preserve_range=True)
                
#                if 'MTMG-HEX' in embryo_folder:
#                    vid_hex = fio.read_multiimg_PIL(hex_unwrap_file)
                        
            # =============================================================================
            #       Load the relevant unwrap 3D.                          
            # =============================================================================
                unwrap_param_file_ve = ve_unwrap_file.replace(unwrapped_folder, unwrapped_params_folder).replace(proj_type+'.tif', 'unwrap_params-geodesic.mat')
                unwrap_params_ve = spio.loadmat(unwrap_param_file_ve); unwrap_params_ve = unwrap_params_ve['ref_map_'+proj_type+'_xyz']
                    
            # =============================================================================
            #       Load the relevant VE tracks depending on if information is available in the migration phase or not.                         
            # =============================================================================
            
                if migration_exist:
                    try:
                        ve_track_file_stage = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_%s_' %(n_spixels,staging_phase)+unwrapped_condition+'.mat')
                        ve_track = spio.loadmat(ve_track_file_stage)['meantracks_ve']
                    except:
                        basename = [proj_condition[0], '', proj_condition[1], proj_condition[2]]
                        ve_track_file_stage = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_%s_' %(n_spixels,staging_phase)+'-'.join(basename)+'.mat')
                        ve_track = spio.loadmat(ve_track_file_stage)['meantracks_ve']
                        
                        
                    migration_times = spio.loadmat(ve_track_file_stage)['start_end'] #- 1
                    migration_times = migration_times.astype(np.int).ravel()
                    growth_correction = temp_tform_scales[migration_times[0]:migration_times[1]+1]
                else:
                    
                    try:
                        ve_track_file = os.path.join(savefulltrackfolder, 'deepflow-meantracks-%d_' %(n_spixels)+unwrapped_condition+'.mat')
    
                        try:
                            ve_track = spio.loadmat(ve_track_file)['meantracks']
                        except:
                            ve_track = spio.loadmat(ve_track_file)['meantracks_ve']
                    except:
                        basename = [proj_condition[0], '', proj_condition[1], proj_condition[2]]
                        ve_track_file = os.path.join(savefulltrackfolder, 'deepflow-meantracks-%d_' %(n_spixels)+'-'.join(basename)+'.mat')
    
                        try:
                            ve_track = spio.loadmat(ve_track_file)['meantracks']
                        except:
                            ve_track = spio.loadmat(ve_track_file)['meantracks_ve']
                        
                        
                    growth_correction = temp_tform_scales.copy()
              
            # =============================================================================
            #       Load the relevant VE track segmentations.                     
            # =============================================================================
                ve_seg_params_file = 'deepflow-meantracks-1000_' + os.path.split(unwrap_param_file_ve)[-1].replace('_unwrap_params-geodesic.mat',  '_polar-ve_aligned.mat')
                ve_seg_params_file = os.path.join(savevesegfolder, ve_seg_params_file)
                ve_seg_params_obj = spio.loadmat(ve_seg_params_file)                             

#                ve_seg_select_rect_valid = ve_seg_params_obj['density_rect_pts']
#                ve_seg_select_rect_valid = ve_seg_select_rect_valid >= np.mean(ve_seg_select_rect_valid) + 1 * np.std(ve_seg_select_rect_valid)
                ve_seg_select_rect_valid = ve_seg_params_obj['density_rect_pts_mask'] # get the polar associated mask. 
                ve_seg_migration_times = ve_seg_params_obj['migration_stage_times'][0] - 1 # this should subtract 1 ! 
            
                if proj_type == 'polar':
                    all_embryos_polar.append(unwrapped_condition)
                    
                    if embryo_inversion:
                        unwrap_params_ve = unwrap_params_ve[::-1]
                        unwrap_params_ve[...,2] = embryo_im_shape[2] - unwrap_params_ve[...,2]
                        
                        # invert the VE tracks
                        ve_track[...,0] = (vid_ve[0].shape[0]-1) - ve_track[...,0]       
                        
                        vid_ve = vid_ve[:,::-1] # flip the y-axis
                        
                    # get the associated segmentation 
                    ve_seg_polar = ve_seg_params_obj['density_rect_pts_mask']
                    
                    
                    """
                    label the tracks.
                    """
                    ve_select = ve_seg_polar[ve_track[:,0,0], 
                                             ve_track[:,0,1]] > 0
                    
                    
                    """
                    The inferred direction should be found in 3D and then mapped to 2D?   
                    """
                    track_2D_mean_disps = np.mean( ( ve_track[:,1:] - ve_track[:,:-1]) * growth_correction[None,1:,None], axis=1)
                    
                    # compute the 3D displacements according to the unwrap parameters. 
                    track_3D_polar_AVE = unwrap_params_ve[ve_track[ve_select>0, : ,0], 
                                                          ve_track[ve_select>0, : ,1]]
                    
                    mean_pos = np.mean(ve_track[ve_select>0,0], axis=0)
                    mean_pos_3D = np.mean(track_3D_polar_AVE[:,0], axis=0)
                    
                    
                    disps_3D_polar_AVE = track_3D_polar_AVE[:,1:] - track_3D_polar_AVE[:,:-1]; 
                    
                    disps_3D_polar_AVE = disps_3D_polar_AVE.astype(np.float32) * growth_correction[None,1:,None]
                   
                    mean_ve_polar_directions_3D = np.mean(disps_3D_polar_AVE[:,:], axis=1)
            #        mean_ve_polar_directions_3D = np.mean(disps_3D_polar_AVE[:,:]/(np.linalg.norm(disps_3D_polar_AVE, axis=-1) + 1e-8 )[...,None], axis=1)
                    # lets normalise?
#                    mean_ve_polar_directions_3D = mean_ve_polar_directions_3D/(np.linalg.norm(mean_ve_polar_directions_3D, axis=-1) + 1e-8)[:,None]
                    mean_ve_polar_directions_reg_3D = np.median(mean_ve_polar_directions_3D, axis=0) # use the mean?
                    
                    mean_ve_polar_directions_reg_3D_proj2D = lookup_3D_to_2D_directions(unwrap_params_ve, 
                                                                            mean_pos, 
                                                                            mean_pos_3D, 
                                                                            mean_ve_polar_directions_reg_3D,
                                                                            scale=None) # use a scale? 
        
                    angle_vertical = np.arctan2(mean_ve_polar_directions_reg_3D_proj2D[1], 
                                                mean_ve_polar_directions_reg_3D_proj2D[0])

                    correct_angle = 180-angle_vertical/np.pi*180; all_correct_angles_polar.append(correct_angle)
                    print(180-angle_vertical/np.pi*180)
                    
                    mean_ve_polar_directions_reg = mean_ve_polar_directions_reg_3D_proj2D
                    mean_ve_polar_directions_reg = mean_ve_polar_directions_reg / float(np.linalg.norm(mean_ve_polar_directions_reg) + 1e-8)
        
        
                    mean_ve_directions_polar = np.hstack([mean_ve_polar_directions_reg_3D/float(np.linalg.norm(mean_ve_polar_directions_reg_3D) + 1e-8),
                                                          mean_ve_polar_directions_reg])
                    
                    all_ve_direction_displacements_polar.append(mean_ve_directions_polar)
        
                    # draw the angle!. 
                    fig, ax = plt.subplots(figsize=(10,10))
                    plt.title(str(ve_seg_params_obj['correct_angle']))
                    ax.imshow(vid_ve[0], cmap='gray')
                    ax.plot(ve_track[ve_select,0,1], 
                            ve_track[ve_select,0,0], 'go')
                    polar_center = mean_pos
                    ax.plot(polar_center[1], polar_center[0], 'wo', ms=15)
                    ax.plot([polar_center[1], polar_center[1]+100*mean_ve_polar_directions_reg[1]],
                            [polar_center[0], polar_center[0]+100*mean_ve_polar_directions_reg[0]],'w--', lw=5)
            #        ax.plot(np.hstack([rect_polar_pts[convexhull_rect.vertices,1], rect_polar_pts[convexhull_rect.vertices[0],1]]), 
            #                np.hstack([rect_polar_pts[convexhull_rect.vertices,0], rect_polar_pts[convexhull_rect.vertices[0],0]]), 'w-', lw=2)
#                    ax.imshow(density_rect_pts_mask>0, cmap='Reds', alpha=0.5)
                    ax.quiver(ve_track[:,0,1], 
                              ve_track[:,0,0], 
                              track_2D_mean_disps[:,1],
                              -track_2D_mean_disps[:,0], color = 'r', scale=20)
                    fig.savefig(os.path.join(all_save_plots_folder, unwrapped_condition+'_polar_direction3D.svg'), bbox_inches='tight')
                    plt.show()
                    
                    
                if proj_type == 'rect':
                    all_embryos_rect.append(unwrapped_condition)
                    
                    unwrap_params_ve = unwrap_params_ve[::-1] # to be in the same orientation as the image. 
                    
                    if embryo_inversion:
                        unwrap_params_ve = unwrap_params_ve[:,::-1]
                        unwrap_params_ve[...,2] = embryo_im_shape[2] - unwrap_params_ve[...,2]
                        
                        vid_ve = vid_ve[:,:,::-1] # reverse the x axis. 
                            
                        # invert the VE tracks
                        ve_track[...,1] = (vid_ve[0].shape[1]-1) - ve_track[...,1]     
                        
                    invalid_yx = unwrap_params_ve[...,0] == 0
                        
                    # get the associated segmentation 
                    ve_seg_rect = ve_seg_params_obj['density_polar_pts_mask']
                    ve_seg_rect_cont = find_contours(ve_seg_rect,0)[0]
                
                    ve_select = ve_seg_rect[ve_track[:,0,0], 
                                            ve_track[:,0,1]] > 0
                    
                    ve_select_pts = ve_track[ve_select,0]
                    ve_select_pts_valid = invalid_yx[ve_select_pts[:,0],
                                                     ve_select_pts[:,1]]
                    ve_select_pts = ve_select_pts[ve_select_pts_valid==0]
                    
                    track_2D_mean_disps = np.mean( ( ve_track[:,1:] - ve_track[:,:-1]) * growth_correction[None,1:,None], axis=1)
                    
                    
                    """
                    Construct the radial vectors and angular normalised vectors
                    """
                    radial_direction_vectors, theta_direction_vectors = construct_radial_theta_vectors_rect(unwrap_params_ve, 
                                                                                                            smooth_theta_window=10, 
                                                                                                            smooth_dist=vid_ve[0].shape[0]//4)
    
                    radial_direction_vectors[...,0] = -radial_direction_vectors[...,0] # point upwards from the distal bottom to the top.
                    
#                    # verify this with radial plotting.
#                    fig = plt.figure()
#                    ax = fig.add_subplot(111, projection='3d')
#                    ax.set_aspect('equal')
#
#                    ax.quiver(unwrap_params_ve[::20,::20,0], 
#                               unwrap_params_ve[::20,::20,1],
#                               unwrap_params_ve[::20,::20,2], 
#                               radial_direction_vectors[::20,::20,0], 
#                               radial_direction_vectors[::20,::20,1],
#                               radial_direction_vectors[::20,::20,2], 
#                               length=10., color='r', normalize=True)
#                    
#                    set_axes_equal(ax)
#                    plt.show()
                    
                    
                    # compute the 3D displacements according to the unwrap parameters. 
                    track_3D_polar_AVE = unwrap_params_ve[ve_track[ve_select>0, : ,0], 
                                                          ve_track[ve_select>0, : ,1]]
                    
#                    track_3D_polar_AVE_radial_direct = radial_direction_vectors[ve_track[ve_select>0, : ,0], 
#                                                                                ve_track[ve_select>0, : ,1]]
#                    
#                    mean_pos = np.mean(ve_track[ve_select>0,0], axis=0)
#                    mean_pos_3D = np.median(track_3D_polar_AVE[:,0], axis=0)
#                    
#                    disps_3D_polar_AVE = track_3D_polar_AVE[:,1:] - track_3D_polar_AVE[:,:-1]; 
#                    disps_3D_polar_AVE = disps_3D_polar_AVE.astype(np.float32) * growth_correction[None,1:,None]
###                    disps_3D_polar_AVE_mean = disps_3D_polar_AVE.mean(axis=1)
#                    
#                    # project the disps_3D_polar_AVE onto the tangential directions.
#                    proj_tangential_disps_3D_polar_AVE = np.sum( disps_3D_polar_AVE * track_3D_polar_AVE_radial_direct[:,:-1], axis=-1)
#                    proj_tangential_disps_3D_polar_AVE_mean = np.nanmean(proj_tangential_disps_3D_polar_AVE, axis=1)
                    if migration_exist:
                        polar_ave_tracks_3D_diff, (polar_ave_surf_dist, polar_ave_surf_lines), (polar_ave_tracks_3D_diff_mean, polar_ave_tracks_3D_diff_mean_curvilinear) = tra3Dtools.compute_curvilinear_3D_disps_tracks(ve_track[ve_select>0], 
                                                                                                                                                                                                   unwrap_params_ve, 
                                                                                                                                                                                                   n_samples=20, nearestK=1, 
                                                                                                                                                                                                   temporal_weights=growth_correction[1:])
                    else:
                        polar_ave_tracks_3D_diff, (polar_ave_surf_dist, polar_ave_surf_lines), (polar_ave_tracks_3D_diff_mean, polar_ave_tracks_3D_diff_mean_curvilinear) = tra3Dtools.compute_curvilinear_3D_disps_tracks(ve_track[ve_select>0], 
                                                                                                                                                                                                   unwrap_params_ve, 
                                                                                                                                                                                                   n_samples=20, nearestK=1, 
                                                                                                                                                                                                   temporal_weights=temp_tform_scales[1:])
#                    
#                    polar_ave_tracks_mean_speed = np.linalg.norm(polar_ave_tracks_3D_diff_mean, axis=-1)
#                    # for this example seems to combine the best of both.?
##                    disps_3D_polar_AVE_mean = np.linalg.norm(polar_ave_tracks_3D_diff_mean_curvilinear, axis=-1)
                    disps_3D_polar_AVE_mean = polar_ave_tracks_3D_diff_mean_curvilinear
                    disps_3D_polar_AVE_mean = disps_3D_polar_AVE_mean[ve_select_pts_valid==0]
#                    proj_tangential_disps_3D_polar_AVE_mean = proj_tangential_disps_3D_polar_AVE_mean[ve_select_pts_valid==0]
                    disps_2D_polar_AVE_mean = track_2D_mean_disps[ve_select][ve_select_pts_valid==0]
                    
                    #### bin the x axis.
                    n_x_bins = 20
                    x_axis_bins = np.linspace(0, vid_ve[0].shape[1], n_x_bins+1)
                    
                    x_bins_s_component = []
                    
                    outliers = np.mean(disps_3D_polar_AVE_mean[:,0]) - 2* np.std(disps_3D_polar_AVE_mean[:,0])
#                    outliers = np.mean(proj_tangential_disps_3D_polar_AVE_mean) + 2* np.std(proj_tangential_disps_3D_polar_AVE_mean)
                    
                    for bb in range(len(x_axis_bins)-1):
                        
                        select = np.logical_and(ve_select_pts[:,1] >= x_axis_bins[bb],
                                                ve_select_pts[:,1] < x_axis_bins[bb + 1])
                        
                        if np.sum(select) > 0:
                            
                            vals = disps_3D_polar_AVE_mean[select][:,0]
                            vals = vals[vals>outliers]
                            
                            if len(vals)>0:
                                x_bins_s_component.append(np.nanmedian(vals))
                            else:
                                x_bins_s_component.append(0)
#                            x_bins_s_component.append(np.nanmedian(disps_2D_polar_AVE_mean[select][:,0])) # y direction.
#                            x_bins_s_component.append(np.nanmedian(disps_2D_polar_AVE_mean[select][:,0]))
#                            x_bins_s_component.append(np.nanmedian(np.linalg.norm(disps_3D_polar_AVE_mean[select], axis=-1)))
                        else:
                            x_bins_s_component.append(0)
                            
#                    # use for the tangetial 3D projected speeds.
#                    for bb in range(len(x_axis_bins)-1):
#                        
#                        select = np.logical_and(ve_select_pts[:,1] >= x_axis_bins[bb],
#                                                ve_select_pts[:,1] < x_axis_bins[bb + 1])
#                        
#                        if np.sum(select) > 0:
#                            
#                            vals = proj_tangential_disps_3D_polar_AVE_mean[select]
#                            vals = vals[vals<outliers]
#                            x_bins_s_component.append(np.nanmedian(vals))
##                            x_bins_s_component.append(np.nanmedian(disps_2D_polar_AVE_mean[select][:,0])) # y direction.
##                            x_bins_s_component.append(np.nanmedian(disps_2D_polar_AVE_mean[select][:,0]))
##                            x_bins_s_component.append(np.nanmedian(np.linalg.norm(disps_3D_polar_AVE_mean[select], axis=-1)))
#                        else:
#                            x_bins_s_component.append(0)
                            
                    x_bins_s_component = np.hstack(x_bins_s_component)
                    x_bins_s_component[np.isnan(x_bins_s_component)] = 0
                    # remove massive outlier
#                    x_bins_s_component[x_bins_s_component<-7] = 0
                                   
#                    """
#                    Use this when using the rectangular coordinates.
#                    """
#                    from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
#
#                    spl_x = UnivariateSpline(.5*(x_axis_bins[1:] + x_axis_bins[:-1]), 
#                                             x_bins_s_component, 
#                                             s = 0.02,
#                                             ext=1)
#                    
##                    spl_x = LSQUnivariateSpline(.5*(x_axis_bins[1:] + x_axis_bins[:-1]), 
##                                                x_bins_s_component, 
##                                                t=[0, vid_ve[0].shape[1]])         
#                    xline = np.arange(0, vid_ve[0].shape[1])
#                    yline = spl_x(xline)
#                    yline[np.isnan(yline)] = 0
#                    yline[yline<-25] = 0
#                    max_x = xline[np.argmin(yline)]
                    
                    from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
                        
                    border_padding = 3 # use 3 bins worth of padding.
                    xspacing = x_axis_bins[1] - x_axis_bins[0]
                    
                    xinterp = .5*(x_axis_bins[1:] + x_axis_bins[:-1])
                    yinterp = x_bins_s_component
                    xinterp = np.hstack([xinterp[0] - np.arange(1,border_padding+1)[::-1]*xspacing, 
                                         xinterp, 
                                         xinterp[-1] + np.arange(1,border_padding+1)*xspacing])
                    yinterp = np.hstack([yinterp[-border_padding:], 
                                         yinterp, 
                                         yinterp[:border_padding]])

                    spl_x = UnivariateSpline(xinterp, 
                                             yinterp, 
                                             s = 0.02,
                                             ext=1)
                    
#                    spl_x = LSQUnivariateSpline(.5*(x_axis_bins[1:] + x_axis_bins[:-1]), 
#                                                x_bins_s_component, 
#                                                t=[0, vid_ve[0].shape[1]])         
                    xline = np.arange(0, vid_ve[0].shape[1])
                    yline = spl_x(xline)
                    yline[np.isnan(yline)] = 0
#                    yline[yline>25] = 0
                    yline[yline<-25] = 0
                    
#                    max_x = xline[np.argmax(yline)]
                    max_x = xline[np.argmin(yline)]
                    
                    correct_angle = (max_x - vid_ve[0].shape[1]/2.) / float(vid_ve[0].shape[1]) * 360 
                    all_correct_angles_rect.append(correct_angle)
                    
                    print(correct_angle)
                    
                    fig, ax = plt.subplots()
                    ax.plot(xinterp, yinterp)
                    ax.plot(xline, yline)
                    plt.show()
                   
                    mean_ve_rect_directions_3D = np.mean(disps_3D_polar_AVE[:,:], axis=1)
            #        mean_ve_polar_directions_3D = np.mean(disps_3D_polar_AVE[:,:]/(np.linalg.norm(disps_3D_polar_AVE, axis=-1) + 1e-8 )[...,None], axis=1)
                    # lets normalise?
                    mean_ve_rect_directions_3D = mean_ve_rect_directions_3D/(np.linalg.norm(mean_ve_rect_directions_3D, axis=-1) + 1e-8)[:,None]
                    mean_ve_rect_directions_reg_3D = np.median(mean_ve_rect_directions_3D, axis=0) # use the mean?
                    
                    mean_ve_rect_directions_reg_3D_proj2D = lookup_3D_to_2D_directions(unwrap_params_ve, 
                                                                                       mean_pos, 
                                                                                       mean_pos_3D, 
                                                                                       mean_ve_rect_directions_reg_3D,
                                                                                       scale=None) # use a scale? 
                    
                    mean_pos_3D_2D = lookup_3D_to_2D_pos(unwrap_params_ve, mean_pos_3D)
                    
                    print(mean_pos_3D_2D)
                    print(mean_pos)
#                    angle_vertical = np.arctan2(mean_ve_rect_directions_reg_3D_proj2D[1], 
#                                                mean_ve_rect_directions_reg_3D_proj2D[0])
#
#                    correct_angle = 180-angle_vertical/np.pi*180; all_correct_angles_polar.append(correct_angle)
#                    print(180-angle_vertical/np.pi*180)
                    
                    mean_ve_rect_directions_reg = mean_ve_rect_directions_reg_3D_proj2D
                    mean_ve_rect_directions_reg = mean_ve_rect_directions_reg / float(np.linalg.norm(mean_ve_rect_directions_reg) + 1e-8)
        
                    
                    # draw the angle!. 
                    fig, ax = plt.subplots(figsize=(10,10))
                    ax.imshow(vid_ve[0], cmap='gray')
                    
#                    rect_center = np.hstack([ve_seg_rect.shape[0]-10, mean_pos[1]])
                    rect_center = mean_pos
                    ax.plot(rect_center[1], rect_center[0], 'wo', ms=15)
                    ax.plot(mean_pos_3D_2D[1], mean_pos_3D_2D[0], 'go', ms=15)
                    ax.plot([rect_center[1], rect_center[1]+100*mean_ve_rect_directions_reg[1]],
                            [rect_center[0], rect_center[0]+100*mean_ve_rect_directions_reg[0]],'w--', lw=5)
                    ax.plot([mean_pos_3D_2D[1], mean_pos_3D_2D[1]+300*mean_ve_rect_directions_reg[1]],
                            [mean_pos_3D_2D[0], mean_pos_3D_2D[0]+300*mean_ve_rect_directions_reg[0]],'w--', lw=5)
            #        ax.plot(np.hstack([rect_polar_pts[convexhull_rect.vertices,1], rect_polar_pts[convexhull_rect.vertices[0],1]]), 
            #                np.hstack([rect_polar_pts[convexhull_rect.vertices,0], rect_polar_pts[convexhull_rect.vertices[0],0]]), 'w-', lw=2)
#                    ax.imshow(density_rect_pts_mask>0, cmap='Reds', alpha=0.5)
#                    ax.quiver(ve_track[:,0,1], 
#                              ve_track[:,0,0], 
#                              track_2D_mean_disps[:,1],
#                              -track_2D_mean_disps[:,0], color = 'r', scale=20)
                    ax.quiver(ve_track[:,0,1], 
                              ve_track[:,0,0], 
                              track_2D_mean_disps[:,1],
                              -track_2D_mean_disps[:,0], color = 'r')
                    ax.plot(ve_seg_rect_cont[:,1], 
                            ve_seg_rect_cont[:,0], 'g--', lw=5)
                    
                    plt.vlines(max_x, 0, vid_ve[0].shape[0]-1, colors='w')
                    plt.vlines(vid_ve[0].shape[1]/2., 0, vid_ve[0].shape[0]-1, colors='k')
                    fig.savefig(os.path.join(all_save_plots_folder, unwrapped_condition+'_rect_direction3D.svg'), bbox_inches='tight')
#                    fig.savefig(os.path.join(all_save_plots_folder, embryo_polar+'_polar_direction3D.svg'), bbox_inches='tight')
                    plt.show()


         
        # summarise with tables. 
        all_polar_table_output = pd.DataFrame(np.vstack([all_embryos_polar,
                                                         all_correct_angles_polar]).T, columns=['Embryo', 'Angle'], index=None)
        all_rect_table_output = pd.DataFrame(np.vstack([all_embryos_rect,
                                                         all_correct_angles_rect]).T, columns=['Embryo', 'Angle'], index=None)
            
        all_polar_table_output.to_csv(os.path.join(all_save_plots_folder, 
                                                   os.path.split(rootfolder)[-1]+'_polar_angles_AVE_classifier.csv'), index=None)
        
        all_rect_table_output.to_csv(os.path.join(all_save_plots_folder, 
                                                   os.path.split(rootfolder)[-1]+'_rect_angles_AVE_classifier.csv'), index=None)
        
        
#        all_ve_direction_displacements_polar = np.vstack(all_ve_direction_displacements_polar)
#        
#        # save out the polar directions for radial grid analysis! 
#        spio.savemat(os.path.join(all_save_plots_folder, 
#                                  os.path.split(rootfolder)[-1]+'_polar_AVE_classifier_displacements.mat', 
#                                  {'embryos': all_embryos_polar, 
#                                   'directions_3D': all_ve_direction_displacements_polar[:,:3], 
#                                   'directions_2D': all_ve_direction_displacements_polar[:, 3:]})
        
        
#        
##        # apply a minimum magnitude
##        if np.linalg.norm(mean_ve_polar_directions_reg_3D) < 0.85:
##            mean_ve_polar_directions_reg_3D_proj2D[0] = 0 # zero the y component. 
#        
#        angle_vertical = np.arctan2(mean_ve_polar_directions_reg_3D_proj2D[1], 
#                                    mean_ve_polar_directions_reg_3D_proj2D[0])
#        print(180-angle_vertical/np.pi*180)
#        
#        mean_ve_polar_directions_reg = mean_ve_polar_directions_reg_3D_proj2D
#        mean_ve_polar_directions_reg = mean_ve_polar_directions_reg / float(np.linalg.norm(mean_ve_polar_directions_reg) + 1e-8)
#        
##        plt.figure()
##        plt.imshow(all_ve_vid_frame0_polar[track_i], cmap='gray')
##        polar_center = mean_pos
##        plt.plot(polar_center[1], polar_center[0], 'wo', ms=15)
##        plt.plot([polar_center[1], polar_center[1]+100*mean_ve_polar_directions_reg[1]],
##                [polar_center[0], polar_center[0]+100*mean_ve_polar_directions_reg[0]],'w--', lw=5)
##        plt.plot([polar_center[1], polar_center[1]+10*mean_ve_polar_directions_reg_3D_proj2D[1]],
##                [polar_center[0], polar_center[0]+10*mean_ve_polar_directions_reg_3D_proj2D[0]],'r--', lw=5)
##        plt.show()
#        
#        """
#        Plot the average 2D directions.... 
#        """
#        track_2D_mean_disps = track_2D_int_polar[:,1:] - track_2D_int_polar[:,:-1]
#        track_2D_mean_disps = track_2D_disps.mean(axis=1) 
#        
#  
#        
#        fig, ax = plt.subplots(figsize=(10,10))
#        ax.imshow(all_ve_vid_frame0_polar[track_i], cmap='gray')
##        ax.plot(track_label_rect_polar_ids_j[track_label_rect>0], 
##                track_label_rect_polar_ids_i[track_label_rect>0], 'w.')
#        ax.plot(track_2D[polar_ve_select,0,1]*all_2d_shapes_polar[track_i][1], 
#                track_2D[polar_ve_select,0,0]*all_2d_shapes_polar[track_i][0], 'go')
#        
##        polar_center = np.hstack(all_ve_vid_frame0_polar[track_i].shape)/2.
#        polar_center = mean_pos
#        ax.plot(polar_center[1], polar_center[0], 'wo', ms=15)
#        ax.plot([polar_center[1], polar_center[1]+100*mean_ve_polar_directions_reg[1]],
#                [polar_center[0], polar_center[0]+100*mean_ve_polar_directions_reg[0]],'w--', lw=5)
##        ax.plot(np.hstack([rect_polar_pts[convexhull_rect.vertices,1], rect_polar_pts[convexhull_rect.vertices[0],1]]), 
##                np.hstack([rect_polar_pts[convexhull_rect.vertices,0], rect_polar_pts[convexhull_rect.vertices[0],0]]), 'w-', lw=2)
#        ax.imshow(density_rect_pts>np.mean(density_rect_pts) + 1.*np.std(density_rect_pts), cmap='Reds', alpha=0.5)
#        ax.quiver(track_2D[:,0,1]*all_2d_shapes_polar[track_i][1], 
#                  track_2D[:,0,0]*all_2d_shapes_polar[track_i][0], 
#                  track_2D_mean_disps[:,1]*all_2d_shapes_polar[track_i][1],
#                  -track_2D_mean_disps[:,0]*all_2d_shapes_polar[track_i][0], color = 'r', scale=20)
#        fig.savefig(os.path.join(saveoutfolder, embryo_polar+'_SVM_polar_w_direction3D.svg'), bbox_inches='tight')
#        plt.show()
#        
#        
#        fig, ax = plt.subplots(figsize=(10,10))
#        ax.imshow(all_ve_vid_frame0_polar[track_i], cmap='gray')
##        ax.plot(track_label_rect_polar_ids_j[track_label_rect>0], 
##                track_label_rect_polar_ids_i[track_label_rect>0], 'w.')
#        ax.plot(track_2D[polar_ve_select,0,1]*all_2d_shapes_polar[track_i][1], 
#                track_2D[polar_ve_select,0,0]*all_2d_shapes_polar[track_i][0], 'go')
#        
##        polar_center = np.hstack(all_ve_vid_frame0_polar[track_i].shape)/2.
#        polar_center = mean_pos
#        ax.plot(polar_center[1], polar_center[0], 'wo', ms=15)
#        ax.plot([polar_center[1], polar_center[1]+100*mean_ve_polar_directions_reg[1]],
#                [polar_center[0], polar_center[0]+100*mean_ve_polar_directions_reg[0]],'w--', lw=5)
##        ax.plot(np.hstack([rect_polar_pts[convexhull_rect.vertices,1], rect_polar_pts[convexhull_rect.vertices[0],1]]), 
##                np.hstack([rect_polar_pts[convexhull_rect.vertices,0], rect_polar_pts[convexhull_rect.vertices[0],0]]), 'w-', lw=2)
#        ax.imshow(density_rect_pts_mask>0, cmap='Reds', alpha=0.5)
#        ax.quiver(track_2D[:,0,1]*all_2d_shapes_polar[track_i][1], 
#                  track_2D[:,0,0]*all_2d_shapes_polar[track_i][0], 
#                  track_2D_mean_disps[:,1]*all_2d_shapes_polar[track_i][1],
#                  -track_2D_mean_disps[:,0]*all_2d_shapes_polar[track_i][0], color = 'r', scale=20)
#        fig.savefig(os.path.join(saveoutfolder, embryo_polar+'_SVM_polar_persistent_filter_w_direction3D.svg'), bbox_inches='tight')
#        plt.show()
        
#        
#        """
#        now we need to compute and save the correct statistics to transform by .... 
#        """
#        # this is the aligned meantracksfile
#        aligned_meantrack_file_in = all_embryos_polar_savefolders[track_i][0]
#        aligned_meantrack_file_out = aligned_meantrack_file_in.replace(savetracksfolderkey, newsavetracksfolderkey).replace('.mat', '-ve_aligned.mat')
#        saveoutfolder = os.path.split(aligned_meantrack_file_out)[0]
#        fio.mkdir(saveoutfolder)
#        
##        print(saveoutfolder)
##        aligned_meantrack_file_out = aligned_meantrack_file_out.replace('-5000','-1000')
## =============================================================================
##         reparse the aligned track we use for plotting purposes. 
## =============================================================================
#        if migration_exist:
#            savetrackfolder_ve_stage = os.path.join(all_embryos_polar_savefolders[track_i][0].split('/MOSES_analysis')[0], 'MOSES_analysis', 'tracks_all_time-aligned-5000_%s-manual' %(staging_phase))
#            ve_track_file_stage_polar = os.path.join(savetrackfolder_ve_stage, 'deepflow-meantracks-%d_%s_' %(n_spixels,staging_phase)+embryo_polar+'.mat')
#            
#            aligned_in_ve_track = spio.loadmat(ve_track_file_stage_polar)['meantracks_ve']
#            aligned_in_epi_track = spio.loadmat(ve_track_file_stage_polar)['meantracks_epi']
#        else:
##            aligned_in_ve_track = spio.loadmat(aligned_meantrack_file_in)['meantracks_ve']
#            try:
#                aligned_in_ve_track = spio.loadmat(aligned_meantrack_file_in)['meantracks']
#            except:
#                aligned_in_ve_track = spio.loadmat(aligned_meantrack_file_in)['meantracks_ve']
##            aligned_in_epi_track = spio.loadmat(aligned_meantrack_file_in)['meantracks_epi']
#        
#        # invert the tracks if need be... 
#        if embryo_inversion:
#            aligned_in_ve_track[...,0] = embryo_polar_im.shape[0] - aligned_in_ve_track[...,0]
#            aligned_in_epi_track[...,0] = embryo_polar_im.shape[0] - aligned_in_epi_track[...,0]
#        
##        
##        fig, ax = plt.subplots(figsize=(15,15))
##        ax.imshow(embryo_polar_im, cmap='gray', alpha=0.25)
##        plot_tracks(aligned_in_ve_track[:,:30], ax=ax, color='r')
##        plt.show()
##        
#        rot_center = np.hstack(embryo_polar_im.shape)/2. 
#        correct_angle = 180-angle_vertical/np.pi*180
#        
#        # get the rotated version of the tracks. 
#        rotated_tracks_ve = rotate_tracks(aligned_in_ve_track, angle=-correct_angle, center=rot_center)
#        rotated_tracks_epi = rotate_tracks(aligned_in_epi_track, angle=-correct_angle, center=rot_center)
#        
#        rotated_motion_center = rotate_pts(mean_pos[::-1][None,:], angle=-correct_angle, center=rot_center).ravel()[::-1]
#        rotated_mean_vector = rotate_pts(mean_ve_polar_directions_reg[::-1][None,:], angle=-correct_angle, center=[0,0]).ravel()[::-1]
#        
#        #. 0. save the non rotated density mask. 
#        rotate_density_image = sktform.rotate(density_rect_pts_mask, angle=correct_angle, preserve_range=True)
#        rotate_initial_image = sktform.rotate(embryo_polar_im, angle=correct_angle, preserve_range=True)
#        
#        # 1. rotation angle + the rotation (motion center) which is required to set up the radial segments. 
##        from scipy.ndimage.measurements import center_of_mass
##        COM_VE = np.hstack(list(center_of_mass(rotate_density_image>0, labels=rotate_density_image>0)))
#        
#        all_correct_angles.append(correct_angle)
#        
#        """
#        save out the inferred parameters which is used for analysis (VE mask, VE motion angle, center + ve_classified points)
#        """
#        print(aligned_meantrack_file_out)
#        print(len(track_label_rect_valid) , len(track_2D_int_rect))
#        assert(len(track_label_rect_valid) == len(track_2D_int_rect))
#        spio.savemat(aligned_meantrack_file_out, {#'meantracks_ve':rotated_tracks_ve.astype(np.float), 
#                                                  #'meantracks_epi':rotated_tracks_epi.astype(np.float), 
#                                                  've_angle': angle_vertical, 
#                                                  've_center': mean_pos,
#                                                  've_vector': mean_ve_polar_directions_reg,
#                                                  'correct_angle': correct_angle,
#                                                  'density_rect_pts': density_rect_pts,
#                                                  'density_polar_pts': density_rect_pts,
#                                                  'density_rect_pts_mask_SVM': density_rect_pts_mask_SVM,
#                                                  'density_polar_pts_mask_SVM': density_polar_pts_mask_SVM,
#                                                  'density_rect_pts_mask': density_rect_pts_mask,
#                                                  'density_polar_pts_mask': density_polar_pts_mask,
#                                                  'embryo_inversion': embryo_inversion,
#                                                  'migration_exist':migration_exist,
#                                                  'migration_stage_times': migration_stage, 
#                                                  'embryo_polar': embryo_polar,
#                                                  'embryo_rect': embryo_rect,
##                                                  've_rect_select': track_label_rect,
##                                                  've_rect_select_valid': track_label_rect_valid,
##                                                  'rect_polar_yx': np.vstack([track_label_rect_polar_ids_i, 
##                                                                              track_label_rect_polar_ids_j]).T,
#                                                  'track_file': aligned_meantrack_file_in
#                                                  })
#        
#        # 2. compute the associated rotated tracks  based on the image statistics. 
#        fig, ax = plt.subplots(figsize=(10,10))
#        ax.imshow(rotate_initial_image, cmap='gray')
##        ax.imshow(rotate_density_image, cmap='Reds', alpha=0.5) # this needs to be blended? 
#        rotate_density_image_cmap = np.zeros((rotate_density_image.shape[0],
#                                              rotate_density_image.shape[1], 
#                                              4))
#        rotate_density_image_cmap[...,1] = rotate_density_image
#        rotate_density_image_cmap[np.logical_not(rotate_density_image),-1] = 0
#        rotate_density_image_cmap[rotate_density_image>0,-1] = .25
#        ax.imshow(rotate_density_image_cmap) # this needs to be blended? 
#        
##        plot_tracks(rotated_tracks_ve[:,:50,::-1], ax=ax, color='r')
#        rot_track_vect = np.mean(rotated_tracks_ve[:,1:] -rotated_tracks_ve[:,:-1], axis=1)
#        ax.plot(rotated_motion_center[1], rotated_motion_center[0], 'wo', ms=15)
##        ax.plot(COM_VE[1], COM_VE[0], 'go', ms=5,zorder=1e6)
#        ax.plot([rotated_motion_center[1], rotated_motion_center[1]+100*rotated_mean_vector[1]],
#                [rotated_motion_center[0], rotated_motion_center[0]+100*rotated_mean_vector[0]],'w--', lw=5)
#        
#        ax.quiver(rotated_tracks_ve[:,0,1], 
#                  rotated_tracks_ve[:,0,0], 
#                  rot_track_vect[:,1],
#                  -rot_track_vect[:,0], color = 'r')
#        if embryo_inversion:
#            fig.savefig(aligned_meantrack_file_out.replace('.mat', '-inverted.svg'), dpi=300, bbox_inches='tight')
#        else:
#            fig.savefig(aligned_meantrack_file_out.replace('.mat', '.svg'), dpi=300, bbox_inches='tight')
#        plt.show()
#        
        
        
        
        
        
        
        
        
        
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
    


    
    