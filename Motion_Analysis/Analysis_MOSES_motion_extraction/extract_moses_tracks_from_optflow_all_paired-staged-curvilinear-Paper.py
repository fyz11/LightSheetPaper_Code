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
    import skimage.transform as sktform
    from tqdm import tqdm 
    import pandas as pd 
    
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    
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
    
    """
    set which experiment folder. 
    """
    
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-new_hex-smooth-lam1.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG-HEX-smooth-lam1.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES-MTMG_TTR-smooth-lam1.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/auto_staging_MOSES.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct/auto_staging_MOSES-LifeAct-smooth-lam10-correct_norot.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct/auto_staging_MOSES-MTMG-TTR-smooth-lam10-correct_norot.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct/auto_staging_MOSES-MTMG-HEX-smooth-lam10-correct_norot.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct/auto_staging_MOSES-MTMG-new-HEX-smooth-lam10-correct_norot.xlsx'
    
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-LifeAct-smooth-lam10-correct_norot.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-MTMG-TTR-smooth-lam10-correct_norot.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-MTMG-HEX-smooth-lam10-correct_norot.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_uniform-scale/auto_staging_MOSES-new_hex-smooth-lam10-correct_norot.xlsx'
    
    
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Paper/auto_staging_MOSES-LifeAct-smooth-lam5-correct_norot_consensus.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Paper/auto_staging_MOSES-MTMG-TTR-smooth-lam5-correct_norot_consensus.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Paper/auto_staging_MOSES-MTMG-HEX-smooth-lam5-correct_norot_consensus.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Paper/auto_staging_MOSES-MTMG-new-HEX-smooth-lam5-correct_norot_consensus.xlsx'

    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-LifeAct-smooth-lam5-correct_norot_consensus.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-TTR-smooth-lam5-correct_norot_consensus.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-HEX-smooth-lam5-correct_norot_consensus.xlsx'
#    auto_staging_file = '/media/felix/Srinivas4/Data_Info/Manual_Staging/Auto_Staging_MOSES_correct_norot_3D_Polar-Curvilinear-Paper/auto_staging_MOSES-MTMG-new-HEX-smooth-lam5-correct_norot_consensus.xlsx'

    
    auto_staging_table = pd.read_excel(auto_staging_file)
    all_stages = auto_staging_table.columns[1:4]
    
    for embryo_folder in embryo_folders[:]:
        
        # find the geodesic-rotmatrix. 
        unwrapped_folder = locate_unwrapped_im_folder(embryo_folder, key='geodesic-rotmatrix')
        
        if len(unwrapped_folder) == 1:
            
            """
            get the embryo name 
            """
            embryo_base_name = os.path.split(embryo_folder)[-1]
            
            print(embryo_base_name)
            print('processing: ', unwrapped_folder[0])
            
            unwrapped_folder = unwrapped_folder[0]
            # determine the savefolder. 
            saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels))
#            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time-aligned-%d' %(n_spixels))
#            print(savetrackfolder)
#            fio.mkdir(savetrackfolder) # create this folder. 
            
            unwrap_im_files = np.hstack(glob.glob(os.path.join(unwrapped_folder, '*.tif')))

            if 'MTMG-TTR' in rootfolder or 'LifeAct' in rootfolder:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_lifeact_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_lifeact_mtmg_embryo(unwrap_im_files, unwrap_meta_files)
            
            else:
                unwrap_meta_files = np.vstack([np.hstack(parse_condition_fnames_hex_unwrapped(ff)) for ff in unwrap_im_files])
                paired_unwrap_file_condition, paired_unwrap_files = pair_meta_file_hex_embryo(unwrap_im_files, unwrap_meta_files)
            
            # iterate over the pairs 
            for ii in tqdm(range(len(paired_unwrap_file_condition))[:]):
                
                unwrapped_condition = paired_unwrap_file_condition[ii]
                
                if 'MTMG-TTR' in rootfolder or 'LifeAct' in rootfolder:
                    ve_unwrap_file, epi_unwrap_file = paired_unwrap_files[ii]
                else:
                    ve_unwrap_file, epi_unwrap_file, hex_unwrap_file = paired_unwrap_files[ii]
                    
#                unwrap_im_file = unwrap_im_files[ii]
#                print(unwrap_im_file)
                vid_ve = fio.read_multiimg_PIL(ve_unwrap_file)
                vid_epi = fio.read_multiimg_PIL(epi_unwrap_file)
                vid_epi_resize = sktform.resize(vid_epi, vid_ve.shape, preserve_range=True)
                
                if 'MTMG-HEX' in rootfolder:
                    vid_hex = fio.read_multiimg_PIL(hex_unwrap_file)
                
                """
                1. load the flow file for VE and Epi surfaces. 
                """
                flowfile_ve = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(ve_unwrap_file)[-1].replace('.tif', '.mat'))
                flow_ve = spio.loadmat(flowfile_ve)['flow']
                
                flowfile_epi = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(epi_unwrap_file)[-1].replace('.tif', '.mat'))
                flow_epi = spio.loadmat(flowfile_epi)['flow']
                flow_epi_resize = sktform.resize(flow_epi, flow_ve.shape, preserve_range=True)
                
                if 'MTMG-HEX' in rootfolder:
                    flowfile_hex = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(hex_unwrap_file)[-1].replace('.tif', '.mat'))
                    flow_hex = spio.loadmat(flowfile_hex)['flow']
                    
                    
                """
                2. set the number of superpixels and produce tracks for each stage if available. 
                """
                select_stage_row = auto_staging_table.loc[auto_staging_table['Embryo'].values == embryo_base_name]
                select_stage_row = select_stage_row.iloc[0] # just choose the first one. 
                
                for stage_i in range(len(all_stages))[:]:
                    
                    stage_times_all = parse_start_end_times(select_stage_row[all_stages]) #.astype(np.int) # get all stages. 
#                    stage_times = parse_start_end_times(select_stage_row[all_stages[stage_i]]).ravel()
                    stage_times = stage_times_all[stage_i]
                    
                    if len(stage_times) > 0 and ~np.isnan(stage_times[0]):
                        # we find that the stage exists.... 
                        stage_times = stage_times.astype(np.int)
                        print(stage_i, stage_times)
                        
#                        savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_stage-aligned-correct_uniform_scale-%d_%s' %(n_spixels, all_stages[stage_i]))
                        savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_stage-auto-polar-curvilinear-aligned-Paper-%d_%s' %(n_spixels, all_stages[stage_i]))
                        print(savetrackfolder)
                        fio.mkdir(savetrackfolder) # create this folder. 
                        
                        
#                        # this is not correct if we have different stages present.!. 
#                        if stage_i==0:
#                            start = stage_times[0]-1
#                            end = stage_times[1]
#                        if stage_i==1:
#                            start = stage_times[0]
#                            end = stage_times[1]
#                        if stage_i == 2:
#                            start = stage_times[0]
#                            end = len(vid_ve) # auto give the end of the video. 
                            
                        start = stage_times[0] - 1
                        end = stage_times[1] - 1
                        
                        if len(vid_ve) - end == 1:
                            end = len(vid_ve)
                            
                        print(stage_i, stage_times, [start, end])
                        
                        
                        if np.isnan(start) == True:
                            continue
                        else:
                            start = int(start)
                            end = int(end)
                            
                            print('start_end: ', np.hstack([start, end]))
                            
                            meantracks_ve = compute_grayscale_vid_superpixel_tracks_w_optflow(vid_ve[start:end], 
                                                                                              flow_ve[start:end], 
                                                                                              n_spixels=n_spixels, 
                                                                                              direction='F', 
                                                                                              dense=False, 
                                                                                              mindensity=1)
                            
                            # compute the resized version. 
                            meantracks_epi = compute_grayscale_vid_superpixel_tracks_w_optflow(vid_epi_resize[start:end], 
                                                                                               flow_epi_resize[start:end], 
                                                                                               n_spixels=n_spixels, 
                                                                                               direction='F', 
                                                                                               dense=False, 
                                                                                               mindensity=1)
                            if 'MTMG-HEX' in rootfolder:
                                meantracks_hex = compute_grayscale_vid_superpixel_tracks_w_optflow(vid_hex[start:end], 
                                                                                                   flow_hex[start:end], 
                                                                                                   n_spixels=n_spixels, 
                                                                                                   direction='F', 
                                                                                                   dense=False, 
                                                                                                   mindensity=1)
                            
            ##                # mask out the irrelevant tracks by the image intensity which deepflow method suffers from. 
            #                meantracks_filt = flow_mag_mask[meantracks[:,0,0], 
            #                                                meantracks[:,0,1]]
            #                meantracks_ = meantracks.copy()
            #                meantracks_[meantracks_filt] = meantracks_[meantracks_filt,0][:,None]
            #                
            ###                flow, meantracks = compute_grayscale_vid_superpixel_tracks(vid, 
            ###                                                                           opt_flow_fn=farnebackflow, 
            ###                                                                           n_spixels=n_spixels, 
            ###                                                                           mask=None, 
            ###                                                                           adaptive_mask=False, 
            ###                                                                           ksize_mask=3, 
            ###                                                                           direction='F', 
            ###                                                                           dense=False, 
            ###                                                                           mindensity=1, 
            ###                                                                           params=optical_flow_params)
            
                            fig, ax = plt.subplots(figsize=(10,10))
                            ax.imshow(vid_ve[0], cmap='gray')
                            plot_tracks(meantracks_ve[:,:], ax=ax, color='g')
    #                        plot_tracks(meantracks_epi, ax=ax, color='r')
    #                        plot_tracks(meantracks_hex, ax=ax, color='b')
                            plt.show()
                            
            #                fig, ax = plt.subplots(figsize=(10,10))
            #                ax.imshow(vid_ve[0], cmap='gray')
            #                ax.plot(meantracks_ve[:,0,1], meantracks_ve[:,0,0],'go')
            #                ax.plot(meantracks_epi[:,0,1], meantracks_ve[:,0,0],'r.')
            ##                plot_tracks(meantracks_hex, ax=ax, color='b')
            #                plt.show()
                            
                            """
                            3. save the tracks with min_I?
                            """
                            trackfile = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_%s_' %(n_spixels, all_stages[stage_i]) + unwrapped_condition+'.mat')
                            print(trackfile)
                            
                            
                            if 'MTMG-HEX' in rootfolder:
                                spio.savemat(trackfile, {'meantracks_ve': meantracks_ve, 
                                                         'meantracks_epi': meantracks_epi,
                                                         'meantracks_hex': meantracks_hex,
                                                         'condition':unwrapped_condition,
                                                         'n_spixels':n_spixels,
                                                         'start_end': np.hstack([start, end])})
                            else:
                                spio.savemat(trackfile, {'meantracks_ve': meantracks_ve, 
                                                         'meantracks_epi': meantracks_epi,
                                                         'condition':unwrapped_condition,
                                                         'n_spixels':n_spixels,
                                                         'start_end': np.hstack([start, end])})
                    
                    

