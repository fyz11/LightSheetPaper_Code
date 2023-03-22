# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 23:12:59 2020

@author: felix
"""

import numpy as np

def parse2array(tracks, start_col_id=1):
    
#    times = np.hstack([int(t.split('_')[1]) for t in tracks.columns[1:]]) - 1
    times = np.arange(len(tracks.columns[start_col_id:]))
#    track_ids = np.unique(tracks.iloc[:,0].values)
    print(times)
    all_tra_ids = tracks.iloc[:,0].values
    uniq_tra_ids = np.unique(all_tra_ids)
    
    tra_array = np.zeros((len(uniq_tra_ids), len(times), 2))
    print(tra_array.shape)
    
    for ii, uniq_tra in enumerate(uniq_tra_ids):
        
        select = all_tra_ids == uniq_tra
        data = tracks.loc[select].values
#        print(uniq_tra, data.shape)
        data = data[:,start_col_id:].astype(np.float)
        data[data==0] = np.nan
        
#        print(data.shape, uniq_tra)
        tra_array[ii] = data[:2].T
        
    return uniq_tra_ids, tra_array
    

def compute_start_end_times(tracks):
    
    start_end_times = []
    times = np.arange(tracks.shape[1])
    
    for tra in tracks:
        t_tra = times[~np.isnan(tra[...,0])]
        if len(t_tra) > 0:
            start_end_times.append([t_tra[0],
                                    t_tra[-1]])
        else:
            start_end_times.append([0,
                                    0])
    
    start_end_times = np.vstack(start_end_times)
    
    return start_end_times


def parse_cell_tracks_only( all_track_ids, all_track_info, len_cutoff=1):
    
    start_end_times = compute_start_end_times(all_track_info)
    tracklentimes = start_end_times[:,1] + 1 - start_end_times[:,0]
    
    select = tracklentimes > len_cutoff
    
    return start_end_times[select], tracklentimes[select], all_track_ids[select], all_track_info[select]


def parse_cell_divs(divtable):
    
    """
    Time_stamp, mother_track_id, mother_x , mother_y, daug1_track _id, daug1_x, daug1_y, daug2_track _id, daug2_x, daug2_y
    """
    div_vals = divtable.values
    
    div_time = div_vals[:,0]
    mother_tra_ids = div_vals[:,1]
    mother_xy = np.vstack([div_vals[:,2], 
                           div_vals[:,3]]).T
    daughter1_tra_ids = div_vals[:,4]
    daughter1_xy = np.vstack([div_vals[:,5], 
                              div_vals[:,6]]).T
    daughter2_tra_ids = div_vals[:,7]
    daughter2_xy = np.vstack([div_vals[:,8], 
                              div_vals[:,9]]).T
    
    return div_time, (mother_tra_ids, mother_xy), (daughter1_tra_ids, daughter1_xy), (daughter2_tra_ids, daughter2_xy)

def assoc_area_tracks( cell_tracks_ids, measurements_files, key='areas_3D'):
    
    import pandas as pd 
    cell_tracks_measures = np.zeros(cell_tracks_ids.shape)
    
    n_tracks, n_frames = cell_tracks_ids.shape[:2]
    
    for ii in np.arange(n_frames):
        measure_table = pd.read_csv(measurements_files[ii])
        measure_ids = measure_table['Index'].values
        
        for jj in np.arange(n_tracks):
            spixel_jj = cell_tracks_ids[jj,ii]
            
            if spixel_jj > 0:
                select_measure_index = measure_ids == spixel_jj
                select_area = float(measure_table.loc[select_measure_index,key].values)
                cell_tracks_measures[jj,ii] = select_area
    
    return cell_tracks_measures


def mkdir(directory):
    """ check if directory exists and create it through Python if it does not yet.

    Parameters
    ----------
    directory : str
        the directory path (absolute or relative) you wish to create (or check that it exists)

    Returns
    -------
        void function, no return
    """
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return []


def smooth_curves(curves, win_size=3, win_mode='reflect', avg_func=np.nanmean):
    
    import numpy as np 
    
    change_curves = np.zeros(curves.shape)
#    curves_pad = np.pad(curves, pad_width=[[0,0], [win_size//2, win_size//2]], mode=win_mode)
#    print(curves_pad.shape)
    
    n_tracks, n_frames = curves.shape[:2]
    
    for ii in np.arange(n_tracks):
        curve_ii = curves[ii]
        
        select_curve = curve_ii > 0 
        curve_ii_select = curve_ii[select_curve]
        select_curve_ind = np.arange(len(curve_ii))[select_curve]
        
        N = len(curve_ii_select)
        curve_ii_select_pad = np.pad(curve_ii_select, pad_width=[win_size//2, win_size//2], mode=win_mode)
        
        for frame_no in np.arange(N):
            vals = curve_ii_select_pad[frame_no:frame_no+win_size]
            vals_avg = avg_func(vals)
            change_curves[ii, select_curve_ind[frame_no]] = vals_avg
            
    return change_curves
    

def create_segmentation_mask(cells, img, cell_ids=None):
    
    from skimage.color import label2rgb
    
    img_ = np.zeros(img.shape, dtype=np.int)
    img_[img>0] = img[img>0]*1
    
    for cell_id in cell_ids:
        binary = cells == cell_id
        img_[binary>0] = cell_id + 2
        
    img_color = label2rgb(img_, bg_label=0)
        
    return img_color
    

def iou_mask(mask1, mask2):
    
    intersection = np.sum(np.abs(mask1*mask2))
    union = np.sum(mask1) + np.sum(mask2) - intersection # this is the union area. 
    
    overlap = intersection / float(union + 1e-8)
    
    return overlap 



def tracks2array_multi(tracks, n_frames, cell_segs):
    
    YY, XX = np.indices(cell_segs.shape[1:])
    
#    times = np.hstack([int(t.split('_')[1]) for t in tracks.columns[1:]]) - 1
    times = np.arange(n_frames)
    all_tra_ids = range(len(tracks))
    
    tra_array = np.zeros((len(all_tra_ids), len(times), 2))
    tra_array[:] = np.nan
    
    for ii in all_tra_ids:
        
        tra = tracks[ii]
        
        for tt in tra:
            if len(tt) > 0: 
                frame = tt[0]
                spixel_region = tt[1]
                
                spixel_region_mask = np.zeros(cell_segs[0].shape, dtype=np.bool)
                
                for ss in spixel_region:
                    spixel_region_mask[cell_segs[frame]==ss] = 1
                
                tra_y = np.mean(YY[spixel_region_mask])
                tra_x = np.mean(XX[spixel_region_mask])
                
                tra_array[ii, frame, 0] = tra_x
                tra_array[ii, frame, 1] = tra_y
                
    return tra_array


def track_cell_ids_2_array(tracks, n_frames):
    
    times = np.arange(n_frames)
    all_tra_ids = range(len(tracks))
          
    tra_array = np.zeros((len(all_tra_ids), len(times))).astype(np.int) # to save space. 0 = background id anyway. 
#    tra_array[:] = np.nan
    
    for ii in all_tra_ids:
        
        tra = tracks[ii]
        
        for tt in tra:
            if len(tt) > 0: 
                frame = tt[0]
                spixel_region = int(tt[1][0])
                
                tra_array[ii, frame] = spixel_region
                
    return tra_array


if __name__=="__main__":
    
    import numpy as np 
    import pandas as pd 
    import pylab as plt 
    import skimage.io as skio 
    import glob 
    import os 
    import scipy.io as spio
    from scipy.signal import find_peaks
    
    
# =============================================================================
#   Step 0: Data processing and reading in 
# =============================================================================
#    tiffile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\Annotated_Cell_Divisions\\L491_final_version_polar_2nd_edit_resize_MS_Edit.tif'
#    trackfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\Annotated_Cell_Divisions\\Holly_polar_distal_corrected_nn_tracking-full-dist30.csv'
#    divfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\Annotated_Cell_Divisions\\div_polar.csv'
    
    
#     saveoutfolder = 'L455_E3 EDITED BINARY -20201005T094221Z-001/L455_E3 EDITED BINARY'
    
    
# #    tiffile = 'L455_E3 EDITED BINARY -20201005T094221Z-001/L455_E3 EDITED BINARY/L455_E3_Composite_Fusion_Seam_Edited_MS_ver2-1x.tif'
# #    trackfile = 'L455_E3 EDITED BINARY -20201005T094221Z-001/L455_E3 EDITED BINARY/L455_E3_Composite_Fusion_Seam_Edited_MS_ver2-1x-manual_polar_distal_corrected_nn_tracking-full-dist30.csv'
# #    divfile = 'L455_E3 EDITED BINARY -20201005T094221Z-001/L455_E3 EDITED BINARY/L455_E3_div_polar_rectangular_only.csv'
    
#     """
#     2nd version
#     """
# #    tiffile = 'L455_E3 EDITED BINARY/L455_E3_Composite_Fusion_Seam_Edited_MS_ver3-1x.tif'
# #    trackfile = 'L455_E3 EDITED BINARY/L455_E3_Composite_Fusion_Seam_Edited_MS_ver3-1x-manual_polar_distal_corrected_nn_tracking-full-dist30.csv'
# #    divfile = 'L455_E3 EDITED BINARY/L455_E3_div_polar_rectangular_only_refine_MS_p_edit1.csv'
    
# #    tiffile = 'L455_E3 EDITED BINARY/L455_E3_Composite_Fusion_Seam_Edited_MS_ver4-1x.tif'
# #    trackfile = 'L455_E3 EDITED BINARY/L455_E3_Composite_Fusion_Seam_Edited_MS_ver4-1x-manual_polar_distal_corrected_nn_tracking-full-dist30.csv'
# #    divfile = 'L455_E3 EDITED BINARY/L455_E3_div_polar_rectangular_only_refine_MS_p_edit2_refine.csv'
    
#     tiffile = 'L455_E3 EDITED BINARY/L455_E3_Composite_Fusion_Seam_Edited_MS_ver4-1x.tif'
#     trackfile = 'L455_E3 EDITED BINARY/L455_E3_Composite_Fusion_Seam_Edited_MS_ver4-1x-manual_polar_distal_corrected_nn_tracking-full-dist30.csv'
# #    divfile = 'L455_E3 EDITED BINARY/L455_E3_div_polar_rectangular_only_refine_MS_p_edit3_refine2.csv'
#     divfile = 'L455_E3 EDITED BINARY/L455_E3_div_polar_rectangular_only_refine_MS_p_edit4_refine3.csv'
    
    # saveoutfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505' # dump here temporarily. 
    # saveoutfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L864\\2020-12-30\\refine'
    
    saveoutfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2' # dump here temporarily. 
    # saveoutfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2' # dump here temporarily. 

#    tiffile = 'L455_E3 EDITED BINARY -20201005T094221Z-001/L455_E3 EDITED BINARY/L455_E3_Composite_Fusion_Seam_Edited_MS_ver2-1x.tif'
#    trackfile = 'L455_E3 EDITED BINARY -20201005T094221Z-001/L455_E3 EDITED BINARY/L455_E3_Composite_Fusion_Seam_Edited_MS_ver2-1x-manual_polar_distal_corrected_nn_tracking-full-dist30.csv'
#    divfile = 'L455_E3 EDITED BINARY -20201005T094221Z-001/L455_E3 EDITED BINARY/L455_E3_div_polar_rectangular_only.csv'
    
    """
    2nd version
    """

#     # L505 ver. 
#     tiffile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\ver2020-12-31_2\\L505_ve_120_MS_edit_ver5_30_12_2020.tif'
#     trackfile = os.path.join(saveoutfolder, 
#                               'ver2020-12-31_2', 'refine',  
#                               'L505_ve_120_MS_edit_ver5_30_12_2020-manual_polar_distal_corrected_nn_tracking-full-dist30.csv')
# #    divfile = 'L455_E3 EDITED BINARY/L455_E3_div_polar_rectangular_only_refine_MS_p_edit3_refine2.csv'
#     divfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\ver2020-12-31_2\\L505_ve_120_MS_edit_ver5_31_12_2020_cell_div_MS.csv_refine.csv'
    
#     # L864
#     tiffile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L864\\2020-12-30\\L864_composite_matt_binary_edit_ver1.tif'
#     trackfile = os.path.join(saveoutfolder, 
#                              'L864_composite_matt_binary_edit_ver1-manual_polar_distal_corrected_nn_tracking-full-dist30.csv')
# #    divfile = 'L455_E3 EDITED BINARY/L455_E3_div_polar_rectangular_only_refine_MS_p_edit3_refine2.csv'
#     divfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L864\\2020-12-30\\L864_cell_div_ms_edit_30_09_20.csv'
    
    # L455_Emb2 
#     tiffile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\2021-03-22\\22Mar_up2_binary-2x-direct_MS_edit1.tif'
#     trackfile = os.path.join(saveoutfolder, 
#                               '2021-03-22', 'refine',  
#                               '22Mar_up2_binary-2x-direct_MS_edit1-manual_polar_distal_corrected_nn_tracking-full-dist30.csv')
# #    divfile = 'L455_E3 EDITED BINARY/L455_E3_div_polar_rectangular_only_refine_MS_p_edit3_refine2.csv'
#     divfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\2021-03-22\\L455_E2_div_polar_MS_Dec_edit.csv'
    
    tiffile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\2021-03-25\\25Mar_up2_binary-2x-direct_MS_edit2.tif'
    trackfile = os.path.join(saveoutfolder, 
                              '2021-03-25_v2', 'refine',  
                              '25Mar_up2_binary-2x-direct_MS_edit3-manual_polar_distal_corrected_nn_tracking-full-dist30.csv')
# #    divfile = 'L455_E3 EDITED BINARY/L455_E3_div_polar_rectangular_only_refine_MS_p_edit3_refine2.csv'
#     divfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\2021-03-22\\L455_E2_div_polar_MS_Dec_edit.csv'
    # divfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\2021-03-25\\25Mar_L455_E2_div_polar_MS_Dec_edit2.csv'
    divfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\2021-03-25_v2\\25Mar_L455_E2_div_polar_MS_Dec_edit3.csv'

    base_fname = os.path.split(tiffile)[-1].split('.tif')[0]
    
#    measurements_folder = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\L491_ve_000_polar_manual_cells_w_3D_measurements'
#    measurements_files = glob.glob(os.path.join(measurements_folder, '*.csv'))
    
    # tif = skio.imread(tiffile)[:,0] # get the binary 
    tif = skio.imread(tiffile)[:,1]
    tracks = pd.read_csv(trackfile)
#    divs = pd.read_csv(divfile, header=None)
    divs = pd.read_csv(divfile)
    
    """
    0. need to load in the unwrap_params_3D information to lookup 3D distancing information. 
    """
#    unwrapfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\LightSheet_Unwrap-params_Analysis\\L491_ve_000_unwrap_params_unwrap_params-geodesic.mat'
    # unwrapfile = 'L455_E3 EDITED BINARY/L455_Emb3_ve_000_unwrap_params_unwrap_params-geodesic.mat'
    
    # # L505
    # unwrapfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\L505_ve_120_unwrap_params_unwrap_params-geodesic.mat'

    # # L864
    # unwrapfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L864\\L864_ve_180_unwrap_params_unwrap_params-geodesic.mat' # get this file.     
    
    # L455_Emb2
    unwrapfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\L455_Emb2_ve_225_unwrap_params_unwrap_params-geodesic.mat' # get this file. 
    
    unwrap_params_obj = spio.loadmat(unwrapfile)
    unwrap_params = unwrap_params_obj['ref_map_polar_xyz']
    
    
    """
    1. parse the tracks. (not very useful) -> use the original. 
    """
#    all_ids, all_track_info = parse2array(tracks, start_col_id=2)
#    cell_track_start_end, cell_track_lengths, cell_track_ids, cell_tracks = parse_cell_tracks_only( all_ids, all_track_info, len_cutoff=0) 
# =============================================================================
#     Below is the original data from all the tested tracking. 
# =============================================================================
    # use the direct outputs of the original tracking and parse in the given format. 
#    matfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\cell_tracks_w_ids_Holly_polar_distal_corrected_nn_tracking-full-dist30.mat'
#    matfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\cell_tracks_w_ids_Holly_polar_distal_corrected_nn_tracking-full-dist30.mat'
#    matfile = 'L455_E3 EDITED BINARY -20201005T094221Z-001/L455_E3 EDITED BINARY/cell_tracks_w_ids_L455_E3_Composite_Fusion_Seam_Edited_MS_ver2-1x-manual_polar_distal_corrected_nn_tracking-full-dist30.mat'
#    matfile = 'L455_E3 EDITED BINARY/cell_tracks_w_ids_L455_E3_Composite_Fusion_Seam_Edited_MS_ver3-1x-manual_polar_distal_corrected_nn_tracking-full-dist30.mat'
    # matfile = 'L455_E3 EDITED BINARY/cell_tracks_w_ids_L455_E3_Composite_Fusion_Seam_Edited_MS_ver4-1x-manual_polar_distal_corrected_nn_tracking-full-dist30.mat'
    
    # # L505
    # matfile = os.path.join(saveoutfolder, 
    #                         'ver2020-12-31_2', 'refine',  
    #                         'cell_tracks_w_ids_L505_ve_120_MS_edit_ver5_30_12_2020-manual_polar_distal_corrected_nn_tracking-full-dist30')

    # # L864
    # matfile = os.path.join(saveoutfolder, 
    #                        'cell_tracks_w_ids_L864_composite_matt_binary_edit_ver1-manual_polar_distal_corrected_nn_tracking-full-dist30.mat')
    
    # # L455_Emb2
    # matfile = os.path.join(saveoutfolder, 
    #                         '2021-03-22', 'refine',  
    #                         'cell_tracks_w_ids_22Mar_up2_binary-2x-direct_MS_edit1-manual_polar_distal_corrected_nn_tracking-full-dist30.mat')
    # matfile = os.path.join(saveoutfolder, 
    #                         '2021-03-25', 'refine',  
    #                         'cell_tracks_w_ids_25Mar_up2_binary-2x-direct_MS_edit2-manual_polar_distal_corrected_nn_tracking-full-dist30.mat')
    matfile = os.path.join(saveoutfolder, 
                           '2021-03-25_v2', 'refine', 
                           'cell_tracks_w_ids_25Mar_up2_binary-2x-direct_MS_edit3-manual_polar_distal_corrected_nn_tracking-full-dist30.mat')
                  
    
    trackmat = spio.loadmat(matfile)
    cell_tracks_ids = trackmat['cell_tracks_ids']
    cell_tracks_ids_xy = trackmat['cell_tracks_xy']
#     load the original segmented image from which statistics taken from. 
#    cellfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\cells_L491_final_version_polar_2nd_edit_merge.tif'
#    cellfile = 'L455_E3 EDITED BINARY -20201005T094221Z-001/L455_E3 EDITED BINARY/cells_L455_E3_Composite_Fusion_Seam_Edited_MS_ver2-1x.tif'
#    cellfile = 'L455_E3 EDITED BINARY/cells_L455_E3_Composite_Fusion_Seam_Edited_MS_ver3-1x.tif'
    # cellfile = 'L455_E3 EDITED BINARY/cells_L455_E3_Composite_Fusion_Seam_Edited_MS_ver4-1x.tif'
    
    # # L505
    # cellfile = os.path.join(saveoutfolder, 
    #                         'ver2020-12-31_2', 'refine',  
    #                         'cells_L505_ve_120_MS_edit_ver5_30_12_2020.tif')
    # L864
    # cellfile = os.path.join(saveoutfolder, 
    #                        'cells_L864_composite_matt_binary_edit_ver1.tif')
    
    # L455_Emb2
    # cellfile = os.path.join(saveoutfolder, 
    #                         'annot_2021-02-21', 'refine',  
    #                         'cells_L455_E2_Mared_Composite_Distal_Edited-1-77.tif')
    # cellfile = os.path.join(saveoutfolder, 
    #                         '2021-03-22', 'refine',  
    #                         'cells_22Mar_up2_binary-2x-direct_MS_edit1.tif')
    cellfile = os.path.join(saveoutfolder, 
                            '2021-03-25_v2', 'refine',  
                            'cells_25Mar_up2_binary-2x-direct_MS_edit3.tif')
    cell_img = skio.imread(cellfile)
    
    """
    2. parse the annotated division -> these need to be refined and associated with the proper superpixel ids then to the appropriate tracks:
        “Time_stamp, mother_track_id, mother_x , mother_y, daug1_track _id, daug1_x, daug1_y, daug2_track _id, daug2_x, daug2_y”
    """
    # ids don't work. 
    div_time, (mother_tra_ids, mother_xy), (daughter1_tra_ids, daughter1_xy), (daughter2_tra_ids, daughter2_xy) = parse_cell_divs(divs)
    
    div_time = div_time.astype(np.int) # this is 0-indexed and is fine. 
    mother_xy = mother_xy.astype(np.int)
    daughter1_xy = daughter1_xy.astype(np.int)
    daughter2_xy = daughter2_xy.astype(np.int)
    
    mother_tra_ids = mother_tra_ids.astype(np.int)
    daughter1_tra_ids = daughter1_tra_ids.astype(np.int)
    daughter2_tra_ids = daughter2_tra_ids.astype(np.int)
    
    # think this should be reversed? -> just to check if this was reversed or not ... corresponding to rotation of matrix. 
    plt.figure(figsize=(8,8))
    # plt.imshow(cell_img[0])
    plt.imshow(tif[0], cmap='gray')
    plt.plot(mother_xy[:,0],
             mother_xy[:,1], 'r.')
    plt.plot([daughter1_xy[:,0], daughter2_xy[:,0]],
             [daughter1_xy[:,1], daughter2_xy[:,1]],'g.-')
    # plt.plot(daughter2_xy[:,1],
             # daughter2_xy[:,0],'g.')
    plt.show()
    
# =============================================================================
#   Step 1. track refinement using IoU + superpixel ids. -> objective is to produce the correct time, associated mother superpixel id + daughter superpixel ids. 
# =============================================================================

    """ couple of params """
    time_test = 2 # check +/-2 frames either side. 
    iou_div_thresh = .05
    single_iou_div_thresh = 0.1 
    
    refined_div_times = []
    refined_div_times_original_ids = [] # this is just to check the number we were able to reassign. 
    
    debug_viz = False
    
    for ii in np.arange(len(div_time))[:]:
        
        # get the division times from the annotation data of Matt. 
        div_time_matt = div_time[ii]
        mother_xy_matt = mother_xy[ii]
        daughter1_xy_matt = daughter1_xy[ii]
        daughter2_xy_matt = daughter2_xy[ii]
        
        
        # div_time must be smaller than the len(time)
        if div_time_matt < len(cell_img) - 1:
            
            """
            - 1) check the mother superpixel and check if we can detect valid divisions based on IoU coverage at the given annotated timepoint.  
            """
            # 2 options: search the track..... , or get the superpixel id area. 
            mother_spixel = cell_img[div_time_matt, 
                                     int(mother_xy_matt[1]), # (y,x)
                                     int(mother_xy_matt[0])]
            
            # check whether the next point is a cell division. 
            daughter_spixels = np.setdiff1d(np.unique(cell_img[div_time_matt+1, 
                                                      cell_img[div_time_matt]==mother_spixel]), 0)
            
            # check minimum coverage. -> iou ?
            iou_daughters = [iou_mask(cell_img[div_time_matt]==mother_spixel, 
                                      cell_img[div_time_matt+1]==dd_spixel) for dd_spixel in daughter_spixels]
           
            if len(iou_daughters) > 0:
                # if there are daughters. 
                iou_daughters_valid = np.sum(np.hstack(iou_daughters) > iou_div_thresh)
                
                division_good = iou_daughters_valid >= 2
                
                if division_good: 
                    # submit to another test. 
                    if np.max(iou_daughters_valid) >= 0.6 and np.sort(iou_daughters)[::-1][1] < 0.1:
                        # the objective of this test is to check it isn't just due to significant movement.... 
                        division_good = False
                
                # if still good. 
                if division_good: 
                    
                    mother_spixel_refine = mother_spixel
                    daughter_spixels_refine = daughter_spixels[np.argsort(iou_daughters)[::-1][:2]]
                    
                    refined_div_times.append(np.hstack([div_time_matt, mother_xy_matt, daughter1_xy_matt, daughter2_xy_matt,
                                              mother_spixel_refine, daughter_spixels_refine])) # this should now contain enough information. 
                    refined_div_times_original_ids.append(ii)
                    
                    if debug_viz:
                        plt.figure(figsize=(15,15))
        #                plt.suptitle(str(div_time_matt) + ' '+str(division_good))
                        plt.subplot(121)
                        plt.title('div_id %s'%(str(ii))+' '+str(div_time_matt) + ' '+str(division_good) + str(np.sort(iou_daughters)[::-1][:2]))
                        plt.imshow(tif[div_time_matt])
                        plt.plot(mother_xy_matt[0], 
                                 mother_xy_matt[1], 'r.')
                        plt.subplot(122)
                        plt.imshow(tif[div_time_matt+1])
                        plt.plot(daughter1_xy_matt[0], 
                                 daughter1_xy_matt[1], 'w.')
                        plt.plot(daughter2_xy_matt[0], 
                                 daughter2_xy_matt[1], 'w.')
                        plt.show()        
                    
                else:
                    
                    if debug_viz:
                        plt.figure(figsize=(15,15))
        #                plt.suptitle(str(div_time_matt) + ' '+str(division_good))
                        plt.subplot(121)
        #                plt.title(str(div_time_matt) + ' '+str(division_good))
                        plt.title('div_id %s'%(str(ii))+' '+str(div_time_matt) + ' '+str(division_good) + str(np.sort(iou_daughters)[::-1][:2]))
                        plt.imshow(tif[div_time_matt])
                        plt.plot(mother_xy_matt[0], 
                                 mother_xy_matt[1], 'r.')
                        plt.subplot(122)
                        plt.imshow(tif[div_time_matt+1])
                        plt.plot(daughter1_xy_matt[0], 
                                 daughter1_xy_matt[1], 'w.')
                        plt.plot(daughter2_xy_matt[0], 
                                 daughter2_xy_matt[1], 'w.')
                        plt.show()  
                    
                    # we do search at -1, -2, and try to refine this. 
                    check_indices_minus = - np.arange(time_test+1) + div_time_matt
                    check_indices_minus = check_indices_minus[1:]
                    check_indices_minus = np.hstack([np.maximum(check_indices_minus_, 0) for check_indices_minus_ in check_indices_minus])
        #            check_indices_minus = np.unique(check_indices_minus)
                    
                    check_indices_plus = np.arange(time_test+1) + div_time_matt
                    check_indices_plus = check_indices_plus[1:]
                    check_indices_plus = np.hstack([np.maximum(check_indices_plus_, 0) for check_indices_plus_ in check_indices_plus])
        #            check_indices_plus = np.unique(check_indices_plus)
                    check_indices = np.hstack([check_indices_minus, check_indices_plus])
                    check_indices = np.unique(check_indices)
                    
                    # arrange check indices in order of closeness 
                    check_indices_sort = np.argsort(np.abs(check_indices - div_time_matt))
                    check_indices = check_indices[check_indices_sort]
                
                    # check indices must only be len(vid) - 1
                    check_indices = check_indices[check_indices<len(cell_img)-1]
                
                    """
                    for each of these we do check from the lowest frame. 
                    """
                    for cand_mother_time in check_indices:
                
                        cand_mother_spixel = cell_img[cand_mother_time, 
                                                      int(mother_xy_matt[1]),
                                                      int(mother_xy_matt[0])]
    #                    plt.figure(figsize=(10,10))
    #                    plt.imshow(cell_img[cand_mother_time] == cand_mother_spixel)
    #                    plt.show()
                        # check whether the next point is a cell division. 
                        cand_daughter_spixels = np.setdiff1d(np.unique(cell_img[cand_mother_time+1, 
                                                                       cell_img[cand_mother_time]==cand_mother_spixel]), 0)
    #                    print(cand_daughter_spixels)
                        
                        # check minimum coverage. -> iou ?
                        cand_iou_daughters = [iou_mask(cell_img[cand_mother_time]==cand_mother_spixel, 
                                                       cell_img[cand_mother_time+1]==dd_spixel) for dd_spixel in cand_daughter_spixels]
    #                    print(ii, iou_daughters)
                        if len(cand_iou_daughters) > 0: 
                            iou_daughters_valid = np.sum(np.hstack(cand_iou_daughters) > iou_div_thresh)
                            division_good = iou_daughters_valid >= 2 
                            
                            if division_good: 
                                # submit to another test. 
                                if np.max(cand_iou_daughters) >= 0.6 and np.sort(cand_iou_daughters)[::-1][1] < 0.1:
                                    # this would suggest non division 
                                    division_good = False
                        else:
                            division_good = False
                            
    #                    print(cand_daughter_spixels, cand_iou_daughters, division_good)
                        if division_good:
                            
                            mother_spixel_refine = cand_mother_spixel
                            daughter_spixels_refine = cand_daughter_spixels[np.argsort(cand_iou_daughters)[::-1][:2]]
    
                            refined_div_times.append(np.hstack([cand_mother_time, mother_xy_matt, daughter1_xy_matt, daughter2_xy_matt,
                                                                mother_spixel_refine, daughter_spixels_refine]))
                            
                            refined_div_times_original_ids.append(ii)
                            
                            if debug_viz:               
                                
                                plt.figure(figsize=(15,15))
                                plt.subplot(121)
        #                        plt.title(str(cand_mother_time) + ' '+str(division_good))
                                plt.title('div_id %s'%(str(ii))+' '+str(cand_mother_time) + ' '+str(division_good) + str(np.sort(cand_iou_daughters)[::-1][:2]))
                                plt.imshow(tif[cand_mother_time])
                                plt.plot(mother_xy_matt[0], 
                                         mother_xy_matt[1], 'r.')
                                plt.subplot(122)
                                plt.imshow(tif[cand_mother_time+1])
                                plt.plot(daughter1_xy_matt[0], 
                                         daughter1_xy_matt[1], 'w.')
                                plt.plot(daughter2_xy_matt[0], 
                                         daughter2_xy_matt[1], 'w.')
                                plt.show()        
                                
                                break
                        
#        print('================================')
            
    if len(refined_div_times) > 0:
        
        refined_div_times = np.vstack(refined_div_times)
        refined_div_times_original_ids = np.hstack(refined_div_times_original_ids)
        
        print(refined_div_times.shape)
        print(refined_div_times_original_ids.shape)
        
        
# =============================================================================
#   Use the refined div times to plot. 
# =============================================================================
#        div_time, (mother_tra_ids, mother_xy), (daughter1_tra_ids, daughter1_xy), (daughter2_tra_ids, daughter2_xy) = parse_cell_divs(divs)
        
    saveplotfolder = os.path.join(saveoutfolder, 'debug_%s_div_parsing_Matt_edit-auto_refine' %(base_fname))
    mkdir(saveplotfolder)
#    
    nframes = len(tif)
    
#    for frame_no in np.arange(nframes-1):
#                
#        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
##        plt.suptitle('Matt-Edit')
##        plt.subplot(121)
#        ax[0,0].set_title('Matt: Frame-%s' %(str(frame_no+1).zfill(3)))
#        ax[0,0].imshow(tif[int(frame_no)], cmap='gray')
#        ax[0,1].set_title('Matt: Frame-%s' %(str(frame_no+2).zfill(3)))
#        ax[0,1].imshow(tif[int(frame_no+1)], cmap='gray')
#        
#        
#        for jj in np.arange(len(divs))[:]:
#        
#            div_time_matt = div_time[ii]
#            mother_xy_matt = mother_xy[ii]
#            daughter1_xy_matt = daughter1_xy[ii]
#            daughter2_xy_matt = daughter2_xy[ii]
#            
#            if div_time_matt == frame_no:
#        
#                ax[0,0].plot(mother_xy_matt[0],mother_xy_matt[1], 'r.')
##                ax[0,0].plot(tra_div_xy[0],tra_div_xy[1], 'y.')
##                ax[0,1].plot([daughter1_xy_matt[0], daughter2_xy_matt[0]],
##                         [daughter1_xy_matt[1], daughter2_xy_matt[1]], 'g.-')
##                ax[1,0].plot(mother_xy_matt[0],mother_xy_matt[1], 'r.')
##                ax[1,0].plot(tra_div_xy[0],tra_div_xy[1], 'y.')
#                ax[0,1].plot([daughter1_xy_matt[0], daughter2_xy_matt[0]],
#                         [daughter1_xy_matt[1], daughter2_xy_matt[1]], 'g.-')
#        
#        
#        ax[1,0].set_title('Matt auto refine: Frame-%s' %(str(frame_no+1).zfill(3)))
#        ax[1,0].imshow(tif[int(frame_no)], cmap='gray')
#        ax[1,1].set_title('Matt auto refine: Frame-%s' %(str(frame_no+2).zfill(3)))
#        ax[1,1].imshow(tif[int(frame_no)+1], cmap='gray')
#        
#        
#        for jj in np.arange(len(refined_div_times))[:]:
#            
#            div_refine_time = refined_div_times[jj][0]
#            mother_refine_xy = refined_div_times[jj][1:3]
#            daughter1_refine_xy = refined_div_times[jj][3:5]
#            daughter2_refine_xy = refined_div_times[jj][5:7]
#            
#            if div_refine_time == frame_no:
#        
#                ax[1,0].plot(mother_refine_xy[0],mother_refine_xy[1], 'r.')
##                ax[0,0].plot(tra_div_xy[0],tra_div_xy[1], 'y.')
##                ax[0,1].plot([daughter1_refine_xy[0], daughter1_refine_xy[0]],
##                         [daughter2_refine_xy[1], daughter2_refine_xy[1]], 'g.-')
##                ax[1,0].plot(mother_xy_matt[0],mother_xy_matt[1], 'r.')
##                ax[1,0].plot(tra_div_xy[0],tra_div_xy[1], 'y.')
#                ax[1,1].plot([daughter1_refine_xy[0], daughter2_refine_xy[0]],
#                         [daughter1_refine_xy[1], daughter2_refine_xy[1]], 'g.')
                
    for frame_no in np.arange(nframes):
                
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
#        plt.suptitle('Matt-Edit')
#        plt.subplot(121)
        ax[0].set_title('Matt: Frame-%s' %(str(frame_no+1).zfill(3)))
        ax[0].imshow(tif[int(frame_no)], cmap='gray')
        ax[1].set_title('Matt auto refine: Frame-%s' %(str(frame_no+1).zfill(3)))
        ax[1].imshow(tif[int(frame_no)], cmap='gray')
        
        for jj in np.arange(len(divs))[:]:
        
            div_time_matt = div_time[jj]
            mother_xy_matt = mother_xy[jj]
            daughter1_xy_matt = daughter1_xy[jj]
            daughter2_xy_matt = daughter2_xy[jj]
            
            if div_time_matt == frame_no:
                # plot the mother
                ax[0].plot(mother_xy_matt[0],mother_xy_matt[1], 'r.')
            if div_time_matt + 1 == frame_no:
                ax[0].plot([daughter1_xy_matt[0], daughter2_xy_matt[0]],
                           [daughter1_xy_matt[1], daughter2_xy_matt[1]], 'g.-')
#        
        for jj in np.arange(len(refined_div_times))[:]:
            
            div_refine_time = refined_div_times[jj][0]
            mother_refine_xy = refined_div_times[jj][1:3]
            daughter1_refine_xy = refined_div_times[jj][3:5]
            daughter2_refine_xy = refined_div_times[jj][5:7]
            
            if div_refine_time == frame_no:
                ax[1].plot(mother_refine_xy[0],mother_refine_xy[1], 'r.')
            if div_refine_time + 1 == frame_no:
                ax[1].plot([daughter1_refine_xy[0], daughter2_refine_xy[0]],
                           [daughter1_refine_xy[1], daughter2_refine_xy[1]], 'g.-')
                
                
        fig.savefig(os.path.join(saveplotfolder, 'Frame-%s.png' %(str(frame_no+1).zfill(3))), bbox_inches='tight')
        plt.show()
    
        
# =============================================================================
#   Having refined, use this to get the associated tracks and try to break? / associate ... ?
# =============================================================================
        
    assigned_cell_track_ids = []
    
    # make a carbon copy of the current set of track ids to show whether to break the tracks or not in light of the refined tracking data.  
    cell_tracks_ids_symbols = np.zeros_like(cell_tracks_ids)
    cell_tracks_ids_symbols[cell_tracks_ids>0] = 1
    
    # a breakage symbol will be given -1. 
    # a start symbol? 
    
    
    for ii in np.arange(len(refined_div_times))[:]:
        
        tp_div_ii, mother_x, mother_y, daughter1_x, daughter1_y, daughter2_x, daughter2_y, mother_spixel_ii, daughter1_spixel, daughter2_spixel = refined_div_times[ii]
                                                            
        
        mother_cell_tracks_id = np.arange(len(cell_tracks_ids))[cell_tracks_ids[:,tp_div_ii] == mother_spixel_ii][0]
        daughter_1_cell_tracks_id = np.arange(len(cell_tracks_ids))[cell_tracks_ids[:,tp_div_ii+1] == daughter1_spixel]
        daughter_2_cell_tracks_id = np.arange(len(cell_tracks_ids))[cell_tracks_ids[:,tp_div_ii+1] == daughter2_spixel]
        
        
        # for the last time point one of the daughters may not be present..... 
        if len(daughter_1_cell_tracks_id) > 0:
            daughter_1_cell_tracks_id = daughter_1_cell_tracks_id[0]
        else:
            daughter_1_cell_tracks_id = -1
        
        if len(daughter_2_cell_tracks_id) > 0:
            daughter_2_cell_tracks_id = daughter_2_cell_tracks_id[0]
        else:
            daughter_2_cell_tracks_id = -1 
        
        
        # hopefully one of the daughters shares id with that of the mother. 
        assigned_cell_track_ids.append([mother_cell_tracks_id, daughter_1_cell_tracks_id, daughter_2_cell_tracks_id])
        
        """
        the easiest way to allow duplicate editting and 'breaking of tracks' would be to insert special characters..... 
        """
        # for mother cells, we insert a termination character at the given timepoint. For daughter cells we need to insert a begin symbol. 
        # think we just need to terminate..... at this point. for all involved tracks. 
        cell_tracks_ids_symbols[mother_cell_tracks_id, tp_div_ii] = -1 # terminate
        
        if daughter_1_cell_tracks_id >= 0:
            cell_tracks_ids_symbols[daughter_1_cell_tracks_id, tp_div_ii] = -1 # terminate -> this is to handle case of termination in the middle of track. 
        if daughter_2_cell_tracks_id >= 0:
            cell_tracks_ids_symbols[daughter_2_cell_tracks_id, tp_div_ii] = -1 # terminate -> this is handle case of termination in the middle of track. 
        
        
    assigned_cell_track_ids = np.vstack(assigned_cell_track_ids)
    
    # debugging visualisation? 
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(cell_tracks_ids_symbols==-1)
    ax.set_aspect('auto')
    plt.show()
    
# =============================================================================
#   Generate new track ids and associated data given the breakage symbols.
# =============================================================================
    
#    new_track_ids = []
    new_cell_id_tracks = []
    
#    start_tra_id = 0 # counter. 
    
    for ii in np.arange(len(cell_tracks_ids_symbols))[:]:
        
        cell_track_id_ii = cell_tracks_ids[ii]
        cell_track_id_symbol = cell_tracks_ids_symbols[ii]
    
        cell_tra = []
        
        for jj in np.arange(len(cell_track_id_ii)):
            
            if cell_track_id_ii[jj] > 0:
                # then look at the associated symbol.
                action = cell_track_id_symbol[jj]
                
                if action == 1:
                    # keep continuing
                    cell_tra.append([jj, [cell_track_id_ii[jj]]])
                if action == -1:
                    cell_tra.append([jj, [cell_track_id_ii[jj]]])
                    new_cell_id_tracks.append(cell_tra)
                    cell_tra = [] # reset. 
                    
        if len(cell_tra) > 0:
            new_cell_id_tracks.append(cell_tra)
            
            
    # we can now convert the tracks data to the correct array form given the cell image. 
    
# =============================================================================
#     For convenience build the cell tracks all as n_tracks x n_time of cell ids. 
# =============================================================================
    new_cell_tracks_ids_time = track_cell_ids_2_array(new_cell_id_tracks, len(cell_img)) # coerces. 
    
# =============================================================================
#    Build the tracks and plot  
# =============================================================================
#    temporal_merged_tras = temporal_merged_tras + [merged_cell_tras[xxx] for xxx in non_merged_temporal_tracks]
    new_cell_tracks_xy = tracks2array_multi(new_cell_id_tracks, len(cell_img), cell_img)
    
    
# =============================================================================
#     Write out the final form of the merging 
#       1) .mat format easier to work with, including intermediates? 
#       2) .csv format for re-editting. 
#       3) .csv format of positions of cell division events.  
# =============================================================================

    """
    1 ) save tracks out in the .mat format for easy working with.  
    """
#    savematfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\Annotated_Cell_Divisions\\cell_tracks_w_ids_Holly_polar_distal_corrected_nn_tracking-full-dist30_cell-div-refined-Matt.mat'
    savematfile = os.path.join(saveoutfolder, 'cell_tracks_w_ids_'+os.path.split(trackfile)[-1].split('-manual')[0]+'_div-refine.mat')
#    savematfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\Annotated_Cell_Divisions\\cell_tracks_w_ids_Holly_polar_distal_corrected_nn_tracking-full-dist30_cell-div-refined-Matt.mat'
    
    spio.savemat(savematfile, 
                 {'refined_div_times':refined_div_times[:,0],
                  'mother_xy': refined_div_times[:,1:3], 
                  'daughter_1_xy': refined_div_times[:,3:5],
                  'daughter_2_xy': refined_div_times[:,5:7],
                  'mother_cell_id': refined_div_times[7],
                  'daughter_1_cell_id': refined_div_times[8],
                  'daughter_2_cell_id': refined_div_times[9],
                  'original_div_id': refined_div_times_original_ids,
                  'assoc_track_ids': assigned_cell_track_ids, 
                  'cell_track_cell_ids_time': new_cell_tracks_ids_time, 
                  'cell_tracks_xy': new_cell_tracks_xy})
    
    """
    2 ) save tracks out in the same format. 
    """
    
    import pandas as pd 
    
    tracktable_csv = []
    
    for ii in np.arange(len(new_cell_tracks_xy)):
        
        tra = new_cell_tracks_xy[ii]
        tra[np.isnan(tra)] = 0 
        
        tracktable_csv.append(np.hstack([ii, 0, tra[:,0]]))
        tracktable_csv.append(np.hstack([ii, 0, tra[:,1]]))
        
    tracktable_csv = np.vstack(tracktable_csv)
    tracktable = pd.DataFrame(tracktable_csv, index=None,
                              columns = np.hstack(['track_id', 'checked', np.hstack(['Frame_%d' %(jj) for jj in np.arange(new_cell_tracks_xy.shape[1])])]))
    tracktable['track_id'] = tracktable['track_id'].astype(np.int)
    tracktable['checked'] = tracktable['checked'].astype(np.int)
    
    
#    savecsvtrackfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\Annotated_Cell_Divisions\\cell_tracks_w_ids_Holly_polar_distal_corrected_nn_tracking-full-dist30_cell-div-refined-Matt.csv'
    savecsvtrackfile = os.path.join(saveoutfolder, 'cell_tracks_w_ids_'+os.path.split(trackfile)[-1].split('-manual')[0]+'_div-refine.csv')
    tracktable.to_csv(savecsvtrackfile, index=None)
    
    
    """
    3 ) save the refined division file in the same format. 
    """
#    savedivcsvfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\Annotated_Cell_Divisions\\div_polar_Matt_edit_refine.csv'
    savedivcsvfile = os.path.join(saveoutfolder, base_fname+'_'+os.path.split(divfile)[-1]+'_refine.csv')
    
    
    savedivtable = np.hstack([refined_div_times[:,0][:,None],
                              assigned_cell_track_ids[:,0][:,None],
                              refined_div_times[:,1:3], 
                              assigned_cell_track_ids[:,1][:,None],
                              refined_div_times[:,3:5],
                              assigned_cell_track_ids[:,2][:,None],
                              refined_div_times[:,5:7]])
    # save this .
    savedivtable = pd.DataFrame(savedivtable, 
                                index=None,
                                columns = None)
    savedivtable.to_csv(savedivcsvfile, index=None)
    
    
    
    
    
            
                
                    
                
                    
    
    
        
    
    

        
        
        
        
        
        
        
        
        
    
#            plt.figure()
#            plt.imshow(tif[div_time_matt], cmap='gray')
#            plt.plot(mother_xy_matt[0], 
#                     mother_xy_matt[1], 'r.')
#            plt.show()
#            
#            plt.figure()
#            plt.imshow(cell_img[div_time_matt]==mother_spixel)
#            plt.plot(mother_xy_matt[0], 
#                     mother_xy_matt[1], 'r.')
#            plt.show()
        

#        time_diff = np.abs(div_events_time - div_time_matt)
#        xy_diff = np.linalg.norm(div_events_xy - mother_xy_matt[None,:], axis=-1) # this is bad! -> we need to parse in 3D!. otherwise border is very very bad !. 
#        # 3D version ? 
##        unwrap_params
#        xy_diff = np.linalg.norm(unwrap_params[div_events_xy[:,1].astype(np.int), div_events_xy[:,0].astype(np.int)] - unwrap_params[int(mother_xy_matt[1]), int(mother_xy_matt[0])][None,:], axis=-1)
#        
#        # sort by the xy_diff
#        sort_xy_diff = np.argsort(xy_diff)
        
        
    





#    """
#    3. attempt to detect based on 3D area changes -> all tracks which contain / cover a cell division event. 
#    """
#    matfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\cell_tracks_w_ids_Holly_polar_distal_corrected_nn_tracking-full-dist30.mat'
#    trackmat = spio.loadmat(matfile)
#    cell_tracks_ids = trackmat['cell_tracks_ids']
#    cell_tracks_ids_xy = trackmat['cell_tracks_xy']
#    
#    cell_tracks_ids_areas = assoc_area_tracks( cell_tracks_ids, measurements_files, key='areas_3D')# parse and get the associated measurements at this time. 
#    
#    # try to check and detect all tracks with some cell division events -> use smoothing. 
#    cell_track_ids_areas_smooth = smooth_curves(cell_tracks_ids_areas, 
#                                                win_size=15, 
#                                                win_mode='reflect', 
#                                                avg_func=np.nanmean)
#    
#    """
#    try and automate the amount of change that is acceptable. 
#        (think only convolution will work here?)
#        - don't need to be accurate, just need to pull out a series of candidates. 
#    """
#    div_detection_thresh = 0.6
#    
#    div_events = []
#    div_events_time = []
#    div_events_xy = []
#    div_tra_id = []
#    
#    for tra_ii in np.arange(len(cell_track_ids_areas_smooth))[:]:
#        
#        cell_track_ii = cell_tracks_ids_xy[tra_ii]
#        tra = cell_track_ids_areas_smooth[tra_ii]
#        
#        select = tra!=0
#        cell_track_ii = cell_track_ii[select]
#        tra = tra[select]
#        tra_time = np.arange(len(cell_tracks_ids_xy[tra_ii]))[select]
#        
#        tra_diff_delta = np.std(tra)
#        """
#        do a convolution based change detection.
#        """
#        # what about detection by step filter convolving (matched filter)
#        up_step = np.hstack([-1*np.ones(len(tra)//2), 1*np.ones(len(tra)-len(tra)//2)])
###                up_step = np.hstack([-1*np.ones(len(diff_curve_component)), 1*np.ones(len(diff_curve_component)-len(diff_curve_component))])
##        down_step = np.hstack([1*np.ones(len(diff_curve_component)//2), -1*np.ones(len(diff_curve_component)-len(diff_curve_component)//2)])
#        conv_res = np.convolve((tra - np.mean(tra))/(np.std(tra)+1e-8)/len(tra), up_step, mode='same')
#        
#        # peak detection 
#        peaks_conv = find_peaks(conv_res, 
#                                distance=5, 
#                                height=div_detection_thresh,
#                                prominence=0.25)[0]
#        
#        if len(peaks_conv) > 0:
#            time = tra_time[peaks_conv]
#            cell_track_ii_pts = cell_track_ii[peaks_conv]
#            
#            print(time, cell_track_ii_pts)
#            div_events_time.append(time)
#            div_events_xy.append(cell_track_ii_pts)
#            div_tra_id.append(tra_ii)
#        
##        plt.figure()
##        plt.subplot(311)
##        plt.plot(tra)
##        plt.subplot(312)
##        plt.plot((tra - np.mean(tra))/(np.std(tra)+1e-8)/len(tra))
##        plt.subplot(313)
##        plt.plot(conv_res)
##        plt.hlines(div_detection_thresh, 0, len(conv_res), color='r')
##        plt.plot(peaks_conv, conv_res[peaks_conv], 'o')
##        plt.ylim([-1.1,1.1])
##        plt.show()
#        
##        div_event = np.sum(conv_res>div_detection_thresh) > 0
#        if len(peaks_conv) > 0:
#            div_event = 1
#        else:
#            div_event = 0 
#        div_events.append(div_event)
#        
#    div_events = np.hstack(div_events)
#    div_events_time = np.hstack(div_events_time)
#    div_events_xy = np.vstack(div_events_xy)
#    div_tra_id = np.hstack(div_tra_id)
#    
#    print(len(div_events))
#    print(np.sum(div_events))
#    
#    
#    fig, ax = plt.subplots(figsize=(10,10))
#    ax.imshow(tif[0], cmap='gray')
#    ax.plot(div_events_xy[:,0], 
#            div_events_xy[:,1], 'ro')
#    plt.show()
#    
#    
#    fig, ax = plt.subplots(figsize=(10,10))
#    ax.imshow(tif[0], cmap='gray')
#    ax.plot(div_events_xy[:,0], 
#            div_events_xy[:,1], 'ro')
#    ax.plot(mother_xy[:,0],
#             mother_xy[:,1], 'go')
#    plt.show()
#    
#    """
#    attempt to match the parsed divisions of Matt with those of the detected binary. 
#    """
#    dist_match = 50 # if we relax it to 50? -> we can get a match ?
#    time_match = 5
#    
#    matched_events = [] 
#    
#    for ii in np.arange(len(div_time))[:]:
#        
#        # need to jointly minimise time and distance. 
#        div_time_matt = div_time[ii]
#        mother_xy_matt = mother_xy[ii]
#        daughter1_xy_matt = daughter1_xy[ii]
#        daughter2_xy_matt = daughter2_xy[ii]
#        
#        time_diff = np.abs(div_events_time - div_time_matt)
#        xy_diff = np.linalg.norm(div_events_xy - mother_xy_matt[None,:], axis=-1) # this is bad! -> we need to parse in 3D!. otherwise border is very very bad !. 
#        # 3D version ? 
##        unwrap_params
#        xy_diff = np.linalg.norm(unwrap_params[div_events_xy[:,1].astype(np.int), div_events_xy[:,0].astype(np.int)] - unwrap_params[int(mother_xy_matt[1]), int(mother_xy_matt[0])][None,:], axis=-1)
#        
#        # sort by the xy_diff
#        sort_xy_diff = np.argsort(xy_diff)
#        
#        for jj in np.arange(len(sort_xy_diff)):
#            tra_index = sort_xy_diff[jj]
#            dist = xy_diff[tra_index]
#            dist_time = time_diff[tra_index]
#            div_tra_id_select = div_tra_id[tra_index]
#            
#            # is difficult and still a bit off? -> is there other ways to confirm? -> should we do instead local search? 
#            tra_div_xy = div_events_xy[tra_index]
#            tra_div_time = div_events_time[tra_index]
#            
#            
#            if dist <= dist_match and dist_time <= time_match:
##                matched_events.append(np.hstack([ii, div_time_matt, mother_xy_matt, daughter1_xy_matt, daughter2_xy_matt,
##                                       div_tra_id_select, tra_div_time, tra_div_xy]))
#                matched_events.append([ii, div_time_matt, mother_xy_matt, daughter1_xy_matt, daughter2_xy_matt,
#                                       div_tra_id_select, tra_div_time, tra_div_xy])
#                break
#        
#        
#    """
#    To do: test the match, by plotting the inferred time from convolving the smoothed. 
#    """
##    matched_events = np.vstack(matched_events)
#        
#    ##### further post-processsing? to check for duplicate assignments? 
#    
#    ##### iterate through the matched events...... ( we haven't checked for duplicates .....)
#    for jj in np.arange(len(matched_events))[:1]:
#        
#        ii, div_time_matt, mother_xy_matt, daughter1_xy_matt, daughter2_xy_matt, div_tra_id_select, tra_div_time, tra_div_xy = matched_events[jj]
#                                       
#        # do a plot. 
#        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
##        plt.suptitle('Matt-Edit')
##        plt.subplot(121)
#        ax[0,0].set_title('Matt')
#        ax[0,0].imshow(tif[int(div_time_matt)], cmap='gray')
#        ax[0,0].plot(mother_xy_matt[0],mother_xy_matt[1], 'r.')
##        plt.subplot(122)
#        ax[0,1].imshow(tif[int(div_time_matt+1)], cmap='gray')
#        ax[0,1].plot([daughter1_xy_matt[0], daughter2_xy_matt[0]],
#                 [daughter1_xy_matt[1], daughter2_xy_matt[1]], 'g.')
#        
#        
##        fig, ax = plt.subplots(figsize=(10,10))
##        plt.suptitle('Auto-Edit')
##        plt.subplot(121)
#        ax[1,0].set_title('Auto')
#        ax[1,0].imshow(tif[int(tra_div_time)], cmap='gray')
#        ax[1,0].plot(mother_xy_matt[0],mother_xy_matt[1], 'r.')
##        plt.subplot(122)
#        ax[1,1].imshow(tif[int(tra_div_time)+1], cmap='gray')
#        ax[1,1].plot([daughter1_xy_matt[0], daughter2_xy_matt[0]],
#                 [daughter1_xy_matt[1], daughter2_xy_matt[1]], 'g.')
#        
#        plt.show()
#        
#        
#        
#    """
#    The xy doesn't seem to work. -> output a full video of division positions to check. 
#    """
#    saveplotfolder = 'debug_L491_div_parsing_Matt_edit-Matching'
#    mkdir(saveplotfolder)
#    
#    nframes = len(tif)
#    
#    for frame_no in np.arange(nframes-1):
#                
#        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
##        plt.suptitle('Matt-Edit')
##        plt.subplot(121)
#        ax[0,0].set_title('Matt')
#        ax[0,0].imshow(tif[int(frame_no)], cmap='gray')
#        ax[0,1].imshow(tif[int(frame_no+1)], cmap='gray')
#        
##        fig, ax = plt.subplots(figsize=(10,10))
##        plt.suptitle('Auto-Edit')
##        plt.subplot(121)
#        ax[1,0].set_title('Auto')
#        ax[1,0].imshow(tif[int(frame_no)], cmap='gray')
#        ax[1,1].imshow(tif[int(frame_no)+1], cmap='gray')
#        
#        for jj in np.arange(len(matched_events))[:]:
#        
#            ii, div_time_matt, mother_xy_matt, daughter1_xy_matt, daughter2_xy_matt, div_tra_id_select, tra_div_time, tra_div_xy = matched_events[jj]
#            
#            if div_time_matt == frame_no:
#        
#                ax[0,0].plot(mother_xy_matt[0],mother_xy_matt[1], 'r.')
#                ax[0,0].plot(tra_div_xy[0],tra_div_xy[1], 'y.')
#                ax[0,1].plot([daughter1_xy_matt[0], daughter2_xy_matt[0]],
#                         [daughter1_xy_matt[1], daughter2_xy_matt[1]], 'g.')
#                ax[1,0].plot(mother_xy_matt[0],mother_xy_matt[1], 'r.')
#                ax[1,0].plot(tra_div_xy[0],tra_div_xy[1], 'y.')
#                ax[1,1].plot([daughter1_xy_matt[0], daughter2_xy_matt[0]],
#                         [daughter1_xy_matt[1], daughter2_xy_matt[1]], 'g.')
#                
#                ax[1,0].imshow(tif[int(tra_div_time)], cmap='gray')
#                ax[1,1].imshow(tif[int(tra_div_time)+1], cmap='gray')
#                ax[0,0].imshow(tif[int(div_time_matt)], cmap='gray')
#                ax[0,1].imshow(tif[int(div_time_matt+1)], cmap='gray')
#            
#        plt.show()
#        
#        fig.savefig(os.path.join(saveplotfolder, 'Frame-%s.png' %(str(frame_no+1).zfill(3))), bbox_inches='tight')
#        plt.show()
#    
    
        
                                       
    
    
    
    
        
        
        
#    div_time, (mother_tra_ids, mother_xy), (daughter1_tra_ids, daughter1_xy), (daughter2_tra_ids, daughter2_xy)
    
        
        
         
#        # put in the filtered. 
#        in_signal = np.diff(diff_curve_component_)
##                in_signal =  np.diff(np.cumsum(in_signal)/(np.arange(len(in_signal)) + 1))
#        conv_res_up = np.convolve((in_signal - np.mean(in_signal))/(np.std(in_signal)+1e-8)/len(in_signal), down_step, mode='same')
#        conv_res_down = np.convolve((in_signal - np.mean(in_signal))/(np.std(in_signal)+1e-8)/len(in_signal), up_step, mode='same')
#        
##                peaks_conv = find_peaks(np.abs(conv_res_up), distance=5, height=0.15, prominence=0.05)[0]
#        peaks_conv = find_peaks(np.abs(conv_res_up), distance=5, height=0.15)[0]
#        peaks_conv = np.hstack([p for p in peaks_conv if p>2 and p < len(diff_curve_component_)-2])
#                

##        change_tra_thresh = np.mean(tra_diff) - 3*tra_diff_delta
##        div_event = np.sum(tra_diff < tra_diff_delta) > 0
##        div_events.append(div_event)
#        plt.figure()
#        plt.plot(tra)
#        plt.hlines(np.mean(tra) - 3*tra_diff_delta, 0, len(tra))
#        plt.show()
#        print(len(tra))
#    
#    div_events = np.hstack(div_events)
    

## =============================================================================
## =============================================================================
## =============================================================================
## # #     Checking. 
## =============================================================================
## =============================================================================
## =============================================================================
#    """
#    The xy doesn't seem to work. -> output a full video of division positions to check. 
#    """
#    saveplotfolder = 'debug_L491_div_parsing_Matt_edit'
#    mkdir(saveplotfolder)
#    
#    
#    nframes = len(tif)
#    
#    for frame_no in np.arange(nframes-1):
#        
#        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
#        ax[0].imshow(tif[frame_no], cmap='gray')
#        ax[1].imshow(tif[frame_no+1], cmap='gray')
#        
##    fig, ax = plt.subplots(figsize=(10,10))
##    ax.imshow(tif[0], cmap='gray')
#        for ii in np.arange(len(div_time))[:]:
#            div_ii_time = div_time[ii]
#            
#            if div_ii_time == frame_no:
#    #            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
#    #            ax[0].imshow(tif[div_ii_time])
#                ax[0].plot(mother_xy[ii,0], 
#                        mother_xy[ii,1],'r.')
#                ax[0].plot([daughter1_xy[ii,0], daughter2_xy[ii,0]], 
#                        [daughter1_xy[ii,1], daughter2_xy[ii,1]], 'g.-')
#    #            ax[1].imshow(tif[div_ii_time+1])
#                ax[1].plot(mother_xy[ii,0], 
#                        mother_xy[ii,1],'r.')
#                ax[1].plot([daughter1_xy[ii,0], daughter2_xy[ii,0]], 
#                        [daughter1_xy[ii,1], daughter2_xy[ii,1]], 'g.-')
#                
#                
#        fig.savefig(os.path.join(saveplotfolder, 'Frame-%s.png' %(str(frame_no+1).zfill(3))), bbox_inches='tight')
#        plt.show()
#    
    
#    """
#    3. Match the divs to changes in geometry along the tracks. This loads the original uneditted file! 
#        -> this is actually a bit of a problem ..... 
#    """
#    
#    matfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\cell_tracks_w_ids_Holly_polar_distal_corrected_nn_tracking-full-dist30.mat'
#    trackmat = spio.loadmat(matfile)
#    cell_tracks_ids = trackmat['cell_tracks_ids']
#    cell_tracks_ids_xy = trackmat['cell_tracks_xy']
#    
#    cell_tracks_ids_areas = assoc_area_tracks( cell_tracks_ids, measurements_files, key='areas_3D')# parse and get the associated measurements at this time. 
#    
#    """
#    apply restriction to tracks
#    """
#    cell_tracks = cell_tracks[:len(cell_tracks_ids)]
#    
##    for xxx in np.arange(10):
##        plt.figure(figsize=(10,10))
##        plt.plot(cell_tracks[xxx][...,1], 
##                 cell_tracks[xxx][...,0], color='r')
##        plt.plot(cell_tracks_ids_xy[xxx][...,1], 
##                 cell_tracks_ids_xy[xxx][...,0], color='g')
##        plt.show()
#    
#    
#    # load the original segmented image from which statistics taken from. 
#    cellfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Annotated_Cell_Tracking\\cells_L491_final_version_polar_2nd_edit_merge.tif'
#    cell_img = skio.imread(cellfile)
#    
#    """
#    4. Question, can we use the actual division to redit the tracking. 
#    """
#    
#    # step 1: associate the mother and daughters to superpixel ids. 
#    spixel_id_mothers = cell_img[div_time, mother_xy[:,1], mother_xy[:,0]]
#    spixel_id_daughters_1 = cell_img[div_time+1, daughter1_xy[:,1], daughter1_xy[:,0]]
#    spixel_id_daughters_2 = cell_img[div_time+1, daughter2_xy[:,1], daughter2_xy[:,0]]
#    
#    # step 2: match to cell track ids.... 
#    
#    time_tol = 5
#    tra_id_div = []
#    
#    for ss in np.arange(len(spixel_id_mothers))[:20]:
#        
#        split_time = div_time[ss]
#        spixel_id_mother_time = spixel_id_mothers[ss]
#        spixel_id_daughters_1_time = spixel_id_daughters_1[ss]
#        spixel_id_daughters_2_time = spixel_id_daughters_2[ss]
#        
#        
#        assoc_track_id_mother = np.arange(len(cell_tracks_ids))[cell_tracks_ids[:,split_time] == spixel_id_mother_time][0]
#        
#        plt.figure()
#        plt.plot(cell_tracks_ids_areas[assoc_track_id_mother])
#        plt.vlines(split_time, cell_tracks_ids_areas[assoc_track_id_mother].min(), cell_tracks_ids_areas[assoc_track_id_mother].max())
#        plt.show()
#    
    
    
    
#    fig, ax = plt.subplots(figsize=(10,10))
#    ax.imshow(tif[0])
#    ax.plot(mother_xy[:,0], 
#            mother_xy[:,1], 'ro')
#    plt.show()
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    