# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:58:50 2021

@author: fyz11
"""

def preprocess_feats(X, std_tformer, pow_tformer):

    # apply learnt transformations to input features X.
    X_tfm = std_tformer.transform(pow_tformer.transform(X))

    return X_tfm


def transform_feats(X, apply_tform_cols, noapplytform_cols, 
                    std_tformer=None, pow_tformer=None, 
                    std_tformer_2=None, pow_tformer_2=None, 
                    X_max=None, return_model=True):

    from sklearn.preprocessing import StandardScaler, PowerTransformer
    
    if std_tformer is None:
        return_model=True
        X_ = X[:, apply_tform_cols].copy()
        std_tformer = StandardScaler()
        pow_tformer = PowerTransformer()
        std_tformer.fit(X_) # apply the power transformation first!. 
        pow_tformer.fit(std_tformer.transform(X_))
        # X_max = np.nanmax(np.abs(X[:,noapplytform_cols]), axis=0)
        
        # can we just use this to spread the distribution!. 
        std_tformer_2 = StandardScaler(with_mean=False) # we only scale the std. 
        pow_tformer_2 = PowerTransformer()
        std_tformer_2.fit(X[:, noapplytform_cols])
        pow_tformer_2.fit(std_tformer_2.transform(X[:, noapplytform_cols]))
        
        # # X_max = 3 * np.nanmax((X[:,noapplytform_cols]), axis=0)
        # X_max = np.nanstd((pow_tformer_2.transform(X[:,noapplytform_cols])), axis=0)
        
    # # apply learnt transformations to input features X. also apply max scaling to the transformed features. 
    # X_tfm = std_tformer.transform(pow_tformer.transform(X[:, apply_tform_cols]))
    # # X_tfm_2 = X[:, noapplytform_cols] / (X_max[None,:])
    # # X_tfm_2 = pow_tformer_2.transform(X[:, noapplytform_cols]) / (X_max[None,:])
    # X_tfm_2 = std_tformer_2.transform(pow_tformer_2.transform(X[:, noapplytform_cols]))
    
    X_tfm = pow_tformer.transform(std_tformer.transform(X[:, apply_tform_cols]))
    # X_tfm_2 = X[:, noapplytform_cols] / (X_max[None,:])
    # X_tfm_2 = pow_tformer_2.transform(X[:, noapplytform_cols]) / (X_max[None,:])
    X_tfm_2 = pow_tformer_2.transform(std_tformer_2.transform(X[:, noapplytform_cols]))

    
    X_out = np.zeros(X.shape)
    X_out[:,apply_tform_cols] = X_tfm.copy()
    X_out[:,noapplytform_cols] = X_tfm_2.copy()

    if return_model:
        # print('hello')
        return X_out, (std_tformer, pow_tformer, std_tformer_2, pow_tformer_2, X_max)
    else:
        return X_out



# construct a function to lookup single cell for tracks. 
def lookup_track_single_cell_data(track_table, cell_table):

    n_tracks = len(track_table)
    
    cell_track_id_array = []
    cell_track_ids = []
    
    for ii in np.arange(n_tracks):
        track_ii = track_table.iloc[ii]
        track_ii_id = track_ii['Track_ID']
        track_ii_tp = track_ii.loc['TP_1':].values.astype(np.int)
        
        # track_ii_tp_cell_ids = np.ones(len(track_ii_tp)) * -1
        # track_ii_tp_cell_ids[track_ii_tp>=0] = cell_table.iloc[track_ii_tp[track_ii_tp>=0]]['Cell_ID'] # only -1 entries are not valid. 
        track_ii_tp_cell_data = np.zeros((len(track_ii_tp), cell_table.shape[1]))
        track_ii_tp_cell_data[:] = np.nan
        track_ii_tp_cell_data = track_ii_tp_cell_data.astype(np.object)
        track_ii_tp_cell_data[track_ii_tp>=0] = cell_table.iloc[track_ii_tp[track_ii_tp>=0]].values.copy()
        
        cell_track_ids.append(track_ii_id)
        cell_track_id_array.append(track_ii_tp_cell_data)
        
    return np.hstack(cell_track_ids).astype(np.int), np.array(cell_track_id_array)
        

def draw_pie(dist, 
             xpos, 
             ypos, 
             size,
             colors=None, 
             ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
    pie = [0] + cumsum.tolist()

    cnt = 0 
    for r1, r2 in zip(pie[:-1], pie[1:]):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])

        if colors is not None:
            ax.scatter([xpos], [ypos], marker=xy, s=size, color=colors[cnt])
            cnt += 1
        else:
            ax.scatter([xpos], [ypos], marker=xy, s=size)

    return ax

def draw_pie_xy(dist, 
                 xpos, 
                 ypos, 
                 size,
                 startangle=None,
                 colors=None, 
                 ax=None):
    
    ax.pie(dist, colors=colors, startangle=startangle, 
           radius=size, 
           center=(xpos, ypos))
    
    return []
    
    
def fit_gaussian_kde_pts(xy_train, grid=None, thresh=None, thresh_sigma=None, norm=False):
    
    from scipy.stats import gaussian_kde
    from skimage.measure import label
    from skimage.measure import find_contours 
    
    # fit the gaussian_kde and evaluate at the grid of points. 
    grid_pts = grid.reshape(-1,2)
    z = gaussian_kde(xy_train.T)(grid_pts.T) 
    z_grid = z.reshape(grid.shape[:-1])
    
    if norm:
        z_grid = z_grid / float(np.sum(z_grid)) # normalize density 
    
    if thresh is None:
        thresh = np.mean(z_grid) + thresh_sigma*np.std(z_grid)
    
    # plt.figure()
    # plt.imshow(z_grid, cmap='coolwarm')
    # plt.show()
    
    binary = z_grid>= thresh
    cont = find_contours(z_grid, thresh)
    labelled = label(binary, connectivity=1)
    
    uniq_regions = np.setdiff1d(np.unique(labelled),0)
    
    if len(uniq_regions) > 0: 
        peak_I = np.hstack([np.mean(z_grid[labelled==reg]) for reg in uniq_regions])
        # use a weighted threshold
        # peak_xy = np.vstack([np.mean(grid[labelled==reg], axis=0) for reg in uniq_regions])
        peak_xy = np.vstack([np.sum(grid[labelled==reg]*z_grid[labelled==reg][:,None], axis=0) / (np.sum(z_grid[labelled==reg])) for reg in uniq_regions])
        
        
        cont_xy = [grid[cnt[...,0].astype(np.int), cnt[...,1].astype(np.int)] for cnt in cont]
        print(peak_I, peak_xy)
        return z_grid, np.hstack([peak_xy, peak_I[:,None]]), cont_xy
    else:
        return z_grid, [], []
    
    # print(np.unique(labelled))
    
    # plt.figure()
    # plt.imshow(z_grid>thresh, cmap='coolwarm')
    # plt.show()
    
    # return z_grid
    
    
def batch_map_stats_back_to_image(id_array, metadict_id_array, metadict_embryo, return_global=False):
    
    import skimage.io as skio 
    import numpy as np 
    import pandas as pd 
    
    """
    create new arrays for each  
    """
    # unroll the wrapped params. 
    embryo_all = metadict_id_array['embryo_all']
    TP_all = metadict_id_array['TP_all']
    cell_id_all = metadict_id_array['cell_id_all']
    row_id_all = metadict_id_array['row_id_all']
    track_id_all = metadict_id_array['track_id_all']
    
    # determine the embryos we need. 
    uniq_embryos = np.unique(embryo_all)
    # embryo_key = {uniq_embryos[ii]:ii for ii in np.arange(len(uniq_embryos))}
    
    # Preload the necessary lookups. 
    uniq_embryos_data = { emb:{'img': skio.imread(metadict_embryo[emb]['polarfile']), 
                               'seg': skio.imread(metadict_embryo[emb]['segfile']),
                               'singlecell': pd.read_csv(metadict_embryo[emb]['singlecellstatfile']), 
                               'tralookup': pd.read_csv(metadict_embryo[emb]['tracklookupfile'])} for emb in uniq_embryos} # load and create a new dict to store everything. 
    # convert this to a dict!. 
    # uniq_embryos_data = {uniq_embryos[iii]:uniq_embryos_data[iii]}
    
    # can we parallelise this further? 
    N = len(id_array)
    
    id_array_track_data = [] # collect in a list is fine. 
     
    for ii in np.arange(N):
        
        test_cell_entry = id_array[ii]
        emb_entry = embryo_all[test_cell_entry]
        # ref_key = embryo_key[emb_entry] # tells us which reference to reference. 
        # tp_entry = TP_all[test_cell_entry]
        # cell_id_entry = cell_id_all[test_cell_entry]
        row_id_entry = row_id_all[test_cell_entry]
        track_id_entry = track_id_all[test_cell_entry]
        
        # # pull down the information of the whole selected_track. so we can use it as we want. 
        # seg_entry =  uniq_embryos_data[ref_key]['seg'] skio.imread(metadata_entry['segfile'])#[int(tp_entry)])# == cell_id_entry
        # binary_entry = seg_entry[int(tp_entry)] == cell_id_entry
        # img_entry = skio.imread(metadata_entry['polarfile'])[int(tp_entry)]
        tra_lookup = uniq_embryos_data[emb_entry]['tralookup'] # pd.read_csv(metadata_entry['tracklookupfile'])
        singlestattab = uniq_embryos_data[emb_entry]['singlecell'] # pd.read_csv(metadata_entry['singlecellstatfile'])
        
        # pull down the information of the whole associated track 
        tra_select = tra_lookup['Track_ID'].values == track_id_entry
        tra_row = tra_lookup.loc[tra_select]
        tra_cells = tra_row.iloc[:,1:].values[tra_row.iloc[:,1:].values > 0] # get the associate track cells. 
        # this can be used to straight up pool the xy positions and times. 
        tra_cell_data = singlestattab.iloc[tra_cells].copy()
        tra_cell_data.index = np.arange(len(tra_cell_data)) # rewrite this so we can manipulate as separate table. 
        
        # immediately useful just to get the immediate data on this cell.? 
        id_array_track_data.append([singlestattab.iloc[row_id_entry], tra_cell_data]) # pull down its own data + the whole track data. 

               
    # return a list of dicts ... containing the relevant information. 
    if return_global:
        return uniq_embryos_data, id_array_track_data # returns whole embryo level data + that of the individual queried ID.
    else:
        return id_array_track_data


    

if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    import scipy.io as spio 
    import os 
    import glob
    import pandas as pd 
    import statsmodels.api as sm
    import Utility_Functions.file_io as fio
    from sklearn.metrics.pairwise import pairwise_distances
    
    
    
    
# =============================================================================
#     create the mega savefolder to save all the plots for editting. 
# =============================================================================

    # we should also save the models used. i.e. the kmeans clusters etc. 
    # saveplotfolder = '2021-05-19_PCA_analysis'
    # saveplotfolder = '../../../Data_Analysis/LifeAct/PseudoAnalysis/Data/Cell_Statistics_3/2022-01-19_PCA_analysis'
    # saveplotfolder = '../../../Data_Analysis/LifeAct/PseudoAnalysis/Data/Cell_Statistics_3/2022-02-09_UMAP_analysis_ALL-Cells'
    saveplotfolder = '../../../Data_Analysis/LifeAct/PseudoAnalysis/Data/Cell_Statistics_3/2022-11-10_UMAP_analysis_ALL-Cells_LifeAct'
    fio.mkdir(saveplotfolder)
    
    
    voxel_size = 0.363
    
# =============================================================================
#     Load all the exported metric files. 
# =============================================================================
    
    metricfiles = glob.glob(os.path.join('../../../Data_Analysis/LifeAct/PseudoAnalysis/Data/Cell_Statistics_3', '*.csv'))
    # metricfiles = glob.glob(os.path.join('../../../Data_Analysis/LifeAct/PseudoAnalysis/Data/Cell_Statistics_2 - Backup', '*.csv'))
    metricfiles = np.hstack([ff for ff in metricfiles if 'single_cell_statistics_table_export-ecc-realxyz-from-demons_' in os.path.split(ff)[-1] and 'vcorr' not in os.path.split(ff)[-1]])
    metrictrackfiles = np.hstack([ff.replace('single_cell_statistics_table_export-ecc-realxyz-from-demons_', 'single_cell_track_stats_lookup_table_') for ff in metricfiles])
    
    metrics_tables = [] # over all cells. 
    metrics_tables_embryo = [] 
    cell_lookup_tables = [] 
    metrics_cell_tracks_tables = [] # after parsing it in the form of tracks. 
    cell_track_data_tables = []
    
    embryo_TP0_frame = {} # initialise a dict. 
    
    
    for ii in np.arange(len(metricfiles))[:]:
        
        embryo = os.path.split(metricfiles[ii])[-1].split('single_cell_statistics_table_export-ecc-realxyz-from-demons_')[1].split('.csv')[0]
        metric_table = pd.read_csv(metricfiles[ii]) # embryo is already in here great .... 
        # metrics_tables.append(metric_table) # but we don't want diff
        
        cell_lookup_table = pd.read_csv(metrictrackfiles[ii]) # get the cell_lookup table. 
        cell_lookup_tables.append(cell_lookup_table)
        
        metric_table_copy = metric_table.copy()
        # replace with the number of cell neighbors.
        n_neighbors = np.zeros(len(metric_table_copy))
        
        for dd_ii, dd in enumerate(metric_table_copy['Cell_Neighbours_Track_ID'].values):
            try:
                if ':' in dd:
                    n_neighbors[dd_ii] = len(dd.split(':'))
                else:
                    n_neighbors[dd_ii] = 1
            except:
                if np.isnan(dd): # empty
                    n_neighbors[dd_ii] = 0
                # else:
                # print(dd, np.isnan(dd))
        # metric_table_copy['Cell_Neighbours_Track_ID'] = np.hstack([ len(dd.split(':')) for dd in metric_table_copy['Cell_Neighbours_Track_ID'].values])
        metric_table_copy['Cell_Neighbours_Track_ID'] = n_neighbors.copy()
        
        cell_track_data_table = lookup_track_single_cell_data(cell_lookup_table, metric_table_copy)[1]
        cell_track_data_tables.append(cell_track_data_table)
        
        # some of these we get infs? 
        metric_table_copy['mean_curvature_VE'] = metric_table_copy['mean_curvature_VE'].astype(np.float)
        metric_table_copy['mean_curvature_Epi'] = metric_table_copy['mean_curvature_Epi'].astype(np.float)
        
        # TP0_frame = metric_table_copy['Frame'].values[metric_table_copy['Stage'].values=='Migration to Boundary']
        # TP0_frame = TP0_frame.min() # this would be the starting time 
        # metric_table_copy['Frame'] = metric_table_copy['Frame'].values - TP0_frame # itself will be 0 else negative.
        TP0_frame = metric_table_copy['Frame'].values[metric_table_copy['Stage'].values=='Migration to Boundary']
        TP0_frame = TP0_frame.min() # this would be the starting time 
        embryo_TP0_frame[embryo] = TP0_frame
        # add a new column which is 
        metric_table_copy['Time'] = metric_table_copy['Frame'].values - TP0_frame # itself will be 0 else negative.
        
        metrics_tables.append(metric_table_copy)
        feat_names = metric_table.columns
        metrics_tables_embryo.append([embryo] * len(metric_table_copy))
        
        
        print(embryo, len(np.unique(metric_table['Cell_Track_ID'])))
        print('----')
        
    metrics_tables = pd.concat(metrics_tables, ignore_index=True) # save this out. 
    metrics_tables_embryo = np.hstack(metrics_tables_embryo)
    # metrics_tables['Embryo_Name'] = metrics_tables_embryo # already have embryo saved... 
    
    print('Original full table size.')
    print(metrics_tables.shape)
    print(metrics_tables_embryo.shape)
    
#     # cell_track_data_tables = np.concatenate(cell_track_data_tables, axis=0)
# #     # savecombmetricsfile = 'combined_cell_feats_Shankar_no-nan.csv'
# #     # metrics_tables.to_csv(savecombmetricsfile, index=None)


# =============================================================================
#   Selectively build the entries of interest. 
# =============================================================================
    
    # metadata of interest. 
    embryo = metrics_tables['embryo'].values
    frame = metrics_tables['Time'].values
    TP = metrics_tables['Frame'].values
    stage = metrics_tables['Stage'].values
    static_quad_id = metrics_tables['Static_Epi_Contour_Quad_ID'].values
    dynamic_quad_id = metrics_tables['Dynamic_Track_Quad_ID'].values
    # dist_ratio = metrics_tables['']
    row_id = metrics_tables['row_id'].values # we might need this specifically for track reconstruction !. 
    cell_id = metrics_tables['Cell_ID'].values
    track_id = metrics_tables['Cell_Track_ID'].values
    # dist_ratio = metrics_tables['']

    # metrics of interest. 
    area = metrics_tables['Area'].values
    perim = metrics_tables['Perimeter_Proj'].values
    shape_index = metrics_tables['Shape_Index_Proj'].values
    eccentricity = metrics_tables['Eccentricity'].values
    thickness = metrics_tables['Thickness'].values
    n_neighbors = metrics_tables['Cell_Neighbours_Track_ID'].values
    ve_speed = metrics_tables['AVE_direction_VE_speed_3D_ref'].values
    epi_speed = metrics_tables['AVE_direction_Epi_speed_3D_ref'].values
    cum_ave = metrics_tables['AVE_cum_dist_coord_3D_ref'].values # just for PCA!. 
    epi_contour_dist_ratio = metrics_tables['cell_epi_contour_dist_ratio'].values
    
    normal_diff = metrics_tables['demons_Norm_dir_VE'].values - metrics_tables['demons_Norm_dir_Epi'].values
    # normal_ve = metrics_tables['demons_Norm_dir_VE'].values
    # normal_epi = metrics_tables['demons_Norm_dir_Epi'].values
    curvature_VE = metrics_tables['mean_curvature_VE'].values
    curvature_Epi = metrics_tables['mean_curvature_Epi'].values
    gauss_curvature_VE = metrics_tables['gauss_curvature_VE'].values
    gauss_curvature_Epi = metrics_tables['gauss_curvature_Epi'].values
    
    # augment with the actin levels # ..... 
    apical_actin_VE = metrics_tables['norm_apical_actin_intensity']
    perimeter_actin_VE = metrics_tables['norm_perimeter_actin_intensity']
    
    
# =============================================================================
#     Double check the total number of unique cells!. 
# =============================================================================
    unique_cell_IDs_full = np.hstack([embryo[kkkk]+'_'+str(int(track_id[kkkk])).zfill(5) for kkkk in np.arange(len(embryo))])
    
    
    print('num unique cell_IDs')
    print(len(np.unique(unique_cell_IDs_full)))
    print('------')
    
    
    """
    with focus now on VE - we might want to withdraw all Epi relevant params. 
    """
    # derive the rows to drop!. -> both infs and nans! -> if any row has this then it is not complete!. 
    metrics_data = np.vstack([area,
                              perim,
                              shape_index,
                              eccentricity, 
                               thickness,
                              n_neighbors, 
                              ve_speed,
                                # epi_speed,
                              cum_ave,
                                # epi_contour_dist_ratio, 
                               # normal_diff,
                              # normal_ve,
                              # normal_epi,
                              curvature_VE,  # putting in curvature complicates. 
                                curvature_Epi,
                               gauss_curvature_VE,
                                gauss_curvature_Epi, #]).T
                                apical_actin_VE, 
                                perimeter_actin_VE
                                ]).T


    """
    Ok so this is the main part where things got deleted...... 
    """
    # this last is more restrictive... hm.... 
    invalid_rows = np.hstack([np.sum(np.isnan(metrics_data[row])) for row in np.arange(len(metrics_data))]) + np.hstack([np.sum(np.isinf(metrics_data[row])) for row in np.arange(len(metrics_data))])
    invalid_meta = np.logical_not(np.logical_and(~np.isnan(dynamic_quad_id), ~np.isnan(static_quad_id)))
    
    keep_rows = np.logical_and(invalid_rows==0, invalid_meta==0)   ### this is the subset that we keep! to annotate!.

    keep_rows_original_select = np.arange(len(invalid_rows))[keep_rows>0]  # keep a record of this index ..... !!!!!
    
    # col_name_select = ['Area', 
    #                     'Perimeter', 
    #                     'Shape_Index', 
    #                     'Eccentricity', 
    #                     'Thickness',
    #                     'Num_Neighbors',
    #                     'A_P_speed_VE', 
    #                     # 'A_P_speed_Epi', 
    #                     'cum_A_P_VE_speed', 
    #                     # 'epi_contour_dist', # can't have this. 
    #                     # 'Normal_VE_minus_Epi', # doesn seem to do anything? 
    #                     # 'Normal VE', 
    #                     # 'Normal Epi', 
    #                     'Mean_Curvature_VE', 
    #                     # 'Mean_Curvature_Epi',
    #                     'Gauss_Curvature_VE', 
    #                     # 'Gauss_Curvature_Epi', #]
    #                     'norm_apical_actin', 
    #                     'norm_perimeter_actin']
    col_name_select = ['Area', 
                        'Perimeter', 
                        'Shape_Index', 
                        'Eccentricity', 
                        'Thickness',
                        'Num_Neighbors',
                        'A_P_speed_VE', 
                        # 'A_P_speed_Epi', 
                        'cum_A_P_VE_speed', 
                        # 'epi_contour_dist', # can't have this. 
                        # 'Normal_VE_minus_Epi', # doesn seem to do anything? 
                        # 'Normal VE', 
                        # 'Normal Epi', 
                        'Mean_Curvature_VE', 
                        'Mean_Curvature_Epi',
                        'Gauss_Curvature_VE', 
                        'Gauss_Curvature_Epi', #]
                        'norm_apical_actin', 
                        'norm_perimeter_actin']
    
    # save a version of the full for reference !
    embryo_full_raw = embryo.copy()
    frame_full_raw = frame.copy()
    TP_full_row = TP.copy()
    stage_full_row = stage.copy()
    static_quad_id_full_row = static_quad_id.copy() #[keep_rows]
    dynamic_quad_id_full_row = dynamic_quad_id.copy() #[keep_rows]
    row_id_full_row = row_id.copy() #[keep_rows] # we might need this specifically for track reconstruction !. 
    cell_id_full_row = cell_id.copy() #[keep_rows] # .values
    track_id_full_row = track_id.copy() #[keep_rows]
    epi_contour_dist_ratio_full_row = epi_contour_dist_ratio.copy() #[keep_rows]
    
    metrics_data_full_raw = metrics_data.copy() #[keep_rows].copy()
    
    
    
    print('full table size.')
    print(metrics_data_full_raw.shape)
    # print(embryo.shape)
    """
    We fix the metrics... by inputting nans!. 
    """

    # finalised. (used to train)
    embryo = embryo[keep_rows]
    frame = frame[keep_rows]
    TP = TP[keep_rows]
    stage = stage[keep_rows]
    static_quad_id = static_quad_id[keep_rows]
    dynamic_quad_id = dynamic_quad_id[keep_rows]
    row_id = row_id[keep_rows] # we might need this specifically for track reconstruction !. 
    cell_id = cell_id[keep_rows] # .values
    track_id = track_id[keep_rows]
    epi_contour_dist_ratio = epi_contour_dist_ratio[keep_rows]
    
    metrics_data = metrics_data[keep_rows].copy()
    
    
    print('filtered table size before STD filtering')
    print(metrics_data.shape)
    print(embryo.shape)
    
    corr_matrix = pairwise_distances(metrics_data.T, metric='correlation', force_all_finite=False) # v. strange. 
  
    plt.figure(figsize=(10,10))
    plt.imshow(1.-corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
    plt.yticks(np.arange(len(corr_matrix)),
                col_name_select[:])
    plt.xticks(np.arange(len(corr_matrix)),
                col_name_select[:], rotation=45, horizontalalignment='right')
    # plt.yticklabels(col_name_select[2:])
    plt.show()
  


    """
    Computing the features!. 
    """
    
#     # we should do a mapper..... 
    uniq_embryo = np.unique(embryo)
    
    # # finalised. 
    # embryo = embryo[keep_rows]
    # frame = frame[keep_rows]
    # stage = stage[keep_rows]
    # static_quad_id = static_quad_id[keep_rows]
    # dynamic_quad_id = dynamic_quad_id[keep_rows]
    
    X_feats_all = []
    X_feats_all_raw = []
    embryo_all = []
    frame_all = []
    stage_all = []
    static_quad_id_all = []
    dynamic_quad_id_all = []
    epi_contour_dist_ratio_all = []
    
    
    row_id_all = []
    cell_id_all = [] 
    track_id_all = []
    
    TP_all = []



    X_feats_all_full = []
    X_feats_all_raw_full = []
    embryo_all_full = []
    frame_all_full = []
    stage_all_full = []
    static_quad_id_all_full = []
    dynamic_quad_id_all_full = []
    epi_contour_dist_ratio_all_full = []
    
    
    row_id_all_full = []
    cell_id_all_full = [] 
    track_id_all_full = []
    
    TP_all_full = []
    
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    from sklearn.impute import SimpleImputer
    
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') # note the missing imputation. 
    
    
    # iterate over each embryo and do z-score normalisation. 
    for emb in uniq_embryo[:]:
        select_row = embryo == emb
        # # for some reason this does not centralizae the features? 
        # X_feats, tforms = transform_feats(metrics_data.astype(np.float)[select_row], 
        #                                     # apply_tform_cols=[0,1,2,3,4,5],
        #                                     # noapplytform_cols=[6,7,8,9,10,11,12,13,14,15], 
        #                                     # noapplytform_cols=[6,7,8,9,10,11,12,13],
        #                                     apply_tform_cols=[0,1,2,3,4,5],
        #                                     noapplytform_cols=[6,7,8,9,10,11], #,12,13], # where there were natural signage.... then we put those variables here. 
        #                                     std_tformer=None, 
        #                                     pow_tformer=None, 
        #                                     X_max=None)
        
        # for some reason this does not centralizae the features? 
        X_feats, tforms = transform_feats(metrics_data.astype(np.float)[select_row], 
                                            # apply_tform_cols=[0,1,2,3,4,5],
                                            # noapplytform_cols=[6,7,8,9,10,11,12,13,14,15], 
                                            # noapplytform_cols=[6,7,8,9,10,11,12,13],
                                            apply_tform_cols=[0,1,2,3,4,5],
                                            noapplytform_cols=[6,7,8,9,10,11,12,13], # where there were natural signage.... then we put those variables here. 
                                            std_tformer=None, 
                                            pow_tformer=None, 
                                            X_max=None)
        
        
        
        """
        Compute on the full raw
        """
        select_row_full = embryo_full_raw == emb
        
        input_X_full_raw = metrics_data_full_raw.astype(np.float)[select_row_full].copy()
        input_X_full_raw[np.isinf(input_X_full_raw)] = np.nan
        # iterate over the columns and impute missing values first! by the mean.  
        input_X_full_raw_impute = imp_mean.fit_transform(input_X_full_raw)
        
        
        X_feats_full, _ = transform_feats(input_X_full_raw_impute, 
                                               # apply_tform_cols=[0,1,2,3,4,5],
                                               # noapplytform_cols=[6,7,8,9,10,11,12,13,14,15], 
                                               # noapplytform_cols=[6,7,8,9,10,11,12,13],
                                               apply_tform_cols=[0,1,2,3,4,5],
                                               noapplytform_cols=[6,7,8,9,10,11,12,13], # where there were natural signage.... then we put those variables here. 
                                               std_tformer=tforms[0],  # supplying this will just apply the training!. 
                                               pow_tformer=tforms[1], 
                                               std_tformer_2=tforms[2],  # supplying this will just apply the training!. 
                                               pow_tformer_2=tforms[3], 
                                               X_max=tforms[-1])

        # embryo_full_raw = embryo.copy()
        # frame_full_raw = frame.copy()
        # TP_full_row = TP.copy()
        # stage_full_row = stage.copy()
        # static_quad_id_full_row = static_quad_id.copy() #[keep_rows]
        # dynamic_quad_id_full_row = dynamic_quad_id.copy() #[keep_rows]
        # row_id_full_row = row_id.copy() #[keep_rows] # we might need this specifically for track reconstruction !. 
        # cell_id_full_row = cell_id.copy() #[keep_rows] # .values
        # track_id_full_row = track_id.copy() #[keep_rows]
        # epi_contour_dist_ratio_full_row = epi_contour_dist_ratio.copy() #[keep_rows]
        
        # metrics_data_full_raw = metrics_data.copy() #[keep_rows].copy()
        
        
        # """
        # will it work better if we just processed all the features the same!. 
        # """
        # X_ = (metrics_data.astype(np.float)[select_row]).copy()
        # std_tformer = StandardScaler()
        # pow_tformer = PowerTransformer()
        # std_tformer.fit(X_) # apply the power transformation first!. 
        # pow_tformer.fit(std_tformer.transform(X_))
       
        # X_feats = pow_tformer.transform(std_tformer.transform(X_))
        
        # col_name_select = ['Area', 0
        #                 'Perimeter', 1 
        #                 'Shape_Index', 2 
        #                 'Eccentricity',  3
        #                 'Thickness', 4
        #                 'Num_Neighbors', 5
        #                 'A_P_speed_VE', 6
        #                 # 'A_P_speed_Epi', 
        #                 'cum_A_P_VE_speed', 7
        #                 # 'epi_contour_dist', # can't have this. 
        #                 # 'Normal_VE_minus_Epi', # doesn seem to do anything? 
        #                 # 'Normal VE', 
        #                 # 'Normal Epi', 
        #                 'Mean_Curvature_VE', 8
        #                 'Mean_Curvature_Epi', 9
        #                 'Gauss_Curvature_VE', 10
        #                 'Gauss_Curvature_Epi', #] 11
        #                 'norm_apical_actin', 12
        #                 'norm_perimeter_actin'] 13
        
        X_feats_all.append(X_feats)
        X_feats_all_raw.append(metrics_data.astype(np.float)[select_row])
        embryo_all.append(embryo[select_row])
        frame_all.append(frame[select_row])
        stage_all.append(stage[select_row])#
        static_quad_id_all.append(static_quad_id[select_row])
        dynamic_quad_id_all.append(dynamic_quad_id[select_row])
        epi_contour_dist_ratio_all.append(epi_contour_dist_ratio[select_row])
        
        row_id_all.append(row_id[select_row])
        cell_id_all.append(cell_id[select_row])
        track_id_all.append(track_id[select_row])
        
        TP_all.append(TP[select_row])
        
        
        
        
        X_feats_all_full.append(X_feats_full)
        X_feats_all_raw_full.append(metrics_data_full_raw.astype(np.float)[select_row_full])
        embryo_all_full.append(embryo_full_raw[select_row_full])
        frame_all_full.append(frame_full_raw[select_row_full])
        stage_all_full.append(stage_full_row[select_row_full])#
        static_quad_id_all_full.append(static_quad_id_full_row[select_row_full])
        dynamic_quad_id_all_full.append(dynamic_quad_id_full_row[select_row_full])
        epi_contour_dist_ratio_all_full.append(epi_contour_dist_ratio_full_row[select_row_full])
        
        row_id_all_full.append(row_id_full_row[select_row_full])
        cell_id_all_full.append(cell_id_full_row[select_row_full])
        track_id_all_full.append(track_id_full_row[select_row_full])
        
        TP_all_full.append(TP_full_row[select_row_full])
        
        # # save a version of the full for reference !
        # embryo_full_raw = embryo.copy()
        # frame_full_raw = frame.copy()
        # TP_full_row = TP.copy()
        # stage_full_row = stage.copy()
        # static_quad_id_full_row = static_quad_id.copy() #[keep_rows]
        # dynamic_quad_id_full_row = dynamic_quad_id.copy() #[keep_rows]
        # row_id_full_row = row_id.copy() #[keep_rows] # we might need this specifically for track reconstruction !. 
        # cell_id_full_row = cell_id.copy() #[keep_rows] # .values
        # track_id_full_row = track_id.copy() #[keep_rows]
        # epi_contour_dist_ratio_full_row = epi_contour_dist_ratio.copy() #[keep_rows]
        
        # metrics_data_full_raw = metrics_data.copy() #[keep_rows].copy()
        
    
    X_feats = np.vstack(X_feats_all)
    X_feats_raw = np.vstack(X_feats_all_raw)
    frame_all = np.hstack(frame_all) # this is the times. 
    embryo_all = np.hstack(embryo_all)
    stage_all = np.hstack(stage_all)
    static_quad_id_all = np.hstack(static_quad_id_all)
    dynamic_quad_id_all = np.hstack(dynamic_quad_id_all)
    epi_contour_dist_ratio_all = np.hstack(epi_contour_dist_ratio_all)
    
    row_id_all = np.hstack(row_id_all)
    cell_id_all = np.hstack(cell_id_all)
    track_id_all = np.hstack(track_id_all)
    TP_all = np.hstack(TP_all)
    
    """
    We need the following for the full full 
    """
    X_feats_all_full = np.vstack(X_feats_all_full)
    X_feats_all_raw_full = np.vstack(X_feats_all_raw_full)
    embryo_all_full = np.hstack(embryo_all_full)
    frame_all_full = np.hstack(frame_all_full)
    stage_all_full = np.hstack(stage_all_full)
    static_quad_id_all_full = np.hstack(static_quad_id_all_full)
    dynamic_quad_id_all_full = np.hstack(dynamic_quad_id_all_full)
    epi_contour_dist_ratio_all_full = np.hstack(epi_contour_dist_ratio_all_full)
    
    row_id_all_full = np.hstack(row_id_all_full)
    cell_id_all_full = np.hstack(cell_id_all_full)
    track_id_all_full = np.hstack(track_id_all_full)
    
    TP_all_full = np.hstack(TP_all_full)
    
    
    
    from sklearn.decomposition import PCA
    
    pca_model = PCA(n_components=2)
    pca_model.fit(X_feats)
    
    Y_1 = pca_model.transform(X_feats)
    Y_2 = pca_model.transform(X_feats_all_full)
    
    plt.figure()
    plt.plot(Y_1[:,0], 
             Y_1[:,1], 'g.')
    plt.plot(Y_2[:,0], 
             Y_2[:,1], 'r.', alpha=0.5)
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.show()
    
    
    
    
# =============================================================================
#     We shouldn't need to do this and hopefully this will vastly boost the number of cells we capture!. 
# =============================================================================
    
    ##### outlier gates. 
    # further remove outliers.
    outliers = np.abs(X_feats).max(axis=1)
    outliers_gate = outliers>3
    # print(X_feats.shape)
    # print()
    
    
    """
    map the outliers gate now back to the original 
    """
    outliers_gate_original = keep_rows_original_select[outliers_gate>0] # this gives you indices but with respect to the original table indices. 
    
# =============================================================================
#     Version prior to embryo correction - save these X_feats too!. 
# =============================================================================
    # X_feats = X_feats[np.logical_not(outliers_gate)]
    # X_feats_raw = X_feats_raw[np.logical_not(outliers_gate)]
    # frame_all = frame_all[np.logical_not(outliers_gate)] # this need to be converted to TP=0 
    # embryo_all = embryo_all[np.logical_not(outliers_gate)]
    # stage_all = stage_all[np.logical_not(outliers_gate)]
    # static_quad_id_all = static_quad_id_all[np.logical_not(outliers_gate)]
    # dynamic_quad_id_all = dynamic_quad_id_all[np.logical_not(outliers_gate)]
    # epi_contour_dist_ratio_all = epi_contour_dist_ratio_all[np.logical_not(outliers_gate)]
    
    # row_id_all = row_id_all[np.logical_not(outliers_gate)].astype(np.int)
    # cell_id_all = cell_id_all[np.logical_not(outliers_gate)].astype(np.int)
    # track_id_all = track_id_all[np.logical_not(outliers_gate)].astype(np.int)
    # TP_all = TP_all[np.logical_not(outliers_gate)].astype(np.int)
    

    # X_feats_nobatch = X_feats.copy()
    print(X_feats.shape)
    print(X_feats_raw.shape)
# =============================================================================
#     Further implement linear batch correction over embryos? 
# =============================================================================

    import statsmodels.formula.api as smf
    import pandas as pd 
    import sklearn.preprocessing as preprocessing
    
    le = preprocessing.LabelEncoder()        
    # all_feats_table = pd.DataFrame(np.hstack([np.vstack([X_timelapse_feats, 
    #                                           X_ctrl_feats[:,:-1]]), 
    #                                           np.hstack([embryo_all, 
    #                                                       all_emb_names])[:,None]]), 
    #                                 columns= np.hstack([ctrl_feat_names[:-1], 'embryo']))  
    all_feats_table = pd.DataFrame( np.hstack([X_feats, embryo_all[:,None]]),  
                                    columns= np.hstack([col_name_select, 'embryo']))  
    
    for var in all_feats_table.columns[:-1]:
        all_feats_table[var] = all_feats_table[var].values.astype(np.float)
    all_feats_table['embryo'] = all_feats_table['embryo'].values.astype(np.str)
    
    
    
    """
    preproccesing
    """
    all_feats_table_full = pd.DataFrame(np.hstack([X_feats_all_full, embryo_all_full[:,None]]),
                                        columns= np.hstack([col_name_select, 'embryo']))
    for var in all_feats_table_full.columns[:-1]:
        all_feats_table_full[var] = all_feats_table_full[var].values.astype(np.float)
    all_feats_table_full['embryo'] = all_feats_table_full['embryo'].values.astype(np.str)
    
    
    # le.fit(all_feats_table['embryo'])
    # embryo_int = le.transform(all_feats_table['embryo'])
    # all_feats_table['embryo_int'] = embryo_int.astype(np.int)
    
    all_feats_corrected = []
    all_feats_corrected_original = [] 
    
    for var in all_feats_table.columns[:-1]:
        
        """
        fit based on the original 
        """
        mod = smf.ols(formula='%s ~ embryo' %(var), data=all_feats_table)
        res = mod.fit()
        # print(res.params)
        
        correct_var = res.params['Intercept'] + res.resid # do we include or don't include intercept? 
        # # correct_var = res.resid.copy() # looks like doing this is slightly better. 
        all_feats_corrected_original.append(correct_var)
        
        """
        apply to the full! 
        """
        resid = all_feats_table_full[var] - res.predict(exog=all_feats_table_full['embryo']) 
        correct_var = res.params['Intercept'] + resid
        
        all_feats_corrected.append(correct_var)
        
    all_feats_corrected_original = np.array(all_feats_corrected_original).T
    all_feats_corrected = np.array(all_feats_corrected).T
    # all_feats_corrected = np.vstack([X_timelapse_feats, 
    #                                           X_ctrl_feats[:,:-1]])

    corr_matrix = pairwise_distances(X_feats.T, metric='correlation', force_all_finite=False) # v. strange. 
  
    plt.figure(figsize=(10,10))
    plt.imshow(1.-corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
    plt.yticks(np.arange(len(corr_matrix)),
                col_name_select[:])
    plt.xticks(np.arange(len(corr_matrix)),
                col_name_select[:], rotation=45, horizontalalignment='right')
    # plt.yticklabels(col_name_select[2:])
    plt.show()
    

    corr_matrix = pairwise_distances(all_feats_corrected.T, metric='correlation', force_all_finite=False) # v. strange. 
  
    plt.figure(figsize=(10,10))
    plt.imshow(1.-corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
    plt.yticks(np.arange(len(corr_matrix)),
                col_name_select[:])
    plt.xticks(np.arange(len(corr_matrix)),
                col_name_select[:], rotation=45, horizontalalignment='right')
    # plt.yticklabels(col_name_select[2:])
    plt.show()

        
    """
    Correction or not doesn't make much difference -> will largely erase the differences. 
    """
    X_feats = all_feats_corrected.copy() # these are none corrrected? already ?
    
    # stick in an arbitrary clip!. 
    # X_feats = np.clip(X_feats, -3.5, 3.5)

# # =============================================================================
# #   based on these normalizations we start performing Lasso. ! 
# # =============================================================================
#     """
#     Creating subset indices for further analysis / testing of feature relationships.  
#     """
#     quads_select = [[8,1,2],
#                     [4,5,6],
#                     [16,9,10], 
#                     [12,13,14],
#                     [24,17,18], 
#                     [20,21,22],
#                     [32,25,26],
#                     [28,29,30]]
#     # quads_select = np.arange(32)+1
    
#     # quads_select = [[1],
#     #                 [4,5,6],
#     #                 [16,9,10], 
#     #                 [12,13,14],
#     #                 [24,17,18], 
#     #                 [20,21,22],
#     #                 [32,25,26]]
    
#     quads_select_names = ['Epi-Distal-Anterior', 
#                           'Epi-Distal-Posterior',
#                           'Epi-Prox-Anterior', 
#                           'Epi-Prox-Posterior',
#                           'ExE-Distal-Anterior',
#                           'ExE-Distal-Posterior',
#                           'ExE-Prox-Anterior',
#                           'ExE-Prox-Posterior']
    
#     # quads_select_names = np.hstack([str(qq) for qq in quads_select])
    
    
#     stages = ['Pre-migration', 
#               'Migration to Boundary']
    
    
#     dynamic_quad_id_main = np.zeros_like(dynamic_quad_id_all)
#     for qq_ii, qq in enumerate(quads_select):
#         for qq_ss in qq:
#             dynamic_quad_id_main[dynamic_quad_id_all==qq_ss] = qq_ii+1
#     dynamic_quad_id_main = np.hstack(dynamic_quad_id_main)

# # =============================================================================
# #     Feature transformed PCA
# # =============================================================================

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    pca_model = PCA(n_components=2)
    pca_model.fit(X_feats)
    
#     print(pca_model.explained_variance_ratio_)
#     # what do each component mainly refer to ? 
    
#     plt.figure()
#     plt.title('Component 1')
#     plt.plot(pca_model.components_[0])
#     plt.xticks(np.arange(X_feats.shape[1]), col_name_select, rotation=45, ha='right')
    
    
#     plt.figure()
#     plt.title('Component 2')
#     plt.plot(pca_model.components_[1])
#     plt.xticks(np.arange(X_feats.shape[1]), col_name_select, rotation=45, ha='right')
    
    
    
#     plt.figure()
#     plt.title('Component 1')
#     plt.plot(np.abs(pca_model.components_[0]))
#     plt.xticks(np.arange(X_feats.shape[1]), col_name_select, rotation=45, ha='right')
    
    
#     plt.figure()
#     plt.title('Component 2')
#     plt.plot(np.abs(pca_model.components_[1]))
#     plt.xticks(np.arange(X_feats.shape[1]), col_name_select, rotation=45, ha='right')
    
    
    """
    PCA model
    """
    # pca_proj = pca_model.transform(X_feats)
    
    # plt.figure()
    # plt.plot(pca_proj[:,0], 
    #          pca_proj[:,1], '.')
    # plt.xlim([-1,1])
    # plt.show()
#     import umap 
    
#     # umap_proj = 
#     noneighborindex = np.setdiff1d(np.arange(len(col_name_select)), 5) # this is now the 5th one!. 
#     # 15 worked well here.
#     fit = umap.UMAP(random_state=0, 
#                     n_neighbors=100, # 100 seems ideal or 200 ....
#                     # init = pca_proj,  # use PCA to initialise. 
#                     # local_connectivity=1, 
#                     output_metric='euclidean',
#                     # output_metric='correlation', ### I see if correlation then we go to PCA!!!. # correlation metric seems to work better? ---> does this allow 2 to show up without inclusion of other params? 
#                     spread=1.) # a big problem is effect of 1 variable... 
#     # fit = PCA(n_components=2)
#     umap_proj = fit.fit_transform(X_feats[:,noneighborindex]) # what if we don't do this ? 
    
#     # high number of clusters actually resulted in better representation  ? ---> how does this effect automatic hclustering? 
#     kmean_cluster = KMeans(n_clusters=100, random_state=0) # is 100 clusters the best? 
#     pca_proj_cluster = kmean_cluster.fit_predict(pca_proj) # how this change if we do this on the original feats? 
#     # pca_proj_cluster = kmean_cluster.fit_predict(X_feats) # how this change if we do this on the original feats? 
    
    
#     cluster_ids = np.unique(pca_proj_cluster)
#     cluster_ids_members = [np.arange(len(pca_proj_cluster))[pca_proj_cluster==cc] for cc in cluster_ids]
#     cluster_ids_centroids = np.vstack([np.mean(pca_proj[cc], axis=0) for cc in cluster_ids_members])
    
    
#     umap_proj_cluster = kmean_cluster.fit_predict(umap_proj) # how this change if we do this on the original feats? 
#     # pca_proj_cluster = kmean_cluster.fit_predict(X_feats) # how this change if we do this on the original feats? 
    
#     umap_cluster_ids = np.unique(umap_proj_cluster)
#     umap_cluster_ids_members = [np.arange(len(umap_proj_cluster))[umap_proj_cluster==cc] for cc in umap_cluster_ids]
#     umap_cluster_ids_centroids = np.vstack([np.mean(umap_proj[cc], axis=0) for cc in umap_cluster_ids_members])
    
# =============================================================================
#     Export all the associated statistics into one .mat file so it is easy to investigate plotting without re-running everything 
# =============================================================================

    import scipy.io as spio 
    
    # # savefile = os.path.join(saveplotfolder, 'pca_umap_analysis_data.mat')
    # savefile = os.path.join(saveplotfolder, '2021-09-02_pca_umap_analysis_data-Neighbors200_EuclideanMetric-ALLCells.mat')
    # # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_CorrelationMetric.mat') # this is similar to 200!
    # # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_EuclideanMetric_NoEpiKappa.mat') # this is similar to 200!
    # # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_EuclideanMetric_NoEpiKappa.mat') # this is similar to 200!
    
    # spio.savemat(savefile, 
    #               {'X_feats': X_feats,
    #                'X_feats_nobatch' : X_feats_nobatch,
    #               'X_feats_raw': X_feats_raw,
    #               'PCA_feats': pca_proj,
    #               'UMAP_feats': umap_proj,
    #               'Frame':frame_all, 
    #               'Embryo': embryo_all,
    #               'Stage': stage_all, 
    #               'Static_quad': static_quad_id_all, 
    #               'Dynamic_quad' : dynamic_quad_id_all, 
    #               'Dynamic_quad_main': dynamic_quad_id_main,
    #               'epi_contour_dist': epi_contour_dist_ratio_all, 
    #               'feat_names': col_name_select,
    #               'PCA_kmeans_cluster': pca_proj_cluster,
    #               'PCA_kmeans_cluster_id': cluster_ids, 
    #               'PCA_cluster_centroids': cluster_ids_centroids, 
    #               'UMAP_kmeans_cluster': umap_proj_cluster,
    #               'UMAP_kmeans_cluster_id': umap_cluster_ids, 
    #               'UMAP_cluster_centroids': umap_cluster_ids_centroids,
    #               'row_id': row_id_all, 
    #               'TP':TP_all, 
    #               'cell_id':cell_id_all, 
    #               'track_id':track_id_all})
    
    
# # =============================================================================
# #     Load the ward clustering results. 
# # =============================================================================
#     # saveplotfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-19_PCA_analysis'
#     saveplotfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-19_PCA_analysis'
    
#     savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_CorrelationMetric.mat') # this version seems most specific -> use this version!. 
   
#     # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors500_EuclideanMetric.mat') # this is similar to 200!
#     # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_CorrelationMetric.mat') # this is similar to 200!
#     # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_CorrelationMetric.mat')
#     # spio.savemat(savefile, 
#     #              {'X_feats': X_feats,
#     #               'Frame':frame_all, 
#     #               'Embryo': embryo_all,
#     #               'Stage': stage_all, 
#     #               'Static_quad': static_quad_id_all, 
#     #               'Dynamic_quad' : dynamic_quad_id_all, 
#     #               'epi_contour_dist': epi_contour_dist_ratio_all, 
#     #               'feat_names': col_name_select,
#     #               'PCA_kmeans_cluster': pca_proj_cluster,
#     #               'PCA_kmeans_cluster_id': cluster_ids, 
#     #               'cluster_centroids': cluster_ids_centroids})
#     saveobj = spio.loadmat(savefile)
    
#     # saveplotfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-24_UMAP_Correlation_analysis'
#     # fio.mkdir(saveplotfolder)
    
#     # """
#     # Load the series of features. 
#     # """
#     X_feats_select = saveobj['X_feats'].copy()
#     X_feats_raw_select = saveobj['X_feats_raw'].copy()
#     # frame_all = saveobj['Frame'].ravel()
#     # embryo_all = saveobj['Embryo'].ravel()
#     # stage_all = saveobj['Stage'].ravel()
#     # static_quad_id_all = saveobj['Static_quad'].ravel()
#     # dynamic_quad_id_all = saveobj['Dynamic_quad'].ravel()
#     # dynamic_quad_id_main = saveobj['Dynamic_quad_main'].ravel()
#     # epi_contour_dist = saveobj['epi_contour_dist'].ravel()
#     # col_name_select = saveobj['feat_names'].ravel()
    
#     # # # parsing the preclustered data - here we use the UMAP. 
#     # # pca_proj = saveobj['PCA_feats']
#     # # pca_proj_cluster = saveobj['PCA_kmeans_cluster'].ravel()
#     # # cluster_ids = saveobj['PCA_kmeans_cluster_id'].ravel()
#     # # cluster_ids_centroids = saveobj['PCA_cluster_centroids']
#     # pca_proj = saveobj['UMAP_feats']
#     # pca_proj_cluster = saveobj['UMAP_kmeans_cluster'].ravel()
#     # cluster_ids = saveobj['UMAP_kmeans_cluster_id'].ravel()
#     # cluster_ids_centroids = saveobj['UMAP_cluster_centroids']
    
#     # row_id_all = np.hstack(saveobj['row_id'].ravel())
#     # TP_all = np.hstack(saveobj['TP'].ravel())
#     # cell_id_all = np.hstack(saveobj['cell_id'].ravel())
#     # track_id_all = np.hstack(saveobj['track_id'].ravel())
    
    
    
    
    # #### check correlation # that the gating was done properly!
    # X_feats_NoOutlier = X_feats[np.logical_not(outliers_gate)].copy()
    
    # plt.figure()
    # plt.plot(X_feats_[:,0], 
    #          X_feats_select[:,0], '.')
    # plt.show()
    
    
# =============================================================================
#     Load the umap clusters. 
# =============================================================================
    
    # saveplotfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-24_UMAP_Correlation_analysis'
    saveplotfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-24_UMAP_Correlation_analysis'
    
    clustering_savefolder = os.path.join(saveplotfolder, 'cluster_stats');
    # fio.mkdir(clustering_savefolder)
    saveclustering_matfile = os.path.join(clustering_savefolder, 'ward_hcluster_stats_groups_all_embryos.mat')
    # spio.savemat(savematfile, 
    #              {'inputfile': savefile, # use this file input to get the other parameters for analysis., 
    #               'region_cluster_labels': region_cluster_labels, 
    #               'region_cluster_labels_colors' : region_cluster_labels_color, 
    #               'individual_umap_cluster_labels': umap_phenotype_cluster_labels, 
    #               # 'cluster_dendrogram' : g, # can't save this... hm... 
    #               'mean_heatmap_matrix': mean_heatmap_matrix, 
    #               'std_heatmap_matrix': sd_heatmap_matrix, 
    #               'homo_classes_pre': homo_classes, 
    #               'homo_classes_after': class_weights,
    #               'dendrogram_homo_all': homogeneity_all, 
    #               'dendrogram_labels_all': labels_all,
    #               'mean_heatmap_matrix_raw': mean_heatmap_matrix_raw,  
    #               'sd_heatmap_matrix_raw': sd_heatmap_matrix_raw}) 
    
    region_clustering_obj = spio.loadmat(saveclustering_matfile)
    
    region_cluster_labels = np.hstack(region_clustering_obj['region_cluster_labels'])
    region_cluster_labels_colors = region_clustering_obj['region_cluster_labels_colors'] 
    umap_cluster_labels = np.hstack(region_clustering_obj['individual_umap_cluster_labels'])
   
    
    # saveplotfolder_umap_region = os.path.join(saveplotfolder, 'UMAP_cluster_cells_mapback')
    # fio.mkdir(saveplotfolder_umap_region)
    
    # create a gating by time .... to do the trajectory analysis 
# =============================================================================
#     Derive trajectories for the phenomic clusters including -1 
# =============================================================================

    unique_umap_clusters = np.unique(umap_cluster_labels)

    import matplotlib 
    import seaborn as sns 
    
    region_cluster_labels_colors = np.vstack(sns.color_palette('colorblind', 16))[np.unique(region_cluster_labels)[1:]]
    region_cluster_labels_color = np.vstack([matplotlib.colors.to_rgb('lightgrey'),
                                              region_cluster_labels_colors])



# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # Keypoint here. ------ use relevant subset to try labelling the missing cells e.g. labelling propagation ? 
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


    from sklearn.neighbors import NearestNeighbors
    # from sklearn.cluster import AffinityPropagation
    from sklearn.semi_supervised import LabelPropagation, LabelSpreading
    
    
    # kNN_model_umap = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(X_feats_NoOutlier) # make this the same. 
    # distances, indices = nbrs.kneighbors(X_feats) # predict the nearest neighbor? 
    # X_feats_NoOutlier
    
    """
    to do: compare to label spreading (i.e. use mesh laplacian)
    """
    # label_prop_model = LabelPropagation(kernel='knn', n_neighbors=100) # use same as inpu to UMAP. 
    label_prop_model = LabelSpreading(kernel='knn', n_neighbors=15, alpha=0.01, n_jobs=4)
    # label_prop_model = LabelSpreading(kernel='rbf', n_neighbors=1, alpha=0.01, n_jobs=4)


    labels_in = -1*np.ones(len(X_feats), dtype=np.int64)
    # labels_in[np.logical_not(outliers_gate)] = umap_cluster_labels + 1 # to ensure positivity
    labelled_indices = np.setdiff1d(keep_rows_original_select, outliers_gate_original)
    labels_in[labelled_indices] = umap_cluster_labels + 1 # to ensure positivity

    label_prop_model.fit(X_feats, labels_in)
    
    labels_out = label_prop_model.label_distributions_.copy()
    labels_out_max = np.unique(umap_cluster_labels)[np.argmax(labels_out,axis=1)]
    # check 
    
    # # can be used to check clamping!. (uncomment to do so)
    # plt.plot(umap_cluster_labels, labels_out_max[~outliers_gate],'.')
    
    
#     import numpy as np
# >>> from sklearn import datasets
# >>> from sklearn.semi_supervised import LabelPropagation
# >>> label_prop_model = LabelPropagation()
# >>> iris = datasets.load_iris()
# >>> rng = np.random.RandomState(42)
# >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
# >>> labels = np.copy(iris.target)
# >>> labels[random_unlabeled_points] = -1
# >>> label_prop_model.fit(iris.data, labels)
# LabelPropagation(...)


# =============================================================================
# =============================================================================
# # Export the imputed labels with the 
# =============================================================================
# =============================================================================

    save_all_stats_folder = os.path.join('../../../Data_Analysis/LifeAct/PseudoAnalysis/Data/Cell_Statistics_3_Compiled_w_UMAP_clusters')
    fio.mkdir(save_all_stats_folder)
    
    # need to use dict to convert the labels into a group ID! (which is that of the phenomic cluster )
    # need to additionally add a column for the regional IDs. 

    metrics_tables['UMAP_labels_num'] = labels_out_max.copy()
    
    # mark out which was imputed and therefore shouldn't be considered as 'unique' 
    impute_bool = np.ones(len(metrics_tables)); impute_bool[labelled_indices]=0
    metrics_tables['UMAP_labels_imputed'] = impute_bool

    # remap. 
    UMAP_cluster_names = np.zeros(len(labels_out_max), dtype=np.str)
    UMAP_unique_labels = np.unique(labels_out_max)
    order_cluster = np.hstack([2, 1, 4, 3, 0])  
    UMAP_cluster_names_unique = ['A','B','C','D','E']
    
    for lab_ii, lab in enumerate(UMAP_unique_labels[order_cluster]):
        UMAP_cluster_names[labels_out_max==lab] = UMAP_cluster_names_unique[lab_ii]
        
    metrics_tables['UMAP_clusters_names'] = UMAP_cluster_names
    
    
    # unique cell ID should indeed be this just the embryoID + the cell track ID!. 
    embryo_ID = metrics_tables['embryo'].values
    track_ID = metrics_tables['Cell_Track_ID'].values
    
    unique_cell_IDs = np.hstack([embryo_ID[kkk]+'_'+str(int(track_ID[kkk])).zfill(4) for kkk in np.arange(len(track_ID))])
    metrics_tables['unique_cell_IDs'] = unique_cell_IDs   ### so this number is correct! 
     
    metrics_tables.to_csv(os.path.join(save_all_stats_folder, '2023-01-18_all_embryo_compiled_with_UMAP_clusters.csv' ), index=None)


# =============================================================================
# =============================================================================
# =============================================================================
# # Proceed to get the mean and averages in any case. 
# =============================================================================
# =============================================================================
# =============================================================================


    # grab the average over cell instances. 
    # # col_name_select = ['Area', 
    #                     'Perimeter', 
    #                     'Shape_Index', 
    #                     'Eccentricity', 
    #                     'Thickness',
    #                     'Num_Neighbors',
    #                     'A_P_speed_VE', 
    #                     # 'A_P_speed_Epi', 
    #                     'cum_A_P_VE_speed', 
    #                     # 'epi_contour_dist', # can't have this. 
    #                     # 'Normal_VE_minus_Epi', # doesn seem to do anything? 
    #                     # 'Normal VE', 
    #                     # 'Normal Epi', 
    #                     'Mean_Curvature_VE', 
    #                     'Mean_Curvature_Epi',
    #                     'Gauss_Curvature_VE', 
    #                     'Gauss_Curvature_Epi', #]
    #                     'norm_apical_actin', 
    #                     'norm_perimeter_actin']
    
    impute_bool_nan = impute_bool.copy(); impute_bool_nan[impute_bool_nan==1] = np.nan; impute_bool_nan[impute_bool_nan==0] = 1 # using nan allows masked mean. 
    # all_cell_class = 
    
    
    # # metrics of interest. 
    # area = metrics_tables['Area'].values
    # perim = metrics_tables['Perimeter_Proj'].values
    # shape_index = metrics_tables['Shape_Index_Proj'].values
    # eccentricity = metrics_tables['Eccentricity'].values
    # thickness = metrics_tables['Thickness'].values
    # n_neighbors = metrics_tables['Cell_Neighbours_Track_ID'].values
    # ve_speed = metrics_tables['AVE_direction_VE_speed_3D_ref'].values
    # epi_speed = metrics_tables['AVE_direction_Epi_speed_3D_ref'].values
    # cum_ave = metrics_tables['AVE_cum_dist_coord_3D_ref'].values # just for PCA!. 
    # epi_contour_dist_ratio = metrics_tables['cell_epi_contour_dist_ratio'].values
    
    # normal_diff = metrics_tables['demons_Norm_dir_VE'].values - metrics_tables['demons_Norm_dir_Epi'].values
    # # normal_ve = metrics_tables['demons_Norm_dir_VE'].values
    # # normal_epi = metrics_tables['demons_Norm_dir_Epi'].values
    # curvature_VE = metrics_tables['mean_curvature_VE'].values
    # curvature_Epi = metrics_tables['mean_curvature_Epi'].values
    # gauss_curvature_VE = metrics_tables['gauss_curvature_VE'].values
    # gauss_curvature_Epi = metrics_tables['gauss_curvature_Epi'].values
    
    # # augment with the actin levels # ..... 
    # apical_actin_VE = metrics_tables['norm_apical_actin_intensity']
    # perimeter_actin_VE = metrics_tables['norm_perimeter_actin_intensity']

    metrics_select = ['Area', 
                        'Perimeter_Proj', 
                        'Shape_Index_Proj', 
                        'Eccentricity', 
                        'Thickness',
                        'Cell_Neighbours_Track_ID',
                        'AVE_direction_VE_speed_3D_ref', 
                        # 'A_P_speed_Epi', 
                        'AVE_cum_dist_coord_3D_ref', 
                        # 'epi_contour_dist', # can't have this. 
                        # 'Normal_VE_minus_Epi', # doesn seem to do anything? 
                        # 'Normal VE', 
                        # 'Normal Epi', 
                        'mean_curvature_VE', 
                        'mean_curvature_Epi',
                        'gauss_curvature_VE', 
                        'gauss_curvature_Epi', #]
                        'norm_apical_actin_intensity', 
                        'norm_perimeter_actin_intensity']
    
    # # augment with the actin levels # ..... 
    # apical_actin_VE = metrics_tables['norm_apical_actin_intensity']
    # perimeter_actin_VE = metrics_tables['norm_perimeter_actin_intensity']
    
    
    uniq_cell_instance_num = []
    cell_instance_num = []
    average_parameters = []
    sd_parameters = []
    
    for umap_name in ['A','B','C','D','E']:

        umap_name_select = metrics_tables['UMAP_clusters_names'].values == umap_name
        total_select = np.logical_and(umap_name_select, impute_bool_nan==1)
        
        # compute the unique cell instances. 
        unique_cell_IDs = metrics_tables['unique_cell_IDs'].values.copy()
        unique_cell_IDs_select = np.unique(unique_cell_IDs[total_select>0])
        
        uniq_cell_instance_num.append(len(unique_cell_IDs_select))
        
        n_cell_instance_num = np.sum(total_select)
        cell_instance_num.append(n_cell_instance_num)
        
        average_parameters_feat = []
        sd_parameters_feat = []
        
        for feature in metrics_select:
            data_feature = metrics_tables[feature].values.copy()
            data_feature = data_feature*impute_bool_nan
    
            average_parameters_feat.append(np.mean(data_feature[total_select>0]))
            sd_parameters_feat.append(np.std(data_feature[total_select>0]))
        # compute the average and s.d. for each phenotypic cluster. 
        
        average_parameters.append(average_parameters_feat)
        sd_parameters.append(sd_parameters_feat)
        
    
    uniq_cell_instance_num = np.hstack(uniq_cell_instance_num)
    cell_instance_num = np.hstack(cell_instance_num)
    average_parameters = np.array(average_parameters)
    sd_parameters = np.array(sd_parameters)
    
    average_sd_parameters = np.zeros((len(average_parameters), 2*average_parameters.shape[1]))
    average_sd_parameters[:,2*np.arange(average_parameters.shape[1])] = average_parameters.copy()
    average_sd_parameters[:,2*np.arange(average_parameters.shape[1])+1] = sd_parameters.copy()
    
    
    mean_stats_table = np.hstack([np.hstack(['A','B','C','D','E'])[:,None],
                                  uniq_cell_instance_num[:,None], 
                                  cell_instance_num[:,None],
                                  average_sd_parameters])
    mean_stats_table_col = np.hstack(['UMAP_cluster', 
                            '#uniq_cells(tracks)', 
                            '#cell_instances(points)',
                            np.hstack([np.hstack(['mean_'+col_name_select[kkk], 'std_'+ col_name_select[kkk]]) for kkk in np.arange(len(col_name_select))])])
    
    mean_stats_table_pandas = pd.DataFrame(mean_stats_table, 
                                           columns=mean_stats_table_col ,
                                           index=None)

    mean_stats_table_pandas.to_csv(os.path.join(save_all_stats_folder, '2023-01-18_UMAP_clusters_mean_cell_stats.csv' ), index=None)
    


# # =============================================================================
# #   retrieve ... and mapback the revised clusters for visual checking!. 
# # =============================================================================

#     import skimage.io as skio
    
#     embryo_all = np.hstack(embryo_all)
#     uniq_embryo = np.unique(embryo_all)
    
#     # TP0_uniq_embryo = np.hstack([np.abs(frame_all[embryo_all==emb]).min() for emb in uniq_embryo])
#     embryo_TP0_frame = {emb:np.abs(frame_all[embryo_all==emb]).min() for emb in uniq_embryo}
    
#     embryo_file_dict = {}
    
#     # metricfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3'
#     # imgfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\CellImages'
#     metricfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3'
#     imgfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\CellImages'
    
    
#     for emb in uniq_embryo:
    
#         # metricfile: 
#         # segfile = 
#         # polarfile =  # rotated? or not .... -> should do rotated? -> rotate this and save out.... for this.  
#         tracklookupfile = os.path.join(metricfolder, 'single_cell_track_stats_lookup_table_'+emb+'.csv')
#         singlecellstatfile = os.path.join(metricfolder, 'single_cell_statistics_table_export-ecc-realxyz-from-demons_'+emb+'.csv')
        
#         polarfile = os.path.join(imgfolder, emb, 'A-P_rotated_data','AVE_rot-%s' %(emb) + '.tif')
#         segfile = os.path.join(imgfolder, emb, 'A-P_rotated_data','AVE_rot-cells_%s' %(emb) + '.tif')
        
#         # create a matrix to save the UMAP clusters. 
#         cells_emb = skio.imread(segfile)
#         cells_emb_umap = np.zeros_like(cells_emb)
        
#         metadict = {'segfile': segfile, 
#                     'tracklookupfile': tracklookupfile,
#                     'singlecellstatfile': singlecellstatfile,
#                     'polarfile': polarfile,
#                     'cells': cells_emb,
#                     'cells_umap_cluster': cells_emb_umap} # the last one will be updated.! 
#         embryo_file_dict[emb] = metadict
    
# # =============================================================================
# #   Can we save this in some manner ... ? 
# # =============================================================================

#     # # wrap all the meta into one dict. 
#     # metadict_id_array = {'embryo_all':embryo_all, 
#     #                       'TP_all':TP_all, 
#     #                       'cell_id_all': cell_id_all, 
#     #                       'row_id_all': row_id_all, 
#     #                       'track_id_all': track_id_all}
    
#     metadict_id_array = {'embryo_all':embryo_all_full, 
#                           'TP_all':TP_all_full, 
#                           'cell_id_all': cell_id_all_full, 
#                           'row_id_all': row_id_all_full, 
#                           'track_id_all': track_id_all_full}

#     # i think this is only taking the goodness. 
#     valid_region_clusters = np.unique(region_cluster_labels) 
#     valid_region_clusters = valid_region_clusters[valid_region_clusters>=-1]

#     # NSWE_complexes = [np.arange(len(region_cluster_labels))[region_cluster_labels==iii] for iii in range(region_cluster_labels.max()+1)]
#     NSWE_complexes = [np.arange(len(region_cluster_labels))[region_cluster_labels==iii] for iii in valid_region_clusters]
#     # NSWE_colors = sns.color_palette('Set2', n_colors=len(NSWE_complexes)) # make these the same as the region colors!.
#     # NSWE_colors = np.vstack([region_cluster_labels_color[cc] for cc in valid_region_clusters])
#     NSWE_colors = region_cluster_labels_color.copy()

    
# # =============================================================================
# #   Initialise the new colors !. 
# # =============================================================================
#     order_cluster = np.hstack([2, 1, 4, 3, 0])
    
#     # redo the coloring.... 
#     NSWE_colors_new = NSWE_colors.copy()
#     NSWE_colors_new[order_cluster[0]] = NSWE_colors[3] # AVE 
#     NSWE_colors_new[order_cluster[1]] = NSWE_colors[2] # AVE 
#     # NSWE_colors_new[order_cluster[2]] = NSWE_colors[2] # AVE 
#     NSWE_colors_new[order_cluster[3]] = NSWE_colors[1] # AVE 
#     # NSWE_colors_new[order_cluster[1]] = NSWE_colors[2] # AVE 
    
#     NSWE_colors = NSWE_colors_new.copy()

# # =============================================================================
# #     Initialise all the embryo cells simultaneously? 
# # =============================================================================
    
#     # actually the best way is to use batch_map_stats_back_to_image to pull the embryo level data per embryo....!!! # this will save all the lookups!. 

#     from tqdm import tqdm 
    
#     for emb in uniq_embryo[:]:
        
#         # emb_select = embryo_all == emb
#         emb_select = embryo_all_full == emb
#         # umap_cluster_labels_select = umap_cluster_labels[emb_select].copy()
#         umap_cluster_labels_select = labels_out_max[emb_select].copy()
    
#         pts_ids = np.arange(len(X_feats))[emb_select]
        
#         embryo_level_data, ent_data = batch_map_stats_back_to_image(pts_ids, 
#                                                                     metadict_id_array, 
#                                                                     embryo_file_dict,
#                                                                     return_global=True)
            
#         # get the frame of each unique_embryo
#         uniq_embryo = np.hstack(list(embryo_level_data.keys()))
#         uniq_embryo_ind = {uniq_embryo[ii]:ii for ii in np.arange(len(uniq_embryo))}
#         uniq_embryo_TP0 = np.hstack([embryo_TP0_frame[emb] for emb in uniq_embryo])
        
#         TP0_frames = [embryo_level_data[uniq_embryo[iii]]['img'][uniq_embryo_TP0[iii]] for iii in np.arange(len(uniq_embryo_TP0))]
        
        
#         cells_umap_ee = embryo_file_dict[emb]['cells_umap_cluster'].astype(np.float)
#         cells_umap_ee[:] = np.nan
#         cell_mask = embryo_file_dict[emb]['cells'].copy()
        
#         for ee_ii, ee in tqdm(enumerate(ent_data)):
#             emb = ee[0]['embryo']
#             emb_ind = uniq_embryo_ind[emb]
#             emb_cell_id = int(ee[0]['Cell_ID'])
#             xy = np.hstack([ee[0]['pos_2D_x_rot-AVE'], 
#                             ee[0]['pos_2D_y_rot-AVE']])
            
#             xytime = ee[0]['Frame'] - uniq_embryo_TP0[emb_ind]
            
#             umap_cluster_ee = umap_cluster_labels_select[ee_ii]
#             cells_umap_ee[xytime, cell_mask[xytime]==emb_cell_id] = umap_cluster_ee
            
#         cells_umap_ee_color = np.zeros((cells_umap_ee.shape[0], 
#                                         cells_umap_ee.shape[1], 
#                                         cells_umap_ee.shape[2],3))
#         # cells_umap_ee_color[:] = np.nan
        
#         for cc_ii, cc in enumerate(valid_region_clusters):
#             cells_umap_ee_color[cells_umap_ee == cc] = NSWE_colors[cc_ii][None,:]
        
#         # create transparency alpha? 
#         cells_umap_ee_color = np.concatenate([cells_umap_ee_color, 
#                                               np.ones_like(cells_umap_ee_color[...,0])[...,None]],
#                                               axis=-1)
#         cells_umap_ee_color[np.isnan(cells_umap_ee),-1] = 0
        
        
#         """
#         Also save the coloring into .tif -> this is required in order to map back to the 3D space.... 
#         """
#         cells_umap_ee_color_save_img = np.uint8(255*cells_umap_ee_color[...,:3])
#         cells_umap_ee_color_save_img[np.isnan(cells_umap_ee),:] = 255
        
        
#         # output the image frame by frame
#         saveimagefolder = emb + '_labelprop-all-filled_2022-11-14'
#         fio.mkdir(saveimagefolder)
        
#         distribution_emb = []        
        
#         for tt in np.arange(len(cells_umap_ee_color)):
            
#             umap_cells_tt = cells_umap_ee[tt].copy()
#             counts_cells = np.hstack([len(np.unique(cell_mask[tt,umap_cells_tt==cc])) for cc in valid_region_clusters])
#             distribution_emb.append(counts_cells)
            
#             fig, ax = plt.subplots(figsize=(10,10))
#             ax.imshow(cells_umap_ee_color[tt])
#             plt.axis('off')
#             plt.grid('off')
#             plt.savefig(os.path.join(saveimagefolder, 'Frame-%s.png' %(str(tt).zfill(3))), 
#                         dpi=300, bbox_inches='tight')
#             plt.show()
        
#         skio.imsave(os.path.join(saveimagefolder, 'colored_umap_clusters-'+emb+'.tif'), cells_umap_ee_color_save_img)
        
#         distribution_emb = np.vstack(distribution_emb)
        
#         """
#         save this output out
#         """
#         spio.savemat('UMAP_cellclusters-'+emb+'.mat', 
#                       {'cells_umap_ee_color': cells_umap_ee_color, 
#                       'cells_umap_ee': cells_umap_ee,
#                       'distribution_emb': distribution_emb})
        
#         # visualise the distribution_variation in stacked barplot form? and save this just for the clusters that are not ... background?
        
#         order_cluster = np.hstack([2, 1, 4, 3, 0])
#         cluster_order = order_cluster[::-1] # this has already been applied!!!! in this instance.! check again with Xiaoyue!
#         # cluster_order = np.arange(len(order_cluster))[::-1]
#         data = distribution_emb.copy()
#         data = data[:, cluster_order].copy() # put in the correct order for plotting. 
#         data = data / (data.sum(axis=1) + 1e-8)[:,None]
#         # create the cumulative totals for bar graph plotting. 
#         data_cumsums = np.cumsum(data, axis=1)
# #        ax.bar(labels, men_means, width, yerr=men_std, label='Men')
# #        ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
# #               label='Women')

#         fig, ax = plt.subplots(figsize=(5,5))
#         # plt.title(le.classes_[jjjj])WW
#         # this requires us to build and program the composition up. in order of the histogram 
#         for ord_ii in np.arange(data.shape[1]):
#             if ord_ii>0:
#                 ax.bar(np.arange(data.shape[0]), 
#                         data[:,ord_ii], 
#                         bottom = data_cumsums[:,ord_ii-1], 
#                         color=region_cluster_labels_color[order_cluster[::-1][ord_ii]],
#                         # color=region_cluster_labels_color[::-1][ord_ii],
#                         width=1,
#                         edgecolor='k')
#             else:
#                 ax.bar(np.arange(data.shape[0]), 
#                         data[:,ord_ii], 
#                         color=region_cluster_labels_color[order_cluster[::-1][ord_ii]],
#                         # color=region_cluster_labelWs_color[::-1][ord_ii],
#                         width=1,
#                         edgecolor='k')
#         plt.xlim([0-0.5,data.shape[0]-0.5])
#         plt.ylim([0,1])
        
#         # save this now. 
#         plt.savefig(os.path.join(saveimagefolder, 'UMAP_cell-cluster_distribution_%s.svg' %(str(emb).strip())), 
#                         dpi=300, bbox_inches='tight')
        
    
    
    
    






    
    
