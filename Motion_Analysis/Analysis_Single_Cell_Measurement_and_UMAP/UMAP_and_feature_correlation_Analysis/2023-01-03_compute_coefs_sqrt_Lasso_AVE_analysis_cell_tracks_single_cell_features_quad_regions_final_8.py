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
                    std_tformer=None, pow_tformer=None, X_max=None):

    from sklearn.preprocessing import StandardScaler, PowerTransformer
    
    return_model=False
    
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
        return X_out, (std_tformer, pow_tformer, X_max)
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
        

if __name__=="__main__":
    
    
    """
    We should make sure to do the same processing as for the UMAP analysis!!!!!. 
    """
    
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
    saveplotfolder = '../../../Data_Analysis/LifeAct/PseudoAnalysis/Data/Cell_Statistics_3/2022-01-19_PCA_analysis'
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
        
    metrics_tables = pd.concat(metrics_tables, ignore_index=True) # save this out. 
    metrics_tables_embryo = np.hstack(metrics_tables_embryo)
    # metrics_tables['Embryo_Name'] = metrics_tables_embryo # already have embryo saved... 
    
    print(metrics_tables.shape)
    print(metrics_tables_embryo.shape)
    
    # cell_track_data_tables = np.concatenate(cell_track_data_tables, axis=0)
#     # savecombmetricsfile = 'combined_cell_feats_Shankar_no-nan.csv'
#     # metrics_tables.to_csv(savecombmetricsfile, index=None)

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
                                # curvature_Epi,
                               gauss_curvature_VE,
                                # gauss_curvature_Epi, #]).T
                                apical_actin_VE, 
                                perimeter_actin_VE
                                ]).T
    
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


    invalid_rows = np.hstack([np.sum(np.isnan(metrics_data[row])) for row in np.arange(len(metrics_data))]) + np.hstack([np.sum(np.isinf(metrics_data[row])) for row in np.arange(len(metrics_data))])
    invalid_meta = np.logical_not(np.logical_and(~np.isnan(dynamic_quad_id), ~np.isnan(static_quad_id)))
    
    keep_rows = np.logical_and(invalid_rows==0, invalid_meta==0)
    
    
    col_name_select = ['Area', 
                        'Perimeter', 
                        'Shape_Index', 
                        'Eccentricity', 
                        'Thickness', #? i did include thickness? 
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
    
    # finalised. 
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
    
    
    corr_matrix = pairwise_distances(metrics_data.T, metric='correlation', force_all_finite=False) # v. strange. 
  
    plt.figure(figsize=(10,10))
    plt.imshow(1.-corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
    plt.yticks(np.arange(len(corr_matrix)),
                col_name_select[:])
    plt.xticks(np.arange(len(corr_matrix)),
                col_name_select[:], rotation=45, horizontalalignment='right')
    # plt.yticklabels(col_name_select[2:])
    plt.show()
    

# # =============================================================================
# #     1. Create the features, transform the features and create the labels of the data columns.  
# # =============================================================================

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
    
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    
    
    # iterate over each embryo and do z-score normalisation. 
    for emb in uniq_embryo:
        select_row = embryo == emb
        # for some reason this does not centralizae the features? 
        X_feats, tforms = transform_feats(metrics_data.astype(np.float)[select_row], 
                                            # apply_tform_cols=[0,1,2,3,4,5],
                                            # noapplytform_cols=[6,7,8,9,10,11,12,13,14,15], 
                                            # noapplytform_cols=[6,7,8,9,10,11,12,13],
                                            apply_tform_cols=[0,1,2,3,4,5],
                                            # noapplytform_cols=[6,7,8,9,10,11], #,12,13], # where there were natural signage.... then we put those variables here. 
                                            noapplytform_cols=[6,7,8,9,10,11,12,13],
                                            std_tformer=None, 
                                            pow_tformer=None, 
                                            X_max=None)
                                            # apply_tform_cols=[0,1,2,3,4,5],
                                            # noapplytform_cols=[6,7,8,9,10,11,12,13],
        
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
    
    # further remove outliers.
    outliers = np.abs(X_feats).max(axis=1)
    outliers_gate = outliers>3
    
    print(X_feats.shape)
    
# =============================================================================
#     Version prior to embryo correction - save these X_feats too!. 
# =============================================================================
    X_feats = X_feats[np.logical_not(outliers_gate)]
    X_feats_raw = X_feats_raw[np.logical_not(outliers_gate)]
    frame_all = frame_all[np.logical_not(outliers_gate)] # this need to be converted to TP=0 
    embryo_all = embryo_all[np.logical_not(outliers_gate)]
    stage_all = stage_all[np.logical_not(outliers_gate)]
    static_quad_id_all = static_quad_id_all[np.logical_not(outliers_gate)]
    dynamic_quad_id_all = dynamic_quad_id_all[np.logical_not(outliers_gate)]
    epi_contour_dist_ratio_all = epi_contour_dist_ratio_all[np.logical_not(outliers_gate)]
    
    row_id_all = row_id_all[np.logical_not(outliers_gate)].astype(np.int)
    cell_id_all = cell_id_all[np.logical_not(outliers_gate)].astype(np.int)
    track_id_all = track_id_all[np.logical_not(outliers_gate)].astype(np.int)
    TP_all = TP_all[np.logical_not(outliers_gate)].astype(np.int)
    
    X_feats_nobatch = X_feats.copy()
    
# =============================================================================
#     Further implement linear batch correction over embryos
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
    # le.fit(all_feats_table['embryo'])
    # embryo_int = le.transform(all_feats_table['embryo'])
    # all_feats_table['embryo_int'] = embryo_int.astype(np.int)
    
    all_feats_corrected = []
    
    for var in all_feats_table.columns[:-1]:
        mod = smf.ols(formula='%s ~ embryo' %(var), data=all_feats_table)
        res = mod.fit()
        # print(res.params)
        
        correct_var = res.params['Intercept'] + res.resid # do we include or don't include intercept? 
        # correct_var = res.resid.copy() # looks like doing this is slightly better. 
        all_feats_corrected.append(correct_var)
        
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

        
    # make the corrected the new X_feats. 
    X_feats = all_feats_corrected.copy() # these are none corrrected? already ?
    
    
    """
    Creating subset indices for further analysis / testing of feature relationships.  
    """
    quads_select = [[8,1,2],
                    [4,5,6],
                    [16,9,10], 
                    [12,13,14],
                    [24,17,18], 
                    [20,21,22],
                    [32,25,26],
                    [28,29,30]]
    # quads_select = np.arange(32)+1
    
    # quads_select = [[1],
    #                 [4,5,6],
    #                 [16,9,10], 
    #                 [12,13,14],
    #                 [24,17,18], 
    #                 [20,21,22],
    #                 [32,25,26]]
    
    quads_select_names = ['Epi-Distal-Anterior', 
                          'Epi-Distal-Posterior',
                          'Epi-Prox-Anterior', 
                          'Epi-Prox-Posterior',
                          'ExE-Distal-Anterior',
                          'ExE-Distal-Posterior',
                          'ExE-Prox-Anterior',
                          'ExE-Prox-Posterior']
    
    # quads_select_names = np.hstack([str(qq) for qq in quads_select])
    
    
    stages = ['Pre-migration', 
              'Migration to Boundary']
    
    
    # dynamic_quad_id_main = np.zeros_like(dynamic_quad_id_all)
    # for qq_ii, qq in enumerate(quads_select):
    #     for qq_ss in qq:
    #         dynamic_quad_id_main[dynamic_quad_id_all==qq_ss] = qq_ii+1
    # dynamic_quad_id_main = np.hstack(dynamic_quad_id_main)
    
    
    
    dynamic_quad = dynamic_quad_id_all.copy() # the code below already handles the subsetting? 
    static_quad = static_quad_id_all.copy()
    
    """
    Iterative subsetting
    """
    from sklearn.linear_model import Lasso
    import networkx as nx 
    import netgraph
    import pickle
    import itertools
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-PreMig-0.01'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-Mig-0.01'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-PreMig-0.05'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-Mig-0.05'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-PreMig-Auto'

    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-8quads-dynamic'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-32quads-static'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-32quads-dynamic'
    
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-rescale_coef'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2-rescale_coef'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-rescale_coef_post'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2-rescale_coef_post'
    
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-with_coef-std'
    
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_ref'
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_ref'
    
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_ref'
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_ref_nocum'
    analysis_model_save_folder = '2023-01-03_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_ref'
    
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_realxyz'
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_realxyz'
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_realxyz-nocum'
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_realxyz-nocum'
    
    
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-stdscale_coef'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2-stdscale_coef'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2'
    #create the new folder. 
    fio.mkdir(analysis_model_save_folder)
    
    lasso_alpha = .05 # this doesn't affect anything. 
    labellingtype = 'dynamic'
    # labellingtype = 'static'
    
    for stage_name_ii in stages[:]:
        
        print(stage_name_ii)
        for quad_ii in np.arange(len(quads_select))[:]:
            
            quad_ii_regions = quads_select[quad_ii]
            quads_ii_regions_names = quads_select_names[quad_ii]
            
            data_select = stage_all == stage_name_ii
            # quad_select_mask = static_quad == quad_ii_regions[0]
            # quad_select_mask = dynamic_quad == quad_ii_regions
            
            if labellingtype == 'dynamic':
                print('dynamic')
                quad_select_mask = dynamic_quad == quad_ii_regions[0]
                for jjj in np.arange(len(quad_ii_regions)-1):
                    # quad_select_mask = np.logical_or(quad_select_mask, 
                                                        # static_quad == quad_ii_regions[jjj+1])
                    quad_select_mask = np.logical_or(quad_select_mask, 
                                                        dynamic_quad == quad_ii_regions[jjj+1])
                print(np.unique(dynamic_quad[quad_select_mask>0]))
                print('-----')
                
            if labellingtype == 'static':
                print('static')
                quad_select_mask = static_quad == quad_ii_regions[0]
                for jjj in np.arange(len(quad_ii_regions)-1):
                    # quad_select_mask = np.logical_or(quad_select_mask, 
                                                        # static_quad == quad_ii_regions[jjj+1])
                    quad_select_mask = np.logical_or(quad_select_mask, 
                                                        static_quad == quad_ii_regions[jjj+1])
                print(np.unique(static_quad[quad_select_mask>0]))
                print('-----')
            
            # print(quad_select_mask.sum()) # seems to check out. 
            # np.logical_or(np.logical_or(static_quad==1, static_quad==2) , static_quad==8)
            data_select = np.logical_and(data_select, 
                                         quad_select_mask)
            
            # apply the masking. 
            X_feats_quad = X_feats[data_select].copy()
            X_feats_quad_std = X_feats_quad.std(axis=0)
            # X_feats_quad_std = X_feats_quad.std(axis=0)
            # X_feats_quad = X_feats_quad / (X_feats_quad.std(axis=0)[None,:]) # this is to allow the coefficient importance comparison.
            # X_feats_quad = StandardScaler().fit_transform(X_feats_quad)
            # X_feats_quad = X_feats[quad_select_mask].copy()
            
            # corr_matrix = pairwise_distances(X_feats_quad.T, 
            #                                  metric='correlation', 
            #                                  force_all_finite=False)
            
            # double check the nature of this calculation -> ascribe pvalue...             
            print(quads_ii_regions_names, quad_ii_regions)
            print(X_feats_quad.shape)
            N_cells = len(X_feats_quad)
            
            corr_matrix = np.zeros((X_feats_quad.shape[1], X_feats_quad.shape[1]))
            corr_matrix_R2 = np.zeros((X_feats_quad.shape[1], X_feats_quad.shape[1])) # equivalent of the pvalue.
            
            n_features = X_feats_quad.shape[1]
            
            for pred_ind in np.arange(n_features)[:]:
                
                # =============================================================================
                #   I think yeah... we need to do some choosing of parameters.                  
                # =============================================================================
                # clf = Lasso(alpha=lasso_alpha, normalize=False, 
                #                 fit_intercept=True) # check the regularization amount.
                # clf1 = LassoLars(alpha=lasso_alpha, normalize=False, 
                #                 fit_intercept=True)
                
                nonpred_ind = np.setdiff1d(np.arange(n_features), pred_ind)
                
                # do an over feat. 
                YY = X_feats_quad[:, pred_ind].copy()
                XX = X_feats_quad[:, nonpred_ind].copy()
                # clf.fit(XX, YY[:,None])
                # clf1.fit(XX,YY[:,None])
                
                mod = sm.OLS(YY, XX)
                # res = mod.fit()
                res = mod.fit_regularized(method='sqrt_lasso', L1_wt=1.)
                corr_matrix[pred_ind, nonpred_ind] = res.params #* X_feats_quad_std[nonpred_ind]
                
                YY_pred = res.predict(XX)
                r2 = r2_score(YY,YY_pred, multioutput='uniform_average')
                
                corr_matrix_R2[pred_ind,:] = r2
                
            # corr_matrix = np.zeros((X_feats_quad.shape[1], X_feats_quad.shape[1]))
            # corr_matrix_pval = np.zeros((X_feats_quad.shape[1], X_feats_quad.shape[1])) # this will act as our filter for reliable signals. 
            
            # for ii, jj in itertools.combinations(np.arange(X_feats_quad.shape[1]), 2):
            #     X_ii = X_feats_quad[:,ii]
            #     X_jj = X_feats_quad[:,jj]
                
            #     r, pval = pearsonr(X_ii, X_jj)
                
            #     corr_matrix[ii,jj] = r 
            #     corr_matrix[jj,ii] = r
            #     corr_matrix_pval[ii,jj] = pval; 
            #     corr_matrix_pval[jj,ii] = pval
                
            #     corr_matrix[ii,ii] = 1
            #     corr_matrix[jj,jj] = 1
                        
            # plt.figure()
            # plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            # plt.show()
            
            
            # # """
            # # Construction of the Lasso prediction models
            # # """
            # # # initialise and build a graph network? 
            # # G_weights = nx.DiGraph() # just those that have weights. 
            # # n_features = X_feats.shape[1]
            # # G_weights.add_nodes_from(np.arange(n_features)) # this is to hack the graph to always plot at the same position irrespective if there is an edge. 
            
            # # save_analysis_name = stage_name_ii+'_'+quads_ii_regions_names+'_'+'-'.join([str(q_ii) for q_ii in quad_ii_regions])
            save_analysis_name = stage_name_ii+'_'+quads_ii_regions_names + '_' + labellingtype
            
            fig, ax = plt.subplots(figsize=(15,15))
            plt.title('Coef Matrix, Normalised Feats %s, %s' %( stage_name_ii, quad_ii_regions), fontsize=32)
            plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
            plt.xticks(np.arange(len(corr_matrix)), col_name_select, rotation=45, horizontalalignment='right', fontsize=24)
            plt.yticks(np.arange(len(corr_matrix)), col_name_select, rotation=0, horizontalalignment='right', fontsize=24)
            fig.savefig(os.path.join(analysis_model_save_folder, save_analysis_name+'.svg'), bbox_inches='tight')
            plt.show()
            
            # pickle the current network connections etc. ready for comparison 
            filename = os.path.join(analysis_model_save_folder, save_analysis_name+'.p')
            savedict = {'lasso_matrix': corr_matrix, 
                        'lasso_matrix_r2': corr_matrix_R2,
                        'X_feats_std': X_feats_quad_std,
                        'X_feats':X_feats_quad,
                        'lasso_param_names': col_name_select,
                        'n_cells':N_cells}
                # lasso_results.append(clf.coef_)
                # lasso_results_ind.append(nonpred_ind)})
            
            with open(filename, 'wb') as filehandler:
                pickle.dump(savedict, filehandler)
    
    

    

        