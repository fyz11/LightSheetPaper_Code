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
    

# # # =============================================================================
# # # Construct and isolate the table entry indexes comprising the metadata of interest. 
# # # =============================================================================
# #     col_name_select = [ 'Embryo_Name',
# #                         'Frame', 
# #                         'Static_Epi_Contour_Quad_ID',
# #                         'Dynamic_Track_Quad_ID', # extremely close.
# #                         'Stage',
# #                         'Area', 
# #                         'Change_Rate_Area_Frac',
# #                         'Perimeter', 
# #                         'Perimeter_Proj',
# #                         'Change_Rate_Perimeter_Frac',
# #                         'Shape_Index',
# #                         'Shape_Index_Proj',
# #                         'Change_Rate_Shape_Index',
# #                         'Eccentricity', 
# #                         'Change_Rate_Eccentricity',
# #                         # 'Eccentricity_Angle_AVE',
# #                         'Eccentricity_Angle_vect_y',
# #                         'Eccentricity_Angle_vect_x',
# #                         # 'Eccentricity_Angle_rel_vect_radial',
# #                         # 'Eccentricity_Angle_rel_vect_theta',
# #                         'Cell_Neighbours_Track_ID',
# #                         'AVE_direction_VE_speed_3D_ref',
# #                         'AVE_direction_Epi_speed_3D_ref',
# #                         # 'AVE_direction_VE_speed_3D_real',
# #                         # 'AVE_direction_Epi_speed_3D_real',
# #                         # 'AVE_cum_dist_coord_3D_real',
# #                         'AVE_cum_dist_coord_3D_ref', # this is important 
# #                         'cell_epi_contour_dist_ratio', # this is important 
# #                         'demons_AVE_dir_VE', 
# #                         'demons_AVE-perp_dir_VE', 
# #                         'demons_Norm_dir_VE', 
# #                         'demons_AVE_dir_Epi',
# #                         'demons_AVE-perp_dir_Epi', 
# #                         'demons_Norm_dir_Epi', 
# #                         'gauss_curvature_VE',
# #                         'gauss_curvature_Epi',
# #                         'mean_curvature_VE',
# #                         'mean_curvature_Epi']
    
# #     print('=================')
# #     print(metrics_tables.columns)
# #     print('=================')
# #     col_id_select = np.hstack([np.arange(len(metrics_tables.columns))[metrics_tables.columns==col_name] for col_name in col_name_select])

# #     assert(len(col_name_select) == len(col_id_select))




# #     data_select = metrics_tables.iloc[:,col_id_select].copy()
# #     data_select.index = np.arange(len(data_select))
        
# #     # # translate the Cell_Neighbours_Track_ID one into one that just counts the number of neighbors if not nan
# #     # num_neighbors = np.hstack([ len(dd.split(':')) for dd in data_select['Cell_Neighbours_Track_ID'].values])
# #     # data_select['Cell_Neighbours_Track_ID'] = num_neighbors.astype(np.int)    
    
# #     """
# #     Use only the selected data to drop na values.  
# #     """
# #     # d = metrics_tables.columns == 'Diff_Cell_Neighbours_Track_ID'
# #     # metrics_tables = metrics_tables.loc[:,~d]
# #     # metrics_tables = metrics_tables.dropna() # this is critical -> drop all non full statistics.  -> we are dropping a huge amount? 
# #     # print(metrics_tables.shape)
# #     metrics_tables = data_select.copy()
# #     # metrics_tables = metrics_tables.astype(np.float)
# #     metrics_tables.replace([np.inf, -np.inf], np.nan, inplace=True)
# #     metrics_tables = metrics_tables.dropna() # should have dropped.... all the nan.... # why not?
# #     print(metrics_tables.shape)
    
# #     data = metrics_tables.values#.astype(np.float)
    
# #     TP = data[:,0].copy().astype(np.int)
# #     static_quad = data[:,1].copy().astype(np.int)
# #     dynamic_quad = data[:,2].copy().astype(np.int)
# #     stage = data[:,3].copy()
# #     metrics_data = data[:,4:].copy().astype(np.float)
    
# #     # cell_track_data_tables_data = cell_track_data_tables[:,:,col_id_select[2:]].copy()
# #     # # cell_track_data_tables_data[...,9] = np.hstack()
# #     # cell_track_data_tables_data = cell_track_data_tables_data.astype(np.float)
    
# # # # # =============================================================================
# # # # #     check the mutual info correlation vs time ?????  
# # # # # =============================================================================
# # # #     from sklearn.feature_selection import mutual_info_regression

# # # #     mi = mutual_info_regression(metrics_data, TP)
# # # #     mi_norm = mi/np.max(mi)

# # # #     fig, ax = plt.subplots(figsize=(10,5))
# # # #     ax.plot(mi, '-o')
# # # #     ax.set_xticks(np.arange(len(col_name_select))[:-2])
# # # #     ax.set_xticklabels(col_name_select[2:], rotation=45, horizontalalignment="right")
# # # #     plt.show()

# # # # # =============================================================================
# # # # #     have a little look at the infomration along cell tracks.  
# # # # # =============================================================================
# # # #     sum_nan = np.nansum(~np.isnan(cell_track_data_tables_data[...,0]), axis=1)
    
# # # #     cell_track_data_tables_data = cell_track_data_tables_data[sum_nan>0]
# # # #     sum_nan = sum_nan[sum_nan>0]
    
# # # #     sort_order = np.argsort(np.nansum(~np.isnan(cell_track_data_tables_data[...,0]), axis=1))[::-1]
    
# # # #     fig, ax = plt.subplots(figsize=(10,10))
# # # #     plt.title('area')
# # # #     ax.imshow(cell_track_data_tables_data[...,0][sort_order], cmap='Reds')
# # # #     ax.set_aspect('auto')
# # # #     plt.show()
    
# # # #     fig, ax = plt.subplots(figsize=(10,10))
# # # #     plt.title('eccentricity')
# # # #     ax.imshow(cell_track_data_tables_data[...,6][sort_order], cmap='Reds', vmin=1., vmax=2.)
# # # #     ax.set_aspect('auto')
# # # #     plt.show()
    
# # # #     # fig, ax = plt.subplots(figsize=(10,10))
# # # #     # ax.imshow(cell_track_data_tables_data[...,-4][sort_order], cmap='coolwarm')
# # # #     # ax.set_aspect('auto')
# # # #     # plt.show()
    
# # # #     fig, ax = plt.subplots(figsize=(10,10))
# # # #     plt.title(col_name_select[2:][np.argmax(mi)])
# # # #     ax.imshow(cell_track_data_tables_data[...,np.argmax(mi)][sort_order], cmap='Reds')
# # # #     ax.set_aspect('auto')
# # # #     plt.show()

# # # #     mean_curve0 = np.hstack([np.nanmean(metrics_data[:,0][TP==tp]) for tp in np.unique(TP)])
    
# # # #     plt.figure()
# # # #     plt.plot(TP, metrics_data[:,0], '.', alpha=0.01)
# # # #     plt.plot(np.unique(TP), mean_curve0)
# # # #     plt.show()

# # # #     mean_curve12 = np.hstack([np.nanmean(metrics_data[:,12][TP==tp]) for tp in np.unique(TP)])
    
# # # #     plt.figure()
# # # #     plt.plot(TP, metrics_data[:,12], '.', alpha=0.01)
# # # #     plt.plot(np.unique(TP), mean_curve12)
# # # #     plt.show()
    
# # # #     mean_curve11 = np.hstack([np.nanmean(metrics_data[:,11][TP==tp]) for tp in np.unique(TP)])
    
# # # #     plt.figure()
# # # #     plt.plot(TP, metrics_data[:,11], '.', alpha=0.01)
# # # #     plt.plot(np.unique(TP), mean_curve11)
# # # #     plt.show()

# # # # =============================================================================
# # # #     Check the co-correlation of features - no strong co-correlation of the unnormalised features. 
# # # # =============================================================================
    
# #     from sklearn.metrics.pairwise import pairwise_distances
    
    
# #     # wasn't able to drop. 
# #     metrics_data_rows = np.hstack([np.sum(np.isnan(metrics_data[row])) for row in np.arange(len(metrics_data))])
    
    
# #     # metrics_tables = metrics_tables.dropna() # this is critical
# #     # all_cell_feats = metrics_tables.loc[:,'area':].values.astype(np.float32) # interesting..... why is there nan in here? 

# #     # # these features are not fully correlated at all la. -> does the diff features confuse? => we have to do 1-(to check things.)
# #     # corr_matrix = pairwise_distances(all_cell_feats.T, metric='correlation', force_all_finite=False)
# #     corr_matrix = pairwise_distances(metrics_data.T, metric='correlation', force_all_finite=False) # v. strange. 
  
# #     plt.figure(figsize=(10,10))
# #     plt.imshow(1.-corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
# #     plt.yticks(np.arange(len(corr_matrix)),
# #                 col_name_select[4:])
# #     plt.xticks(np.arange(len(corr_matrix)),
# #                 col_name_select[4:], rotation=45, horizontalalignment='right')
# #     # plt.yticklabels(col_name_select[2:])
# #     plt.show()
    

# # # # =============================================================================
# # # #     1. Create the features, transform the features and create the labels of the data columns.  
# # # # =============================================================================

# #     col_name_select = ['Area', 
# #                         'Change_Rate_Area_Frac',
# #                         'Perimeter', 
# #                         'Change_Rate_Perimeter_Frac',
# #                         'Shape_Index',
# #                         'Change_Rate_Shape_Index',
# #                         'Eccentricity', 
# #                         'Change_Rate_Eccentricity',
# #                         # 'Eccentricity_Angle_AVE',
# #                         'Eccentricity_Angle_vect_y',
# #                         'Eccentricity_Angle_vect_x',
# #                         # 'Eccentricity_Angle_rel_vect_radial',
# #                         # 'Eccentricity_Angle_rel_vect_theta',
# #                         '#_Cell_Neighbours',
# #                         'AVE_direction_VE_speed_3D_ref',
# #                         'AVE_direction_Epi_speed_3D_ref',
# #                         # 'AVE_direction_VE_speed_3D_real',
# #                         # 'AVE_direction_Epi_speed_3D_real',
# #                         # 'AVE_cum_dist_coord_3D_real',
# #                         # 'AVE_cum_dist_coord_3D_ref',
# #                         # 'cell_epi_contour_dist_ratio',
# #                         'demons_AVE_dir_VE', 
# #                         'demons_AVE-perp_dir_VE', 
# #                         'demons_Norm_dir_VE', 
# #                         'demons_AVE_dir_Epi',
# #                         'demons_AVE-perp_dir_Epi', 
# #                         'demons_Norm_dir_Epi', 
# #                         'curvature_VE',
# #                         'curvature_Epi']
    
    
# #     col_name_select = np.hstack(col_name_select) # convert to numpy array for indexing. 

# #     # for this test don't use the eccentricity angle ...
# #     select_inds = np.arange(len(col_name_select))
# #     # select_inds = np.setdiff1d(select_inds, [8]) # cannot include discrete entities. ?

# #     from sklearn.preprocessing import StandardScaler, PowerTransformer
# #     import joblib  
    
# #     # all_embryos = metrics_tables['embryo'].values
# #     # all_embryo_track_ids = metrics_tables['track_id'].values # -> right we should also add this in 
# #     # all_track_ids = np.hstack([all_embryos[iii]+'_'+ str(all_embryo_track_ids[iii]).zfill(3) for iii in range(len(all_embryos))])
# #     # unique_track_ids = np.unique(all_track_ids) # get the unique cell track ids. -> we use this to split the feats for training.
    
    
# #     # # extraction of just the numeric features of the table. 
# #     # all_cell_feats = metrics_tables.loc[:,'area':].values.astype(np.float32)
# #     all_cell_feats = metrics_data.copy()
# #     all_cell_feats = all_cell_feats[:, select_inds].copy()
# #     col_name_select = col_name_select[select_inds].copy()
    
# #     # apply standard scaling transformations just for the cell feats like area etc. 
    
# #     # for some reason this does not centralizae the features? 
# #     X_feats, tforms = transform_feats(all_cell_feats, 
# #                                       apply_tform_cols=[0,2,4,6,8,9,10],
# #                                       noapplytform_cols=[1,3,5,7,11,12,13,14,15,16,17,18,19,20], 
# #                                       std_tformer=None, 
# #                                       pow_tformer=None, 
# #                                       X_max=None)
    
# #     print(X_feats.shape)


# # =============================================================================
# #     Feature transformed correlations. 
# # =============================================================================

#     X_feats, tforms = transform_feats(metrics_data.astype(np.float), 
#                                           apply_tform_cols=[0,1,2,3,4],
#                                           # noapplytform_cols=[5,6,7,8,9,10,11], 
#                                           noapplytform_cols=[5,6,7,8,9,10], 
#                                           std_tformer=None, 
#                                           pow_tformer=None, 
#                                           X_max=None)

#     # based on this we should further remove # discard particularly noisy cells!. 
#     mean_cols = np.nanmean(X_feats,axis=0)
#     std_cols = np.nanstd(X_feats,axis=0)
    
#     X_feats_extremal = np.abs(X_feats) > 4; 
#     sum_extremal = X_feats_extremal.sum(axis=1)
    
#     valid_X_feats = sum_extremal ==0 
#     X_feats = X_feats[valid_X_feats]
    
#     dynamic_quad_id = dynamic_quad_id[valid_X_feats]
#     static_quad_id = static_quad_id[valid_X_feats]
    
    
# # # =============================================================================
# # #     try minisom 
# # # =============================================================================

# #     from minisom import MiniSom # does seem to work .... -> and the proportions etc.    
# #     som = MiniSom(32, 32, X_feats.shape[1], sigma=5., learning_rate=0.5) # initialization of 6x6 SOM
# #     som.train(X_feats, 10000) # trains the SOM with 100 iterations
    
# #     lab_map = som.labels_map(X_feats, dynamic_quad_id.astype(np.int))
    
# #     lab_map_counts = np.zeros((32,32,32))
    
# #     for coord in set(lab_map.keys()):
# #         coord_count = lab_map[coord]
# #         for key in coord_count.keys():
# #             if np.abs(key)>=0:
# #                 lab_map_counts[coord[0], coord[1], key-1] = coord_count[key]
    
    
    
# #     plt.figure(figsize=(8, 8))
# #     wmap = {}
# #     im = 0
# #     for x, t in zip(X_feats, dynamic_quad_id):  # scatterplot
# #         w = som.winner(x)
# #         wmap[w] = im
# #         plt. text(w[0]+.5,  w[1]+.5,  str(int(t)),
# #                   color=plt.cm.rainbow( t / 32.), fontdict={'weight': 'bold',  'size': 11})
# #         im = im + 1
# #     plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
# #     # plt.savefig('resulting_images/som_digts.png')
# #     plt.show()
    
    
    
    
    
    
#     from py_pcha import PCHA

#     # dimensions = 15
#     # examples = 100
#     # X = np.random.random((dimensions, examples))
#     XC, S, C, SSE, varexpl = PCHA(X_feats.T, noc=4, delta=0.1) # ask for 3 prototypes. 
#     print(varexpl)


#     # ok. we need a tetrahedral to visualise ... 4 prototypes... 
#     from sklearn.decomposition import PCA
#     # project S.
#     convexhull = PCA(n_components=2)
#     YY = convexhull.fit_transform(S.T)
#     min_id = [np.argmin(np.linalg.norm(X_feats- XC[:,kk].T, axis=-1)) for kk in range(4)] 
#     # YY_prototype = convexhull.transform(XC.T)
#     # YY = convexhull.fit_transform(C)
    
#     fig, ax = plt.subplots()
#     plt.plot(YY[:,0], 
#              YY[:,1], '.')
#     plt.plot(YY[min_id,0], 
#              YY[min_id,1], 'ro')
#     ax.set_aspect(1)
#     plt.show()
    
    
#     fig, ax = plt.subplots()
#     plt.scatter(YY[:,0], 
#                 YY[:,1], c=X_feats[:,0], cmap='coolwarm')
#     ax.set_aspect(1)
#     plt.show()
    
#     fig, ax = plt.subplots()
#     plt.scatter(YY[:,0], 
#                 YY[:,1], c=X_feats[:,1], cmap='coolwarm')
#     ax.set_aspect(1)
#     plt.show()
    
#     fig, ax = plt.subplots()
#     plt.scatter(YY[:,0], 
#                 YY[:,1], c=X_feats[:,-5], cmap='coolwarm')#, vmin=-0.5,vmax=0.5)
#     ax.set_aspect(1)
#     plt.show()
    
    
    
#     fig, ax = plt.subplots()
#     plt.scatter(YY[:,0], 
#                 YY[:,1], c=dynamic_quad_id, cmap='coolwarm', s=1)#, vmin=-0.5,vmax=0.5)
#     ax.set_aspect(1)
#     plt.show()
    
    
#     fig, ax = plt.subplots()
#     plt.scatter(YY[:,0], 
#                 YY[:,1], c=static_quad_id, cmap='coolwarm', s=1)#, vmin=-0.5,vmax=0.5)
#     ax.set_aspect(1)
#     plt.show()
    
    
    
# # =============================================================================
# #     also try a umap to see if we can't separate more... 
# # =============================================================================
#     import umap 
    
#     umapfit = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.1)
#     u = umapfit.fit_transform(X_feats)

#     fig, ax = plt.subplots()
#     plt.plot(u[:,0], 
#              u[:,1], '.')
#     ax.set_aspect(1)
#     plt.show()
    
    
#     fig, ax = plt.subplots()
#     plt.scatter(u[:,0], 
#                 u[:,1], c=X_feats[:,-5], cmap='coolwarm')
#     ax.set_aspect(1)
#     plt.show()
    
#     fig, ax = plt.subplots()
#     plt.scatter(u[:,0], 
#                 u[:,1], c=X_feats[:,3], s=1, cmap='coolwarm')
#     ax.set_aspect(1)
#     plt.show()
    
#     fig, ax = plt.subplots()
#     plt.scatter(u[:,0], 
#                 u[:,1], c=X_feats[:,8], s=1, cmap='coolwarm')
#     ax.set_aspect(1)
#     plt.show()


#     fig, ax = plt.subplots()
#     plt.scatter(u[:,0], 
#                 u[:,1], c=dynamic_quad_id, cmap='coolwarm', s=1)#, vmin=-0.5,vmax=0.5)
#     ax.set_aspect(1)
#     plt.show()
    
#     fig, ax = plt.subplots()
#     plt.scatter(u[:,0], 
#                 u[:,1], c=static_quad_id, cmap='coolwarm', s=1)#, vmin=-0.5,vmax=0.5)
#     ax.set_aspect(1)
#     plt.show()
    
    
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
                                            noapplytform_cols=[6,7,8,9,10,11,12,13], # where there were natural signage.... then we put those variables here. 
                                            std_tformer=None, 
                                            pow_tformer=None, 
                                            X_max=None)
        
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

        
    """
    Correction or not doesn't make much difference -> will largely erase the differences. 
    """
    X_feats = all_feats_corrected.copy() # these are none corrrected? already ?
    
    
# # =============================================================================
# #     Gate and only keep the pre migration cells. 
# # =============================================================================
#     select_premigrate = stage_all == 'Pre-migration'
    
#     # apply this filter. 
#     X_feats = X_feats[select_premigrate].copy()
#     X_feats_raw = X_feats_raw[select_premigrate].copy()
#     frame_all = frame_all[select_premigrate].copy() # this need to be converted to TP=0 
#     embryo_all = embryo_all[select_premigrate].copy()
#     stage_all = stage_all[select_premigrate].copy()
#     static_quad_id_all = static_quad_id_all[select_premigrate]
#     dynamic_quad_id_all = dynamic_quad_id_all[select_premigrate]
#     epi_contour_dist_ratio_all = epi_contour_dist_ratio_all[select_premigrate]
    
#     row_id_all = row_id_all[select_premigrate].astype(np.int)
#     cell_id_all = cell_id_all[select_premigrate].astype(np.int)
#     track_id_all = track_id_all[select_premigrate].astype(np.int)
#     TP_all = TP_all[select_premigrate].astype(np.int)
    
# =============================================================================
#   based on these normalizations we start performing Lasso. ! 
# =============================================================================
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
    
    
    dynamic_quad_id_main = np.zeros_like(dynamic_quad_id_all)
    for qq_ii, qq in enumerate(quads_select):
        for qq_ss in qq:
            dynamic_quad_id_main[dynamic_quad_id_all==qq_ss] = qq_ii+1
    dynamic_quad_id_main = np.hstack(dynamic_quad_id_main)

# =============================================================================
#     Feature transformed PCA
# =============================================================================

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    pca_model = PCA(n_components=2)
    pca_model.fit(X_feats)
    
    print(pca_model.explained_variance_ratio_)
    # what do each component mainly refer to ? 
    
    plt.figure()
    plt.title('Component 1')
    plt.plot(pca_model.components_[0])
    plt.xticks(np.arange(X_feats.shape[1]), col_name_select, rotation=45, ha='right')
    
    
    plt.figure()
    plt.title('Component 2')
    plt.plot(pca_model.components_[1])
    plt.xticks(np.arange(X_feats.shape[1]), col_name_select, rotation=45, ha='right')
    
    
    
    plt.figure()
    plt.title('Component 1')
    plt.plot(np.abs(pca_model.components_[0]))
    plt.xticks(np.arange(X_feats.shape[1]), col_name_select, rotation=45, ha='right')
    
    
    plt.figure()
    plt.title('Component 2')
    plt.plot(np.abs(pca_model.components_[1]))
    plt.xticks(np.arange(X_feats.shape[1]), col_name_select, rotation=45, ha='right')
    
    
    """
    PCA model
    """
    pca_proj = pca_model.transform(X_feats)
    
    import umap 
    
    # umap_proj = 
    noneighborindex = np.setdiff1d(np.arange(len(col_name_select)), 5) # this is now the 5th one!. 
    # 15 worked well here.
    fit = umap.UMAP(random_state=0, 
                    n_neighbors=200, # 100 seems ideal or 200 ....
                    init = pca_proj,  # use PCA to initialise. 
                    # local_connectivity=1, 
                    # output_metric='euclidean',
                    output_metric='correlation', # correlation metric seems to work better? ---> does this allow 2 to show up without inclusion of other params? 
                    spread=1.) # a big problem is effect of 1 variable... 
    # fit = PCA(n_components=2)
    umap_proj = fit.fit_transform(X_feats[:,noneighborindex]) # what if we don't do this ? 
    
    # high number of clusters actually resulted in better representation  ? ---> how does this effect automatic hclustering? 
    kmean_cluster = KMeans(n_clusters=100, random_state=0) # is 100 clusters the best? 
    pca_proj_cluster = kmean_cluster.fit_predict(pca_proj) # how this change if we do this on the original feats? 
    # pca_proj_cluster = kmean_cluster.fit_predict(X_feats) # how this change if we do this on the original feats? 
    
    
    cluster_ids = np.unique(pca_proj_cluster)
    cluster_ids_members = [np.arange(len(pca_proj_cluster))[pca_proj_cluster==cc] for cc in cluster_ids]
    cluster_ids_centroids = np.vstack([np.mean(pca_proj[cc], axis=0) for cc in cluster_ids_members])
    
    
    umap_proj_cluster = kmean_cluster.fit_predict(umap_proj) # how this change if we do this on the original feats? 
    # pca_proj_cluster = kmean_cluster.fit_predict(X_feats) # how this change if we do this on the original feats? 
    
    umap_cluster_ids = np.unique(umap_proj_cluster)
    umap_cluster_ids_members = [np.arange(len(umap_proj_cluster))[umap_proj_cluster==cc] for cc in umap_cluster_ids]
    umap_cluster_ids_centroids = np.vstack([np.mean(umap_proj[cc], axis=0) for cc in umap_cluster_ids_members])
    
# =============================================================================
#     Export all the associated statistics into one .mat file so it is easy to investigate plotting without re-running everything 
# =============================================================================

    import scipy.io as spio 
    
    # savefile = os.path.join(saveplotfolder, 'pca_umap_analysis_data.mat')
    # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors200_CorrelationMetric.mat')
    # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_CorrelationMetric.mat') # this is similar to 200!
    # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_EuclideanMetric_NoEpiKappa.mat') # this is similar to 200!
    # savefile = os.path.join(saveplotfolder, '2021-01-26_pca_umap_analysis_data-Neighbors100_CorrelationMetric_PreMigrationOnly.mat') # this is similar to 200!
    # savefile = os.path.join(saveplotfolder, '2021-01-26_pca_umap_analysis_data-Neighbors100_CorrelationMetric_Both.mat')
    savefile = os.path.join(saveplotfolder, '2021-01-26_pca_umap_analysis_data-Neighbors100_CorrelationMetric_Both2.mat')
    
    spio.savemat(savefile, 
                  {'X_feats': X_feats,
                   'X_feats_nobatch': X_feats_nobatch,
                  'X_feats_raw': X_feats_raw,
                  'PCA_feats': pca_proj,
                  'UMAP_feats': umap_proj,
                  'Frame':frame_all, 
                  'Embryo': embryo_all,
                  'Stage': stage_all, 
                  'Static_quad': static_quad_id_all, 
                  'Dynamic_quad' : dynamic_quad_id_all, 
                  'Dynamic_quad_main': dynamic_quad_id_main,
                  'epi_contour_dist': epi_contour_dist_ratio_all, 
                  'feat_names': col_name_select,
                  'PCA_kmeans_cluster': pca_proj_cluster,
                  'PCA_kmeans_cluster_id': cluster_ids, 
                  'PCA_cluster_centroids': cluster_ids_centroids, 
                  'UMAP_kmeans_cluster': umap_proj_cluster,
                  'UMAP_kmeans_cluster_id': umap_cluster_ids, 
                  'UMAP_cluster_centroids': umap_cluster_ids_centroids,
                  'row_id': row_id_all, 
                  'TP':TP_all, 
                  'cell_id':cell_id_all, 
                  'track_id':track_id_all, 
                  'keep_rows_filter': keep_rows,
                  'outliers_gate': outliers_gate})
    
    
# =============================================================================
#     Note: we need to save also the kmeans cluster model so we can use this for the static!!!!! 
# =============================================================================

    # save_kmeans_modelfile = os.path.join(saveplotfolder, '2021-01-26_pca_umap_analysis_data-Neighbors100_CorrelationMetric_Both_kMeansModel.joblib')
    save_kmeans_modelfile = os.path.join(saveplotfolder, '2021-01-26_pca_umap_analysis_data-Neighbors100_CorrelationMetric_Both_kMeansModel2.joblib')
    
    from joblib import dump, load
    dump(kmean_cluster, save_kmeans_modelfile) 
    
    # # wrap all the meta into one dict. 
    # metadict_id_array = {'embryo_all':embryo_all, 
    #                      'TP_all':TP_all, 
    #                      'cell_id_all': cell_id_all, 
    #                      'row_id_all': row_id_all, 
    #                      'track_id_all': track_id_all}
    
# =============================================================================
#     detect the number of unique embryos
# =============================================================================
    uniq_embryos = np.unique(embryo_all)
    
    print(len(uniq_embryos), ' embryos ')
    print('====')
    print(uniq_embryos)
    
    
    from scipy.stats import mode
    cluster_ids_majority_valid = []
    cluster_ids_majority_valid_num = [] 
    cluster_ids_majority_main = []
    cluster_ids_prop = []
    cluster_ids_prop_stage = []
    cluster_ids_prop_embryo = []
    
    for cc in cluster_ids_members:
        member = dynamic_quad_id_all[cc]
        member = member[member>0]
        cluster_ids_majority_valid_num.append(len(member))
        cluster_ids_majority_valid.append(mode(member)[0])
        
        member = dynamic_quad_id_main[cc]
        member = member[member>0]
        cluster_ids_majority_main.append(mode(member)[0])
        
        prop = np.hstack([np.sum(member==group) for group in np.arange(1,9)])
        cluster_ids_prop.append(prop)
        
        member_stage = stage_all[cc]
        member_stage_int = np.zeros(len(member_stage))
        member_stage_int[member_stage == 'Migration to Boundary'] = 1
        prop = np.hstack([np.sum(member_stage_int==group) for group in np.arange(2)])
        cluster_ids_prop_stage.append(prop)
        
        member_embryo = embryo_all[cc]
        # member_embryo_int = np.zeros(len(member_embryo))
        prop_embryo = []
        for emb_ii, emb in enumerate(uniq_embryos):
            prop_embryo.append(np.sum(member_embryo==emb))
        cluster_ids_prop_embryo.append(prop_embryo)
            
        
        
    cluster_ids_majority_valid = np.hstack(cluster_ids_majority_valid)
    cluster_ids_majority_valid_num = np.hstack(cluster_ids_majority_valid_num)
    cluster_ids_majority_main = np.hstack(cluster_ids_majority_main)
    cluster_ids_prop = np.vstack(cluster_ids_prop)
    cluster_ids_prop_stage = np.vstack(cluster_ids_prop_stage)
    cluster_ids_prop_embryo = np.vstack(cluster_ids_prop_embryo)
    # np.hstack([mode(dynamic_quad_id_all[cc])[0] for cc in cluster_ids_members])
    
    
    plt.figure()
    plt.plot(pca_proj[:,0], 
              pca_proj[:,1],'.')
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(15,15))
    # plt.plot(pca_proj[:,0], 
    #           pca_proj[:,1],'.')
    plt.scatter(pca_proj[:,0], 
                pca_proj[:,1], c=pca_proj_cluster, cmap='hsv')
    # plt.plot(cluster_ids_centroids[:,0], 
    #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    for c_ii in np.arange(len(cluster_ids_majority_valid)):
        ax.text(cluster_ids_centroids[c_ii,0], 
                cluster_ids_centroids[c_ii,1], 
                str(int(cluster_ids_majority_valid[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
    
    ax.set_aspect(1)
    # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(15,15))
    # plt.plot(pca_proj[:,0], 
    #           pca_proj[:,1],'.')
    plt.scatter(pca_proj[:,0], 
                pca_proj[:,1], c=pca_proj_cluster, cmap='hsv')
    # plt.plot(cluster_ids_centroids[:,0], 
    #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    for c_ii in np.arange(len(cluster_ids_majority_valid)):
        ax.text(cluster_ids_centroids[c_ii,0], 
                cluster_ids_centroids[c_ii,1], 
                str(int(cluster_ids_majority_main[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
    ax.set_aspect(1)
    # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning_and_majority_vote.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
# =============================================================================
#     Separate the pie chart out a bit. 
# =============================================================================
    import seaborn as sns
    from skimage.color import rgb2hsv, hsv2rgb
    
    def adjust_color_lightness(rgb, factor):
        hls = rgb2hsv(rgb[None,None,:])
        hls = np.squeeze(hls)
        l = max(min(hls[2] * factor, 1.0), 0.0)
        # hls[1] = l 
        hls[2] = l 
        rgb = hsv2rgb(hls[None,None,:])
        rgb = np.squeeze(rgb)
        # return rgb2hex(int(r * 255), int(g * 255), int(b * 255))
        return rgb
    
    def darken_color(rgb, factor=0.1):
        return adjust_color_lightness(rgb, 1 - factor)

    
    # pie_colors_A = sns.color_palette('magma_r', n_colors=8)[::2] # to be consistent. #
    pie_colors_A = sns.color_palette('coolwarm', n_colors=8)[::2] # to be consistent. 
    pie_colors_A = sns.color_palette('coolwarm', n_colors=4)
    pie_colors_A = np.vstack(pie_colors_A)
    pie_colors_A = pie_colors_A[[0,1,2,3]]
    # # pie_colors_P = sns.color_palette('coolwarm', n_colors=8)
    # pie_colors_P = np.vstack([darken_color(c, factor=0.1) for c in pie_colors_A])
    pie_colors_P = sns.color_palette('magma_r', n_colors=8)[::2]
    pie_colors_P = np.vstack(pie_colors_P)
    # dist_colors = sns.color_palette('magma_r', 8)[::2]
    # AP_colors = sns.color_palette('coolwarm',3)
    pie_colors = np.vstack([[pie_colors_A[iii], pie_colors_P[iii]] for iii in np.arange(len(pie_colors_A))])
    
    
# =============================================================================
#     Plotting the PCA overlaid with proportion occupied by each of the 8 gated regions. 
# =============================================================================
    max_val = 5
    # show the proportion:
    fig, ax = plt.subplots(figsize=(15,15))
    # plt.plot(pca_proj[:,0], 
    #           pca_proj[:,1],'.')
    # ax.scatter(pca_proj[:,0], 
    #             pca_proj[:,1], c=pca_proj_cluster, cmap='hsv', alpha=.5)
    ax.scatter(pca_proj[:,0], 
                pca_proj[:,1], c=pca_proj_cluster, cmap='Greys_r')
    # plt.plot(cluster_ids_centroids[:,0], 
    #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    for c_ii in np.arange(len(cluster_ids_majority_valid)):
        ax.text(cluster_ids_centroids[c_ii,0], 
                cluster_ids_centroids[c_ii,1], 
                str(int(cluster_ids_majority_main[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
        draw_pie(cluster_ids_prop[c_ii], 
                  cluster_ids_centroids[c_ii,0], 
                  cluster_ids_centroids[c_ii,1],
                  size=1200, # size is the size of scatter marker. 
                  colors=pie_colors,
                  ax=ax)
        # draw_pie_xy(cluster_ids_prop[c_ii], 
        #              cluster_ids_centroids[c_ii,1], 
        #              cluster_ids_centroids[c_ii,0], 
        #              size=.01,
        #              startangle=None,
        #              colors=None, 
        #              ax=ax)
    plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
    plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning_and_majority_vote_w_pie.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    
# =============================================================================
#     Plotting the PCA overlaid with proportion occupied by each of the 8 gated regions + Stage.  
# =============================================================================
    
    max_val = 5

    base_num = np.hstack([np.sum(stage_all=='Pre-migration'), 
                          np.sum(stage_all=='Migration to Boundary')])
    base_num_prop = base_num / float(np.sum(base_num))
    pie_colors_stage = sns.color_palette('Spectral', 2)

    # show the proportion:
    fig, ax = plt.subplots(figsize=(15,15))
    # plt.plot(pca_proj[:,0], 
    #           pca_proj[:,1],'.')
    # ax.scatter(pca_proj[:,0], 
    #             pca_proj[:,1], c=pca_proj_cluster, cmap='hsv', alpha=.5)
    ax.scatter(pca_proj[:,0], 
                pca_proj[:,1], c=pca_proj_cluster, cmap='Greys_r')
    # plt.plot(cluster_ids_centroids[:,0], 
    #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    for c_ii in np.arange(len(cluster_ids_majority_valid)):
        
        # observed_stage = cluster_ids_prop_stage[c_ii]
        # observed_stage_prop = observed_stage / float(np.sum(observed_stage))
        
        # change_frac = (observed_stage_prop - base_num_prop)/ base_num_prop
        # # change_frac = change_frac / (base_num_prop[::-1]) * 50
        # # multiply by the original 50 % 
        # change_prop_ratio = np.abs(change_frac)
        ax.text(cluster_ids_centroids[c_ii,0], 
                cluster_ids_centroids[c_ii,1], 
                str(int(cluster_ids_majority_main[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
        draw_pie(cluster_ids_prop_stage[c_ii], 
                  cluster_ids_centroids[c_ii,0], 
                  cluster_ids_centroids[c_ii,1],
                  size=1200, # size is the size of scatter marker. 
                  colors=pie_colors_stage,
                  ax=ax)
        
    plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
    plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_majority_vote_w_pie_Stage.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    
# =============================================================================
# =============================================================================
# =============================================================================
# # #     Do the above plots but for the umap clusters as debug 
# =============================================================================
# =============================================================================
# =============================================================================

    uniq_embryos = np.unique(embryo_all)
    
    print(len(uniq_embryos), ' embryos ')
    print('====')
    print(uniq_embryos)
    
    
    
    cluster_ids = umap_cluster_ids.copy()
    cluster_ids_members = list(umap_cluster_ids_members)
    cluster_ids_centroids = umap_cluster_ids_centroids.copy()
    
    
    from scipy.stats import mode
    cluster_ids_majority_valid = []
    cluster_ids_majority_valid_num = [] 
    cluster_ids_majority_main = []
    cluster_ids_prop = []
    cluster_ids_prop_stage = []
    cluster_ids_prop_embryo = []
    
    for cc in cluster_ids_members:
        member = dynamic_quad_id_all[cc]
        member = member[member>0]
        cluster_ids_majority_valid_num.append(len(member))
        cluster_ids_majority_valid.append(mode(member)[0])
        
        member = dynamic_quad_id_main[cc]
        member = member[member>0]
        cluster_ids_majority_main.append(mode(member)[0])
        
        prop = np.hstack([np.sum(member==group) for group in np.arange(1,9)])
        cluster_ids_prop.append(prop)
        
        member_stage = stage_all[cc]
        member_stage_int = np.zeros(len(member_stage))
        member_stage_int[member_stage == 'Migration to Boundary'] = 1
        prop = np.hstack([np.sum(member_stage_int==group) for group in np.arange(2)])
        cluster_ids_prop_stage.append(prop)
        
        member_embryo = embryo_all[cc]
        # member_embryo_int = np.zeros(len(member_embryo))
        prop_embryo = []
        for emb_ii, emb in enumerate(uniq_embryos):
            prop_embryo.append(np.sum(member_embryo==emb))
        cluster_ids_prop_embryo.append(prop_embryo)
            
        
        
    cluster_ids_majority_valid = np.hstack(cluster_ids_majority_valid)
    cluster_ids_majority_valid_num = np.hstack(cluster_ids_majority_valid_num)
    cluster_ids_majority_main = np.hstack(cluster_ids_majority_main)
    cluster_ids_prop = np.vstack(cluster_ids_prop)
    cluster_ids_prop_stage = np.vstack(cluster_ids_prop_stage)
    cluster_ids_prop_embryo = np.vstack(cluster_ids_prop_embryo)
    # np.hstack([mode(dynamic_quad_id_all[cc])[0] for cc in cluster_ids_members])
    
    
    plt.figure()
    plt.plot(umap_proj[:,0], 
              umap_proj[:,1],'.')
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(15,15))
    # plt.plot(pca_proj[:,0], 
    #           pca_proj[:,1],'.')
    plt.scatter(umap_proj[:,0], 
                umap_proj[:,1], c=umap_proj_cluster, cmap='hsv')
    # plt.plot(cluster_ids_centroids[:,0], 
    #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    for c_ii in np.arange(len(cluster_ids_majority_valid)):
        ax.text(cluster_ids_centroids[c_ii,0], 
                cluster_ids_centroids[c_ii,1], 
                str(int(cluster_ids_majority_valid[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
    
    ax.set_aspect(1)
    # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(15,15))
    # plt.plot(pca_proj[:,0], 
    #           pca_proj[:,1],'.')
    plt.scatter(umap_proj[:,0], 
                umap_proj[:,1], c=umap_proj_cluster, cmap='hsv')
    # plt.plot(cluster_ids_centroids[:,0], 
    #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    for c_ii in np.arange(len(cluster_ids_majority_valid)):
        ax.text(cluster_ids_centroids[c_ii,0], 
                cluster_ids_centroids[c_ii,1], 
                str(int(cluster_ids_majority_main[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
    ax.set_aspect(1)
    # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning_and_majority_vote.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
# =============================================================================
#     Separate the pie chart out a bit. 
# =============================================================================
    import seaborn as sns
    from skimage.color import rgb2hsv, hsv2rgb
    
    def adjust_color_lightness(rgb, factor):
        hls = rgb2hsv(rgb[None,None,:])
        hls = np.squeeze(hls)
        l = max(min(hls[2] * factor, 1.0), 0.0)
        # hls[1] = l 
        hls[2] = l 
        rgb = hsv2rgb(hls[None,None,:])
        rgb = np.squeeze(rgb)
        # return rgb2hex(int(r * 255), int(g * 255), int(b * 255))
        return rgb
    
    def darken_color(rgb, factor=0.1):
        return adjust_color_lightness(rgb, 1 - factor)

    
    # pie_colors_A = sns.color_palette('magma_r', n_colors=8)[::2] # to be consistent. #
    pie_colors_A = sns.color_palette('coolwarm', n_colors=8)[::2] # to be consistent. 
    pie_colors_A = sns.color_palette('coolwarm', n_colors=4)
    pie_colors_A = np.vstack(pie_colors_A)
    pie_colors_A = pie_colors_A[[0,1,2,3]]
    # # pie_colors_P = sns.color_palette('coolwarm', n_colors=8)
    # pie_colors_P = np.vstack([darken_color(c, factor=0.1) for c in pie_colors_A])
    pie_colors_P = sns.color_palette('magma_r', n_colors=8)[::2]
    pie_colors_P = np.vstack(pie_colors_P)
    # dist_colors = sns.color_palette('magma_r', 8)[::2]
    # AP_colors = sns.color_palette('coolwarm',3)
    pie_colors = np.vstack([[pie_colors_A[iii], pie_colors_P[iii]] for iii in np.arange(len(pie_colors_A))])
    
    
# =============================================================================
#     Plotting the PCA overlaid with proportion occupied by each of the 8 gated regions. 
# =============================================================================
    max_val = 5
    # show the proportion:
    fig, ax = plt.subplots(figsize=(15,15))
    # plt.plot(pca_proj[:,0], 
    #           pca_proj[:,1],'.')
    # ax.scatter(pca_proj[:,0], 
    #             pca_proj[:,1], c=pca_proj_cluster, cmap='hsv', alpha=.5)
    ax.scatter(umap_proj[:,0], 
               umap_proj[:,1], c=umap_proj_cluster, cmap='Greys_r')
    # plt.plot(cluster_ids_centroids[:,0], 
    #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    for c_ii in np.arange(len(cluster_ids_majority_valid)):
        ax.text(cluster_ids_centroids[c_ii,0], 
                cluster_ids_centroids[c_ii,1], 
                str(int(cluster_ids_majority_main[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
        draw_pie(cluster_ids_prop[c_ii], 
                  cluster_ids_centroids[c_ii,0], 
                  cluster_ids_centroids[c_ii,1],
                  size=1200, # size is the size of scatter marker. 
                  colors=pie_colors,
                  ax=ax)
        # draw_pie_xy(cluster_ids_prop[c_ii], 
        #              cluster_ids_centroids[c_ii,1], 
        #              cluster_ids_centroids[c_ii,0], 
        #              size=.01,
        #              startangle=None,
        #              colors=None, 
        #              ax=ax)
    # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
    # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
    center_umap = np.mean(umap_proj, axis=0)
    sd_umap = np.std(umap_proj, axis=0)
        
    plt.vlines(center_umap[0], -3*sd_umap[1] + center_umap[0], 3*sd_umap[1] + center_umap[0], linestyles='dashed', lw=3, zorder=1)
    plt.hlines(center_umap[1], -3*sd_umap[0] + center_umap[1], 3*sd_umap[0] + center_umap[1], linestyles='dashed', lw=3, zorder=1)
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning_and_majority_vote_w_pie.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    
# =============================================================================
#     Plotting the PCA overlaid with proportion occupied by each of the 8 gated regions + Stage.  
# =============================================================================
    
    max_val = 5

    base_num = np.hstack([np.sum(stage_all=='Pre-migration'), 
                          np.sum(stage_all=='Migration to Boundary')])
    base_num_prop = base_num / float(np.sum(base_num))
    pie_colors_stage = sns.color_palette('Spectral', 2)

    # show the proportion:
    fig, ax = plt.subplots(figsize=(15,15))
    # plt.plot(pca_proj[:,0], 
    #           pca_proj[:,1],'.')
    # ax.scatter(pca_proj[:,0], 
    #             pca_proj[:,1], c=pca_proj_cluster, cmap='hsv', alpha=.5)
    ax.scatter(umap_proj[:,0], 
               umap_proj[:,1], c=umap_proj_cluster, cmap='Greys_r')
    # plt.plot(cluster_ids_centroids[:,0], 
    #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    for c_ii in np.arange(len(cluster_ids_majority_valid)):
        
        # observed_stage = cluster_ids_prop_stage[c_ii]
        # observed_stage_prop = observed_stage / float(np.sum(observed_stage))
        
        # change_frac = (observed_stage_prop - base_num_prop)/ base_num_prop
        # # change_frac = change_frac / (base_num_prop[::-1]) * 50
        # # multiply by the original 50 % 
        # change_prop_ratio = np.abs(change_frac)
        ax.text(cluster_ids_centroids[c_ii,0], 
                cluster_ids_centroids[c_ii,1], 
                str(int(cluster_ids_majority_main[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
        draw_pie(cluster_ids_prop_stage[c_ii], 
                  cluster_ids_centroids[c_ii,0], 
                  cluster_ids_centroids[c_ii,1],
                  size=1200, # size is the size of scatter marker. 
                  colors=pie_colors_stage,
                  ax=ax)
        
    center_umap = np.mean(umap_proj, axis=0)
    sd_umap = np.std(umap_proj, axis=0)
        
    plt.vlines(center_umap[0], -3*sd_umap[1] + center_umap[0], 3*sd_umap[1] + center_umap[0], linestyles='dashed', lw=3, zorder=1)
    plt.hlines(center_umap[1], -3*sd_umap[0] + center_umap[1], 3*sd_umap[0] + center_umap[1], linestyles='dashed', lw=3, zorder=1)
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_majority_vote_w_pie_Stage.png'), dpi=300, bbox_inches='tight')
    plt.show()




    
    
    
    
# # =============================================================================
# #     Control plot of the representation of each embryo in each region of the PCA.   
# # =============================================================================
    
    
    
    
    
#     cluster_ids_prop_embryo
#     max_val = 5

#     base_num = np.hstack([np.sum(embryo_all==emb_) for emb_ in uniq_embryos])
    
    
    
#     # base_num_prop = base_num / float(np.sum(base_num))
#     pie_colors_stage = sns.color_palette('Spectral_r', cluster_ids_prop_embryo.shape[1])
    
#     # base pie chart.
#     fig, ax = plt.subplots(figsize=(10,10))
#     draw_pie(base_num, 
#              0, 
#              0,
#              size=2400, # size is the size of scatter marker. 
#              colors=pie_colors_stage,
#              ax=ax)
#     plt.grid('off')
#     plt.axis('off')
#     plt.show()
    
#     plt.figure(figsize=(10,10))
#     sns.palplot(pie_colors_stage)
#     plt.show()

#     # show the proportion:
#     fig, ax = plt.subplots(figsize=(15,15))
#     # plt.plot(pca_proj[:,0], 
#     #           pca_proj[:,1],'.')
#     # ax.scatter(pca_proj[:,0], 
#     #             pca_proj[:,1], c=pca_proj_cluster, cmap='hsv', alpha=.5)
#     ax.scatter(pca_proj[:,0], 
#                 pca_proj[:,1], c=pca_proj_cluster, cmap='Greys_r')
#     # plt.plot(cluster_ids_centroids[:,0], 
#     #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
#     for c_ii in np.arange(len(cluster_ids_majority_valid)):
        
#         # observed_stage = cluster_ids_prop_stage[c_ii]
#         # observed_stage_prop = observed_stage / float(np.sum(observed_stage))
        
#         # change_frac = (observed_stage_prop - base_num_prop)/ base_num_prop
#         # # change_frac = change_frac / (base_num_prop[::-1]) * 50
#         # # multiply by the original 50 % 
#         # change_prop_ratio = np.abs(change_frac)
#         ax.text(cluster_ids_centroids[c_ii,0], 
#                 cluster_ids_centroids[c_ii,1], 
#                 str(int(cluster_ids_majority_main[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
#         draw_pie(cluster_ids_prop_embryo[c_ii], 
#                   cluster_ids_centroids[c_ii,0], 
#                   cluster_ids_centroids[c_ii,1],
#                   size=1200, # size is the size of scatter marker. 
#                   colors=pie_colors_stage,
#                   ax=ax)
        
#     plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
#     plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
#     ax.set_aspect(1)
#     plt.grid('off')
#     plt.axis('off')
#     # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_majority_vote_w_pie_Embryo.png'), dpi=300, bbox_inches='tight')
#     plt.show()
    
    
    
# # # =============================================================================
# # #     Supplementary plot of every variable used. on the same axes. 
# # # =============================================================================
     
# #     # # generate all of these... sorting points by extremals. 
# #     # col_name_select = ['Area', 
# #     #                    'Perimeter', 
# #     #                    'Shape_Index', 
# #     #                    'Eccentricity', 
# #     #                    'Thickness',
# #     #                    '#Neighbors',
# #     #                    'A-P_speed_VE', 
# #     #                    'A-P_speed_Epi', 
# #     #                    'cum_A-P_VE_speed', 
# #     #                     # 'epi_contour_dist', # can't have this. 
# #     #                     'Normal_VE-Epi', # doesn seem to do anything? 
# #     #                    # 'Normal VE', 
# #     #                    # 'Normal Epi', 
# #     #                    'Mean Curvature_VE', 
# #     #                    'Mean Curvature_Epi',
# #     #                    'Gauss Curvature_VE', 
# #     #                    'Gauss Curvature_Epi']
    
# #     """
# #     plots one the extremal behaviour in the PCA regions. 
# #     """
    
# #     for feat_ii in np.arange(len(col_name_select)):
    
# #         fig, ax = plt.subplots(figsize=(15,15))
# #         plt.title('%s'%(col_name_select[feat_ii]))
        
# #         z = np.abs(X_feats[:,feat_ii]); idx = z.argsort()
        
# #         plt.scatter(pca_proj[idx,0], 
# #                     pca_proj[idx,1], c=X_feats[idx, feat_ii], cmap='coolwarm', vmin=-3,vmax=3)
# #         plt.plot(cluster_ids_centroids[:,0], 
# #                   cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
        
# #         plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #         plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #         ax.set_aspect(1)
# #         plt.grid('off')
# #         plt.axis('off')
# #         plt.show()
        
        
# #     """
# #     plots one the mean behaviour in the clustered voronoi regions. ( these would be more interpretable ) 
# #     """
# #     for feat_ii in np.arange(len(col_name_select)):
    
# #         fig, ax = plt.subplots(figsize=(15,15))
# #         plt.title('%s'%(col_name_select[feat_ii]))
        
# #         discrete_feats = np.zeros(len(X_feats))
# #         for cc in cluster_ids_members:
# #             cc_feats = X_feats[cc,feat_ii].copy()
# #             cc_feats_mean = np.nanmedian(cc_feats)
# #             discrete_feats[cc] = cc_feats_mean.copy()
# #         # z = np.abs(X_feats[:,feat_ii]); idx = z.argsort()
        
# #         # plt.scatter(pca_proj[idx,0], 
# #                     # pca_proj[idx,1], c=X_feats[idx, feat_ii], cmap='coolwarm', vmin=-3,vmax=3)
                    
# #         # max_scale = np.quantile(np.abs(discrete_feats), 0.75)
# #         max_scale = np.std(np.abs(X_feats[:,feat_ii]))*4
# #         plt.scatter(pca_proj[:,0], 
# #                     pca_proj[:,1], c=discrete_feats, cmap='coolwarm', vmin=-max_scale,vmax=max_scale)
# #         plt.plot(cluster_ids_centroids[:,0], 
# #                   cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
        
# #         plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #         plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #         ax.set_aspect(1)
# #         plt.grid('off')
# #         plt.axis('off')
# #         plt.savefig(os.path.join(saveplotfolder, 'PCA_%s_mean_regions.png' %(col_name_select[feat_ii])), dpi=300, bbox_inches='tight')
# #         plt.show()
        
        
# # # =============================================================================
# # #         additionally we map the meta statistics. namely epi contour distance to get a relative sense. + the times. 
# # # =============================================================================

# #     fig, ax = plt.subplots(figsize=(15,15))
# #     plt.title('Time')
    
# #     discrete_feats = np.zeros(len(X_feats))
# #     for cc in cluster_ids_members:
# #         cc_feats = frame_all[cc].copy()
# #         cc_feats_mean = np.nanmean(cc_feats) # better to use the mean time. 
# #         discrete_feats[cc] = cc_feats_mean.copy()
    
# #     # max_scale = np.quantile(np.abs(discrete_feats), 0.75)
# #     max_scale = np.std(np.abs(frame_all))*4
# #     plt.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1], c=discrete_feats, cmap='coolwarm', vmin=-max_scale,vmax=max_scale)
# #     plt.plot(cluster_ids_centroids[:,0], 
# #               cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    
# #     plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #     plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #     ax.set_aspect(1)
# #     plt.grid('off')
# #     plt.axis('off')
# #     plt.savefig(os.path.join(saveplotfolder, 'PCA_Time_mean_regions.png'), dpi=300, bbox_inches='tight')
# #     plt.show()
    
    
# #     fig, ax = plt.subplots(figsize=(15,15))
# #     plt.title('Epi_Contour_Dist_ratio')
    
# #     discrete_feats = np.zeros(len(X_feats))
# #     for cc in cluster_ids_members:
# #         cc_feats = epi_contour_dist_ratio_all[cc].copy()
# #         cc_feats_mean = np.nanmean(cc_feats) # better to use the mean time. 
# #         discrete_feats[cc] = cc_feats_mean.copy()
    
# #     # max_scale = np.quantile(np.abs(discrete_feats), 0.75)
# #     max_scale = np.std(np.abs(frame_all))*4
# #     plt.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1], c=discrete_feats, cmap='coolwarm', vmin=-1,vmax=1)
# #     plt.plot(cluster_ids_centroids[:,0], 
# #               cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    
# #     plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #     plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #     ax.set_aspect(1)
# #     plt.grid('off')
# #     plt.axis('off')
# #     # plt.savefig(os.path.join(saveplotfolder, 'PCA_Epi-ExE_distance_mean_regions.png'), dpi=300, bbox_inches='tight')
# #     plt.show()
    
    
    
# #     fig, ax = plt.subplots(figsize=(15,15))
# #     plt.title('Number of Points')
    
# #     discrete_feats = np.zeros(len(X_feats))
# #     for cc in cluster_ids_members:
# #         # cc_feats = epi_contour_dist_ratio_all[cc].copy()
# #         # cc_feats_mean = np.nanmedian(cc_feats) # better to use the mean time. 
# #         discrete_feats[cc] = len(cc)
    
# #     # # max_scale = np.quantile(np.abs(discrete_feats), 0.75)
# #     # max_scale = np.std(np.abs(frame_all))*4
# #     plt.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1], c=discrete_feats, cmap='coolwarm', vmin=0,vmax=discrete_feats.max())
# #     plt.plot(cluster_ids_centroids[:,0], 
# #               cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    
# #     plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #     plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #     ax.set_aspect(1)
# #     plt.grid('off')
# #     plt.axis('off')
# #     plt.savefig(os.path.join(saveplotfolder, 'PCA_density_num_mean_regions.png'), dpi=300, bbox_inches='tight')
# #     plt.show()
    
        
        
# # # =============================================================================
# # #    Understand the general movement of the gated group of cells over time. !      
# # # =============================================================================
        
# #     from scipy.stats import gaussian_kde # why is this different? 
# #     from skimage.filters import threshold_otsu
    
# #     grid_pca = np.meshgrid(np.linspace(-5,5, 500), np.linspace(-5,5,500))
# #     grid_pca = np.dstack(grid_pca)
# #     # discretize time
# #     time_discrete = np.linspace(-3,6, 9+1) # only up to 4 hrs. 
    
# #     # set up the time colors. 
# #     TP_start = np.arange(len(time_discrete))[time_discrete ==0][0]
# #     time_discrete_colors = np.vstack([sns.color_palette('coolwarm', 2*(TP_start+1)+1)])[:(TP_start)]
# #     time_discrete_colors = np.vstack([time_discrete_colors, 
# #                                       sns.color_palette('magma_r', len(time_discrete)-TP_start)])
    
# #     import seaborn as sns
    
# #     plt.figure()
# #     sns.palplot(time_discrete_colors)
# #     plt.show()
    
    
# #     # org_day_colors = sns.color_palette('coolwarm', 8) 
    
# #     all_times_all_pts_line = []
# #     all_times_all_pts_line_I = [] # records the density threshold used. 
# #     all_times_all_contours = []
# #     all_times_all_occupancy = [] # we need to know the number of cells that went into this. -> if this is bad. then... =_=
    
    
# #     for quad_id in np.arange(1,9):
    
# #         gate_prox_epi = dynamic_quad_id_main == quad_id
# #         times_all_gate = frame_all[gate_prox_epi]
# #         pca_proj_gate = pca_proj[gate_prox_epi]
        
# #         times_all_gate_bins = np.digitize(times_all_gate*5/60., time_discrete ).astype(np.int) # convert to hours. 
    
    
# #         times_all_pts_line = []
# #         times_all_pts_line_I = [] # records the density threshold used. 
# #         times_all_contours = []
# #         times_all_occupancy = []
        
        
# #         # =============================================================================
# #         #   plot the evolution. 
# #         # =============================================================================
    
# #         # fig, ax = plt.subplots(figsize=(15,15))
# #         # ax.scatter(pca_proj[:,0], 
# #         #             pca_proj[:,1],alpha=0.1,
# #         #             c=pca_proj_cluster, cmap='Greys_r') # always plot onto this. 
        
# #         # easier to design our own contourf and kde estimate like for organoids!. 
# #         # compute the kde estimate... 
# #         for iii in np.arange(len(time_discrete)):
            
# #             fig, ax = plt.subplots(figsize=(15,15))
# #             ax.scatter(pca_proj[:,0], 
# #                         pca_proj[:,1],alpha=0.1)
            
# #             select_plot = times_all_gate_bins == iii
# #             times_all_occupancy.append(select_plot.sum())
# #             # Fit a kde
# #             xy = pca_proj_gate[select_plot].T
# #             z = gaussian_kde(xy)(pca_proj_gate.T) 
# #             thresh = np.mean(z) + 1*np.std(z)
# #             # thresh = threshold_otsu(z)
            
# #             # z_grid, peak_stats, z_grid_cont = fit_gaussian_kde_pts((xy.T), grid=grid_pca, thresh=None, thresh_sigma=3) # create a dense version ... 
# #             z_grid, peak_stats, z_grid_cont = fit_gaussian_kde_pts((xy.T), 
# #                                                                    grid=grid_pca, 
# #                                                                    thresh=thresh, thresh_sigma=3) # create a dense version ... 
# #             print(iii, xy.shape)
# #             # eval the KDE on this domain. 
# #             # z = gaussian_kde(xy)(pca_proj.T) # why is this so inaccurate? 
# #             z = gaussian_kde(xy)(pca_proj_gate.T) 
# #             # z = gaussian_kde(xy, 'scott')(xy)
# #             max_z_id = np.argmax(z)
# #             # now this works.! 
# #             thresh = np.mean(z) + 1*np.std(z)
            
# #             # modal point -> this is absolutely necessary due to so much outliers. 
# #             # pt_max_xy = (xy.T)[max_z_id]  
# #             pt_max_xy = (pca_proj_gate)[max_z_id]  
# #             idx = z.argsort() # plot by density size. 
# #             # ax.scatter(pca_proj[idx,0], 
# #                         # pca_proj[idx,1], c=z[idx], s=100, cmap='coolwarm')
# #             ax.scatter(pca_proj_gate[idx,0], 
# #                         pca_proj_gate[idx,1], c=z[idx], s=100, cmap='coolwarm')
# #             # ax.scatter((xy.T)[idx,0], 
# #             #             (xy.T)[idx,1], c=z[idx],  s=10, cmap='coolwarm')
# #             ax.plot(pt_max_xy[0], pt_max_xy[1], 'ko', ms=20, zorder=1000)
# #             ax.plot(pca_proj_gate[z>=thresh, 0], 
# #                     pca_proj_gate[z>=thresh, 1], 'g.', zorder=1000)
# #             # ax.plot((xy.T)[idx,0][-20:], 
# #             #         (xy.T)[idx,1][-20:], 'go', ms=10, zorder=1000)
# #             ax.plot(peak_stats[:,0], 
# #                     peak_stats[:,1], 'bo', zorder=10000)
            
# #             for cnt in z_grid_cont:
# #                 ax.plot(cnt[:,0], cnt[:,1], 'k-')
                
# #             # if more than 1 then weight it.... 
# #             weighted_mean = np.nansum(peak_stats[:,:2]*peak_stats[:,2][:,None], axis=0) / float(np.nansum(peak_stats[:,2]))
            
# #             # times_all_pts_line.append(np.nanmean(peak_stats[:,:2], axis=0))
# #             times_all_pts_line.append(weighted_mean)
# #             # times_all_pts_line.append(peak_stats[:,:2])
            
# #             times_all_pts_line_I.append(thresh)
# #             times_all_contours.append(z_grid_cont)
            
# #         plt.grid('off')
# #         plt.axis('off')    
# #         plt.show()           
            
# #         times_all_pts_line = np.vstack(times_all_pts_line)
# #         times_all_pts_line_I = np.hstack(times_all_pts_line_I)
# #         # times_all_pts_line = []
# #         # times_all_pts_line_I = [] # records the density threshold used. 
# #         # times_all_contours = []
        
# #         all_times_all_pts_line.append(times_all_pts_line)
# #         all_times_all_pts_line_I.append(times_all_pts_line_I) # records the density threshold used. 
# #         all_times_all_contours.append(times_all_contours)
# #         all_times_all_occupancy.append(np.hstack(times_all_occupancy))
    
    
# #     all_times_all_occupancy = np.array(all_times_all_occupancy)
    
# #     fig, ax = plt.subplots(figsize=(15,15))
# #     ax.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1],alpha=0.1,
# #                 c=pca_proj_cluster, cmap='Greys_r')
# #     for line_ii, line in enumerate(all_times_all_pts_line):
# #         # ax.plot(line[:,0], 
# #         #         line[:,1], 'ro-', zorder=100)
# #         plt.text(line[0,0], 
# #                  line[0,1], str(line_ii+1), fontsize=24, color='g', zorder=100)
        
        
# #     # color each point by time. 
# #     time_colors = sns.color_palette('magma_r', n_colors=len(all_times_all_pts_line[1]))
# #     plt.plot(all_times_all_pts_line[1][:,0], 
# #               all_times_all_pts_line[1][:,1], 'bo', ms=20, zorder=100, lw=3)
    
# #     # easier to plot by line segments... 
# #     for jj in np.arange(len(all_times_all_pts_line[1])-1):
# #         plt.plot([all_times_all_pts_line[1][jj,0], all_times_all_pts_line[1][jj+1,0]], 
# #                  [all_times_all_pts_line[1][jj,1], all_times_all_pts_line[1][jj+1,1]], 
# #                  '-', color=time_colors[jj], zorder=100, lw=3)
    
# #     # plt.scatter(all_times_all_pts_line[1][:,0], 
# #     #             all_times_all_pts_line[1][:,1], 
# #     #             c=time_colors, zorder=1000, s=25)
# #     # # times_all_pts_line = np.vstack(times_all_pts_line)
# #     # ax.plot(times_all_pts_line[:,0], 
# #     #         times_all_pts_line[:,1], 'ro-')
# #     ax.set_aspect(1)
# #     plt.show()


# #     plt.figure()
# #     for all_time_occ in all_times_all_occupancy:
# #         plt.plot(time_discrete, all_time_occ)
# #     plt.show()
    
    
    
# # # =============================================================================
# # #     Proper plotting over time and saving. 
# # # =============================================================================
    

# #     # all_times_all_pts_line.append(times_all_pts_line)
# #     #     all_times_all_pts_line_I.append(times_all_pts_line_I) # records the density threshold used. 
# #     #     all_times_all_contours.append(times_all_contours)
# #     #     all_times_all_occupancy.append(np.hstack(times_all_occupancy))

# #     for reg_ii in np.arange(len(all_times_all_pts_line))[:]:
    
# #         fig, ax = plt.subplots(figsize=(15,15))
# #         plt.title('Reg_'+str(reg_ii+1))
# #         ax.scatter(pca_proj[:,0], 
# #                     pca_proj[:,1],alpha=0.01,
# #                     c=pca_proj_cluster, cmap='Greys_r')
# #         plt.plot(cluster_ids_centroids[:,0], 
# #                  cluster_ids_centroids[:,1], 'k.', zorder=100, ms=10)

# #         contour_reg_ii = all_times_all_contours[reg_ii]

# #         for tp in np.arange(len(contour_reg_ii)):    
# #             contour_tp = contour_reg_ii[tp]
# #             for cnt in contour_tp:
# #                 ax.plot(cnt[:,0], 
# #                         cnt[:,1], color=time_discrete_colors[tp], lw=3, zorder=100)
# #         # for tp in np.arange(len(contour_reg_ii)-1):   
# #         #     plt.plot([all_times_all_pts_line[reg_ii][tp,0], all_times_all_pts_line[reg_ii][tp+1,0]], 
# #         #              [all_times_all_pts_line[reg_ii][tp,1], all_times_all_pts_line[reg_ii][tp+1,1]], 
# #         #              '-', color=time_discrete_colors[tp], zorder=100, lw=3)
# #         ax.set_aspect(1)
# #         plt.axis('off')
# #         plt.grid('off')
# #         plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #         plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #         plt.savefig(os.path.join(saveplotfolder, 'PCA_density_contour-Reg_%s.png' %(str(reg_ii+1).zfill(3))), dpi=300, bbox_inches='tight')
# #         plt.show()
        
        
# #         fig, ax = plt.subplots(figsize=(15,15))
# #         plt.title('Reg_'+str(reg_ii+1))
# #         ax.scatter(pca_proj[:,0], 
# #                     pca_proj[:,1],alpha=0.01,
# #                     c=pca_proj_cluster, cmap='Greys_r')
# #         plt.plot(cluster_ids_centroids[:,0], 
# #                  cluster_ids_centroids[:,1], 'k.', zorder=100, ms=10)

# #         # contour_reg_ii = all_times_all_contours[reg_ii]
# #         # for tp in np.arange(len(contour_reg_ii)):    
# #         #     contour_tp = contour_reg_ii[tp]
# #         #     for cnt in contour_tp:
# #         #         ax.plot(cnt[:,0], 
# #         #                 cnt[:,1], color=time_discrete_colors[tp], lw=3, zorder=100)
# #         for tp in np.arange(len(contour_reg_ii)-1):   
# #             plt.plot([all_times_all_pts_line[reg_ii][tp,0], all_times_all_pts_line[reg_ii][tp+1,0]], 
# #                      [all_times_all_pts_line[reg_ii][tp,1], all_times_all_pts_line[reg_ii][tp+1,1]], 
# #                      '-', color=time_discrete_colors[tp], zorder=100, lw=10)
            
# #         plt.plot(all_times_all_pts_line[reg_ii][:,0], 
# #                  all_times_all_pts_line[reg_ii][:,1], 'ro', ms=10, zorder=100)
        
# #         # add the cross axes. 
# #         ax.set_aspect(1)
# #         plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #         plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
# #         plt.axis('off')
# #         plt.grid('off')
# #         plt.savefig(os.path.join(saveplotfolder, 'PCA_density_track-Reg_%s.png' %(str(reg_ii+1).zfill(3))), dpi=300, bbox_inches='tight')
# #         plt.show()
        
# #         # for line_ii, line in enumerate(all_times_all_pts_line):

# #         #     ax.plot(line[:,0], 
# #         #             line[:,1], 'ro-', zorder=100)
# #         #     # plt.text(line[0,0], 
# #         #     #          line[0,1], str(line_ii+1), fontsize=24, color='g', zorder=100)
        
# #         # plt.show()
    




    

#     # import pandas as pd 
#     # kde_table = pd.DataFrame(np.hstack([pca_proj_gate, times_all_gate_bins[:,None]]), 
#     #                           columns=['x', 'y', 'time'])
    
#     # kde_table['time'] = kde_table['time'].astype(np.int)
    
#     # fig, ax = plt.subplots(figsize=(15,15))
#     # ax.scatter(pca_proj[:,0], 
#     #             pca_proj[:,1],alpha=0.1)
#     # sns.kdeplot(x=kde_table["x"], 
#     #             y=kde_table["y"], 
#     #             hue=kde_table['time'].values, ax=ax,
#     #             bw_adjust=1,
#     #             levels=[.7,.8,.9], fill=False)
#     # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
#     # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
#     # ax.set_aspect(1)
#     # plt.grid('off')
#     # plt.axis('off')
#     # plt.show()

    
#     # fig, ax = plt.subplots(figsize=(15,15))
#     # # plt.plot(pca_proj[:,0], 
#     # #           pca_proj[:,1],'.')
#     # # ax.scatter(pca_proj[:,0], 
#     # #             pca_proj[:,1], c=pca_proj_cluster, cmap='hsv', alpha=.5)
#     # ax.scatter(pca_proj[:,0], 
#     #             pca_proj[:,1], c=pca_proj_cluster, cmap='Greys_r')
#     # # plt.plot(cluster_ids_centroids[:,0], 
#     # #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
#     # for c_ii in np.arange(len(cluster_ids_majority_valid)):
#     #     ax.text(cluster_ids_centroids[c_ii,0], 
#     #             cluster_ids_centroids[c_ii,1], 
#     #             str(int(cluster_ids_majority_main[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
#     #     draw_pie(cluster_ids_prop[c_ii], 
#     #               cluster_ids_centroids[c_ii,0], 
#     #               cluster_ids_centroids[c_ii,1],
#     #               size=1200, # size is the size of scatter marker. 
#     #               colors=pie_colors,
#     #               ax=ax)
#     #     # draw_pie_xy(cluster_ids_prop[c_ii], 
#     #     #              cluster_ids_centroids[c_ii,1], 
#     #     #              cluster_ids_centroids[c_ii,0], 
#     #     #              size=.01,
#     #     #              startangle=None,
#     #     #              colors=None, 
#     #     #              ax=ax)
#     # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
#     # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
#     # ax.set_aspect(1)
#     # plt.grid('off')
#     # plt.axis('off')
#     # plt.show()
    
    
    
    
    
    
    
#     # fig, ax = plt.subplots(figsize=(15,15))
#     # plt.title('eccentricity')
#     # # plt.plot(pca_proj[:,0], 
#     # #           pca_proj[:,1],'.')
#     # plt.scatter(pca_proj[:,0], 
#     #             pca_proj[:,1], c=X_feats[:,3], cmap='coolwarm')
#     # # plt.plot(cluster_ids_centroids[:,0], 
#     # #          cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
    
#     # # for c_ii in np.arange(len(cluster_ids_majority)):
#     # #     ax.text(cluster_ids_centroids[c_ii,0], 
#     # #             cluster_ids_centroids[c_ii,1], 
#     # #             str(int(cluster_ids_majority[c_ii])), fontsize=24, color='k', ha='center', va='center', zorder=1000)
    
#     # ax.set_aspect(1)
#     # plt.grid('off')
#     # plt.axis('off')
#     # plt.show()
    
    
    
    
# #     plt.figure()
# #     plt.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1], c=X_feats[:,0], cmap='coolwarm')
# #     plt.show()
    
# #     plt.figure()
# #     plt.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1], c=X_feats[:,-1], cmap='coolwarm', s=1, vmin=-1, vmax=1)
# #     plt.show()
    
# #     plt.figure()
# #     plt.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1], c=X_feats[:,-2], cmap='coolwarm', s=1, vmin=-1, vmax=1)
# #     plt.show()
    
# #     plt.figure()
# #     plt.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1], c=X_feats[:,3], cmap='coolwarm', s=1)
# #     plt.show()
    
# #     plt.figure()
# #     plt.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1], c=X_feats[:,-4], cmap='coolwarm', s=1)
# #     # plt.xlim([-5,5])
# #     # plt.ylim([-5,5])
# #     plt.show()
    
    
# #     plt.figure()
# #     plt.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1], c=X_feats[:,8], cmap='coolwarm', s=1)
# #     # plt.xlim([-5,5])
# #     # plt.ylim([-5,5])
# #     plt.show()
    
# #     plt.figure()
# #     plt.scatter(pca_proj[:,0], 
# #                 pca_proj[:,1], c=X_feats[:,7], cmap='coolwarm', s=1)
# #     # plt.xlim([-5,5])
# #     # plt.ylim([-5,5])
# #     plt.show()
    
    
    
# #     # plt.figure()
# #     # plt.scatter(pca_proj[:,0], 
# #     #             pca_proj[:,1], c=X_feats[:,-4], cmap='coolwarm', s=1)
# #     # plt.show()


# #     # col_name_select = ['Area', 
# #     #                    'Perimeter', 
# #     #                    'Shape_Index', 
# #     #                    'Eccentricity', 
# #     #                    '#Neighbors',
# #     #                    'A-P_speed_VE', 
# #     #                    'A-P_speed_Epi', 
# #     #                    'cum_A-P_VE_speed', 
# #     #                    'epi_contour_dist',
# #     #                    'Normal_VE-Epi', 
# #     #                    'Curvature_VE', 
# #     #                    'Curvature_Epi']



    
# # #     """
# # #     Creating subset indices for further analysis / testing of feature relationships.  
# # #     """
# # #     quads_select = [[8,1,2],
# # #                     [4,5,6],
# # #                     [16,9,10], 
# # #                     [12,13,14],
# # #                     [24,17,18], 
# # #                     [20,21,22],
# # #                     [32,25,26],
# # #                     [28,29,30]]
# # #     # quads_select = np.arange(32)+1
    
# # #     # quads_select = [[1],
# # #     #                 [4,5,6],
# # #     #                 [16,9,10], 
# # #     #                 [12,13,14],
# # #     #                 [24,17,18], 
# # #     #                 [20,21,22],
# # #     #                 [32,25,26]]
    
# # #     quads_select_names = ['Epi-Distal-Anterior', 
# # #                           'Epi-Distal-Posterior',
# # #                           'Epi-Prox-Anterior', 
# # #                           'Epi-Prox-Posterior',
# # #                           'ExE-Distal-Anterior',
# # #                           'ExE-Distal-Posterior',
# # #                           'ExE-Prox-Anterior',
# # #                           'ExE-Prox-Posterior']
    
# # #     # quads_select_names = np.hstack([str(qq) for qq in quads_select])
    
    
# # #     stages = ['Pre-migration', 
# # #               'Migration to Boundary']
    
# # #     """
# # #     Iterative subsetting
# # #     """
    
# # # #     # X_feats = X_feats[all_embryos=='455-Emb3-000-polar'] 
# # # #     # X_feats = X_feats[all_embryos=='491--000-polar']
# # # #     # X_feats = X_feats[all_embryos=='505--120-polar']
# # # #     # X_feats = X_feats[all_embryos=='864--180-polar']
    
# # # #     # separate by quads. 
# # # #     # anterior -> 1,2,8
    
# # # #     # anterior_select = np.logical_or(np.logical_or(static_quad==1, static_quad==2) , static_quad==8)
# # # #     # # quads 9,10,16
# # # #     anterior_select = np.logical_or(np.logical_or(static_quad==9, static_quad==10) , static_quad==16)
# # # #     X_feats = X_feats[anterior_select].copy()
    
# # # #     # posterior_select = np.logical_or(np.logical_or(static_quad==4, static_quad==5) , static_quad==6)
    
# # # #     # posterior_select = np.logical_or(np.logical_or(static_quad==12, static_quad==13) , static_quad==14)
# # # #     # X_feats = X_feats[posterior_select].copy()
# # # #     print(X_feats.shape)
    
# # # #     # # just in case also get the maximums of the standardscaling columns
# # # #     # tforms_std_scales = np.nanmax(np.abs(X_feats[:,[0,2,4,6,8]]), axis=0)
    
# # # #     # X_feats = X_feats/np.nanmax(np.abs(X_feats),axis=0)[None,:]
    
# # # #     # X_feats[:,[0,2,4,6,8]] = X_feats[:,[0,2,4,6,8]] / (tforms_std_scales[None,:])
# # # #     # X_feats = X_feats[:,:-1] # we drop speed. in its entirety. 
    
# # # #     # from sklearn.preprocessing import StandardScaler, PowerTransformer
# # # #     # # for some reason this bit here makes it better? -> seems to somehow don't allow the discrete feature to completely stand out now..... 
# # # #     # X_feats = StandardScaler().fit_transform(PowerTransformer().fit_transform(all_cell_feats)) # this doesn't change it, but the power transformer is able to suppress somewhat the effect of outliers!. 
    
# # # # #     # The below is hugely important in influencing the 'weighting' of features!. 
# # # #     # X_feats = X_feats / np.nanmax(np.abs(X_feats), axis=0)[None,:] # in some ways don't like this... 
# # # #     # X_feats = StandardScaler().fit_transform(all_cell_feats)
# # # #     # X_feats = all_cell_feats.copy()
# # # # #     # X_feats = X_feats[:,:8]

# # # # # =============================================================================
# # # # #   Understanding the correlation between the variables overall in the embryo as a whole.  
# # # # # =============================================================================
# # # #     metric_col_names = np.hstack(col_name_select)[select_inds]
    
# # # #     corr_matrix = pairwise_distances(X_feats.T, metric='correlation', force_all_finite=False)
    
# # # #     plt.figure()
# # # #     plt.title('Corr Matrix, Normalised Feats')
# # # #     plt.imshow(1.-corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
# # # #     plt.xticks(np.arange(len(corr_matrix)), metric_col_names, rotation=45, horizontalalignment='right')
# # # #     plt.yticks(np.arange(len(corr_matrix)), metric_col_names, rotation=0, horizontalalignment='right')
# # # #     plt.show()

# # #     from sklearn.linear_model import Lasso
# # #     import networkx as nx 
# # #     import netgraph
# # #     import pickle
# # #     import itertools
# # #     from scipy.stats import pearsonr
# # #     from sklearn.metrics import r2_score
# # #     from sklearn.preprocessing import StandardScaler

# # #     # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-PreMig-0.01'
# # #     # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-Mig-0.01'
# # #     # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-PreMig-0.05'
# # #     # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-Mig-0.05'
# # #     # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-PreMig-Auto'

# # #     # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-8quads-dynamic'
# # #     # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-32quads-static'
# # #     # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-32quads-dynamic'
    
# # #     # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-rescale_coef'
# # #     # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2-rescale_coef'
# # #     # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-rescale_coef_post'
# # #     # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2-rescale_coef_post'
    
# # #     # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-with_coef-std'
    
# # #     # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_ref'
# # #     # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_ref'
    
# # #     # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_ref'
# # #     analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_ref_nocum'
    
# # #     # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_realxyz'
# # #     # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_realxyz'
# # #     # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_realxyz-nocum'
# # #     # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_realxyz-nocum'
    
    
# # #     # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-stdscale_coef'
# # #     # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2-stdscale_coef'
# # #     # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2'
# # #     #create the new folder. 
# # #     fio.mkdir(analysis_model_save_folder)
    
# # #     lasso_alpha = .05
    
# # #     for stage_name_ii in stages[:]:
        
# # #         print(stage_name_ii)
# # #         for quad_ii in np.arange(len(quads_select))[:]:
            
# # #             quad_ii_regions = quads_select[quad_ii]
# # #             quads_ii_regions_names = quads_select_names[quad_ii]
            
# # #             data_select = stage == stage_name_ii
# # #             # quad_select_mask = static_quad == quad_ii_regions[0]
# # #             # quad_select_mask = dynamic_quad == quad_ii_regions
# # #             quad_select_mask = dynamic_quad == quad_ii_regions[0]
# # #             for jjj in np.arange(len(quad_ii_regions)-1):
# # #                 # quad_select_mask = np.logical_or(quad_select_mask, 
# # #                                                     # static_quad == quad_ii_regions[jjj+1])
# # #                 quad_select_mask = np.logical_or(quad_select_mask, 
# # #                                                     dynamic_quad == quad_ii_regions[jjj+1])
            
# # #             # print(quad_select_mask.sum()) # seems to check out. 
# # #             # np.logical_or(np.logical_or(static_quad==1, static_quad==2) , static_quad==8)
# # #             data_select = np.logical_and(data_select, 
# # #                                          quad_select_mask)
            
# # #             # apply the masking. 
# # #             X_feats_quad = X_feats[data_select].copy()
# # #             X_feats_quad_std = X_feats_quad.std(axis=0)
# # #             # X_feats_quad_std = X_feats_quad.std(axis=0)
# # #             # X_feats_quad = X_feats_quad / (X_feats_quad.std(axis=0)[None,:]) # this is to allow the coefficient importance comparison.
# # #             # X_feats_quad = StandardScaler().fit_transform(X_feats_quad)
# # #             # X_feats_quad = X_feats[quad_select_mask].copy()
            
# # #             # corr_matrix = pairwise_distances(X_feats_quad.T, 
# # #             #                                  metric='correlation', 
# # #             #                                  force_all_finite=False)
            
# # #             # double check the nature of this calculation -> ascribe pvalue...             
# # #             print(quads_ii_regions_names, quad_ii_regions)
# # #             print(X_feats_quad.shape)
# # #             N_cells = len(X_feats_quad)
            
# # #             corr_matrix = np.zeros((X_feats_quad.shape[1], X_feats_quad.shape[1]))
# # #             corr_matrix_R2 = np.zeros((X_feats_quad.shape[1], X_feats_quad.shape[1])) # equivalent of the pvalue.
            
# # #             n_features = X_feats_quad.shape[1]
            
# # #             for pred_ind in np.arange(n_features)[:]:
                
# # #                 # =============================================================================
# # #                 #   I think yeah... we need to do some choosing of parameters.                  
# # #                 # =============================================================================
# # #                 # clf = Lasso(alpha=lasso_alpha, normalize=False, 
# # #                 #                 fit_intercept=True) # check the regularization amount.
# # #                 # clf1 = LassoLars(alpha=lasso_alpha, normalize=False, 
# # #                 #                 fit_intercept=True)
                
# # #                 nonpred_ind = np.setdiff1d(np.arange(n_features), pred_ind)
                
# # #                 # do an over feat. 
# # #                 YY = X_feats_quad[:, pred_ind].copy()
# # #                 XX = X_feats_quad[:, nonpred_ind].copy()
# # #                 # clf.fit(XX, YY[:,None])
# # #                 # clf1.fit(XX,YY[:,None])
                
# # #                 mod = sm.OLS(YY, XX)
# # #                 # res = mod.fit()
# # #                 res = mod.fit_regularized(method='sqrt_lasso', L1_wt=1.)
# # #                 corr_matrix[pred_ind, nonpred_ind] = res.params #* X_feats_quad_std[nonpred_ind]
                
# # #                 YY_pred = res.predict(XX)
# # #                 r2 = r2_score(YY,YY_pred, multioutput='uniform_average')
                
# # #                 corr_matrix_R2[pred_ind,:] = r2
                
# # #             # corr_matrix = np.zeros((X_feats_quad.shape[1], X_feats_quad.shape[1]))
# # #             # corr_matrix_pval = np.zeros((X_feats_quad.shape[1], X_feats_quad.shape[1])) # this will act as our filter for reliable signals. 
            
# # #             # for ii, jj in itertools.combinations(np.arange(X_feats_quad.shape[1]), 2):
# # #             #     X_ii = X_feats_quad[:,ii]
# # #             #     X_jj = X_feats_quad[:,jj]
                
# # #             #     r, pval = pearsonr(X_ii, X_jj)
                
# # #             #     corr_matrix[ii,jj] = r 
# # #             #     corr_matrix[jj,ii] = r
# # #             #     corr_matrix_pval[ii,jj] = pval; 
# # #             #     corr_matrix_pval[jj,ii] = pval
                
# # #             #     corr_matrix[ii,ii] = 1
# # #             #     corr_matrix[jj,jj] = 1
                        
# # #             # plt.figure()
# # #             # plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
# # #             # plt.show()
            
            
# # #             # # """
# # #             # # Construction of the Lasso prediction models
# # #             # # """
# # #             # # # initialise and build a graph network? 
# # #             # # G_weights = nx.DiGraph() # just those that have weights. 
# # #             # # n_features = X_feats.shape[1]
# # #             # # G_weights.add_nodes_from(np.arange(n_features)) # this is to hack the graph to always plot at the same position irrespective if there is an edge. 
            
# # #             # # save_analysis_name = stage_name_ii+'_'+quads_ii_regions_names+'_'+'-'.join([str(q_ii) for q_ii in quad_ii_regions])
# # #             save_analysis_name = stage_name_ii+'_'+quads_ii_regions_names
            
# # #             fig, ax = plt.subplots(figsize=(15,15))
# # #             plt.title('Coef Matrix, Normalised Feats %s, %s' %( stage_name_ii, quad_ii_regions), fontsize=32)
# # #             plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
# # #             plt.xticks(np.arange(len(corr_matrix)), col_name_select, rotation=45, horizontalalignment='right', fontsize=24)
# # #             plt.yticks(np.arange(len(corr_matrix)), col_name_select, rotation=0, horizontalalignment='right', fontsize=24)
# # #             fig.savefig(os.path.join(analysis_model_save_folder, save_analysis_name+'.svg'), bbox_inches='tight')
# # #             plt.show()
            
# # #             # pickle the current network connections etc. ready for comparison 
# # #             filename = os.path.join(analysis_model_save_folder, save_analysis_name+'.p')
# # #             savedict = {'lasso_matrix': corr_matrix, 
# # #                         'lasso_matrix_r2': corr_matrix_R2,
# # #                         'X_feats_std': X_feats_quad_std,
# # #                         'X_feats':X_feats_quad,
# # #                         'lasso_param_names': col_name_select,
# # #                         'n_cells':N_cells}
# # #                 # lasso_results.append(clf.coef_)
# # #                 # lasso_results_ind.append(nonpred_ind)})
            
# # #             with open(filename, 'wb') as filehandler:
# # #                 pickle.dump(savedict, filehandler)
    
    

    

        