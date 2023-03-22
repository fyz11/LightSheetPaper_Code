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
    
def convexhull_pts(pts, ptslabels, uniq_labels):

    from scipy.spatial import ConvexHull    
    
    lab_hull = []
    
    for lab in uniq_labels:
        pts_lab = pts[ptslabels==lab]
        
        hull = ConvexHull(pts_lab)
        hull_pts = pts_lab[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        lab_hull.append(hull_pts)
        
    return lab_hull
    
    
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
    import seaborn as sns 
    import pandas as pd 
    
# =============================================================================
#     create the mega savefolder to save all the plots for editting. 
# =============================================================================

    # we should also save the models used. i.e. the kmeans clusters etc. 
    # saveplotfolder = '2021-05-19_PCA_analysis'
    # fio.mkdir(saveplotfolder)
    # saveplotfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-19_PCA_analysis'
    # # saveplotfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-24_UMAP_analysis'
    
    saveplotfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-19_PCA_analysis'
    
# =============================================================================
#     Export all the associated statistics into one .mat file so it is easy to investigate plotting without re-running everything 
# =============================================================================

    import scipy.io as spio 
    import scipy.stats as stats # load the stats module. 
    
    # savefile = os.path.join(saveplotfolder, 'pca_umap_analysis_data.mat')
    # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors200_CorrelationMetric.mat')
    savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_CorrelationMetric.mat') # this version seems most specific -> use this version!. 
    # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_EuclideanMetric.mat') # weird connectivity shape if we include it ... 
    # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_EuclideanMetric_NoEpiKappa.mat') # this is similar to 200!
    
    # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors500_EuclideanMetric.mat') # this is similar to 200!
    # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_CorrelationMetric.mat') # this is similar to 200!
    # savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_CorrelationMetric.mat')
    # spio.savemat(savefile, 
    #              {'X_feats': X_feats,
    #               'Frame':frame_all, 
    #               'Embryo': embryo_all,
    #               'Stage': stage_all, 
    #               'Static_quad': static_quad_id_all, 
    #               'Dynamic_quad' : dynamic_quad_id_all, 
    #               'epi_contour_dist': epi_contour_dist_ratio_all, 
    #               'feat_names': col_name_select,
    #               'PCA_kmeans_cluster': pca_proj_cluster,
    #               'PCA_kmeans_cluster_id': cluster_ids, 
    #               'cluster_centroids': cluster_ids_centroids})
    saveobj = spio.loadmat(savefile)
    
    # saveplotfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-24_UMAP_Correlation_analysis'
    saveplotfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-24_UMAP_Correlation_analysis'
    fio.mkdir(saveplotfolder)
    
    
    X_feats = saveobj['X_feats']
    X_feats_raw = saveobj['X_feats_raw']
    frame_all = saveobj['Frame'].ravel()
    embryo_all = np.hstack(saveobj['Embryo'].ravel())
    stage_all = saveobj['Stage'].ravel()
    static_quad_id_all = saveobj['Static_quad'].ravel()
    dynamic_quad_id_all = saveobj['Dynamic_quad'].ravel()
    dynamic_quad_id_main = saveobj['Dynamic_quad_main'].ravel()
    epi_contour_dist = saveobj['epi_contour_dist'].ravel() # we should be able to use this to further segment out sector 1. 
    col_name_select = saveobj['feat_names'].ravel()
    
    # # parsing the preclustered data - here we use the UMAP. 
    # pca_proj = saveobj['PCA_feats']
    # pca_proj_cluster = saveobj['PCA_kmeans_cluster'].ravel()
    # cluster_ids = saveobj['PCA_kmeans_cluster_id'].ravel()
    # cluster_ids_centroids = saveobj['PCA_cluster_centroids']
    pca_proj = saveobj['UMAP_feats']
    pca_proj_cluster = saveobj['UMAP_kmeans_cluster'].ravel()
    cluster_ids = saveobj['UMAP_kmeans_cluster_id'].ravel()
    cluster_ids_centroids = saveobj['UMAP_cluster_centroids']
    
    row_id_all = np.hstack(saveobj['row_id'].ravel())
    TP_all = np.hstack(saveobj['TP'].ravel())
    cell_id_all = np.hstack(saveobj['cell_id'].ravel())
    track_id_all = np.hstack(saveobj['track_id'].ravel())
    
    #### combine track and embryo id to make unique_track_id_all
    embryo_track_id_all = np.hstack([embryo_all[iii] +'_' + str(track_id_all[iii]).zfill(3) for iii in np.arange(len(track_id_all))])
    
    # row_id': row_id_all, 
    #               'TP':TP_all, 
    #               'cell_id':cell_id_all, 
    #               'track_id':track_id_all
    
    """
    recluster pca_project_cluster using alternative schemas 
    """
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
    from sklearn.cluster import Birch, MiniBatchKMeans
    # n_clusters = 27  # number of regions
    # ward = AgglomerativeClustering(
        # n_clusters=100, linkage="ward", connectivity=None)
    # ward = AgglomerativeClustering(n_clusters=100, linkage="ward")
    # ward.fit(pca_proj)
    
    # # db = DBSCAN(eps=1, min_samples=20).fit(pca_proj)
    # bir = Birch(threshold=.5, n_clusters=50).fit(pca_proj) # you need to tune this to have a minima number
    # pca_proj_cluster_bir = bir.labels_.copy()
    # pca_proj_cluster = pca_proj_cluster_bir.copy()

    # # try gaussian mixture. 
    # from sklearn.mixture import GaussianMixture
    
    # model = GaussianMixture(n_components=80, random_state=0) # try with 50, 80 and 100 
    # model.fit(pca_proj)
    # pca_proj_cluster = model.predict(pca_proj)
    
    # recluster with 80 ? 
    # pca_proj_cluster = KMeans(n_clusters=80, random_state=0).fit_predict(pca_proj) # this looks to be quite nice? 
    
    # cluster_ids = np.unique(pca_proj_cluster)
    # cluster_ids_centroids = np.vstack([np.nanmean(pca_proj[pca_proj_cluster==lab],axis=0) for lab in cluster_ids])

    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.scatter(pca_proj[:,0], 
    #            pca_proj[:,1], 
    #            c=pca_proj_cluster_bir)
    # plt.show()
    
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.scatter(pca_proj[:,0], 
    #            pca_proj[:,1], 
    #            c=pca_proj_cluster)
    # plt.show()
    
    
    
    cluster_ids_members = [np.arange(len(pca_proj_cluster))[pca_proj_cluster==cc] for cc in cluster_ids]
    cluster_hull = convexhull_pts(pca_proj, 
                                  ptslabels=pca_proj_cluster, 
                                  uniq_labels=cluster_ids)
    
    fig, ax = plt.subplots(figsize=(15,15))
    ax.plot(pca_proj[:,0], 
            pca_proj[:,1], '.')
    # for hull in cluster_hull:
    #     ax.plot(hull[:,0], 
    #             hull[:,1], 'k-')
    # draw and label the centroid
    for ii in np.arange(len(cluster_ids_centroids)):
        ax.text(cluster_ids_centroids[ii,0], 
                cluster_ids_centroids[ii,1], 
                str(ii), va='center', ha='center', fontsize=28)
    ax.set_aspect(1)
    plt.show()
    
# =============================================================================
#     compute the static clusters.
# =============================================================================
    quads_select = [[8,1,2],
                    [4,5,6],
                    [16,9,10], 
                    [12,13,14],
                    [24,17,18], 
                    [20,21,22],
                    [32,25,26],
                    [28,29,30]]

    stages = ['Pre-migration', 
              'Migration to Boundary']
    
    
    stad_quad_id_main = np.zeros_like(dynamic_quad_id_all)
    for qq_ii, qq in enumerate(quads_select):
        for qq_ss in qq:
            stad_quad_id_main[static_quad_id_all==qq_ss] = qq_ii+1
    stad_quad_id_main = np.hstack(stad_quad_id_main)
    stad_quad_id_main = stad_quad_id_main.astype(np.int)
    
    
    """
    Create a new subgrouping of regions splitting region 1. 
    """
    stad_quad_id_main_split = stad_quad_id_main.copy()
    stad_quad_id_main_split[np.logical_and(stad_quad_id_main==1, epi_contour_dist>0.75)] = 9 # lower half! 
    stad_quad_id_main_split[np.logical_and(stad_quad_id_main==1, epi_contour_dist<=0.75)] = 1 # keep as one # upper half!. 
    
    
    # now modify the dynamic track ids.... which will be based on time 0 tracks.... 
    dynamic_quad_id_main_split = dynamic_quad_id_main.copy()
    tracks_lower_1 = embryo_track_id_all[np.logical_and(np.logical_and(dynamic_quad_id_main==1, frame_all==0), epi_contour_dist>0.75)]
    tracks_lower_2 = embryo_track_id_all[np.logical_and(np.logical_and(dynamic_quad_id_main==1, frame_all==0), epi_contour_dist<=0.75)]
    
    # iterate and turn all of these track ids into the approriate IDs....
    for tra_id in tracks_lower_1:
        dynamic_quad_id_main_split[embryo_track_id_all==tra_id] = 9
    for tra_id in tracks_lower_2:
        dynamic_quad_id_main_split[embryo_track_id_all==tra_id] = 1
    

# =============================================================================
#     detect the number of unique embryos
# =============================================================================
    uniq_embryos = np.unique(embryo_all)
    
    print(len(uniq_embryos), ' embryos ')
    print('====')
    print(uniq_embryos)
    
    f_main_quads = np.hstack([np.sum(dynamic_quad_id_main==cc) for cc in np.setdiff1d(np.unique(dynamic_quad_id_main), 0)])
    f_main_quads = f_main_quads / float(np.sum(f_main_quads))
    
    from scipy.stats import mode
    cluster_ids_majority_valid = []
    cluster_ids_majority_valid_num = [] 
    cluster_ids_majority_main = []
    cluster_ids_prop = []
    static_cluster_ids_prop = []
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
        
        
        #### understanding the static. 
        static_member = stad_quad_id_main[cc]
        static_member = static_member[static_member>0]
        static_cluster_ids_prop.append( np.hstack([np.sum(static_member==group) for group in np.arange(1,9)]))
        
    cluster_ids_majority_valid = np.hstack(cluster_ids_majority_valid)
    cluster_ids_majority_valid_num = np.hstack(cluster_ids_majority_valid_num)
    cluster_ids_majority_main = np.hstack(cluster_ids_majority_main)
    cluster_ids_prop = np.vstack(cluster_ids_prop)
    cluster_ids_prop_stage = np.vstack(cluster_ids_prop_stage)
    cluster_ids_prop_embryo = np.vstack(cluster_ids_prop_embryo)
    static_cluster_ids_prop = np.vstack(static_cluster_ids_prop)
    
    # conduct statistical enrichment test ... with exact fisher... 
    # heirarchically cluster to arrange this ? after normalization.... 
    # hierarchically cluster to arrange .... 
    dynamic_cluster_ids_frac = cluster_ids_prop / (np.sum(cluster_ids_prop, axis=1)[:,None])
    static_cluster_ids_frac = static_cluster_ids_prop / (np.sum(static_cluster_ids_prop, axis=1)[:,None])
    
    quad_id_names = np.hstack([['Hist'+str(ii+1).zfill(2) for ii in range(8)], 
                               ['Static'+str(ii+1).zfill(2) for ii in range(8)]])
    
# =============================================================================
#   Understanding the distribution in the PCA space between static and dynamic quad. -> so we selectively color this for display!
# =============================================================================
    
    df = pd.DataFrame(np.vstack([dynamic_cluster_ids_frac.T,
                                 static_cluster_ids_frac.T]), 
                      index=quad_id_names, 
                      columns=None)
    
    
# =============================================================================
#     Automatic Hierarchical Clustering of the kmeans based regions.... 
# =============================================================================
    
    # # fig, ax = plt.subplots(figsize=(.5*100,.5*16))
    # g = sns.clustermap(df, 
    #                    method='average',
    #                    metric='euclidean',
    #                    cmap='vlag',
    #                    figsize=(.5*100,.5*16),
    #                    dendrogram_ratio=(0.05,0.1))
    # g.ax_cbar.remove()
    # g.ax_heatmap.xaxis.set_ticks(np.arange(len(g.dendrogram_col.reordered_ind))+.5)
    # g.ax_heatmap.xaxis.set_ticklabels(g.dendrogram_col.reordered_ind)
    # g.savefig(os.path.join(saveplotfolder, 'UMAP_regions_dendrogram_cluster.svg'), 
    #           bbox_inches='tight', pad_inches=0, dpi=300)
    # # ax.set_aspect(1)
    

#     # allocate them into clusters.
#     region_cluster_labels_color = sns.color_palette('Set2', 8)
#     region_cluster_labels = -1*np.ones(len(g.dendrogram_col.reordered_ind), dtype=np.int)    
#     region_cluster_labels[g.dendrogram_col.reordered_ind[:16]] = 0
#     region_cluster_labels[g.dendrogram_col.reordered_ind[16:16+13]] = 1
#     region_cluster_labels[g.dendrogram_col.reordered_ind[16+13:16+13+6]] = 2
    
# =============================================================================
#     dendrogram cluster
# =============================================================================

    dynamic_cluster_ids_frac = cluster_ids_prop / (np.sum(cluster_ids_prop, axis=1)[:,None])
    static_cluster_ids_frac = static_cluster_ids_prop / (np.sum(static_cluster_ids_prop, axis=1)[:,None])
    
    quad_id_names = np.hstack([['Hist'+str(ii+1).zfill(2) for ii in range(8)], 
                                ['Static'+str(ii+1).zfill(2) for ii in range(8)]])

    df = pd.DataFrame(np.vstack([dynamic_cluster_ids_frac.T,
                                  static_cluster_ids_frac.T]), 
                      index=quad_id_names, 
                      columns=None)
    # df_entropy_col = entropy(df, axis=0)
    df_max_col = df.max( axis=0)
    
    
    """
    what if only doing static labels? 
    """
    # quad_id_names = np.hstack([['Hist'+str(ii+1).zfill(2) for ii in range(8)], 
    #                             ['Static'+str(ii+1).zfill(2) for ii in range(8)]])

    # df = pd.DataFrame(dynamic_cluster_ids_frac.T, 
    #                   index=['Hist'+str(ii+1).zfill(2) for ii in range(8)], 
    #                   columns=None)
    # df = pd.DataFrame(static_cluster_ids_frac.T, 
                      # index=['Static'+str(ii+1).zfill(2) for ii in range(8)], 
                      # columns=None)
    
    # fig, ax = plt.subplots(figsize=(.5*100,.5*16))
    g = sns.clustermap(df, 
                        # ward gives the least noise .... 
                        method='ward', # use ward? # average # using ward is best to minimise the number of crazy clusters. 
                        metric='euclidean',
                        cmap='vlag',
                        figsize=(.5*100,.5*16),
                        dendrogram_ratio=(0.05,0.1))
    g.ax_cbar.remove()
    g.ax_heatmap.xaxis.set_ticks(np.arange(len(g.dendrogram_col.reordered_ind))+.5)
    g.ax_heatmap.xaxis.set_ticklabels(g.dendrogram_col.reordered_ind)
    g.savefig(os.path.join(saveplotfolder, 'UMAP_regions_dendrogram_cluster-jointPCA-average.svg'), 
              bbox_inches='tight', pad_inches=0, dpi=300)
    # # ax.set_aspect(1)
    
    
    from scipy.cluster.hierarchy import fcluster
    from skimage.filters import threshold_otsu
    # # thresh = threshold_otsu(g.dendrogram_col.calculated_linkage[:,2])
    # thresh = np.mean(g.dendrogram_col.calculated_linkage[:,2]) + 1*np.std(g.dendrogram_col.calculated_linkage[:,2])
    # gcluster = fcluster(g.dendrogram_col.calculated_linkage, 
    #                     t=thresh, criterion='distance', depth=2, R=None, monocrit=None) - 1#[source]
    # print(gcluster)
        
    # idea is we can analyse the breaks that occur and figure out how long it took... to get an idea of how many clusters to go for.
    cluster_formation_cols = np.sort(g.dendrogram_col.calculated_linkage[:,-1])
    cluster_formation_cols_diff = np.diff(cluster_formation_cols)
    cluster_formation_cols_break_pts = np.arange(len(cluster_formation_cols_diff))[cluster_formation_cols_diff>0]
    cluster_formation_cols_break_pts_len = np.diff(cluster_formation_cols_break_pts)
    
    # set the min detected length ...  
    min_cluster_len = 1
    max_clust_len = 2 + np.arange(len(cluster_formation_cols_break_pts_len))[cluster_formation_cols_break_pts_len==min_cluster_len][1] # tolerate the 2nd ? 
    
    homogeneity = []
    labels_all = []
    homogeneity_all = []
    
    # then use this to scan the max.... -> we know each step adds a cluster. 
    for link_ii in np.arange(len(g.dendrogram_col.calculated_linkage))[:max_clust_len+30]:
        lab = fcluster(g.dendrogram_col.calculated_linkage, 
                        t=g.dendrogram_col.calculated_linkage[-link_ii-1,2], 
                        criterion='distance', depth=2, R=None, monocrit=None)
        labels_all.append(lab)
        print(link_ii, len(np.unique(lab)))
        print([np.arange(len(cluster_ids))[lab==lab_uniq] for lab_uniq in np.unique(lab)])
        
        max_p = [np.mean(df_max_col[lab==lab_uniq]) for lab_uniq in np.unique(lab)]
        homogeneity_all.append(max_p)
        print(max_p)
        homogeneity.append(np.min(max_p))
        print('===')
        
    # try and separate out the clustering based on this. 
    labels_all = np.vstack(labels_all)
    homogeneity = np.hstack(homogeneity)
    
    # based on the stability of homogeneity .... predict.
    pred = 0
    count = 0
    homogeneity_diff = np.abs(np.diff(homogeneity))
    for hh in homogeneity_diff:
        if np.abs(hh) > 0:
            pred+=1
        else:
            if count > 2 : # relax this..? # use 1.   # one finds a larger cluster size... or we do 1 to be more specific.  2 is good and justified. 
                break
            else:
                count+=1    
    pred_index = pred + 1
    
    plt.figure()
    plt.plot(homogeneity,'.')
    plt.show()

    # print('n_clusters', pred_index+2) # this is wrong.... should be + 1 only since the unique groupings start from 1. 
    print('n_clusters', pred_index+1)
    
    
    # allocate them into clusters.
    region_cluster_labels_color = sns.color_palette('colorblind', 16) # and shuffle this.
    # np.random.shuffle(region_cluster_labels_color)
    
    
    region_cluster_labels = -1*np.ones(len(g.dendrogram_col.reordered_ind), dtype=np.int)        
    # region_cluster_labels = gcluster.copy()

    homogeneity_thresh = .3
# =============================================================================
#     figure out automated cluster determination. -> we want the minimal number of cluster for which we have relatively good grouping (single mode (low entropy configs)) 
# =============================================================================
    region_cluster_labels = labels_all[pred_index] - 1 
    region_cluster_labels_copy = region_cluster_labels.copy()
        
# =============================================================================
#   Before dropping compile the characteristics of cells in the different clusters as a heatmap. 
# =============================================================================
    
    mean_heatmap_matrix = []
    pca_timelapse_cluster_labels = region_cluster_labels[pca_proj_cluster]
    
    for clust in np.unique(region_cluster_labels):
        mean_timelapse_feats = np.nanmean(X_feats[pca_timelapse_cluster_labels==clust], axis=0)
        mean_heatmap_matrix.append(mean_timelapse_feats)
    mean_heatmap_matrix = np.vstack(mean_heatmap_matrix)
    
    plt.figure()
    plt.imshow( mean_heatmap_matrix, cmap='coolwarm')
    plt.xticks(np.arange(X_feats.shape[1]), col_name_select[:], rotation=45, ha='right')
    plt.show()
    
    
    # dropping clusters that are not super good. 
    drop_cluster_ids = np.arange(len(homogeneity_all[pred_index]))[np.hstack(homogeneity_all[pred_index])<=homogeneity_thresh] # include all cells? 
    
    for clust_id in drop_cluster_ids:
        region_cluster_labels[region_cluster_labels_copy==clust_id] = -1 # invalid. don't use
    
    homo_classes = np.hstack(homogeneity_all[pred_index])
    
    # we don't need this -> as this is essentially the regression coefficient. 
    class_weights = np.hstack([np.mean(homo_classes[homo_classes<=homogeneity_thresh]), 
                               homo_classes[homo_classes>homogeneity_thresh]])    
    
        
    mean_heatmap_matrix = []
    sd_heatmap_matrix = [] #  is this also required? 
    pca_timelapse_cluster_labels = region_cluster_labels[pca_proj_cluster]
    
    for clust in np.unique(region_cluster_labels):
        # if clust > -1: # we want all clusters. 
        mean_timelapse_feats = np.nanmean(X_feats[pca_timelapse_cluster_labels==clust], axis=0)
        mean_heatmap_matrix.append(mean_timelapse_feats)
        sd_heatmap_matrix.append(np.nanstd(X_feats[pca_timelapse_cluster_labels==clust], axis=0))
    mean_heatmap_matrix = np.vstack(mean_heatmap_matrix)
    sd_heatmap_matrix = np.vstack(sd_heatmap_matrix)
    
    plt.figure(figsize=(10,10))
    plt.imshow( mean_heatmap_matrix, cmap='coolwarm')
    plt.xticks(np.arange(X_feats.shape[1]), col_name_select[:], rotation=45, ha='right')
    plt.savefig(os.path.join(saveplotfolder, 'mean_heatmap_matrix_groups_valid.svg'), bbox_inches='tight')
    plt.show()
    
    
    ### get an ordered version......
    order_cluster = np.hstack([1,0,3,2,-1]) + 1
    plt.figure(figsize=(10,10))
    plt.imshow( mean_heatmap_matrix[order_cluster], cmap='coolwarm')
    plt.xticks(np.arange(X_feats.shape[1]), col_name_select[:], rotation=45, ha='right')
    plt.savefig(os.path.join(saveplotfolder, 'mean_heatmap_matrix_groups_valid_reordered.svg'), bbox_inches='tight')
    plt.show()
    
# =============================================================================
#     Go ahead here to plot the barchart characterisation in the same manner as that for the organoid paper. 
# =============================================================================
    
    def get_colors(inp, colormap, vmin=None, vmax=None):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))
    
    """
    alternative to plot each of these as a bar graph to show stats. 
    """
    
    order_cluster = np.hstack([1,0,3,2,-1]) + 1
    
    # iterate over the phenotype cluster.
    for ii in np.arange(mean_heatmap_matrix.shape[0]):

        # map the values to a colormap namely blue and red.         

        fig, ax = plt.subplots(figsize=(2, X_feats.shape[1]))
        data = mean_heatmap_matrix[order_cluster[ii], ::-1]
        bar_colors = get_colors(data, colormap=plt.cm.coolwarm, 
                                vmin=-2, vmax=2)
        
        for jj in np.arange(len(data)):
            bar_color = bar_colors[jj]
            ax.barh(jj, 
                    data[jj], edgecolor='k', color=bar_colors[jj])
        plt.vlines(0, -0.5, mean_heatmap_matrix.shape[1]-0.5, color='k')
        plt.yticks(np.arange(mean_heatmap_matrix.shape[1]), 
                   col_name_select[::-1])
        plt.xlim([-2,2])
        plt.ylim([-0.5, mean_heatmap_matrix.shape[1]-0.5])
        # make the right and top spines away. 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(np.linspace(-2,2,5))
        # plt.savefig('BrittanyConfocal_cluster_mean_SAM_features-Bar_Cluster_%s.svg' %(str(ii).zfill(3)), 
        #             dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(saveplotfolder, 'mean_heatmap_matrix_groups_valid_%s.svg' %(str(ii).zfill(3))), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.tick_params(length=5)
        plt.show()
    
    
    
# =============================================================================
#     Recompile the mean statistics
# =============================================================================
    mean_heatmap_matrix_raw = []
    sd_heatmap_matrix_raw = [] #  is this also required? 
    # pca_timelapse_cluster_labels = region_cluster_labels[pca_proj_cluster]
    
    for clust in np.unique(region_cluster_labels):
        # if clust > -1:
        mean_timelapse_feats = np.nanmean(X_feats_raw[pca_timelapse_cluster_labels==clust], axis=0)
        mean_heatmap_matrix_raw.append(mean_timelapse_feats)
        sd_heatmap_matrix_raw.append(np.nanstd(X_feats_raw[pca_timelapse_cluster_labels==clust], axis=0))
    mean_heatmap_matrix_raw = np.vstack(mean_heatmap_matrix_raw)
    sd_heatmap_matrix_raw = np.vstack(sd_heatmap_matrix_raw)

    # region_cluster_labels[[1,66,32,91,58,60,48,40,73,8,93,24,26,3,51,88,27,77]] = 1
    # region_cluster_labels[[87,20,44,33,50,83,17,84]] = 2 
    # region_cluster_labels[[39,10,18,22,92,38,62]] = 3 
    
    # region_cluster_labels[[18,40,67,13,84,57,35,94,45,6,46,77,72,90,96,49,91]] = 3 
    # region_cluster_labels[[40,82,30,60,5,66,59,16,65]] = 5  
    
    # region_cluster_labels[[78,67,36,87,    72,31,64]] = 2  
    # # region_cluster_labels[[90,83,15,88,3,10,29,20,61,8,26,55,76]] = 3  
    # region_cluster_labels[[90,83,15,88]] = 3  


    # region_cluster_labels[[13,34,48,80, 95,89,7,43,58]] = 0
    # region_cluster_labels[[85,71,12,54,42,47,33,21,81]] = 4
    
    # region_cluster_labels[[53,39,6,84]] = 1 
    # region_cluster_labels[[40,82,30,60,5,66,59,16,65]] = 5  
    
    # region_cluster_labels[[78,67,36,87,    72,31,64]] = 2  
    # # region_cluster_labels[[90,83,15,88,3,10,29,20,61,8,26,55,76]] = 3  
    # region_cluster_labels[[90,83,15,88]] = 3  
    
    # fig, ax = plt.subplots(figsize=(15,15))
    # # ax.scatter(pca_timelapse[:,0], 
    # #            pca_timelapse[:,1], color='lightgrey')
    # # for hull_ii, hull in enumerate(cluster_hull):
    # #     hull_label = region_cluster_labels[hull_ii]
    # #     ax.plot(hull[:,0], 
    # #             hull[:,1], 'k-',zorder=1)
    # #     if hull_label >= 0: 
    # #         ax.fill(hull[:,0], 
    # #                 hull[:,1], color=region_cluster_labels_color[hull_label], zorder=1)
    # # # draw and label the centroid
    # for ii in np.arange(len(cluster_ids_centroids)):
    #     hull_label = region_cluster_labels[ii]
    #     # get the modal label in this region. 
    #     mode_label = cluster_ids_majority_main[ii]
        
    #     if hull_label>=0:
    #         ax.text(cluster_ids_centroids[ii,0], 
    #                 cluster_ids_centroids[ii,1], 
    #                 str(int(mode_label)), va='center', ha='center', fontsize=28)
    # ax.set_aspect(1)
    # plt.grid('off')
    # plt.axis('off')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster.svg'), bbox_inches='tight')
    # plt.show()
    
    # print(region_cluster_labels[g.dendrogram_col.reordered_ind])
    

    fig, ax = plt.subplots(figsize=(15,15))
    # ax.scatter(pca_timelapse[:,0], 
    #            pca_timelapse[:,1], color='lightgrey')
    # for hull_ii, hull in enumerate(cluster_hull):
    #     hull_label = region_cluster_labels[hull_ii]
    #     ax.plot(hull[:,0], 
    #             hull[:,1], 'k-',zorder=1)
    #     if hull_label >= 0: 
    #         ax.fill(hull[:,0], 
    #                 hull[:,1], color=region_cluster_labels_color[hull_label], zorder=1)
    # # draw and label the centroid
    for ii in np.arange(len(cluster_ids_centroids)):
        hull_label = region_cluster_labels[ii]
        # get the modal label in this region. 
        mode_label = cluster_ids_majority_main[ii]
        
        if hull_label>=0:
            ax.plot(cluster_ids_centroids[ii,0], 
                    cluster_ids_centroids[ii,1], '.')
            ax.text(cluster_ids_centroids[ii,0], 
                    cluster_ids_centroids[ii,1], 
                    str(int(mode_label)), va='center', ha='center', fontsize=28)
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster.svg'), bbox_inches='tight')
    plt.show()
    
    
    

# =============================================================================
#     Shankar's suggestion is to color the umap ... so we can do this instead of doing the fill.... 
# =============================================================================


# =============================================================================
# =============================================================================
# # Suggestion to recolor such that  
# =============================================================================
# =============================================================================
    
    # map the region cluster labels now to the umap cluster labels (defined now over all points!. )
    umap_phenotype_cluster_labels = np.zeros(len(pca_proj_cluster), dtype=np.int)  
    for clust_ii, clust in enumerate(np.unique(pca_proj_cluster)):
        umap_phenotype_cluster_labels[pca_proj_cluster==clust] = region_cluster_labels[clust_ii]

    print(region_cluster_labels[g.dendrogram_col.reordered_ind])
    
    all_umap_cluster_colors = np.vstack(region_cluster_labels_color)[np.unique(umap_phenotype_cluster_labels)]
    from matplotlib import colors
    # print(colors.to_rgba('lightgrey'))
    all_umap_cluster_colors[0] = np.hstack(colors.to_rgba('lightgrey')[:3])
    
    all_umap_cluster_colors = all_umap_cluster_colors[[0,2,3,1,4]] # ok so this is the new coloring for the umap positions ! 
    
    fig, ax = plt.subplots(figsize=(15,15))
    ax.scatter(pca_proj[:,0], 
               pca_proj[:,1], color='lightgrey')
    for clust_ii, clust in enumerate(np.unique(umap_phenotype_cluster_labels)):
        if clust != -1: 
            ax.plot(pca_proj[umap_phenotype_cluster_labels == int(clust), 0 ], 
                    pca_proj[umap_phenotype_cluster_labels == int(clust), 1 ], '.',
                    color = all_umap_cluster_colors[int(clust_ii)])
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
    plt.show()
    
    
    """ Draw the above and the region ids..
    """
    fig, ax = plt.subplots(figsize=(15,15))
    # ax.scatter(pca_timelapse[:,0], 
    #            pca_timelapse[:,1], color='lightgrey')
    ax.scatter(pca_proj[:,0], 
               pca_proj[:,1], color='lightgrey')
    for clust_ii, clust in enumerate(np.unique(umap_phenotype_cluster_labels)):
        if clust != -1: 
            ax.plot(pca_proj[umap_phenotype_cluster_labels == int(clust), 0 ], 
                    pca_proj[umap_phenotype_cluster_labels == int(clust), 1 ], '.',
                    color = all_umap_cluster_colors[int(clust_ii)])
    for hull_ii, hull in enumerate(cluster_hull):
        hull_label = region_cluster_labels[hull_ii]
        ax.plot(hull[:,0], 
                hull[:,1], 'k-',zorder=1000)
        # if hull_label >= 0: 
        #     ax.fill(hull[:,0], 
        #             hull[:,1], 
        #             color=region_cluster_labels_color[hull_label], zorder=1000, alpha=0)
    # # # draw and label the centroid
    # for ii in np.arange(len(cluster_ids_centroids)):
    #     hull_label = region_cluster_labels[ii]
    #     # get the modal label in this region. 
    #     mode_label = cluster_ids_majority_main[ii]
        
    #     if hull_label>=0:
    #         ax.plot(cluster_ids_centroids[ii,0], 
    #                 cluster_ids_centroids[ii,1], '.')
    #         ax.text(cluster_ids_centroids[ii,0], 
    #                 cluster_ids_centroids[ii,1], 
    #                 str(int(mode_label)), va='center', ha='center', fontsize=28)
    for ii in np.arange(len(cluster_ids_centroids)):
        ax.text(cluster_ids_centroids[ii,0], 
                cluster_ids_centroids[ii,1], 
                str(ii), va='center', ha='center', fontsize=24, fontname='Arial')
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points_with_text_and_voronoi.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points_with_text_and_voronoi.png'), bbox_inches='tight', dpi=600)
    
    plt.show()
    
    
    
# =============================================================================
#     Make piechart distribution plots for each cluster. ! 
# =============================================================================

    dynamic_quad_id_distribution_in_phenotype_clusters = []
    for pheno_cluster in np.unique(umap_phenotype_cluster_labels):
        dynamic_ids = dynamic_quad_id_main[umap_phenotype_cluster_labels==pheno_cluster]
        # get the histogram over the region. 
        counts_dynamic_ids = np.hstack([np.sum(dynamic_ids.astype(np.int)==jjj) for jjj in np.arange(8)+1])
        dynamic_quad_id_distribution_in_phenotype_clusters.append(counts_dynamic_ids)
    dynamic_quad_id_distribution_in_phenotype_clusters = np.array(dynamic_quad_id_distribution_in_phenotype_clusters)

    pie_region_colors = sns.color_palette('magma_r', n_colors=8)[:] # to be consistent. 
    # iterate and plot the clusters... distributions in the natural order. --- i.e. the plotting color order. 
    for clust_ii in np.arange(len(dynamic_quad_id_distribution_in_phenotype_clusters)):
        
        fig, ax = plt.subplots(figsize=(5,5))
        draw_pie_xy(dynamic_quad_id_distribution_in_phenotype_clusters[clust_ii], 
                      0, 
                      0, 
                      size=10,
                      startangle=0,
                      colors=pie_region_colors, 
                      ax=ax)
        # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        ax.set_aspect(1)
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(saveplotfolder, 'dynamic_id_distribution-PhenoCluster_%s.svg' %(str(clust_ii).zfill(3))), 
                    bbox_inches='tight')
        # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning_and_majority_vote_w_pie.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        
    # give the static might be more meaningful!. 
    static_quad_id_distribution_in_phenotype_clusters = []
    for pheno_cluster in np.unique(umap_phenotype_cluster_labels):
        static_ids = stad_quad_id_main[umap_phenotype_cluster_labels==pheno_cluster]
        # get the histogram over the region. 
        counts_static_ids = np.hstack([np.sum(static_ids.astype(np.int)==jjj) for jjj in np.arange(8)+1])
        static_quad_id_distribution_in_phenotype_clusters.append(counts_static_ids)
    static_quad_id_distribution_in_phenotype_clusters = np.array(static_quad_id_distribution_in_phenotype_clusters)

    pie_region_colors = sns.color_palette('magma_r', n_colors=8)[:] # to be consistent. 
    # iterate and plot the clusters... distributions in the natural order. --- i.e. the plotting color order. 
    for clust_ii in np.arange(len(static_quad_id_distribution_in_phenotype_clusters)):
        
        fig, ax = plt.subplots(figsize=(5,5))
        draw_pie_xy(static_quad_id_distribution_in_phenotype_clusters[clust_ii], 
                      0, 
                      0, 
                      size=10,
                      startangle=0,
                      colors=pie_region_colors, 
                      ax=ax)
        # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        ax.set_aspect(1)
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(saveplotfolder, 'static_id_distribution-PhenoCluster_%s.svg' %(str(clust_ii).zfill(3))), 
                    bbox_inches='tight')
        # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning_and_majority_vote_w_pie.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        
        
# =============================================================================
#     Make the reverse piechart of 8 i.e. for each dynamic and static regions of 8 what are the distributions of cells labelled a particular phenotype. 
# =============================================================================
    
    phenotype_distriution_in_dynamic_quad_id_clusters = []
    # for pheno_cluster in np.unique(umap_phenotype_cluster_labels):
    for region_cluster in np.arange(8)+1:
        # dynamic_ids = dynamic_quad_id_main[umap_phenotype_cluster_labels==pheno_cluster]
        pheno_ids = umap_phenotype_cluster_labels[dynamic_quad_id_main==region_cluster]
        # get the histogram over the region. 
        counts_pheno_ids = np.hstack([np.sum(pheno_ids.astype(np.int)==jjj) for jjj in np.unique(umap_phenotype_cluster_labels)])
        phenotype_distriution_in_dynamic_quad_id_clusters.append(counts_pheno_ids)
    phenotype_distriution_in_dynamic_quad_id_clusters = np.array(phenotype_distriution_in_dynamic_quad_id_clusters)

    # pie_region_colors = sns.color_palette('magma_r', n_colors=8)[:] # to be consistent. 
    pie_region_colors = all_umap_cluster_colors.copy()
    # iterate and plot the clusters... distributions in the natural order. --- i.e. the plotting color order. 
    for clust_ii in np.arange(len(phenotype_distriution_in_dynamic_quad_id_clusters)):
        
        fig, ax = plt.subplots(figsize=(5,5))
        draw_pie_xy(phenotype_distriution_in_dynamic_quad_id_clusters[clust_ii], 
                      0, 
                      0, 
                      size=10,
                      startangle=0,
                      colors=pie_region_colors, 
                      ax=ax)
        # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        ax.set_aspect(1)
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(saveplotfolder, 'phenotypic_id_distribution-DynamicRegions_%s.svg' %(str(clust_ii+1).zfill(3))), 
                    bbox_inches='tight')
        # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning_and_majority_vote_w_pie.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        
        
    """
    For static region cluster ids. 
    """
    phenotype_distribution_in_static_quad_id_clusters = []
    for region_cluster in np.arange(8)+1:
        # dynamic_ids = dynamic_quad_id_main[umap_phenotype_cluster_labels==pheno_cluster]
        pheno_ids = umap_phenotype_cluster_labels[stad_quad_id_main==region_cluster]
        # get the histogram over the region. 
        counts_pheno_ids = np.hstack([np.sum(pheno_ids.astype(np.int)==jjj) for jjj in np.unique(umap_phenotype_cluster_labels)])
        phenotype_distribution_in_static_quad_id_clusters.append(counts_pheno_ids)
    phenotype_distribution_in_static_quad_id_clusters = np.array(phenotype_distribution_in_static_quad_id_clusters)

    # pie_region_colors = sns.color_palette('magma_r', n_colors=8)[:] # to be consistent. 
    pie_region_colors = all_umap_cluster_colors.copy()
    # iterate and plot the clusters... distributions in the natural order. --- i.e. the plotting color order. 
    for clust_ii in np.arange(len(phenotype_distribution_in_static_quad_id_clusters)):
        
        fig, ax = plt.subplots(figsize=(5,5))
        draw_pie_xy(phenotype_distribution_in_static_quad_id_clusters[clust_ii], 
                      0, 
                      0, 
                      size=10,
                      startangle=0,
                      colors=pie_region_colors, 
                      ax=ax)
        # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        ax.set_aspect(1)
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(saveplotfolder, 'phenotypic_id_distribution-StaticRegions_%s.svg' %(str(clust_ii+1).zfill(3))), 
                    bbox_inches='tight')
        # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning_and_majority_vote_w_pie.png'), dpi=300, bbox_inches='tight')
        plt.show()



# =============================================================================
# =============================================================================
# #  Making the above but now splitting region 1 (based on lineage and static IDs)
# =============================================================================
# =============================================================================

    phenotype_distriution_in_dynamic_quad_id_clusters = []
    # for pheno_cluster in np.unique(umap_phenotype_cluster_labels):
    for region_cluster in np.arange(9)+1:
        # dynamic_ids = dynamic_quad_id_main[umap_phenotype_cluster_labels==pheno_cluster]
        pheno_ids = umap_phenotype_cluster_labels[dynamic_quad_id_main_split==region_cluster]
        # get the histogram over the region. 
        counts_pheno_ids = np.hstack([np.sum(pheno_ids.astype(np.int)==jjj) for jjj in np.unique(umap_phenotype_cluster_labels)])
        phenotype_distriution_in_dynamic_quad_id_clusters.append(counts_pheno_ids)
    phenotype_distriution_in_dynamic_quad_id_clusters = np.array(phenotype_distriution_in_dynamic_quad_id_clusters)

    # pie_region_colors = sns.color_palette('magma_r', n_colors=8)[:] # to be consistent. 
    pie_region_colors = all_umap_cluster_colors.copy()
    # iterate and plot the clusters... distributions in the natural order. --- i.e. the plotting color order. 
    for clust_ii in np.arange(len(phenotype_distriution_in_dynamic_quad_id_clusters)):
        
        fig, ax = plt.subplots(figsize=(5,5))
        draw_pie_xy(phenotype_distriution_in_dynamic_quad_id_clusters[clust_ii], 
                      0, 
                      0, 
                      size=10,
                      startangle=0,
                      colors=pie_region_colors, 
                      ax=ax)
        # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        ax.set_aspect(1)
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(saveplotfolder, 'phenotypic_id_distribution-DynamicRegionsSplit_%s.svg' %(str(clust_ii+1).zfill(3))), 
                    bbox_inches='tight')
        # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning_and_majority_vote_w_pie.png'), dpi=300, bbox_inches='tight')
        plt.show()


    """
    For static region cluster ids. 
    """
    phenotype_distribution_in_static_quad_id_clusters = []
    for region_cluster in np.arange(9)+1:
        # dynamic_ids = dynamic_quad_id_main[umap_phenotype_cluster_labels==pheno_cluster]
        pheno_ids = umap_phenotype_cluster_labels[stad_quad_id_main_split==region_cluster]
        # get the histogram over the region. 
        counts_pheno_ids = np.hstack([np.sum(pheno_ids.astype(np.int)==jjj) for jjj in np.unique(umap_phenotype_cluster_labels)])
        phenotype_distribution_in_static_quad_id_clusters.append(counts_pheno_ids)
    phenotype_distribution_in_static_quad_id_clusters = np.array(phenotype_distribution_in_static_quad_id_clusters)

    # pie_region_colors = sns.color_palette('magma_r', n_colors=8)[:] # to be consistent. 
    pie_region_colors = all_umap_cluster_colors.copy()
    # iterate and plot the clusters... distributions in the natural order. --- i.e. the plotting color order. 
    for clust_ii in np.arange(len(phenotype_distribution_in_static_quad_id_clusters)):
        
        fig, ax = plt.subplots(figsize=(5,5))
        draw_pie_xy(phenotype_distribution_in_static_quad_id_clusters[clust_ii], 
                      0, 
                      0, 
                      size=10,
                      startangle=0,
                      colors=pie_region_colors, 
                      ax=ax)
        # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        ax.set_aspect(1)
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(saveplotfolder, 'phenotypic_id_distribution-StaticRegionsSplit_%s.svg' %(str(clust_ii+1).zfill(3))), 
                    bbox_inches='tight')
        # plt.savefig(os.path.join(saveplotfolder, 'PCA_kmeans_partitioning_and_majority_vote_w_pie.png'), dpi=300, bbox_inches='tight')
        plt.show()












# =============================================================================
#   Make plots of all the key statistics and scores.. in the umap space. 
# =============================================================================

    # for feat_ii in np.arange(len(col_name_select)):
    # plot the mean in each voronoi regions. 
    for feat_ii in np.arange(len(col_name_select))[:]:
    # for feat_ii in np.arange(len(col_name_select))[11:12]:
    
        fig, ax = plt.subplots(figsize=(15,15))
        # plt.title('%s'%(col_name_select[feat_ii]))
        
        discrete_feats = np.zeros(len(X_feats))
        
        # for hull in cluster_hull:
        #     ax.plot(hull[:,0], 
        #             hull[:,1], 'k-')
        for cc in cluster_ids_members:
            cc_feats = X_feats[cc,feat_ii].copy()
            cc_feats_mean = np.nanmedian(cc_feats)
            discrete_feats[cc] = cc_feats_mean.copy()
        # z = np.abs(X_feats[:,feat_ii]); idx = z.argsort()
        
        if 'Epi' in col_name_select[feat_ii] and 'speed' in col_name_select[feat_ii]:
            max_scale = np.std(np.abs(X_feats[:,feat_ii]))*1
        elif 'Normal_VE' in col_name_select[feat_ii]:
            max_scale = np.std(np.abs(X_feats[:,feat_ii]))*.5
        elif 'Mean Curvature' in col_name_select[feat_ii] and 'Epi' in col_name_select[feat_ii]:
            max_scale = np.std(np.abs(X_feats[:,feat_ii]))*2
        else:
        # max_scale = np.quantile(np.abs(discrete_feats), 0.75)
            max_scale = np.std(np.abs(X_feats[:,feat_ii]))*4
        plt.scatter(pca_proj[:,0], 
                    pca_proj[:,1], c=discrete_feats, cmap='coolwarm', vmin=-max_scale,vmax=max_scale)
        # plt.plot(cluster_ids_centroids[:,0], 
                  # cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
        
        # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        ax.set_aspect(1)
        plt.grid('off')
        plt.axis('off')
        # plt.savefig(os.path.join(saveplotfolder, 'umap_%s_mean_regions.svg' %(col_name_select[feat_ii]).strip()), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(saveplotfolder, 'umap_%s_mean_regions.png' %(col_name_select[feat_ii]).strip()), dpi=300, bbox_inches='tight')
        plt.show()
        
        
        
        # also plot a version where we simply overlay the points? 
        fig, ax = plt.subplots(figsize=(15,15))
        # plt.title('%s'%(col_name_select[feat_ii]))
        
        # discrete_feats = np.zeros(len(X_feats))
        
        # # for hull in cluster_hull:
        # #     ax.plot(hull[:,0], 
        # #             hull[:,1], 'k-')
        # for cc in cluster_ids_members:
        #     cc_feats = X_feats[cc,feat_ii].copy()
        #     cc_feats_mean = np.nanmedian(cc_feats)
        #     discrete_feats[cc] = cc_feats_mean.copy()
        # # z = np.abs(X_feats[:,feat_ii]); idx = z.argsort()
        if 'Epi' in col_name_select[feat_ii] and 'speed' in col_name_select[feat_ii]:
            max_scale = np.std(np.abs(X_feats[:,feat_ii]))*1
        elif 'Normal_VE' in col_name_select[feat_ii]:
            max_scale = np.std(np.abs(X_feats[:,feat_ii]))*.5
        elif 'Mean Curvature' in col_name_select[feat_ii] and 'Epi' in col_name_select[feat_ii]:
            max_scale = np.std(np.abs(X_feats[:,feat_ii]))*2
        else:
        # max_scale = np.quantile(np.abs(discrete_feats), 0.75)
            max_scale = np.std(np.abs(X_feats[:,feat_ii]))*4
        plt.scatter(pca_proj[:,0], 
                    pca_proj[:,1], 
                    c=X_feats[:,feat_ii], 
                    cmap='coolwarm', 
                    vmin=-max_scale,vmax=max_scale)
        # plt.plot(cluster_ids_centroids[:,0], 
                  # cluster_ids_centroids[:,1], 'k.', zorder=100, ms=20)
        
        # plt.vlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        # plt.hlines(0, -max_val,max_val, linestyles='dashed', lw=3, zorder=1)
        ax.set_aspect(1)
        plt.grid('off')
        plt.axis('off')
        # plt.savefig(os.path.join(saveplotfolder, 'umap_%s_mean_regions_continuous.svg' %(col_name_select[feat_ii]).strip()), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(saveplotfolder, 'umap_%s_mean_regions_continuous.png' %(col_name_select[feat_ii]).strip()), dpi=300, bbox_inches='tight')
        plt.show()
        
        
    
# # =============================================================================
# #     Now we need to map the static points into the same .... 
# # =============================================================================
#     fig, ax = plt.subplots(figsize=(15,15))
#     ax.scatter(pca_timelapse[:,0], 
#                 pca_timelapse[:,1], color='lightgrey', alpha=.5)
#     for hull_ii, hull in enumerate(cluster_hull):
#         hull_label = region_cluster_labels[hull_ii]
#         ax.plot(hull[:,0], 
#                 hull[:,1], 'k-',zorder=1, alpha=.5)
#         if hull_label >= 0: 
#             ax.fill(hull[:,0], 
#                     hull[:,1], color=region_cluster_labels_color[hull_label], zorder=1, alpha=.5)
#     # draw and label the centroid
#     for ii in np.arange(len(cluster_ids_centroids)):
#         hull_label = region_cluster_labels[ii]
#         # get the modal label in this region. 
#         mode_label = cluster_ids_majority_main[ii]
        
# # =============================================================================
# #     Do a clustering again but just with the dynamic (more explanatory)
# # =============================================================================
        
#     # this version more accurate. 
#     df = pd.DataFrame(np.vstack([dynamic_cluster_ids_frac.T]), 
#                       index=quad_id_names[:8], 
#                       columns=None)
    
#     # fig, ax = plt.subplots(figsize=(.5*100,.5*16))
#     g = sns.clustermap(df, 
#                        method='ward',
#                        metric='euclidean',
#                        cmap='vlag',
#                        figsize=(.5*100,.5*8),
#                        dendrogram_ratio=(0.05,0.1))
#     g.ax_cbar.remove()
#     g.ax_heatmap.xaxis.set_ticks(np.arange(len(g.dendrogram_col.reordered_ind))+.5)
#     g.ax_heatmap.xaxis.set_ticklabels(g.dendrogram_col.reordered_ind)
#     g.savefig(os.path.join(saveplotfolder, 'PCA_regions_dendrogram_cluster_dynamic_only.svg'), 
#               bbox_inches='tight', pad_inches=0, dpi=300)
#     # ax.set_aspect(1)
    

#     # allocate them into clusters.
#     region_cluster_labels_color = sns.color_palette('Set2', 8)
#     region_cluster_labels = -1*np.ones(len(g.dendrogram_col.reordered_ind), dtype=np.int)    
#     region_cluster_labels[g.dendrogram_col.reordered_ind[84:]] = 0 # color for AVE 
#     region_cluster_labels[g.dendrogram_col.reordered_ind[71:84]] = 1
#     region_cluster_labels[g.dendrogram_col.reordered_ind[26:34]] = 2
#     region_cluster_labels[g.dendrogram_col.reordered_ind[34:34+9]] = 3 # less extremal. 
      
# #     cluster_ids_majority_valid = np.hstack(cluster_ids_majority_valid)
# #     cluster_ids_majority_valid_num = np.hstack(cluster_ids_majority_valid_num)
# #     cluster_ids_majority_main = np.hstack(cluster_ids_majority_main)
# #     cluster_ids_prop = np.vstack(cluster_ids_prop)
# #     cluster_ids_prop_stage = np.vstack(cluster_ids_prop_stage)
# #     cluster_ids_prop_embryo = np.vstack(cluster_ids_prop_embryo)
# #     # np.hstack([mode(dynamic_quad_id_all[cc])[0] for cc in cluster_ids_members])


    clustering_savefolder = os.path.join(saveplotfolder, 'cluster_stats');
    fio.mkdir(clustering_savefolder)
    savematfile = os.path.join(clustering_savefolder, 'ward_hcluster_stats_groups_all_embryos.mat')
    
    spio.savemat(savematfile, 
                 {'inputfile': savefile, # use this file input to get the other parameters for analysis., 
                  'region_cluster_labels': region_cluster_labels, 
                  'region_cluster_labels_colors' : region_cluster_labels_color, 
                  'individual_umap_cluster_labels': umap_phenotype_cluster_labels, 
                  # 'cluster_dendrogram' : g, # can't save this... hm... 
                  'mean_heatmap_matrix': mean_heatmap_matrix, 
                  'std_heatmap_matrix': sd_heatmap_matrix, 
                  'homo_classes_pre': homo_classes, 
                  'homo_classes_after': class_weights,
                  'dendrogram_homo_all': homogeneity_all, 
                  'dendrogram_labels_all': labels_all,
                  'mean_heatmap_matrix_raw': mean_heatmap_matrix_raw,  
                  'sd_heatmap_matrix_raw': sd_heatmap_matrix_raw}) 
    
    
    """
    separately save the dendrogram (pickle?)
    """
    # this can be done!. 
    dendrogram_obj_file = os.path.join(clustering_savefolder, 'ward_hcluster_obj.npy')
    np.save(dendrogram_obj_file, g, allow_pickle=True, fix_imports=True)

    
    saveplotfolder_final = os.path.join(saveplotfolder, 'refine')
    fio.mkdir(saveplotfolder_final)
    
# =============================================================================
#   retrieve ... and mapback. 
# =============================================================================

    embryo_all = np.hstack(embryo_all)
    uniq_embryo = np.unique(embryo_all)
    
    # TP0_uniq_embryo = np.hstack([np.abs(frame_all[embryo_all==emb]).min() for emb in uniq_embryo])
    embryo_TP0_frame = {emb:np.abs(frame_all[embryo_all==emb]).min() for emb in uniq_embryo}
    
    embryo_file_dict = {}
    
    # metricfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3'
    # imgfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\CellImages'
    metricfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3'
    imgfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\CellImages'
    
    
    for emb in uniq_embryo:
    
        # metricfile: 
        # segfile = 
        # polarfile =  # rotated? or not .... -> should do rotated? -> rotate this and save out.... for this.  
        tracklookupfile = os.path.join(metricfolder, 'single_cell_track_stats_lookup_table_'+emb+'.csv')
        singlecellstatfile = os.path.join(metricfolder, 'single_cell_statistics_table_export-ecc-realxyz-from-demons_'+emb+'.csv')
        
        polarfile = os.path.join(imgfolder, emb, 'A-P_rotated_data','AVE_rot-%s' %(emb) + '.tif')
        segfile = os.path.join(imgfolder, emb, 'A-P_rotated_data','AVE_rot-cells_%s' %(emb) + '.tif')
        
        metadict = {'segfile': segfile, 
                    'tracklookupfile': tracklookupfile,
                    'singlecellstatfile': singlecellstatfile,
                    'polarfile': polarfile}
        
        embryo_file_dict[emb] = metadict
    

# =============================================================================
#   Can we save this in some manner ... ? 
# =============================================================================

    # wrap all the meta into one dict. 
    metadict_id_array = {'embryo_all':embryo_all, 
                          'TP_all':TP_all, 
                          'cell_id_all': cell_id_all, 
                          'row_id_all': row_id_all, 
                          'track_id_all': track_id_all}

    # i think this is only taking the goodness. 
    valid_region_clusters = np.unique(region_cluster_labels) 
    valid_region_clusters = valid_region_clusters[valid_region_clusters>=0]

    # NSWE_complexes = [np.arange(len(region_cluster_labels))[region_cluster_labels==iii] for iii in range(region_cluster_labels.max()+1)]
    NSWE_complexes = [np.arange(len(region_cluster_labels))[region_cluster_labels==iii] for iii in valid_region_clusters]
    # NSWE_colors = sns.color_palette('Set2', n_colors=len(NSWE_complexes)) # make these the same as the region colors!.
    NSWE_colors = np.vstack([region_cluster_labels_color[cc] for cc in valid_region_clusters])
    NSWE_colors = NSWE_colors[[1,2,0,3]] #all_umap_cluster_colors = all_umap_cluster_colors[[0,2,3,1,4]]

    for cc_ii, cc in enumerate(NSWE_complexes[:]):
        
        print(cc_ii)
        cc_select = pca_proj_cluster == cc[0]
        for c in cc[1:]:
            cc_select = np.logical_or(cc_select, pca_proj_cluster == c)
            
        pts_ids = np.arange(len(pca_proj))[cc_select] # convert to numerical ids.
    
        if cc_ii == 0: 
            embryo_level_data, ent_data = batch_map_stats_back_to_image(pts_ids, 
                                                                        metadict_id_array, 
                                                                        embryo_file_dict,
                                                                        return_global=True)
            
            # get the frame of each unique_embryo
            uniq_embryo = np.hstack(list(embryo_level_data.keys()))
            uniq_embryo_ind = {uniq_embryo[ii]:ii for ii in np.arange(len(uniq_embryo))}
            uniq_embryo_TP0 = np.hstack([embryo_TP0_frame[emb] for emb in uniq_embryo])
            
            TP0_frames = [embryo_level_data[uniq_embryo[iii]]['img'][uniq_embryo_TP0[iii]] for iii in np.arange(len(uniq_embryo_TP0))]
        
            complex_pts_all = [[[] for jjj in np.arange(len(uniq_embryo))] for iii in np.arange(len(NSWE_complexes))]
            complex_tracks_all = [[[] for jjj in np.arange(len(uniq_embryo))] for iii in np.arange(len(NSWE_complexes))]
            complex_times_all = [[[] for jjj in np.arange(len(uniq_embryo))] for iii in np.arange(len(NSWE_complexes))]
            complex_track_times_all = [[[] for jjj in np.arange(len(uniq_embryo))] for iii in np.arange(len(NSWE_complexes))]
            
        
            # iterate and add to the correct list. 
            for ee in ent_data:
                emb = ee[0]['embryo']
                emb_ind = uniq_embryo_ind[emb]
                
                xy = np.hstack([ee[0]['pos_2D_x_rot-AVE'], 
                                ee[0]['pos_2D_y_rot-AVE']])
                
                xytime = ee[0]['Frame'] - uniq_embryo_TP0[emb_ind]
                xytrack = np.vstack([ee[1]['pos_2D_x_rot-AVE'], 
                                      ee[1]['pos_2D_y_rot-AVE']]).T
                
                complex_pts_all[cc_ii][emb_ind].append(xy)
                complex_tracks_all[cc_ii][emb_ind].append(xytrack) # this is not unique!!!!. 
                complex_times_all[cc_ii][emb_ind].append(xytime)
                complex_track_times_all[cc_ii][emb_ind].append(ee[1]['Frame'].values)
                
            # # plot here is just for debugging!. 
            # fig, ax = plt.subplots(nrows=1, ncols=len(uniq_embryo), figsize=(15,15))
            # for ii in np.arange(len(uniq_embryo)):
            #     ax[ii].set_title(uniq_embryo[ii])
            #     ax[ii].imshow(TP0_frames[ii], cmap='gray')
            #     ax[ii].grid('off')
            #     ax[ii].axis('off')
                
            # # now iterate and plot the xy points. 
            # for ee in ent_data:
            #     emb = ee[0]['embryo']
            #     emb_ind = uniq_embryo_ind[emb]
                
            #     xy = np.hstack([ee[0]['pos_2D_x_rot-AVE'], 
            #                     ee[0]['pos_2D_y_rot-AVE']])
                
            #     ax[emb_ind].plot(xy[0], xy[1], '.', color=NSWE_colors[cc_ii])
        else:
            print('hello')
            ent_data = batch_map_stats_back_to_image(pts_ids, 
                                                    metadict_id_array, 
                                                    embryo_file_dict,
                                                    return_global=False)    
            # now iterate and plot the xy points. 
            for ee in ent_data:
                emb = ee[0]['embryo']
                emb_ind = uniq_embryo_ind[emb]
                
                xy = np.hstack([ee[0]['pos_2D_x_rot-AVE'], 
                                ee[0]['pos_2D_y_rot-AVE']])
                xytime = ee[0]['Frame'] - uniq_embryo_TP0[emb_ind]
                
                xytrack = np.vstack([ee[1]['pos_2D_x_rot-AVE'], 
                                      ee[1]['pos_2D_y_rot-AVE']]).T
                
                complex_pts_all[cc_ii][emb_ind].append(xy)
                complex_tracks_all[cc_ii][emb_ind].append(xytrack)
                complex_times_all[cc_ii][emb_ind].append(xytime)
                complex_track_times_all[cc_ii][emb_ind].append(ee[1]['Frame'].values)
                
            #     ax[emb_ind].plot(xy[0], xy[1], '.', color=NSWE_colors[cc_ii], ms=1)
            
    # plt.show()
    
# =============================================================================
#     Mapback postional plots. 
# =============================================================================
    
    """
    Plot all the points on one image.... and for each embryo load in the TP_mig boundary lines...... (save those..... for easy usage here. )
    """



    n_embs = len(complex_pts_all[0])
    
    for emb_ii in np.arange(n_embs):
        
        ### grab the contour. 
        emb = uniq_embryo[emb_ii]
        embmatfile = os.path.join(imgfolder, emb+'_mig_Epi-contour-line.mat')
        embmatobj = spio.loadmat(embmatfile)
        # spio.savemat(savematfile, 
        #              {'epi_contours' : epi_contour_line, 
        #               'epi_contour_TPmig' : epi_contour_line[TP_mig_start],
        #               'TP_mig_start' : TP_mig_start,
        #               'center' : rot_center,
        #               'embryo' : emb, 
        #               'flipping' : flipping })
        epi_contour_line = np.squeeze(embmatobj['epi_contour_TPmig'])
        polar_center = np.squeeze(embmatobj['center'])
        
        fig, ax = plt.subplots(figsize=(15,15))
        ax.set_title(uniq_embryo[emb_ii])
        ax.imshow(TP0_frames[emb_ii], cmap='gray')
        ax.grid('off')
        ax.axis('off')

        for cc_ii in np.arange(len(complex_pts_all)):
            
            pts = complex_pts_all[cc_ii][emb_ii]
            pts = np.vstack(pts)
            ax.plot(pts[:,0], 
                    pts[:,1], '.', color=NSWE_colors[cc_ii])
        ax.plot(epi_contour_line[:,0], 
                epi_contour_line[:,1], 'r--', lw=5)
        ax.plot(polar_center[0],
                polar_center[1], color='w', marker='P', markersize=20)
        fig.savefig(os.path.join(saveplotfolder_final, 'PCA_pullback_xy_pts_%s_w_contour.svg' %(uniq_embryo[emb_ii])), 
                    bbox_inches='tight', 
                    dpi=300)
        plt.show()
        
    saveplotfolder_final_region = os.path.join(saveplotfolder_final, 'xy_pts_umap_cluster')
    fio.mkdir(saveplotfolder_final_region)
    
    
    for emb_ii in np.arange(n_embs):
        
        emb = uniq_embryo[emb_ii]
        embmatfile = os.path.join(imgfolder, emb+'_mig_Epi-contour-line.mat')
        embmatobj = spio.loadmat(embmatfile)
        epi_contour_line = np.squeeze(embmatobj['epi_contour_TPmig'])
        polar_center = np.squeeze(embmatobj['center'])
        
        for cc_ii in np.arange(len(complex_pts_all)):
            
            fig, ax = plt.subplots(figsize=(15,15))
            ax.set_title(uniq_embryo[emb_ii])
            ax.imshow(TP0_frames[emb_ii], cmap='gray')
            ax.grid('off')
            ax.axis('off')
            
            pts = complex_pts_all[cc_ii][emb_ii]
            pts = np.vstack(pts)
            ax.plot(pts[:,0], 
                    pts[:,1], '.', color=NSWE_colors[cc_ii])
            
            ax.plot(epi_contour_line[:,0], 
                epi_contour_line[:,1], 'r--', lw=5)
            ax.plot(polar_center[0],
                polar_center[1], color='w', marker='P', markersize=20)
            
            fig.savefig(os.path.join(saveplotfolder_final_region, 'PCA_pullback_xy_pts_%s_Grp%s_w_contour.svg' %(uniq_embryo[emb_ii], str(cc_ii))), 
                        bbox_inches='tight', 
                        dpi=300)
            plt.show()
    
    
        
    # for emb_ii in np.arange(n_embs):
        
    #     fig, ax = plt.subplots(figsize=(15,15))
    #     ax.set_title(uniq_embryo[emb_ii])
    #     ax.imshow(TP0_frames[emb_ii], cmap='gray')
    #     ax.grid('off')
    #     ax.axis('off')

    #     for cc_ii in np.arange(len(complex_pts_all)):
            
    #         pts = complex_pts_all[cc_ii][emb_ii]
    #         pts = np.vstack(pts)
    #         pts_times = complex_times_all[cc_ii][emb_ii]
    #         pts_times = np.hstack(pts_times)* 5./60 # h.
    #         # ax.scatter(pts[:,0], 
    #                    # pts[:,1], c=pts_times<=0, cmap='coolwarm', vmin=-2, vmax=4)
    #         ax.scatter(pts[:,0], 
    #                    pts[:,1], c=pts_times<=0, cmap='coolwarm', vmin=0, vmax=1)
    #     fig.savefig(os.path.join(saveplotfolder_final, 'PCA_pullback_xy_pts_tracks.svg'), bbox_inches='tight')
    #     plt.show()
        
        
    # # using time to gate. 
    # for emb_ii in np.arange(n_embs):
        
    #     fig, ax = plt.subplots(figsize=(15,15))
    #     ax.set_title(uniq_embryo[emb_ii])
    #     ax.imshow(TP0_frames[emb_ii], cmap='gray')
    #     ax.grid('off')
    #     ax.axis('off')

    #     for cc_ii in np.arange(len(complex_pts_all)):
            
    #         pts = complex_pts_all[cc_ii][emb_ii]
    #         pts = np.vstack(pts)
    #         pts_times = complex_times_all[cc_ii][emb_ii]
    #         pts_times = np.hstack(pts_times)* 5./60 # h.
    #         pts_mask = pts_times<=0
    #         ax.plot(pts[pts_mask,0], 
    #                 pts[pts_mask,1], 'o', color=NSWE_colors[cc_ii])
            
    #     plt.show()
        
        
    # where is the blue? 
    for emb_ii in np.arange(n_embs):
        
        emb = uniq_embryo[emb_ii]
        embmatfile = os.path.join(imgfolder, emb+'_mig_Epi-contour-line.mat')
        embmatobj = spio.loadmat(embmatfile)
        epi_contour_line = np.squeeze(embmatobj['epi_contour_TPmig'])
        polar_center = np.squeeze(embmatobj['center'])
        
        fig, ax = plt.subplots(figsize=(15,15))
        ax.set_title(uniq_embryo[emb_ii])
        ax.imshow(TP0_frames[emb_ii], cmap='gray')
        ax.grid('off')
        ax.axis('off')

        for cc_ii in np.arange(len(complex_tracks_all)):
            
            pts = complex_pts_all[cc_ii][emb_ii]
            pts = np.vstack(pts)
            ax.plot(pts[:,0], 
                    pts[:,1], 'k.')# 
            
            tracks = complex_tracks_all[cc_ii][emb_ii]
            for tra in tracks:
                ax.plot(tra[:,0], 
                        tra[:,1], '-', color=NSWE_colors[cc_ii]) 
                
        ax.plot(epi_contour_line[:,0], 
                epi_contour_line[:,1], 'r--', lw=5)
        ax.plot(polar_center[0],
            polar_center[1], color='w', marker='P', markersize=20)
                
        fig.savefig(os.path.join(saveplotfolder_final, 'PCA_pullback_xy_pts_tracks_%s_w_contour.svg' %(uniq_embryo[emb_ii])), bbox_inches='tight')
        plt.show()
        
        # separate this all out. 
        
    saveplotfolder_final_region = os.path.join(saveplotfolder_final, 'xy_pts_tracks_umap_cluster')
    fio.mkdir(saveplotfolder_final_region)
    
    
    for emb_ii in np.arange(n_embs):
        
        emb = uniq_embryo[emb_ii]
        embmatfile = os.path.join(imgfolder, emb+'_mig_Epi-contour-line.mat')
        embmatobj = spio.loadmat(embmatfile)
        epi_contour_line = np.squeeze(embmatobj['epi_contour_TPmig'])
        polar_center = np.squeeze(embmatobj['center'])
        
        for cc_ii in np.arange(len(complex_pts_all)):
            
            fig, ax = plt.subplots(figsize=(15,15))
            ax.set_title(uniq_embryo[emb_ii])
            ax.imshow(TP0_frames[emb_ii], cmap='gray')
            ax.grid('off')
            ax.axis('off')
            
            # pts = complex_pts_all[cc_ii][emb_ii]
            # pts = np.vstack(pts)
            # ax.plot(pts[:,0], 
            #         pts[:,1], '.', color=NSWE_colors[cc_ii])
            tracks = complex_tracks_all[cc_ii][emb_ii]
            for tra in tracks:
                ax.plot(tra[:,0], 
                        tra[:,1], '-', color=NSWE_colors[cc_ii]) 
            
            ax.plot(epi_contour_line[:,0], 
                epi_contour_line[:,1], 'r--', lw=5)
            ax.plot(polar_center[0],
                polar_center[1], color='w', marker='P', markersize=20)
            
            fig.savefig(os.path.join(saveplotfolder_final_region, 'PCA_pullback_xy_pts_tracks_%s_Grp%s_w_contour.svg' %(uniq_embryo[emb_ii], str(cc_ii))), 
                        bbox_inches='tight', 
                        dpi=300)
            plt.show()
    
        
    # # plot the track before tp 0
    # for emb_ii in np.arange(n_embs):
        
    #     fig, ax = plt.subplots(figsize=(15,15))
    #     ax.set_title(uniq_embryo[emb_ii])
    #     ax.imshow(TP0_frames[emb_ii], cmap='gray')
    #     ax.grid('off')
    #     ax.axis('off')

    #     for cc_ii in np.arange(len(complex_tracks_all)):
            
    #         pts = complex_pts_all[cc_ii][emb_ii]
    #         pts = np.vstack(pts)
    #         # ax.plot(pts[:,0], 
    #                 # pts[:,1], 'k.')# 
    #         # pts_times = complex_times_all[cc_ii][emb_ii]
    #         # pts_times = np.hstack(pts_times)* 5./60 # h.
    #         # pts_mask = pts_times<=0
            
    #         tracks = complex_tracks_all[cc_ii][emb_ii]
    #         tracks_time = complex_track_times_all[cc_ii][emb_ii]
    #         for tra_ii, tra in enumerate(tracks):
    #             plot_range = (tracks_time[tra_ii]-uniq_embryo_TP0[emb_ii])<=0
    #             if np.sum(plot_range) > 0: 
    #                 ax.plot(tra[plot_range,0], 
    #                         tra[plot_range,1], '-', color=NSWE_colors[cc_ii]) 
    
    #     plt.show()



        