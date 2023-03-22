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




# =============================================================================
#   wrapping a function to infer trajectory. 
# =============================================================================


def compute_umap_trajectory( umap_coords, umap_kmeans_cluster, gate_select, time_discrete, frame_all, resampling_grid_size, max_lim, tri, Ts=5./60) :
    
    
        print(max_lim, resampling_grid_size)
        
        pca_proj = umap_coords.copy()
        gate_prox_epi = gate_select.copy()
        
        # this restricts the main condition. 
        times_all_gate = frame_all[gate_prox_epi].copy()
        pca_proj_gate = pca_proj[gate_prox_epi].copy()
        pca_proj_gate_cluster = umap_kmeans_cluster[gate_prox_epi].copy()
        
        plt.figure()
        plt.plot(pca_proj[:,0], 
                 pca_proj[:,1], 
                 '.', color='lightgrey')
        plt.plot(pca_proj_gate[:,0], 
                 pca_proj_gate[:,1], 
                 '.', color='k')
        plt.show()
        
        # pca_proj_gate_cluster = selection_vector[gate_prox_epi] # this could be anything.... 
        
        # # don't trust this. 
        # times_all_gate_bins = np.digitize(times_all_gate*5/60., time_discrete ).astype(np.int) # convert to hours. # the invalid bin = 0 !
        # times_all_gate_bins = times_all_gate_bins - 1
        times_all_gate_bins = -1*np.ones(len(times_all_gate))
        for bin_ii in np.arange(len(time_discrete)-1):
            select = np.logical_and(times_all_gate*Ts >= time_discrete[bin_ii], 
                                    times_all_gate*Ts <= time_discrete[bin_ii+1])
            times_all_gate_bins[select] = bin_ii
        valid = times_all_gate_bins >= 0 # allow 0. 
        
        # gate the current PCA.
        pca_proj_gate = pca_proj_gate[valid].copy()
        pca_proj_gate_cluster = pca_proj_gate_cluster[valid]
        times_all_gate_bins = times_all_gate_bins[valid].copy()
        
        print(times_all_gate_bins)
        
        times_all_pts_line = []
        times_all_pts_contour_thresh = [] # records the density threshold used. 
        times_all_contours = [] # records all the contours -> best for visualisation. 
        times_all_occupancy = [] # make a note of the occupancy.... for each voronoi region. 
        density_images_tri = [] # 
        density_images_voronoi = []
        density_images_smooth = [] # this is the final image we use. 
        density_images_smooth_sigma = []
        # =============================================================================
        #   plot the evolution. 
        # =============================================================================
        
        max_pts = []
        
        for iii in np.arange(len(time_discrete)-1)[:]:
            
            select_plot = times_all_gate_bins == iii
            times_all_occupancy.append(select_plot.sum()) # total number of points that were gated.             
            
            """
            get the x,y positions to produce an image 
            """
            xy = pca_proj_gate[select_plot].T
            
            # do in addition a delaunay count... to supplement measurements. 
            grid_tri = np.zeros(resampling_grid_size)
            xy_tranform = (xy.T-(max_lim[0][0]))/float(max_lim[0][1] - max_lim[0][0]) * resampling_grid_size[0]
            
            # hull_areas = []
            for iii in np.arange(len(tri.simplices)):
                triangle_iii = tri.simplices[iii]
                triangle_iii_pts = tri.points[triangle_iii] # 3 x 2 
                triangle_iii_pts = (triangle_iii_pts-(max_lim[0][0]))/float(max_lim[0][1] - max_lim[0][0]) * resampling_grid_size[0]
                rr, cc = polygon(triangle_iii_pts[:,0], triangle_iii_pts[:,1], shape=grid_tri.shape)
                binary = np.zeros(grid_tri.shape)
                binary[rr,cc] = 1
                xy_transform_count = binary[xy_tranform[:,0].astype(np.int), 
                                            xy_tranform[:,1].astype(np.int)]
                grid_tri[rr,cc] = np.sum(xy_transform_count)
               
            xy_cluster = pca_proj_gate_cluster[select_plot] # region!
            # # do a count. 
            counts_cluster = np.hstack([np.sum(xy_cluster==clust) for clust in cluster_ids])
            # # map this to a mesh. 
            # # cluster_centroids_ = (cluster_ids_centroids-(-5))/10. * 200
            grid = np.zeros(resampling_grid_size)
            
            hull_areas = []
            for hull_ii, hull_ in enumerate(cluster_hull):
                hull = hull_.copy()
                hull[:,0] = (hull[:,0] - max_lim[0][0])/float(max_lim[0][1] - max_lim[0][0]) * resampling_grid_size[0]
                hull[:,1] = (hull[:,1] - max_lim[1][0])/float(max_lim[1][1] - max_lim[1][0]) * resampling_grid_size[1]
                # hull = (hull-(-6))/12. * 500
                rr, cc = polygon(hull[:,0], hull[:,1], shape=grid.shape)
                grid[rr,cc] = counts_cluster[hull_ii]
                hull_areas.append(len(rr))
                
            # append grids before averaging. 
            density_images_tri.append(grid_tri)
            density_images_voronoi.append(grid)
                
            # merge.
            grid = .5*(grid+grid_tri) # naively. 
            smooth_sigma = np.sqrt(np.mean(np.hstack(hull_areas))) *1. #*1.5 #/ 2. #* 1.5 # when only single. 
            density_images_smooth_sigma.append(smooth_sigma)
            
            # smooth 
            grid = gaussian(grid, sigma=smooth_sigma, preserve_range=True)
            
            density_images_smooth.append(grid) # this is the final image we use.     
            times_all_pts_contour_thresh.append(np.mean(grid)+2*np.std(grid))
            grid_cont = find_contours(grid, np.mean(grid)+2*np.std(grid))
            
            times_all_contours.append(grid_cont) # records all the contours -> best for visualisation. 
                
            # # this is for the line only.
            # grid_max_pos = np.argmax(grid)
            # grid_max_pos = np.unravel_index(grid_max_pos, shape=grid.shape)
            # max_pts.append(grid_max_pos)
            
            # lets do the mean instead. 
            binary_density = grid>=np.mean(grid)+2*np.std(grid)
            grid_max_pos = np.mean(np.array(np.where(binary_density>0)).T, axis=0)
            max_pts.append(grid_max_pos)
            
            plt.figure()
            plt.imshow(grid, cmap='coolwarm')
            plt.plot(grid_max_pos[1], 
                     grid_max_pos[0], 'go')
            for cc in grid_cont:
                plt.plot(cc[:,1], 
                         cc[:,0], 'g-')
            for hull_ii, hull_ in enumerate(cluster_hull):
                hull = hull_.copy()
                # hull = (hull-(-6))/12. * 500
                hull = (hull-(max_lim[0][0]))/float(max_lim[0][1] - max_lim[0][0]) * resampling_grid_size[0]
                plt.plot(hull[:,1], 
                         hull[:,0], 'k-')
            plt.show()
            
        
        max_pts = np.vstack(max_pts)  
        times_all_pts_line =  max_pts.copy().astype(np.float) # else will all be casted!. 
        times_all_pts_line[:,0] = times_all_pts_line[:,0]/float(resampling_grid_size[0]) * (max_lim[0][1] - max_lim[0][0]) + max_lim[0][0] 
        times_all_pts_line[:,1] = times_all_pts_line[:,1]/float(resampling_grid_size[1]) * (max_lim[1][1] - max_lim[1][0]) + max_lim[1][0]
        # max_pts/500.*12 + (-6.)
        # resampling_grid_size = (500,500) # 
        # max_lim = [[-6,6], [-6,6]] # make the same here. # in general we don't....  

        return times_all_pts_line, times_all_pts_contour_thresh, times_all_contours, times_all_occupancy, density_images_tri, density_images_voronoi , density_images_smooth, density_images_smooth_sigma


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
    saveplotfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-19_PCA_analysis'
    # saveplotfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-24_UMAP_analysis'
    
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
    epi_contour_dist = saveobj['epi_contour_dist'].ravel()
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
    
    embryo_track_id_all = np.hstack([embryo_all[iii] +'_' + str(track_id_all[iii]).zfill(3) for iii in np.arange(len(track_id_all))])
    
    # row_id': row_id_all, 
    #               'TP':TP_all, 
    #               'cell_id':cell_id_all, 
    #               'track_id':track_id_all
    

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

    
    
    
# # =============================================================================
# #     Plots of the 8 regions onto UMAP space. 
# # =============================================================================

#     saveplotfolder_umap_region = os.path.join(saveplotfolder, 'UMAP_8main_regions')
#     fio.mkdir(saveplotfolder_umap_region)

#     from scipy.stats import gaussian_kde

#     for region_ii in np.arange(1,8+1):
        
#         select_region = dynamic_quad_id_main.astype(np.int) == region_ii
        
#         fig, ax = plt.subplots(figsize=(15,15))
#         ax.scatter(pca_proj[:,0], 
#                    pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
        
#         # graph by density? 
        
#         # Calculate the point density
#         x = pca_proj[select_region,0]; y = pca_proj[select_region,1]
#         xy = np.vstack([x,y])
#         z = gaussian_kde(xy)(xy)
        
#         # Sort the points by density, so that the densest points are plotted last
#         idx = z.argsort()
#         x, y, z = x[idx], y[idx], z[idx]
        
#         # ax.scatter(pca_proj[select_region,0],
#         #            pca_proj[select_region,1], 
#         #            color='k')
#         ax.scatter(x, y, c=z, cmap='coolwarm') #, s=50)
            
#         ax.set_aspect(1)
#         plt.grid('off')
#         plt.axis('off')
#         plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_points_Region-%s_density.png' %(str(region_ii).zfill(2))), 
#                     dpi=600,
#                     bbox_inches='tight')
#         # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
#         plt.show()
    




# =============================================================================
#     Include the information from the regional clusters. 
# =============================================================================

    
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
   
    
    region_cluster_labels_colors = np.vstack(sns.color_palette('colorblind', 16))#[np.unique(region_cluster_labels)[1:]]
    all_umap_cluster_colors = np.vstack(region_cluster_labels_colors)[np.unique(umap_cluster_labels)]
    from matplotlib import colors
    # print(colors.to_rgba('lightgrey'))
    all_umap_cluster_colors[0] = np.hstack(colors.to_rgba('lightgrey')[:3])
    all_umap_cluster_colors = all_umap_cluster_colors[[0,2,3,1,4]] # ok so this is the new coloring for the umap positions ! 
    
    all_umap_phenotype_label_colors = np.zeros((len(umap_cluster_labels),3))
    for clust_ii in np.arange(len(np.unique(umap_cluster_labels))):
        all_umap_phenotype_label_colors[umap_cluster_labels==np.unique(umap_cluster_labels)[clust_ii]] = all_umap_cluster_colors[clust_ii]
        
    
    saveplotfolder_umap_region = os.path.join(saveplotfolder, 'UMAP_phenomic_trajectories_regions')
    fio.mkdir(saveplotfolder_umap_region)
    
    
    # create a gating by time .... to do the trajectory analysis 
# =============================================================================
#     Derive trajectories for the phenomic clusters including -1 
# =============================================================================

    unique_umap_clusters = np.unique(umap_cluster_labels)


# =============================================================================
#   Construct the delauny!. 
# =============================================================================
    from scipy.spatial import Delaunay
    
    tri = Delaunay(cluster_ids_centroids) 
    
    def angle_tri( verts, faces, max_angle=90):
        
        # use cosine rule to get all angles and remove triangles that contain super obtuse angle!. 
        face_pts = verts[faces] # n x 3 x 2
        a = np.linalg.norm(face_pts[:,1]-face_pts[:,2], axis=-1)
        b = np.linalg.norm(face_pts[:,0]-face_pts[:,2], axis=-1)
        c = np.linalg.norm(face_pts[:,0]-face_pts[:,1], axis=-1)
        
        a2 = a**2
        b2 = b**2
        c2 = c**2
        
        # cosine law. 
        alpha = np.arccos((b2 + c2 - a2) / (2*b*c))
        beta = np.arccos((a2 + c2 - b2) / (2*a*c))
        gamma = np.arccos((a2 + b2 - c2) / (2*a*b))
        
        # convert to degree
        alpha = alpha * 180. / np.pi
        beta = beta * 180. / np.pi 
        gamma = gamma * 180. / np.pi
        
        remove_triangles = np.logical_or(alpha>=max_angle, np.logical_or(beta>=max_angle, gamma>=max_angle))
        
        return remove_triangles
    
    # #function to get vertex neigbors
    # def find_neighbors(pindex, triang):
    #     return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]
    find_neighbors = lambda x,triang: list(set(indx for simplex in triang.simplices if x in simplex for indx in simplex if indx !=x))
    
    tri_angle_remove = angle_tri( tri.points, tri.simplices, max_angle=120)
    tri.simplices = tri.simplices[~tri_angle_remove] # update
    # pca_proj_tri_ids = tri.find_simplex(pca_proj) # this doesn't quite work... so instead we will try and do this more generally from image. 
    

    # generate for supplementary figures.  
    fig, ax = plt.subplots(figsize=(15,15))
    for hull_ii, hull in enumerate(cluster_hull):
        ax.plot(hull[:,0], 
                hull[:,1], 'k-')
    ax.triplot(tri.points[:,0],
               tri.points[:,1], 
               tri.simplices, color='r')
    # plt.triplot(cluster_ids_centroids[:,0], 
    #             cluster_ids_centroids[:,1], tri.simplices, color='g', lw=5, alpha=.5)
    ax.set_aspect(1)
    plt.show()      
    
    
    """
    save the processing properties. 
    """
    # save the mesh properties (delaunay and voronoi with the pca points. )
    savemeshfile = os.path.join(saveplotfolder, 'delaunay_voronoi_regions.mat')
    spio.savemat(savemeshfile, 
                 {'voronoi': convexhull_pts, 
                  'delaunay_pts': tri.points, 
                  'delaunay_simplex': tri.simplices})


    from scipy.stats import gaussian_kde # why is this different? 
    from skimage.filters import threshold_otsu
    
    """
    Set the parameters of the analysis.
    """
    # can also tile with squares -> occupancy issues... 
    resampling_grid_size = (500,500) # 
    # max_lim = [[-6,6], [-6,6]] # make the same here. # in general we don't....  
    max_lim = [[-2,12], [-2,12]] # must be larger than the range of umap space. 
    
    time_discrete = np.linspace(-3,6, 9+1) # only up to 4 hrs. 
    # # time_discrete = np.linspace(-3,6, 2*9+1) # only up to 4 hrs. 
    # time_discrete = np.linspace(-2,6, 2*8+1) # only up to 4 hrs.
    # time_discrete = np.linspace(-1,4, 1*5+1) # only up to 4 hrs.  
    
    threshold = 2 # later timepoints is cut ! -> 
    # set up the time colors. 
    TP_start = np.arange(len(time_discrete))[time_discrete ==0][0]
    time_discrete_colors = np.vstack([sns.color_palette('coolwarm', 2*(TP_start+1)+1)])[:(TP_start)]
    time_discrete_colors = np.vstack([time_discrete_colors, 
                                      sns.color_palette('magma_r', len(time_discrete)-TP_start)])
    
    # check the time discrete_colors.     
    import seaborn as sns
    # import matplotlib.tri as tri
    from skimage.draw import polygon
    from skimage.filters import gaussian
    from skimage.measure import find_contours
    
    plt.figure()
    sns.palplot(time_discrete_colors)
    plt.show()

    

    """
    Main derivation.
    """
    all_times_all_pts_line = []
    all_times_all_pts_contour_thresh = [] # records the density threshold used. 
    all_times_all_contours = [] # records all the contours -> best for visualisation. 
    all_times_all_occupancy = [] # make a note of the occupancy.... for each voronoi region. 
    all_density_images_tri = [] # 
    all_density_images_voronoi = []
    all_density_images_smooth = [] # this is the final image we use. 
    all_density_images_smooth_sigma = []
    
    
    # order_cluster = np.hstack([1,0,3,2,-1]) + 1
    
    # use the dynamic clusters
    order_cluster = np.arange(8)
    
    # for quad_id in unique_umap_clusters[order_cluster]:#[4:5]: # rearrange to the same order as paper! 
    for quad_id in np.arange(1,8+1):
        gate_prox_epi = umap_cluster_labels == quad_id
        times_all_gate = frame_all[gate_prox_epi]
        pca_proj_gate = pca_proj[gate_prox_epi]
        pca_proj_gate_cluster = dynamic_quad_id_main[gate_prox_epi]
        
        # # don't trust this. 
        # times_all_gate_bins = np.digitize(times_all_gate*5/60., time_discrete ).astype(np.int) # convert to hours. # the invalid bin = 0 !
        # times_all_gate_bins = times_all_gate_bins - 1
        times_all_gate_bins = -1*np.ones(len(times_all_gate))
        for bin_ii in np.arange(len(time_discrete)-1):
            select = np.logical_and(times_all_gate*5/60. >= time_discrete[bin_ii], 
                                    times_all_gate*5/60. <= time_discrete[bin_ii+1])
            times_all_gate_bins[select] = bin_ii
        valid = times_all_gate_bins >= 0 # allow 0. 
        
        # gate the current PCA.
        pca_proj_gate = pca_proj_gate[valid]
        pca_proj_gate_cluster = pca_proj_gate_cluster[valid]
        times_all_gate_bins = times_all_gate_bins[valid]
        
        
        times_all_pts_line = []
        times_all_pts_contour_thresh = [] # records the density threshold used. 
        times_all_contours = [] # records all the contours -> best for visualisation. 
        times_all_occupancy = [] # make a note of the occupancy.... for each voronoi region. 
        density_images_tri = [] # 
        density_images_voronoi = []
        density_images_smooth = [] # this is the final image we use. 
        density_images_smooth_sigma = []
        # =============================================================================
        #   plot the evolution. 
        # =============================================================================
        
        max_pts = []
        
        for iii in np.arange(len(time_discrete)-1)[:]:
            
            select_plot = times_all_gate_bins == iii
            times_all_occupancy.append(select_plot.sum()) # total number of points that were gated.             
            
            """
            get the x,y positions to produce an image 
            """
            xy = pca_proj_gate[select_plot].T
            
            # do in addition a delaunay count... to supplement measurements. 
            grid_tri = np.zeros(resampling_grid_size)
            xy_tranform = (xy.T-(max_lim[0][0]))/float(max_lim[0][1] - max_lim[0][0]) * resampling_grid_size[0]
            
            # hull_areas = []
            for iii in np.arange(len(tri.simplices)):
                triangle_iii = tri.simplices[iii]
                triangle_iii_pts = tri.points[triangle_iii] # 3 x 2 
                triangle_iii_pts = (triangle_iii_pts-(max_lim[0][0]))/float(max_lim[0][1] - max_lim[0][0]) * resampling_grid_size[0]
                rr, cc = polygon(triangle_iii_pts[:,0], triangle_iii_pts[:,1], shape=grid_tri.shape)
                binary = np.zeros(grid_tri.shape)
                binary[rr,cc] = 1
                xy_transform_count = binary[xy_tranform[:,0].astype(np.int), 
                                            xy_tranform[:,1].astype(np.int)]
                grid_tri[rr,cc] = np.sum(xy_transform_count)
               
            xy_cluster = pca_proj_gate_cluster[select_plot]
            # do a count. 
            counts_cluster = np.hstack([np.sum(xy_cluster==clust) for clust in cluster_ids])
            # map this to a mesh. 
            # cluster_centroids_ = (cluster_ids_centroids-(-5))/10. * 200
            grid = np.zeros(resampling_grid_size)
            
            hull_areas = []
            for hull_ii, hull_ in enumerate(cluster_hull):
                hull = hull_.copy()
                hull[:,0] = (hull[:,0] - max_lim[0][0])/float(max_lim[0][1] - max_lim[0][0]) * resampling_grid_size[0]
                hull[:,1] = (hull[:,1] - max_lim[1][0])/float(max_lim[1][1] - max_lim[1][0]) * resampling_grid_size[1]
                # hull = (hull-(-6))/12. * 500
                rr, cc = polygon(hull[:,0], hull[:,1], shape=grid.shape)
                grid[rr,cc] = counts_cluster[hull_ii]
                hull_areas.append(len(rr))
                
            # append grids before averaging. 
            density_images_tri.append(grid_tri)
            density_images_voronoi.append(grid)
                
            # merge.
            grid = .5*(grid+grid_tri) # naively. 
            smooth_sigma = np.sqrt(np.mean(np.hstack(hull_areas))) *1. #*1.5 #/ 2. #* 1.5 # when only single. 
            density_images_smooth_sigma.append(smooth_sigma)
            # smooth 
            grid = gaussian(grid, sigma=smooth_sigma, preserve_range=True)
            
            density_images_smooth.append(grid) # this is the final image we use.     
            times_all_pts_contour_thresh.append(np.mean(grid)+2*np.std(grid))
            grid_cont = find_contours(grid, np.mean(grid)+2*np.std(grid))
            
            times_all_contours.append(grid_cont) # records all the contours -> best for visualisation. 
                
            # # this is for the line only.
            # grid_max_pos = np.argmax(grid)
            # grid_max_pos = np.unravel_index(grid_max_pos, shape=grid.shape)
            # max_pts.append(grid_max_pos)
            
            # lets do the mean instead. 
            binary_density = grid>=np.mean(grid)+2*np.std(grid)
            grid_max_pos = np.mean(np.array(np.where(binary_density>0)).T, axis=0)
            max_pts.append(grid_max_pos)
            
            plt.figure()
            plt.imshow(grid, cmap='coolwarm')
            plt.plot(grid_max_pos[1], 
                      grid_max_pos[0], 'go')
            for cc in grid_cont:
                plt.plot(cc[:,1], 
                          cc[:,0], 'g-')
            for hull_ii, hull_ in enumerate(cluster_hull):
                hull = hull_.copy()
                # hull = (hull-(-6))/12. * 500
                hull = (hull-(max_lim[0][0]))/float(max_lim[0][1] - max_lim[0][0]) * resampling_grid_size[0]
                plt.plot(hull[:,1], 
                          hull[:,0], 'k-')
            plt.show()
            
        
        max_pts = np.vstack(max_pts)  
        times_all_pts_line =  max_pts.copy().astype(np.float) # else will all be casted!. 
        times_all_pts_line[:,0] = times_all_pts_line[:,0]/float(resampling_grid_size[0]) * (max_lim[0][1] - max_lim[0][0]) + max_lim[0][0] 
        times_all_pts_line[:,1] = times_all_pts_line[:,1]/float(resampling_grid_size[1]) * (max_lim[1][1] - max_lim[1][0]) + max_lim[1][0]
        # max_pts/500.*12 + (-6.)
        # resampling_grid_size = (500,500) # 
        # max_lim = [[-6,6], [-6,6]] # make the same here. # in general we don't....  
        
        plt.figure()
        plt.title(str(quad_id))
        plt.imshow(grid, cmap='coolwarm', alpha=0)
        for hull_ii, hull in enumerate(cluster_hull):
            hull = (hull-(max_lim[0][0]))/float(max_lim[0][1] - max_lim[0][0]) * resampling_grid_size[0]
            plt.plot(hull[:,1], 
                      hull[:,0], 'k-')
        plt.plot(np.vstack(max_pts)[:,1], 
                np.vstack(max_pts)[:,0], 'g-')
        plt.show()
        
        # transform back to normal space? 
        plt.figure(figsize=(15,15))
        for hull_ii, hull in enumerate(cluster_hull):
            # hull = (hull-(-6))/12. * 500
            plt.plot(hull[:,0], 
                      hull[:,1], 'k-')
        max_pts = np.vstack(max_pts)
        max_pts_ = max_pts/resampling_grid_size[0]*float(max_lim[0][1] - max_lim[0][0]) + (max_lim[0][0])
        plt.plot(max_pts_[:,0], 
                  max_pts_[:,1], 'g-')
        ax.set_aspect(1)
        plt.show()
        
        all_times_all_pts_line.append(np.vstack(times_all_pts_line))
        all_times_all_pts_contour_thresh.append(np.hstack(times_all_pts_contour_thresh)) # records the density threshold used. 
        all_times_all_contours.append(times_all_contours) # records all the contours -> best for visualisation. 
        all_times_all_occupancy.append(np.hstack(times_all_occupancy)) # make a note of the occupancy.... for each voronoi region. 
        all_density_images_tri.append(np.array(density_images_tri)) # 
        all_density_images_voronoi.append(np.array(density_images_voronoi))
        all_density_images_smooth.append(np.array(density_images_smooth)) # this is the final image we use. 
        all_density_images_smooth_sigma.append(np.hstack(density_images_smooth_sigma))
        
    
    """
    Save all this computation for now!. 
    """
    saveoutfile = os.path.join(saveplotfolder_umap_region, 
                                'umap_trajectories_all_quads_mean_SAVE.mat')
    
    region_cluster_labels_colors = np.vstack(sns.color_palette('colorblind', 16))[np.unique(region_cluster_labels)[1:]]
    
    spio.savemat(saveoutfile, 
                  {'all_traj_pts': all_times_all_pts_line, 
                  'all_traj_contours': all_times_all_pts_contour_thresh, 
                  'all_times_all_contours': all_times_all_contours,
                  'all_traj_occupancy': all_times_all_occupancy, 
                  'all_density_tri_mesh': all_density_images_tri, 
                  'all_density_voronoi_mesh': all_density_images_voronoi,
                  'all_density_images_smooth_mesh': all_density_images_smooth, 
                  'all_density_images_smooth_sigma': all_density_images_smooth_sigma, 
                  'grid_size': np.hstack(resampling_grid_size), 
                  'max_lim': np.vstack(max_lim),
                  'time_discrete': np.hstack(time_discrete),
                  'time_discrete_colors': time_discrete_colors, 
                  'TP_start_index': TP_start, 
                  'voronoi_region_cluster_colors': region_cluster_labels_colors, # colors. 
                  'voronoi_region_cluster_labels': region_cluster_labels}) # unique labels. 
    
    
    
    """
    Plot the phenomic trajectory
    """
    time_discrete = np.hstack(time_discrete)
    
    # since these are in order we just iterate 
    for phenomic_clust_ii in np.arange(len(all_times_all_pts_line)):
        
        after_seg = .5*(time_discrete[1:] + time_discrete[:-1])>=0
        
        fig, ax = plt.subplots(figsize=(10,10))
        # ax.scatter(pca_proj[:,0], 
        #             pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
        ax.scatter(pca_proj[:,0], 
                   pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
                
        traj = all_times_all_pts_line[phenomic_clust_ii]
        ax.plot(traj[:,0], 
                traj[:,1], color = 'k', 
                lw=5) #, s=50)
        ax.plot(traj[after_seg,0], 
                traj[after_seg,1], color = 'green', 
                lw=5)
        # ax.plot(traj[:,0], 
        #         traj[:,1], 
        #         color = 'k',
        #         ms=10)
            
        ax.set_aspect(1)
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_PhenomicCluster-%s_colorbg.svg' %(str(phenomic_clust_ii).zfill(2))), 
                    dpi=600,
                    bbox_inches='tight')
        # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
        plt.show()
    



# =============================================================================
#     Specifically derive posterior as well as the ExE_VE(all....)
# =============================================================================

    gate_select = np.logical_or(dynamic_quad_id_main.astype(np.int) == 2, 
                                dynamic_quad_id_main.astype(np.int) == 4)
    
    # gate_select = dynamic_quad_id_main.astype(np.int) == 1
    # gate_select = umap_cluster_labels == 1
    
    traj_P_EpiVE = compute_umap_trajectory( pca_proj, 
                                            pca_proj_cluster,
                                            gate_select, 
                                            time_discrete, 
                                            frame_all, 
                                            resampling_grid_size, 
                                            max_lim, 
                                            tri, 
                                            Ts=5/60.) 


    fig, ax = plt.subplots(figsize=(10,10))
    # ax.scatter(pca_proj[:,0], 
    #             pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
    ax.scatter(pca_proj[:,0], 
                   pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
            
    traj = traj_P_EpiVE[0]
    ax.plot(traj[:,0], 
            traj[:,1], color = 'k', 
            lw=5) #, s=50)
    ax.plot(traj[after_seg,0], 
            traj[after_seg,1], color = 'green', 
            lw=5)
    
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_P-EpiVE_colorbg.svg' ), 
                dpi=600,
                bbox_inches='tight')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
    plt.show()
    
    
    
    gate_select = dynamic_quad_id_all >= 17
    
    # gate_select = dynamic_quad_id_main.astype(np.int) == 1
    # gate_select = umap_cluster_labels == 1
    
    traj_P_EpiVE = compute_umap_trajectory( pca_proj, 
                                            pca_proj_cluster,
                                            gate_select, 
                                            time_discrete, 
                                            frame_all, 
                                            resampling_grid_size, 
                                            max_lim, 
                                            tri, 
                                            Ts=5/60.) 


    fig, ax = plt.subplots(figsize=(10,10))
    # ax.scatter(pca_proj[:,0], 
                # pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
    ax.scatter(pca_proj[:,0], 
                   pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
            
    traj = traj_P_EpiVE[0]
    ax.plot(traj[:,0], 
            traj[:,1], color = 'k', 
            lw=5) #, s=50)
    ax.plot(traj[after_seg,0], 
            traj[after_seg,1], color = 'green', 
            lw=5)
    
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_ExE-VE_colorbg.svg' ), 
                dpi=600,
                bbox_inches='tight')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
    plt.show()


# =============================================================================
# =============================================================================
# =============================================================================
# # # Check the regions. 
# =============================================================================
# =============================================================================
# =============================================================================

    all_traj_regions_major8 = []    

    for reg_id in np.arange(1,8+1):
        
        gate_select = dynamic_quad_id_main == reg_id
        
        # gate_select = dynamic_quad_id_main.astype(np.int) == 1
        # gate_select = umap_cluster_labels == 1
        
        traj_P_EpiVE = compute_umap_trajectory( pca_proj, 
                                                pca_proj_cluster,
                                                gate_select, 
                                                time_discrete, 
                                                frame_all, 
                                                resampling_grid_size, 
                                                max_lim, 
                                                tri, 
                                                Ts=5/60.) 
    
    
        fig, ax = plt.subplots(figsize=(10,10))
        # ax.scatter(pca_proj[:,0], 
                    # pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
        ax.scatter(pca_proj[:,0], 
                   pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
                
        traj = traj_P_EpiVE[0]
        ax.plot(traj[:,0], 
                traj[:,1], color = 'k', 
                lw=5) #, s=50)
        ax.plot(traj[after_seg,0], 
                traj[after_seg,1], color = 'green', 
                lw=5)
        
        ax.set_aspect(1)
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_Dyn_Region-%d_colorbg.svg' %(reg_id) ), 
                    dpi=600,
                    bbox_inches='tight')
        # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
        plt.show()

        # save the trajectory. 
        all_traj_regions_major8.append(traj)


# =============================================================================
# =============================================================================
# =============================================================================
# # # Do the regions for prox/distal split of region 1.
# =============================================================================
# =============================================================================
# =============================================================================

    
    gate_select = dynamic_quad_id_main_split == 1 # use upper. 
    
    # gate_select = dynamic_quad_id_main.astype(np.int) == 1
    # gate_select = umap_cluster_labels == 1
    traj_P_EpiVE = compute_umap_trajectory( pca_proj, 
                                            pca_proj_cluster,
                                            gate_select, 
                                            time_discrete, 
                                            frame_all, 
                                            resampling_grid_size, 
                                            max_lim, 
                                            tri, 
                                            Ts=5/60.) 


    fig, ax = plt.subplots(figsize=(10,10))
    # ax.scatter(pca_proj[:,0], 
                # pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
    ax.scatter(pca_proj[:,0], 
                   pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
            
    traj = traj_P_EpiVE[0]
    ax.plot(traj[:,0], 
            traj[:,1], color = 'k', 
            lw=5) #, s=50)
    ax.plot(traj[after_seg,0], 
            traj[after_seg,1], color = 'green', 
            lw=5)
    
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_Region-1prox_colorbg.svg' ), 
                dpi=600,
                bbox_inches='tight')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
    plt.show()
    
    
    all_traj_regions_major8.append(traj) # this will add 1A. 
    
    
    gate_select = dynamic_quad_id_main_split == 9 # use lower. 
    
    # gate_select = dynamic_quad_id_main.astype(np.int) == 1
    # gate_select = umap_cluster_labels == 1
    traj_P_EpiVE = compute_umap_trajectory( pca_proj, 
                                            pca_proj_cluster,
                                            gate_select, 
                                            time_discrete, 
                                            frame_all, 
                                            resampling_grid_size, 
                                            max_lim, 
                                            tri, 
                                            Ts=5/60.) 


    fig, ax = plt.subplots(figsize=(10,10))
    # ax.scatter(pca_proj[:,0], 
                # pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
    ax.scatter(pca_proj[:,0], 
                   pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
            
    traj = traj_P_EpiVE[0]
    ax.plot(traj[:,0], 
            traj[:,1], color = 'k', 
            lw=5) #, s=50)
    ax.plot(traj[after_seg,0], 
            traj[after_seg,1], color = 'green', 
            lw=5)
    
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_Region-1distal_colorbg.svg' ), 
                dpi=600,
                bbox_inches='tight')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
    plt.show()


    all_traj_regions_major8.append(traj) # this will add 1B.
    
    
    
# =============================================================================
# =============================================================================
# #     Make coplot of the Epi-VE half 
# =============================================================================
# =============================================================================
    
    
    index_TP0 = np.arange(len(after_seg))[after_seg>0][0]
    
    track_labels = ['2', '3', '4', '1prox', '1distal']
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    # ax.scatter(pca_proj[:,0], 
                # pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
    ax.scatter(pca_proj[:,0], 
               pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
            
    # plot these as segments!. 
    
    # plot regions 2 - 4
    counter = 0
    for iii in np.arange(2-1,4-1+1):
        
        traj = all_traj_regions_major8[iii].copy()
        ax.scatter(traj[0,0],
                   traj[0,1], s=100, color='k')
        
        for index_ in np.arange(index_TP0):
            # ax.plot(traj[index_:index_+2,0], 
            #         traj[index_:index_+2,1], color = 'k', 
            #         lw=5) #, s=50)
            ax.plot([traj[index_,0], traj[index_+1,0]], 
                    [traj[index_,1], traj[index_+1,1]], 
                    color='k', lw=5)
            
        for index_ in np.arange(index_TP0, len(after_seg)-1):
            ax.plot(traj[index_:index_+2,0], 
                    traj[index_:index_+2,1], color = 'green', 
                    lw=5) #, s=50)
            
        # add in a label at the end point of the track. 
        ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
        ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
        
        ax.text(traj[-1,0],
                traj[-1,1], 
                track_labels[counter],
                va='center', 
                ha='center', 
                fontsize=18,
                fontname='Arial', zorder=1000)
        
        counter+=1
        
    # 1A 
    traj = all_traj_regions_major8[-2].copy()
    ax.scatter(traj[0,0],
               traj[0,1], s=100, color='k')
    for index_ in np.arange(index_TP0):
        # ax.plot(traj[index_:index_+2,0], 
        #         traj[index_:index_+2,1], color = 'k', 
        #         lw=5) #, s=50)
        ax.plot([traj[index_,0], traj[index_+1,0]], 
                [traj[index_,1], traj[index_+1,1]], 
                color='k', lw=5)
        
    for index_ in np.arange(index_TP0, len(after_seg)-1):
            ax.plot(traj[index_:index_+2,0], 
                    traj[index_:index_+2,1], color = 'green', 
                    lw=5) #, s=50)
    ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
    
    ax.text(traj[-1,0],
                traj[-1,1], 
                track_labels[-2],
                va='center', 
                ha='center', 
                fontsize=18,
                fontname='Arial', zorder=1000)
        
    # 1B
    traj = all_traj_regions_major8[-1].copy()
    ax.scatter(traj[0,0],
               traj[0,1], s=100, color='k')
    for index_ in np.arange(index_TP0):
        # ax.plot(traj[index_:index_+2,0], 
        #         traj[index_:index_+2,1], color = 'k', 
        #         lw=5) #, s=50)
        ax.plot([traj[index_,0], traj[index_+1,0]], 
                [traj[index_,1], traj[index_+1,1]], 
                color='k', lw=5)
        
    for index_ in np.arange(index_TP0, len(after_seg)-1):
            ax.plot(traj[index_:index_+2,0], 
                    traj[index_:index_+2,1], color = 'green', 
                    lw=5) #, s=50)
    ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
        
    ax.text(traj[-1,0],
                traj[-1,1], 
                track_labels[-1],
                va='center', 
                ha='center', 
                fontsize=18,
                fontname='Arial', zorder=1000)
    
    # Add plot of regions 1A, 1B
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_Region-Em-VE_colorbg.svg' ), 
                dpi=600,
                bbox_inches='tight')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
    plt.show()
    
    
    
    index_TP0 = np.arange(len(after_seg))[after_seg>0][0]
    
    track_labels = ['2', '3', '4', '1prox', '1distal']
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(pca_proj[:,0], 
                pca_proj[:,1], color='lightgrey', s=10) # color all the backgroud grey 
    # ax.scatter(pca_proj[:,0], 
    #            pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
            
    # plot these as segments!. 
    
    # plot regions 2 - 4
    counter = 0
    for iii in np.arange(2-1,4-1+1):
        
        traj = all_traj_regions_major8[iii].copy()
        ax.scatter(traj[0,0],
                   traj[0,1], s=100, color='k')
        ax.text(traj[0,0],
                traj[0,1], 
                track_labels[counter],
                va='center', 
                ha='center', 
                fontsize=24,
                fontname='Arial', zorder=1000)
        
        for index_ in np.arange(index_TP0):
            # ax.plot(traj[index_:index_+2,0], 
            #         traj[index_:index_+2,1], color = 'k', 
            #         lw=5) #, s=50)
            ax.plot([traj[index_,0], traj[index_+1,0]], 
                    [traj[index_,1], traj[index_+1,1]], 
                    color='k', lw=5)
            
        for index_ in np.arange(index_TP0, len(after_seg)-1):
            ax.plot(traj[index_:index_+2,0], 
                    traj[index_:index_+2,1], color = 'green', 
                    lw=5) #, s=50)
            
        # add in a label at the end point of the track. 
        ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
        ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
        
        ax.text(traj[-1,0],
                traj[-1,1], 
                track_labels[counter],
                va='center', 
                ha='center', 
                fontsize=18,
                fontname='Arial', zorder=1000)
        
        counter+=1
        
    # 1A 
    traj = all_traj_regions_major8[-2].copy()
    ax.scatter(traj[0,0],
               traj[0,1], s=100, color='k')
    ax.text(traj[0,0],
                traj[0,1], 
                track_labels[-2],
                va='center', 
                ha='center', 
                fontsize=24,
                fontname='Arial', zorder=1000)
    for index_ in np.arange(index_TP0):
        # ax.plot(traj[index_:index_+2,0], 
        #         traj[index_:index_+2,1], color = 'k', 
        #         lw=5) #, s=50)
        ax.plot([traj[index_,0], traj[index_+1,0]], 
                [traj[index_,1], traj[index_+1,1]], 
                color='k', lw=5)
        
    for index_ in np.arange(index_TP0, len(after_seg)-1):
            ax.plot(traj[index_:index_+2,0], 
                    traj[index_:index_+2,1], color = 'green', 
                    lw=5) #, s=50)
    ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
    
    ax.text(traj[-1,0],
                traj[-1,1], 
                track_labels[-2],
                va='center', 
                ha='center', 
                fontsize=18,
                fontname='Arial', zorder=1000)
        
    # 1B
    traj = all_traj_regions_major8[-1].copy()
    ax.scatter(traj[0,0],
               traj[0,1], s=100, color='k')
    ax.text(traj[0,0],
                traj[0,1], 
                track_labels[-1],
                va='center', 
                ha='center', 
                fontsize=24,
                fontname='Arial', zorder=1000)
    for index_ in np.arange(index_TP0):
        # ax.plot(traj[index_:index_+2,0], 
        #         traj[index_:index_+2,1], color = 'k', 
        #         lw=5) #, s=50)
        ax.plot([traj[index_,0], traj[index_+1,0]], 
                [traj[index_,1], traj[index_+1,1]], 
                color='k', lw=5)
        
    for index_ in np.arange(index_TP0, len(after_seg)-1):
            ax.plot(traj[index_:index_+2,0], 
                    traj[index_:index_+2,1], color = 'green', 
                    lw=5) #, s=50)
    ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
        
    ax.text(traj[-1,0],
                traj[-1,1], 
                track_labels[-1],
                va='center', 
                ha='center', 
                fontsize=18,
                fontname='Arial', zorder=1000)
    
    # Add plot of regions 1A, 1B
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_Region-Em-VE_graybg.svg' ), 
                dpi=600,
                bbox_inches='tight')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
    plt.show()
    
    
    
# =============================================================================
# =============================================================================
# #     Make coplot of the ExE-VE half 
# =============================================================================
# =============================================================================
    
    index_TP0 = np.arange(len(after_seg))[after_seg>0][0]
    
    track_labels = ['5', '6', '7', '8']
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    # ax.scatter(pca_proj[:,0], 
                # pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
    ax.scatter(pca_proj[:,0], 
               pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
    
    # plot these as segments!. 
    
    # plot regions 2 - 4
    counter = 0
    for iii in np.arange(5-1,8-1+1):
        
        traj = all_traj_regions_major8[iii].copy()
        ax.scatter(traj[0,0],
                   traj[0,1], s=100, color='k')
        ax.text(traj[0,0],
                traj[0,1], 
                track_labels[counter],
                va='center', 
                ha='center', 
                fontsize=24,
                fontname='Arial', zorder=1000)
        
        for index_ in np.arange(index_TP0):
            # ax.plot(traj[index_:index_+2,0], 
            #         traj[index_:index_+2,1], color = 'k', 
            #         lw=5) #, s=50)
            ax.plot([traj[index_,0], traj[index_+1,0]], 
                    [traj[index_,1], traj[index_+1,1]], 
                    color='k', lw=5)
            
        for index_ in np.arange(index_TP0, len(after_seg)-1):
            ax.plot(traj[index_:index_+2,0], 
                    traj[index_:index_+2,1], color = 'green', 
                    lw=5) #, s=50)
            
        # add in a label at the end point of the track. 
        ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
        ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
        
        ax.text(traj[-1,0],
                traj[-1,1], 
                track_labels[counter],
                va='center', 
                ha='center', 
                fontsize=24,
                fontname='Arial', zorder=1000)
        
        counter+=1
        
    
    # Add plot of regions 1A, 1B
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_Region-Ex-VE_colorbg.svg' ), 
                dpi=600,
                bbox_inches='tight')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
    plt.show()
    



    index_TP0 = np.arange(len(after_seg))[after_seg>0][0]
    
    track_labels = ['5', '6', '7', '8']
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(pca_proj[:,0], 
                pca_proj[:,1], color='lightgrey', s=10) # color all the backgroud grey 
    # ax.scatter(pca_proj[:,0], 
    #            pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
    
    # plot these as segments!. 
    # plot regions 2 - 4
    counter = 0
    for iii in np.arange(5-1,8-1+1):
        
        traj = all_traj_regions_major8[iii].copy()
        ax.scatter(traj[0,0],
                   traj[0,1], s=100, color='k')
        ax.text(traj[0,0],
                traj[0,1], 
                track_labels[counter],
                va='center', 
                ha='center', 
                fontsize=24,
                fontname='Arial', zorder=1000)
        
        for index_ in np.arange(index_TP0):
            # ax.plot(traj[index_:index_+2,0], 
            #         traj[index_:index_+2,1], color = 'k', 
            #         lw=5) #, s=50)
            ax.plot([traj[index_,0], traj[index_+1,0]], 
                    [traj[index_,1], traj[index_+1,1]], 
                    color='k', lw=5)
            
        for index_ in np.arange(index_TP0, len(after_seg)-1):
            ax.plot(traj[index_:index_+2,0], 
                    traj[index_:index_+2,1], color = 'green', 
                    lw=5) #, s=50)
            
        # add in a label at the end point of the track. 
        ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
        ax.scatter(traj[-1,0],
               traj[-1,1], s=100, color='g')
        
        ax.text(traj[-1,0],
                traj[-1,1], 
                track_labels[counter],
                va='center', 
                ha='center', 
                fontsize=24,
                fontname='Arial', zorder=1000)
        
        counter+=1
        
    
    # Add plot of regions 1A, 1B
    ax.set_aspect(1)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_Region-Ex-VE_graybg.svg' ), 
                dpi=600,
                bbox_inches='tight')
    # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
    plt.show()



# # =============================================================================
# # =============================================================================
# # =============================================================================
# # # # Do the regions for phenotypic split of region 1.
# # =============================================================================
# # =============================================================================
# # =============================================================================

#     # fig, ax = plt.subplots(figsize=(10,10))
#     # # ax.scatter(pca_proj[:,0], 
#     #             # pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
#     # ax.scatter(pca_proj[:,0], 
#     #                pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
#     # ax.plot(pca_proj[umap_cluster_labels==0,0], 
#     #         pca_proj[umap_cluster_labels==0,1], 'r.')
#     # plt.show()


#     # 1A 
#     gate_select = np.logical_and(dynamic_quad_id_main == 1, umap_cluster_labels == 1) # use upper. 
    
#     # gate_select = dynamic_quad_id_main.astype(np.int) == 1
#     # gate_select = umap_cluster_labels == 1
#     traj_P_EpiVE = compute_umap_trajectory( pca_proj, 
#                                             pca_proj_cluster,
#                                             gate_select, 
#                                             time_discrete, 
#                                             frame_all, 
#                                             resampling_grid_size, 
#                                             max_lim, 
#                                             tri, 
#                                             Ts=5/60.) 


#     fig, ax = plt.subplots(figsize=(10,10))
#     # ax.scatter(pca_proj[:,0], 
#                 # pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
#     ax.scatter(pca_proj[:,0], 
#                    pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
            
#     traj = traj_P_EpiVE[0]
#     ax.plot(traj[:,0], 
#             traj[:,1], color = 'k', 
#             lw=5) #, s=50)
#     ax.plot(traj[after_seg,0], 
#             traj[after_seg,1], color = 'green', 
#             lw=5)
    
#     ax.set_aspect(1)
#     plt.grid('off')
#     plt.axis('off')
#     plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_Region-1A_colorbg.svg' ), 
#                 dpi=600,
#                 bbox_inches='tight')
#     # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
#     plt.show()
    
    
#     # 1B
#     gate_select = np.logical_and(dynamic_quad_id_main == 1, umap_cluster_labels == 0) # use upper.
    
#     # gate_select = dynamic_quad_id_main.astype(np.int) == 1
#     # gate_select = umap_cluster_labels == 1
#     traj_P_EpiVE = compute_umap_trajectory( pca_proj, 
#                                             pca_proj_cluster,
#                                             gate_select, 
#                                             time_discrete, 
#                                             frame_all, 
#                                             resampling_grid_size, 
#                                             max_lim, 
#                                             tri, 
#                                             Ts=5/60.) 


#     fig, ax = plt.subplots(figsize=(10,10))
#     # ax.scatter(pca_proj[:,0], 
#                 # pca_proj[:,1], color='lightgrey') # color all the backgroud grey 
#     ax.scatter(pca_proj[:,0], 
#                    pca_proj[:,1], c=all_umap_phenotype_label_colors, alpha=0.5, s=10) # color all the backgroud grey 
            
#     traj = traj_P_EpiVE[0]
#     ax.plot(traj[:,0], 
#             traj[:,1], color = 'k', 
#             lw=5) #, s=50)
#     ax.plot(traj[after_seg,0], 
#             traj[after_seg,1], color = 'green', 
#             lw=5)
    
#     ax.set_aspect(1)
#     plt.grid('off')
#     plt.axis('off')
#     plt.savefig(os.path.join(saveplotfolder_umap_region, 'UMAP_trajectory_Region-1B_colorbg.svg' ), 
#                 dpi=600,
#                 bbox_inches='tight')
#     # plt.savefig(os.path.join(saveplotfolder, 'UMAP_kmeans_partitioning_and_hcluster_points.svg'), bbox_inches='tight')
#     plt.show()