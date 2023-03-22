# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:57:39 2022

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



def _make_color_wheel():
    
    import numpy as np 
    """ :returns: [ncols, 3] ndarray colorwheel """
    # how many hues ("cols") separate each color
    # (for this color wheel)
    RY = 15  # red-yellow
    YG = 6   # yellow-green
    GC = 4   # green-cyan
    CB = 11  # cyan-blue
    BM = 13  # blue-magenta
    MR = 6   # magenta-red
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)  # r g b

    col = 0
    # RY
    colorwheel[col:col+RY, 0] = 255
    colorwheel[col:col+RY, 1] = np.floor(255*np.arange(RY)/RY)
    col = col + RY
    # YG
    colorwheel[col:col+YG, 0] = np.ceil(255*np.arange(YG, 0, -1)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(GC)/GC)
    col = col + GC
    # CB
    colorwheel[col:col+CB, 1] = np.ceil(255*np.arange(CB, 0, -1)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(BM)/BM)
    col = col + BM
    # MR
    colorwheel[col:col+MR, 2] = np.ceil(255*np.arange(MR, 0, -1)/MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


_colorwheel = _make_color_wheel()


def flow_to_rgb(flo_data):
    """
        :param flo_data: [h, w, 2] ndarray (flow data)
                         (must be normalized to 0-1 range)
        :returns: [h, w, 3] ndarray (rgb data) using color wheel
    """
    ncols = len(_colorwheel)

    fu = flo_data[:, :, 0]
    fv = flo_data[:, :, 1]

    [h, w] = fu.shape
    rgb_data = np.empty([h, w, 3], dtype=np.uint8)

    rad = np.sqrt(fu ** 2 + fv ** 2)
    a = np.arctan2(-fv, -fu) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 mapped to 1~ncols
    k0 = fk.astype(np.uint8)
    k1 = (k0 + 1) % ncols
    f = fk - k0
    for i in range(3):  # r g b
        col0 = _colorwheel[k0, i]/255.0
        col1 = _colorwheel[k1, i]/255.0
        col = np.multiply(1.0-f, col0) + np.multiply(f, col1)

        # increase saturation with radius
        col = 1.0 - np.multiply(rad, 1.0 - col)

        # save to data channel i
        rgb_data[:, :, i] = np.floor(col * 255).astype(np.uint8)

    return rgb_data


def get_colors(inp, colormap, vmin=None, vmax=None):
    import pylab as plt 
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


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


if __name__=="__main__":
    
    import numpy as np 
    import seaborn as sns
    import pylab as plt 
    import skimage.io as skio
    import scipy.io as spio 
    import os 
    import glob
    import pandas as pd 
    import statsmodels.api as sm
    import Utility_Functions.file_io as fio
    from sklearn.metrics.pairwise import pairwise_distances
    import seaborn as sns 
    import pandas as pd 
    import skimage
    from flow_vis import flow_to_color, make_colorwheel
    from matplotlib.collections import LineCollection
    from skimage.color import rgb2hsv, hsv2rgb
    from matplotlib import cm
    import matplotlib 
    
    save_out_statsfolder = os.path.join('.', '2022-03-07_Compiled_single_cell_UMAP_clustering')
    # fio.mkdir(save_out_statsfolder)
    metricfile = os.path.join(save_out_statsfolder, '2022-03-07_Master-Timelapse-cluster-single-cell-stats-table.csv')
    
    # this will load all the labelling for all embryos. 
    metrics_tables = pd.read_csv(metricfile)
    
# =============================================================================
#     Load all the metadata
# =============================================================================
    embryo_all = metrics_tables['Uniq_Embryo_Name'].values.copy()
    embryo_all = np.hstack(embryo_all)
    uniq_embryo = np.unique(embryo_all)
    

    metricfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3'
    imgfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\CellImages'
    
    # mastersavefolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\2022-03-08_Colored-Single-Cell-Masks'
    # mastersavefolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\2022-03-09_Colored-Single-Cell-Tracks_Direction'
    # mastersavefolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\2022-03-09_Colored-Single-Cell-Tracks_Direction-Ice'
    # mastersavefolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\2022-08-03_Colored-Single-Cell-Tracks_Umap-Cluster'
    
    mastersavefolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\2022-11-10_Colored-Single-Cell-Tracks_Region-Cluster-EpiHalf'
    fio.mkdir(mastersavefolder)
    
    
    """
    Check the colormap
    """
    # create a colour wheel
    wheel = np.zeros((100,100,2)); wheel_x, wheel_y = np.indices(wheel[...,0].shape)
    wheel[...,0] = wheel_x - wheel.shape[0]//2 
    wheel[...,1] = wheel_y - wheel.shape[1]//2 
    
    wheel_color = flow_to_color(wheel, convert_to_bgr=False)
    
    plt.figure()
    plt.imshow(wheel_color.transpose(1,0,2))
    plt.show()
    
    
    embfolders = ['L455_Emb2_TP1-90',
                  'L455_Emb3_TP1-111',
                  'L491_Emb2_TP1-100',
                  'L505_Emb2_TP1-110',
                  'L864_Emb1_TP1-115']
    
    
    # add flipping 
    embryos_flipping = {'455-Emb2-225-polar': True,
                        '455-Emb3-000-polar': False,
                        '491--000-polar': False,
                        '505--120-polar': False,
                        '864--180-polar': False}
    
    # add the csv folder to lookup the angle rotation for plotting the midlines.... 
    angle_file = 'F:\\Data_Info\\Migration_Angles-curvilinear_curved_mean\\LifeAct_polar_angles_AVE_classifier-consensus.csv'
    angle_tab = pd.read_csv(angle_file)
    
    
    for emb_iii, emb in enumerate(uniq_embryo[:]):
    # for emb_iii in np.arange(len(uniq_embryo))[4:]:
        emb = uniq_embryo[emb_iii]
        """
        Grab the unwrapping params in order to remap . 
        """
    
        # metricfile: 
        # segfile = 
        
        """
        Grab the umap saved cluster images. 
        """
        umap_cluster_emb_mat = spio.loadmat('UMAP_cellclusters-'+emb+'.mat')
        cells_umap_ee = umap_cluster_emb_mat['cells_umap_ee']
        cells_umap_ee_color = umap_cluster_emb_mat['cells_umap_ee_color']
        # spio.savemat('UMAP_cellclusters-'+emb+'.mat', 
        #              {'cells_umap_ee_color': cells_umap_ee_color, 
        #               'cells_umap_ee': cells_umap_ee,
        #               'distribution_emb': distribution_emb})
        
        
        """
        Read in the cell labels corresponding to the table!. 
        """
        saveplotfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-19_PCA_analysis'
        savefile = os.path.join(saveplotfolder, '2021-01-24_pca_umap_analysis_data-Neighbors100_CorrelationMetric.mat') # this version seems most specific -> use this version!. 
        saveobj = spio.loadmat(savefile)
        
        frame_all = saveobj['Frame'].ravel()
        embryo_all = np.hstack(saveobj['Embryo'].ravel())
        stage_all = saveobj['Stage'].ravel()
        static_quad_id_all = saveobj['Static_quad'].ravel()
        dynamic_quad_id_all = saveobj['Dynamic_quad'].ravel()
        dynamic_quad_id_main = saveobj['Dynamic_quad_main'].ravel()
        epi_contour_dist = saveobj['epi_contour_dist'].ravel()
        col_name_select = saveobj['feat_names'].ravel()
        
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
            
        row_id_all = np.hstack(saveobj['row_id'].ravel())
        TP_all = np.hstack(saveobj['TP'].ravel())
        cell_id_all = np.hstack(saveobj['cell_id'].ravel())
        track_id_all = np.hstack(saveobj['track_id'].ravel())
        
        #### combine track and embryo id to make unique_track_id_all
        embryo_track_id_all = np.hstack([embryo_all[iii] +'_' + str(track_id_all[iii]).zfill(3) for iii in np.arange(len(track_id_all))])
        
        """
        Derive the prox/distal clusterings.... 
        """
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
    
        
        saveplotfolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\PseudoAnalysis\\Data\\Cell_Statistics_3\\2022-01-24_UMAP_Correlation_analysis'
        clustering_savefolder = os.path.join(saveplotfolder, 'cluster_stats');
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
   
        region_cluster_labels_colors = np.vstack(sns.color_palette('colorblind', 16))[np.unique(region_cluster_labels)[1:]]
        region_cluster_labels_color = np.vstack([matplotlib.colors.to_rgb('lightgrey'),
                                                 region_cluster_labels_colors])
        region_cluster_labels_color = region_cluster_labels_color[[0,2,3,1,4]]


        """
        We want to isolate all the data for the embryo in question!. 
        """
        embryo_all_select = embryo_all == emb

        """
        Gate out the key parameters.
        """
        
        track_id_all = track_id_all[embryo_all_select>0].copy() # we just need this really .... 
        stad_quad_id_main = stad_quad_id_main[embryo_all_select>0]
        dynamic_quad_id_main = dynamic_quad_id_main[embryo_all_select>0]
        stad_quad_id_main_split = stad_quad_id_main_split[embryo_all_select>0]
        dynamic_quad_id_main_split = dynamic_quad_id_main_split[embryo_all_select>0]
        umap_cluster_labels = umap_cluster_labels[embryo_all_select>0]


        """
        Compile all tracks 
        """
        
        # polarfile =  # rotated? or not .... -> should do rotated? -> rotate this and save out.... for this.  
        tracklookupfile = os.path.join(metricfolder, 'single_cell_track_stats_lookup_table_'+emb+'.csv')
        singlecellstatfile = os.path.join(metricfolder, 'single_cell_statistics_table_export-ecc-realxyz-from-demons_'+emb+'.csv')
        
        # we can parse the cell trajectory for each cell to trajectories. 
        single_cell_table = pd.read_csv(singlecellstatfile)
        track_lookup_table = pd.read_csv(tracklookupfile)
        
        # use the AVE_rotated. 
        xy_AVE_rot = np.vstack([single_cell_table['pos_2D_x_rot-AVE'].values, 
                                single_cell_table['pos_2D_y_rot-AVE'].values]).T
        # xy_AVE_rot = np.vstack([single_cell_table['pos_2D_x'].values, 
                                # single_cell_table['pos_2D_y'].values]).T
        xy_AVE_tracks = []
        
        polarfile = os.path.join(imgfolder, emb, 'A-P_rotated_data','AVE_rot-%s' %(emb) + '.tif')
        segfile = os.path.join(imgfolder, emb, 'A-P_rotated_data','AVE_rot-cells_%s' %(emb) + '.tif')
        
        # create a matrix to save the UMAP clusters. 
        cells_emb = skio.imread(segfile) # segmented cell image.
        img_emb = skio.imread(polarfile) # raw image 
        TP_mig_end =  single_cell_table['Frame'].max()
        TP_mig_start = np.min(single_cell_table['Frame'][single_cell_table['Stage']=='Migration to Boundary'].values) # determine the frame at the start of migration. 
        
        
        """
        Viz 1 by mean direction 
        """   
        emb_save_folder = os.path.join(mastersavefolder, emb); fio.mkdir(emb_save_folder)
        
        
        # emb_save_folder_tracks_w_bg = os.path.join(emb_save_folder, 'single_cell_tracks_direction'); fio.mkdir(emb_save_folder_tracks_w_bg)
        # emb_save_folder_tracks_no_bg = os.path.join(emb_save_folder, 'single_cell_tracks_direction_no-overlay'); fio.mkdir(emb_save_folder_tracks_no_bg)
        
        all_tracks_xy = []
        all_tracks_xyz = [] # get the 3D positions for computation!. 
        all_tracks_xy_time = []
        all_tracks_umap_clusters_time = []
        # all_tracks_xy_mean_vector = []
        
        for ii in np.arange(len(track_lookup_table))[:]:
            
            inds = track_lookup_table.iloc[ii].values[1:].copy()
            track_xy_time = np.arange(len(inds))[inds>-1] # invalid is -1 
            
            # if len(track_xy_time)> 1: # must have more than 1 point in order to get any color 
            track_xy_AVE = xy_AVE_rot[inds[track_xy_time]].copy()
                # track_xy_AVE_ = track_xy_AVE.reshape(-1, 1, track_xy_AVE.shape[-1])
        #         segments = np.concatenate([track_xy_AVE_[:-1], track_xy_AVE_[1:]], axis=1)
        #         track_xy_AVE_dxdy = segments[...,1] - segments[...,0]
        #         track_xy_AVE_dxdy = track_xy_AVE_dxdy[...,::-1] # normalize? 
                
        #         # colour the directionality of the segments. 
        #         segments_color = flow_to_color(track_xy_AVE_dxdy[None,...], 
        #                                        clip_flow=None, 
        #                                        convert_to_bgr=True)
        #         segments_color = segments_color[0] / 255.
                
        #         lc = LineCollection(segments, colors=segments_color)
        #         line = ax.add_collection(lc)
        # #     # plot line segments with prescribed colours 
        
                # mean_track_vector = np.mean(track_xy_AVE[1:] - track_xy_AVE[:-1], axis=0)
                # # mean_track_vector = mean_track_vector/(np.linalg.norm(mean_track_vector) + 1e-12)
            all_tracks_xy.append(track_xy_AVE)
            all_tracks_xy_time.append(track_xy_time)
            if len(track_xy_AVE) > 0:
                all_tracks_umap_clusters_time.append(cells_umap_ee[track_xy_time, track_xy_AVE[:,1].astype(np.int), track_xy_AVE[:,0].astype(np.int)])
            else:
                all_tracks_umap_clusters_time.append([])
                # all_tracks_xy_mean_vector.append(mean_track_vector)

                # mean_track_colors = flow_to_color(mean_track_vector[None,None,:], #[...,::-1], 
                #                                   # clip_flow=[-2.,2.],
                #                                   convert_to_bgr=False)
                # mean_track_colors = mean_track_colors.astype(np.float32) /255.
                # mean_track_color = np.squeeze(mean_track_colors)
                # print(mean_track_color.min(), mean_track_color.max())
                # # mean_track_color = 
                # # # reduce the saturation. 
                # # mean_track_color_hsv = rgb2hsv(mean_track_color); mean_track_color_hsv[0] = .5*mean_track_color_hsv[0]
                # # mean_track_color = hsv2rgb(mean_track_color_hsv)
                
        # all_tracks_xy_mean_vector = np.vstack(all_tracks_xy_mean_vector)
        # all_mags = np.linalg.norm(all_tracks_xy_mean_vector, axis=-1); lim = np.percentile(all_mags,90)
        # clip_lims = np.hstack([-lim, lim])
        
        # all_mean_track_colors = flow_to_color(np.vstack(all_tracks_xy_mean_vector)[None,...], #[...,::-1], 
        #                                            clip_flow=clip_lims,
        #                                           convert_to_bgr=False)
        # all_mean_track_colors = np.squeeze(all_mean_track_colors).astype(np.float32)/255.
        
        """ 
        1. viz by different umap clusters overlaid on the cells at TP_mig. 
        """ 
        emb_save_folder_tracks_umap_clusters = os.path.join(emb_save_folder, 'single_cell_tracks_umap_clusters'); fio.mkdir(emb_save_folder_tracks_umap_clusters)
        emb_save_folder_tracks_region_clusters = os.path.join(emb_save_folder, 'single_cell_tracks_region_clusters'); fio.mkdir(emb_save_folder_tracks_region_clusters)
        
        uniq_UMAP_clusters = np.unique(cells_umap_ee)
        
        
        """
        load the boundary line 
        """
        embfolder = embfolders[emb_iii]
        embryo_folder = 'F:\\LifeAct\\'+embfolder 
        saveboundaryfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow-epi-boundary')
        epi_boundary_file = os.path.join(saveboundaryfolder, emb,'inferred_epi_boundary-'+emb+'.mat')
        epi_boundary_time = spio.loadmat(epi_boundary_file)['contour_line']
        epi_contour_line = epi_boundary_time # just want the initial division line of the Epi.
        
        
        # flipping. 
        flipping = embryos_flipping[emb]
        if flipping:
            epi_contour_line[...,1] = (cells_emb[0].shape[0]-1) - epi_contour_line[...,1] 
        
        
        ve_angle_move = angle_tab.loc[angle_tab['Embryo'].values == emb]['Angle_consenus'].values[0]
        ve_angle_move = 90 - ve_angle_move # to get back to the rectangular angle coordinate? 
        print('ve movement direction: ', ve_angle_move)
        
        # use these angles to rotate the borders!.
        rot_center = (np.hstack(cells_emb[0].shape)/2.)[::-1]
        epi_contour_line = rotate_tracks(epi_contour_line, 
                                         angle=-(90-ve_angle_move), 
                                         center=rot_center, transpose=False)
        
        """
        2. plot temporal tracks following the umap clusters. 
        """
        
        # # idea is to gate by umap cluster -> identify tracks -> identify TP0_frame -> color the track by this cell?
        
        # for clust_ii, clust_id in enumerate(np.unique(umap_cluster_labels)[:]):
        
        #     if clust_id > -1:
                
        #         # do we attempt to assign each track uniquely? / by majority? 
        #         gate = umap_cluster_labels == clust_id    
        #         # tp0_gate = 
        #         cell_track_ids = np.unique(track_id_all[gate>0])
        #         cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
            
        #         # # now do we do an exclusivity check? I think this works? but we need to exclude out -1 and nan. --- this last removes too much
        #         # cell_track_ids_clusters = [all_tracks_umap_clusters_time[tra_id] for tra_id in cell_track_ids]
        #         # dominant_ids = []
        #         # for cc in cell_track_ids_clusters:
        #         #     cc = np.hstack(cc)
        #         #     cc = cc[np.isnan(cc) == 0].astype(np.int) + 1
        #         #     dominant_id = np.argsort(np.bincount(cc)[np.unique(cc)])[::-1]
        #         #     dominant_id = np.unique(cc)[dominant_id] - 1
        #         #     dominant_id = np.setdiff1d(dominant_id,-1)[0]
        #         #     dominant_ids.append(dominant_id)
        #         # cell_track_ids = [cell_track_ids[jjj] for jjj in np.arange(len(cell_track_ids)) if dominant_ids[jjj]==clust_id]
            
        #         fig, ax = plt.subplots(figsize=(10,10))
        #         ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray') # generically show the cell outline!. 
                
        #         for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): 
        #             plt.plot(tra[:,0],
        #                       tra[:,1],
        #                       color=region_cluster_labels_color[clust_ii], 
        #                       lw=2)
        #                       # alpha=.8) # color by mean! vector
                
        #         ax.plot(epi_contour_line[TP_mig_start,:,0],
        #                 epi_contour_line[TP_mig_start,:,1], 'r--', lw=5)
                
        #         ax.plot(rot_center[0], 
        #                 rot_center[1], 'P', ms=20, color='w')
                
        #         plt.grid('off')
        #         plt.axis('off')
        #         plt.savefig(os.path.join(emb_save_folder, 
        #                           'single_cell_tracks_Grp-%s.svg' %(str(clust_id))), dpi=300, bbox_inches='tight')
        #         plt.show()
                
                
        # # altogether on one plot. 
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray') # generically show the cell outline!. 
        
        # for clust_ii, clust_id in enumerate(np.unique(umap_cluster_labels)[:]):
        #     if clust_id > -1:
                    
        #         # do we attempt to assign each track uniquely? / by majority? 
        #         gate = umap_cluster_labels == clust_id    
        #         # tp0_gate = 
        #         cell_track_ids = np.unique(track_id_all[gate>0])
        #         cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
            
            
        #         for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): 
        #             ax.plot(tra[:,0],
        #                     tra[:,1],
        #                       color=region_cluster_labels_color[clust_ii], 
        #                       lw=2)
        #                       # alpha=.8) # color by mean! vector
                
        # ax.plot(epi_contour_line[TP_mig_start,:,0],
        #         epi_contour_line[TP_mig_start,:,1], 'r--', lw=5)
        
        # ax.plot(rot_center[0], 
        #         rot_center[1], 'P', ms=20, color='w')
        
        # plt.grid('off')
        # plt.axis('off')
        # plt.savefig(os.path.join(emb_save_folder, 
        #                   'single_cell_tracks_all_UMAP.svg'), dpi=300, bbox_inches='tight')
        # plt.show()
        
        
        
        # """
        # Make individual track videos for each umap cluster.... and one together.... and make it without background ..... or with alpha background!. 
        # """
        
        # for clust_ii, clust_id in enumerate(np.unique(umap_cluster_labels)[:]):
        
        #     if clust_id > -1:
                
        #         # do we attempt to assign each track uniquely? / by majority? 
        #         gate = umap_cluster_labels == clust_id    
        #         # tp0_gate = 
        #         cell_track_ids = np.unique(track_id_all[gate>0])
        #         cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
            
        #         # # now do we do an exclusivity check? I think this works? but we need to exclude out -1 and nan. --- this last removes too much
        #         # cell_track_ids_clusters = [all_tracks_umap_clusters_time[tra_id] for tra_id in cell_track_ids]
        #         # dominant_ids = []
        #         # for cc in cell_track_ids_clusters:
        #         #     cc = np.hstack(cc)
        #         #     cc = cc[np.isnan(cc) == 0].astype(np.int) + 1
        #         #     dominant_id = np.argsort(np.bincount(cc)[np.unique(cc)])[::-1]
        #         #     dominant_id = np.unique(cc)[dominant_id] - 1
        #         #     dominant_id = np.setdiff1d(dominant_id,-1)[0]
        #         #     dominant_ids.append(dominant_id)
        #         # cell_track_ids = [cell_track_ids[jjj] for jjj in np.arange(len(cell_track_ids)) if dominant_ids[jjj]==clust_id]
                
        #         cluster_plot_folder = os.path.join(emb_save_folder, 'UMAP_tracks_Grp-'+str(clust_ii).zfill(3))
        #         fio.mkdir(cluster_plot_folder)
                
        #         worm_length=10
        #         for frame_ii in np.arange(len(cells_emb)):
                    
        #             fig, ax = plt.subplots(figsize=(10,10))
        #             ax.imshow(cells_emb[frame_ii] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
                
        #             # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): \
        #             for tra_ii in cell_track_ids: 
        #                 tra = all_tracks_xy[tra_ii]
        #                 time_tra = all_tracks_xy_time[tra_ii]
        #                 # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                        
        #                 # check if we need to plot the current track!. 
        #                 if frame_ii in time_tra:
        #                     # we need to plot. 
        #                     end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
        #                     start_ind = np.maximum(0, end_ind-worm_length+1)
                            
        #                     ax.plot(tra[end_ind,0],
        #                             tra[end_ind,1], 'ko',zorder=100, ms=3)
        #                     ax.plot(tra[start_ind:end_ind+1, 0],
        #                             tra[start_ind:end_ind+1, 1], '-', lw=2, color=region_cluster_labels_color[clust_ii])
                            
        #             ax.plot(epi_contour_line[frame_ii,:,0],
        #                     epi_contour_line[frame_ii,:,1], 'r--', lw=5)
                
        #             ax.plot(rot_center[0], 
        #                     rot_center[1], 'P', ms=20, color='w')
                            
        #             plt.grid('off')
        #             plt.axis('off')
        #             plt.savefig(os.path.join(cluster_plot_folder, 
        #                                       'time_%s.png' %(str(frame_ii).zfill(3))), 
        #                         dpi=300, bbox_inches='tight')
        #             plt.show()
                
                
        # """
        # Here is a good opportunity to apply the 1A/1B splitting of regionality.... based on which is more proximal.... the other possibility is splitting by? 
        # """
        
        # gate = np.logical_and(umap_cluster_labels == 1, dynamic_quad_id_main_split==1)
        # # tp0_gate = 
        # cell_track_ids = np.unique(track_id_all[gate>0])
        # cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
        
        # cluster_plot_folder = os.path.join(emb_save_folder, 'UMAP_tracks_Grp-1A')
        # fio.mkdir(cluster_plot_folder)
        
        # worm_length=10
        # for frame_ii in np.arange(len(cells_emb)):
            
        #     fig, ax = plt.subplots(figsize=(10,10))
        #     ax.imshow(cells_emb[frame_ii] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
        
        #     # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): \
        #     for tra_ii in cell_track_ids: 
        #         tra = all_tracks_xy[tra_ii]
        #         time_tra = all_tracks_xy_time[tra_ii]
        #         # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                
        #         # check if we need to plot the current track!. 
        #         if frame_ii in time_tra:
        #             # we need to plot. 
        #             end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
        #             start_ind = np.maximum(0, end_ind-worm_length+1)
                    
        #             ax.plot(tra[end_ind,0],
        #                     tra[end_ind,1], 'ko',zorder=100, ms=3)
        #             ax.plot(tra[start_ind:end_ind+1, 0],
        #                     tra[start_ind:end_ind+1, 1], '-', lw=2, color=region_cluster_labels_color[2], alpha=0.8)
                    
        #     ax.plot(epi_contour_line[frame_ii,:,0],
        #             epi_contour_line[frame_ii,:,1], 'r--', lw=5)
        
        #     ax.plot(rot_center[0], 
        #             rot_center[1], 'P', ms=20, color='w')
                    
        #     plt.grid('off')
        #     plt.axis('off')
        #     plt.savefig(os.path.join(cluster_plot_folder, 
        #                               'time_%s.png' %(str(frame_ii).zfill(3))), 
        #                 dpi=300, bbox_inches='tight')
        #     plt.show()
            
            
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray') # generically show the cell outline!. 
        
        # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): 
        #     plt.plot(tra[:,0],
        #               tra[:,1],
        #               color=region_cluster_labels_color[2], alpha=0.8, 
        #               lw=2)
        #               # alpha=.8) # color by mean! vector
        
        # ax.plot(epi_contour_line[TP_mig_start,:,0],
        #         epi_contour_line[TP_mig_start,:,1], 'r--', lw=5)
        
        # ax.plot(rot_center[0], 
        #         rot_center[1], 'P', ms=20, color='w')
        
        # plt.grid('off')
        # plt.axis('off')
        # plt.savefig(os.path.join(emb_save_folder, 
        #                   'single_cell_tracks_Grp-1A.svg'), dpi=300, bbox_inches='tight')
        # plt.show()
        
            
            
        # gate = np.logical_and(umap_cluster_labels == 1, dynamic_quad_id_main_split==9)
        # # tp0_gate = 
        # cell_track_ids = np.unique(track_id_all[gate>0])
        
        # # can only do this if this exists!. 
        # if len(cell_track_ids) > 0:
        #     cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
            
        #     cluster_plot_folder = os.path.join(emb_save_folder, 'UMAP_tracks_Grp-1B')
        #     fio.mkdir(cluster_plot_folder)
            
        #     worm_length=10
        #     for frame_ii in np.arange(len(cells_emb)):
                
        #         fig, ax = plt.subplots(figsize=(10,10))
        #         ax.imshow(cells_emb[frame_ii] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
            
        #         # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): \
        #         for tra_ii in cell_track_ids: 
        #             tra = all_tracks_xy[tra_ii]
        #             time_tra = all_tracks_xy_time[tra_ii]
        #             # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                    
        #             # check if we need to plot the current track!. 
        #             if frame_ii in time_tra:
        #                 # we need to plot. 
        #                 end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
        #                 start_ind = np.maximum(0, end_ind-worm_length+1)
                        
        #                 ax.plot(tra[end_ind,0],
        #                         tra[end_ind,1], 'ko',zorder=100, ms=3)
        #                 ax.plot(tra[start_ind:end_ind+1, 0],
        #                         tra[start_ind:end_ind+1, 1], '-', lw=2, color=region_cluster_labels_color[2], alpha=0.5)
                        
        #         ax.plot(epi_contour_line[frame_ii,:,0],
        #                 epi_contour_line[frame_ii,:,1], 'r--', lw=5)
            
        #         ax.plot(rot_center[0], 
        #                 rot_center[1], 'P', ms=20, color='w')
                        
        #         plt.grid('off')
        #         plt.axis('off')
        #         plt.savefig(os.path.join(cluster_plot_folder, 
        #                                   'time_%s.png' %(str(frame_ii).zfill(3))), 
        #                     dpi=300, bbox_inches='tight')
        #         plt.show()
            
            
        #     """
        #     Plot the static version.
        #     """
        #     fig, ax = plt.subplots(figsize=(10,10))
        #     ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray') # generically show the cell outline!. 
            
        #     for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): 
        #         plt.plot(tra[:,0],
        #                   tra[:,1],
        #                   color=region_cluster_labels_color[2], alpha=0.5, 
        #                   lw=2)
        #                   # alpha=.8) # color by mean! vector
            
        #     ax.plot(epi_contour_line[TP_mig_start,:,0],
        #             epi_contour_line[TP_mig_start,:,1], 'r--', lw=5)
            
        #     ax.plot(rot_center[0], 
        #             rot_center[1], 'P', ms=20, color='w')
            
        #     plt.grid('off')
        #     plt.axis('off')
        #     plt.savefig(os.path.join(emb_save_folder, 
        #                       'single_cell_tracks_Grp-1B.svg'), dpi=300, bbox_inches='tight')
        #     plt.show()
            
        
        """
        Make individual track videos for each region cluster ( dynamic ID) 
        and one together.... and make it without background ..... or with alpha background!. 
        """
        
        region_colors_plot = sns.color_palette('magma_r', 8)
        
        region_5_color = np.hstack([192,179,155])/255.
        region_6_color = np.hstack([117,77,36])/255.
        region_7_color = np.hstack([77,77,77])/255.
        region_8_color = np.hstack([0,0,0])/255.
        
        region_colors_plot = np.vstack([region_colors_plot[0], 
                                        region_colors_plot[1],
                                        region_colors_plot[2],
                                        region_colors_plot[3], 
                                        region_5_color, 
                                        region_6_color,
                                        region_7_color,
                                        region_8_color])
        # create an extended version. 
        
        
        
        # for region_ii in np.arange(1,4+1):
        for region_ii in np.arange(5,8+1): # this is the exe portion! 
            
            # cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-%s' %(str(region_ii).zfill(3))) 
            # fio.mkdir(cluster_plot_folder)
             
            # gate = np.logical_and(umap_cluster_labels == 1, dynamic_quad_id_main_split==9)
            gate = dynamic_quad_id_main == region_ii
           
            # tp0_gate = 
            cell_track_ids = np.unique(track_id_all[gate>0])
            
            # can only do this if this exists!. 
            if len(cell_track_ids) > 0:
                cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
                
                # cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-%s' %(str(region_ii).zfill(3))) 
                cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-%s_continuous' %(str(region_ii).zfill(3))) 
                fio.mkdir(cluster_plot_folder)
               
                worm_length=10
                for frame_ii in np.arange(len(cells_emb)):
                    
                    fig, ax = plt.subplots(figsize=(10,10))
                    ax.imshow(cells_emb[frame_ii] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
                
                    # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): \
                    for tra_ii in cell_track_ids: 
                        tra = all_tracks_xy[tra_ii]
                        time_tra = all_tracks_xy_time[tra_ii]
                        # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                            
                        
                        # # check if we need to plot the current track!. 
                        # if frame_ii in time_tra:
                        # we need to plot. 
                        if time_tra[0] <=frame_ii:
                            start_ind = 0
                            if time_tra[-1] <=frame_ii:
                                end_ind = len(time_tra) - 1
                            else:
                                end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
                            # end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
                            # start_ind = np.maximum(0, end_ind-worm_length+1)
                        
                
                        # check if we need to plot the current track!. 
                        # if frame_ii in time_tra:
                        if time_tra[0] <=frame_ii:
                            # we need to plot. 
                            start_ind = 0
                            if time_tra[-1] <=frame_ii:
                                end_ind = len(time_tra) - 1
                            else:
                                # end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
                                end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
                            # start_ind = np.maximum(0, end_ind-worm_length+1)
                            ax.plot(tra[end_ind,0],
                                    tra[end_ind,1], 'ko',zorder=100, ms=3)
                            ax.plot(tra[start_ind:end_ind+1, 0],
                                    tra[start_ind:end_ind+1, 1], '-', lw=2, color=region_colors_plot[region_ii-1], alpha=1)
                            
                    ax.plot(epi_contour_line[frame_ii,:,0],
                            epi_contour_line[frame_ii,:,1], 'r--', lw=5)
               
                    ax.plot(rot_center[0], 
                              rot_center[1], 'P', ms=20, color='w')
                             
                    plt.grid('off')
                    plt.axis('off')
                    plt.savefig(os.path.join(cluster_plot_folder, 
                                                'time_%s.png' %(str(frame_ii).zfill(3))), 
                                  dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    
                    
                    
        for region_ii in np.arange(5,8+1): # this is the exe portion! 
            
            # cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-%s' %(str(region_ii).zfill(3))) 
            # fio.mkdir(cluster_plot_folder)
             
            # gate = np.logical_and(umap_cluster_labels == 1, dynamic_quad_id_main_split==9)
            gate = dynamic_quad_id_main == region_ii
           
            # tp0_gate = 
            cell_track_ids = np.unique(track_id_all[gate>0])
            
            # can only do this if this exists!. 
            if len(cell_track_ids) > 0:
                cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
                
                # cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-%s' %(str(region_ii).zfill(3))) 
                cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-%s_continuous-Static' %(str(region_ii).zfill(3))) 
                fio.mkdir(cluster_plot_folder)
               
                    
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
                
                # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): \
                for tra_ii in cell_track_ids: 
                    tra = all_tracks_xy[tra_ii]
                    time_tra = all_tracks_xy_time[tra_ii]
                    # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                            
                    # ax.plot(tra[end_ind,0],
                            # tra[end_ind,1], 'ko',zorder=100, ms=3)
                    ax.plot(tra[:, 0],
                            tra[:, 1], '-', lw=2, color=region_colors_plot[region_ii-1], alpha=1)
                        
                ax.plot(epi_contour_line[frame_ii,:,0],
                        epi_contour_line[frame_ii,:,1], 'r--', lw=5)
           
                ax.plot(rot_center[0], 
                          rot_center[1], 'P', ms=20, color='w')
                         
                plt.grid('off')
                plt.axis('off')
                plt.savefig(os.path.join(cluster_plot_folder, 
                                            'time_%s.svg' %(str(TP_mig_start).zfill(3))), 
                              dpi=300, bbox_inches='tight')
                plt.show()
            
                
                
                
# =============================================================================
#       Create a joint!. 
# =============================================================================

        # cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-%s' %(str(region_ii).zfill(3))) 
        cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-%s_continuous' %('Ex-VE')) 
        fio.mkdir(cluster_plot_folder)
        

        # for region_ii in np.arange(1,4+1):
               
        worm_length=10
        for frame_ii in np.arange(len(cells_emb)):
            
            fig, ax = plt.subplots(figsize=(10,10))
            ax.imshow(cells_emb[frame_ii] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
        
        
        
            for region_ii in np.arange(5,8+1): # this is the exe portion! 
             
                # gate = np.logical_and(umap_cluster_labels == 1, dynamic_quad_id_main_split==9)
                gate = dynamic_quad_id_main == region_ii
               
                # tp0_gate = 
                cell_track_ids = np.unique(track_id_all[gate>0])
                
                # can only do this if this exists!. 
                if len(cell_track_ids) > 0:
                    cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
                    
            
                # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): \
                for tra_ii in cell_track_ids: 
                    tra = all_tracks_xy[tra_ii]
                    time_tra = all_tracks_xy_time[tra_ii]
                    # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                        
                    
                    # # check if we need to plot the current track!. 
                    # if frame_ii in time_tra:
                    # we need to plot. 
                    if time_tra[0] <=frame_ii:
                        start_ind = 0
                        if time_tra[-1] <=frame_ii:
                            end_ind = len(time_tra) - 1
                        else:
                            end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
                        # end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
                        # start_ind = np.maximum(0, end_ind-worm_length+1)
                    
            
                    # check if we need to plot the current track!. 
                    # if frame_ii in time_tra:
                    if time_tra[0] <=frame_ii:
                        # we need to plot. 
                        start_ind = 0
                        if time_tra[-1] <=frame_ii:
                            end_ind = len(time_tra) - 1
                        else:
                            # end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
                            end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
                        # start_ind = np.maximum(0, end_ind-worm_length+1)
                        ax.plot(tra[end_ind,0],
                                tra[end_ind,1], 'ko',zorder=100, ms=3)
                        ax.plot(tra[start_ind:end_ind+1, 0],
                                tra[start_ind:end_ind+1, 1], '-', lw=2, color=region_colors_plot[region_ii-1], alpha=1)
                    
            ax.plot(epi_contour_line[frame_ii,:,0],
                    epi_contour_line[frame_ii,:,1], 'r--', lw=5)
       
            ax.plot(rot_center[0], 
                      rot_center[1], 'P', ms=20, color='w')
                     
            plt.grid('off')
            plt.axis('off')
            plt.savefig(os.path.join(cluster_plot_folder, 
                                        'time_%s.png' %(str(frame_ii).zfill(3))), 
                          dpi=300, bbox_inches='tight')
            plt.show()
            
            
            
            
            
        cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-%s_continuous_Static' %('Ex-VE')) 
        fio.mkdir(cluster_plot_folder)
        
        # for region_ii in np.arange(1,4+1):       
        worm_length=10
            
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
        

        for region_ii in np.arange(5,8+1): # this is the exe portion! 
         
            # gate = np.logical_and(umap_cluster_labels == 1, dynamic_quad_id_main_split==9)
            gate = dynamic_quad_id_main == region_ii
           
            # tp0_gate = 
            cell_track_ids = np.unique(track_id_all[gate>0])
            
            # can only do this if this exists!. 
            if len(cell_track_ids) > 0:
                cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
                
        
            # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): \
            for tra_ii in cell_track_ids: 
                tra = all_tracks_xy[tra_ii]
                time_tra = all_tracks_xy_time[tra_ii]
                # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                    
                
                # ax.plot(tra[end_ind,0],
                        # tra[end_ind,1], 'ko',zorder=100, ms=3)
                ax.plot(tra[:, 0],
                        tra[:, 1], '-', lw=2, color=region_colors_plot[region_ii-1], alpha=1)
            
        ax.plot(epi_contour_line[frame_ii,:,0],
                epi_contour_line[frame_ii,:,1], 'r--', lw=5)
   
        ax.plot(rot_center[0], 
                  rot_center[1], 'P', ms=20, color='w')
                 
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(cluster_plot_folder, 
                                    'time_%s.svg' %(str(TP_mig_start).zfill(3))), 
                      dpi=600, bbox_inches='tight')
        plt.show()
                
                
        
        
# =============================================================================
# =============================================================================
# =============================================================================
# # #        Give the static plots!          
# =============================================================================
# =============================================================================
# =============================================================================
                
        
                
                
                
                # """
                # Plot the static version.
                # """
                # fig, ax = plt.subplots(figsize=(10,10))
                # ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
                
                # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): 
                #     ax.plot(tra[:,0],
                #             tra[:,1],
                #             color=region_colors_plot[region_ii-1], alpha=1, 
                #             lw=2)
                #                   # alpha=.8) # color by mean! vector
                    
                # ax.plot(epi_contour_line[TP_mig_start,:,0],
                #         epi_contour_line[TP_mig_start,:,1], 'r--', lw=5)
                    
                # ax.plot(rot_center[0], 
                #         rot_center[1], 'P', ms=20, color='w')
                    
                # plt.grid('off')
                # plt.axis('off')
                # plt.savefig(os.path.join(emb_save_folder, 
                #                       'single_cell_tracks_Region-%s_static.svg' %(str(region_ii).zfill(3))), dpi=300, bbox_inches='tight')
                # plt.show()
                    
                
                # """
                # Plot the static version of only just the cells,,, instantaneously classified .... as the single label.... 
                # """
                # fig, ax = plt.subplots(figsize=(10,10))
                # ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
                
                # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): 
                #     ax.plot(tra[:,0],
                #             tra[:,1],
                #             color=region_colors_plot[region_ii-1], alpha=1, 
                #             lw=2)
                #                   # alpha=.8) # color by mean! vector
                    
                # ax.plot(epi_contour_line[TP_mig_start,:,0],
                #         epi_contour_line[TP_mig_start,:,1], 'r--', lw=5)
                    
                # ax.plot(rot_center[0], 
                #         rot_center[1], 'P', ms=20, color='w')
                    
                # plt.grid('off')
                # plt.axis('off')
                # plt.savefig(os.path.join(emb_save_folder, 
                #                       'single_cells_Region-%s_static.svg' %(str(region_ii).zfill(3))), dpi=300, bbox_inches='tight')
                # plt.show()
                
                    
                    
        # """
        # Generate this but now splitting physical region 1 with umap relevant clusters..... namely A,B,C? 
        # """
        
        # for umap_lab_ii, umap_lab in enumerate(np.unique(umap_cluster_labels)): 
            
        #     if umap_lab > -1:
                
        #         # gate = np.logical_and(umap_cluster_labels == 1, dynamic_quad_id_main_split==9)
        #         gate = np.logical_and(umap_cluster_labels == umap_lab, 
        #                               dynamic_quad_id_main == 1)
        #         cell_track_ids = np.unique(track_id_all[gate>0])
                
                 
        #         # can only do this if this exists!. 
        #         if len(cell_track_ids) > 0:
        #             cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
                     
        #             cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-1_UMAP-clust-%s' %(str(umap_lab_ii+1).zfill(3))) 
        #             fio.mkdir(cluster_plot_folder)
                    
        #             worm_length=10
        #             for frame_ii in np.arange(len(cells_emb)):
                         
        #                 fig, ax = plt.subplots(figsize=(10,10))
        #                 ax.imshow(cells_emb[frame_ii] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
                     
        #                 # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): \
        #                 for tra_ii in cell_track_ids: 
        #                      tra = all_tracks_xy[tra_ii]
        #                      time_tra = all_tracks_xy_time[tra_ii]
        #                      # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                                 
        #                      # check if we need to plot the current track!. 
        #                      if frame_ii in time_tra:
        #                          # we need to plot. 
        #                          end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
        #                          start_ind = np.maximum(0, end_ind-worm_length+1)
                                     
        #                          ax.plot(tra[end_ind,0],
        #                                  tra[end_ind,1], 'ko',zorder=100, ms=3)
        #                          ax.plot(tra[start_ind:end_ind+1, 0],
        #                                  tra[start_ind:end_ind+1, 1], '-', lw=2, 
        #                                  color=region_cluster_labels_color[umap_lab_ii], alpha=1)
                                 
        #                 ax.plot(epi_contour_line[frame_ii,:,0],
        #                          epi_contour_line[frame_ii,:,1], 'r--', lw=5)
                    
        #                 ax.plot(rot_center[0], 
        #                           rot_center[1], 'P', ms=20, color='w')
                                  
        #                 plt.grid('off')
        #                 plt.axis('off')
        #                 plt.savefig(os.path.join(cluster_plot_folder, 
        #                                             'time_%s.png' %(str(frame_ii).zfill(3))), 
        #                               dpi=300, bbox_inches='tight')
        #                 plt.show()
                     
                     
        #             """
        #             Plot the static version.
        #             """
        #             fig, ax = plt.subplots(figsize=(10,10))
        #             ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray') # generically show the cell outline!. 
                     
        #             for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): 
        #                 ax.plot(tra[:,0],
        #                          tra[:,1],
        #                          color=region_cluster_labels_color[umap_lab_ii], alpha=1, 
        #                          lw=2)
        #                                # alpha=.8) # color by mean! vector
                         
        #             ax.plot(epi_contour_line[TP_mig_start,:,0],
        #                     epi_contour_line[TP_mig_start,:,1], 'r--', lw=5)
                         
        #             ax.plot(rot_center[0], 
        #                     rot_center[1], 'P', ms=20, color='w')
                         
        #             plt.grid('off')
        #             plt.axis('off')
        #             plt.savefig(os.path.join(emb_save_folder, 
        #                                    'single_cell_tracks_Region-1_UMAP-clust-%s.svg' %(str(umap_lab_ii+1).zfill(3))), 
        #                         dpi=300, bbox_inches='tight')
        #             plt.show()
        
        
        # """
        # Generate splitting physical region 1 with prox distal. 
        # """
        # # for umap_lab_ii, umap_lab in enumerate(np.unique(umap_cluster_labels)): 
        # #     if umap_lab > -1:
        # prox_colors_plot = np.hstack([63,65,151])/255.; 
        # distal_colors_plot = sns.color_palette('colorblind', 8)[1]
        
        
        
            
        # # gate = np.logical_and(umap_cluster_labels == 1, dynamic_quad_id_main_split==9)
        # gate = dynamic_quad_id_main_split == 1
        # cell_track_ids = np.unique(track_id_all[gate>0])
        
        # # can only do this if this exists!. 
        # if len(cell_track_ids) > 0:
        #     cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
             
        #     cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-1_Prox_continuous') 
        #     fio.mkdir(cluster_plot_folder)
            
        #     worm_length=10
        #     for frame_ii in np.arange(len(cells_emb)):
                 
        #         fig, ax = plt.subplots(figsize=(10,10))
        #         ax.imshow(cells_emb[frame_ii] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
             
        #         # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): \
        #         for tra_ii in cell_track_ids: 
        #               tra = all_tracks_xy[tra_ii]
        #               time_tra = all_tracks_xy_time[tra_ii]
        #               # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                         
        #               # check if we need to plot the current track!. 
        #               if time_tra[0] <=frame_ii:
        #                   start_ind = 0
        #                   if time_tra[-1] <=frame_ii:
        #                       end_ind = len(time_tra) - 1
        #                   else:
        #                       end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
                             
        #                   ax.plot(tra[end_ind,0],
        #                           tra[end_ind,1], 'ko',zorder=100, ms=3)
        #                   ax.plot(tra[start_ind:end_ind+1, 0],
        #                           tra[start_ind:end_ind+1, 1], '-', lw=2, 
        #                           color=prox_colors_plot, alpha=1)
                         
        #         ax.plot(epi_contour_line[frame_ii,:,0],
        #                   epi_contour_line[frame_ii,:,1], 'r--', lw=5)
            
        #         ax.plot(rot_center[0], 
        #                   rot_center[1], 'P', ms=20, color='w')
                          
        #         plt.grid('off')
        #         plt.axis('off')
        #         plt.savefig(os.path.join(cluster_plot_folder, 
        #                                     'time_%s.png' %(str(frame_ii).zfill(3))), 
        #                       dpi=300, bbox_inches='tight')
        #         plt.show()
             
             
        #     # """
        #     # Plot the static version.
        #     # """
        #     # fig, ax = plt.subplots(figsize=(10,10))
        #     # ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
             
        #     # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): 
        #     #     ax.plot(tra[:,0],
        #     #               tra[:,1],
        #     #               color=prox_colors_plot, alpha=1, 
        #     #               lw=2)
        #     #                     # alpha=.8) # color by mean! vector
                 
        #     # ax.plot(epi_contour_line[TP_mig_start,:,0],
        #     #         epi_contour_line[TP_mig_start,:,1], 'r--', lw=5)
                 
        #     # ax.plot(rot_center[0], 
        #     #         rot_center[1], 'P', ms=20, color='w')
                 
        #     # plt.grid('off')
        #     # plt.axis('off')
        #     # plt.savefig(os.path.join(emb_save_folder, 
        #     #                         'single_cell_tracks_Region_tracks_Region-1_Prox.svg'), 
        #     #             dpi=300, bbox_inches='tight')
        #     # plt.show()
            
            
        # gate = dynamic_quad_id_main_split == 9
        # cell_track_ids = np.unique(track_id_all[gate>0])
        
        # # can only do this if this exists!. 
        # if len(cell_track_ids) > 0:
        #     cell_track_ids = np.hstack([cell_track_ids[jj] for jj in np.arange(len(cell_track_ids)) if TP_mig_start in all_tracks_xy_time[cell_track_ids[jj]]])
             
        #     cluster_plot_folder = os.path.join(emb_save_folder, 'Region_tracks_Region-1_Distal_continuous') 
        #     fio.mkdir(cluster_plot_folder)
            
        #     worm_length=10
        #     for frame_ii in np.arange(len(cells_emb)):
                 
        #         fig, ax = plt.subplots(figsize=(10,10))
        #         ax.imshow(cells_emb[frame_ii] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
             
        #         # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): \
        #         for tra_ii in cell_track_ids: 
        #               tra = all_tracks_xy[tra_ii]
        #               time_tra = all_tracks_xy_time[tra_ii]
        #               # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                         
        #               # check if we need to plot the current track!. 
        #               if time_tra[0] <=frame_ii:
        #                   start_ind = 0
        #                   if time_tra[-1] <=frame_ii:
        #                       end_ind = len(time_tra) - 1
        #                   else:
        #                       end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
                             
        #                   ax.plot(tra[end_ind,0],
        #                           tra[end_ind,1], 'ko',zorder=100, ms=3)
        #                   ax.plot(tra[start_ind:end_ind+1, 0],
        #                           tra[start_ind:end_ind+1, 1], '-', lw=2, 
        #                           color=distal_colors_plot, alpha=0.5)
                         
        #         ax.plot(epi_contour_line[frame_ii,:,0],
        #                   epi_contour_line[frame_ii,:,1], 'r--', lw=5)
            
        #         ax.plot(rot_center[0], 
        #                   rot_center[1], 'P', ms=20, color='w')
                          
        #         plt.grid('off')
        #         plt.axis('off')
        #         plt.savefig(os.path.join(cluster_plot_folder, 
        #                                     'time_%s.png' %(str(frame_ii).zfill(3))), 
        #                       dpi=300, bbox_inches='tight')
        #         plt.show()
             
             
        #     # """
        #     # Plot the static version.
        #     # """
        #     # fig, ax = plt.subplots(figsize=(10,10))
        #     # ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray', alpha=0.25) # generically show the cell outline!. 
             
        #     # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): 
        #     #     ax.plot(tra[:,0],
        #     #               tra[:,1],
        #     #               color=distal_colors_plot, alpha=0.5, 
        #     #               lw=2)
        #     #                     # alpha=.8) # color by mean! vector
                 
        #     # ax.plot(epi_contour_line[TP_mig_start,:,0],
        #     #         epi_contour_line[TP_mig_start,:,1], 'r--', lw=5)
                 
        #     # ax.plot(rot_center[0], 
        #     #         rot_center[1], 'P', ms=20, color='w')
                 
        #     # plt.grid('off')
        #     # plt.axis('off')
        #     # plt.savefig(os.path.join(emb_save_folder, 
        #     #                         'single_cell_tracks_Region_tracks_Region-1_Distal.svg'), 
        #     #             dpi=300, bbox_inches='tight')
        #     # plt.show()




        
                # fig, ax = plt.subplots(figsize=(10,10))
                # ax.imshow(cells_emb[TP_mig_start] == 0, cmap='gray') # generically show the cell outline!. 
                
                # for tra_ii, tra in enumerate([all_tracks_xy[iii] for iii in cell_track_ids] ): 
                #     plt.plot(tra[:,0],
                #               tra[:,1],
                #               color=region_cluster_labels_color[clust_ii], 
                #               lw=2)
                #               # alpha=.8) # color by mean! vector
                
                # ax.plot(epi_contour_line[TP_mig_start,:,0],
                #         epi_contour_line[TP_mig_start,:,1], 'r--', lw=5)
                
                # ax.plot(rot_center[0], 
                #         rot_center[1], 'P', ms=20, color='w')
                
                # plt.grid('off')
                # plt.axis('off')
                # plt.savefig(os.path.join(emb_save_folder, 
                #                   'single_cell_tracks_Grp-%s.svg' %(str(clust_id))), dpi=300, bbox_inches='tight')
                # plt.show()
        
        
        
        
        
        
            # cell_track_ids
        # for ii in np.arange()
        
        
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(cells_emb[TP_mig_end] == 0, cmap='gray')
        # for tra_ii, tra in enumerate(all_tracks_xy): 
        #     plt.plot(tra[:,0],
        #               tra[:,1],
        #               color=all_mean_track_colors[tra_ii], 
        #               lw=2)
        #               # alpha=.8) # color by mean! vector
        # plt.grid('off')
        # plt.axis('off')
        # plt.savefig(os.path.join(emb_save_folder, 
        #                          'single_cell_tracks_direction.png'), dpi=120, bbox_inches='tight')
        # plt.show()
        
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(np.zeros_like(cells_emb[TP_mig_end] == 0), cmap='gray')
        # for tra_ii, tra in enumerate(all_tracks_xy): 
        #     plt.plot(tra[:,0],
        #               tra[:,1],
        #               color=all_mean_track_colors[tra_ii], 
        #               lw=2)
        #               # alpha=.8) # color by mean! vector
        # plt.grid('off')
        # plt.axis('off')
        # plt.savefig(os.path.join(emb_save_folder, 
        #                          'single_cell_tracks_direction_no-overlay.png'), dpi=120, bbox_inches='tight')
        # plt.show()
        
        
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(cells_emb[TP_mig_end]==0, cmap='gray')
        # # for tra_ii, tra in enumerate(all_tracks_xy): 
        # #     plt.plot(tra[:,0],
        # #               tra[:,1],
        # #               color=all_mean_track_colors[tra_ii], 
        # #               lw=2)
        # #               # alpha=.8) # color by mean! vector
        # plt.grid('off')
        # plt.axis('off')
        # plt.savefig(os.path.join(emb_save_folder, 
        #                          'single_cells_TP_mig-end.png'), dpi=120, bbox_inches='tight')
        # plt.show()
        
        
        
        
        # # alternative coloring with time? 
        # # this we need to do line segments.... 
        # """ 
        # viz by time using segments
        # """ 
        # # get_colors(inp, colormap, vmin=None, vmax=None)
        
        # # all_tracks_xy = []
        # # all_tracks_xy_time = []
        
        
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(cells_emb[TP_mig_end] == 0, cmap='gray')
        # for tra_ii, tra in enumerate(all_tracks_xy): 
            
        #     time_tra = all_tracks_xy_time[tra_ii]
        #     tra_ = tra.reshape(-1, 1, tra.shape[-1])
        #     segments = np.concatenate([tra_[:-1], tra_[1:]], axis=1)
        #     segments_color = get_colors(time_tra, colormap=cm.Reds, vmin=0, vmax=TP_mig_end)
        #     # # turn into segments? 
        #     # # colour the directionality of the segments. 
        #     # segments_color = flow_to_color(track_xy_AVE_dxdy[None,...], 
        #     #                                 clip_flow=None, 
        #     #                                 convert_to_bgr=True)
        #     # segments_color = segments_color[0] / 255.
            
        #     lc = LineCollection(segments, colors=segments_color)
        #     line = ax.add_collection(lc)
            
        #     # plt.plot(tra[:,0],
        #     #           tra[:,1],
        #     #           color=all_mean_track_colors[tra_ii], 
        #     #           lw=2)
        #     #           # alpha=.8) # color by mean! vector
        # plt.grid('off')
        # plt.axis('off')
        # plt.savefig(os.path.join(emb_save_folder, 
        #                          'single_cell_tracks_time.png'), dpi=120, bbox_inches='tight')
        # plt.show()
        
        
        
        # # here we want to make a movie of the temporal evolution of the tracks. 
        # emb_save_folder_movie_tracks = os.path.join(emb_save_folder, 'movies_2D_tracks')
        # fio.mkdir(emb_save_folder_movie_tracks)
        
        # worm_length=10
        # for frame_ii in np.arange(len(cells_emb)):
            
        #     fig, ax = plt.subplots(figsize=(10,10))
        #     ax.imshow(cells_emb[frame_ii] == 0, cmap='gray')
            
        #     for tra_ii, tra in enumerate(all_tracks_xy): 
        #         time_tra = all_tracks_xy_time[tra_ii]
        #         # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                
        #         # check if we need to plot the current track!. 
        #         if frame_ii in time_tra:
        #             # we need to plot. 
        #             end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0]
        #             start_ind = np.maximum(0, end_ind-worm_length+1)
                    
        #             ax.plot(tra[end_ind,0],
        #                     tra[end_ind,1], 'wo',zorder=100, ms=3)
        #             ax.plot(tra[start_ind:end_ind+1, 0],
        #                     tra[start_ind:end_ind+1, 1], '-', lw=2, color='cyan')
        #     plt.grid('off')
        #     plt.axis('off')
        #     plt.savefig(os.path.join(emb_save_folder_movie_tracks, 
        #                              'single_cell_tracks_time_%s.png' %(str(frame_ii).zfill(3))), 
        #                 dpi=300, bbox_inches='tight')
        #     plt.show()
                
        
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(np.zeros_like(cells_emb[TP_mig_end] == 0), cmap='gray')
        # for tra_ii, tra in enumerate(all_tracks_xy): 
            
        #     time_tra = all_tracks_xy_time[tra_ii]
        #     tra_ = tra.reshape(-1, 1, tra.shape[-1])
        #     segments = np.concatenate([tra_[:-1], tra_[1:]], axis=1)
        #     segments_color = get_colors(time_tra, colormap=cm.coolwarm, vmin=0, vmax=TP_mig_end)
        #     # # turn into segments? 
        #     # # colour the directionality of the segments. 
        #     # segments_color = flow_to_color(track_xy_AVE_dxdy[None,...], 
        #     #                                 clip_flow=None, 
        #     #                                 convert_to_bgr=True)
        #     # segments_color = segments_color[0] / 255.
            
        #     lc = LineCollection(segments, colors=segments_color)
        #     line = ax.add_collection(lc)
            
        #     # plt.plot(tra[:,0],
        #     #           tra[:,1],
        #     #           color=all_mean_track_colors[tra_ii], 
        #     #           lw=2)
        #     #           # alpha=.8) # color by mean! vector
        # plt.grid('off')
        # plt.axis('off')
        # plt.savefig(os.path.join(emb_save_folder, 
        #                          'single_cell_tracks_time_no-overlay.png'), dpi=120, bbox_inches='tight')
        # plt.show()
        
        
        
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)

# fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

# # Create a continuous norm to map from data points to colors
# norm = plt.Normalize(dydx.min(), dydx.max())
# lc = LineCollection(segments, cmap='viridis', norm=norm)
# # Set the values used for colormapping
# lc.set_array(dydx)
# lc.set_linewidth(2)
# line = axs[0].add_collection(lc)
# fig.colorbar(line, ax=axs[0])

# # Use a boundary norm instead
# cmap = ListedColormap(['r', 'g', 'b'])
# norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
# lc = LineCollection(segments, cmap=cmap, norm=norm)
# lc.set_array(dydx)
# lc.set_linewidth(2)
# line = axs[1].add_collection(lc)
# fig.colorbar(line, ax=axs[1])

# axs[0].set_xlim(x.min(), x.max())
# axs[0].set_ylim(-1.1, 1.1)
# plt.show()
        
        
        # polarfile = os.path.join(imgfolder, emb, 'A-P_rotated_data','AVE_rot-%s' %(emb) + '.tif')
        # segfile = os.path.join(imgfolder, emb, 'A-P_rotated_data','AVE_rot-cells_%s' %(emb) + '.tif')
        
        # # create a matrix to save the UMAP clusters. 
        # cells_emb = skio.imread(segfile)
        # # cells_emb_umap = np.zeros_like(cells_emb)
        
        # # metadict = {'segfile': segfile, 
        # #             'tracklookupfile': tracklookupfile,
        # #             'singlecellstatfile': singlecellstatfile,
        # #             'polarfile': polarfile,
        # #             'cells': cells_emb,
        # #             'cells_umap_cluster': cells_emb_umap} # the last one will be updated.! 
        # # embryo_file_dict[emb] = metadict
        # color_palette = np.vstack(sns.color_palette('Spectral', n_colors=16))
        
        # cells_emb_color = skimage.color.label2rgb(cells_emb, bg_label=0, colors=color_palette)
        # # extract just the luminance value of the colors..... 
        
        # # cells_emb_color_lab = skimage.color.rgb2lab(cells_emb_color)
        # cells_emb_color_grey = skimage.color.rgb2gray(cells_emb_color) # this has more contrast.... -> do this instead.
        
        # savefolder =  os.path.join(mastersavefolder, emb); 
        # fio.mkdir(savefolder)
        
        # save_color_file = os.path.join(mastersavefolder, emb, 'color+cell_segmentation_%s.tif' %(emb))
        # skio.imsave(save_color_file, np.uint8(255*cells_emb_color))
        # save_grey_file = os.path.join(mastersavefolder, emb, 'grey+cell_segmentation_%s.tif' %(emb))
        # skio.imsave(save_grey_file, np.uint8(255*cells_emb_color_grey))

