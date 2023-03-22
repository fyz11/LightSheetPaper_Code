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


##### functions for creating a custom gridding scheme given the boundary masks. 
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def draw_polygon_mask(xy, shape):
    
    from skimage.draw import polygon
    img = np.zeros(shape, dtype=np.bool)
    rr, cc = polygon(xy[:,1].astype(np.int),
                     xy[:,0].astype(np.int))
    img[rr, cc] = 1
    
    return img

def tile_uniform_windows_radial_guided_line(imsize, n_r, n_theta, max_r, mid_line, center=None, bound_r=True, zero_angle=None, return_grid_pts=False):
    
    from skimage.segmentation import mark_boundaries
    m, n = imsize
    
    XX, YY = np.meshgrid(range(n), range(m))
    
    if center is None:
        center = np.hstack([m/2., n/2.])

    r2 = (XX - center[1])**2  + (YY - center[0])**2
    r = np.sqrt(r2)
    theta = np.arctan2(YY-center[0],XX-center[1])
    
    if max_r is None:
        if bound_r: 
            max_r = np.minimum(np.abs(np.max(XX-center[1])),
                               np.abs(np.max(YY-center[0])))
        else:
            max_r = np.maximum(np.max(np.abs(XX-center[1])),
                               np.max(np.abs(YY-center[0])))

    """
    construct contour lines that bipartition the space. 
    """
    mid_line_polar = np.vstack(cart2pol(mid_line[:,0] - center[1], mid_line[:,1] - center[0])).T
    mid_line_central = np.vstack(pol2cart(mid_line_polar[:,0], mid_line_polar[:,1])).T
    
    # derive lower and upper boundary lines -> make sure to 
    contour_r_lower_polar = np.array([np.linspace(0, l, n_r//2+1) for l in mid_line_polar[:,0]]).T
    contour_r_upper_polar = np.array([np.linspace(l, max_r, n_r//2+1) for l in mid_line_polar[:,0]]).T
    
    contour_r_lower_lines = [np.vstack(pol2cart(l, mid_line_polar[:,1])).T for l in contour_r_lower_polar][:-1]
    contour_r_upper_lines = [np.vstack(pol2cart(l, mid_line_polar[:,1])).T for l in contour_r_upper_polar][1:]
    
    all_dist_lines = contour_r_lower_lines + [mid_line_central] + contour_r_upper_lines
    all_dist_lines = [np.vstack([ll[:,0] + center[1], ll[:,1] + center[0]]).T  for ll in all_dist_lines]
    all_dist_masks = [draw_polygon_mask(ll, imsize) for ll in all_dist_lines]
    # now construct binary masks. 
    all_dist_masks = [np.logical_xor(all_dist_masks[ii+1], all_dist_masks[ii]) for ii in range(len(all_dist_masks)-1)] # ascending distance. 
    
    """
    construct angle masks to partition the angle space. 
    """
    # 2. partition the angular direction
    angle_masks_list = [] # list of the binary masks in ascending angles.
    
    theta = theta + np.pi
    
    if zero_angle is None:
        theta_bounds = np.linspace(0, 2*np.pi, n_theta+1)
    else:
#        print(np.linspace(0, 2*np.pi, n_theta+1) + (180+360./n_theta/2.)/180.*np.pi + zero_angle/180.*np.pi)
#        theta_bounds = np.mod(np.linspace(0, 2*np.pi, n_theta+1), 2*np.pi) 
        theta_bounds = np.mod(np.linspace(0, 2*np.pi, n_theta+1) + (180.-360./n_theta/2.)/180.*np.pi - zero_angle/180.*np.pi, 2*np.pi)
        print(theta_bounds)

    for ii in range(len(theta_bounds)-1):
        
        #### this works if all angles are within the 0 to 2 pi range. 
        if theta_bounds[ii+1] > theta_bounds[ii]:
            mask_theta = np.logical_and( theta>=theta_bounds[ii], theta <= theta_bounds[ii+1])
        else:
            mask_theta = np.logical_or(np.logical_and(theta>=theta_bounds[ii], theta<=2*np.pi), 
                                       np.logical_and(theta>=0, theta<=theta_bounds[ii+1]))
        angle_masks_list.append(mask_theta)
        
    """
    construct the final set of masks (which is for the angles. )  
    """  
    spixels = np.zeros((m,n), dtype=np.int)
    
    counter = 1
    for ii in range(len(all_dist_masks)):
        for jj in range(len(angle_masks_list)):
            mask = np.logical_and(all_dist_masks[ii], angle_masks_list[jj])

            spixels[mask] = counter 
            counter += 1
        
    if return_grid_pts:
        return spixels, [all_dist_lines, theta_bounds]
    else:
        return spixels 

def resample_curve(x,y,s=0, n_samples=10):
    
    import scipy.interpolate
    
    tck, u = scipy.interpolate.splprep([x,y], s=s)
    unew = np.linspace(0, 1., n_samples)
    out = scipy.interpolate.splev(unew, tck) 
    
    return np.vstack(out).T

def infer_control_pts_xy(radial_lines,thetas, ref_pt,s=0, resolution=360):

    from sklearn.metrics.pairwise import pairwise_distances
    from scipy.optimize import linear_sum_assignment
    # given closed curves and a set of angles, pull out the intersection points.
    grid_pts = []

    for rad_line in radial_lines[1:]:

        rad_line_interp = resample_curve(rad_line[:,0], rad_line[:,1], s=s, n_samples=resolution+1)
        rad_line_thetas = np.arctan2(rad_line_interp[:,1]-ref_pt[1], 
                                     rad_line_interp[:,0]-ref_pt[0])

        # rad_line_thetas[rad_line_thetas<0]  = rad_line_thetas[rad_line_thetas<0] + 2*np.pi # make [0, 2*np.pi]
        rad_line_thetas = rad_line_thetas + np.pi # to be consistent with the way grid is done.

        # find the minimum distance to the query thetas.
        dist_matrix = pairwise_distances(thetas[:,None], rad_line_thetas[:,None])

        i, j = linear_sum_assignment(dist_matrix)
        # print(i,j)
        
        found_pts = np.squeeze(rad_line_interp[j])
        dist_ref = np.linalg.norm(found_pts - ref_pt[None,:], axis=-1)

        # the reconstruction doesn't quite match...
        constructed_pts = np.vstack([dist_ref*np.cos(thetas-np.pi) + ref_pt[0], 
                                     dist_ref*np.sin(thetas-np.pi) + ref_pt[1]]).T
        grid_pts.append(constructed_pts[:-1])
        # grid_pts.append(found_pts[:-1])

    return np.array(grid_pts)


def parse_grid_squares_from_pts(gridlines):

    # points are arranged by increasing radial distance, increasing polar angle. 
    # therefore should be a case of --->|
    #                                   |
    #                                   \/
    grid_int = np.arange(len(gridlines[...,0].ravel())).reshape(gridlines[...,0].shape) # reshape back... 
    
    grid_squares = []
    grid_squares_connectivity = [] 

    n_r, n_theta, _ = gridlines.shape

    for ii in range(n_r-1):
        for jj in range(n_theta-1):
            top_left = gridlines[ii,jj]
            top_right = gridlines[ii,jj+1]
            bottom_right = gridlines[ii+1,jj+1]
            bottom_left = gridlines[ii+1,jj]

            grid_coords = np.vstack([top_left, top_right, bottom_right, bottom_left, top_left]) # create a complete circulation.
            grid_squares.append(grid_coords)
            
            grid_coords_connectivity = np.vstack([grid_int[ii,jj], 
                                                  grid_int[ii,jj+1], 
                                                  grid_int[ii+1,jj+1], 
                                                  grid_int[ii+1,jj]])
            grid_squares_connectivity.append(grid_coords_connectivity)

    grid_squares = np.array(grid_squares)
    grid_squares_connectivity = np.array(grid_squares_connectivity)

    return grid_squares, grid_squares_connectivity


def plot_grid_squares_ax(ax, squarelines, color='w', lw=1):

    for squareline in squarelines:
        if color is not None:
            ax.plot(squareline[:,0], 
                    squareline[:,1], color=color, lw=lw)
        else:
            ax.plot(squareline[:,0], 
                    squareline[:,1], lw=lw)

    return []

def map_intensity_interp2(query_pts, grid_shape, I_ref, method='spline', cast_uint8=False, s=0):

    import numpy as np 
    from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator 
    
    if method == 'spline':
        spl = RectBivariateSpline(np.arange(grid_shape[0]), 
                                  np.arange(grid_shape[1]), 
                                  I_ref,
                                  s=s)
        I_query = spl.ev(query_pts[...,0], 
                         query_pts[...,1])
    else:
        spl = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                       np.arange(grid_shape[1])), 
                                       I_ref, method=method, bounds_error=False, fill_value=0)
        I_query = spl((query_pts[...,0], 
                       query_pts[...,1]))

    if cast_uint8:
        I_query = np.uint8(I_query)
    
    return I_query




# this version doesn't support mesh laplacian if using the corresponding 3D coordinates because the first row and last row maps to the same point and generates the triangles. - use the above triangle version. 
def get_uv_grid_quad_connectivity(grid, return_triangles=False, bounds='spherical'): 
    
    """
    grid must be even!. 
    """
    
    import trimesh
    import numpy as np 
    
    m, n = grid.shape[:2]
    img_grid_indices = np.arange(np.prod(grid.shape[:2])).reshape(grid.shape[:2])
        
    if bounds == 'spherical':

        img_grid_indices_main = np.hstack([img_grid_indices, img_grid_indices[:,0][:,None]])
        squares_main = np.vstack([img_grid_indices_main[:m-1, :n].ravel(),
                                  img_grid_indices_main[1:m, :n].ravel(), 
                                  img_grid_indices_main[1:m, 1:n+1].ravel(),
                                  img_grid_indices_main[:m-1, 1:n+1].ravel()]).T

        # then handle the top and bottom strips separately ... 
        img_grid_indices_top = np.vstack([img_grid_indices[0,n//2:][::-1], 
                                          img_grid_indices[0,:n//2]])
        squares_top = np.vstack([img_grid_indices_top[0, :img_grid_indices_top.shape[1]-1].ravel(),
                                 img_grid_indices_top[1, :img_grid_indices_top.shape[1]-1].ravel(), 
                                 img_grid_indices_top[1, 1:img_grid_indices_top.shape[1]].ravel(),
                                 img_grid_indices_top[0, 1:img_grid_indices_top.shape[1]].ravel()]).T
        
        img_grid_indices_bottom = np.vstack([img_grid_indices[-1,:n//2], 
                                             img_grid_indices[-1,n//2:][::-1]])
        squares_bottom = np.vstack([img_grid_indices_bottom[0, :img_grid_indices_bottom.shape[1]-1].ravel(),
                                    img_grid_indices_bottom[1, :img_grid_indices_bottom.shape[1]-1].ravel(), 
                                    img_grid_indices_bottom[1, 1:img_grid_indices_bottom.shape[1]].ravel(),
                                    img_grid_indices_bottom[0, 1:img_grid_indices_bottom.shape[1]].ravel()]).T

        
        all_squares = np.vstack([squares_main, 
                                 squares_top,
                                 squares_bottom])
    if bounds == 'none':
        all_squares = np.vstack([img_grid_indices[:m-1, :n-1].ravel(),
                                 img_grid_indices[1:m, :n-1].ravel(), 
                                 img_grid_indices[1:m, 1:n].ravel(),
                                 img_grid_indices[:m-1, 1:n].ravel()]).T
    all_squares = all_squares[:,::-1]
    
    if return_triangles:
        all_squares_to_triangles = trimesh.geometry.triangulate_quads(all_squares)
        
        return all_squares, all_squares_to_triangles
    else:
        return all_squares
    
    
def create_mesh(vertices,faces,vertex_colors=None, face_colors=None):

    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices,
                            faces=faces, 
                            process=False,
                            validate=False, 
                            vertex_colors=vertex_colors, 
                            face_colors=face_colors)
    
    return mesh 
    
    

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
    import trimesh
    
    from MOSES.Motion_Analysis import tracks3D_tools as tra3Dtools
    
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
    mastersavefolder = 'E:\\Work\\Projects\\Shankar-AVE_Migration\\Data_Analysis\\LifeAct\\2022-03-09_Colored-Single-Cell-Tracks_Direction-Ice-Magenta3D_cont'
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
    
    
    #### does the raw need flipping or not flipping?
    """
    embryos
    """
    embryos_flipping = {'455-Emb2-225-polar': True,
                        '455-Emb3-000-polar': False,
                        '491--000-polar': False,
                        '505--120-polar': False,
                        '864--180-polar': False}
    
    
    for emb in uniq_embryo[2:3]:
        
        """
        Grab the epicontours information to get the rotation angle 
        """
        savematfile = os.path.join(imgfolder, emb+'_mig_Epi-contour-line.mat')
        contour_mat = spio.loadmat(savematfile)
        
        ve_angle_move = float(contour_mat['ve_angle_move'])
        rot_center = np.hstack(contour_mat['center'])
        
        """
        Get the tracks specifically from the annotation!. 
        """
        # trackfile = r'G:\Work\Research\Shankar\Annotation\Curations\L491\L491 Dec 2020 Final\cell_tracks_w_ids_Holly_polar_distal_corrected_nn_tracking-full-dist30_cell-div-refined-Matt_11-05.csv'
        # annot_tracks = pd.read_csv(trackfile)
        
        # annot_tracks_x = annot_tracks.values[::2, 2:].copy()
        # annot_tracks_y = annot_tracks.values[1::2,2:].copy()
        # annot_tracks = np.array([annot_tracks_x, 
        #                           annot_tracks_y]).transpose(1,2,0)
    
        """
        Grab the unwrapping params in order to remap onto a registered 3D outline.  
        """
        # metricfile: 
        # segfile = 
        # polarfile =  # rotated? or not .... -> should do rotated? -> rotate this and save out.... for this.  
        tracklookupfile = os.path.join(metricfolder, 'single_cell_track_stats_lookup_table_'+emb+'.csv')
        singlecellstatfile = os.path.join(metricfolder, 'single_cell_statistics_table_export-ecc-realxyz-from-demons_'+emb+'.csv')
        
        # we can parse the cell trajectory for each cell to trajectories. 
        single_cell_table = pd.read_csv(singlecellstatfile)
        track_lookup_table = pd.read_csv(tracklookupfile)
        
        TP_mig = single_cell_table['Frame'].values[single_cell_table['Stage'].values=='Migration to Boundary']
        TP_mig = np.min(TP_mig)
        
        # use the AVE_rotated. 
        xy_AVE_rot = np.vstack([single_cell_table['pos_2D_x_rot-AVE'].values, 
                                single_cell_table['pos_2D_y_rot-AVE'].values]).T
        
        # are these already rotated? 
        xy_norot = np.vstack([single_cell_table['pos_2D_x'].values, 
                              single_cell_table['pos_2D_y'].values]).T
        
        xy_AVE_tracks = []
        xy_AVE_tracks_norot = []
        
        polarfile = os.path.join(imgfolder, emb, 'A-P_rotated_data','AVE_rot-%s' %(emb) + '.tif')
        segfile = os.path.join(imgfolder, emb, 'A-P_rotated_data','AVE_rot-cells_%s' %(emb) + '.tif')
        segfile_norot = os.path.join(imgfolder, emb, 'A-P_rotated_data','cells_%s' %(emb) + '.tif')
        unwrap_params_ve =  os.path.join(imgfolder, emb, 'A-P_rotated_data','%s' %(emb) + '_ve_unwrap_params_unwrap_params-geodesic.mat') 
        
        # need to add the original unwrapped image!. 
        polarfile_original = os.path.join(imgfolder, emb, 'A-P_rotated_data','%s_unwrap_ve-epi_vid_geodesic_polar.tif' %(emb))
        img_original = skio.imread(polarfile_original) 
        
        """
        This is the wrapping back to 3D!. 
        """
        unwrap_params_ve = spio.loadmat(unwrap_params_ve)['ref_map_polar_xyz'] # load the unwrapping parameters...... # we need to build a grid... on this!. 
        
        # create a matrix to save the UMAP clusters. 
        cells_emb = skio.imread(segfile)
        cells_emb_norot = skio.imread(segfile_norot)
        
        if embryos_flipping[emb] == True:
            cells_emb_norot = cells_emb_norot[:,::-1].copy()
            # flip also the unwrap_params_ve!. 
            # unwrap_params_ve[....,[1,2]] = unwrap_params_ve[....,[2,1]] # flip this and flip 
            unwrap_params_ve = unwrap_params_ve[::-1] # reverse y direction
            # properly invert the 3D. 
            unwrap_params_ve[...,2] =  - unwrap_params_ve[...,2] + unwrap_params_ve[...,2].max() # does this matter? # we can center this how we like...
            
            
            img_original = img_original[:,::-1].copy() # flip on the y-axis 
        
        TP_mig_end =  single_cell_table['Frame'].max()
        
        
        """
        Viz 1 by mean direction 
        """   
        emb_save_folder = os.path.join(mastersavefolder, emb); fio.mkdir(emb_save_folder)
        
        # emb_save_folder_tracks_w_bg = os.path.join(emb_save_folder, 'single_cell_tracks_direction'); fio.mkdir(emb_save_folder_tracks_w_bg)
        # emb_save_folder_tracks_no_bg = os.path.join(emb_save_folder, 'single_cell_tracks_direction_no-overlay'); fio.mkdir(emb_save_folder_tracks_no_bg)
        
        all_tracks_xy = []
        all_tracks_xy_time = []
        all_tracks_xy_mean_vector = []
        
        all_tracks_xy_norot = [] 
        
        for ii in np.arange(len(track_lookup_table))[:]:
            
            inds = track_lookup_table.iloc[ii].values[1:].copy()
            track_xy_time = np.arange(len(inds))[inds>-1] # invalid is -1 
            
            if len(track_xy_time)> 1: # must have more than 1 point in order to get any color 
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
        
                mean_track_vector = np.mean(track_xy_AVE[1:] - track_xy_AVE[:-1], axis=0)
                # mean_track_vector = mean_track_vector/(np.linalg.norm(mean_track_vector) + 1e-12)
                all_tracks_xy.append(track_xy_AVE)
                all_tracks_xy_time.append(track_xy_time)
                all_tracks_xy_mean_vector.append(mean_track_vector)
                
                
                all_tracks_xy_norot.append(xy_norot[inds[track_xy_time]])
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
                
        all_tracks_xy_mean_vector = np.vstack(all_tracks_xy_mean_vector)
        all_mags = np.linalg.norm(all_tracks_xy_mean_vector, axis=-1); lim = np.percentile(all_mags,90)
        clip_lims = np.hstack([-lim, lim])
        
        all_mean_track_colors = flow_to_color(np.vstack(all_tracks_xy_mean_vector)[None,...], #[...,::-1], 
                                                   clip_flow=clip_lims,
                                                  convert_to_bgr=False)
        all_mean_track_colors = np.squeeze(all_mean_track_colors).astype(np.float32)/255.
        
        # """ 
        # viz by angle.
        # """ 
        # emb_save_folder_tracks_w_bg = os.path.join(emb_save_folder, 'single_cell_tracks_direction'); fio.mkdir(emb_save_folder_tracks_w_bg)
        # emb_save_folder_tracks_no_bg = os.path.join(emb_save_folder, 'single_cell_tracks_direction_no-overlay'); fio.mkdir(emb_save_folder_tracks_no_bg)
        
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
        
        
        # check non-rotation plots fine!. 
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('Tracks-norot')
        # ax.imshow(np.zeros_like(cells_emb_norot[TP_mig_end] == 0), cmap='gray')
        ax.imshow(cells_emb_norot[TP_mig_end] == 0, cmap='gray')
        for tra_ii, tra in enumerate(all_tracks_xy_norot): 
            plt.plot(tra[:,0],
                      tra[:,1],
                      color=all_mean_track_colors[tra_ii], 
                      lw=2, #)
                       alpha=1) # color by mean! vector
        plt.grid('off')
        plt.axis('off')
        # plt.savefig(os.path.join(emb_save_folder, 
        #                          'single_cell_tracks_direction_no-overlay.png'), dpi=120, bbox_inches='tight')
        plt.show()
        # all_tracks_xy_norot
        
        
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(cells_emb[TP_mig_end]==0, cmap='gray')
        # for tra_ii, tra in enumerate(all_tracks_xy): 
        #     plt.plot(tra[:,0],
        #               tra[:,1],
        #               color=all_mean_track_colors[tra_ii], 
        #               lw=2)
        #               # alpha=.8) # color by mean! vector
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(emb_save_folder, 
                                 'single_cells_TP_mig-end.png'), dpi=120, bbox_inches='tight')
        plt.show()
        
        
        
        
        # alternative coloring with time? 
        # this we need to do line segments.... 
        """ 
        viz by time using segments
        """ 
        # get_colors(inp, colormap, vmin=None, vmax=None)
        
        # all_tracks_xy = []
        # all_tracks_xy_time = []
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(cells_emb[TP_mig_end] == 0, cmap='gray')
        for tra_ii, tra in enumerate(all_tracks_xy): 
            
            time_tra = all_tracks_xy_time[tra_ii]
            tra_ = tra.reshape(-1, 1, tra.shape[-1])
            segments = np.concatenate([tra_[:-1], tra_[1:]], axis=1)
            segments_color = get_colors(time_tra, colormap=cm.Reds, vmin=0, vmax=TP_mig_end)
            # # turn into segments? 
            # # colour the directionality of the segments. 
            # segments_color = flow_to_color(track_xy_AVE_dxdy[None,...], 
            #                                 clip_flow=None, 
            #                                 convert_to_bgr=True)
            # segments_color = segments_color[0] / 255.
            
            lc = LineCollection(segments, colors=segments_color)
            line = ax.add_collection(lc)
            
            # plt.plot(tra[:,0],
            #           tra[:,1],
            #           color=all_mean_track_colors[tra_ii], 
            #           lw=2)
            #           # alpha=.8) # color by mean! vector
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(emb_save_folder, 
                                 'single_cell_tracks_time.png'), dpi=120, bbox_inches='tight')
        plt.show()
        
        
        
        # # here we want to make a movie of the temporal evolution of the tracks. 
        # emb_save_folder_movie_tracks = os.path.join(emb_save_folder, 'movies_2D_tracks')
        # fio.mkdir(emb_save_folder_movie_tracks)
        
        # """
        # Worm plots. 
        # """
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
        #                               'single_cell_tracks_time_%s.png' %(str(frame_ii).zfill(3))), 
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
        
        
        """
        smooth the unwrap_params_ve!
        """
        smooth_r_dist = 15 # was 5
        smooth_theta_dist= 15 # 15 # to align the parameters with what was used to determine the curvature of the shape.  
        smooth_radial_dist=20
        smooth_tip_sigma=25
        smooth_tip_radius=50
        
        unwrap_params_ve = tra3Dtools.radial_smooth(unwrap_params_ve, 
                                                        r_dist=smooth_r_dist,  # think we should do more? 
                                                        smooth_theta_dist=smooth_theta_dist, # hm... 
                                                        smooth_radial_dist=smooth_radial_dist, 
                                                        smooth_tip_sigma=smooth_tip_sigma, # this fixes the tip?
                                                        smooth_tip_radius=smooth_tip_radius)
        
        """
        Upsize this!. 
        """
        upsample_size = 1. 
        
        import skimage.transform as sktform 
        unwrap_params_ve = np.array([sktform.resize(unwrap_params_ve[...,ch], output_shape=(np.hstack(unwrap_params_ve.shape[:2])*upsample_size).astype(np.int), 
                                                    preserve_range=True, order=1) for ch in np.arange(3)]).transpose(1,2,0)
        unwrap_params_ve = unwrap_params_ve*upsample_size 
        # img_original = np.uint8(sktform.resize(img_original, output_shape=np.hstack([len(img_original), unwrap_params_ve.shape[0]*upsample_size, unwrap_params_ve.shape[1]*upsample_size]), preserve_range=True))
        
        ### get the valid polar coordinates .... and construct polar meshes in3D .
        YY, XX = np.indices(unwrap_params_ve.shape[:-1])
        M, N = unwrap_params_ve.shape[:-1]
        polar_valid_mask = np.sqrt((XX-N/2.)**2 + (YY-M/2.)**2) <= N/2./1.2 
        
        # Build grid
        
        epi_contour_line = np.array([N/2./1.2 /2.* np.cos(np.linspace(0,2*np.pi)) + N/2., N/2./1.2 /2.* np.sin(np.linspace(0,2*np.pi)) + M/2.]).T 
        
        plt.figure()
        plt.imshow(polar_valid_mask)
        plt.plot(epi_contour_line[:,0], 
                 epi_contour_line[:,1], 'r.')
        plt.show()
        
        
        # this will be the master polar grid region .
        polar_grid = tile_uniform_windows_radial_guided_line(imsize=unwrap_params_ve.shape[:-1], 
                                                            n_r=4, 
                                                            n_theta=8,
                                                            max_r=unwrap_params_ve.shape[0]/2./1.2,
                                                            mid_line = epi_contour_line[:,:],
                                                            center=None, 
                                                            bound_r=True,
                                                            zero_angle = 0,#values[0],
                                                            return_grid_pts=False)
    
        # uniq_polar_regions = np.unique(polar_grid)[1:]
        grid_center = np.hstack([polar_grid.shape[1]/2.,
                                 polar_grid.shape[0]/2.])
        uniq_polar_regions = np.unique(polar_grid)[1:]
        
        # assign tracks to unique regions. 
        # uniq_polar_spixels = [(polar_grid==r_id)[meantracks_ve[:,0,0], meantracks_ve[:,0,1]] for r_id in uniq_polar_regions]
        
        
        supersample_r = 12
        supersample_theta = 12

        # # prop a finer grid which we will use for obtaining canonical PCA deformation maps and computing average areal changes.
        # polar_grid_fine, polar_grid_fine_ctrl_pts = tile_uniform_windows_radial_guided_line(imsize=unwrap_params_ve.shape[:-1], 
        #                                                             n_r=supersample_r*4, 
        #                                                             n_theta=(supersample_theta+1)*8,
        #                                                             max_r=unwrap_params_ve.shape[0]/2./1.2,
        #                                                             mid_line = epi_contour_line[:,:],
        #                                                             center=None, 
        #                                                             bound_r=True,
        #                                                             zero_angle = 0, #.values[0], 
        #                                                             return_grid_pts=True)
    
        # # # assign tracks to unique regions. 
        # # uniq_polar_spixels = [(polar_grid==r_id)[meantracks_ve[:,0,0], meantracks_ve[:,0,1]] for r_id in uniq_polar_regions]
        # polar_grid_fine_coordinates = infer_control_pts_xy(polar_grid_fine_ctrl_pts[0],
        #                                                 polar_grid_fine_ctrl_pts[1], 
        #                                                 ref_pt=grid_center,s=0, resolution=360)

        # polar_grid_fine_coordinates = np.squeeze(polar_grid_fine_coordinates)
        
        # #create periodic boundary conditions.
        # polar_grid_fine_coordinates = np.hstack([polar_grid_fine_coordinates, 
        #                                         polar_grid_fine_coordinates[:,0][:,None]])

        # grid_center_vect = np.array([grid_center]*polar_grid_fine_coordinates.shape[1])
        # polar_grid_fine_coordinates = np.vstack([grid_center_vect[None,:],
        #                                         polar_grid_fine_coordinates])

        # # additionally save polar_grid_fine_coordinates.
        # plt.figure()
        # plt.imshow(polar_grid_fine)
        # plt.plot(polar_grid_fine_coordinates[:,:,1],
        #          polar_grid_fine_coordinates[:,:,0], 'r.')
        # plt.show()
        
        # """
        # parse the grid squares point
        # """
        # # polar_grid_squares = parse_grid_squares_from_pts(polar_grid_coordinates)
        # polar_grid_fine_squares, polar_grid_fine_squares_quads = parse_grid_squares_from_pts(polar_grid_fine_coordinates) # add the connectivity here!. 
        # polar_grid_fine_squares_quads = np.squeeze(polar_grid_fine_squares_quads)        
        # # cut to tri
        # polar_grid_fine_squares_tri = trimesh.geometry.triangulate_quads(polar_grid_fine_squares_quads)
        
        # # unwrap_params_ve
        # # create mesh!. 
        # polar_grid_fine_coordinates_3D = np.array([map_intensity_interp2(query_pts=polar_grid_fine_coordinates.reshape(-1,2)[:,::-1], 
        #                                                        grid_shape=unwrap_params_ve.shape[:-1], 
        #                                                        I_ref=unwrap_params_ve[...,ch], 
        #                                                        method='linear', 
        #                                                        cast_uint8=False, 
        #                                                        s=0) for ch in np.arange(3)] ).T
        
        # mesh3D = trimesh.Trimesh(vertices=polar_grid_fine_coordinates_3D, 
        #                          faces=polar_grid_fine_squares_tri[:,::-1],
        #                          process=False,
        #                          validate=False) # ok so the coordinates is kinda of not good... so we just apply some Laplacian smoothing. 
        # # trimesh.smoothing.filter_laplacian(mesh3D, 
        # #                                    lamb=0.5, 
        # #                                    iterations=10, 
        # #                                    implicit_time_integration=True, 
        # #                                    volume_constraint=True, 
        # #                                    laplacian_operator=trimesh.smoothing.laplacian_calculation(mesh3D, equal_weight=False))
        # # mesh3D.export('test_remap3d.obj')
        
        """
        Texture the 3D. 
        """        
        # emb_save_folder_movie_tracks3D = os.path.join(emb_save_folder, 'movies_3D_tracks_mesh2-continuous-Hi')
        emb_save_folder_movie_tracks3D = os.path.join(emb_save_folder, 'movies_3D_pastel_cells')
        fio.mkdir(emb_save_folder_movie_tracks3D)
        
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        def set_axes_equal(ax: plt.Axes):
            """Set 3D plot axes to equal scale.
        
            Make axes of 3D plot have equal scale so that spheres appear as
            spheres and cubes as cubes.  Required since `ax.axis('equal')`
            and `ax.set_aspect('equal')` don't work on 3D.
            """
            import numpy as np 
            limits = np.array([
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ])
            origin = np.mean(limits, axis=1)
            radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
            _set_axes_radius(ax, origin, radius)
        
            return []
        
        def _set_axes_radius(ax, origin, radius):
            x, y, z = origin
            ax.set_xlim3d([x - radius, x + radius])
            ax.set_ylim3d([y - radius, y + radius])
            ax.set_zlim3d([z - radius, z + radius])
        
            return []
        
        
        def get_colors(inp, colormap, vmin=None, vmax=None, bg_label=None):
        	import pylab as plt 
        	norm = plt.Normalize(vmin, vmax)
        
        	colored = colormap(norm(inp))
        	if bg_label is not None:
        		colored[inp==bg_label] = 0 # make these all black!
        
        	return colored
        
        from skimage.draw import line_aa, disk, line_nd
        import skimage.morphology as skmorph
        import skimage.segmentation as sksegmentation 
        import skimage.exposure as skexposure 
        import scipy.ndimage as ndimage 
        import skimage.filters as skfilters 
        # sizes = np.shape(cells_emb[0])
        # height = float(sizes[0])
        # width = float(sizes[1])
                
        from matplotlib import cm 
        
        """
        rotate the unwrap_params! to align with the rotated version!
        """
        unwrap_params_ve_rot = unwrap_params_ve.copy()
        for ch_no in range(unwrap_params_ve.shape[-1]):
            unwrap_params_ve_rot[...,ch_no] = sktform.rotate(unwrap_params_ve[...,ch_no], angle=90-ve_angle_move, preserve_range=True)
            
        # copy over this variable!. 
        unwrap_params_ve = unwrap_params_ve_rot.copy()
        
        
        color_palette = np.vstack(sns.color_palette('pastel', n_colors=16)) 
        cells_emb_color = skimage.color.label2rgb(cells_emb, bg_label=0, colors=color_palette) # just save this out too!. 
        # cells_emb_color[cells_emb==0] = 1
        skio.imsave(os.path.join(emb_save_folder_movie_tracks3D, 'pastel_cells-'+emb+'.tif'), 
                    np.uint8(255*cells_emb_color))
        
        
        # cells_emb_color = np.uint8(255*cells_emb_color)
        
        worm_length=10
        for frame_ii in np.arange(len(cells_emb))[:]:
            
            print(frame_ii)
            cell_img_plot = cells_emb_color[frame_ii].copy()
            
            
            # for tra_ii, tra in enumerate(annot_tracks): 
            #     # time_tra = all_tracks_xy_time[tra_ii] 
            #     # tra = tra * upsample_size # apply the appropriate resize!. 
                
            #     time_tra = np.arange(len(tra))[tra[:,0]>1]
            #     tra = tra[tra[:,0]>1].copy() # only take the subset that is valid!. 
                
            #     # tra = rotate_tracks(tra[None,...], angle=-(90-ve_angle_move), 
            #     #                     center=rot_center, transpose=False)[0]
            #     tra = tra * upsample_size
            #     # tra = tra[...,::-1]
            #     # tra_ = tra.reshape(-1, 1, tra.shape[-1])
                
            #     # check if we need to plot the current track!. 
            #     # if frame_ii in time_tra:
            #         # we need to plot. 
                
            #     if time_tra[0] <=frame_ii : 
            #         # trajectory should be plotted. 
            #         start_ind = 0 
                    
            #         if time_tra[-1] <=frame_ii:
            #             end_ind = len(time_tra) # plot the full trajectory as much of it!. 
            #         else:
            #             end_ind = np.arange(len(time_tra))[time_tra==frame_ii][0] # plot only up to the current timepoint!. 
                    
            #         # # draw the track in cyan. 
            #         # for seg_ii in np.arange(worm_length-1):
            #         for jj in np.arange(start_ind, end_ind-1):
            #             # track_draw = line_aa(int(tra[start_ind, 1]), int(tra[start_ind,0]), 
            #             #                      int(tra[end_ind,1]), int(tra[end_ind,0]))
            #             track_draw = line_aa(int(tra[jj, 1]), int(tra[jj,0]), 
            #                                  int(tra[jj+1,1]), int(tra[jj+1,0]))
            #             # track_draw = line_nd(tra[start_ind,::-1], tra[end_ind,::-1], endpoint=True, integer=False)
                        
            #             # track_mask = np.zeros(gray_img.shape, dtype=np.bool)
            #             # track_mask[track_draw[0].astype(np.int), 
            #             #             track_draw[1].astype(np.int)] = 1
            #             # track_mask = ndimage.gaussian_filter(track_mask*1., sigma=3) 
            #             # track_mask = track_mask/(track_mask.max()+1e-8)> 0.5
            #             # # track_mask = skmorph.binary_dilation(track_mask, skmorph.disk(1))
            #             # # plt.figure()
            #             # # plt.imshow(track_mask)
            #             # # plt.show()
            #             # # track_mask = skmorph.binary_dilation(track_mask, skmorph.disk(1))
            #             cell_img_plot[track_draw[0].astype(np.int), 
            #                           track_draw[1].astype(np.int)] = np.hstack([0,1,1])[None,:]
            #             # cell_img_plot[track_mask>0] = np.hstack([0,1,1])[None,:]
                        
            #         # # create a  disk
            #         # disk_draw = disk(tra[end_ind,::-1], 4, shape=gray_img.shape)
            #         # cell_img_plot[disk_draw[0].astype(np.int), 
            #         #               disk_draw[1].astype(np.int)] = np.hstack([1,1,1])[None,:]
                    
            
            plt.figure()
            plt.imshow(cell_img_plot)
            plt.show()
                    
            #### ok so now we can map the valid region!. 
            grid_quads, grid_tri = get_uv_grid_quad_connectivity(polar_valid_mask, 
                                                                    return_triangles=True, 
                                                                    bounds='none')
    
            grid_pts = np.dstack(np.indices(polar_valid_mask.shape)).reshape(-1,2)
            grid_pts = np.hstack([np.zeros(len(grid_pts))[:,None], 
                                  grid_pts])
            grid_bool = polar_valid_mask.ravel()
            
            
            # remove all triangles with background edges!. 
            # invalid_pts = np.arange(len(grid_bool))[grid_bool==0]
            face_bool = grid_bool[grid_tri].copy()
            face_invalid_index = np.unique(np.argwhere(face_bool==0)[:,0])
            face_keep_index = np.setdiff1d(np.arange(len(grid_tri)), face_invalid_index)
            # keep_tri_indices = np.hstack([iii for iii in np.arange(len(grid_tri)) if np.sum(np.intersect1d(grid_tri, invalid_pts))==0])
            
            mesh_2D = create_mesh(grid_pts[:,:], grid_tri[face_keep_index][:,::-1])
            mesh_2D_comps = mesh_2D.split(only_watertight=False)
            mesh_2D_submesh = mesh_2D_comps[np.argmax([len(ccc.vertices) for ccc in mesh_2D_comps])]
                    
            ### now get the colors..... 
            I_verts_ch = cell_img_plot[mesh_2D_submesh.vertices[:,1].astype(np.int), 
                                        mesh_2D_submesh.vertices[:,2].astype(np.int)].copy()
            
            # create a new 3D mesh!. 
            mesh3D_pts = unwrap_params_ve[mesh_2D_submesh.vertices[:,1].astype(np.int), 
                                          mesh_2D_submesh.vertices[:,2].astype(np.int)].copy()
            
            
            """
            Export the mesh to take a look at !. 
            """
            mesh3D_cells = create_mesh(vertices=mesh3D_pts,
                                        faces=mesh_2D_submesh.faces[:,::-1],
                                        vertex_colors=np.uint8(255*I_verts_ch))
            
            mesh3D_cells.export(os.path.join(emb_save_folder_movie_tracks3D, 
                                              'single_cells_time_%s.obj' %(str(frame_ii).zfill(3))))
        

        """
        Do a static plot!. 
        """
        
        
        # """
        # Export the mesh to take a look at !. 
        # """
        # gray_img = img_original[TP_mig] / 255. ; 
        # gray_img = gray_img[...,1].copy()
        # gray_img = sktform.resize(gray_img, output_shape=np.hstack(img_original[0].shape[:2])*upsample_size, preserve_range=True)
        # # cell_img_overlay = cells_emb_norot[frame_ii] == 0; cell_img_overlay = sktform.resize(cell_img_overlay, output_shape=np.hstack(img_original[0].shape[:2])*upsample_size, preserve_range=True)
        # cell_img_overlay = sktform.resize(cells_emb_norot[TP_mig], output_shape=np.hstack(img_original[0].shape[:2])*upsample_size, preserve_range=True, order=1)
        # # cell_img_plot = 5.*img_original[frame_ii] / 255. + (1-0.5)*get_colors(cells_emb_norot[frame_ii] == 0, 
        # #                                                                        cm.gray, 
        # #                                                                        vmin=0, vmax=1, bg_label=None)[...,:3]
        # cell_img_plot = 1*get_colors(skexposure.equalize_adapthist(gray_img*1., clip_limit=0.01), cm.gray)[...,:3] # + (1.-0.5) * get_colors(cell_img_overlay>0.5, cm.gray, vmin=0, vmax=1, bg_label=None)[...,:3]
        # # cell_img_plot = sksegmentation.mark_boundaries(cell_img_plot, cell_img_overlay.astype(np.int64), color=(1,1,0))
        # # cell_img_plot_overlay = get_colors(cells_emb_norot[frame_ii] == 0, 
        # #                                    cm.gray, 
        # #                                    vmin=0, vmax=1, bg_label=None)[...,:3]
        # cell_img_overlay_binary = (cell_img_overlay>0.5) == 0
        # # cell_img_overlay_binary = ndimage.gaussian_filter(cell_img_overlay_binary*1., sigma=1) > 0.15
        # # plt.figure()
        # # plt.imshow(cell_img_overlay_binary)
        # # plt.show()
        # # cell_img_overlay_binary = cell_img_overlay_binary/(cell_img_overlay_binary.max()+1e-8) > 0.5
        # cell_img_overlay_binary = skmorph.binary_dilation(cell_img_overlay_binary, skmorph.square(1))
        # cell_img_overlay_binary_fill = ndimage.morphology.binary_fill_holes(skmorph.binary_dilation(cell_img_overlay_binary==0, skmorph.disk(3)))
        # cell_img_overlay_binary = np.logical_and(cell_img_overlay_binary_fill, cell_img_overlay_binary)
        # cell_img_plot[cell_img_overlay_binary>0] = np.hstack([1,1,0])[None,:]
            
        
        # for tra_ii, tra in enumerate(all_tracks_xy_norot): 
        #     time_tra = all_tracks_xy_time[tra_ii] 
        #     tra = tra * upsample_size # apply the appropriate resize!. 
        #     # tra_ = tra.reshape(-1, 1, tra.shape[-1])
            
        #     # check if we need to plot the current track!. 
        #     # if frame_ii in time_tra:
        #     # we need to plot. 
        #     end_ind = len(time_tra)-1
        #     start_ind = 0
    
        #     # draw the track in cyan. 
        #     # for seg_ii in np.arange(worm_length-1):
                
        #     for jj in np.arange(start_ind, end_ind-1):
        #         # track_draw = line_aa(int(tra[start_ind, 1]), int(tra[start_ind,0]), 
        #         #                      int(tra[end_ind,1]), int(tra[end_ind,0]))
        #         track_draw = line_aa(int(tra[jj, 1]), int(tra[jj,0]), 
        #                              int(tra[jj+1,1]), int(tra[jj+1,0]))
        #         # track_draw = line_nd(tra[start_ind,::-1], tra[end_ind,::-1], endpoint=True, integer=False)
                
        #         # track_mask = np.zeros(gray_img.shape, dtype=np.bool)
        #         # track_mask[track_draw[0].astype(np.int), 
        #         #             track_draw[1].astype(np.int)] = 1
        #         # track_mask = ndimage.gaussian_filter(track_mask*1., sigma=3) 
        #         # track_mask = track_mask/(track_mask.max()+1e-8)> 0.5
        #         # # track_mask = skmorph.binary_dilation(track_mask, skmorph.disk(1))
        #         # # plt.figure()
        #         # # plt.imshow(track_mask)
        #         # # plt.show()
        #         # # track_mask = skmorph.binary_dilation(track_mask, skmorph.disk(1))
        #         cell_img_plot[track_draw[0].astype(np.int), 
        #                       track_draw[1].astype(np.int)] = np.hstack([0,1,1])[None,:]
        
        # ### now get the colors..... 
        # I_verts_ch = cell_img_plot[mesh_2D_submesh.vertices[:,1].astype(np.int), 
        #                            mesh_2D_submesh.vertices[:,2].astype(np.int)].copy()
        
        # # create a new 3D mesh!. 
        # mesh3D_pts = unwrap_params_ve[mesh_2D_submesh.vertices[:,1].astype(np.int), 
        #                               mesh_2D_submesh.vertices[:,2].astype(np.int)].copy()
        
        # mesh3D_cells = create_mesh(vertices=mesh3D_pts,
        #                            faces=mesh_2D_submesh.faces[:,::-1],
        #                            vertex_colors=np.uint8(255*I_verts_ch))
        
        # emb_save_folder_movie_tracks3D_all = os.path.join(emb_save_folder_movie_tracks3D, 
        #                                                   'full')
        # fio.mkdir(emb_save_folder_movie_tracks3D_all)
        # mesh3D_cells.export(os.path.join(emb_save_folder_movie_tracks3D_all, 
        #                                  'single_cell_tracks_time_all.obj'))
        
        
        
            # # plt.figure()
            # # plt.imshow(cell_img_plot)
            # # ax.grid('off')
            # # ax.axis('off')
            
            # # mesh3D_colors = np.array([map_intensity_interp2(polar_grid_fine_coordinates.reshape(-1,2), 
            # #                                       grid_shape=unwrap_params_ve.shape[:-1], 
            # #                                       I_ref=cell_img_plot[...,ch], 
            # #                                       method='nearest', 
            # #                                       cast_uint8=False, 
            # #                                       s=0) for ch in np.arange(3)] ).T
            # # # now mapping this to the 3D vertex. 
            # # mesh3D = trimesh.Trimesh(vertices=polar_grid_fine_coordinates_3D, 
            # #                      faces=polar_grid_fine_squares_tri[:,::-1],
            # #                      vertex_colors=np.uint8(255*mesh3D_colors),
            # #                      process=False,
            # #                      validate=False) # ok so the coordinates is kinda of not good... so we just apply some Laplacian smoothing. 
            # # mesh3D.export('test_remap3d.obj')
            
            
            # # Alternatively do a 3D plot.  # ok this works! and maps exactly.  
            # fig = plt.figure(figsize=(15,15))
            # ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho') # give the orthogonal view!. 
            # ax.scatter(unwrap_params_ve[polar_valid_mask>0, 0],
            #            unwrap_params_ve[polar_valid_mask>0, 1],
            #            unwrap_params_ve[polar_valid_mask>0, 2], 
            #            c=cell_img_plot[polar_valid_mask>0])
            # ax.grid('off')
            # ax.axis('off')
            # ax.view_init(0,90) # this is the orthogonal side view. 
            # set_axes_equal(ax)
            # # # # To remove the huge white borders
            # # # ax.margins(0)
            # # # canvas.draw()       # draw the canvas, cache the renderer
            # # # width, height = fig.get_size_inches() * fig.get_dpi() 
            # # # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            # plt.savefig(os.path.join(emb_save_folder_movie_tracks3D, 
            #                           'single_cell_tracks_time_%s.png' %(str(frame_ii).zfill(3))), 
            #                             dpi=300)#, bbox_inches='tight')
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

