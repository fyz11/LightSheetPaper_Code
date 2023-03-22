#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:11:36 2018

@author: felix

track statistics tools

"""

import numpy as np 
  
def set_axes_equal(ax):
    
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    import numpy as np 

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = np.abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = np.abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = np.abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*np.max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    return []

def tracks2D_to_3D( tracks, ref_map):
    """
    ref map gives the matching corresponding look up table in 3D coordinates., assume integer inputs and cast down.
    """
    tracks3D = ref_map[tracks[:,:,0].astype(np.int), tracks[:,:,1].astype(np.int),:]

    return tracks3D


def learn_nearest_neighbor_surf_embedding_model(unwrap_params_3D, nearest_K=1):
    
    # useful when we have to use this a lot.
    from sklearn.neighbors import NearestNeighbors
    
    pts3D = unwrap_params_3D.reshape(-1,unwrap_params_3D.shape[-1])
    nbrs = NearestNeighbors(n_neighbors=nearest_K, algorithm='auto').fit(pts3D.reshape(-1,3))
    
    return nbrs, pts3D
    

def lookup_3D_to_2D_directions_single(unwrap_params, ref_pos_2D, disp_3D, scale=10.):
  
    disp_3D_ = disp_3D / float(np.linalg.norm(disp_3D) + 1e-8) # MUST normalise
    ref_pos_3D = unwrap_params[int(ref_pos_2D[0]), 
                               int(ref_pos_2D[1])]
    
    new_pos_3D = ref_pos_3D + scale*disp_3D_
    
    # to do: modify to incorporate surface normals.
    min_dists = np.linalg.norm(unwrap_params - new_pos_3D[None,None,:], axis=-1) # require the surface normal ? 
    
    min_pos = np.argmin(min_dists)
    new_pos_2D = np.unravel_index(min_pos, unwrap_params.shape[:-1]) # returns as the index coords.
    
    direction_2D = new_pos_2D - ref_pos_2D

    return direction_2D, np.arctan2(direction_2D[1], direction_2D[0])

def lookup_3D_mean(pts2D, unwrap_params):
    
    pts3D = unwrap_params[pts2D[:,0].astype(np.int), 
                          pts2D[:,1].astype(np.int)]
    
    pts3D_mean = np.median(pts3D, axis=0)
    
    pts2D_mean_remap = np.argmin(np.linalg.norm(unwrap_params - pts3D_mean[None,None,:], axis=-1))
    pts2D_mean_remap = np.unravel_index(pts2D_mean_remap, 
                                        unwrap_params.shape[:-1])
    
    return np.hstack(pts2D_mean_remap)


def uniform_sample_line(pt_array, n_samples = 10):
    # simples way to interpolate an n-D curve given the end points.    
    ndim = pt_array.shape[-1]
    
    lines = []
    dist_pt = np.linalg.norm(pt_array[0] - pt_array[1])
    
    N = int(np.minimum(n_samples, dist_pt))
    
    for dim in range(ndim):
        lines.append(np.linspace(pt_array[0,dim], pt_array[1,dim], N+1))
    
    line = np.vstack(lines).T
    
    return line
    

def compute_geodesic_3D_distances_tracks(pt_array_time, unwrap_params, n_samples=10, nearestK=1):

    from sklearn.neighbors import NearestNeighbors
    
    # first build the neighorhood model.
    pts3D_all = unwrap_params.reshape(-1,unwrap_params.shape[-1])
    nbrs = NearestNeighbors(n_neighbors=nearestK, algorithm='auto').fit(pts3D_all) # simplest, alternative is average.
    
    # resample the individual lines. 
    pt_array_dist_lines = []
    pt_array_dists = []
    
    for pt in pt_array_time:
        pt3D = unwrap_params[pt[:,0].astype(np.int), 
                             pt[:,1].astype(np.int)]

        line3D = uniform_sample_line(pt3D, n_samples = n_samples)
        # map to surface
        line3D_indices = nbrs.kneighbors(line3D, return_distance=False)
        line3D_surf = pts3D_all[line3D_indices]; 
        line3D_surf = line3D_surf.mean(axis=1) # this is to remove the singleton dimension due to the kneighbours.

        pt_array_dist_lines.append(line3D_surf) # save the sample line. 
        
        line3D_surf_dist = np.sum(np.linalg.norm(line3D_surf[1:] - line3D_surf[:-1], axis=-1))
        pt_array_dists.append(line3D_surf_dist)
        
    pt_array_dist_lines = np.array(pt_array_dist_lines)
    pt_array_dists = np.array(pt_array_dists)
    
    return pt_array_dists, pt_array_dist_lines

def compute_curvilinear_3D_disps_tracks(tracks_time_2D, unwrap_params, n_samples=20, nearestK=1, temporal_weights=None):

    # temporal_weights is for taking into account the affect of time. 
    # 1. use linearity to sum 'vectors' and obtain unit length directional vectors with optionally temporal weights.
    tracks_3D = tracks2D_to_3D( tracks_time_2D, unwrap_params) # (y,x) convention.
                    
    if temporal_weights is None:
        temporal_weights = np.ones(tracks_3D.shape[1]-1)
    tracks_3D_diff_linear = (tracks_3D[:,1:] - tracks_3D[:,:-1]) * temporal_weights[None,:,None] # apply weights to scale temporal dimension.
    
    # Linear Mean to get directionality vector. 
    tracks_3D_diff_linear_mean_scale = np.nanmean(tracks_3D_diff_linear, axis=1)
    
    # this works.
    tracks_3D_diff_linear_mean_scale_normalised = tracks_3D_diff_linear_mean_scale/(np.linalg.norm(tracks_3D_diff_linear_mean_scale, axis=-1)[...,None]+1e-8)
    # compute effective isotropic scaling factor to correct curvilinear distance.
#    print(tracks_3D_diff_linear_mean_scale_normalised.max())
    """
     sum_i (w_i * v_i) = k * sum(v_i)
    """
    if temporal_weights is None:
        eff_scale = 1.
    else:
        tracks_3D_diff_linear_mean_no_scale = np.mean((tracks_3D[:,1:] - tracks_3D[:,:-1]), axis=1)
    
        eff_scales = np.linalg.norm(tracks_3D_diff_linear_mean_scale, axis=-1) / np.linalg.norm(tracks_3D_diff_linear_mean_no_scale, axis=-1)
#        print(eff_scales.min(), np.nanmedian(eff_scales))
        eff_scales_select = eff_scales[np.logical_and(~np.isnan(eff_scales), 
                                                      ~np.isinf(eff_scales))] <= 100
#        eff_scale = np.nanmean(eff_scales[np.logical_and(~np.isnan(eff_scales), 
#                                                         ~np.isinf(eff_scales))])
        eff_scale = np.nanmean((eff_scales[np.logical_and(~np.isnan(eff_scales), 
                                                      ~np.isinf(eff_scales))])[eff_scales_select])
    
#    print(eff_scale)
#    print(np.nanmedian((eff_scales[np.logical_and(~np.isnan(eff_scales), 
#                                                      ~np.isinf(eff_scales))])[eff_scales_select]))
    # using start and end point only allows us to compute geodesic...
    track_surf_total_dist, track_surf_total_lines = compute_geodesic_3D_distances_tracks(tracks_time_2D[:,[0,-1]], 
                                                                                        unwrap_params, 
                                                                                        n_samples=n_samples)
    
    track_surf_total_dist = track_surf_total_dist * eff_scale
    track_surf_mean_dist = track_surf_total_dist/float(tracks_3D_diff_linear.shape[1])

    # returns the small magnitude temporal displacements, (effective total_geodesic displacement, geodesic_lines use for computation), (mean_vector_line, mean_vector_geodesic_corrected_distance)
#    if return_dist_model == False:
    return tracks_3D_diff_linear, (track_surf_total_dist, track_surf_total_lines), (tracks_3D_diff_linear_mean_scale, tracks_3D_diff_linear_mean_scale_normalised*track_surf_mean_dist[:,None])
#    else:
#        return tracks_3D_diff_linear, (track_surf_total_dist, track_surf_total_lines), (tracks_3D_diff_linear_mean_scale, tracks_3D_diff_linear_mean_scale_normalised*track_surf_mean_dist[:,None])
#    

def construct_surface_tangent_vectors(pt_array_time, 
                                      unwrap_params, direction_vector_3D, K_neighbors=1, 
                                      nbr_model=None, pts3D_all=None, dist_sample=5, return_dist_model=False):

    # primarily an 
    from sklearn.neighbors import NearestNeighbors
    
    # pts contains all the points at which we wish to query the tangent vectors for. 
    if nbr_model is None:
        # fit the model.
        pts3D_all = unwrap_params.reshape(-1,unwrap_params.shape[-1])
        nbr_model = NearestNeighbors(n_neighbors=K_neighbors, algorithm='auto').fit(pts3D_all)
#    else:
#        pts3D_all = unwrap_params.reshape(-1,unwrap_params.shape[-1])
        
    # n_points x n_time x 3
    pt_array_3D = unwrap_params[pt_array_time[...,0].astype(np.int), 
                                pt_array_time[...,1].astype(np.int)]
    
    # we need to construct a new pt in the direction of the given direction
    direction_vector_3D_unit = direction_vector_3D / (np.linalg.norm(direction_vector_3D,axis=-1)[...,None] + 1e-8)
    pt_array_3D_next = pt_array_3D + dist_sample*direction_vector_3D_unit # should dist_sample be the same?
    
    pt_array_3D_next = pt_array_3D_next.reshape(-1,3)
    pt_array_3D_next_surf_indices = nbr_model.kneighbors(pt_array_3D_next, 
                                                         return_distance=False)
    
    pt_array_3D_next_surf = np.median(pts3D_all[pt_array_3D_next_surf_indices], axis=1)
    pt_array_3D_next_surf = pt_array_3D_next_surf.reshape(pt_array_3D.shape)
    
    surf_dir_vectors = pt_array_3D_next_surf - pt_array_3D
    surf_dir_vectors = surf_dir_vectors/(np.linalg.norm(surf_dir_vectors, axis=-1)[...,None] + 1e-8)
    
    if return_dist_model:
        return surf_dir_vectors, nbr_model
    else:
        return surf_dir_vectors
    

def downsample_gaussian(img, sigma):
    
    from skimage.filters import gaussian
    from skimage.transform import resize
    
    ds_level = int(np.rint(np.log2(sigma)))
    
    img_ds = resize(img, np.hstack([np.hstack(img.shape[:2])//ds_level, img.shape[-1]]), preserve_range=True)
    print(img_ds.shape, img.shape)
    img_ds_smooth = gaussian(img_ds, sigma=1, preserve_range=True, multichannel=True)
    print(img_ds_smooth.shape, img.shape)
    img_smooth = resize(img_ds_smooth, img.shape, preserve_range=True)
                
    return img_smooth
  
def radial_smooth(vects, r_dist=5, smooth_theta_dist=10, smooth_radial_dist=10, smooth_tip_sigma=None, smooth_tip_radius=None):
    
    from skimage.filters import gaussian
    
    m, n = vects.shape[:2]
    YY,XX = np.indices(vects.shape[:2])
    
    # central displacements.
    disps_XX = (XX-XX.shape[1]/2.)
    disps_YY = (YY-YY.shape[0]/2.)
    
    # parametrise the circular grid.
    dist_grid = np.sqrt((disps_XX)**2 + (disps_YY)**2)
    arg_grid = np.arctan2(disps_YY, disps_XX)
    
    
    # first implement smoothing at tip.
    if smooth_tip_sigma is not None:
        print('smoothing tip')
        smooth_tip_mask = dist_grid <= smooth_tip_radius
        
        if smooth_tip_sigma > 7:
            vects_tip = downsample_gaussian(vects, sigma=smooth_tip_sigma)
        else:
            vects_tip = gaussian(vects, sigma=smooth_tip_sigma, preserve_range=True, multichannel=True)
            
        vects[smooth_tip_mask>0] = vects_tip[smooth_tip_mask>0]
    
    # smooth in a line manner. -> build consistent theta lines...
    XX_theta_line = dist_grid[...,None] * (np.cos(arg_grid[...,None]+np.arange(-smooth_theta_dist//2, smooth_theta_dist//2+1)[None,None,:]/180.*np.pi)) + XX.shape[1]/2.
    YY_theta_line = dist_grid[...,None] * (np.sin(arg_grid[...,None]+np.arange(-smooth_theta_dist//2, smooth_theta_dist//2+1)[None,None,:]/180.*np.pi)) + YY.shape[0]/2.
    
    # # smooth in a line manner. -> build consistent dist lines...
    # XX + (r_dist*disps_XX/(dist_grid+1e-8))
    
    XX_radial_line = XX[...,None] + (disps_XX/(dist_grid+1e-8))[...,None] * np.arange(-smooth_radial_dist//2, smooth_radial_dist//2+1)[None,None,:] 
    YY_radial_line = YY[...,None] + (disps_YY/(dist_grid+1e-8))[...,None] * np.arange(-smooth_radial_dist//2, smooth_radial_dist//2+1)[None,None,:] 

    XX_theta_line = np.clip(XX_theta_line, 0, n-1); 
    YY_theta_line = np.clip(YY_theta_line, 0, m-1); 
    XX_radial_line = np.clip(XX_radial_line, 0, n-1);
    YY_radial_line = np.clip(YY_radial_line, 0, m-1);

    vects = np.nanmean(np.array([vects[YY_theta_line[...,jj].astype(np.int), XX_theta_line[...,jj].astype(np.int)] for jj in range(XX_theta_line.shape[-1])]), axis=0)
    
    # we should also do filtering along dist.... -> this is 0-180 degrees filtering lines. 
    vects = np.nanmean(np.array([vects[YY_radial_line[...,jj].astype(np.int), XX_radial_line[...,jj].astype(np.int)] for jj in range(XX_radial_line.shape[-1])]), axis=0)
    
    return vects
    

# script to project 2D vectors back into 3D.
def proj_2D_direction_3D_surface(unwrap_params, 
                                 direction_vector_2D,
                                 pt_array_time_2D = None,
                                 K_neighbors=1, 
                                 nbr_model=None, 
                                 nbr_model_pts =None,
                                 dist_sample=10, 
                                 return_dist_model=False,
                                 mode='sparse',
                                 smooth=False,
                                 r_dist=5, 
                                 smooth_theta_dist=10, 
                                 smooth_radial_dist=10, 
                                 smooth_tip_sigma=None, smooth_tip_radius=None):

    from sklearn.neighbors import NearestNeighbors 
    
    # pts contains all the points at which we wish to query the tangent vectors for. 
    if nbr_model is None:
        # fit the model.
        pts3D_all = unwrap_params.reshape(-1,unwrap_params.shape[-1])
        nbr_model = NearestNeighbors(n_neighbors=K_neighbors, algorithm='auto').fit(pts3D_all)
    else:
        nbr_model = nbr_model
#        pts3D_all = unwrap_params.reshape(-1,unwrap_params.shape[-1])
        pts3D_all = nbr_model_pts.copy()
        
    if mode == 'dense':
        YY, XX = np.indices(unwrap_params.shape[:2])
        pt_array_time_2D = np.dstack([YY,XX])
        
    pt_array_time_2D_next = pt_array_time_2D + dist_sample*direction_vector_2D
    pt_array_time_2D_next[...,0] = np.clip(pt_array_time_2D_next[...,0], 0, unwrap_params.shape[0]-1)
    pt_array_time_2D_next[...,1] = np.clip(pt_array_time_2D_next[...,1], 0, unwrap_params.shape[1]-1)
     
    # n_points x n_time x 3
    pt_array_3D = unwrap_params[pt_array_time_2D[...,0].astype(np.int), 
                                pt_array_time_2D[...,1].astype(np.int)]
    
    pt_array_3D_next = unwrap_params[pt_array_time_2D_next[...,0].astype(np.int), 
                                     pt_array_time_2D_next[...,1].astype(np.int)]
        
    # we need to construct a new pt. (why is this bit needed?)
    
    if mode == 'sparse':
        pt_array_3D_next = pt_array_3D_next.reshape(-1,3)
        pt_array_3D_next_surf_indices = nbr_model.kneighbors(pt_array_3D_next, 
                                                             return_distance=False)
        
    #    pt_array_3D_next_surf = pts3D_all[pt_array_3D_next_surf_indices].mean(axis=1)
        pt_array_3D_next_surf = np.median(pts3D_all[pt_array_3D_next_surf_indices], axis=1)
        pt_array_3D_next_surf = pt_array_3D_next_surf.reshape(pt_array_3D.shape)
    else:
        pt_array_3D_next_surf = pt_array_3D_next.copy()
    
    surf_dir_vectors = pt_array_3D_next_surf - pt_array_3D
    
    """
    implement smoothing in dense mode.... 
    """
    if mode == 'dense':
        if smooth:
            surf_dir_vectors = radial_smooth(surf_dir_vectors, 
                                             r_dist=r_dist, 
                                             smooth_theta_dist=smooth_theta_dist, 
                                             smooth_radial_dist=smooth_radial_dist, 
                                             smooth_tip_sigma=smooth_tip_sigma, 
                                             smooth_tip_radius=smooth_tip_radius)
    
    surf_dir_vectors = surf_dir_vectors/(np.linalg.norm(surf_dir_vectors, axis=-1)[...,None] + 1e-8)
#    surf_dir_vectors = surf_dir_vectors.reshape(pt_array_3D.shape) # return in the same shape format.
    
    
    if return_dist_model:
        return surf_dir_vectors, nbr_model
    else:
        return surf_dir_vectors    



def lookup_3D_to_2D_loc_and_directions(unwrap_params, mean_3D, disp_3D, scale=10., nbrs=None):

    from sklearn.neighbors import NearestNeighbors
    
    if nbrs is None:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(unwrap_params.reshape(-1,3))
#    else:
    
    # normalise the 3D displacement.
    disp_3D_ = disp_3D / (np.linalg.norm(disp_3D, axis=-1)[...,None] + 1e-8) # MUST normalise
    new_pos_3D_ = mean_3D + scale*disp_3D_
    
    pos2D_map_indices = nbrs.kneighbors(mean_3D, return_distance=False)
    pos2D_map_indices_disp = nbrs.kneighbors(new_pos_3D_, return_distance=False)
     
    pos2D_map_indices = pos2D_map_indices.ravel()
    pos2D_map_indices_disp = pos2D_map_indices_disp.ravel()
    
    pos2D_map_indices_ij = np.unravel_index(pos2D_map_indices, unwrap_params.shape[:-1])
    pos2D_map_indices_disp_ij = np.unravel_index(pos2D_map_indices_disp, unwrap_params.shape[:-1])
   
    pos2D_map_indices_ij = np.vstack(pos2D_map_indices_ij).T
    pos2D_map_indices_disp_ij = np.vstack(pos2D_map_indices_disp_ij).T
    
#    print(pos2D_map_indices_disp_ij.shape)

    direction_2D = pos2D_map_indices_disp_ij - pos2D_map_indices_ij
    direction_2D = direction_2D / (np.linalg.norm(direction_2D, axis=-1)[...,None] + 1e-8 )
    
    return pos2D_map_indices_ij, direction_2D




















    