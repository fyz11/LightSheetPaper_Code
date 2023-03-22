#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:11:36 2018

@author: felix

track statistics tools

"""

import numpy as np 
  
def tracks2D_to_3D( tracks, ref_map):
    """
    ref map gives the matching corresponding look up table in 3D coordinates., assume integer inputs and cast down.
    """
    tracks3D = ref_map[tracks[:,:,0].astype(np.int), tracks[:,:,1].astype(np.int),:]

    return tracks3D

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
        temporal_weights = np.ones(tracks_3D.shape[1])
    tracks_3D_diff_linear = (tracks_3D[:,1:] - tracks_3D[:,:-1]) * temporal_weights[None,:] # apply weights to scale temporal dimension.
    
    # Linear Mean to get directionality vector. 
    tracks_3D_diff_linear_mean_scale = np.nanmean(tracks_3D_diff_linear, axis=1)
    tracks_3D_diff_linear_mean_scale_normalised = tracks_3D_diff_linear_mean_scale/(np.linalg.norm(tracks_3D_diff_linear_mean_scale, axis=-1)+1e-8)
    # compute effective isotropic scaling factor to correct curvilinear distance.
    """
     sum_i (w_i * v_i) = k * sum(v_i)
    """
    if temporal_weights is None:
        eff_scale = 1.
    else:
        tracks_3D_diff_linear_mean_no_scale = np.mean((tracks_3D[:,1:] - tracks_3D[:,:-1]), axis=1)
    
        eff_scales = np.linalg.norm(tracks_3D_diff_linear_mean_scale, axis=-1) / np.linalg.norm(tracks_3D_diff_linear_mean_no_scale, axis=-1)
        eff_scale = np.mean(eff_scale[np.logical_and(~np.isnan(eff_scales), 
                                                     ~np.isinf(eff_scales))])
    
    # using start and end point only allows us to compute geodesic...
    track_surf_total_dist, track_surf_total_lines = compute_geodesic_3D_distances_tracks(tracks_time_2D[:,[0,-1]], 
                                                                                        unwrap_params, 
                                                                                        n_samples=n_samples)
    
    track_surf_total_dist = track_surf_total_dist * eff_scale
    track_surf_mean_dist = track_surf_total_dist/float(tracks_3D_diff_linear.shape[1])

    # returns the small magnitude temporal displacements, (effective total_geodesic displacement, geodesic_lines use for computation), (mean_vector_line, mean_vector_geodesic_corrected_distance)
    return tracks_3D_diff_linear, (track_surf_total_dist, track_surf_total_lines), (tracks_3D_diff_linear_mean_scale, tracks_3D_diff_linear_mean_scale_normalised*track_surf_mean_dist)


