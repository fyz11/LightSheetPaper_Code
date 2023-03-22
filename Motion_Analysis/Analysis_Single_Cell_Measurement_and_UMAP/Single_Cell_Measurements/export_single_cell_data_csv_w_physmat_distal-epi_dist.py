# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:58:50 2021

@author: fyz11
"""

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

def rotate_axis_to_AVE_dirs(vects_2D_time_list, 
                            angle,
                            yx_convention=True,
                            center = [0,0]):
    """
    this works. 
    """
    # ave_vector etc... uses (x,y) convention but vects_2D_time_list is yx
    vects_2D_time_list_out = []
    center = np.hstack(center)

    for vect2D in vects_2D_time_list:

        if yx_convention == True:
            vect2D = vect2D[:,::-1] # reverse the axis. 
            
        vect2D_out = vect2D.copy() # make a copy. 
        nonnanselect = ~np.isnan(vect2D[:,0])
        vect2D_out[nonnanselect] = rotate_pts(vect2D[nonnanselect], 
                                              angle=angle,
                                              center=center)
        
        if yx_convention == True:
            vect2D_out = vect2D_out[:,::-1]

        vects_2D_time_list_out.append(vect2D_out)

    return vects_2D_time_list_out

def align_polar_orientation_axis(pos2D_time_list,
                                 vects2D_time_list,
                                 shape, 
                                 yx_convention=True,
                                 norm_vectors=True,
                                 return_polar_convention=True):
    
    # modify this to project the orientation of the polar -> and deconvolve it to express it in terms of the position 
    
    # this essentially first aligns the vectors and then points all orientations according the theta direction on polar, axis.
    # result of these transforms should allow normal averaging to take place.

    import numpy as np 

    smooth_theta_dist = 10 # arbitrary, just to ensure a small angular displacement. 
    """
    Construct reusable normalise theta vectors
    """
    YY,XX = np.indices(shape)
    
    theta = np.arctan2(YY-shape[0]/2.,XX-shape[1]/2.)
    dists = np.sqrt((YY-shape[0]/2.) ** 2 + (XX-shape[1]/2.)**2)
    
    XX_theta_dir = dists[...,None] * (np.cos(theta[...,None]+np.arange(-smooth_theta_dist//2, smooth_theta_dist//2+1, 2)[None,None,:]/180.*np.pi)) + XX.shape[1]/2.
    YY_theta_dir = dists[...,None] * (np.sin(theta[...,None]+np.arange(-smooth_theta_dist//2, smooth_theta_dist//2+1, 2)[None,None,:]/180.*np.pi)) + YY.shape[0]/2.
    
    # print(YY_theta_dir.shape)
    # normalised theta direction angles. 
    theta_dir_vectors = np.dstack([XX_theta_dir[...,1] - XX_theta_dir[...,0], 
                                   YY_theta_dir[...,1] - YY_theta_dir[...,0]])
    theta_dir_vectors = theta_dir_vectors / (np.linalg.norm(theta_dir_vectors)[...,None] + 1e-8) # normalisation. 
    
    
    # create also the same directions in the radial direction to project onto. 
    # central displacements. going in the positive direction. 
    disps_XX = (XX-XX.shape[1]/2.)
    disps_YY = (YY-YY.shape[0]/2.)
    dist_grid = np.sqrt((disps_XX)**2 + (disps_YY)**2)
    
    radial_dir_vectors = np.dstack([disps_XX, 
                                    disps_YY])
    radial_dir_vectors = radial_dir_vectors / (dist_grid[...,None] + 1e-8)
    
    # fig, ax = plt.subplots()
    # plt.imshow(dist_grid)
    # ax.quiver(XX[::20,::20], 
    #           YY[::20,::20],
    #           radial_dir_vectors[::20,::20,0], 
    #           -radial_dir_vectors[::20,::20,1], color='r')
    # plt.show()
    
    """
    primary processing. 
    """
    vects_2D_time_list_out = [] 
    vects_2D_time_list_out_rel_xy = [] # expresses in relative angular terms of r, theta direction. 

    for ii, vect2D in enumerate(vects2D_time_list):

        pos2D = pos2D_time_list[ii] # should be in yx convention.
        
        if yx_convention == False:
            vect2D = vect2D[:,::-1] # reverse the axis. 
            pos2D = pos2D[:,::-1]

        vect2D_out = vect2D.copy() # make a copy.
        vect2D_out = vect2D_out.astype(np.float32) # must to prevent 0's 
        nonnanselect = ~np.isnan(vect2D[:,0])
        
        vect2D_out_rel_xy = vect2D_out.copy()
        
        pos2D_ = pos2D[nonnanselect].copy()
        vect2D_angle = np.arctan2(vect2D[nonnanselect,1], 
                                  vect2D[nonnanselect,0])

        vect2D_angle[vect2D_angle<-np.pi/2.] = vect2D_angle[vect2D_angle<-np.pi/2.] + np.pi
        vect2D_angle[vect2D_angle>np.pi/2.] = vect2D_angle[vect2D_angle>np.pi/2.] - np.pi 
        assert(np.sum(np.logical_or(vect2D_angle<-np.pi/2, vect2D_angle>np.pi/2.)) == 0)

        # reconstruct the vectors in x,y convention! 
        vect2D_ = np.vstack([np.sin(vect2D_angle), 
                             np.cos(vect2D_angle)]).T

        # reorient vectors to align with polar. 
        vects2D_norm = vect2D_ / (np.linalg.norm(vect2D_, axis=-1)[...,None] + 1e-8) # normalised ver.
        vects2D_norm_theta_dir = theta_dir_vectors[pos2D_[:,0].astype(np.int), 
                                                   pos2D_[:,1].astype(np.int)]
        vects2D_norm_radial_dir = radial_dir_vectors[pos2D_[:,0].astype(np.int), 
                                                     pos2D_[:,1].astype(np.int)]

        sign_vects2D = np.sign(np.sum(vects2D_norm*vects2D_norm_theta_dir, axis=-1))

        # for those not in the same orientation then we do a rotation of 180 ( fast way is to )
        vects2D_correct = vect2D_.copy()
        vects2D_correct[sign_vects2D < 0 ] = -1 * vect2D_[sign_vects2D < 0 ]
            
        # produce a version of the angle directions but in the local polar axis. 
        vects2D_correct_rel_xy = np.vstack([np.sum(vects2D_correct*vects2D_norm_radial_dir, axis=-1), 
                                            np.sum(vects2D_correct*vects2D_norm_theta_dir, axis=-1)]).T
        
        if norm_vectors:
            vects2D_correct = vects2D_correct / (np.linalg.norm(vects2D_correct, axis=-1)[:,None] + 1e-8) # do a normalisation. 
            vects2D_correct_rel_xy = vects2D_correct_rel_xy / (np.linalg.norm(vects2D_correct_rel_xy, axis=-1)[:,None] + 1e-8)
        
        # put back the coordinate convention to yx 
        if yx_convention == True:
            vects2D_correct = vects2D_correct[:,::-1]

        # ok now we put this into the out array. 
        vect2D_out[nonnanselect] = vects2D_correct.copy() # why this is not put back? 
        vects_2D_time_list_out.append(vect2D_out)
        
        vect2D_out_rel_xy[nonnanselect] = vects2D_correct_rel_xy.copy()
        vects_2D_time_list_out_rel_xy.append(vect2D_out_rel_xy)

    return vects_2D_time_list_out, vects_2D_time_list_out_rel_xy


def map_tra_cell_ids_to_row_ids(cell_tracks_cell_ids, 
                                cell_ids_list, 
                                cell_ids_row_list,
                                times):
    
    """
    cell_tracks_ids : tracks given as an array of cell ids with cell_id = 0 being the background 
    cell_ids_list : list of all available cell_ids 
    cell_stats_list : this the particular stat we are trying to map onto the track. 
    """

    n_tracks, n_time = cell_tracks_cell_ids.shape
    cell_track_row_ids_out = np.ones((n_tracks, n_time), dtype=np.int) * -1 # if np.nan 
    
    for track_ii in np.arange(n_tracks):
        for time_ii in times: # evaluate at the specific given times. 

            track_id_time = cell_tracks_cell_ids[track_ii, time_ii]

            if track_id_time > 0: # if not background 
                cell_id_list = np.squeeze(cell_ids_list[time_ii]) # load up all available cell ids at this timepoint. 
                cell_id_select = np.arange(len(cell_id_list))[cell_id_list==track_id_time] # try to match up.
#                print(cell_id_select)
                if len(cell_id_select) > 0:
                    cell_id_select = cell_id_select[0]
#                    print(cell_id_select)
#                    for stat_ii in np.arange(n_stats):
                    cell_track_row_ids_out[track_ii, time_ii] = np.squeeze(cell_ids_row_list[time_ii])[cell_id_select]
                    
    return cell_track_row_ids_out


def read_cell_table_track(tra_row_ids, cell_table):
    
    valid_index = np.arange(len(tra_row_ids))[tra_row_ids>=0]
    lookup_index = tra_row_ids[valid_index]
    
    return valid_index, cell_table.iloc[lookup_index]

def compute_time_diff_neighbor_inds_tracks(stage_cell_neighbors_tra_ids):
    
    n_tra, n_time = stage_cell_neighbors_tra_ids.shape
    
    out = []
    
    for tra_ii in np.arange(n_tra):
        out_tra_ii = []
        for time_ii in np.arange(n_time-1):
            
            t0 = stage_cell_neighbors_tra_ids[tra_ii, time_ii].ravel()
            t1 = stage_cell_neighbors_tra_ids[tra_ii, time_ii+1].ravel()
            
            if len(t0)>0 and len(t1)>0:
                all_ids = np.unique(np.hstack([t0,t1]))
                diff_ids = np.setdiff1d(all_ids, t0) # this will just give the different ids in t1.
                out_tra_ii.append(diff_ids)
            else:
                out_tra_ii.append(np.array([])) # empty
                
        out.append(out_tra_ii)
        
    return out
        

"""
Set up code to do inference of points on the contour with cell centroids. -> would this distance be reliable ? 
# interesting .... -> this notion of time could actually be placed into the regression? 
"""
def resample_curve(x,y,s=0, k=1, n_samples=10):
    
    import scipy.interpolate
    
    tck, u = scipy.interpolate.splprep([x,y], s=s,k=k)
    unew = np.linspace(0, 1., n_samples)
    out = scipy.interpolate.splev(unew, tck) 
    
    return np.vstack(out).T

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

# primary function.
def compute_epi_contour_distance(contour, cell_coords, shape, n_samples=720):
    
    # shape = img shape. 
    # contour in x,y coordinates, 
    # cell_coords also in (x,y)
    YY, XX = np.indices(shape)
    center = np.hstack([shape[1]/2., shape[0]/2.])
    
    # resample in absolute coordinates. 
    contour_resample = resample_curve(contour[:,0], 
                                      contour[:,1], n_samples=n_samples+1)
    
    # get the contour in terms of arg
    contour_resample_rel = contour_resample - center[None,:]
    contour_resample_rel_arg = np.arctan2(contour_resample_rel[:,1],
                                          contour_resample_rel[:,0])
    
    cell_coords_rel = cell_coords - center[None,:]
    cell_coords_arg = np.arctan2(cell_coords_rel[:,1], 
                                 cell_coords_rel[:,0])
    cell_coords_dist_center = np.linalg.norm(cell_coords_rel, axis=-1) # distance from distal tip.  
                                 
    # iterate and match every cell coordinate to a boundary point. 
    match_contour_pt = []
    match_contour_pt_dist_center = []

    # iterating over each cell.     
    for ii in np.arange(len(cell_coords_arg)):
        
        contour_pt_id = np.argmin(np.abs(contour_resample_rel_arg[:-1] - cell_coords_arg[ii]))
        contour_pt = contour_resample[contour_pt_id] ; match_contour_pt.append(contour_pt)
        contour_pt_rel = contour_resample_rel[contour_pt_id]
        dist_contour_cell = np.linalg.norm(contour_pt_rel); match_contour_pt_dist_center.append(dist_contour_cell)
    
    match_contour_pt = np.vstack(match_contour_pt)
    match_contour_pt_dist_center = np.hstack(match_contour_pt_dist_center)
    
    return match_contour_pt, np.vstack([match_contour_pt_dist_center, cell_coords_dist_center]).T
    
    
        
if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    import scipy.io as spio 
    import os 
    import glob
    import pandas as pd 
    import skimage.transform as sktform
    
    voxel_size = 0.3630 #um
    Ts = 5. #min

# =============================================================================
#     File loading 
# =============================================================================
    matfilesfolder = 'C:/Users/fyz11/Documents/Work/Projects/Shankar-AVE_Migration/Data_Analysis/LifeAct/PseudoAnalysis/Data/Cell_Track_Statistics'
    
    matfiles = glob.glob(os.path.join(matfilesfolder, '*.mat'))
    
    # split these into shape and velocity stats.
    shapematfiles = [ff for ff in matfiles if 'cell_track_and_polargrid_stats_' in os.path.split(ff)[-1]]
    # based on these create the corresponding list of velocity stats 
    velocitymatfiles = [os.path.join(os.path.split(ff)[0], os.path.split(ff)[1].replace('cell_track_and_polargrid_stats_', 'cell_track_velocity_stats_')) for ff in shapematfiles]
    cellmatfiles = [os.path.join(os.path.split(ff)[0], os.path.split(ff)[1].replace('cell_track_and_polargrid_stats_', 'cell_stats_')) for ff in shapematfiles]
    cellphysmatfiles = [os.path.join(os.path.split(ff)[0], os.path.split(ff)[1].replace('cell_track_and_polargrid_stats_', 'cell_physical_stats_')) for ff in shapematfiles]
    # cellflowfiles_real = [os.path.join(os.path.split(ff)[0], os.path.split(ff)[1].replace('cell_track_and_polargrid_stats_', 'cell_ve-epi-flow_3Dreal_')) for ff in shapematfiles]
    # celltrack3Dfiles_real = [os.path.join(os.path.split(ff)[0], os.path.split(ff)[1].replace('cell_track_and_polargrid_stats_', 'cell_track_velocity_stats_real_xyz_')) for ff in shapematfiles]
    # # we also need the cell tracks -> this will allow us to get cell ids that precisely correspond. 
    cellflowfiles_real = [os.path.join(os.path.split(ff)[0], os.path.split(ff)[1].replace('cell_track_and_polargrid_stats_', 'cell_ve-epi-flow_3Dreal_')) for ff in shapematfiles]
    celltrack3Dfiles_real = [os.path.join(os.path.split(ff)[0], os.path.split(ff)[1].replace('cell_track_and_polargrid_stats_', 'cell_track_velocity_stats_real_xyz_')) for ff in shapematfiles]
        
# =============================================================================
#     Cell_Track features parsing. 
#     Rationale:
#           - use the start of migration to boundary to understand, the initial position of cells.  
#           - generate all the metadata to ascribe to tracks and to tabulate them all up for learning. 
# =============================================================================

    # first one is L455_Emb3
    for ii in np.arange(len(shapematfiles))[:]: 
        
        # shapematfile = shapematfiles[ii]
        # shapematobj = spio.loadmat(shapematfile)
        # velocitymatfile = velocitymatfiles[ii]
        # velocitymatobj = spio.loadmat(velocitymatfile)
        cellmatfile = cellmatfiles[ii]
        cellmatobj = spio.loadmat(cellmatfile)
        
        physicalmatfile = cellphysmatfiles[ii] # these are both defined on the cell level. ! 
        cellphysmatobj = spio.loadmat(physicalmatfile)
        
        flowfile = cellflowfiles_real[ii]
        cellflowobj = spio.loadmat(flowfile)
        
        celltrackfile3Dreal = celltrack3Dfiles_real[ii]
        celltrack3Dobj = spio.loadmat(celltrackfile3Dreal) #inversion corrected and in true 3d, should need absolutely no other correction. 
        
        print(cellmatfile)
        print('====')
        """
        Load up the associated statistics. # diff areas etc. leave for Matt to compute.  
        """
        embryo = os.path.split(cellmatfile)[-1].split('.mat')[0].split('cell_stats_')[1]
        
        
        """
        load in the relevant cell tracks
        """
        if '455' in embryo:
            # hardcode this in for every embryo to export. 
            celltrackfile = os.path.join(os.path.split(cellmatfiles[ii])[0], 'cell_tracks_w_ids_L455_E3_Composite_Fusion_Seam_Edited_MS_ver4-1x_div-refine_v2.mat')
        if '505' in embryo:
            celltrackfile = os.path.join(os.path.split(cellmatfiles[ii])[0], 'cell_tracks_w_ids_L505_ve_120_MS_edit_ver5_30_12_2020_div-refine.mat')
        if '864' in embryo:
            celltrackfile = os.path.join(os.path.split(cellmatfiles[ii])[0], 'cell_tracks_w_ids_L864_composite_matt_binary_edit_ver1_div-refine.mat')
        if '491' in embryo:
            celltrackfile = os.path.join(os.path.split(cellmatfiles[ii])[0], 'cell_tracks_w_ids_Holly_polar_distal_corrected_nn_tracking-full-dist30_cell-div-refined-Matt_11-05.mat')
        celltrackmatobj = spio.loadmat(celltrackfile)
        
        """
        cell track ids. 
        """
        try:
            cell_tracks_cell_ids = celltrackmatobj['cell_tracks_ids']; # this is in terms of the superpixels? 
        except:
            cell_tracks_cell_ids = celltrackmatobj['cell_track_cell_ids_time']
        
        
        cell_time = np.squeeze(cellmatobj['time'])
        cell_stage = np.squeeze(cellmatobj['stage']); cell_stage = np.hstack([s.strip() for s in cell_stage])
        
        cell_stage_times = [cell_time[cell_stage==stage] for stage in ['Pre-migration', 'Migration to Boundary']]
        
        cell_ve_angle = cellmatobj['ve_angle'].ravel()[0] # get the ve angle. 
        cell_ids = np.squeeze(cellmatobj['cell_ids'])
        cell_areas = np.squeeze(cellmatobj['cell_areas'])
        cell_perims = np.squeeze(cellmatobj['cell_perims'])
        cell_shape_index = np.squeeze(cellmatobj['cell_shape_index'])
        cell_major_length = np.squeeze(cellmatobj['cell_major_length'])
        cell_minor_length = np.squeeze(cellmatobj['cell_minor_length'])
        cell_eccentricity = np.squeeze(cellmatobj['cell_eccentricity'])
        
        # reorient this. and produce angles both in original + corrected coordinates. 
        cell_major_axis_2D = np.squeeze(cellmatobj['cell_major_axis_2D'])
        
        # 'squish' into an array? / string. 
        cell_neighbors_ids = np.squeeze(cellmatobj['cell_neighbors']) # not super useful to give out ... in this format.. # better to give as neighboring track id. 
        # convert to track neighbors. 
        
        # use instead the epi_contour. 
        # polar_grids_coarse = np.squeeze(cellmatobj['polar_grids_coarse'])
        # polar_grids_coarse = np.squeeze(cellmatobj['polar_grids_coarse']) # this ver. is the MOSES propagated ver of the grids. 
        polar_grids_coarse = np.squeeze(cellphysmatobj['polar_grids_coarse_contour']) # lets do this. 
        polar_grids_stage = [polar_grids_coarse[tt[0]] for tt in cell_stage_times] # the polar grid at the start of each phase. specifically add this. 


        # # # cell_minor_axis_2D = np.squeeze(cellmatobj['cell_minor_axis_2D'])
        # # actual frame no. 
        # cell_flow_VE_3D = np.squeeze(cellphysmatobj['cell_flow_VE_3D']) # need further projection onto cell AVE directions as with the cell track speeds. 
        # cell_flow_Epi_3D = np.squeeze(cellphysmatobj['cell_flow_Epi_3D'])
        
        # less 2 frames. 
        cell_demons_flow_VE_AVE_dir = np.squeeze(cellphysmatobj['cell_demons_flow_VE_AVE_dir']) # VE ave 
        cell_demons_flow_VE_AVE_perp_dir = np.squeeze(cellphysmatobj['cell_demons_flow_VE_AVE_perp_dir'])
        cell_demons_flow_VE_AVE_norm_dir = np.squeeze(cellphysmatobj['cell_demons_flow_VE_AVE_norm_dir'])
        
        cell_demons_flow_Epi_AVE_dir = np.squeeze(cellphysmatobj['cell_demons_flow_Epi_AVE_dir'])
        cell_demons_flow_Epi_AVE_perp_dir = np.squeeze(cellphysmatobj['cell_demons_flow_Epi_AVE_perp_dir'])
        cell_demons_flow_Epi_AVE_norm_dir = np.squeeze(cellphysmatobj['cell_demons_flow_Epi_AVE_norm_dir'])
        # actual frame no 
        cell_curvature_VE = np.squeeze(cellphysmatobj['cell_curvature_VE'])
        cell_curvature_Epi = np.squeeze(cellphysmatobj['cell_curvature_Epi'])
        
        # =============================================================================
        #   Computing the distal distance of a cell centroid w.r.t the epi-contour boundary.           
        # =============================================================================
        cell_flow_VE_3D = np.squeeze(cellphysmatobj['cell_flow_VE_3D']) # alternative to the above, 
        cell_flow_Epi_3D = np.squeeze(cellphysmatobj['cell_flow_Epi_3D'])
        
        cell_flow_VE_3D_real = np.squeeze(cellflowobj['cell_flow_VE_3D']) # alternative to the above, 
        cell_flow_Epi_3D_real = np.squeeze(cellflowobj['cell_flow_Epi_3D'])
        
        epi_boundary_line = cellflowobj['epi_contour_line'] # n_time x n_points x 2 (#x,y, coordinate order)
        # resample all the lines at every point ... to get at least 360/720 points...
        
        # =============================================================================
        #      Centroid + Major Axis rotation based on ve_angle. to get consistent angle data. 
        # =============================================================================
        # add centroids 2D_rot         
        cell_centroids_2D = np.squeeze(cellmatobj['cell_centroids_2D']) # rotate this. 
        
        # deduce the distanc of cell to center + distance from cell to closest epi contour point. 
        matched_contour_pts_time = []
        matched_contour_distal_distances_time = []
        
        # compute this for all time. 
        for tt in np.arange(len(cell_centroids_2D))[:]:
            matched_contour_pts, matched_contour_distal_distances = compute_epi_contour_distance(contour=epi_boundary_line[tt], 
                                                                                                 cell_coords=cell_centroids_2D[tt][:,::-1], # (x,y) convention 
                                                                                                 shape=polar_grids_coarse.shape[1:], 
                                                                                                 n_samples=720)
            matched_contour_pts_time.append(matched_contour_pts)
            matched_contour_distal_distances_time.append(matched_contour_distal_distances)
            
        dist_ratio_plot =  (matched_contour_distal_distances_time[0][:,0] - matched_contour_distal_distances_time[0][:,1]) / (matched_contour_distal_distances_time[0][:,0]+0.1)
        
        
        # plot check of computed distances. 
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(polar_grids_coarse[0])
        # ax.scatter(cell_centroids_2D[0][:,1],
        #            cell_centroids_2D[0][:,0], 'r.')
        ax.plot(epi_boundary_line[0,:,0],
                epi_boundary_line[0,:,1],'g-')
        ax.plot(matched_contour_pts_time[0][:,0],
                matched_contour_pts_time[0][:,1],'go', ms=5)
        
        # check by drawing some distance pairings.
        for iii in np.arange(len(matched_contour_pts_time[0]))[::5]:
            ax.plot([matched_contour_pts_time[0][iii,0], cell_centroids_2D[0][iii,1]],
                    [matched_contour_pts_time[0][iii,1], cell_centroids_2D[0][iii,0]],'b-')
        ax.scatter(cell_centroids_2D[0][:,1],
                   cell_centroids_2D[0][:,0], 
                   cmap='coolwarm',
                   c=dist_ratio_plot, vmin=-1,vmax=1)
        plt.grid('off')
        plt.axis('off')
        plt.show()
        
        
        
        # rotation of points to compute relative eccentricity displacements. (check this computation)
        cell_centroids_2D_rot = np.array([rotate_pts(cc_centroids_2D[:,::-1],
                                                      angle=-(90-cell_ve_angle), 
                                                      center=[polar_grids_coarse[0].shape[1]/2.,
                                                              polar_grids_coarse[0].shape[0]/2.])[:,::-1] for cc_centroids_2D in cell_centroids_2D])
        
        # align and rotate the cell major axis. 
        cell_major_axis_2D_rot = rotate_axis_to_AVE_dirs(cell_major_axis_2D, 
                                                                angle = -(90-cell_ve_angle))
        # something wrong here? 
        cell_major_axis_2D_rot, cell_major_axis_2D_rot_rel_polar = align_polar_orientation_axis(cell_centroids_2D_rot, 
                                                                                                cell_major_axis_2D_rot, 
                                                                                                shape=polar_grids_coarse[0].shape)
        # can we now change this to a proper orientation in the same manner as velocity. 
        cell_major_axis_2D_rot_angles = [np.arctan2(vect[:,1], -vect[:,0]) for vect in cell_major_axis_2D_rot] # if we do a 'negative' here should be good? 
        cell_major_axis_2D_rot_angles_vect = [np.vstack([np.sin(angle), np.cos(angle)]).T for angle in cell_major_axis_2D_rot_angles]
        
        # double check the rotation. 
        rot_img = np.uint16(sktform.rotate(polar_grids_coarse[0], 
                                            angle=(90-cell_ve_angle), 
                                            preserve_range=True))
        
        plt.figure(figsize=(10,10))
        plt.imshow(polar_grids_coarse[0])
        plt.plot(cell_centroids_2D[0][:,1], 
                  cell_centroids_2D[0][:,0], 'r.')
        plt.quiver(cell_centroids_2D[0][:,1], 
                    cell_centroids_2D[0][:,0],
                    cell_major_axis_2D[0][:,1],
                    -cell_major_axis_2D[0][:,0], color='r')
        plt.show()
        
        plt.figure(figsize=(10,10))
        plt.imshow(rot_img)
        plt.plot(cell_centroids_2D_rot[0][:,1], 
                  cell_centroids_2D_rot[0][:,0], 'r.')
        plt.quiver(cell_centroids_2D_rot[0][:,1], 
                    cell_centroids_2D_rot[0][:,0],
                    cell_major_axis_2D_rot[0][:,1],
                    -cell_major_axis_2D_rot[0][:,0], color='r')
        plt.show()
        
        plt.figure(figsize=(10,10))
        plt.imshow(rot_img)
        plt.plot(cell_centroids_2D_rot[0][:,1], 
                  cell_centroids_2D_rot[0][:,0], 'r.')
        plt.quiver(cell_centroids_2D_rot[0][:,1], 
                    cell_centroids_2D_rot[0][:,0],
                    cell_major_axis_2D_rot[0][:,1],
                    -cell_major_axis_2D_rot[0][:,0], color='r')
        plt.quiver(cell_centroids_2D_rot[0][:,1], 
                    cell_centroids_2D_rot[0][:,0],
                    np.abs(cell_major_axis_2D_rot[0][:,1]),
                    -cell_major_axis_2D_rot[0][:,0]*0, color='g')
        plt.quiver(cell_centroids_2D_rot[0][:,1], 
                    cell_centroids_2D_rot[0][:,0],
                    cell_major_axis_2D_rot[0][:,1]*0,
                    -np.abs(cell_major_axis_2D_rot[0][:,0]), color='b')
        plt.show()
        
        
        # # checking the reorientation. 
        
        # plt.figure(figsize=(10,10))
        # plt.imshow(rot_img)
        # plt.plot(cell_centroids_2D_rot[0][:,1], 
        #           cell_centroids_2D_rot[0][:,0], 'r.')
        
        # flip = np.hstack(cell_major_axis_2D_rot[0][:,0] > 0)
        
        # plt.quiver(cell_centroids_2D_rot[0][:,1], 
        #             cell_centroids_2D_rot[0][:,0],
        #             cell_major_axis_2D_rot[0][:,1]*-1*flip,
        #             -cell_major_axis_2D_rot[0][:,0]*-1*flip, color='r')
        # plt.show()
        
        # -1 * (np.sign(eccentricity_angle_vect_y_ii) > 0)
        
        
        
        # plt.figure(figsize=(10,10))
        # plt.imshow(rot_img)
        # plt.plot(cell_centroids_2D_rot[0][:,1], 
        #          cell_centroids_2D_rot[0][:,0], 'r.')
        # plt.quiver(cell_centroids_2D_rot[0][:,1], 
        #            cell_centroids_2D_rot[0][:,0],
        #            cell_major_axis_2D_rot_angles_vect[0][:,0],
        #            cell_major_axis_2D_rot_angles_vect[0][:,1], color='r')
        # plt.show()
                   
        """
        1. construct the cell table for individual cells at every timepoint. 
        """
        cell_table = [] # this is the flattened ver. 
        cell_row_ids = [] # for fast lookup when integrating with track data. ? 
        cell_row_counter = 0
        # cell_row_counter_array = []
        
        for jj in np.arange(len(cell_time)): 
            
            cell_row_counter_array_ii = []
            # iterate over time
            TP = cell_time[jj]
            TP_stage = cell_stage[jj]
            
            cell_ids_ii = np.squeeze(cell_ids[jj]) # important to export this out. 
            
            """cell shape measurements"""
            areas_ii = np.squeeze(cell_areas[jj])
            perims_ii = np.squeeze(cell_perims[jj])
            shape_index_ii = np.squeeze(cell_shape_index[jj])
            major_length_ii = np.squeeze(cell_major_length[jj]) 
            minor_length_ii = np.squeeze(cell_minor_length[jj])
            eccentricity_ii = np.squeeze(cell_eccentricity[jj])
            eccentricity_angle_ii = np.squeeze(cell_major_axis_2D_rot_angles[jj])
            
            # retain these to get the proper angles. between [0, np.pi]
            eccentricity_angle_vect_x_ii = np.squeeze(cell_major_axis_2D_rot[jj][:,1]) * eccentricity_ii
            eccentricity_angle_vect_y_ii = np.squeeze(cell_major_axis_2D_rot[jj][:,0]) * eccentricity_ii
            
            # x = perp AVE, y = AVE? 
            eccentricity_angle_vect_x_ii[np.sign(eccentricity_angle_vect_y_ii) < 0] = eccentricity_angle_vect_x_ii[np.sign(eccentricity_angle_vect_y_ii) < 0] * -1 
            eccentricity_angle_vect_y_ii[np.sign(eccentricity_angle_vect_y_ii) < 0] = eccentricity_angle_vect_y_ii[np.sign(eccentricity_angle_vect_y_ii) < 0] * -1 # positive y image down. 
            
            eccentricity_angle_vect_x_ii = np.abs(eccentricity_angle_vect_x_ii)
            eccentricity_angle_vect_y_ii = np.abs(eccentricity_angle_vect_y_ii)
            
            eccentricity_angle_vect_rel_polar_radial_ii = np.squeeze(cell_major_axis_2D_rot_rel_polar[jj][:,0]) * eccentricity_ii
            eccentricity_angle_vect_rel_polar_theta_ii = np.squeeze(cell_major_axis_2D_rot_rel_polar[jj][:,1]) * eccentricity_ii
            
            cell_centroids_2D_ii = np.squeeze(cell_centroids_2D[jj])
            cell_centroids_2D_rot_ii = np.squeeze(cell_centroids_2D_rot[jj])
            
            
            # cell centroid position relative to the epi contour
            matched_contour_pts_time_ii = matched_contour_pts_time[jj]
            matched_contour_distal_distances_time_ii = matched_contour_distal_distances_time[jj] # contour 1st then cell in column
            distal_to_epi_contour_ii = matched_contour_distal_distances_time_ii[:,0].copy()
            distal_to_cell_ii = matched_contour_distal_distances_time_ii[:,1].copy()
            cell_to_epi_contour_ratio_ii = (distal_to_epi_contour_ii - distal_to_cell_ii) / (distal_to_epi_contour_ii+0.1) # don't need the regularization in denominator, but just in case.
        
            # figure out the static cell quad_id # using the cell centroids at this time point.
            
            # if TP_stage == 'Pre-migration':
            #     polar_grid_stage_ii = polar_grids_stage[0]
            # if TP_stage == 'Migration to Boundary':
            #     polar_grid_stage_ii = polar_grids_stage[1]
                
            # # this is referencing the static at the start of the stage? 
            # cell_quad_ids_static = polar_grid_stage_ii[cell_centroids_2D_ii[:,0].astype(np.int), 
            #                                            cell_centroids_2D_ii[:,1].astype(np.int)]
            
            # give the quad id where they are mapped to over time according to the centroid id? -> are we sure of this ? or ... do we use the cell id? -> just need to check this. 
            cell_quad_ids_static = polar_grids_coarse[jj][cell_centroids_2D_ii[:,0].astype(np.int), 
                                                          cell_centroids_2D_ii[:,1].astype(np.int)]
            
            # the dynamic will need to be inferred once again from the track ancestors. -> get this and write this. for each track. 
            
            """cell physics measurements"""
            flow_VE_3D_ii = cell_flow_VE_3D[TP]
            flow_Epi_3D_ii = cell_flow_Epi_3D[TP]
            
            flow_VE_3D_ii_real = cell_flow_VE_3D_real[TP]
            flow_Epi_3D_ii_real = cell_flow_Epi_3D_real[TP]
            
            curvature_VE_ii = np.squeeze(cell_curvature_VE[jj])
            curvature_Epi_ii = np.squeeze(cell_curvature_Epi[jj])
            
            # need to selectively only update for the various demons
            if TP_stage == 'Pre-migration':
                if TP < cell_stage_times[0][-1]: 
                    # not the last. 
                    demons_flow_VE_AVE_dir_ii = np.squeeze(cell_demons_flow_VE_AVE_dir[TP])
                    demons_flow_VE_AVE_perp_dir_ii = np.squeeze(cell_demons_flow_VE_AVE_perp_dir[TP])
                    demons_flow_VE_AVE_norm_dir_ii = np.squeeze(cell_demons_flow_VE_AVE_norm_dir[TP])
                    demons_flow_Epi_AVE_dir_ii = np.squeeze(cell_demons_flow_Epi_AVE_dir[TP])
                    demons_flow_Epi_AVE_perp_dir_ii = np.squeeze(cell_demons_flow_Epi_AVE_perp_dir[TP])
                    demons_flow_Epi_AVE_norm_dir_ii = np.squeeze(cell_demons_flow_Epi_AVE_norm_dir[TP])
                else:
                    demons_flow_VE_AVE_dir_ii = np.zeros(len(curvature_VE_ii)); demons_flow_VE_AVE_dir_ii[:] = np.nan
                    demons_flow_VE_AVE_perp_dir_ii = np.zeros(len(curvature_VE_ii)); demons_flow_VE_AVE_perp_dir_ii[:] = np.nan
                    demons_flow_VE_AVE_norm_dir_ii = np.zeros(len(curvature_VE_ii)); demons_flow_VE_AVE_norm_dir_ii[:] = np.nan
                    demons_flow_Epi_AVE_dir_ii = np.zeros(len(curvature_Epi_ii)); demons_flow_Epi_AVE_dir_ii[:] = np.nan
                    demons_flow_Epi_AVE_perp_dir_ii = np.zeros(len(curvature_Epi_ii)); demons_flow_Epi_AVE_perp_dir_ii[:] = np.nan
                    demons_flow_Epi_AVE_norm_dir_ii = np.zeros(len(curvature_Epi_ii)); demons_flow_Epi_AVE_norm_dir_ii[:] = np.nan
                    
            if TP_stage == 'Migration to Boundary':
                
                if TP < cell_stage_times[1][-1]: 
                    # not the last. 
                    demons_flow_VE_AVE_dir_ii = np.squeeze(cell_demons_flow_VE_AVE_dir[TP-1]) # note the -1 
                    demons_flow_VE_AVE_perp_dir_ii = np.squeeze(cell_demons_flow_VE_AVE_perp_dir[TP-1])
                    demons_flow_VE_AVE_norm_dir_ii = np.squeeze(cell_demons_flow_VE_AVE_norm_dir[TP-1])
                    demons_flow_Epi_AVE_dir_ii = np.squeeze(cell_demons_flow_Epi_AVE_dir[TP-1])
                    demons_flow_Epi_AVE_perp_dir_ii = np.squeeze(cell_demons_flow_Epi_AVE_perp_dir[TP-1])
                    demons_flow_Epi_AVE_norm_dir_ii = np.squeeze(cell_demons_flow_Epi_AVE_norm_dir[TP-1])
                else:
                    demons_flow_VE_AVE_dir_ii = np.zeros(len(curvature_VE_ii)); demons_flow_VE_AVE_dir_ii[:] = np.nan
                    demons_flow_VE_AVE_perp_dir_ii = np.zeros(len(curvature_VE_ii)); demons_flow_VE_AVE_perp_dir_ii[:] = np.nan
                    demons_flow_VE_AVE_norm_dir_ii = np.zeros(len(curvature_VE_ii)); demons_flow_VE_AVE_norm_dir_ii[:] = np.nan
                    demons_flow_Epi_AVE_dir_ii = np.zeros(len(curvature_Epi_ii)); demons_flow_Epi_AVE_dir_ii[:] = np.nan
                    demons_flow_Epi_AVE_perp_dir_ii = np.zeros(len(curvature_Epi_ii)); demons_flow_Epi_AVE_perp_dir_ii[:] = np.nan
                    demons_flow_Epi_AVE_norm_dir_ii = np.zeros(len(curvature_Epi_ii)); demons_flow_Epi_AVE_norm_dir_ii[:] = np.nan
                    
            # # # cell_minor_axis_2D = np.squeeze(cellmatobj['cell_minor_axis_2D'])
            
            # we now update the rows. 
            n_cells = len(cell_ids_ii)
            
            for cell_ii in np.arange(n_cells):
                cell_info = np.hstack([ cell_row_counter, # a unique row id for lookup. 
                                        embryo, 
                                        TP_stage, 
                                        TP,
                                        cell_quad_ids_static[cell_ii], 
                                        cell_ids_ii[cell_ii],
                                        areas_ii[cell_ii], 
                                        perims_ii[cell_ii], 
                                        shape_index_ii[cell_ii], 
                                        major_length_ii[cell_ii], 
                                        minor_length_ii[cell_ii], 
                                        eccentricity_ii[cell_ii], 
                                        eccentricity_angle_ii[cell_ii], 
                                        eccentricity_angle_vect_x_ii[cell_ii], # verify this ... 
                                        eccentricity_angle_vect_y_ii[cell_ii], 
                                        eccentricity_angle_vect_rel_polar_radial_ii[cell_ii], 
                                        eccentricity_angle_vect_rel_polar_theta_ii[cell_ii], 
                                        cell_centroids_2D_ii[cell_ii][1], 
                                        cell_centroids_2D_ii[cell_ii][0],
                                        cell_centroids_2D_rot_ii[cell_ii][1], 
                                        cell_centroids_2D_rot_ii[cell_ii][0],
                                        # add in the computed distal coordinate position from the epi-contour.(fraction))
                                        matched_contour_pts_time_ii[cell_ii][0], #x-coordinate first. 
                                        matched_contour_pts_time_ii[cell_ii][1],
                                        # matched_contour_distal_distances_time_ii[cell_ii], 
                                        distal_to_epi_contour_ii[cell_ii]*2.*voxel_size, # note the 2! -> polar is half 
                                        distal_to_cell_ii[cell_ii]*2.*voxel_size,
                                        cell_to_epi_contour_ratio_ii[cell_ii], # don't need the regularization in denominator, but just in case.
                                        # for these we need to also apply scaling!. 
                                        demons_flow_VE_AVE_dir_ii[cell_ii] * voxel_size / float(Ts), 
                                        demons_flow_VE_AVE_perp_dir_ii[cell_ii] * voxel_size / float(Ts),
                                        demons_flow_VE_AVE_norm_dir_ii[cell_ii] * voxel_size / float(Ts),
                                        demons_flow_Epi_AVE_dir_ii[cell_ii] * voxel_size / float(Ts),
                                        demons_flow_Epi_AVE_perp_dir_ii[cell_ii] * voxel_size / float(Ts),
                                        demons_flow_Epi_AVE_norm_dir_ii[cell_ii] * voxel_size / float(Ts), # keep in mins only. 
                                        curvature_VE_ii[cell_ii] / voxel_size, # as its units = 1/length
                                        curvature_Epi_ii[cell_ii] / voxel_size,
                                        flow_VE_3D_ii[cell_ii] * voxel_size / float(Ts), # add this in to test. 
                                        flow_Epi_3D_ii[cell_ii] * voxel_size / float(Ts), 
                                        flow_VE_3D_ii_real[cell_ii] * voxel_size / float(Ts),
                                        flow_Epi_3D_ii_real[cell_ii] * voxel_size / float(Ts)])
                
                cell_table.append(cell_info)
                cell_row_counter_array_ii.append(cell_row_counter)
                cell_row_counter+=1 # increment global counter.
                         
            cell_row_ids.append(np.hstack(cell_row_counter_array_ii))
                
        cell_table = np.array(cell_table)
        print(cell_table.shape)
        
        # =============================================================================
        #      Initialise all the export csv table   
        # =============================================================================
        cell_columns = np.hstack(['row_id', 
                                  'embryo',
                                  'Stage', 
                                  'Frame', 
                                  'Static_Epi_Contour_Quad_ID', 
                                  'Cell_ID',
                                  'Area', 
                                  'Perimeter', 
                                  'Shape_Index',
                                  'Major_Length', 
                                  'Minor_Length', 
                                  'Eccentricity', 
                                  'Eccentricity_Angle_AVE',
                                  'Eccentricity_Angle_vect_x',
                                  'Eccentricity_Angle_vect_y',
                                  'Eccentricity_Angle_rel_vect_radial',
                                  'Eccentricity_Angle_rel_vect_theta',
                                  'pos_2D_x', 
                                  'pos_2D_y',
                                  'pos_2D_x_rot-AVE', 
                                  'pos_2D_y_rot-AVE',
                                  'match_epi_contour_x',
                                  'match_epi_contour_y',
                                  'match_epi_contour_distal_dist',
                                  'cell_distal_dist',
                                  'cell_epi_contour_dist_ratio',
                                  'demons_AVE_dir_VE',
                                  'demons_AVE-perp_dir_VE',
                                  'demons_Norm_dir_VE',
                                  'demons_AVE_dir_Epi',
                                  'demons_AVE-perp_dir_Epi',
                                  'demons_Norm_dir_Epi', 
                                  'curvature_VE', 
                                  'curvature_Epi',
                                  'flow_VE_3D_x_ref',
                                  'flow_VE_3D_y_ref',
                                  'flow_VE_3D_z_ref',
                                  'flow_Epi_3D_x_ref',
                                  'flow_Epi_3D_y_ref',
                                  'flow_Epi_3D_z_ref',
                                  'flow_VE_3D_x_real',
                                  'flow_VE_3D_y_real',
                                  'flow_VE_3D_z_real',
                                  'flow_Epi_3D_x_real',
                                  'flow_Epi_3D_y_real',
                                  'flow_Epi_3D_z_real'])
        cell_table = pd.DataFrame(cell_table, index=None, columns=cell_columns)
        
        # add placeholder columns to insert the track statistics into. 
        cell_table['Dynamic_Track_Quad_ID'] = np.nan # this will be the version that inherits that of the 
        cell_table['Change_Rate_Area_Frac'] = np.nan
        cell_table['Change_Rate_Perimeter_Frac'] = np.nan
        cell_table['Change_Rate_Shape_Index'] = np.nan
        cell_table['Change_Rate_Eccentricity'] = np.nan
        cell_table['Cell_Neighbours_Track_ID'] = np.nan # additional to track cell neighbour identities. 
        cell_table['Diff_Cell_Neighbours_Track_ID'] = np.nan
        cell_table['velocity_3D_x_ref'] = np.nan
        cell_table['velocity_3D_y_ref'] = np.nan
        cell_table['velocity_3D_z_ref'] = np.nan
        cell_table['velocity_3D_x_real'] = np.nan
        cell_table['velocity_3D_y_real'] = np.nan
        cell_table['velocity_3D_z_real'] = np.nan
        cell_table['speed_3D_VE_ref'] = np.nan
        cell_table['speed_3D_VE_real'] = np.nan
        cell_table['AVE_direction_VE_speed_3D_ref'] = np.nan
        cell_table['AVE_direction_VE_speed_3D_real'] = np.nan
        cell_table['AVE_cum_dist_coord_3D'] = np.nan # cumulative direction moved by the VE cell according to VE movement. 
        cell_table['AVE_direction_Epi_speed_3D_ref'] = np.nan
        cell_table['AVE_direction_Epi_speed_3D_real'] = np.nan
        
        # we should additionally include columns that explicitly highlight the mother-daughter relationships too !.         
        
        # =============================================================================
        #   Operate on the track data.       
        # =============================================================================
        
        # load the file giving the relationships between tracks. 
        
        # use the cell_time to mask the cell_track_ids. 
        cell_tracks_cell_ids = cell_tracks_cell_ids[:,cell_time] # find only that which is relevant.
        
        # match the cell tracks_cell_ids to the unique row ids in the cell statistics table so we can compute all relevant statistics. 
        cell_tracks_cell_row_ids = map_tra_cell_ids_to_row_ids(cell_tracks_cell_ids, 
                                                                cell_ids, 
                                                                cell_row_ids,
                                                                cell_time)
            
        # with the above lookup we should be able to load and map all stats to the correct static table.
        shapematfile = shapematfiles[ii]
        shapematobj = spio.loadmat(shapematfile)
        velocitymatfile = velocitymatfiles[ii]
        velocitymatobj = spio.loadmat(velocitymatfile)
        
        """
        alright now we update all the positions with cell track data. 
        """
        celltracks_yx_2D = shapematobj['cell_track_centroids_2D']
        
        # shape / position params. 
        celltracks_area = shapematobj['cell_track_areas']
        celltracks_perim = shapematobj['cell_track_perims']
        celltracks_shape_index = shapematobj['cell_track_shape_index']
        celltracks_eccentricity = shapematobj['cell_track_eccentricity'] # neighbors is not a very good metric.
        cell_track_track_neighbors = shapematobj['cell_track_track_neighbors'] # load this up in terms of the tracks that are neighbors. 
        celltracks_pos_3D_absolute = shapematobj['cell_track_centroids_3D'] 
        
        # we compute these in the various stages. 
        celltracks_pos_3D = velocitymatobj['cell_track_3D_xyz_ref'] # note: these are by stage!. 
        celltracks_AVE_dir = velocitymatobj['cell_track_AVE_dir_3D']
        vol_factor = velocitymatobj['vol_factor'].ravel()[0]
        # need to have temp_tform_scales
        temp_tform_scales = velocitymatobj['temp_tform_scales'].ravel() # get this for each timepoint to know the scaling.
        
        """
        parse the saved data for the real coordinates. 
        """
        celltracks_pos_3D_absolute_rel = celltrack3Dobj['cell_track_centroids_3D_real_rel']
        celltracks_AVE_dir_absolute_rel = celltrack3Dobj['AVE_surf_dir_ve_time_3D'] # available for all of time. 
        celltracks_AVE_dir_absolute_rel = celltracks_AVE_dir_absolute_rel.transpose(1,0,2) 
        
        """
        parsing the times to obtain cell track statistics over. 
        """
        # get the times for the stages
        analyse_stages = shapematobj['analyse_stages']; analyse_stages = np.hstack([s.strip() for s in analyse_stages])
        analyse_times = shapematobj['analyse_time_segs']
    
        if analyse_stages.shape[0] == 1:
            analyse_stages = analyse_stages[0]
        if analyse_times.shape[0] == 1:
            analyse_times = analyse_times[0]
        
        """
        assembly of statistics by iterating through all the tracks. 
        """
        mig_time_0_start = analyse_times[1].ravel()[0]
        
        
        """
        use the static quad and track lineage to assign all tracks a 'dynamic' label. 
        """
        # reparse the track lineage dict to flatten it ... and then save out.
        savefolder , basefname = os.path.split(shapematfile)
        lineage_savefile = os.path.join(savefolder, basefname.replace('cell_track_and_polargrid_stats_', 'single_cell_track_lineage_table_'))
        # basefname = basefname.replace('single_cell_track_stats_lookup_table_', 
        #                               'single_cell_track_lineage_table_')
        
        # lineage_savefile = os.path.join(savefolder, basefname).replace('.csv','.mat')
        lineage_mat = spio.loadmat(lineage_savefile)      
        
        track_ids = lineage_mat['tra_ids'].ravel()
        sort_track_ids = np.argsort(track_ids)
        track_ids = track_ids[sort_track_ids]
        track_ids_lineage = lineage_mat['tra_ids_lineage'][0]
        track_ids_lineage = track_ids_lineage[sort_track_ids]
        
        track_cells_quad_mig0 = cell_tracks_cell_row_ids[:,mig_time_0_start]
        track_ids_quad_mig = np.zeros(len(track_cells_quad_mig0), dtype=np.int)
        
        # propagate the initial ids starting from the tracks defined at the migration point. 
        track_ids_quad_mig[track_cells_quad_mig0!=-1] =  cell_table.iloc[track_cells_quad_mig0[track_cells_quad_mig0!=-1]]['Static_Epi_Contour_Quad_ID'].values
        
        mig0_tracks = track_ids[track_ids_quad_mig>0]
        nonmig0_tracks = np.setdiff1d(track_ids, mig0_tracks)
        
        # now iterate over the nonmig0_tracks and assign based on track lineage. 
        for nonmig0_tra_id in nonmig0_tracks[:]:
            
            lineage = track_ids_lineage[nonmig0_tra_id]
            # iterate over the lineage 
            lineage = lineage[0]
            
            for lin in lineage: # iterate and priorities the mother. 
                if lin in mig0_tracks:
                    track_ids_quad_mig[nonmig0_tra_id] = track_ids_quad_mig[lin]
                    # print(nonmig0_tra_id, lin)
                    break
        
        # the total amount even after this is not as much as the total number of tracks. 
        # print(len(track_ids[track_ids_quad_mig>0]))
        
        # use this to write to the respective cells in each track, the dynamic quad id. 
        for tra_ii in np.arange(len(track_ids)):
            tra_id = track_ids[tra_ii]
            dyn_quad_id = track_ids_quad_mig[tra_id]
            cell_tracks_cell_row_ids_tra_ii = cell_tracks_cell_row_ids[tra_id]
            select_tra_ii = cell_tracks_cell_row_ids_tra_ii>0
            select_cell_id_tra_ii = cell_tracks_cell_row_ids_tra_ii[select_tra_ii]
            print(select_cell_id_tra_ii)
            if len(select_cell_id_tra_ii) > 0:
                select_bool = np.zeros(len(cell_table), dtype=np.bool)
                for idd in select_cell_id_tra_ii:
                    select_bool[np.arange(len(cell_table))==idd] = 1 
                
                # use the logical indexers to write into the table. 
                cell_table.loc[select_bool,'Dynamic_Track_Quad_ID'] = dyn_quad_id # fill up with all of this id
        # cell_table['Dynamic_Track_Quad_ID'] = np.nan
        
        # outer loop over stage times. 
        for stage_ii in np.arange(len(analyse_stages))[:]:
            
            stage_name = analyse_stages[stage_ii]
            stage_times = analyse_times[stage_ii]
            
            if len(stage_times.shape) > 1: 
                stage_times = stage_times[0]
            
            stage_polar_grid = polar_grids_stage[stage_ii]
            celltracks_yx_2D_stage00 = celltracks_yx_2D[:,stage_times.ravel()[0]]
            celltracks_yx_2D_stage00_nonnan = ~np.isnan(celltracks_yx_2D_stage00[...,0])
            celltracks_quad_stage = np.zeros(len(celltracks_yx_2D_stage00), dtype=np.int)
            celltracks_quad_stage[celltracks_yx_2D_stage00_nonnan] = stage_polar_grid[celltracks_yx_2D_stage00[celltracks_yx_2D_stage00_nonnan][...,0].astype(np.int), 
                                                                                      celltracks_yx_2D_stage00[celltracks_yx_2D_stage00_nonnan][...,1].astype(np.int)]
            
            cell_tracks_cell_row_ids_stage = cell_tracks_cell_row_ids[:,stage_times] # this is required to know what to write to for each individual cell track. 
            
            # # computations below are only valid for the selected?
            # select_tracks = celltracks_quad_stage > 0 # they are non-zero. 
            
            # state characterisation. 
            
            # # # quad_id: 
            # cell_track_quad_id_stage_ii = cell_track_quad[select_tracks]
            
            # cell_track_coordinates: 
            cell_centroids_stage_yx_ii = celltracks_yx_2D[:,stage_times]
            
            # area:
            stage_areas = celltracks_area[:,stage_times] * (voxel_size**2)
            
            # perim: 
            stage_perims = celltracks_perim[:,stage_times] * (voxel_size)
            
            # shape index:
            stage_shape_index = celltracks_shape_index[:,stage_times] # dimensionless
            
            # eccentricity: 
            stage_eccentricity = celltracks_eccentricity[:,stage_times]
            
            # cell neighbors in terms of the track ids. 
            stage_cell_neighbors_tra_ids = cell_track_track_neighbors[:, stage_times] # need to write this ... into the cell table. 
            
            # =============================================================================
            #   Here are the derivative statistics we must get and nans should propagate.  
            # =============================================================================
            stage_diff_areas = (stage_areas[:,1:] - stage_areas[:,:-1]) / (stage_areas[:,:-1] + .1) / (Ts/60.)
            stage_diff_perims = (stage_perims[:,1:] - stage_perims[:,:-1]) / (stage_perims[:,:-1] + .1) / (Ts/60.)
            stage_diff_shape_index = (stage_shape_index[:,1:] - stage_shape_index[:,:-1]) / (Ts/60.) # converts to rate. 
            stage_diff_eccentricity = (stage_eccentricity[:,1:] - stage_eccentricity[:,:-1]) / (Ts/60.)
            
            # compute the diff in stage_cell_neighbors_tra_ids? # along the track over time? 
            diff_stage_cell_neighbors_tra_ids = compute_time_diff_neighbor_inds_tracks(stage_cell_neighbors_tra_ids) # gives the tra id that is different from timepoint to timepoint. 
            diff_stage_cell_neighbors_tra_ids = np.array(diff_stage_cell_neighbors_tra_ids) # to make things easier. 
            
            # =============================================================================
            #       Speed computations       
            # =============================================================================
            # AVE dir velocity: (question is do we compute for all the timepoints?)
            stage_tracks_3D = celltracks_pos_3D[:,stage_times]
            stage_disps_3D = stage_tracks_3D[:,1:] - stage_tracks_3D[:,:-1]
            stage_disps_3D = stage_disps_3D * temp_tform_scales[stage_times[0]+1:stage_times[-1]+1][None,:,None] # do not divide by volume factor here.
            stage_disps_3D = stage_disps_3D * (voxel_size) / float(Ts) # to get into physical units. 
            stage_AVE_3D_dir = celltracks_AVE_dir[:,stage_times]
            
            # get the ave direction speed. 
            stage_AVE_dir_speed_3D = np.sum(stage_disps_3D*stage_AVE_3D_dir[:,:-1], axis=-1)
            
            # get the total speed. 
            stage_total_speed_3D = np.linalg.norm(stage_disps_3D, axis=-1)
            
            # linear velocity speeds can't be computed... per say ..... need coordinates.... etc. # just give velocity directions? 
            
            # computation as above but not for real coordinates. 
            stage_tracks_3D_real = celltracks_pos_3D_absolute_rel[:,stage_times]
            stage_disps_3D_real = stage_tracks_3D_real[:,1:] - stage_tracks_3D_real[:,:-1]
            stage_disps_3D_real = stage_disps_3D_real * (voxel_size) / float(Ts) # to get into physical units. 
            stage_AVE_3D_dir_real = celltracks_AVE_dir_absolute_rel[:,stage_times]
            
            # get the ave direction speed. 
            stage_AVE_dir_speed_3D_real = np.sum(stage_disps_3D_real*stage_AVE_3D_dir_real[:,:-1], axis=-1)
            
            # get the total speed. 
            stage_total_speed_3D_real = np.linalg.norm(stage_AVE_3D_dir_real, axis=-1)
            
            """
            oh i see, here we have to compute the ave cumulative coordinates and what we need to do is use the full track length else we will have an indexing error. 
            """
            stage_tracks_3D_all = celltracks_pos_3D[:] 
            stage_disps_3D_all = stage_tracks_3D_all[:,1:] - stage_tracks_3D_all[:,:-1]
            # stage_disps_3D = stage_disps_3D / float(vol_factor) * temp_tform_scales[stage_times[0]+1:stage_times[-1]+1][None,:,None]
            stage_disps_3D_all = stage_disps_3D_all * temp_tform_scales[1:stage_tracks_3D_all.shape[1]][None,:,None]
            stage_AVE_3D_dir_all = celltracks_AVE_dir[:].copy()
            
            stage_AVE_speeds_all = np.sum(stage_disps_3D_all*stage_AVE_3D_dir_all[:,:-1], axis=-1) # this preserves nan propagation. 
            # confirmed this has nan propagation. 
            stage_AVE_speeds_all = stage_AVE_speeds_all * (voxel_size) / float(Ts) # as this is just a differential of time. 
            
            # transform the stage_ave_speeds here to instead an AVE coordinate relative to the AVE migration TP.
            stage_AVE_speeds_coords = np.nancumsum(stage_AVE_speeds_all, axis=1)
            stage_AVE_speeds_coords = stage_AVE_speeds_coords - stage_AVE_speeds_coords[:,mig_time_0_start-1][:,None]
            stage_AVE_speeds_coords[np.isnan(stage_AVE_speeds_all)] = np.nan # put back the nan.
            
            
            # plt.figure()
            # plt.plot(stage_AVE_speeds_coords.T)
            # plt.show()
            
            # now apply the masking of time for this stage. 
            stage_AVE_speeds_coords = stage_AVE_speeds_coords[:, stage_times] # only to penultimate as it is a displacement. 
            
            print(stage_AVE_speeds_coords.shape)
            print('====')
            
            """
            repeating the lengthy computation above now for real positions. 
            """
            stage_tracks_3D_all_real = celltracks_pos_3D_absolute_rel[:,:celltracks_AVE_dir_absolute_rel.shape[1]+1].copy()
            stage_disps_3D_all_real = stage_tracks_3D_all_real[:,1:] - stage_tracks_3D_all_real[:,:-1]
            stage_AVE_3D_dir_all_real = celltracks_AVE_dir_absolute_rel[:].copy()
            # stage_disps_3D_all_real = stage_disps_3D_all_real[:,:stage_AVE_3D_dir_all_real.shape[1]-1]
            
            stage_AVE_speeds_all_real = np.sum(stage_disps_3D_all_real*stage_AVE_3D_dir_all_real[:,:], axis=-1) # this preserves nan propagation. 
            # confirmed this has nan propagation. 
            stage_AVE_speeds_all_real = stage_AVE_speeds_all_real * (voxel_size) / float(Ts) # as this is just a differential of time. 
            
            # transform the stage_ave_speeds here to instead an AVE coordinate relative to the AVE migration TP.
            stage_AVE_speeds_coords_real = np.nancumsum(stage_AVE_speeds_all_real, axis=1)
            stage_AVE_speeds_coords_real = stage_AVE_speeds_coords_real - stage_AVE_speeds_coords_real[:,mig_time_0_start-1][:,None]
            stage_AVE_speeds_coords_real[np.isnan(stage_AVE_speeds_all_real)] = np.nan # put back the nan.
            
            # now apply the masking of time for this stage. 
            stage_AVE_speeds_coords_real = stage_AVE_speeds_coords_real[:, stage_times] # only to penultimate as it is a displacement. 
            
            # plt.figure()
            # plt.plot(stage_AVE_speeds_coords_real.T)
            # plt.show()
            # =============================================================================
            # =============================================================================
            # =============================================================================
            # # #   Go through for each track and now write all the computed statistics into place. 
            # =============================================================================
            # =============================================================================
            # =============================================================================
            n_tra = len(cell_tracks_cell_row_ids_stage)
            
            for tra_ii in np.arange(n_tra)[:]:
                
                # go through each track, 
                # look up the valid index. and put the values in the respective columns. 
                cell_tra_ii_cell_row_ids_stage = cell_tracks_cell_row_ids_stage[tra_ii]
                
                # read this. 
                cell_tra_table_index, cell_tra_table_stats = read_cell_table_track(cell_tra_ii_cell_row_ids_stage, 
                                                                                    cell_table)
                
                # if there are any valid entries then we write this. 
                if len(cell_tra_table_index)>0:
                    
                    # construct the index lookup. 
                    bool_index = np.zeros(len(cell_table), dtype=np.bool); 
                    bool_index[cell_tra_ii_cell_row_ids_stage[cell_tra_table_index][:-1]] = True
                    
                    cell_table.loc[bool_index, 'Change_Rate_Area_Frac'] = stage_diff_areas[tra_ii][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'Change_Rate_Perimeter_Frac'] = stage_diff_perims[tra_ii][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'Change_Rate_Shape_Index'] = stage_diff_shape_index[tra_ii][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'Change_Rate_Eccentricity'] = stage_diff_eccentricity[tra_ii][cell_tra_table_index[:-1]]
                    
                    cell_table.loc[bool_index, 'velocity_3D_x_ref'] = stage_disps_3D[tra_ii][...,0][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'velocity_3D_y_ref'] = stage_disps_3D[tra_ii][...,1][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'velocity_3D_z_ref'] = stage_disps_3D[tra_ii][...,2][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'speed_3D_VE_ref'] = stage_total_speed_3D[tra_ii][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'AVE_direction_speed_3D_ref'] = stage_AVE_dir_speed_3D[tra_ii][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'AVE_cum_dist_coord_3D_ref'] = stage_AVE_speeds_coords[tra_ii][cell_tra_table_index[:-1]]
        
                    cell_table.loc[bool_index, 'velocity_3D_x_real'] = stage_disps_3D_real[tra_ii][...,0][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'velocity_3D_y_real'] = stage_disps_3D_real[tra_ii][...,1][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'velocity_3D_z_real'] = stage_disps_3D_real[tra_ii][...,2][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'speed_3D_VE_real'] = stage_total_speed_3D_real[tra_ii][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'AVE_direction_speed_3D_real'] = stage_AVE_dir_speed_3D_real[tra_ii][cell_tra_table_index[:-1]]
                    cell_table.loc[bool_index, 'AVE_cum_dist_coord_3D_real'] = stage_AVE_speeds_coords_real[tra_ii][cell_tra_table_index[:-1]]
        

                    # for the Epi speed displacements we need to reconstruct the 3D vector and project onto the direction given by the cell trajectory!. 
                    tra_cell_AVE_directions_3D = stage_AVE_3D_dir[tra_ii][cell_tra_table_index[:-1]]
                    tra_cell_epi_3D_x = cell_table.loc[bool_index, 'flow_Epi_3D_x_ref'].values.astype(np.float)
                    tra_cell_epi_3D_y = cell_table.loc[bool_index, 'flow_Epi_3D_y_ref'].values.astype(np.float)
                    tra_cell_epi_3D_z = cell_table.loc[bool_index, 'flow_Epi_3D_z_ref'].values.astype(np.float)
                    
                    # definining another one for cumulative ave direction might be overkill here. 
                    if len(tra_cell_epi_3D_x) > 0: 
                        tra_cell_epi_3D_xyz = np.vstack([tra_cell_epi_3D_x, 
                                                          tra_cell_epi_3D_y, 
                                                          tra_cell_epi_3D_z]).T
                        tra_cell_epi_speed_3D_AVE = np.sum(tra_cell_AVE_directions_3D * tra_cell_epi_3D_xyz, axis=-1)
                        cell_table.loc[bool_index, 'AVE_direction_Epi_speed_3D_ref'] = tra_cell_epi_speed_3D_AVE
                        
                    # this one is for the real distances. 
                    # for the Epi speed displacements we need to reconstruct the 3D vector and project onto the direction given by the cell trajectory!. 
                    tra_cell_AVE_directions_3D = stage_AVE_3D_dir_real[tra_ii][cell_tra_table_index[:-1]] # would need to be the real version!.
                    tra_cell_epi_3D_x = cell_table.loc[bool_index, 'flow_Epi_3D_x_real'].values.astype(np.float)
                    tra_cell_epi_3D_y = cell_table.loc[bool_index, 'flow_Epi_3D_y_real'].values.astype(np.float)
                    tra_cell_epi_3D_z = cell_table.loc[bool_index, 'flow_Epi_3D_z_real'].values.astype(np.float)
                    
                    # definining another one for cumulative ave direction might be overkill here. 
                    if len(tra_cell_epi_3D_x) > 0: 
                        tra_cell_epi_3D_xyz = np.vstack([tra_cell_epi_3D_x, 
                                                          tra_cell_epi_3D_y, 
                                                          tra_cell_epi_3D_z]).T
                        tra_cell_epi_speed_3D_AVE = np.sum(tra_cell_AVE_directions_3D * tra_cell_epi_3D_xyz, axis=-1)
                        cell_table.loc[bool_index, 'AVE_direction_Epi_speed_3D_real'] = tra_cell_epi_speed_3D_AVE
                        
                    
                    
                    # print(tra_cell_AVE_directions_3D.shape)
                    # print(tra_cell_epi_3D_x.shape)
                    # print('===')
                    
                    # compute and then save in the table. 
                    # get the ave direction speed. 
                    # stage_AVE_dir_speed_3D = np.sum(stage_disps_3D*stage_AVE_3D_dir[:,:-1], axis=-1)
        
                    # for neighbors we need to write in entry by entry
                    for ind_ii in np.arange(len(cell_tra_table_index)):
                        
                        #create a bool index just for it 
                        row_ind_ii = cell_tra_table_index[ind_ii]
                        row_ind_ii_bool = np.zeros(len(cell_table), dtype=np.bool); row_ind_ii_bool[cell_tra_ii_cell_row_ids_stage[row_ind_ii]] = True
                        
                        cell_neighbor_id = stage_cell_neighbors_tra_ids[tra_ii, row_ind_ii]
                        if len(cell_neighbor_id.shape)>1 and cell_neighbor_id.shape[0]>0:
                            cell_neighbor_id = cell_neighbor_id[0]
                        
                        if len(cell_neighbor_id) > 0: 
                        # diff_stage_cell_neighbors_tra_ids
                            cell_table.loc[row_ind_ii_bool, 'Cell_Neighbours_Track_ID'] = ':'.join([str(nn) for nn in cell_neighbor_id]) 
                        
                    # if this is not the final index then we should additionally provide the difference in neighbors. 
                    for ind_ii in np.arange(len(cell_tra_table_index)-1):
                        
                        #create a bool index just for it 
                        row_ind_ii = cell_tra_table_index[ind_ii]
                        row_ind_ii_bool = np.zeros(len(cell_table), dtype=np.bool); row_ind_ii_bool[cell_tra_ii_cell_row_ids_stage[row_ind_ii]] = True
                        
                        cell_neighbor_id = diff_stage_cell_neighbors_tra_ids[tra_ii, row_ind_ii]
                        
                        if len(cell_neighbor_id.shape)>1:
                            cell_neighbor_id = cell_neighbor_id[0]
                            
                        if len(cell_neighbor_id) > 0: 
                        # diff_stage_cell_neighbors_tra_ids
                            cell_table.loc[row_ind_ii_bool, 'Diff_Cell_Neighbours_Track_ID'] = ':'.join([str(nn) for nn in cell_neighbor_id]) 
                        
        """
        3. save the final cell table, along with the cell row indexes for future lookup and computation. 
        """
        # cell_table = [] # this is the flattened ver. 
        # cell_row_ids = [] # for fast lookup when integrating with track data. ? 
        
        savecsvcellstatsfile = os.path.join(os.path.split(cellmatfiles[ii])[0], 
                                            os.path.split(cellmatfiles[ii])[1].replace('cell_stats_', 'single_cell_statistics_table_export-ecc-realxyz_').replace('.mat', '.csv'))
        cell_table.to_csv(savecsvcellstatsfile, index=None)
    
        savecsvtrackindexfile = os.path.join(os.path.split(cellmatfiles[ii])[0], 
                                              os.path.split(cellmatfiles[ii])[1].replace('cell_stats_', 'single_cell_track_stats_lookup_table_').replace('.mat', '.csv'))
    
        cell_tracks_cell_ids_table = pd.DataFrame(np.hstack([np.arange(len(cell_tracks_cell_ids))[:,None], 
                                                              cell_tracks_cell_row_ids]), 
                                                  index=None,
                                                  columns = np.hstack(['Track_ID', np.hstack(['TP_%d'%(t_ii+1) for t_ii in range(cell_tracks_cell_ids.shape[1])])]))
        cell_tracks_cell_ids_table.to_csv(savecsvtrackindexfile, index=None)
    
        savematcellindexfile = os.path.join(os.path.split(cellmatfiles[ii])[0], 
                                            os.path.split(cellmatfiles[ii])[1].replace('cell_stats_', 'single_cell_rowindex_').replace('.csv', '.mat'))

        # spio.savemat(savematcellindexfile, 
        #               {'embryo':embryo, 
        #               'cellfile': cellmatfile,
        #               'celltrackfile': celltrackfile,
        #               'cell_row_index': cell_row_ids})
        
        
        