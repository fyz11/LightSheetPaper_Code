# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:13:16 2019

@author: Felix
"""

def map_intensity_interp3(query_pts, grid_shape, I_ref):
    
    # interpolate instead of discretising to avoid artifact.
    from scipy.interpolate import RegularGridInterpolator
    
    #ZZ,XX,YY = np.indices(im_array.shape)
    spl_3 = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                     np.arange(grid_shape[1]), 
                                     np.arange(grid_shape[2])), 
                                     I_ref, method='linear', bounds_error=False, fill_value=0)
    
    I_query = np.uint8(spl_3((query_pts[...,0], 
                              query_pts[...,1],
                              query_pts[...,2])))
    
    return I_query
    

def parse_conditions_fnames(vidfile):
    
    import os 
    import re 
    
    fname = os.path.split(vidfile)[-1]
    
    """ get the tissue information """
    tissue = []
    if 'mtmg' in fname or 'epi' in fname:
        tissue = 'epi'
    if 'ttr' in fname or 've' in fname:
        tissue = 've'
        
    """ check if distal projection """
    proj = 'cylindrical'
    
    if 'distal' in fname:
        proj = 'distal'
        
    """ get the embryo number """
    emb_no = re.findall('L\d+', fname)

    if emb_no:
        emb_no = emb_no[0].split('L')[1]
    else:
        emb_no = np.nan
        
    """ get the angle of the unwrapping """
    ang_cand = re.findall(r"\d+", fname)

    ang = '000'
        
    for a in ang_cand:
        if len(a) == 3 and a!=emb_no:
            ang = a
            
    """ get the timepoint of the reference """       
    tp_val = fname.split('_tp')[1].split('.')[0]            
            
    """ check to see if angle is transposed """
    tp = 'No'
    tp_cand = re.findall(r"\d+tp", fname)      
    if len(tp_cand) > 0:
        # do we also need to check for lenght of the digits in front?
        tp = 'Yes'
    
    return tissue, proj, emb_no, ang, tp_val, tp
    
def fetch_assoc_unwrap_params(vidfile, folder, ext='.mat'):
    
    import glob
    import os 
    
    files = np.hstack(glob.glob(os.path.join(folder, '*'+ext)))

    v_condition = '-'.join(parse_conditions_fnames(vidfile))
    unwrap_conditions = np.hstack(['-'.join(parse_conditions_fnames(f)) for f in files])
    
    return files[unwrap_conditions == v_condition]

def find_time_string(name):
    
    import re
    
    time_string = re.findall(r't\d+', name)
    time_string = time_string[0]
#    if len(time_string):
    time = int(time_string.split('t')[1])
        
    return time


def find_time_string_MTMG(name):
    
    print(name)
    if '_' in name:
        cands = name.split('_')
    elif '-' in name:
        cands = name.split('-')
    else:
        print('error in naming')

    time = []
    
    for c in cands:
        
        try:
#            if len(c) == 2:
            c = int(c)
            time = c
        except:
            continue
        
    return time
    

def find_unwrap_params_folder(rootfolder, key='unwrap_params', exclude='geodesic'):
    
    dir_found = []
    for root, dirs, files in os.walk(rootfolder, topdown=False):
        for d in dirs:
            if key in d and exclude not in d:
                dir_found.append(os.path.join(root,d))
                
    return dir_found


def get_rotation_y(theta):
    
    R_z = np.zeros((4,4))
    R_z[-1:] = np.array([0,0,0,1])
    
    R_z[:-1,:-1] = np.array([[np.cos(theta), 0, np.sin(theta)],
                             [0, 1, 0], 
                             [-np.sin(theta), 0, np.cos(theta)]])
    
    return R_z


def preprocess_unwrap_pts(pts3D, im_array, rot_angle_3d=None, transpose_axis=None, clip_pts_to_array=True):
    
    import Registration.registration_new as reg
    import Geometry.geometry as geom
    
    pts = pts3D.copy()
    
    if transpose_axis is not None:
        pts = pts[...,transpose_axis]
        
    if rot_angle_3d is not None:
        
        rot_matrix = np.eye(3)
        if rot_angle_3d != 0:
            rot_matrix = geom.get_rotation_y(-rot_angle_3d/180.*np.pi)[:3,:3]
            
        pts = reg.warp_3D_transforms_xyz_similarity_pts(pts, 
                                                        translation=[0,0,0], 
                                                        rotation=rot_matrix, 
                                                        zoom=[1,1,1], 
                                                        shear=[0,0,0], 
                                                        center_tform=True, 
                                                        im_center = np.hstack(im_array.shape)/2.,
#                                                        im_center = pts.reshape(-1,pts.shape[-1]).mean(axis=0),
                                                        direction='B') # for pts is reverse of image. 

    if clip_pts_to_array==True:
        # clip the values to the array.
        pts[...,0] = np.clip(pts[...,0], 0, im_array.shape[0]-1)
        pts[...,1] = np.clip(pts[...,1], 0, im_array.shape[1]-1)
        pts[...,2] = np.clip(pts[...,2], 0, im_array.shape[2]-1)
    
    return pts

import numpy as np 
def warp_3D_transforms_xyz_similarity_pts(pts3D, translation=[0,0,0], 
                                          rotation=np.eye(3), 
                                          zoom=[1,1,1], shear=[0,0,0], 
                                          center_tform=True, 
                                          im_center=None, 
                                          direction='F'):

    from transforms3d.affines import compose 
    
    # compose the 4 x 4 homogeneous matrix. 
    tmatrix = compose(translation, rotation, zoom, shear)

    if center_tform:
        # im_center = np.array(im.shape)//2
        tmatrix[:-1,-1] = tmatrix[:-1,-1] + np.array(im_center)
        decenter = np.eye(4); decenter[:-1,-1] = -np.array(im_center)
        tmatrix = tmatrix.dot(decenter)

#    if direction == 'F':
#        print(tmatrix)
#    if direction == 'B':
#        print(np.linalg.inv(tmatrix))
        
    # first make homogeneous coordinates.
    xyz = np.vstack([(pts3D[...,0]).ravel().astype(np.float32), 
                      (pts3D[...,1]).ravel().astype(np.float32), 
                      (pts3D[...,2]).ravel().astype(np.float32),
                      np.ones(len(pts3D[...,0].ravel()), dtype=np.float32)])
    
    if direction == 'F':
        xyz_ = tmatrix.dot(xyz)
    if direction == 'B':
        xyz_ = (np.linalg.inv(tmatrix)).dot(xyz)
    
    pts3D_warp = (xyz_[:3].T).reshape(pts3D.shape)

    return pts3D_warp

#def get_rotation_y(theta):
#    
#    R_z = np.zeros((4,4))
#    R_z[-1:] = np.array([0,0,0,1])
#    
#    R_z[:-1,:-1] = np.array([[np.cos(theta), 0, np.sin(theta)],
#                             [0, 1, 0], 
#                             [-np.sin(theta), 0, np.cos(theta)]])
#    
#    return R_z


def preprocess_unwrap_pts_(pts3D, ref_pt, rot_angle_3d=None, transpose_axis=None):
    
    # import Registration.registration_new as reg
    # import Geometry.geometry as geom
    
    pts = pts3D.copy()
    
    if transpose_axis is not None:
        pts = pts[...,transpose_axis]
        ref_pt = ref_pt[transpose_axis]
        
    if rot_angle_3d is not None:
        
        rot_matrix = np.eye(3)
        if rot_angle_3d != 0:
            rot_matrix = get_rotation_y(-rot_angle_3d/180.*np.pi)[:3,:3]
            
        pts = warp_3D_transforms_xyz_similarity_pts(pts, 
                                                    translation=[0,0,0], 
                                                    rotation=rot_matrix, 
                                                    zoom=[1,1,1], 
                                                    shear=[0,0,0], 
                                                    center_tform=True, 
                                                    im_center = ref_pt,
#                                                        im_center = pts.reshape(-1,pts.shape[-1]).mean(axis=0),
                                                    direction='B') # for pts is reverse of image. 
    # # clip the values to the array.
    # pts[...,0] = np.clip(pts[...,0], 0, im_array.shape[0]-1)
    # pts[...,1] = np.clip(pts[...,1], 0, im_array.shape[1]-1)
    # pts[...,2] = np.clip(pts[...,2], 0, im_array.shape[2]-1)
    
    return pts

if __name__=="__main__":
    
    
    import os
    import sys
    
    import Utility_Functions.file_io as fio
    import Unzipping.unzip_new as uzip # unzip new is the latest!. 
    import numpy as np
    import pylab as plt 
    from skimage.exposure import rescale_intensity, equalize_adapthist
    import Geometry.transforms as tf
    
    from skimage.io import imsave
    from tqdm import tqdm 
    import Geometry.meshtools as meshtools
    import skimage.measure as measure 
    
    import scipy.io as spio
    from mpl_toolkits.mplot3d import Axes3D
    import pylab as plt
    
    from mayavi import mlab 
    from scipy.interpolate import RegularGridInterpolator
    from skimage.filters import gaussian
    
    import glob
    import scipy.io as spio 
    import Geometry.geometry as geom
    
    from mpl_toolkits.mplot3d import Axes3D
    from Geometry.transforms import apply_affine_tform
    import Registration.registration_new as reg
    
    """
    Detect and Read in Dataset. 
    """

#    surface_to_unwrap = 've'
#    surface_to_unwrap = 'epi'
    surface_to_unwrap = 'hex'
    
    
#    all_embryo_folder = '/media/felix/Srinivas4/LifeAct'
#    all_embryo_folder = '/media/felix/Srinivas4/MTMG-TTR'
#    all_embryo_folder = '/media/felix/Srinivas4/MTMG-HEX'
    all_embryo_folder = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
    
    embryo_folders = np.hstack([os.path.join(all_embryo_folder, name) for name in os.listdir(all_embryo_folder) if os.path.isdir(os.path.join(all_embryo_folder, name))])
    embryo_folders = np.sort(embryo_folders)
    
#    embryo_folders = np.hstack([e for e in embryo_folders if 'new_hex' not in e])
#    embryo_folders = np.hstack([e for e in embryo_folders if 'new_hex' in e])
    
    
#    for embryo_folder in embryo_folders[-2:-1]:
    for embryo_folder in embryo_folders[1:2]:

        print(embryo_folder)
        print('---')
        # detect step images folder.
        embryo_im_folder = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if 'step6' in name and os.path.isdir(os.path.join(embryo_folder, name))])
        
        if len(embryo_im_folder)>0:
            
            if surface_to_unwrap == 've':
#                embryo_im_folder = [folder for folder in embryo_im_folder if 'ttr' in os.path.split(folder)[-1]][0]
                embryo_im_folder = [folder for folder in embryo_im_folder if 'mtmg' in os.path.split(folder)[-1]][0]
#                embryo_im_folder = [folder for folder in embryo_im_folder if 'mtmg' in os.path.split(folder)[-1] and '-2' not in os.path.split(folder)[-1]][0]
            if surface_to_unwrap == 'epi':
#                embryo_im_folder = [folder for folder in embryo_im_folder if 'mtmg' in os.path.split(folder)[-1]][0]
                embryo_im_folder = [folder for folder in embryo_im_folder if 'mtmg' in os.path.split(folder)[-1]][0]
#                embryo_im_folder = [folder for folder in embryo_im_folder if 'mtmg' in os.path.split(folder)[-1] and '-2' not in os.path.split(folder)[-1]][0]
            if surface_to_unwrap == 'hex':
#                embryo_im_folder = [folder for folder in embryo_im_folder if 'hex' in os.path.split(folder)[-1]][0]
                embryo_im_folder = [folder for folder in embryo_im_folder if 'hex' in os.path.split(folder)[-1]][0]
#                embryo_im_folder = [folder for folder in embryo_im_folder if 'hex' in os.path.split(folder)[-1] and '-2' not in os.path.split(folder)[-1]][0]
            
            print('processing: ', embryo_im_folder)

            # find the video files. 
            embryo_im_files = np.hstack(glob.glob(os.path.join(embryo_im_folder, '*.tif'))); 
            # get the times.
#            frame_nos = np.hstack([find_time_string(os.path.split(f)[-1].split('.tif')[0]) for f in embryo_im_files])
            frame_nos = np.hstack([find_time_string_MTMG(os.path.split(f)[-1].split('.tif')[0]) for f in embryo_im_files])
            embryo_im_files = embryo_im_files[np.argsort(frame_nos)]


            print(embryo_im_files)
            print('===================')

            """
            Load in the unwrap params for the particular dataset
            """
            print('finding unwrapping parameter folder')
            # auto detect and fetch the unwrap_params folder. 
            unwrap_params_folder_path = find_unwrap_params_folder(embryo_folder, key='unwrap_params')#[0]
            
            # =============================================================================
            #       Only for L930      
            # =============================================================================
#            if surface_to_unwrap == 'hex':
##                unwrap_params_folder_path = np.hstack([folder for folder in unwrap_params_folder_path if 've' in folder])
#                unwrap_params_folder_path = np.hstack([folder for folder in unwrap_params_folder_path if 'tr' in folder])
#            else:
#                unwrap_params_folder_path = np.hstack([folder for folder in unwrap_params_folder_path if surface_to_unwrap in folder])
           
            unwrap_params_folder_path = unwrap_params_folder_path[0]
            print(unwrap_params_folder_path)
            
            
            all_unwrap_params_files = glob.glob(os.path.join(unwrap_params_folder_path, '*.mat'))    
            
            # uncomment for L930
#            all_unwrap_params_files = np.hstack([fil for fil in all_unwrap_params_files if '_rr10_' in fil])
            
            # parse the meta details from the param files. for automation. 
            metadata_files = np.vstack([parse_conditions_fnames(f) for f in all_unwrap_params_files])

            """
            Iterate over the unwrap files. 
            """
            
            for file_i in range(len(all_unwrap_params_files))[:3]:
                
                # dont take the parameters for distal projections. 
                type_projection = metadata_files[file_i][1]
                type_surface = metadata_files[file_i][0] 
                
                
                if surface_to_unwrap == 'hex':
                    surface_to_unwrap_ = 've'
                else:
                    surface_to_unwrap_ = surface_to_unwrap
                
                if type_projection == 'cylindrical' and type_surface==surface_to_unwrap_:# and 'rr10-2_' in os.path.split(all_unwrap_params_files[file_i])[-1]:
        #            unwrap_params_file = '/media/felix/Srinivas4/LifeAct/L455_Emb3_TP1-111/unwrapped/unwrap_params/L455_Emb3_epi_000_unwrap_params_tp55.mat'
                    unwrap_params_file = all_unwrap_params_files[file_i]
                    
                    # specify the angle of rotation and the timepoint. 
                    rot_angle = int(metadata_files[file_i][-3]) 
                    ref_file_id = int(metadata_files[file_i][-2]) - 1
                    
                    print(unwrap_params_file)
                    print('Reference timepoint: ', ref_file_id)
                    print('Rotation angle: ', rot_angle)
                    print('---------------')
                         
                    """
                    Specify the final output folder 
                    """
#                    out_unwrapped_folder = os.path.join(embryo_folder, 'unwrapped/geodesic-rotmatrix'); fio.mkdir(out_unwrapped_folder)
#                    out_unwrapped_params_folder = os.path.join(embryo_folder, 'unwrapped/unwrap_params_geodesic-rotmatrix'); fio.mkdir(out_unwrapped_params_folder)
#                        
                    """
                    Specify the specific file and the timepoint to learn the reference mapping for unwrapping
                    """
                    embryo_im_file = embryo_im_files[ref_file_id]
                    im_array = fio.read_multiimg_PIL(embryo_im_file)
                    
                    plt.figure()
                    plt.imshow(im_array.transpose(1,0,2).mean(axis=0))
                    plt.show()                    
                    
                    # unless is angle 0 otherwise use Holly's rotation. 
                    if rot_angle != 0:
                        # get the corresponding rotation file:
                        rottformfile = os.path.join(embryo_folder, 'step6_%s_rotation_tforms.mat' %(str(rot_angle).zfill(3)))
                        rottform = spio.loadmat(rottformfile)['tforms'][0]
                        im_array = apply_affine_tform(im_array, rottform)
                    
                    
                    plt.figure()
                    plt.title('first rotation')
                    plt.imshow(im_array.transpose(1,0,2).mean(axis=0))
                    plt.show()  
                    
                    
#                    extra_rot_angle = 0
#                    extra_rot_angle =  281.524119597077 #from the file.
#                    extra_rot_angle = 45 # this is the extra rotation angle. # easier to distinguish
                    extra_rot_angle = -98.2340664943
                    
                    rottform = geom.get_rotation_y(extra_rot_angle/180.*np.pi)
                    im_array = reg.warp_3D_transforms_xyz_similarity(im_array, 
                                                                        rotation = rottform[:3,:3],
                                                                        zoom = np.hstack([1,1,1]),
                                                                        center = None, 
                                                                        center_tform=True,
                                                                        direction='F',
                                                                        pad=None)
                    
                    plt.figure()
                    plt.title('second rotation')
                    plt.imshow(im_array.transpose(1,0,2).mean(axis=0))
                    plt.show()  
                    
                    
                    im_array = im_array.transpose(1,0,2) # transpose to long axes 

                    plt.figure()
                    plt.imshow(im_array.mean(axis=0))
                    plt.show()
                    
                    
                    
                    """
                    Read in the volume segmentation / binaries to et a set of coordinates. 
                    
                    read in the unwrapping metrics. 
                    """
                    unwrap_params = spio.loadmat(unwrap_params_file)
                    
                    # get the coordinates describing the outer surface. 
                    ref_coords =  unwrap_params['ref_coord_set']
                    polar_scale_factor = 1.1
                    sigma_rect_xyz = 1
                    sigma_polar_xyz = 3
                    
                    print(np.mean(ref_coords, axis=0))
                    print(np.hstack(im_array.shape)/2.)
                    print('===')
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    
                    n = 100
                    ax.scatter(ref_coords[::n,0], 
                               ref_coords[::n,1],
                               ref_coords[::n,2])
#                    plt.show()
                    
                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')
                    
                    
##                    pts3D, ref_pt, rot_angle_3d=None, transpose_axis=None
#                    # we can rotate these coordinates .... and match this to the volume. 
#                    ref_coords_ = preprocess_unwrap_pts_(ref_coords[:,:3], 
#                                                       ref_pt=np.mean(ref_coords[:,:3], axis=0), 
#                                                       rot_angle_3d=-extra_rot_angle, 
#                                                       transpose_axis=[2,0,1])[...,[1,2,0]]
                    ref_coords_ = preprocess_unwrap_pts_(ref_coords[:,:3], 
#                                                       ref_pt=np.mean(ref_coords[:,:3], axis=0), 
                                                       ref_pt = np.hstack(im_array.shape)/2.,
                                                       rot_angle_3d=-extra_rot_angle, 
                                                       transpose_axis=[1,0,2])[...,[1,0,2]]
                    
#                    ref_coords_ = ref_coords[:,:3]
#                    ref_coords_2 = preprocess_unwrap_pts_(ref_coords[:,:3], 
#                                                       ref_pt=np.mean(ref_coords[:,:3], axis=0), 
#                                                       rot_angle_3d=-45, 
#                                                       transpose_axis=[2,0,1])[...,[1,2,0]]
##                    ref_coords_ = preprocess_unwrap_pts(ref_coords[:,:3], 
##                                                       im_array, 
##                                                       rot_angle_3d=-extra_rot_angle, 
##                                                       transpose_axis=[1,0,2])[...,[1,0,2]]
#                    
                    ax.scatter(ref_coords_[::n,0], 
                               ref_coords_[::n,1],
                               ref_coords_[::n,2])
#                    ax.scatter(ref_coords_2[::n,0], 
#                               ref_coords_2[::n,1],
#                               ref_coords_2[::n,2])
                    
                    plt.show()

                    select_z = np.logical_and(ref_coords_[...,2]>=im_array.shape[2]//2-5, 
                                              ref_coords_[...,2]<=im_array.shape[2]//2+5)
                    
                    plt.figure()
                    plt.imshow(im_array[...,im_array.shape[2]//2])
                    plt.plot(ref_coords_[select_z,...,1], 
                             ref_coords_[select_z,...,0], 'r.')
                    plt.show()


                    # =============================================================================
                    #     Compute the filtered geodesic coordinates from the given segmentation
                    # =============================================================================
                    unwrap_params = uzip.compute_geodesic_statistics(ref_coords_, 
                                                                     filt_pts=True, 
                                                                     frac_coords=0, 
                                                                     ref_point=None, 
                                                                     pole='S', 
                                                                     n_angles=480,
                                                                     smooth_dist=2000)
                    
                    geodesic_pts = unwrap_params['coords']
                    
                    s_resolution = unwrap_params['s_resolution']
                    c_resolution = unwrap_params['c_resolution']
                #    min_s = unwrap_params['ranges'][1]
                    min_s = 0
                    max_s = unwrap_params['ranges'][3]
                    s_resolution = max_s - min_s + 1 # recompute. as we have forced min_s to be 0. 
#                    
                    # =============================================================================
                    #     1. Do the cylindrical geodesic unwrapping
                    # =============================================================================
                    # build the mapping space for cylindrical projection. 
                    """
                    Construct reference space and pull back to determine the (i,j) <-> (x,y,z) mapping  for cylindrical  
                    """
                    # build the mapping space. 
                    mapping_space_rect = uzip.build_mapping_space(geodesic_pts[:,:3], 
                                                                   ranges=[-np.pi, min_s, np.pi, max_s], 
                                                                   shape=[c_resolution, s_resolution], 
                                                                   polar_map=False)
                
                    mapped_coords_rect = uzip.match_coords_to_ref_space(geodesic_pts, 
                                                               ref_x=mapping_space_rect[0,:,0].ravel(), 
                                                               ref_y=mapping_space_rect[:,0,1].ravel(), map_index=[-1,-2])
                    
                #    match_coords_to_ref_space_general(query_coords, ref_coords, map_index=[-2,-1], rescale=True)
                    
                    ref_coord_set_rect = np.hstack([geodesic_pts, mapped_coords_rect])
                
                    # rbf does a very poor job preserving the distal mapping. 
                    ref_map_xyz_rect = uzip.gen_ref_map(im_array, 
                                                        ref_coord_set_rect, 
                                                        mapping_space_rect, interp_type='barycentric', interp_method='linear', 
                                                        rbf_samples=1000, 
                                                        rbf_basis='multiquadric', 
                                                        rbf_epsilon=10, 
                                                        rbf_smooth=0.01,
                                                        smooth_check_thresh=5, 
                                                        smooth_check_axis =1,
                                                        smooth_cutoff_thresh=0.5) # how much to smooth? (too much smoothing distorts the distal tip!)
                
                
                    ref_map_xyz_rect_filt = np.dstack([gaussian(ref_map_xyz_rect[...,i_], sigma=sigma_rect_xyz, mode='nearest', preserve_range=True) for i_ in range(3)])
                    # =============================================================================
                    #     2. Do the polar geodesic unwrapping
                    # =============================================================================
                    """
                    Construct reference space and pull back to determine the (i,j) <-> (x,y,z) mapping  for polar distal  
                    """
                    
                    mapping_space_polar = uzip.build_mapping_space(geodesic_pts[:,:3], 
                                                                   ranges=[-polar_scale_factor*max_s, -polar_scale_factor*max_s, polar_scale_factor*max_s, polar_scale_factor*max_s], 
                                                                   shape=[int(np.rint(1*polar_scale_factor*s_resolution)), 
                                                                          int(np.rint(1*polar_scale_factor*s_resolution))], 
                                                                   polar_map=True)
                                       
                    mapped_index_polar = uzip.match_coords_to_ref_space_general(geodesic_pts, 
                                                                                mapping_space_polar.reshape(-1,2), 
                                                                                map_index=[-2,-1], rescale=True)
                    mapped_coords_polar = np.unravel_index(mapped_index_polar, mapping_space_polar[...,0].shape, order='F')
                    mapped_coords_polar = np.hstack(mapped_coords_polar)
                    mapped_index_map = np.zeros(mapping_space_polar[...,0].shape)
                    mapped_index_map[mapped_coords_polar[:,0],
                                     mapped_coords_polar[:,1]] = 1
                
                    
                    ref_coord_set_polar = np.hstack([geodesic_pts, mapped_coords_polar])
                
                    # rbf does a very poor job preserving the distal mapping. 
                    ref_map_xyz_polar = uzip.gen_ref_map(im_array, 
                                                        ref_coord_set_polar, 
                                                        mapping_space_polar, interp_type='barycentric', interp_method='linear', 
                                                        rbf_samples=1000, 
                                                        rbf_basis='multiquadric', 
                                                        rbf_epsilon=10, 
                                                        rbf_smooth=0.01) # how much to smooth? (too much smoothing distorts the distal tip!)
                    
                    
                    # mask the polar projection to ensure a maximum s.
                    XX, YY = np.meshgrid(np.arange(ref_map_xyz_polar.shape[1]),
                                         np.arange(ref_map_xyz_polar.shape[0]))
                    
                    polar_mask = np.sqrt( (XX-ref_map_xyz_polar.shape[1]/2.)**2 + (YY-ref_map_xyz_polar.shape[0]/2.)**2 ) >= int(max_s/2.)
                    ref_map_xyz_polar_mask = ref_map_xyz_polar.copy(); ref_map_xyz_polar_mask[polar_mask,:] = 0
                    
                    
                    ref_map_xyz_polar = ref_map_xyz_polar_mask.copy()
                    
                    
                    ref_map_xyz_polar_filt = np.dstack([gaussian(ref_map_xyz_polar[...,i_], sigma=sigma_polar_xyz, mode='nearest', preserve_range=True) for i_ in range(3)])
                    
                    # =============================================================================
                    #  Save the unwrapping parameters learnt for the geodesic   
                    # =============================================================================
                    
                    if surface_to_unwrap == 've' or surface_to_unwrap=='epi':
                        basename = os.path.split(unwrap_params_file)[-1].split('_tp')[0] # + '_HEX' # to distinguish itself. 
                    else:
                        basename = os.path.split(unwrap_params_file)[-1].split('_tp')[0] + '_HEX' # distinguish with hex. 
                    
##                    savematfile = os.path.join(out_unwrapped_params_folder, basename+'_unwrap_params-geodesic.mat')
##                    spio.savemat(savematfile, { 'ref_coords': ref_coords,
##                                                'unwrap_params': unwrap_params, 
##                                                'mapping_space_polar':mapping_space_polar, 
##                                                'mapping_space_rect':mapping_space_rect, 
##                                                'ref_map_polar_xyz':  ref_map_xyz_polar, 
##                                                'ref_map_rect_xyz': ref_map_xyz_rect, 
##                                                'filt_rect_xyz': sigma_rect_xyz, 
##                                                'filt_polar_xyz': sigma_polar_xyz, 
##                                                'polar_scale_factor':polar_scale_factor} )
                    
                    I_polar_unwrapped = map_intensity_interp3(ref_map_xyz_polar_filt, im_array.shape, im_array)
                        
                    plt.figure(figsize=(10,10))
                    plt.imshow(I_polar_unwrapped, cmap='gray')
                    plt.show()
                    
                    
                    ref_center = ref_map_xyz_polar[ref_map_xyz_polar.shape[0]//2, ref_map_xyz_polar.shape[1]//2]
                    ref_slice = int(ref_center[1])
                    
                    # this is the slice. 
                    plt.figure(figsize=(10,10))
                    plt.imshow(im_array[:,ref_slice], cmap='gray')
                    plt.plot(ref_map_xyz_polar[:,ref_map_xyz_polar.shape[1]//2,2],
                             ref_map_xyz_polar[:,ref_map_xyz_polar.shape[1]//2,0], 'r-')
                    plt.show()
                    
                    
                    
                    
# =============================================================================
#                     Check the cross section... 
# =============================================================================
                    
#                    # =============================================================================
#                    #     Now map the intensities. 
#                    # =============================================================================
#                    list_im_array_rect = []
#                    list_im_array_polar = []
#                    
#                    for embryo_im_file in tqdm(embryo_im_files[:]):
#                    
#                        im_array = fio.read_multiimg_PIL(embryo_im_file)
#        
#                        if rot_angle != 0:
#                            # get the corresponding rotation file:
#                            rottformfile = os.path.join(embryo_folder, 'step6_%s_rotation_tforms.mat' %(str(rot_angle).zfill(3)))
#                            rottform = spio.loadmat(rottformfile)['tforms'][0]
#                            im_array = apply_affine_tform(im_array, rottform)
#                
#                        im_array = im_array.transpose(1,0,2) # transpose to long axes  
#                        
#                        I_rect_unwrapped = map_intensity_interp3(ref_map_xyz_rect_filt, im_array.shape, im_array)
#                        I_polar_unwrapped = map_intensity_interp3(ref_map_xyz_polar_filt, im_array.shape, im_array)
#                        
#                        list_im_array_rect.append(I_rect_unwrapped[::-1])
#                        list_im_array_polar.append(I_polar_unwrapped)
#                        
#                
#                    list_im_array_rect = np.array(list_im_array_rect)
#                    list_im_array_polar = np.array(list_im_array_polar)
###                
###                    """
###                    Save the results 
###                    """
###                    fio.save_multipage_tiff(list_im_array_rect, os.path.join(out_unwrapped_folder, basename+'_rect.tif'))
###                    fio.save_multipage_tiff(list_im_array_polar, os.path.join(out_unwrapped_folder, basename+'_polar.tif'))
##                    
#                
#        
#    
#    
#    
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    