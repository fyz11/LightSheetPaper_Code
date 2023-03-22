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
    
def parse_condition_fnames_comb_unwrapped(vidfile):
    
    import os 
    import re 
    
    fname = os.path.split(vidfile)[-1]
    
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
            
    return emb_no, ang
    

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
    
#    print(name)
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


def preprocess_unwrap_pts(pts3D, im_array, rot_angle_3d=None, transpose_axis=None):
    
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
                                                        direction='B') # for pts is reverse of image. 
    # clip the values to the array.
    pts[...,0] = np.clip(pts[...,0], 0, im_array.shape[0]-1)
    pts[...,1] = np.clip(pts[...,1], 0, im_array.shape[1]-1)
    pts[...,2] = np.clip(pts[...,2], 0, im_array.shape[2]-1)
    
    return pts

import numpy as np 
def remap_pts_rot_angle(pts3D, im_array, dx,dy,dz, translation=[0,0,0], rotation=np.eye(3), zoom=[1,1,1], shear=[0,0,0], center_tform=True):
    
    dx_ = dx[pts3D[...,0], pts3D[...,1], pts3D[...,2]]
    dy_ = dy[pts3D[...,0], pts3D[...,1], pts3D[...,2]]
    dz_ = dz[pts3D[...,0], pts3D[...,1], pts3D[...,2]]
    
    dxyz = np.vstack([dx_,dy_,dz_]).T
    
    demons_rev_xyz = reg.warp_3D_displacements_xyz_pts(pts3D, 
                                                       dxyz,
                                                       direction='F') 
    
    """
    # step 2: recover the rigid registration on top of the demons registration.
    """
    orig_xyz = reg.warp_3D_transforms_xyz_similarity_pts(demons_rev_xyz, 
                                                         translation=translation, 
                                                         rotation=rotation, 
                                                         zoom=zoom, 
                                                         shear=shear, 
                                                         center_tform=center_tform, 
                                                         im_center = np.hstack(im_array.shape)/2.,
                                                         direction='B') # for pts is reverse of image. 
    
    return orig_xyz


def lookup_3D_demons(pts3D, dx,dy,dz):
    
    dx_ = dx[pts3D[...,0], pts3D[...,1], pts3D[...,2]]
    dy_ = dy[pts3D[...,0], pts3D[...,1], pts3D[...,2]]
    dz_ = dz[pts3D[...,0], pts3D[...,1], pts3D[...,2]]

    dxyz = np.vstack([dx_,dy_,dz_]).T
    
    return dxyz
    

if __name__=="__main__":
    
# =============================================================================
#     note: for Hex, the demons will come from the MTMG folder. 
# =============================================================================
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
    from transforms3d.affines import decompose44, compose
    """
    Detect and Read in Dataset. 
    """

#    surface_to_unwrap = 've'
    
    all_embryo_folder = '/media/felix/Srinivas4/MTMG-HEX' # just lifeAct. 
#    all_embryo_folder = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
#    all_embryo_folder = '/media/felix/Srinivas4/MTMG-TTR'
#    all_embryo_folder = '/media/felix/Srinivas4/LifeAct'
    embryo_folders = np.hstack([os.path.join(all_embryo_folder, name) for name in os.listdir(all_embryo_folder) if os.path.isdir(os.path.join(all_embryo_folder, name))])
    embryo_folders = np.sort(embryo_folders)
        
    # why is embryo 5 different?
    for embryo_folder in embryo_folders[3:5]:

        # detect step images folder for the various layers? 
        embryo_im_folder = [os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step6_' in name or 'step6tf_' in name) and os.path.isdir(os.path.join(embryo_folder, name))]
        
        if len(embryo_im_folder) == 0: 
            embryo_im_folder = np.hstack([os.path.join(embryo_folder, name) for name in os.listdir(embryo_folder) if ('step6-' in name or 'step6tf-' in name) and os.path.isdir(os.path.join(embryo_folder, name))])
        else:
            embryo_im_folder = np.hstack(embryo_im_folder)
            
        # =============================================================================
        #   1. parse the folders for the respective VE, Epi and Hex layers.   
        # =============================================================================
        if len(embryo_im_folder) > 0:
            
            """
            only need this to set the video names + im array size. 
            """
            
            # VE or Epi 
            embryo_im_folder_ve_epi = [folder for folder in embryo_im_folder if 'mtmg' in os.path.split(folder)[-1]][0]
            # Hex
            embryo_im_folder_hex = [folder for folder in embryo_im_folder if 'hex' in os.path.split(folder)[-1]][0]
        
            print('processing VE-Epi: ', embryo_im_folder_ve_epi)
            print('processing Hex: ', embryo_im_folder_hex)

            # =============================================================================
            #   Get the associated time reg folder.           
            # =============================================================================
            
#            # the set for L924_TP1-97
#            """
#            1. first temp reg file
#            """
#            time_step4_reg_file_1 = os.path.join(embryo_folder, 'step4_temp_reg_tforms-1.mat')
#            tforms_time_1 = spio.loadmat(time_step4_reg_file_1)['tforms']
#            
#            """
#            2. second temp reg file
#            """
#            time_step4_reg_file_2 = os.path.join(embryo_folder, 'step4_temp_reg_tforms-3.mat')
#            tforms_time_2 = spio.loadmat(time_step4_reg_file_2)['tforms']
            
            
#            # the set for L928_Emb1_TP1-94
#            """
#            1. first temp reg file
#            """
#            time_step4_reg_file_1 = os.path.join(embryo_folder, 'step4_temp_reg_tforms.mat')
#            tforms_time_1 = spio.loadmat(time_step4_reg_file_1)['tforms']
#            
#            """
#            2. second temp reg file
#            """
#            time_step4_reg_file_2 = os.path.join(embryo_folder, 'step4_temp_reg_tforms-2.mat')
#            tforms_time_2 = spio.loadmat(time_step4_reg_file_2)['tforms']
#            
            # =============================================================================
            #   Get the associated time demons reg folder.           
            # =============================================================================
            embryo_demon_files = glob.glob(os.path.join(embryo_im_folder_ve_epi, '*.mat')); 
           
            if len(embryo_demon_files) == 0:
                embryo_demon_files = np.hstack(glob.glob(os.path.join(embryo_im_folder_hex, '*.mat'))); 
            else:
                embryo_demon_files = np.hstack(embryo_demon_files)
                # get the times.
            
            frame_nos = np.hstack([find_time_string(os.path.split(f)[-1]) for f in embryo_demon_files])
           
            # for LEFTY embryo
#            frame_nos = np.hstack([find_time_string_MTMG(os.path.split(f)[-1]) for f in embryo_demon_files])
            embryo_demon_files = embryo_demon_files[np.argsort(frame_nos)]


            print(embryo_demon_files)
            print('===================')

            """
            Load in the unwrap params for the particular dataset
            """
            print('finding unwrapping parameter folder')
            # auto detect and fetch the unwrap_params folder. 
            unwrap_params_folder_path = find_unwrap_params_folder(embryo_folder, key='geodesic_VE_Epi_matched-resize-rotmatrix', exclude='revmap')[0]
            print(unwrap_params_folder_path)

            all_unwrap_params_files = glob.glob(os.path.join(unwrap_params_folder_path, '*.mat'))    
            # parse the meta details from the param files. for automation. 
            metadata_files = np.vstack([parse_condition_fnames_comb_unwrapped(f) for f in all_unwrap_params_files])

            """
            Iterate over the unwrap files. 
            """
            # shit.. was :1.
            for file_i in range(len(all_unwrap_params_files))[:]:
                
                unwrap_params_file = all_unwrap_params_files[file_i]
                    
                # specify the angle of rotation and the timepoint. 
                rot_angle = int(metadata_files[file_i][-1]) 
                
                print(unwrap_params_file)
                print('Rotation angle: ', rot_angle)
                print('---------------')
                
                
                """
                
                """
                if rot_angle  != 0:
                    # load the transforms from Holly's folder. 
                    print('loading rotation angles')
                    rottformfile = os.path.join(embryo_folder, 'step6_%s_rotation_tforms.mat' %(str(rot_angle).zfill(3)))
                    print(rottformfile)
                    rottform = spio.loadmat(rottformfile)['tforms'][0]
#                        rottform = np.linalg.inv(rottform)
                    print(rottform)
                    
#                    unwrap_params_ve = reg.warp_3D_xyz_tmatrix_pts(unwrap_params_ve[...,[1,0,2]], rottform)
#                    unwrap_params_epi = reg.warp_3D_xyz_tmatrix_pts(unwrap_params_epi[...,[1,0,2]], rottform)
#                else:
#                    rottform = np.eye(4)
#                    unwrap_params_ve = unwrap_params_ve[...,[1,0,2]]
#                    unwrap_params_epi = unwrap_params_epi[...,[1,0,2]]
                
                     
                """
                Specify the final output folder for saving the reconstituted binary coordinates.. 
                """
                out_unwrap_folder = os.path.join(unwrap_params_folder_path+'_demons_revmap', os.path.split(unwrap_params_file)[-1].split('.mat')[0]); 
                fio.mkdir(out_unwrap_folder)
                print(out_unwrap_folder)
                
                """
                Read in the relevant coordinates in the mapping. 
                """
                embryo_im_files = np.hstack(glob.glob(os.path.join(embryo_im_folder_ve_epi, '*.tif')))
#                frame_nos = np.hstack([find_time_string_MTMG(os.path.split(f)[-1].split('.tif')[0]) for f in embryo_im_files])
                try:
                    frame_nos = np.hstack([find_time_string(os.path.split(f)[-1].split('.tif')[0]) for f in embryo_im_files])
                except:
                    frame_nos = np.hstack([find_time_string_MTMG(os.path.split(f)[-1].split('.tif')[0]) for f in embryo_im_files])
                    
                embryo_im_files = embryo_im_files[np.argsort(frame_nos)]
                embryo_im_file = embryo_im_files[0]
#                embryo_im_file = embryo_im_files[ref_file_id]
                im_array = fio.read_multiimg_PIL(embryo_im_file)
                
                ###### Epi
                epi_xyz_polar = np.rint(spio.loadmat(spio.loadmat(unwrap_params_file)['unwrap_epi_filepath'][0])['ref_map_polar_xyz']).astype(np.int)
                epi_xyz_rect = np.rint(spio.loadmat(spio.loadmat(unwrap_params_file)['unwrap_epi_filepath'][0])['ref_map_rect_xyz']).astype(np.int)
                ###### VE
                ve_xyz_polar = np.rint(spio.loadmat(spio.loadmat(unwrap_params_file)['unwrap_ve_filepath'][0])['ref_map_polar_xyz']).astype(np.int)
                ve_xyz_rect = np.rint(spio.loadmat(spio.loadmat(unwrap_params_file)['unwrap_ve_filepath'][0])['ref_map_rect_xyz']).astype(np.int)
                ###### HEX
                hex_xyz_polar = np.rint(spio.loadmat(spio.loadmat(unwrap_params_file)['unwrap_hex_filepath'][0])['ref_map_polar_xyz']).astype(np.int)
                hex_xyz_rect = np.rint(spio.loadmat(spio.loadmat(unwrap_params_file)['unwrap_hex_filepath'][0])['ref_map_rect_xyz']).astype(np.int)

                
                """
                
                """
#                if rot_angle!=0:
#                    epi_xyz_polar = reg.warp_3D_xyz_tmatrix_pts(epi_xyz_polar[...,[1,0,2]], rottform)
#                    epi_xyz_rect = reg.warp_3D_xyz_tmatrix_pts(epi_xyz_rect[...,[1,0,2]], rottform)
#                    
#                    ve_xyz_polar = reg.warp_3D_xyz_tmatrix_pts(ve_xyz_polar[...,[1,0,2]], rottform)
#                    ve_xyz_rect = reg.warp_3D_xyz_tmatrix_pts(ve_xyz_rect[...,[1,0,2]], rottform)
#                    
#                    hex_xyz_polar = reg.warp_3D_xyz_tmatrix_pts(hex_xyz_polar[...,[1,0,2]], rottform)
#                    hex_xyz_rect = reg.warp_3D_xyz_tmatrix_pts(hex_xyz_rect[...,[1,0,2]], rottform)
#                else:
#                    epi_xyz_polar = epi_xyz_polar[...,[1,0,2]]
#                    epi_xyz_rect = epi_xyz_rect[...,[1,0,2]]
#                    
#                    ve_xyz_polar = ve_xyz_polar[...,[1,0,2]]
#                    ve_xyz_rect = ve_xyz_rect[...,[1,0,2]]
#                    
#                    hex_xyz_polar = hex_xyz_polar[...,[1,0,2]]
#                    hex_xyz_rect = hex_xyz_rect[...,[1,0,2]]
                    
                """
                preprocess the points to allow them to correspond to the image (so we can apply demons) *THIS iS WRONG'
                """                
                epi_xyz_polar = preprocess_unwrap_pts(epi_xyz_polar, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2])
                epi_xyz_rect = preprocess_unwrap_pts(epi_xyz_rect, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2])
                
                ve_xyz_polar = preprocess_unwrap_pts(ve_xyz_polar, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2])
                ve_xyz_rect = preprocess_unwrap_pts(ve_xyz_rect, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2])
                
                hex_xyz_polar = preprocess_unwrap_pts(hex_xyz_polar, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2])
                hex_xyz_rect = preprocess_unwrap_pts(hex_xyz_rect, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2])
                
#            # =============================================================================
#            #   Visualize the coordinates. 
#            # =============================================================================
#                ve_xyz_polar_rev = ve_xyz_polar.copy()
#                epi_xyz_polar_rev = epi_xyz_polar.copy()
#                vid_demons_temp_rev_1 = im_array.copy()
#                
#                
#                select_z = im_array.shape[1]//2
#                select_coords_ve = np.logical_and(ve_xyz_polar_rev.reshape(-1,3)[:,1] >= select_z-0.5, 
#                                                  ve_xyz_polar_rev.reshape(-1,3)[:,1] < select_z+0.5)
#                select_coords_epi = np.logical_and(epi_xyz_polar_rev.reshape(-1,3)[:,1] >= select_z-0.5, 
#                                                   epi_xyz_polar_rev.reshape(-1,3)[:,1] < select_z + 0.5)
##                    select_coords_hex = np.logical_and(hex_xyz_polar_rev.reshape(-1,3)[:,1] >= select_z-0.5, 
##                                                       hex_xyz_polar_rev.reshape(-1,3)[:,1] < select_z + 0.5)
#                
#                fig, ax = plt.subplots(figsize=(15,15))
#                plt.suptitle('original for')
#                plt.imshow(vid_demons_temp_rev_1[:,select_z])
##                    plt.plot(pts3D[select_coords,0],
##                             pts3D[::1*n,2], 'o', color='g')
#                plt.plot(ve_xyz_polar_rev.reshape(-1,3)[select_coords_ve,2],
#                         ve_xyz_polar_rev.reshape(-1,3)[select_coords_ve,0], 'o', color='g')
#                plt.plot(epi_xyz_polar_rev.reshape(-1,3)[select_coords_epi,2],
#                         epi_xyz_polar_rev.reshape(-1,3)[select_coords_epi,0], '.', color='r')
##                    plt.plot(hex_xyz_polar_rev.reshape(-1,3)[select_coords_hex,2],
##                             hex_xyz_polar_rev.reshape(-1,3)[select_coords_hex,0], '.', color='r')
#                ax.set_aspect(1)
##                    ax.set_ylim([im_array.shape[0], 0])
##                    ax.set_xlim([0,im_array.shape[0]])
#                plt.show()  
                
                
#                hex_xyz_polar = preprocess_unwrap_pts(hex_xyz_polar, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2])
#                hex_xyz_rect = preprocess_unwrap_pts(hex_xyz_rect, im_array, rot_angle_3d=rot_angle, transpose_axis=[1,0,2])
                
                # =============================================================================
                #     Now remap the coordinates per time point.
                # =============================================================================

                if rot_angle != 0: 
                    rot_matrix = geom.get_rotation_y(rot_angle/180.*np.pi)[:3,:3]
                else:
                    rot_matrix = np.eye(3)

#                all_epi_polar_dxyz = []
#                all_epi_rect_dxyz = []
#                all_ve_polar_dxyz = []
#                all_ve_rect_dxyz = []
                from skimage.exposure import equalize_hist
                for frame in tqdm(range(len(embryo_demon_files))[:]):
                    
                    # look up the embryo_demon transform for the respective volume coordinate.
                    
                    """
                    check the application of the transforms on the images. 
                    """
                    emb_file = embryo_im_files[frame]
                    im = fio.read_multiimg_PIL(embryo_im_files[frame])
                    
                    
                    
                    select_z = im.shape[0]//2
                    select_pts = np.logical_and(ve_xyz_polar[...,0] > select_z-1, 
                                                ve_xyz_polar[...,0] < select_z+1)
                    
                    fig, ax = plt.subplots(figsize=(5,5))
                    ax.imshow(equalize_hist(im[select_z]), cmap='gray')
                    ax.plot(ve_xyz_polar[select_pts,2], 
                            ve_xyz_polar[select_pts,1], 'r.')
                    plt.show()
                    
                    
                    tform_file = embryo_demon_files[frame]
                    
#                    """
#                    # step1: recover the demons first demons registration 
#                    """
#                    # the reading of the transform file and its manipulation is very slow? 
#                    
#    #                print('warping demons')
#    #                import time 
#    #                t1 = time.time()
#                    vid_demons_rev = reg.warp_3D_demons_matlab_tform_scipy(im, tform_file, direction='B', pad=None) # 241 sec? really? 
#    #                print(time.time()-t1)
#    
##                    """
##                    # step2: apply the rigid temporal registration 
##                    """
##                    tmatrix_frame_1 = tforms_time_1[frame]
##                    T1, R1, Z1, S1 = decompose44(tmatrix_frame_1)
##                    
##                    tmatrix_frame_2 = tforms_time_2[frame]
##                    T2, R2, Z2, S2 = decompose44(tmatrix_frame_2)
##                    
##                    if rot_angle != 0: 
##                        rot_matrix = geom.get_rotation_y(rot_angle/180.*np.pi)[:3,:3]
##                    else:
##                        rot_matrix = np.eye(3)
##                    
##    #                t1 = time.time()
##                    vid_demons_temp_rev_2 = reg.warp_3D_transforms_xyz_similarity(vid_demons_rev, 
##                                                                                  rotation = rot_matrix,
##                                                                                  zoom = Z2,
##                                                                                  center_tform=True,
##                                                                                  direction='F',
##                                                                                  pad=None)
##                    
##                    vid_demons_temp_rev_1 = reg.warp_3D_transforms_xyz_similarity(vid_demons_temp_rev_2, 
##                                                                                  rotation = np.eye(3),
##                                                                                  zoom = Z1,
##                                                                                  center_tform=True,
##                                                                                  direction='F',
##                                                                                  pad=None)
##                    
###                    stacked = np.dstack([vid_demons_rev.max(axis=1), 
###                                         vid_demons_temp_rev_2.max(axis=1),
###                                         vid_demons_temp_rev_1.max(axis=1)])
#                    
                    """
                    # step1: recover the demons first demons registration 
                    """
                    # read in the 
                    dxyz = fio.read_demons_matlab_tform( embryo_demon_files[frame], im_array.shape)
                    
                    # this is x,y,z -> transpose to fit the pts3D?
                    dx, dy, dz = dxyz
                    
#                    """
#                    # step 2: recover the rigid registration on top of the demons registration.
#                    """
#                    tmatrix_frame_1 = tforms_time_1[frame]
#                    T1, R1, Z1, S1 = decompose44(tmatrix_frame_1)
#                    
#                    tmatrix_frame_2 = tforms_time_2[frame]
#                    T2, R2, Z2, S2 = decompose44(tmatrix_frame_2)
                    
                    if rot_angle != 0: 
                        rot_matrix = geom.get_rotation_y(rot_angle/180.*np.pi)[:3,:3]
                    else:
                        rot_matrix = np.eye(3)
                    
                    # no interp (interp might be more accurate but slower?)
                    epi_polar_dxyz = lookup_3D_demons(np.rint(epi_xyz_polar.reshape(-1,3)).astype(np.int), 
                                                       dx,dy,dz)
                    
                    epi_rect_dxyz = lookup_3D_demons(np.rint(epi_xyz_rect.reshape(-1,3)).astype(np.int), 
                                                       dx,dy,dz)
                    
                    ve_polar_dxyz = lookup_3D_demons(np.rint(ve_xyz_polar.reshape(-1,3)).astype(np.int), 
                                                       dx,dy,dz)
                    
                    ve_rect_dxyz = lookup_3D_demons(np.rint(ve_xyz_rect.reshape(-1,3)).astype(np.int), 
                                                       dx,dy,dz)
                    
                    hex_polar_dxyz = lookup_3D_demons(np.rint(hex_xyz_polar.reshape(-1,3)).astype(np.int), 
                                                       dx,dy,dz)
                    
                    hex_rect_dxyz = lookup_3D_demons(np.rint(hex_xyz_rect.reshape(-1,3)).astype(np.int), 
                                                       dx,dy,dz)
                    
                    epi_polar_dxyz = epi_polar_dxyz.reshape(epi_xyz_polar.shape)
                    epi_rect_dxyz = epi_rect_dxyz.reshape(epi_xyz_rect.shape)
                    ve_polar_dxyz = ve_polar_dxyz.reshape(ve_xyz_polar.shape)
                    ve_rect_dxyz = ve_rect_dxyz.reshape(ve_xyz_rect.shape)
                    hex_polar_dxyz = hex_polar_dxyz.reshape(hex_xyz_polar.shape)
                    hex_rect_dxyz = hex_rect_dxyz.reshape(hex_xyz_rect.shape)
                    
                    
#                    all_epi_polar_dxyz.append(epi_polar_dxyz)
#                    all_epi_rect_dxyz.append(epi_rect_dxyz)
#                    all_ve_polar_dxyz.append(ve_polar_dxyz)
#                    all_ve_rect_dxyz.append(ve_rect_dxyz)
#                    
#                all_epi_polar_dxyz = np.array(all_epi_polar_dxyz).astype(np.float32)
#                all_epi_rect_dxyz = np.array(all_epi_rect_dxyz).astype(np.float32)
#                all_ve_polar_dxyz = np.array(all_ve_polar_dxyz).astype(np.float32)
#                all_ve_rect_dxyz = np.array(all_ve_rect_dxyz).astype(np.float32)
                
                
                
                    
#                    epi_xyz_polar_rev = epi_xyz_polar_rev.reshape(epi_xyz_polar.shape)
#                    epi_xyz_rect_rev = epi_xyz_rect_rev.reshape(epi_xyz_rect.shape)
#                    ve_xyz_polar_rev = ve_xyz_polar_rev.reshape(ve_xyz_polar.shape)
#                    ve_xyz_rect_rev = ve_xyz_rect_rev.reshape(ve_xyz_rect.shape)
#                    hex_xyz_polar_rev = hex_xyz_polar_rev.reshape(hex_xyz_polar.shape)
#                    hex_xyz_rect_rev = hex_xyz_rect_rev.reshape(hex_xyz_rect.shape)
#                    
#                    """
#                    check step  on the volumetric transformed image. 
#                    """
#                    im = fio.read_multiimg_PIL(embryo_im_files[frame])
#                    vid_demons_rev = reg.warp_3D_demons_matlab_tform_scipy(im, embryo_demon_files[frame], direction='B', pad=None) # 241 sec? really? 
#                    vid_demons_temp_rev = reg.warp_3D_transforms_xyz_similarity(vid_demons_rev, 
#                                                                                rotation = rot_matrix,
#                                                                                zoom = Z,
#                                                                                center_tform=True,
#                                                                                direction='F',
#                                                                                pad=None)
#                    
#                    select_z = im_array.shape[1]//2
#                    select_coords_ve = np.logical_and(ve_xyz_polar_rev.reshape(-1,3)[:,1] >= select_z-0.5, 
#                                                      ve_xyz_polar_rev.reshape(-1,3)[:,1] < select_z+0.5)
#                    select_coords_epi = np.logical_and(epi_xyz_polar_rev.reshape(-1,3)[:,1] >= select_z-0.5, 
#                                                       epi_xyz_polar_rev.reshape(-1,3)[:,1] < select_z + 0.5)
#                    select_coords_hex = np.logical_and(hex_xyz_polar_rev.reshape(-1,3)[:,1] >= select_z-0.5, 
#                                                       hex_xyz_polar_rev.reshape(-1,3)[:,1] < select_z + 0.5)
#                    
#                    fig, ax = plt.subplots(figsize=(15,15))
#                    plt.suptitle('original for')
#                    plt.imshow(vid_demons_temp_rev_1[:,select_z])
##                    plt.plot(pts3D[select_coords,0],
##                             pts3D[::1*n,2], 'o', color='g')
#                    plt.plot(ve_xyz_polar_rev.reshape(-1,3)[select_coords_ve,2],
#                             ve_xyz_polar_rev.reshape(-1,3)[select_coords_ve,0], 'o', color='g')
#                    plt.plot(epi_xyz_polar_rev.reshape(-1,3)[select_coords_epi,2],
#                             epi_xyz_polar_rev.reshape(-1,3)[select_coords_epi,0], '.', color='r')
#                    plt.plot(hex_xyz_polar_rev.reshape(-1,3)[select_coords_hex,2],
#                             hex_xyz_polar_rev.reshape(-1,3)[select_coords_hex,0], '.', color='b')
#                    ax.set_aspect(1)
##                    ax.set_ylim([im_array.shape[0], 0])
##                    ax.set_xlim([0,im_array.shape[0]])
#                    plt.show()  
                    
##                    select_z = vid_demons_temp_rev.shape[2]//2
##                    select_coords = np.logical_and(orig_xyz[:,2] >= select_z-0.5, 
##                                                   orig_xyz[:,2] < select_z+0.5)
##                    select_coords_ve = np.logical_and(orig_xyz_ve[:,2] >= select_z-0.5, 
##                                                      orig_xyz_ve[:,2] < select_z + 0.5)
##                    
##                    fig, ax = plt.subplots(figsize=(15,15))
##                    plt.suptitle('original for')
##                    plt.imshow(vid_demons_temp_rev[:,:,select_z])
###                    plt.plot(pts3D[select_coords,0],
###                             pts3D[::1*n,2], 'o', color='g')
##                    
##                    plt.plot(orig_xyz[select_coords,1],
##                             orig_xyz[select_coords,0], 'o', color='g')
##                    plt.plot(orig_xyz_ve[select_coords_ve,1],
##                             orig_xyz_ve[select_coords_ve,0], '.', color='r')
##                    ax.set_aspect(1)
###                    ax.set_ylim([im_array.shape[0], 0])
###                    ax.set_xlim([0,im_array.shape[0]])
##                    plt.show()  
#                    
#                    
##                    select_z = vid_demons_temp_rev.shape[1]//2
##                    select_coords = np.logical_and(orig_xyz[:,1] >= select_z-0.5, 
##                                                   orig_xyz[:,1] < select_z+0.5)
##                    select_coords_ve = np.logical_and(orig_xyz_ve[:,1] >= select_z-0.5, 
##                                                      orig_xyz_ve[:,1] < select_z + 0.5)
##                    
##                    fig, ax = plt.subplots(figsize=(10,10))
##                    plt.suptitle('original for')
##                    plt.imshow(vid_demons_temp_rev[:,select_z])
###                    plt.plot(pts3D[select_coords,0],
###                             pts3D[::1*n,2], 'o', color='g')
##                    plt.plot(orig_xyz[select_coords,2],
##                             orig_xyz[select_coords,0], 'o', color='g')
##                    plt.plot(orig_xyz_ve[select_coords_ve,2],
##                             orig_xyz_ve[select_coords_ve,0], '.', color='r')
##                    ax.set_aspect(1)
###                    ax.set_ylim([im_array.shape[0], 0])
###                    ax.set_xlim([0,im_array.shape[0]])
##                    plt.show()
                    
                    """
                    save out using integers.
                    """
                    filebasename = os.path.split(embryo_demon_files[frame])[-1]
                    filebasename = filebasename.replace('_demons_tform_', '_')
                    savefile = os.path.join(out_unwrap_folder, 'unwrap_xyz_rev_demons_'+filebasename.replace('.tif', '.mat'))
                    
#                    spio.savemat(savefile, 
#                                 {'epi_xyz_polar_rev': epi_xyz_polar_rev.astype(np.int) ,
#                                  'epi_xyz_rect_rev': epi_xyz_rect_rev.astype(np.int),
#                                  've_xyz_polar_rev': ve_xyz_polar_rev.astype(np.int),
#                                  've_xyz_rect_rev': ve_xyz_rect_rev.astype(np.int),
#                                  'hex_xyz_polar_rev': hex_xyz_polar_rev.astype(np.int),
#                                  'hex_xyz_rect_rev': hex_xyz_rect_rev.astype(np.int)})
    
#                    spio.savemat(savefile, 
#                                 {'epi_polar_dxyz': epi_polar_dxyz ,
#                                  'epi_rect_dxyz': epi_rect_dxyz,
#                                  've_polar_dxyz': ve_polar_dxyz,
#                                  've_rect_dxyz': ve_rect_dxyz})
    
                    spio.savemat(savefile, 
                                 {'epi_polar_dxyz': epi_polar_dxyz ,
                                  'epi_rect_dxyz': epi_rect_dxyz,
                                  've_polar_dxyz': ve_polar_dxyz,
                                  've_rect_dxyz': ve_rect_dxyz,
                                  'hex_polar_dxyz': hex_polar_dxyz,
                                  'hex_rect_dxyz': hex_rect_dxyz})
    
    
    

    
    