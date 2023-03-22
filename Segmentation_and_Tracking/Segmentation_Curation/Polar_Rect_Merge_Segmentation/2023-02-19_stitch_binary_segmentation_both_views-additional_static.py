# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:51:28 2021

@author: fyz11
"""

def map_rect_to_polar(img, mapping, target_shape):
    
    img_polar = np.zeros(target_shape)
    img_polar[mapping[0], 
              mapping[1]] = img.copy()
    
    return img_polar


# here we take into account the actual 3D !.. 
def grow_polar_regions_nearest(labels, unwrap_params, dilate=5, metric='manhattan', debug=False):
    
    from scipy.ndimage.morphology import binary_fill_holes
    from sklearn.neighbors import NearestNeighbors
    """
    script will grow the cell regions so that there is 0 distance between each cell region. 
    """
    bg = labels>0
    bg = skmorph.binary_closing(bg, skmorph.disk(dilate))
    valid_mask = binary_fill_holes(bg) # this needs to be a circle... 
    
    # plt.figure()
    # plt.imshow(valid_mask)
    # plt.show()
    
    # # in particular we need the valid mask to estimate a maximal circle!!!. 
    # YY, XX = np.indices(labels.shape)
    # RR = np.sqrt((YY - labels.shape[0]/2.)**2 + (XX - labels.shape[1]/2.)**2)
    # valid_mask = RR <= np.max(RR[valid_mask]) # guarantee a circle. 
    
    # plt.figure()
    # plt.imshow(valid_mask)
    # plt.show()
    
    uncovered_mask = np.logical_and(valid_mask, labels==0)
    covered_mask = np.logical_and(valid_mask, labels > 0)
    
    if debug: 
        plt.figure()
        plt.title('uncovered')
        plt.imshow(uncovered_mask)
        plt.show()
        
        plt.figure()
        plt.title('covered')
        plt.imshow(covered_mask)
        plt.show()
        
    YY, XX = np.indices(labels.shape)
    
    uncovered_coords = np.vstack([YY[uncovered_mask], 
                                  XX[uncovered_mask]]).T
        
    covered_coords = np.vstack([YY[covered_mask], 
                                XX[covered_mask]]).T
    
    uncovered_coords_3D = unwrap_params[uncovered_coords[:,0], 
                                        uncovered_coords[:,1]]
    covered_coords_3D = unwrap_params[covered_coords[:,0], 
                                      covered_coords[:,1]]
    
    """
    build a kNN tree for assigning the remainder. 
    """
    neigh = NearestNeighbors(n_neighbors=1, metric=metric)
    neigh.fit(covered_coords_3D)

    nbr_indices = neigh.kneighbors(uncovered_coords_3D, return_distance=False)
    nbr_indices = nbr_indices.ravel()
    
    """
    look up what labels should be assigned
    """
    labels_new = labels.copy()
    
    nbr_pts = covered_coords[nbr_indices]
    
    labels_new[uncovered_coords[:,0], 
               uncovered_coords[:,1]] = labels[nbr_pts[:,0], nbr_pts[:,1]].copy()
    
    return valid_mask, labels_new


def grow_polar_regions_nearest_rect(labels, unwrap_params, dilate=5, metric='manhattan', debug=False):
    
    from scipy.ndimage.morphology import binary_fill_holes
    from sklearn.neighbors import NearestNeighbors
    """
    script will grow the cell regions so that there is 0 distance between each cell region. 
    """
    uncovered_mask = labels==0
    covered_mask = labels > 0
    
    if debug: 
        plt.figure()
        plt.title('uncovered')
        plt.imshow(uncovered_mask)
        plt.show()
        
        plt.figure()
        plt.title('covered')
        plt.imshow(covered_mask)
        plt.show()
        
    YY, XX = np.indices(labels.shape)
    
    uncovered_coords = np.vstack([YY[uncovered_mask], 
                                  XX[uncovered_mask]]).T
        
    covered_coords = np.vstack([YY[covered_mask], 
                                XX[covered_mask]]).T
    
    uncovered_coords_3D = unwrap_params[uncovered_coords[:,0], 
                                        uncovered_coords[:,1]]
    covered_coords_3D = unwrap_params[covered_coords[:,0], 
                                      covered_coords[:,1]]
    
    """
    build a kNN tree for assigning the remainder. 
    """
    neigh = NearestNeighbors(n_neighbors=1, metric=metric)
    neigh.fit(covered_coords_3D)

    nbr_indices = neigh.kneighbors(uncovered_coords_3D, return_distance=False)
    nbr_indices = nbr_indices.ravel()
    
    """
    look up what labels should be assigned
    """
    labels_new = labels.copy()
    
    nbr_pts = covered_coords[nbr_indices]
    
    labels_new[uncovered_coords[:,0], 
               uncovered_coords[:,1]] = labels[nbr_pts[:,0], nbr_pts[:,1]].copy()
    
    return labels_new



def get_centroids_and_area(labelled):
    
    from skimage.measure import regionprops
    
    regions = regionprops(labelled)
    
    region_areas = np.hstack([re.area for re in regions])
    region_centroids = np.vstack([re.centroid for re in regions])
    
    return region_centroids, region_areas


def parse_cells_from_rect_binary(binary, ksize=1):
    
    from skimage.measure import label 
    
    cell_rect_binary_frame = np.logical_not(skmorph.binary_dilation(binary, skmorph.disk(ksize)))
    
    cell_rect_binary_frame_label = label(cell_rect_binary_frame)
    binary_rect_label_centroids, binary_rect_label_areas = get_centroids_and_area(cell_rect_binary_frame_label)
    
    return cell_rect_binary_frame_label, (binary_rect_label_centroids, binary_rect_label_areas)


def parse_cells_from_polar_binary(binary, ksize=3, valid_factor = 1.1):
    
    from skimage.measure import label 
    
    # create a valid mask ... to exclude border regions. 
    polar_valid_mask_ii, polar_valid_mask_jj = np.indices(binary.shape)
    polar_center = np.hstack(binary.shape[:]) / 2. 

    polar_center_dist = np.sqrt((polar_valid_mask_ii-polar_center[0])**2  + (polar_valid_mask_jj-polar_center[1])**2)
    polar_valid_mask = polar_center_dist <= binary.shape[0]/2./valid_factor
    
    
    cell_polar_binary_frame = np.logical_not(skmorph.binary_dilation(binary, skmorph.square(ksize))) #
    cell_polar_binary_frame = np.logical_and(polar_valid_mask, cell_polar_binary_frame)
    
     # produce the initial labelling. 
    binary_polar_label = label(cell_polar_binary_frame)
    
    # get the areas to flag up super unreliable? 
    binary_polar_label_centroids, binary_polar_label_areas = get_centroids_and_area(binary_polar_label)
    
    return binary_polar_label, (binary_polar_label_centroids, binary_polar_label_areas)

    
def find_embryo_folders(infolder, key):
    
    import os 
    
    folders = os.listdir(infolder)
    folders_found = []
    
    for ff in folders:
        if key in ff:
            folders_found.append(os.path.join(infolder,ff))
    
    if len(folders_found)>0:
        folders_found = np.hstack(folders_found)
        
    return folders_found

def match_folder(query, ref_list):
    
    import os 
    import numpy as np 
    
    look_vect = np.hstack([query in os.path.split(rr)[-1] for rr in ref_list])
    print(look_vect)
    
    return look_vect


def mkdir(directory):
    """ check if directory exists and create it through Python if it does not yet.

    Parameters
    ----------
    directory : str
        the directory path (absolute or relative) you wish to create (or check that it exists)

    Returns
    -------
        void function, no return
    """
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return []
    
if __name__=="__main__":
    
    
# =============================================================================
# =============================================================================
# #     Script takes 1 polar, 1 rect view and turns the polar to rect .... ready for merging of rect to rect.... 
# =============================================================================
# =============================================================================
    
    import numpy as np 
    import pylab as plt 
    import skimage.io as skio 
    import scipy.io as spio 
    import os 
    import pandas as pd 
    import skimage.transform as sktform
    import skimage.morphology as skmorph
    from tqdm import tqdm 
    from skimage.segmentation import find_boundaries
    from tracks3D_tools import radial_smooth
    import glob
    
    date = '2023-02-18'
    binary_ch = 0 # which channel 0 or 1 the binary is in . 
        
    ### Try to do this in an automated manner. 
    
    # annotfolder = 'F:\\Shankar-2\\Control_Embryos_Nodal\\Nodal_Outlining\\Nodal_outling_ver1'
    # annotfolder = 'F:\\Shankar-2\\Control_Embryos_Lefty\\Lefty_fixed_outlines_edit1'
    # annotfolder = 'F:\\Shankar-2\\Control_Embryos_Lefty\\Lefty_fixed_outlines_edit1'
    # annotfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\A_PI DAPI and Actin only\Outline_A_PI DAPI  and Actin only'
    annotfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\B_E5_5 YAP staining\B_5_5_ YAP outlining ver1'
    # annotfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\C_Pre_Induction YAP Staining\C_Pre_Induction YAP Staining outlining ver1'
    
    # find the various embryo folders and use this to get. 
    # allfolders = find_embryo_folders(annotfolder, key='Lit')
    allfolders = find_embryo_folders(annotfolder, key='L')
    # allfolders = find_embryo_folders(annotfolder, key='D4') # for yap 7
    
    
    allfolders = np.sort(allfolders)
    allfolders = np.hstack([ff for ff in allfolders if '.zip' not in ff])
    
    # exclude the only polar annotation only folder. 
    exclude_folder = ['Lit335_Emb3']
    # exclude_folder = ['Lit395_E3']
    # exclude_folder = []
    # allfolders = np.hstack([ff for ff in allfolders if os.path.split(ff)[-1] not in exclude_folder])    
    allfolders = np.hstack([ff for ff in allfolders if exclude_folder[0] not in os.path.split(ff)[-1]])    
    
    # all_emb_folder = 'F:\\Shankar-2\\Control_Embryos_Lefty'
    # all_emb_folder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\A_PI DAPI and Actin only'
    all_emb_folder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\B_E5_5 YAP staining'
    # all_emb_folder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\C_Pre_Induction YAP Staining'
    
    all_emb_unwrap_folders = find_embryo_folders(all_emb_folder, key='_reorient_gui')
    all_emb_unwrap_folders = np.sort(all_emb_unwrap_folders)
    # exclude the .zip
    
    # E3 not good!
    for folder_ii in np.arange(len(allfolders))[2:3]:
        
        emb_folder = allfolders[folder_ii]
        emb_name = os.path.split(emb_folder)[-1]
        
        # find the corresponding emb_name in the proper folder. 
        unwrap_folder = all_emb_unwrap_folders[match_folder(emb_name.split('_outline')[0]+'_', all_emb_unwrap_folders)][0] # to turn into a string. 
        tformfolder = os.path.join(unwrap_folder, 'unwrapped', 'polar_to_rect_tfs')
        
        ###
        # Load the relevant rectfile or polarfile
        ###
        allannotfiles = os.listdir(emb_folder)
        
        """
        channel 0: gray
        channel 1: binary 
        """
        rectfile = [os.path.join(emb_folder, ff) for ff in allannotfiles if 'rect' in ff][0]
        # C:\Users\fyz11\Documents\Work\Projects\Shankar-AVE_Migration\Annotation\Curations\L455_Emb2\2021-03-17
        rect = skio.imread(rectfile); rect = np.squeeze(rect)
        
        polarfile = [os.path.join(emb_folder, ff) for ff in allannotfiles if 'polar' in ff][0]
        # C:\Users\fyz11\Documents\Work\Projects\Shankar-AVE_Migration\Annotation\Curations\L455_Emb2\2021-03-17
        polar = skio.imread(polarfile); polar = np.squeeze(polar)
        
        
        print(rect.shape, polar.shape)
        """
        load the unwrap params. 
        """
        unwrap_params_file = glob.glob(os.path.join(unwrap_folder, 'unwrapped', '*.mat'))[0] 
        # unwrap_params_file = '../Curations/L455_Emb2/L455_Emb2_ve_225_unwrap_params_unwrap_params-geodesic.mat'
        unwrap_obj = spio.loadmat(unwrap_params_file)
        unwrap_params_polar = unwrap_obj['ref_map_polar_xyz']
        # propose to smooth this in order to better solve the growing problem 
        unwrap_params_rect = unwrap_obj['ref_map_rect_xyz']
    
        # # =============================================================================
        # #     Smooth the coordinates. 
        # # =============================================================================
        
        # # did this for the nodal case but don't for lefty! pretty sure it was good. 
        # # radial smoothing parameters for regularizing the surface.
        # smooth_r_dist = 5
        # smooth_theta_dist=100 
        # smooth_radial_dist=20
        # smooth_tip_sigma=25
        # smooth_tip_radius=50
    
        # unwrap_params_polar = radial_smooth(unwrap_params_polar, 
        #                                     r_dist=smooth_r_dist,  # think we should do more? 
        #                                     smooth_theta_dist=smooth_theta_dist, # hm... 
        #                                     smooth_radial_dist=smooth_radial_dist, 
        #                                     smooth_tip_sigma=smooth_tip_sigma, # this fixes the tip?
        #                                     smooth_tip_radius=smooth_tip_radius)
        
        # unwrap_params_polar = radial_smooth(unwrap_params_polar, 
        #                                     r_dist=smooth_r_dist,  # think we should do more? 
        #                                     smooth_theta_dist=smooth_theta_dist, # hm... 
        #                                     smooth_radial_dist=smooth_radial_dist, 
        #                                     smooth_tip_sigma=smooth_tip_sigma, # this fixes the tip?
        #                                     smooth_tip_radius=smooth_tip_radius)
        
    
    # =============================================================================
    #   INPUT 2 : POLAR -> RECT AND RECT -> POLAR TRANSFORMATION FILE FOR THE EMBRYO.
    # =============================================================================
        # tformfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2'
    
        # polar_rect_map_file_000 = os.path.join(tformfolder, 'polar_rect_mapping_L455_Emb2_ve_225_unwrap_params_unwrap_params-geodesic.mat'
    #    polar_rect_map_file_180 = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\LightSheet_Unwrap-params_Analysis\\polar_rect_transformations\\polar_rect_mapping_L491_ve_180_unwrap_params_unwrap_params-geodesic.mat'
        polar_rect_map_file_000_xx = glob.glob(os.path.join(tformfolder, 'polar_rect_mapping_x_*.csv'))[0]
        polar_rect_map_file_000_yy = glob.glob(os.path.join(tformfolder, 'polar_rect_mapping_y*.csv'))[0]
        
        rect_polar_map_file_000_xx = glob.glob(os.path.join(tformfolder, 'rect_polar_mapping_x*.csv'))[0]
        rect_polar_map_file_000_yy = glob.glob(os.path.join(tformfolder, 'rect_polar_mapping_y*.csv'))[0]
        
    #    polar_rect_map_000_rect_polar_ = spio.loadmat(polar_rect_map_file_000)['rect_to_polar_mapping']
    #    polar_rect_map_000_polar_rect_ = spio.loadmat(polar_rect_map_file_000)['polar_to_rect_mapping']
    ##    polar_rect_map_180_rect_polar = spio.loadmat(polar_rect_map_file_180)['rect_to_polar_mapping']
        polar_rect_map_000_polar_rect_ = np.dstack([pd.read_csv(polar_rect_map_file_000_yy), 
                                                    pd.read_csv(polar_rect_map_file_000_xx)]).transpose(2,0,1)
        polar_rect_map_000_rect_polar_ = np.dstack([pd.read_csv(rect_polar_map_file_000_yy), 
                                                    pd.read_csv(rect_polar_map_file_000_xx)]).transpose(2,0,1)
        
        # make a copy
        polar_rect_map_000_rect_polar = polar_rect_map_000_rect_polar_.copy()
        polar_rect_map_000_polar_rect = polar_rect_map_000_polar_rect_.copy()
        
        upsample_factor = 1
        
    # =============================================================================
    #     Transform the original rect to a polar and upscaled? 
    # =============================================================================
    
        distal_dist_polar_keep = 100 # designates the distance around the polar distal tip ids that we don't replace. 
        max_cell_area = 20000 
        # upsample_factor = 1 # how much we need to increase this by ?
        polar_erode_ksize = 5 # whats a good setting? 
        rect_erode_ksize = 3
        min_cell_area = 20 # former 5. 
        
        # # mapping to polar to have a polar version. with upsampling.
        
        # rect2polar_vid = []
        
        # for frame in tqdm(np.arange(len(rect))[:]):
            
        #     binary_img_frame_rect = rect[frame,0].copy()
            
            
        #     # we do and should get the cell segmentation image ... and then we need to do the infilling.... => recomputing the contours -> just smoothing doesn't cut in polar. 
        #     # =============================================================================
        #     #  b) gen individual cell images from rect and polar mapped separately.        
        #     # =============================================================================
        #     cells_rect_label, (cells_rect_centroids, cells_rect_areas) = parse_cells_from_rect_binary(binary_img_frame_rect, 
        #                                                                                               ksize=rect_erode_ksize)
        
        #     # cells_rect_label, (cells_rect_centroids, cells_rect_areas) = parse_cells_from_rect_binary(binary_img_frame_rect, 
        #     #                                                                                           ksize=rect_erode_ksize)
        
            
            
        #     if upsample_factor == 1: 
        #         labels_polar = map_rect_to_polar(cells_rect_label, 
        #                                           polar_rect_map_000_rect_polar_, 
        #                                           target_shape = polar_rect_map_000_polar_rect_.shape[1:] )
                
        #         # labels_polar, (cells_rect_centroids, cells_rect_areas) = parse_cells_from_rect_binary(labels_polar_, 
        #         #                                                                                       ksize=rect_erode_ksize)
        
        #         # labels_polar, (cells_rect_centroids, cells_rect_areas) = parse_cells_from_polar_binary(labels_polar_, 
        #         #                                                                                       ksize=polar_erode_ksize)
        
                
        #         # use this ver. 
        #         valid_mask, cellseg_polar = grow_polar_regions_nearest(labels_polar, 
        #                                                                unwrap_params_polar, 
        #                                                                dilate=5, 
        #                                                                metric='manhattan', 
        #                                                                debug=True)
                            
                
        # #     else:
            
        # #         # applying the resampling. 
        # #         # attempting a larger image. 
        # # #            target_shape = (np.hstack(polar_rect_map_000_polar_rect.shape[1:])).astype(np.int)
        # #         target_shape = (np.hstack(polar_rect_map_000_polar_rect_.shape[1:]) * 2).astype(np.int)
            
        # #         # we need to increase this too. 
        # #         binary_img_frame_rect = sktform.resize(binary_img_frame_rect, ((upsample_factor*polar_rect_map_000_rect_polar_.shape[1], upsample_factor*polar_rect_map_000_rect_polar_.shape[2])), preserve_range=True) > 0
            
        # #         polar_rect_map_000_rect_polar = sktform.resize(polar_rect_map_000_rect_polar_, 
        # #                                                        ((polar_rect_map_000_rect_polar_.shape[0], 
        # #                                                          upsample_factor*polar_rect_map_000_rect_polar_.shape[1], 
        # #                                                          upsample_factor*polar_rect_map_000_rect_polar_.shape[2])), 
        # #                                                        preserve_range=True).astype(np.int)
            
        # #         print(polar_rect_map_000_rect_polar.max(), polar_rect_map_000_rect_polar.min())
            
        # #         binary_img_frame_polar = map_rect_to_polar(binary_img_frame_rect, 
        # #                                                   polar_rect_map_000_rect_polar * upsample_factor, 
        # #                                                   target_shape = target_shape)
                
        #     # # need to close holes. 
        #     # binary_img_frame_polar = skmorph.binary_closing(binary_img_frame_polar>0, skmorph.square(5))
        #     binary_img_frame_polar = find_boundaries(cellseg_polar) 
        #     valid_mask_ = skmorph.binary_erosion(valid_mask, skmorph.disk(3))
        #     binary_img_frame_polar = np.logical_and(binary_img_frame_polar, valid_mask_)
            
        #     plt.figure()
        #     # plt.imshow(cellseg_polar)
        #     plt.imshow(binary_img_frame_polar)
        #     plt.show()
        #     rect2polar_vid.append(binary_img_frame_polar)
            
        # rect2polar_vid = np.array(rect2polar_vid)
    
        # # write out a combined. 
        # skio.imsave('remap_polar2rect.tif', np.uint8(255*rect2polar_vid))
    
    # =============================================================================
    #     Transform the original polar to a rect 
    # =============================================================================
    
        rect = rect[None,:]
        polar = polar[None,:]
        polar2rect_vid = []
        
        
        for frame in tqdm(np.arange(len(polar))[:]):
            
            binary_img_frame_polar = polar[frame,binary_ch].copy()
            
            # parse the cells ... 
            labels_polar, (cells_rect_centroids, cells_rect_areas) = parse_cells_from_polar_binary(binary_img_frame_polar, 
                                                                                                    ksize=polar_erode_ksize)
        
            
            # if upsample_factor == 1: 
            labels_polar_remap = map_rect_to_polar(labels_polar, 
                                                    polar_rect_map_000_polar_rect, 
                                                    target_shape = polar_rect_map_000_rect_polar_.shape[1:] )
        #     else:
            
        #         # applying the resampling. 
        #         # attempting a larger image. 
        # #            target_shape = (np.hstack(polar_rect_map_000_polar_rect.shape[1:])).astype(np.int)
        #         target_shape = (np.hstack(polar_rect_map_000_polar_rect.shape[1:]) * 2).astype(np.int)
            
        #         # we need to increase this too. 
        #         binary_img_frame_rect = sktform.resize(binary_img_frame_rect, ((upsample_factor*binary_img_frame_rect.shape[0], upsample_factor*binary_img_frame_rect.shape[1])), preserve_range=True) > 0
            
        #         polar_rect_map_000_rect_polar = sktform.resize(polar_rect_map_000_rect_polar, ((polar_rect_map_000_rect_polar.shape[0], upsample_factor*polar_rect_map_000_rect_polar.shape[1], upsample_factor*polar_rect_map_000_rect_polar.shape[2])), preserve_range=True).astype(np.int)
            
        #         binary_img_frame_polar = map_rect_to_polar(binary_img_frame_rect, 
        #                                                   polar_rect_map_000_rect_polar * upsample_factor, 
        #                                                   target_shape = target_shape)
        
            cellseg_rect = grow_polar_regions_nearest_rect(labels_polar_remap, 
                                                                        unwrap_params_rect, 
                                                                        metric='euclidean', 
                                                                        debug=True)
        
            # binary_img_frame_polar = find_boundaries(cellseg_polar) 
            binary_img_frame_polar_remap = find_boundaries(cellseg_rect) 
                
            plt.figure()
            plt.imshow(binary_img_frame_polar_remap)
            plt.show()
            
            polar2rect_vid.append(binary_img_frame_polar_remap)
            
            
        savecombfolder = os.path.join(emb_folder, 'Comb_%s' %(date))
        mkdir(savecombfolder)
        
        polar2rect_vid = np.array(polar2rect_vid)[0] # since only one channel. 
        
        savecombfile = os.path.join(savecombfolder, 'remap2polar_' + os.path.split(rectfile)[-1])
        skio.imsave(savecombfile, 
                    np.uint8(255*polar2rect_vid)) # save this out. 




    
    
    