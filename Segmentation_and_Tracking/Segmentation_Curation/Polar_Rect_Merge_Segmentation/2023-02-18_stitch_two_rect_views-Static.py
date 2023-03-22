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
    # annotfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\A_PI DAPI and Actin only\Outline_A_PI DAPI  and Actin only'
    annotfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\B_E5_5 YAP staining\B_5_5_ YAP outlining ver1'
    # annotfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\C_Pre_Induction YAP Staining\C_Pre_Induction YAP Staining outlining ver1'
    
    
    # find the various embryo folders and use this to get. 
    allfolders = find_embryo_folders(annotfolder, key='Lit')
    # allfolders = find_embryo_folders(annotfolder, key='D4')
    
    allfolders = np.sort(allfolders)
    allfolders = np.hstack([ff for ff in allfolders if '.zip' not in ff])
    
    # exclude the only polar annotation only folder. 
    exclude_folder = ['Lit335_Emb3']
    allfolders = np.hstack([ff for ff in allfolders if os.path.split(ff)[-1] not in exclude_folder])    
    
    # all_emb_folder = 'F:\\Shankar-2\\Control_Embryos_Lefty'
    # all_emb_folder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\A_PI DAPI and Actin only'
    all_emb_folder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\B_E5_5 YAP staining'
    # all_emb_folder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\C_Pre_Induction YAP Staining'
    
    all_emb_unwrap_folders = find_embryo_folders(all_emb_folder, key='_reorient_gui')
    all_emb_unwrap_folders = np.sort(all_emb_unwrap_folders)
    
    # comb_row = [290, 250, 270, 280, 230, 320, 260, 210] # this is tailored. 
    
    
    # # for A. 
    # comb_row = [240, 300, 260, 230]
    
    # for B. 
    # comb_row = [300, 300, 0, 260, 230]
    comb_row = [300, 300, 320, 210, 260, 230]
    # comb_row = [280] #D4
    
    # for the missing #2. 
    
    # for C. 
    # comb_row = [300, 270, 300, 390, 250]
    
    
    for folder_ii in np.arange(len(allfolders))[2:3]: # skip 2  # check 3. 
        
        emb_folder = allfolders[folder_ii]
        emb_name = os.path.split(emb_folder)[-1]
        
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
        rect = skio.imread(rectfile)
        
        polarfile = [os.path.join(emb_folder, ff) for ff in allannotfiles if 'polar' in ff][0]
        # C:\Users\fyz11\Documents\Work\Projects\Shankar-AVE_Migration\Annotation\Curations\L455_Emb2\2021-03-17
        polar = skio.imread(polarfile)
        
        
        """
        load the remapped polar file ? 
        """    
        savecombfolder = os.path.join(emb_folder, 'Comb_%s' %(date))
        savecombfile = os.path.join(savecombfolder, 'remap2polar_' + os.path.split(rectfile)[-1])
        # skio.imsave(savecombfile, 
        #             np.uint8(255*polar2rect_vid)) # save this out. 
        
        rect_polar = skio.imread(savecombfile)
        
        
        """
        combining by stitching. 
        """
        comb_level = comb_row[folder_ii]
        
        comb_binary = rect[binary_ch].copy()
        comb_binary[comb_level:] = rect_polar[comb_level:].copy()

        plt.figure(figsize=(10,10))
        plt.imshow(comb_binary)
        plt.show()
        
        
        # save this out. 
        savecombfolder = os.path.join(emb_folder, 'Comb_%s' %(date))
        savecombfile = os.path.join(savecombfolder, 'merged-rect_' + os.path.split(rectfile)[-1])

        skio.imsave(savecombfile, 
                    np.uint8(comb_binary)) # save this out. 
    
    
    