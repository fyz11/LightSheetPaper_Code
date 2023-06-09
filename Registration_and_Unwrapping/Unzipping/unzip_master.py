# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:01:36 2018

@author: felix

Reusable Functions to do unzipping of embryo.
"""
import numpy as np 

def keep_largest_component(mask3D):
    
    """
    remove small objects and keep the largest connected component
    """
    from skimage.measure import label
    import numpy as np 
    labelled = label(mask3D)
    uniq_regs = np.unique(labelled)[1:]
    areas = [np.sum(labelled == reg) for reg in uniq_regs]
             
    keep = labelled == uniq_regs[np.argmax(areas)]
             
    return keep


def remove_small_objects( mask3D, minsize=100):

    import skimage.morphology as morph
    from scipy.ndimage.morphology import binary_fill_holes
    
    st = []
    
    for v in mask3D:
        st.append(binary_fill_holes(morph.remove_small_objects(v, min_size=minsize)))
        
    return np.array(st)


def segment_embryo(im_array, I_thresh=10, k_size=3, minsize=100, apply_morph=True): # From Unzip_new - includes binary_fill_holes
    
    """
    a quick and dirty 3d segmentation.
    im_array : grayscale 3d volume
    """
    from skimage.morphology import ball, binary_closing, binary_dilation
    from scipy.ndimage.morphology import binary_fill_holes
    
    n_z, n_y, n_x = im_array.shape
    
    mask = im_array >= I_thresh
    mask = remove_small_objects(mask, minsize)
    mask = np.concatenate([binary_fill_holes(m)[None,:] for m in mask], axis=0) 
    
    if apply_morph:
        mask = binary_closing(mask, ball(k_size))
        mask = binary_dilation(mask, ball(k_size))
    mask = np.concatenate([binary_fill_holes(m)[None,:] for m in mask], axis=0) 
    
    return mask


#==============================================================================
#   Taken from Holly's 
#==============================================================================

def fnc_binary(im, ref, thresh_plus):
    ''' This function separates the embryo from the background via Otsu thresholding, setting the background to 0.
        Inputs: 
                i - is the current slice number (0 to n)
                self.use_base - should be the slice number from e.g. in the middle of the stack, or you find a suitable one by print(thresh).
                        It is useful to initially print out 'thresh' and plot the binary output to determine a good threshold slice and base.
                self.im_array: is the array where the images are stored, in order (x,z,y)
        Returns: 
                binary_values - a binary shell image
    '''
    from skimage.filters import threshold_otsu
    
    thresh = threshold_otsu(im[ref]) + thresh_plus
    binary_values = im >= (thresh)
    
    return binary_values 
    

def fnc_close(binary, pad_size=6, k_size=5):
    ''' This function creates a smooth binary mask via closing and filling holes in the thresholded image.
        Inputs: 
                binary
        Outputs: 
                none
        returns: 
                binary_close - the updated smoothed mask
    '''
    from scipy import ndimage
    import numpy as np 
    from skimage.morphology import ball, binary_closing
    
    npad = ((pad_size, 0), (0, 0), (0, 0))     # padding required for 3d closing
    binary = np.pad(binary, pad_width=npad, mode='reflect')
    binary_close = binary_closing(binary, ball(k_size))            # Close
    binary_close = np.delete(binary_close, np.s_[:pad_size], axis=0) # deletes the padding. 
    for i in range(0,len(binary_close)):
        binary_close[i] = ndimage.binary_fill_holes(binary_close[i]).astype(int)   #> 0     # Fill holes axially in each 2d slice.
        
    return binary_close    


def fnc_open(embryo_region, pad_size=7, k_size=3):
    import numpy as np
    from skimage.morphology import ball, binary_closing, binary_opening, binary_erosion

    npad = ((pad_size, 0), (0, 0), (0, 0))
    embryo_region = np.pad(embryo_region, pad_width=npad, mode='reflect')
    binary_open = binary_opening(embryo_region, ball(k_size))            # Close

    binary_open = binary_closing(binary_open, ball(k_size))
    binary_open = np.delete(binary_open, np.s_[:pad_size], axis=0)

    return binary_open


def fnc_remove_small_objects(binary_close, min_percentage_area):
    ''' This function removes small objects from the binary mask, and multiples by the original data. May want to play with min_area.
        Inputs: 
                binary_close
                self.im_array
                self.threshold
        Outputs: 
                none
        Returns: 
                threshold_values, mask_values
    '''
    import numpy as np
    from skimage.measure import label
    from skimage.morphology import remove_small_objects

    for i in range(0,len(binary_close)):
        min_area = (np.size(np.nonzero(binary_close[i]))/10)  # Remove objects smaller than roughly 1/10th of the embryo section
        binary_close[i] = remove_small_objects(binary_close[i], min_size=min_area, connectivity=1, in_place=False)
    
    #tifimsave(os.path.join(self.output_path,'test_3binaryrso.tif'), binary_close.astype(np.uint8))              # Saves binary of whole emb into one tif

    binary_rso = label(binary_close,connectivity=1)   #this labels different connected regions with different integer values
    uniq = np.unique(binary_rso) #find how many unique integers, background is usually first - left to right raster scan of pixels
    
#    print('n_unique', len(uniq))
    #find the area exluding the first unique number
    area_list = []
    for i in uniq[1:]:
        area = np.sum(binary_rso==uniq[i])
        area_list.append(area)
    wanted_id = uniq[1:][np.argmax(area_list)]
    
    embryo_region = binary_rso == wanted_id
    
    return embryo_region

    
def segment_embryo_adaptive(im_array, pad_size, k_size, thresh_plus, min_percentage_area, ref):
    
    import numpy as np 
    
    # these are currently hard coded in specifications? -> need to unwrap these. 
    m = fnc_binary(im_array, ref, thresh_plus)
    m = fnc_close(m, pad_size, k_size)
    m = fnc_remove_small_objects(m, min_percentage_area)
    m = fnc_open(m, pad_size, k_size)
    mask = np.multiply(im_array, m)
        
    return mask 

#==============================================================================
#   Taken from Holly's 
#==============================================================================

def contour_seg_mask(im, mask):
    
    import cv2
    import numpy as np
    im2,contours,heirachy = cv2.findContours(np.uint8(255*mask),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)    # Note: plot the contours not the image output!
    
    # index into image. 
    contours_im = np.zeros_like(im)
    
    if len(contours) > 0:
        
        # find all contours. 
        len_contours = [len(cnt) for cnt in contours]
        contours = np.squeeze(np.array(contours)[np.argmax(len_contours)])

        # index into image. 
        contours_im = np.zeros_like(im)
#        print contours_im.shape
        if len(contours.shape) == 2:
            contours_im[contours[:,1], contours[:,0]] = 1
    
    return contours_im


def contour_seg_embryo(im_array, mask_array):
    
    """
    exploits cv2 contour function to create a mask of the contour. 
    """
    import numpy as np 
    
    return np.array([contour_seg_mask(im_array[i], mask_array[i]) for i in range(len(im_array))])


def segment_contour_embryo(im_array, I_thresh=10, k_size=3, pad_size=7, thresh_plus=-10, min_percentage_area=10, fast_flag=True, ref=None):
    
    """
    if fast_flag: then a simple threshold is used to segment which will contain more holes?
    """
    
    if fast_flag:
        mask_array = segment_embryo(im_array, I_thresh=I_thresh, k_size=k_size)
    else:
        print('adaptive')
        mask_array = segment_embryo_adaptive(im_array, pad_size, k_size, thresh_plus, min_percentage_area, ref=ref) # this is more accurate for capturing the surface. 
    contour_array = contour_seg_embryo(im_array, mask_array)  
    
    return mask_array, contour_array


#==============================================================================
#   Taken from Felix's 
#==============================================================================


def map_centre_line_projection(im_array, center_of_mass, coords, depth=20.0, voxel=5.23):
    
    ''' 
    This is SLOW! -> faster way to get projected intensities along a line? 
    
    Given some sorted coordinates this function max projects them and unwraps them onto a 2D map. 
    
        Parameters are ...
            Input path: string, path to files
            Output path: string, pathe to location in which to save files
            gpu: whether the code is run on GPU or not
        Returns:    images of a the unwrapped embryo surface: 'emb_map' 
                    normalised surface: 'emb_map_normalised'
                    normalised surface with front data centered: 'emb_map_normalised_wrap'       
            
    '''

    import numpy as np 
    n,p,q = im_array.shape    
           
    # Loop over each x (in order of phi) 
    # Stack the pixels and assign the max of them to the original pixel coordinates in a flat array (x,order)

    # in standard (x,y) convention. 
    pt2 = np.take(center_of_mass, [1,2]).astype(np.int)   # This gives the y,z, central points of that slice (assumes a completely vertical embryo)
    max_I = []    
    dist_I = [] 
    
    for xpoi in coords:

        fx = int(xpoi[0]) #x
        fz = int(xpoi[1]) #z 
        fy = int(xpoi[2]) #y
        
        pt1 = np.asarray([fx,fz,fy]).astype(np.int)                    # The coordinates of one pixel on the embryo surface

        """
        Create a line and project ( forsake the production of intermediate image and expensive boolean operations. )
        """
        dist_line = np.linalg.norm(pt1[1:3]-pt2); dist_I.append(dist_line)
        mask_coords = np.array([np.linspace(pt1[1:3][0], pt2[0], 2*int(dist_line)+1),
                                np.linspace(pt1[1:3][1], pt2[1], 2*int(dist_line)+1)]).T
        mask_line = im_array[fx,:,:][mask_coords[:,0].astype(np.int), mask_coords[:,1].astype(np.int)]

        # double check this. should only need the depth. 
        distance_line = np.sqrt(np.sum (np.square (((mask_coords.astype(np.float64))-(pt1[1:3].astype(np.float64)))*[voxel,1]), axis=1))
        mask_distance_less = distance_line<=(depth*voxel)   # Keeps only the values which are less than a given distance
        
        if len(mask_line[mask_distance_less]) > 0: 
            max_pixel = np.mean(mask_line[mask_distance_less]) # mean projection
        else:
            max_pixel = 0
#        max_pixel = np.max(mask_line[mask_distance_less]) # mean projection
        max_I.append(max_pixel)
        
    return np.hstack([coords, np.hstack(dist_I)[:,None], np.hstack(max_I)[:,None]])


def map_spherical_intensities(im_array, center_of_mass, coords, depth=20.0, voxel=5.23):
    
    import numpy as np 
    n,p,q = im_array.shape    
  
    # in standard (x,y) convention. 
    pt2 = center_of_mass.astype(np.int) # set the projection centre.
    max_I = []    
    dist_I = [] 
    
    for xpoi in coords:
        
        pt1 = xpoi.astype(np.int)                    # The coordinates of one pixel on the embryo surface

        """
        Create a line and project ( forsake the production of intermediate image and expensive boolean operations. )
        """
        dist_line = np.linalg.norm(pt1-pt2); dist_I.append(dist_line)
        mask_coords = np.array([np.linspace(pt1[0], pt2[0], 2*int(dist_line)+1),
                                np.linspace(pt1[1], pt2[1], 2*int(dist_line)+1),
                                np.linspace(pt1[2], pt2[2], 2*int(dist_line)+1)]).T
        mask_line = im_array[mask_coords[:,0].astype(np.int), mask_coords[:,1].astype(np.int), mask_coords[:,2].astype(np.int)]

        # double check this. should only need the depth. 
        distance_line = np.sqrt(np.sum (np.square (((mask_coords.astype(np.float64))-(pt1.astype(np.float64)))*[voxel,1,1]), axis=1))
        mask_distance_less = distance_line<=(depth*voxel)   # Keeps only the values which are less than a given distance
        
        if len(mask_line[mask_distance_less]) > 0: 
            max_pixel = np.mean(mask_line[mask_distance_less]) # mean projection
        else:
            max_pixel = 0
#        max_pixel = np.max(mask_line[mask_distance_less]) # mean projection
        max_I.append(max_pixel)
        
    return np.hstack([coords, np.hstack(dist_I)[:,None], np.hstack(max_I)[:,None]])


def fit_spline(x,y, smoothing=None):
    
    from scipy.interpolate import UnivariateSpline
    import numpy as np 
    
    max_y = np.max(y)
    y_ = y/float(max_y)   
    
    if smoothing is None:
        spl = UnivariateSpline(x, y_, s=2*np.std(y_))
    else:
        spl = UnivariateSpline(x, y_, s=smoothing)
        
    interp_s = max_y*spl(x)
    
    return interp_s, int(np.max(interp_s))
    

def compute_cylindrical_statistics(contour_mask, coords, smoothing=None, radial_estimate_start=None):
    
    """
    Compute statistics required for projection 
    """
    
    import numpy as np
    from skimage.morphology import skeletonize
    center = coords.mean(axis=0)
    
    # compute the maximum circumference. 
    axial_s = np.hstack([np.sum(skeletonize(cnt)>0) for cnt in contour_mask]) 
    axial_s_smooth, max_s = fit_spline(np.arange(len(axial_s)), axial_s, smoothing=smoothing)
    
    if radial_estimate_start is not None:
        max_s = np.max(axial_s_smooth[radial_estimate_start:]) # only take from a range onward.
    
    
    axial_phi = np.arctan2(coords[:,2]-center[2],  coords[:,1]-center[1])
    axial_r = np.sqrt(np.sum((coords[:,1:]-center[1:][None,:])**2, axis=1))
    
    min_z, max_z = np.min(coords[:,0]), np.max(coords[:,0])
    min_phi, max_phi = -np.pi, np.pi
    
    # augment the xyz with xyz,phi, r
    coords_ = np.hstack([coords, axial_phi[:,None], axial_r[:,None]])
    
    unwrap_params = {'axial_c':axial_s,
                     'axial_c_smooth':axial_s_smooth,
                     'ranges':[min_phi, min_z, max_phi, max_z],
                     'aspect_ratio':[max_s, max_z-min_z+1],
                     'center':center, 
                     'coords':coords_}

    return unwrap_params


def compute_orthographic_statistics(contour_mask, coords, pole='neg'):
    
    """
    Compute statistics required for projection 
    """
    
    import numpy as np
    from skimage.morphology import skeletonize
    import Geometry.geometry as geom
    
    center = coords.mean(axis=0)

    """
    project into the polar space.
    """
    r,lat,lon = geom.xyz_2_longlat(coords[:,0], coords[:,1], coords[:,2], center=center) # put into geometrical coordinates.
    x_p, y_p, sign = geom.azimuthal_ortho_proj3D([r.ravel(),lat.ravel(), lon.ravel()], pole=pole)
    

    min_x_p, max_x_p = np.min(x_p), np.max(x_p)
    min_y_p, max_y_p = np.min(y_p), np.max(y_p)
    
    # augment the xyz with xyz,phi, r
    coords_ = np.hstack([coords[sign], x_p[:,None], y_p[:,None]])
    
    # we should place the zero point in the center!. 
    largest_x = np.maximum(np.abs(min_x_p), np.abs(max_x_p))
    largest_y = np.maximum(np.abs(min_y_p), np.abs(max_y_p))
    
    min_x_p, max_x_p = -largest_x, largest_x
    min_y_p, max_y_p = -largest_y, largest_y

    unwrap_params = {'pole':pole,
                     'select':sign,
                     'ranges':[min_x_p, min_y_p, max_x_p, max_y_p],
                     'aspect_ratio':[max_x_p-min_x_p+1, max_y_p-min_y_p+1],
                     'center':center, 
                     'coords':coords_}

    return unwrap_params


def compute_stereographic_statistics(contour_mask, coords, pole='neg', max_lims=200, clip=True, clip_lim=0):
    
    import numpy as np 
    import Geometry.geometry as geom
    import pylab as plt 

    center = coords.mean(axis=0)
    
# =============================================================================
#   To Do: implement a pole orientation correction here. 
# =============================================================================
    # demean. 
    x_ = coords[:,0] - center[0]
    y_ = coords[:,1] - center[1]
    z_ = coords[:,2] - center[2]
    r_ = np.sqrt(x_**2+y_**2+z_**2) # find the radial distance to the center. # this is not the signed distance ! (we need the signed distance!)
    
    
    elevation = np.arcsin(z_/r_) # what is the elevation relative to the central angle (longitude)
    
    if pole =='neg':
        # this is south and positive? 
        dist_surface = np.sign(elevation) * r_
    else:
        dist_surface = -np.sign(elevation) * r_
        
    """
    this map seems weird, must try to confine the angle!
    """
    # this is the stereo projection for mapping from an origin point that is the center of the 'sphere'    
    x_p = 2*r_/(r_ + z_) * x_
    y_p = 2*r_/(r_ + z_) * y_
    
    
    if clip:
        select = elevation > clip_lim
        x_p = x_p[select]; 
        y_p = y_p[select];
        r_ = r_[select]
        coords = coords[select]
        dist_surface = dist_surface[select]

    x_p = np.clip(x_p, -max_lims, max_lims)
    y_p = np.clip(y_p, -max_lims, max_lims)
    
    min_x_p, max_x_p = np.min(x_p), np.max(x_p)
    min_y_p, max_y_p = np.min(y_p), np.max(y_p)
    
    # augment the xyz with xyz,phi, r
    coords_ = np.hstack([coords, r_[:,None], dist_surface[:,None], x_p[:,None], y_p[:,None]])
    
    # we should place the zero point in the center!. 
    
    unwrap_params = {'pole':pole,
                     'ranges':[min_x_p, min_y_p, max_x_p, max_y_p],
                     'aspect_ratio':[np.rint(max_x_p-min_x_p+1), np.rint(max_y_p-min_y_p+1)],
                     'center':center, 
                     'coords':coords_}
    
    return unwrap_params


def compute_azimuthal_equidistant_statistics(contour_mask, coords, lat1=0, lon0=0, pole='S', max_lims=200):
    
    import numpy as np 
    import Geometry.geometry as geom
    import pylab as plt 

    center = coords.mean(axis=0)
    
# =============================================================================
#   To Do: implement a pole orientation correction here. (the mapping is only valid for one hemisphere)
# =============================================================================
    # demean. 
    
    r,lat, lon = geom.xyz_2_longlat( coords[:,0], coords[:,1], coords[:,2], center=center)
    x_p, y_p = geom.azimuthal_equidistant([r, lat, lon], lat1=lat1, lon0=lon0, center=center)
    
    if pole=='S':
        select = lat >= 0 
    else:
        select = lat <= 0

    """
    limit mapping to one half of the embryo! as required. 
    """
    x_p = x_p[select] # why is select empty? 
    y_p = y_p[select]
 

    min_x_p, max_x_p = np.min(x_p), np.max(x_p)
    min_y_p, max_y_p = np.min(y_p), np.max(y_p)
    
    # augment the xyz with xyz,phi, r
    coords_ = np.hstack([coords[select], np.abs(r[select])[:,None], x_p[:,None], y_p[:,None]])
    
    # we should place the zero point in the center!. 
    
    unwrap_params = {'pole':pole,
                     'ranges':[min_x_p, min_y_p, max_x_p, max_y_p],
                     'aspect_ratio':[np.rint(max_x_p-min_x_p+1), np.rint(max_y_p-min_y_p+1)],
                     'center':center, 
                     'coords':coords_}
    
    return unwrap_params
    


# =============================================================================
#   given the ref_coord_set, generate a reference 1-1 mapping.  
# =============================================================================
def build_mapping_space(ref_coord_set, ranges=None, shape=None):
    """
    Builds a rectilinear interpolation space.  
    """
    import numpy as np 
    
    if ranges is None:
        uniq_x = np.unique(ref_coord_set[:,0])
        uniq_y = np.unique(ref_coord_set[:,1])
    else:
        uniq_x = [ranges[0], ranges[2]] # x1, x2
        uniq_y = [ranges[1], ranges[3]] # y1, y2
        
    if shape is None:
        x_space = np.linspace(uniq_x[0], uniq_x[-1], len(uniq_x))
        y_space = np.linspace(uniq_y[0], uniq_y[-1], len(uniq_y))
    else:
        x_space = np.linspace(uniq_x[0], uniq_x[1], shape[0])
        y_space = np.linspace(uniq_y[0], uniq_y[1], shape[1])
    
    # build the space. 
    ref_map_x, ref_map_y = np.meshgrid(x_space, y_space)
    ref_space = np.dstack([ref_map_x, ref_map_y])
    
    return ref_space

def shrink_emb_mask(mask, pad=10):
    
    from skimage.morphology import binary_erosion, square
    import numpy as np 
    
    new_mask = []
    
    for m in mask:
        new_mask.append(binary_erosion(m, square(pad))[None,:])
    
    new_mask = np.concatenate(new_mask, axis=0)
    
    return new_mask

def shrink_emb_mask_cylindric_morphology(mask, pad=10):
    
    # this uses morphological means to shrink the mask cylindrically. 
    from skimage.morphology import binary_erosion, square, disk
    import numpy as np 
    
    new_mask = []
    
    for m in mask:
        new_mask.append(binary_erosion(m, disk(pad))[None,:])
    
    new_mask = np.concatenate(new_mask, axis=0)
    
    return new_mask

def shrink_emb_mask_spherical_morphology(mask, pad_size=5, k_size=5):
    
    # this uses 3D morphological parameters to erode the surface down!. 
    from skimage.morphology import binary_erosion, ball
    
    npad = ((pad_size,0), (0,0), (0,0))
    new_mask = np.pad(mask, pad_width=npad, mode='reflect')
    
    new_mask = binary_erosion(new_mask, ball(k_size))
    new_mask = np.delete(new_mask, np.s_[:pad_size], axis=0)
    new_mask = np.multiply(mask, new_mask)
    
    return new_mask


def shrink_emb_mask_cylindric(mask, pad=5, center=None):
    
    if center is None:
        # find the centroid of the surface.
        coords = np.array(np.where(mask > 0)).T
        center = np.mean(coords,axis=0)
        
    points = []
    
    for ii, cnt in enumerate(mask):
        pts = np.array(np.where(cnt>0)).T # this is two digits? 
        if len(pts)>0: 
            vec = center[None,1:] - pts # the first axis is the z axis !
            vec = vec / np.linalg.norm(vec, axis=1)[:,None] # unit displacement vector. 
            pts = pts + pad*vec # move inwards radially.
            z_coords = np.hstack(len(pts)*[ii])[:,None]
            points.append(np.hstack([z_coords,pts]))
#        else:
#            points.append([])
            
    # points (list) -> points (np.array)
#    points = np.vstack([c for c in points if len(c)>0])
    if len(points) == 0:
        print('check mask, no surface points detected')
    else:
        points = np.vstack(points)
    return points


#def gen_ref_map(im_array, ref_coord_set, ref_space, interp_method='cubic'):
#    """
#    Builds a 1-1 mapping from mapping space to xyz geometry space. 
#    
#    ref_coord_set: x,y,z, phi, r, mapped_x, mapped_y
#    
#    """
#    import numpy as np 
#    from scipy.interpolate import griddata
#    
#    # enforce a unique 1-1 mapping by filtering. 
#    uniq_rows = np.arange(ref_space.shape[0])
#    uniq_cols = np.arange(ref_space.shape[1])
#    
#    coords = ref_coord_set[:,:3]
#    mapped_coords = ref_coord_set[:,-2:]
#    distance = ref_coord_set[:,-3] # o i see!!! what is the proper distance here? 
#        
#    m_set = np.hstack([mapped_coords, np.arange(len(mapped_coords))[:,None], distance[:,None]])
#    m_group_row = [m_set[m_set[:,1]==r] for r in uniq_rows] # group by row first. 
#    
#    # create a grid-like structure to contain mappings. 
#    mappings = [] 
#    twoD_2_threeD_mappings = np.zeros((ref_space.shape[0], ref_space.shape[1], 3)) # 2D<->3D mapping
#    
#    for i, m in enumerate(m_group_row):
#        col_sort = [m[m[:,0]==c,:] for c in uniq_cols] # get the intensity.
#        
#        for j, n in enumerate(col_sort):
#            if len(col_sort[j]) == 0: 
#                twoD_2_threeD_mappings[i,j] = np.nan
#            else:
#                # instead of average we should take the outermost coordinates with respect to the central axis. 
#                coords3d = coords[(col_sort[j][:,2]).astype(np.int)]
#                coords3d_axial_dist = col_sort[j][:,3]
#                twoD_2_threeD_mappings[i,j,:] = coords3d[np.argmax(coords3d_axial_dist)] # this should give the furthest coordinate. 
##                twoD_2_threeD_mappings[i,j,:] = np.median( coords3d, axis=0) # gain robustness. 
#        mappings.append(col_sort)
#
#
#    """
#    Interpolate the mapping in order to get a dense mapping. 
#    """
#    val_mapping = np.logical_not(np.isnan(twoD_2_threeD_mappings[:,:,0]))
#    rows, cols = np.indices((twoD_2_threeD_mappings[:,:,0]).shape)
#    
#    rows_ = rows[val_mapping]
#    cols_ = cols[val_mapping]
#    
#    mapped_3d_coords = twoD_2_threeD_mappings[val_mapping,:]
#    
#    # can also do unstructured knn type interpolation. ?
#    interp_x = griddata(np.hstack([cols_[:,None], rows_[:,None]]), mapped_3d_coords[:,0],(cols, rows), method=interp_method, fill_value=0, rescale=False)
#    interp_y = griddata(np.hstack([cols_[:,None], rows_[:,None]]), mapped_3d_coords[:,1],(cols, rows), method=interp_method, fill_value=0, rescale=False)
#    interp_z = griddata(np.hstack([cols_[:,None], rows_[:,None]]), mapped_3d_coords[:,2],(cols, rows), method=interp_method, fill_value=0, rescale=False)
#    
#    # clip to ensure mapping to image space. 
#    interp_x = np.clip(interp_x, 0, im_array.shape[0]-1)
#    interp_y = np.clip(interp_y, 0, im_array.shape[1]-1)
#    interp_z = np.clip(interp_z, 0, im_array.shape[2]-1)
#    
#    # output as an image which is easier to work with and visualise!
#    mapped_3d_coords_interp = np.dstack([interp_x, interp_y, interp_z])
#    
#    return mapped_3d_coords_interp


def gen_ref_map(im_array, ref_coord_set, ref_space, interp_method='cubic', interp_type='rbf', rbf_samples=1000, rbf_basis='multiquadric', rbf_epsilon=20, rbf_smooth=0.5, debug=False):
    """
    Builds a 1-1 mapping from mapping space to xyz geometry space. 
    
    ref_coord_set: x,y,z, phi, r, mapped_x, mapped_y
    
    """
    import numpy as np 
    from scipy.interpolate import griddata
    
    # enforce a unique 1-1 mapping by filtering. 
    uniq_rows = np.arange(ref_space.shape[0])
    uniq_cols = np.arange(ref_space.shape[1])
    
    coords = ref_coord_set[:,:3]
    mapped_coords = ref_coord_set[:,-2:]
    distance = ref_coord_set[:,-3] # o i see!!! what is the proper distance here? which distance is this? => is this trye for all?
        
    m_set = np.hstack([mapped_coords, np.arange(len(mapped_coords))[:,None], distance[:,None]])
    m_group_row = [m_set[m_set[:,1]==r] for r in uniq_rows] # group by row first. 
    
    # create a grid-like structure to contain mappings. 
    mappings = [] 
    twoD_2_threeD_mappings = np.zeros((ref_space.shape[0], ref_space.shape[1], 3)) # 2D<->3D mapping
    
    for i, m in enumerate(m_group_row):
        col_sort = [m[m[:,0]==c,:] for c in uniq_cols] # get the intensity.
        
        for j, n in enumerate(col_sort):
            if len(col_sort[j]) == 0: 
                twoD_2_threeD_mappings[i,j] = np.nan
            else:
                # is this really the best way? 
                # instead of average we should take the outermost coordinates with respect to the central axis. 
                coords3d = coords[(col_sort[j][:,2]).astype(np.int)]
#                coords3d_axial_dist = col_sort[j][:,3]
#                twoD_2_threeD_mappings[i,j,:] = coords3d[np.argmax(coords3d_axial_dist)] # this should give the furthest coordinate. 
                twoD_2_threeD_mappings[i,j,:] = np.median( coords3d, axis=0) # gain robustness. 
        mappings.append(col_sort)

    if debug:
        """
        Visually check the raw mapping.
        """
        import pylab as plt 
        plt.figure(figsize=(10,10))
        plt.suptitle('Non-interpolated mapping')
        plt.subplot(131)
        plt.title('x')
        plt.imshow(twoD_2_threeD_mappings[:,:,0])
        plt.subplot(132)
        plt.title('y')
        plt.imshow(twoD_2_threeD_mappings[:,:,1])
        plt.subplot(133)
        plt.title('z')
        plt.imshow(twoD_2_threeD_mappings[:,:,2])
        plt.show()
    
    """
    Interpolate the mapping in order to get a dense mapping. 
    """
    val_mapping = np.logical_not(np.isnan(twoD_2_threeD_mappings[:,:,0])) # identify all non-mapped. # is there a way to get a score? 
    rows, cols = np.indices((twoD_2_threeD_mappings[:,:,0]).shape)
    
    rows_ = rows[val_mapping]
    cols_ = cols[val_mapping]
    
    mapped_3d_coords = twoD_2_threeD_mappings[val_mapping,:]
    
    if interp_type == 'barycentric':
        """
        Method 1: unstructured barycentric interpolation
        """
        # can also do unstructured knn type interpolation. ?
        interp_x = griddata(np.hstack([cols_[:,None], rows_[:,None]]), mapped_3d_coords[:,0],(cols, rows), method=interp_method, fill_value=0, rescale=False)
        interp_y = griddata(np.hstack([cols_[:,None], rows_[:,None]]), mapped_3d_coords[:,1],(cols, rows), method=interp_method, fill_value=0, rescale=False)
        interp_z = griddata(np.hstack([cols_[:,None], rows_[:,None]]), mapped_3d_coords[:,2],(cols, rows), method=interp_method, fill_value=0, rescale=False)

    if interp_type == 'rbf':
        """
        Method 2: RBF (smooth underlying prior)
        """
        from scipy.interpolate import Rbf
        # random sample fitting points. 
        sampler = np.arange(len(cols_))
        np.random.shuffle(sampler)
        sampler = sampler[:rbf_samples] # whats the maximum number of points?
        
        interp_x_fn = Rbf(cols_[sampler], rows_[sampler], mapped_3d_coords[sampler,0], function=rbf_basis, epsilon=rbf_epsilon, smooth=rbf_smooth)
        interp_y_fn = Rbf(cols_[sampler], rows_[sampler], mapped_3d_coords[sampler,1], function=rbf_basis, epsilon=rbf_epsilon, smooth=rbf_smooth)
        interp_z_fn = Rbf(cols_[sampler], rows_[sampler], mapped_3d_coords[sampler,2], function=rbf_basis, epsilon=rbf_epsilon, smooth=rbf_smooth) 
         
        interp_x = interp_x_fn(cols.ravel(), rows.ravel()).reshape(rows.shape)
        interp_y = interp_y_fn(cols.ravel(), rows.ravel()).reshape(rows.shape)
        interp_z = interp_z_fn(cols.ravel(), rows.ravel()).reshape(rows.shape)
    

    # clip to ensure mapping to valid image space. 
    interp_x = np.clip(interp_x, 0, im_array.shape[0]-1)
    interp_y = np.clip(interp_y, 0, im_array.shape[1]-1)
    interp_z = np.clip(interp_z, 0, im_array.shape[2]-1)
    
    # output as an image which is easier to work with and visualise!
    mapped_3d_coords_interp = np.dstack([interp_x, interp_y, interp_z])
    mapped_3d_coords_interp[np.isnan(mapped_3d_coords_interp)] = 0
        
    return mapped_3d_coords_interp



# generic function to map intensities.         
def map_intensities(mapped_coords, query_I, shape, interp=True, distance=None, uniq_rows=None, uniq_cols=None, min_I=0):
    """
    Given a coordinate mapping map the intensities, by default it should take the distance info else it will take the last. Maximum and mean has issues 

    # filters the mapped coordinates uniquely. 

    """
    from scipy.interpolate import griddata
    import numpy as np 
    
    #==============================================================================
    #   Map the coordinate image and resolve duplicate mappings!   
    #   input mapped_coords should be in (x,y) convention but image is (y,x) convention!. 
    #==============================================================================
    # in image coordinate convention. 
    m_coords = mapped_coords.astype(np.int) # convert to int. 
    
    if uniq_rows is None:
        uniq_rows = np.unique(m_coords[:,1])
    if uniq_cols is None:
        uniq_cols = np.unique(m_coords[:,0])
        
    if distance is None:
        mapped_img = np.zeros((len(uniq_rows), len(uniq_cols)))
        mapped_img[m_coords[:,1], m_coords[:,0]] = query_I # we directly map to an image. using the mapped coordinates. 
    else:
        m_set = np.hstack([m_coords, distance[:,None], query_I[:,None]])
        m_group_row = [m_set[m_set[:,0]==r] for r in uniq_rows] #sort into uniq_z
        
        mapped_img = []
        for m in m_group_row:
            col_sort = [m[m[:,0]==c,-2:] for c in uniq_cols] # get the intensity.
            vals = []
            for c in col_sort:
                if len(c) > 0:
#                    vals.append(c[np.argmax(c[:,-2]), -1]) # take the outermost distance! where distance is the 2nd last column!. 
#                    vals.append(np.mean(c[:, -1])) 
                    vals.append(np.median(c[:,-1])) # take the median value to be more robust. 
                else:
                    vals.append(0) # no intensity
            mapped_img.append(vals)
        mapped_img = np.array(mapped_img)
        
    if interp:
        
        interp_coords = np.array(np.where(mapped_img>min_I)).T
        interp_I = mapped_img[mapped_img>min_I]
        
        # grid interpolation        
        im_shape = mapped_img.shape
        grid_x, grid_y = np.meshgrid(range(im_shape[1]), range(im_shape[0]))
        mapped_image_interp = griddata(interp_coords[:,::-1], interp_I, (grid_x, grid_y), method='linear', fill_value=0)

        return mapped_image_interp, mapped_img

    else:

        return mapped_img


def match_coords_to_ref_space(query_coords, ref_x, ref_y, map_index=[-2,-1]):
    """
    given a ref_map assigns the query coordinates onto the positions of the map using fast NN trees.
        note query_coords are ordered.
    """    
    from sklearn.neighbors import NearestNeighbors
    import numpy as np 
    # define the tree
    neigh_x = NearestNeighbors(n_neighbors=1, leaf_size=2, algorithm='kd_tree', n_jobs=4)
    neigh_y = NearestNeighbors(n_neighbors=1, leaf_size=2, algorithm='kd_tree', n_jobs=4)
    
    query = query_coords[:, map_index]
 
    # fit the nearest neighbour interpolator
    neigh_x.fit(ref_x[:,None])
    neigh_y.fit(ref_y[:,None])
    
    neighbor_index_x = neigh_x.kneighbors(query[:,0][:,None], return_distance=False)
    neighbor_index_y = neigh_y.kneighbors(query[:,1][:,None], return_distance=False)
    
    mapped_coords_row = neighbor_index_y
    mapped_coords_col = neighbor_index_x

    mapped_coords = np.hstack([mapped_coords_col, mapped_coords_row])
    
    return mapped_coords
    
    
def map_coords_to_ref_map(query_coords_order, ref_map):
    """
    given a ref_map assigns the query coordinates onto the positions of the map using fast NN trees.
        note query_coords are ordered.
    """    
    from sklearn.neighbors import NearestNeighbors
    import numpy as np 
    # define the tree
    neigh = NearestNeighbors(n_neighbors=1)
    
    neighbor_index = []
    
    for ii, q in enumerate(query_coords_order):
        if len(q) > 0 : 
            query = q[:,3][:,None] # get the query phi
            ref = ref_map[ii]
            neigh.fit(ref[:,None])
            res = neigh.kneighbors(query, return_distance=False)
            neighbor_index.append(np.hstack([ii*np.ones(len(q))[:,None], res]))

    return neighbor_index


  
