# -*- coding: utf-8 -*-
"""
This is a first attempt at a CNN. 
Started by Holly Hathrell on 20 July 2016.
Last updated on: 06/02/2018 
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python basic_CNN.py
"""

def extract_gray_video_tif(invideo):

    """
    output: imgs: n_imgs x n_rows x n_cols x n_channels stack.
    """    
    from PIL import Image
    import numpy as np     
    img = Image.open(invideo)    
    imgs = []
    read = True
    frame = 0    
    while read:
        try:
            img.seek(frame) # select this as the image
            imgs.append(np.array(img)[None,:])
            frame += 1
        except EOFError:
            # Not enough frames in img
            break        
    return np.concatenate(imgs, axis=0)


def fetch_train_files(all_file_list, slices, key='.mat'):
    
    import os 
    
    slice_nums = []
    want_files = []
    
    for f in all_file_list:
        base = (f.split('/')[-1]).split(key)[0]
        base_splits = base.split('_')
        for b in base_splits:
            if 'slice' in b:
                slice_num = int(b.split('-')[1])
                if slice_num in slices:
                    want_files.append(f)
                slice_nums.append(slice_num)
                break
            
    slice_nums = np.hstack(slice_nums)
    
    return want_files, slice_nums
    
    
def detect_all_files(infolder, key='.mat'):
    
    import os 
    
    all_files = os.listdir(infolder)
    
    files = []
    
    for f in all_files:
        if key in f:
            files.append(os.path.join(infolder, f))
            
    return np.hstack(files)


def random_crops(img, delta_y, delta_x, N = 100, final_size=(256,256)):
    
    from skimage.transform import resize
    
    # Note: image_data_format is 'channel_last'
#    assert img.shape[2] == 3
    nrows, ncols = img.shape[:2] 
    min_size=np.min([nrows, ncols])
    height, width = delta_y, delta_x
    
    if len(delta_y) > 1: 
        dy = np.random.randint(np.max([height[0], 100]), np.min([height[1], int(.9*min_size)]), N)
        dx = np.random.randint(np.max([width[0], 100]), np.min([width[1], int(.9*min_size)]), N)
#        print(dx)
    else:
        dy = height[0]
        dx = width[0]
#    print(height, width)
#    print(dx)
#    print(dy)
    crops = []
    
    if len(delta_y) > 1:
        for ii in range(len(dx)):
            # this is to fit within the image bounds.
            x = np.random.randint(0, img.shape[1] - dx[ii] + 1)
            y = np.random.randint(0, img.shape[0] - dy[ii] + 1)
            crop = img[y:(y+dy[ii]), x:(x+dx[ii]), :]
            crops.append( resize(crop, output_shape=final_size, preserve_range=True))
    else:
        for ii in range(N):
            # this is to fit within the image bounds.
            x = np.random.randint(0, img.shape[1] - dx + 1)
            y = np.random.randint(0, img.shape[0] - dy + 1)
            crop = img[y:(y+dy), x:(x+dx), :]
            crops.append( resize(crop, output_shape=final_size, preserve_range=True))
    return np.array(crops)


def masked_mse(mask_value):
    import keras.backend as K 
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        # in case mask_true is 0 everywhere, the error would be nan, therefore divide by at least 1
        # this doesn't change anything as where sum(mask_true)==0, sum(masked_squared_error)==0 as well
        masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
        return masked_mse
    return f


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    import tensorflow as tf
    import keras.backend as K
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def jaccard_distance_loss(smooth=100):
    def jaccard_distance_fixed(y_true, y_pred):
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
        
        The jaccard distance loss is usefull for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
        gradient.
        
        Ref: https://en.wikipedia.org/wiki/Jaccard_index
        
        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        import keras.backend as K
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth
    return jaccard_distance_fixed

from skimage.segmentation import find_boundaries

w0 = 10
sigma = 5

def make_weight_map(masks):
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.
    
    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)
    
    """
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    print(ix)
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss
    return ZZ

#
#def focal_loss(alpha=0.25, gamma=2):
#    """
#    from https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
#    """
#  import tensorflow as tf
#  eps = 1e-7
#  def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
#    weight_a = alpha * (1 - y_pred) ** gamma * targets
#    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
#    
#    return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 
#
#  def loss(y_true, y_pred):
#    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
#    logits = tf.log(y_pred / (1 - y_pred))
#
#    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
#
#    return tf.reduce_mean(loss)
#
#  return loss


def categorical_focal_loss(gamma=2., alpha=.75):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


from keras.layers import Layer
from keras.initializers import Constant
import tensorflow as tf#
import keras.backend as K
eps = 1e-7
#from keras.activations import softplus

class customFocal_loss_categorical(Layer):
    def __init__(self, gamma=2, alpha=0.75, nb_outputs=3, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.is_placeholder = True
        self.nb_outputs = nb_outputs
        super(customFocal_loss_categorical, self).__init__(**kwargs)

    def focal_loss(self, y_true, y_pred):
        
        weights = K.exp(-self.log_vars) # create the precision matrix. 
        y_pred *= weights
        
        # the rest is normal softmax
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate Focal Loss
        loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy
        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = self.add_weight(name='log_vars', shape=(self.nb_outputs,),
                                              initializer=Constant([0 for ii in range(self.nb_outputs)]), trainable=True)
        super(customFocal_loss_categorical, self).build(input_shape)
        
    def multi_loss(self, ys_true, ys_pred):

        loss = self.focal_loss(y_true=ys_true, y_pred=ys_pred)

        return tf.reduce_mean(loss)
    
    def call(self, inputs):
        # parse the list input. 
        ys_true = inputs[0]
        ys_pred = inputs[1]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

class customFocal_loss_categorical_directionality(Layer):
    def __init__(self, gamma=2, alpha=0.75, nb_outputs=3, nb_tasks=2, losses=None, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.is_placeholder = True
        self.nb_outputs = nb_outputs
        self.nb_tasks = nb_tasks
        self.loss_types = losses 
        super(customFocal_loss_categorical_directionality, self).__init__(**kwargs)

    def weighted_softmax_focal_loss(self, y_true, y_pred, weights):
        
#        weights = K.exp(-self.log_vars) # create the precision matrix. 
        y_pred *= weights
        
        # the rest is normal softmax
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon) # predictions are clipped. 

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate Focal Loss
        loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy
        # Sum the losses in mini_batch, and should this not be mean over the -1.
        return K.sum(loss, axis=-1) ####### previously over the 1 axis
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        # add different initialisations for different types of losses. 
        for i in range(self.nb_tasks):
            if self.loss_types[i] == 'categorical':
                self.log_vars+=[self.add_weight(name='log_vars'+str(i), shape=(self.nb_outputs,),
                                                      initializer=Constant([0. for ii in range(self.nb_outputs)]), trainable=True)]
            if self.loss_types[i] == 'L1':
                self.log_vars+=[self.add_weight(name='log_vars'+str(i), shape=(1,),
                                                      initializer=Constant(0.), trainable=True)]

        super(customFocal_loss_categorical_directionality, self).build(input_shape)
        
    def multi_loss(self, ys_true, ys_pred):       
        loss = 0
        for ii, (y_true, y_pred, log_var) in enumerate(zip(ys_true, ys_pred, self.log_vars)):
            precision = K.exp(-log_var[0]) # 1/sigma**2 this gives positivity.
            if self.loss_types[ii] == 'categorical':
                print(ii, 'categorical')
#                loss += K.mean(self.weighted_softmax_focal_loss(y_true, y_pred, precision))
                loss += self.weighted_softmax_focal_loss(y_true, y_pred, precision)
            if self.loss_types[ii] == 'L1':
                print(ii, 'L1')
                mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
#                y_pred = K.clip(y_pred, -1 + K.epsilon(), 1. - K.epsilon()) #### clip the predictions.
                # prevent division by 0 .
                masked_mae = K.sum(mask_true* precision * K.abs(y_true - y_pred) + log_var[0], axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
                loss += masked_mae #K.mean(masked_mae)
#                loss += K.mean(K.sum(precision * K.abs((y_true - y_pred))  + log_var[0], -1))
        return K.mean(loss)
            
    def call(self, inputs):
        # parse the list input. 
        ys_true = inputs[:self.nb_tasks]
        ys_pred = inputs[self.nb_tasks:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)


class customFocal_loss_mse_directionality(Layer):
    def __init__(self, nb_outputs=3, nb_tasks=2, losses=None, **kwargs):
        self.is_placeholder = True
        self.nb_outputs = nb_outputs
        self.nb_tasks = nb_tasks
        self.loss_types = losses 
        super(customFocal_loss_mse_directionality, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        # add different initialisations for different types of losses. 
        for i in range(self.nb_tasks):
            if self.loss_types[i] == 'L2':
                self.log_vars+=[self.add_weight(name='log_vars'+str(i), shape=(1,),
                                                      initializer=Constant(0.), trainable=True)]
            if self.loss_types[i] == 'L2':
                self.log_vars+=[self.add_weight(name='log_vars'+str(i), shape=(1,),
                                                      initializer=Constant(0.), trainable=True)]
        super(customFocal_loss_mse_directionality, self).build(input_shape)
        
    def multi_loss(self, ys_true, ys_pred):       
        loss = 0
        for ii, (y_true, y_pred, log_var) in enumerate(zip(ys_true, ys_pred, self.log_vars)):
            precision = K.exp(-log_var[0]) # 1/sigma**2 this gives positivity.
            log_var = K.softplus(log_var[0]) # ensure positivity
            if self.loss_types[ii] == 'L2':
                print(ii, 'L2')
                epsilon = K.epsilon()
                y_pred = K.clip(y_pred, epsilon, 1. - epsilon) # make sure this is in the proper range
#                loss += K.mean(self.weighted_softmax_focal_loss(y_true, y_pred, precision))
                loss += K.sum( precision * K.abs(y_true - y_pred) + log_var, axis=-1) 
            if self.loss_types[ii] == 'L1':
                print(ii, 'L1')
                mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
#                y_pred = K.clip(y_pred, -1 + K.epsilon(), 1. - K.epsilon()) #### clip the predictions.
                # prevent division by 0 .
#                masked_mae = K.sum(mask_true* precision * K.abs(y_true - y_pred) + log_var[0], axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
                masked_mae = K.sum(mask_true* precision * K.abs(y_true - y_pred) + log_var, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
                loss += masked_mae #K.mean(masked_mae)
#                loss += K.mean(K.sum(precision * K.abs((y_true - y_pred))  + log_var[0], -1))
        return K.mean(loss)
            
    def call(self, inputs):
        # parse the list input. 
        ys_true = inputs[:self.nb_tasks]
        ys_pred = inputs[self.nb_tasks:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)



def softmax_3d(class_dim=-1):
    """ 3D extension of softmax, class is last dim"""
    import keras.backend as K
    def activation(x):
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=class_dim, keepdims=True))
            s = K.sum(e, axis=class_dim, keepdims=True)
            return e / s
        else:
            raise Exception('Cannot apply softmax to a tensor that is not 2D or '
                            '3D. Here, ndim=' + str(ndim))
    return activation

def categorical_crossentropy_3d(class_dim=-1):
    """2D categorical crossentropy loss
    """
    import keras.backend as K
    def loss(y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=class_dim, keepdims=True)
        # avoid numerical instability with _EPSILON clipping
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cce = -K.sum(K.sum(K.sum(y_true * K.log(y_pred), axis=class_dim), axis=-1), axis=-1)
        return cce
    return loss
#def focal_loss(gamma=2., alpha=.75):
#    import tensorflow as tf
#    gamma = float(gamma)
#    alpha = float(alpha)
#
#    def focal_loss_fixed(y_true, y_pred):
#        """Focal loss for multi-classification
#        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
#        Notice: y_pred is probability after softmax
#        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
#        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
#        Focal Loss for Dense Object Detection
#        https://arxiv.org/abs/1708.02002
#
#        Arguments:
#            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
#            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
#
#        Keyword Arguments:
#            gamma {float} -- (default: {2.0})
#            alpha {float} -- (default: {4.0})
#
#        Returns:
#            [tensor] -- loss.
#        """
#        epsilon = 1.e-9
#        y_true = tf.convert_to_tensor(y_true, tf.float32)
#        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
#
#        model_out = tf.add(y_pred, epsilon)
#        ce = tf.multiply(y_true, -tf.log(model_out))
#        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
#        fl = tf.multiply(alpha, tf.multiply(weight, ce))
#        reduced_fl = tf.reduce_max(fl, axis=1)
#        return tf.reduce_mean(reduced_fl)
#    return focal_loss_fixed
    
def compute_direction_centre_map(binary, normalise=True):
    
    from skimage.measure import label, regionprops
    labelled = label(binary)
    regions = regionprops(labelled)
#    regions = np.unique(labelled)[1:]
    
    y_i, x_i = np.indices(binary.shape)
    out_mask = np.zeros((binary.shape[0], binary.shape[1], 2))
    
    for ii,reg in enumerate(regions):
        center = reg.centroid
        y_reg = y_i[labelled==ii+1]
        x_reg = x_i[labelled==ii+1]
        
        out_mask[y_reg,x_reg,0] = (center[0] - y_reg) #####/ np.float(binary.shape[0])
        out_mask[y_reg,x_reg,1] = (center[1] - x_reg) ########## / np.float(binary.shape[1])
        
    return out_mask

def compute_direction_centre_map_multi(binary, normalise=True):
    
    n_channels = binary.shape[-1]
    
    votes_out = []
    
    for i in range(n_channels):
        votes = compute_direction_centre_map(binary[...,i], normalise=normalise)
        votes_out.append(votes)
        
    votes_out = np.array(votes_out)
    
    return votes_out

"""
Code the dynamic image augmentor based on the videofiles in the respective folders.
"""
import imgaug as ia
from imgaug import augmenters as iaa
import keras
from imgaug.augmentables.heatmaps import HeatmapsOnImage

def pad_image(im, image_dimensions):
    
    new_im = np.zeros((image_dimensions[0], image_dimensions[1], im.shape[-1]),dtype=np.float32)
    new_im[:im.shape[0], :im.shape[1]] = im.copy()
    
    return new_im
    

import numpy as np
import keras


#class DataGenerator(keras.utils.Sequence):
#    'Generates data for Keras'
#    def __init__(self, images_paths, labels_paths, batch_size=1, image_dimensions = (576 ,1024 ,3), shuffle=True, augment=True):
#        self.labels_paths = labels_paths        # array of label paths
#        self.images_paths = images_paths        # array of image paths
#        self.dim          = image_dimensions    # image dimensions
#        self.batch_size   = batch_size          # batch size
#        self.shuffle      = shuffle             # shuffle bool
#        self.augment      = augment             # augment data bool
#        self.n_           = 1
#        self.on_epoch_end()
#
#    def __len__(self):
#        'Denotes the number of batches per epoch'
#        return int(np.floor(len(self.images_paths) / self.batch_size))
#
#    def on_epoch_end(self):
#        'Updates indexes after each epoch'
#        self.indexes = np.arange(len(self.images_paths))
#        if self.shuffle:
#            np.random.shuffle(self.indexes)
#
#    def __getitem__(self, index):
##        print('Generate one batch of data')
#        # selects indices of data for next batch
#        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
##        print()
#        # select data resize and load images
#        labels = [(pad_image(extract_gray_video_tif(self.labels_paths[k]).transpose(1,2,0)[...,1::2], self.dim)/255.).astype(np.float32) for k in indexes]
#        images = [np.uint8(pad_image(extract_gray_video_tif(self.images_paths[k])[0], self.dim)) for k in indexes]        
#        
#        # preprocess and augment data
#        if self.augment == True:
#            # augment both the images and heatmap
#            images, labels = self.augmentor(images, labels)
#        
#        images = np.array([img for img in images])
#        labels = np.array([label for label in labels]) 
#        
##        print(images.shape)
#        
#        # compute the centroid displacement maps from the augmented labels.
#        votes = np.array([compute_direction_centre_map(l>threshold_otsu(l), normalise=True) for l in labels])
#        print('hello')
#        print('votes shape', votes.shape)
#        
#        # reshape for the network
#        images = images.transpose(0,3,1,2)[...,None]
#        labels = labels.transpose(0,3,1,2)[...,None]
##        votes = votes.transpose(0,)
##        print(images.shape)
#        
##        return [images, labels, votes], [None]
#        return (images, labels)
##        yield([images, labels, votes], None)
#    
#    
#    def augmentor(self, images, labels):
#        'Apply data augmentation'
#        
#        # create the augmentation sequence.
#        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#        seq = iaa.Sequential(
#                [
#                # apply the following augmenters to most images
#                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
#                iaa.Flipud(0.5),  # vertically flip 20% of all images
#                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
#                # execute 0 to 5 of the following (less important) augmenters per image
#                # don't execute all of them, as that would often be way too strong
#                iaa.SomeOf((0, 5),
#                           [    iaa.OneOf([
#                                       iaa.GaussianBlur((0, 3.0)),
#                                       # blur images with a sigma between 0 and 3.0
#                                       iaa.AverageBlur(k=(3, 11)),
#                                       # blur image using local means with kernel sizes between 2 and 7
#                                       iaa.MedianBlur(k=(3, 11)),
#                                       # blur image using local medians with kernel sizes between 2 and 7
#                               ]),
#                               iaa.AdditiveGaussianNoise(loc=0,
#                                                         scale=(0.0, 0.01 * 255),
#                                                         per_channel=0.5),
#                               # add gaussian noise to images
#                               iaa.OneOf([
#                                       iaa.Dropout((0.01, 0.05), per_channel=0.5),
#                                       # randomly remove up to 10% of the pixels
#                                       iaa.CoarseDropout((0.01, 0.03),
#                                                         size_percent=(0.01, 0.02),
#                                                         per_channel=0.2),
#                               ]),
#                               # invert color channels
#                               iaa.Add((-20, 20), per_channel=0.5),
#                               # change brightness of images (by -10 to 10 of original value)
#                               iaa.OneOf([
#                                       iaa.Multiply((0.7, 1.2), per_channel=0.5),
#                                       iaa.FrequencyNoiseAlpha(
#                                               exponent=(-1, 0),
#                                               first=iaa.Multiply((0.9, 1.1),
#                                                                  per_channel=True),
#                                               second=iaa.ContrastNormalization(
#                                                       (0.9, 1.1))
#                                       )
#                               ])
#                           ],
#                           random_order=True
#                           )
#                ],
#                random_order=True
#        )
#        # Augment images and heatmaps.
#        images_aug = []
#        heatmaps_aug = []
#        
#        for kk in range(len(images)):
#            im = images[kk]
#            mask = HeatmapsOnImage(labels[kk], shape=im.shape, min_value=0.0, max_value=1.0)
#            
#            # how many times to run for. 
#            for _ in range(self.n_):
#                images_aug_i, heatmaps_aug_i = seq(image=im, heatmaps=mask)
#                images_aug.append(images_aug_i)
#                heatmaps_aug.append(heatmaps_aug_i.get_arr())
#
#        return images_aug, heatmaps_aug

#
#def image_generator(files, batch_size = 64, aug_steps=1, n_augs=1):
def augmentor(images, labels, n_=1):
        'Apply data augmentation'
        
        # create the augmentation sequence.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
                [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 20% of all images
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [    iaa.OneOf([
                                       iaa.GaussianBlur((0, 3.0)),
                                       # blur images with a sigma between 0 and 3.0
                                       iaa.AverageBlur(k=(3, 11)),
                                       # blur image using local means with kernel sizes between 2 and 7
                                       iaa.MedianBlur(k=(3, 11)),
                                       # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.AdditiveGaussianNoise(loc=0,
                                                         scale=(0.0, 0.01 * 255),
                                                         per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                       iaa.Dropout((0.01, 0.05), per_channel=0.5),
                                       # randomly remove up to 10% of the pixels
                                       iaa.CoarseDropout((0.01, 0.03),
                                                         size_percent=(0.01, 0.02),
                                                         per_channel=0.2),
                               ]),
                               # invert color channels
                               iaa.Add((-20, 20), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.OneOf([
                                       iaa.Multiply((0.7, 1.2), per_channel=0.5),
                                       iaa.FrequencyNoiseAlpha(
                                               exponent=(-1, 0),
                                               first=iaa.Multiply((0.9, 1.1),
                                                                  per_channel=True),
                                               second=iaa.ContrastNormalization(
                                                       (0.9, 1.1))
                                       )
                               ])
                           ],
                           random_order=True
                           )
                ],
                random_order=True
        )
        # Augment images and heatmaps.
        images_aug = []
        heatmaps_aug = []
        
        for kk in range(len(images)):
            im = images[kk]
            mask = HeatmapsOnImage(labels[kk], shape=im.shape, min_value=0.0, max_value=1.0)
            
            # how many times to run for. 
            for _ in range(n_):
                images_aug_i, heatmaps_aug_i = seq(image=im, heatmaps=mask)
                images_aug.append(images_aug_i)
                heatmaps_aug.append(heatmaps_aug_i.get_arr())

        return images_aug, heatmaps_aug
    
def image_generator(images_paths, labels_paths, batch_size=1, image_dimensions = (576 ,1024 ,3), shuffle=True, augment=True):
    
    from skimage.transform import resize
    while True:
        
        index = np.arange(len(images_paths))
        np.random.shuffle(index)

        # select data resize and load images
#        labels = [(pad_image(extract_gray_video_tif(labels_paths[k]).transpose(1,2,0)[...,1::2], image_dimensions)/255.).astype(np.float32) for k in index[:batch_size]]
#        images = [np.uint8(pad_image(extract_gray_video_tif(images_paths[k])[0], image_dimensions)) for k in index[:batch_size]]
        labels = [(resize(extract_gray_video_tif(labels_paths[k]).transpose(1,2,0)[...,1::2], image_dimensions[:2], preserve_range=True)/255.).astype(np.float32) for k in index[:batch_size]]
        images = [np.uint8(resize(extract_gray_video_tif(images_paths[k])[0], image_dimensions[:2], preserve_range=True)) for k in index[:batch_size]]
        
        # preprocess and augment data
        if augment == True:
            # augment both the images and heatmap
            images, labels = augmentor(images, labels)
        
        images = np.array([img for img in images])
        labels = np.array([label for label in labels]) 

        # compute the centroid displacement maps from the augmented labels.
        votes = np.array([compute_direction_centre_map_multi(l>threshold_otsu(l), normalise=True) for l in labels])
        
        # reshape for the network
        images = (images.transpose(0,3,1,2)[...,None]/255.).astype(np.float32)
        labels = labels.transpose(0,3,1,2)[...,None]
#        votes = votes.transpose(0,)
#        print(images.max())
#        print(images.shape, labels.shape)
#        print(images.max(), images.min())
#        print(labels.max(), labels.min())
        images = np.clip(images,0,1)
        labels = np.clip(labels,0,1)
        
#        return ([images, labels, votes], None)
        yield([images, labels, votes], None)
    
        
#import cv2 # Installed using 'conda install -c https://conda.binstar.org/menpo opencv3'
import os    

#os.environ['THEANO_FLAGS'] = "device=gpu1"  
  
#import theano
#theano.config.floatX = 'float32'
import numpy as np
import keras
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization, Activation, Add, Input
from keras.layers.merge import Concatenate
from keras.layers.pooling import AveragePooling2D
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tifffile import imsave as tifimsave
from skimage.io import imsave, imread
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, disk 

import scipy.io as spio
#from my_generator_test import image_generator
from skimage.segmentation import find_boundaries
print(keras.__version__)
seed = 1000
np.random.seed(seed)    # robust runs that should have the same result for the same dataset

import glob 
#==============================================================================
# Set the folder locations and make the directory
#date = '2017_06_02/Test1/'  # Month the day to order chronologially in folders
nb_epochs=1000

"""
Specify the root directory to the folder. 
"""
trainfolder = '/media/felix/Elements2/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/training_three_frame/train-2'
valfolder = '/media/felix/Elements2/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/training_three_frame/test-2'

train_images = glob.glob(os.path.join(trainfolder, 'img_*.tif'))
train_labels = np.hstack([os.path.join(trainfolder, os.path.split(t)[-1].replace('img_', 'dst_')) for t in train_images])
test_images = glob.glob(os.path.join(valfolder, 'img_*.tif'))
test_labels = np.hstack([os.path.join(valfolder, os.path.split(t)[-1].replace('img_', 'dst_')) for t in test_images])

print(len(train_images), len(test_images))
# create the two data generators. 
#train_data = DataGenerator(train_images, train_labels, batch_size=1, image_dimensions = (576 ,1024 ,3), shuffle=True, augment=True)
#val_data = DataGenerator(test_images, test_labels, batch_size=1, image_dimensions = (576 ,1024 ,3), shuffle=True, augment=True)

train_data = image_generator(train_images, train_labels, batch_size=1, image_dimensions = (512 ,640 ,3), shuffle=True, augment=True)
val_data = image_generator(test_images, test_labels, batch_size=1, image_dimensions = (512, 640 ,3), shuffle=True, augment=True)

"""
Define the network to use. 
"""
# =============================================================================
# Define the model 
# =============================================================================
# keep it small.jav 
batch = 1 # try a large batch size. 
out_channels = 1 # regression problem this time around. 
#dim_order='th'
#
from attention import Self_Attention2D, UnetGatingSignal, AttnGatingBlock
from keras.layers import ConvLSTM2D, TimeDistributed, Conv3D, Bidirectional, BatchNormalization, Conv3DTranspose

"""
Attempt to build a convLSTM using 
"""
input_shape = (3, 512, 640, 1) # requires a 5D input of (samples, time, rows, cols, channels)
y1_shape = (3, 512, 640, out_channels)
y2_shape = (3, 512, 640, 2)
#c = 16
input_img = Input(input_shape, name='input')
y_true_1 = Input(y1_shape, name='y_true_1')
y_true_2 = Input(y2_shape, name='y_true_2')

x = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', padding='same')(input_img)
c1 = Conv3D(filters=44, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(c1)
x = Activation('relu')(x)
x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
#x = Conv3D(filters=44, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)

x = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', padding='same')(x)
c2 = Conv3D(filters=64, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(c2)
x = Activation('relu')(x)
x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
#x = Conv3D(filters=64, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)

x = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', padding='same')(x)
c3 = Conv3D(filters=72, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(c3)
x = Activation('relu')(x)
x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
#x = Conv3D(filters=72, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)

x = Conv3D(filters=72, kernel_size=(3,3,3), activation='relu', padding='same')(x)
c4 = Conv3D(filters=96, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(c4)
x = Activation('relu')(x)
x = TimeDistributed(Activation('relu'))(x)
x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
#x = Conv3D(filters=96, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)

x = Conv3D(filters=72, kernel_size=(3,3,3), activation='relu', padding='same')(x)
c5 = Conv3D(filters=96, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(c5)
x = Activation('relu')(x)
#x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
#x = Conv3D(filters=96, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)
# upsampling branch.
# [ attn 1 ] # how to extend this to work with TimeDistributed?
#gating = TimeDistributed(UnetGatingSignal(x, is_batchnorm=True))
#attn_1 = TimeDistributed(AttnGatingBlock(c4, gating, 96))
###### time distributed is required to create a u-net type architecture.
x = TimeDistributed(UpSampling2D((2,2)))(x)
#x = Conv3DTranspose(filters=96, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
x = Concatenate(axis=-1)([x,c4])
x = Conv3DTranspose(filters=96, kernel_size=(3,3,3), activation='relu', padding='same')(x)
x = Conv3DTranspose(filters=72, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Activation('relu'))(x)

x = TimeDistributed(UpSampling2D((2,2)))(x)
#x = Conv3DTranspose(filters=96, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
x = Concatenate(axis=-1)([x,c3])
x = Conv3DTranspose(filters=96, kernel_size=(3,3,3), activation='relu', padding='same')(x)
x = Conv3DTranspose(filters=72, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Activation('relu'))(x)

x = TimeDistributed(UpSampling2D((2,2)))(x)
#x = Conv3DTranspose(filters=72, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
x = Concatenate(axis=-1)([x,c2])
x = Conv3DTranspose(filters=72, kernel_size=(3,3,3), activation='relu', padding='same')(x)
x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Activation('relu'))(x)

x = TimeDistributed(UpSampling2D((2,2)))(x)
#x = Conv3DTranspose(filters=44, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
x = Concatenate(axis=-1)([x,c1])
x = Conv3DTranspose(filters=44, kernel_size=(3,3,3), activation='relu', padding='same')(x)
x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(x)
#x = TimeDistributed(Activation('relu'))(x)

# predict the softmax output at all timepoints.... # should i increase this?
output1 = TimeDistributed(Activation('relu'))(x)
#output1 = Conv3D(filters=out_channels, kernel_size=(1,1,1), activation=softmax_3d(-1), padding='same')(output1)
output1 = Conv3D(filters=out_channels, kernel_size=(1,1,1), activation='relu', padding='same')(output1)
output2 = Conv3D(filters=16, kernel_size=(3,3,3), activation='linear', padding='same')(x) #### tanh is not working. 
output2 = Conv3D(filters=16, kernel_size=(3,3,3), activation='linear', padding='same')(output2)
output2 = Conv3D(filters=2, kernel_size=(1,1,1), activation='linear', padding='same')(output2)  

prediction_model = Model(inputs=[input_img], outputs=[output1, output2]) # classification branch, centroid regression branch.

y_pred1, y_pred2 = prediction_model(input_img) # gives two outputs. 
##focus loss model. 
output = customFocal_loss_mse_directionality(nb_outputs=out_channels,
                                             nb_tasks=2,
                                             losses=['L2','L1'])([y_true_1, y_true_2, y_pred1, y_pred2])

#output = customFocal_loss_categorical_directionality(gamma=2, alpha=0.75, 
#                                                     nb_outputs=out_channels,
#                                                     nb_tasks=2,
#                                                     losses=['categorical','L1'])([y_true_1, y_true_2, y_pred1, y_pred2])

#x = TimeDistributed(Conv2D(out_channels, (1, 1), activation="softmax", padding="same"))
final_model = Model(inputs=[input_img, y_true_1, y_true_2], outputs=[output])

final_model.summary()

# train the model using Adam + MSE 
opt = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08)


final_model.compile(loss=None, optimizer=opt, metrics=['accuracy'])

# early stopping
EStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min', )


learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
            		                                            patience=5,
            		                                            verbose=1,
            		                                            factor=0.5,
            		                                            min_lr=1e-9)

# checkpoint
checkpoint = keras.callbacks.ModelCheckpoint('cell-seg-unet-Conv3D-full-L1-direction-mae_mae-more_aug-v2-1.h5',
                                            #'cell-seg-unet-Conv3D-full-custom-L2-direction-mae_mae-more_aug.h5', 
                                             verbose=1, monitor='val_loss',save_best_only=True, save_weights_only=True, mode='min')  


#final_model.fit(train_x[...,None], train_y[...,None], 
#                 batch_size = batch, 
#                 epochs=nb_epochs, 
#                 validation_data=(test_x[...,None], test_y[...,None]), 
#                callbacks = [EStop, checkpoint], shuffle=True, verbose = 1)

batch = 1 # try 1.... 
#final_model.fit(train_x[:], train_y[:], 
#                 batch_size = batch, 
#                 epochs=nb_epochs, 
#                 validation_data=(test_x[:], test_y[:]), 
#                 callbacks = [EStop, checkpoint], shuffle=True, verbose = 1)
#final_model.fit([train_x[:], train_y[...,:out_channels], train_y[..., out_channels:]], None,
#                 batch_size = batch, 
#                 epochs=nb_epochs, 
#                 validation_data=([test_x[:], test_y[...,:out_channels], test_y[...,out_channels:]], None), 
#                 callbacks = [EStop, checkpoint], shuffle=True, verbose = 1)


# train on data
history = final_model.fit_generator(generator=train_data,
                                   validation_data=val_data,
                                   epochs=nb_epochs,
                                   steps_per_epoch=10*len(train_labels),
                                   validation_steps = 10*len(test_labels),
                                   callbacks=[learning_rate_reduction, EStop, checkpoint],
                                   verbose=1
                                   )


input_shape = (3, 512, 1024, 1) # requires a 5D input of (samples, time, rows, cols, channels)
y1_shape = (3, 512, 1024, out_channels)
y2_shape = (3, 512, 1024, 2)
#c = 16
input_img = Input(input_shape, name='input')
y_true_1 = Input(y1_shape, name='y_true_1')
y_true_2 = Input(y2_shape, name='y_true_2')

x = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', padding='same')(input_img)
c1 = Conv3D(filters=44, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(c1)
x = Activation('relu')(x)
x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
#x = Conv3D(filters=44, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)

x = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', padding='same')(x)
c2 = Conv3D(filters=64, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(c2)
x = Activation('relu')(x)
x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
#x = Conv3D(filters=64, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)

x = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', padding='same')(x)
c3 = Conv3D(filters=72, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(c3)
x = Activation('relu')(x)
x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
#x = Conv3D(filters=72, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)

x = Conv3D(filters=72, kernel_size=(3,3,3), activation='relu', padding='same')(x)
c4 = Conv3D(filters=96, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(c4)
x = Activation('relu')(x)
x = TimeDistributed(Activation('relu'))(x)
x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
#x = Conv3D(filters=96, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)

x = Conv3D(filters=72, kernel_size=(3,3,3), activation='relu', padding='same')(x)
c5 = Conv3D(filters=96, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(c5)
x = Activation('relu')(x)
#x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
#x = Conv3D(filters=96, kernel_size=(1,2,2), strides=(1,2,2), activation='linear', padding='same')(x)
# upsampling branch.
# [ attn 1 ] # how to extend this to work with TimeDistributed?
#gating = TimeDistributed(UnetGatingSignal(x, is_batchnorm=True))
#attn_1 = TimeDistributed(AttnGatingBlock(c4, gating, 96))
###### time distributed is required to create a u-net type architecture.
x = TimeDistributed(UpSampling2D((2,2)))(x)
#x = Conv3DTranspose(filters=96, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
x = Concatenate(axis=-1)([x,c4])
x = Conv3DTranspose(filters=96, kernel_size=(3,3,3), activation='relu', padding='same')(x)
x = Conv3DTranspose(filters=72, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Activation('relu'))(x)

x = TimeDistributed(UpSampling2D((2,2)))(x)
#x = Conv3DTranspose(filters=96, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
x = Concatenate(axis=-1)([x,c3])
x = Conv3DTranspose(filters=96, kernel_size=(3,3,3), activation='relu', padding='same')(x)
x = Conv3DTranspose(filters=72, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Activation('relu'))(x)

x = TimeDistributed(UpSampling2D((2,2)))(x)
#x = Conv3DTranspose(filters=72, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
x = Concatenate(axis=-1)([x,c2])
x = Conv3DTranspose(filters=72, kernel_size=(3,3,3), activation='relu', padding='same')(x)
x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Activation('relu'))(x)

x = TimeDistributed(UpSampling2D((2,2)))(x)
#x = Conv3DTranspose(filters=44, kernel_size=(1,1,1), strides=(1,2,2), activation='relu', padding='same')(x)
x = Concatenate(axis=-1)([x,c1])
x = Conv3DTranspose(filters=44, kernel_size=(3,3,3), activation='relu', padding='same')(x)
x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), activation='linear', padding='same')(x)
x = TimeDistributed(BatchNormalization())(x)
#x = TimeDistributed(Activation('relu'))(x)

# predict the softmax output at all timepoints.... # should i increase this?
output1 = TimeDistributed(Activation('relu'))(x)
#output1 = Conv3D(filters=out_channels, kernel_size=(1,1,1), activation=softmax_3d(-1), padding='same')(output1)
output1 = Conv3D(filters=out_channels, kernel_size=(1,1,1), activation='relu', padding='same')(output1)
output2 = Conv3D(filters=16, kernel_size=(3,3,3), activation='linear', padding='same')(x) #### tanh is not working. 
output2 = Conv3D(filters=16, kernel_size=(3,3,3), activation='linear', padding='same')(output2)
output2 = Conv3D(filters=2, kernel_size=(1,1,1), activation='linear', padding='same')(output2)  

prediction_model = Model(inputs=[input_img], outputs=[output1, output2]) # classification branch, centroid regression branch.

y_pred1, y_pred2 = prediction_model(input_img) # gives two outputs. 
##focus loss model. 
output = customFocal_loss_mse_directionality(nb_outputs=out_channels,
                                             nb_tasks=2,
                                             losses=['L2','L1'])([y_true_1, y_true_2, y_pred1, y_pred2])

#output = customFocal_loss_categorical_directionality(gamma=2, alpha=0.75, 
#                                                     nb_outputs=out_channels,
#                                                     nb_tasks=2,
#                                                     losses=['categorical','L1'])([y_true_1, y_true_2, y_pred1, y_pred2])

#x = TimeDistributed(Conv2D(out_channels, (1, 1), activation="softmax", padding="same"))
final_model = Model(inputs=[input_img, y_true_1, y_true_2], outputs=[output])

final_model.summary()
final_model.load_weights('cell-seg-unet-Conv3D-full-L1-direction-mae_mae-more_aug-v2-1.h5')

infer_model = prediction_model


# =============================================================================
# now apply. 
# =============================================================================

# load test images. 
"""
load a second set for validation of loss. 
"""
from skimage.transform import resize
import hdbscan 
from skimage.color import label2rgb

clusterer = hdbscan.HDBSCAN(min_cluster_size=150)

# take another dataset as the val dataset.
annotfolder = '/media/felix/Elements2/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L871_Emb2_masks-2/'
#annotfolder = '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L871_Emb1_masks-2/'
imgfolder = '/media/felix/Elements2/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/tiffs'
annotfiles = glob.glob(os.path.join(annotfolder, '*.tif'))

# parse the time points and the image file from annot files
imgs_test = []
#masks_test = []

for a in annotfiles[:1]:
    imgfname = '_'.join(os.path.split(a)[-1].replace('_bleachcorr', '').split('_')[:-1]) + '.tif'
#    imgfname = '_'.join(os.path.split(a)[-1].split('_')[:-1]) + '.tif'
#    tp = int(os.path.split(a)[-1].split('_')[-1].split('.tif')[0])
    
    vidfile = os.path.join(imgfolder, imgfname)
    vid = extract_gray_video_tif(vidfile)
    
    for tp in range(len(vid)):
        vidslice = vid[tp]
        
        imgs_test.append(vidslice)
#    masks_test.append(imread(a).transpose(1,2,0))
imgs_test = np.array(imgs_test)

for kk in range(len(imgs_test)-3)[-5:]:
    test_id = kk
    #pad_shape = (np.ceil(np.array(imgs_test[test_id].shape) / 8. ) * 8.).astype(np.int)
    from skimage.filters import gaussian 
    
    in_im = np.zeros((512,1024,3))
#    in_im[:imgs_test[test_id].shape[0], 
#          :imgs_test[test_id].shape[1]] = gaussian(imgs_test[test_id], .5)
#    in_im[:imgs_test[test_id].shape[0], 
#          :imgs_test[test_id].shape[1],:] = gaussian(imgs_test[test_id:test_id+3].transpose(1,2,0), 1)
    in_im[:imgs_test[test_id].shape[0], 
          :imgs_test[test_id].shape[1],:] = imgs_test[test_id:test_id+3].transpose(1,2,0)
    in_im = in_im[:,:,::-1] # reverse the order. 
    in_im = rescale_intensity(in_im / 255.).astype(np.float32)
    
    #in_im = resize(imgs_test[test_id], (1024,1024))
    #in_im = in_im/255.
    #in_im.max()
#    test_out = infer_model.predict(in_im[None,:,:,None], batch_size=batch)[...,:]
    test_out = infer_model.predict(in_im.transpose(2,0,1)[None,:][...,None], batch_size=batch)
    test_cells, test_votes = test_out # split the two channel outputs. 
    
    # 1.) parse the cell segmentation information. 
    test_cells = np.squeeze(test_cells); # take only the cell channel? 
    test_cells = test_cells[:,:imgs_test[test_id].shape[0],
                            :imgs_test[test_id].shape[1]]; 
    
    # 2.) parse the votes segmentation information to get the instance segmentation 
    test_votes = np.squeeze(test_votes);
    test_votes = test_votes[:,:imgs_test[test_id].shape[0],
                            :imgs_test[test_id].shape[1]]
    
    # threshold the binary and get votes. 
#    cell_binary = test_cells[1] >= threshold_otsu(test_cells[1].ravel()) # for masking the votes. 
    cell_binary = test_cells[1] >= 0.2
#    cell_binary = test_cells[1,...,1] >= 0.6
    mask_y, mask_x = np.indices(cell_binary.shape)
    
    pred_y = test_votes[1,...,0] + mask_y
    pred_x = test_votes[1,...,1] + mask_x 
    
    pred_mask = np.zeros(mask_y.shape, dtype=np.uint16)
    pts = np.vstack([pred_y[cell_binary>0], 
                     pred_x[cell_binary>0]]).T
    pred_cells = clusterer.fit_predict(pts) # return the integer masks
#    pred_cells[pred_cells==-1] = 0
    
    pred_mask[mask_y[cell_binary>0], 
              mask_x[cell_binary>0]] = pred_cells + 1 
    pred_mask = label2rgb(pred_mask, bg_label=0)
    
    fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(20,10))
    ax[0,0].imshow(imgs_test[test_id])
    ax[0,1].imshow(imgs_test[test_id+1])
    ax[0,2].imshow(imgs_test[test_id+2])
    
    ax[1,0].imshow(test_cells[0])
    ax[1,1].imshow(test_cells[1])
    ax[1,2].imshow(test_cells[2])
    
    plt.figure()
    plt.imshow(pred_mask)
    plt.show()
#    plt.imshow(np.squeeze(test_out)[...,1], cmap='coolwarm', alpha=1)
#    plt.imshow(np.squeeze(test_out), cmap='coolwarm', alpha=1)
    plt.show()
    





#"""
#Test OPTICS clustering from sklearn. 
#"""
#mask_test = extract_gray_video_tif(annotfiles[0]) 
#mask_binary = mask_test[1] > threshold_otsu(mask_test[1])
#mask_y, mask_x = np.indices(mask_binary.shape)
#
## compute the direction centre_map
#mask_directions = compute_direction_centre_map(mask_binary)
#
#centres_y = mask_y + mask_directions[...,0]
#centres_x = mask_x + mask_directions[...,1]
#
#plt.figure(figsize=(15,15))
#plt.imshow(mask_binary)
#plt.plot(centres_x[mask_binary==1].ravel(), 
#         centres_y[mask_binary==1].ravel(), '.')
#plt.show()
#
#
## cluster the points
#pts = np.vstack([centres_x[mask_binary==1].ravel(), 
#                 centres_y[mask_binary==1].ravel()]).T
#
#from sklearn.cluster import OPTICS    
#clust = OPTICS(min_samples=5, xi=.05, min_cluster_size=.05)
#
## Run the fit
#clust.fit(pts)
#cells = clust.labels_







    






















#compare_mask = np.dstack([masks_test[test_id][...,1] > threshold_otsu(masks_test[test_id][...,1]), 
#                          test_out[...,1], 
#                          np.zeros_like(masks_test[test_id][...,1])])




