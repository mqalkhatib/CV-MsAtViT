import numpy as np
from keras.layers import Conv2D, Lambda, BatchNormalization, AveragePooling2D, ReLU, concatenate
from keras.layers import Conv3D,  AveragePooling3D

from cvnn.layers import complex_input, ComplexConv2D, ComplexAvgPooling2D, ComplexAvgPooling3D, ComplexConv3D
from cvnn.activations import cart_relu
from keras import backend as K
import tensorflow as tf


def CoordAtt_cmplx(x, reduction = 8):

    def coord_act(x):
        tmpx = cart_relu((x + 3), max_value=6) / 6
        x = x * tmpx
        return x

    x_shape = x.shape.as_list()
    [b, h, w, c] = x_shape
    x_h = ComplexAvgPooling2D(pool_size=(1, w), strides=(1, 1), data_format='channels_last')(x)
    x_w = ComplexAvgPooling2D(pool_size=(h, 1), strides=(1, 1), data_format='channels_last')(x)
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    y = concatenate(inputs=[x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = ComplexConv2D(filters=mip, kernel_size=(1, 1), strides=(1, 1), padding='valid')(y)
    #y = BatchNormalization(trainable=False)(y)
    y = coord_act(y)
    x_h, x_w = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': [h, w]})(y)
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    a_h = ComplexConv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="cart_sigmoid")(x_h)
    a_w = ComplexConv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="cart_sigmoid")(x_w)
    out = x * (a_h * a_w)
    return out


def CoordAtt(x, reduction=32, bn_trainable=False):
    def coord_act(x):
        tmpx = (ReLU(max_value=6)(x + 3)) / 6
        x = x * tmpx
        return x

    x_shape = x.shape.as_list()
    [b, h, w, c] = x_shape
    x_h = AveragePooling2D(pool_size=(1, w), strides=(1, 1), data_format='channels_last')(x)
    x_w = AveragePooling2D(pool_size=(h, 1), strides=(1, 1), data_format='channels_last')(x)
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    y = concatenate(inputs=[x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = Conv2D(filters=mip, kernel_size=(1, 1), strides=(1, 1), padding='valid')(y)
    y = BatchNormalization(trainable=bn_trainable)(y)
    y = coord_act(y)
    x_h, x_w = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': [h, w]})(y)
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    a_h = Conv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="sigmoid")(x_h)
    a_w = Conv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="sigmoid")(x_w)
    out = x * a_h * a_w
    return out


def CoordAtt3D(x, reduction=32, bn_trainable=False):
    def coord_act(x):
        tmpx = (ReLU(max_value=6)(x + 3)) / 6
        x = x * tmpx
        return x

    x_shape = x.shape.as_list()
    [b, h, w, d, c] = x_shape  # 3D shape: [batch, height, width, depth, channels]
    
    # Average pooling along width (w) and depth (d) while keeping height (h)
    x_h = AveragePooling3D(pool_size=(1, w, d), strides=(1, 1, 1), data_format='channels_last')(x)
    
    # Average pooling along height (h) and depth (d) while keeping width (w)
    x_w = AveragePooling3D(pool_size=(h, 1, d), strides=(1, 1, 1), data_format='channels_last')(x)
    
    # Average pooling along height (h) and width (w) while keeping depth (d)
    x_d = AveragePooling3D(pool_size=(h, w, 1), strides=(1, 1, 1), data_format='channels_last')(x)

    # Permute the width and depth pooled features for concatenation
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3, 4])
    x_d = K.permute_dimensions(x_d, [0, 3, 1, 2, 4])
    
    # Concatenate along the spatial and depth dimensions
    y = concatenate(inputs=[x_h, x_w, x_d], axis=1)

    # Apply Conv2D-like operation, but for 3D data, reduce dimensionality using 1x1x1 convolution
    mip = max(8, c // reduction)
    y = Conv3D(filters=mip, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(y)
    y = BatchNormalization(trainable=bn_trainable)(y)
    y = coord_act(y)

    # Split y back into its spatial components
    x_h, x_w, x_d = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': [h, w, d]})(y)
    
    # Permute back to original dimensions
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3, 4])
    x_d = K.permute_dimensions(x_d, [0, 3, 2, 1, 4])

    # Generate the attention maps for each dimension (height, width, depth)
    a_h = Conv3D(filters=c, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', activation="sigmoid")(x_h)
    a_w = Conv3D(filters=c, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', activation="sigmoid")(x_w)
    a_d = Conv3D(filters=c, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', activation="sigmoid")(x_d)

    # Apply the attention weights to the original input
    out = x * (a_h * a_w * a_d)

    return out


def CoordAtt3D_cmplx(x, reduction=32, bn_trainable=False):
    def coord_act(x):
        tmpx = cart_relu((x + 3), max_value=6) / 6
        x = x * tmpx
        return x

    [b, h, w, d, c] = x.shape  # 3D shape: [batch, height, width, depth, channels]
    
    # Average pooling along width (w) and depth (d) while keeping height (h)
    x_h = ComplexAvgPooling3D(pool_size=(1, w, d), strides=(1, 1, 1), data_format='channels_last')(x)
    
    # Average pooling along height (h) and depth (d) while keeping width (w)
    x_w = ComplexAvgPooling3D(pool_size=(h, 1, d), strides=(1, 1, 1), data_format='channels_last')(x)
    
    # Average pooling along height (h) and width (w) while keeping depth (d)
    x_d = ComplexAvgPooling3D(pool_size=(h, w, 1), strides=(1, 1, 1), data_format='channels_last')(x)

    # Permute the width and depth pooled features for concatenation
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3, 4])
    x_d = K.permute_dimensions(x_d, [0, 3, 1, 2, 4])
    
    # Concatenate along the spatial and depth dimensions
    y = concatenate(inputs=[x_h, x_w, x_d], axis=1)

    # Apply Conv3D-like operation, but for 3D data, reduce dimensionality using 1x1x1 convolution
    mip = max(8, c // reduction)
    y = ComplexConv3D(filters=mip, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(y)
    #y = BatchNormalization(trainable=bn_trainable)(y)
    y = coord_act(y)

    # Split y back into its spatial components
    x_h, x_w, x_d = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': [h, w, d]})(y)
    
    # Permute back to original dimensions
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3, 4])
    x_d = K.permute_dimensions(x_d, [0, 3, 2, 1, 4])

    # Generate the attention maps for each dimension (height, width, depth)
    a_h = ComplexConv3D(filters=c, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', activation="cart_sigmoid")(x_h)
    a_w = ComplexConv3D(filters=c, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', activation="cart_sigmoid")(x_w)
    a_d = ComplexConv3D(filters=c, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', activation="cart_sigmoid")(x_d)

    # Apply the attention weights to the original input
    out = x * (a_h * a_w * a_d)

    return out
