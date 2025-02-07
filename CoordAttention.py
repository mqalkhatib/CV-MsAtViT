from keras.layers import  Lambda, concatenate

from cvnn.layers import ComplexConv2D, ComplexAvgPooling2D, ComplexBatchNormalization

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
    y = ComplexBatchNormalization(trainable=False)(y)
    y = coord_act(y)
    x_h, x_w = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': [h, w]})(y)
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    a_h = ComplexConv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="cart_sigmoid")(x_h)
    a_w = ComplexConv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="cart_sigmoid")(x_w)
    out = x * (a_h * a_w)
    return out

