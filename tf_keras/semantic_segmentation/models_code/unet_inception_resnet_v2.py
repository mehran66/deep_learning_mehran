# https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/inception_resnetv2_unet.py

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, ZeroPadding2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_inception_resnetv2_unet(num_classes, encoder, activation):


    """ Encoder """
    s1 = encoder.get_layer("input_1").output  ## (512 x 512)

    s2 = encoder.get_layer("activation").output  ## (255 x 255)
    s2 = ZeroPadding2D(((1, 0), (1, 0)))(s2)  ## (256 x 256)

    s3 = encoder.get_layer("activation_3").output  ## (126 x 126)
    s3 = ZeroPadding2D((1, 1))(s3)  ## (128 x 128)

    s4 = encoder.get_layer("activation_74").output  ## (61 x 61)
    s4 = ZeroPadding2D(((2, 1), (2, 1)))(s4)  ## (64 x 64)

    """ Bridge """
    b1 = encoder.get_layer("activation_161").output  ## (30 x 30)
    b1 = ZeroPadding2D((1, 1))(b1)  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)  ## (512 x 512)

    """ Output """
    dropout = Dropout(0.3)(d4)
    outputs = Conv2D(num_classes, 1, padding="same", activation=activation)(dropout)

    return outputs