from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D, Conv2D
from keras.layers.merge import concatenate
from keras.models import Model

"""
Template for inception v4 with keras 2 API, TF backend
"""


def conv_bn_relu(x, nb_filter, nb_row, nb_col, strides=(1, 1), padding='same', use_bias=False):
    x = Conv2D(nb_filter, (nb_row, nb_col), padding=padding, strides=strides, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def inception_base(inp):
    x = conv_bn_relu(inp, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv_bn_relu(x, 32, 3, 3, padding='valid')
    x = conv_bn_relu(x, 64, 3, 3)

    br1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    br2 = conv_bn_relu(x, 96, 3, 3, strides=(2, 2), padding='valid')

    x = concatenate([br1, br2])

    br1 = conv_bn_relu(x, 64, 1, 1)
    br1 = conv_bn_relu(br1, 96, 3, 3, padding='valid')

    br2 = conv_bn_relu(x, 64, 1, 1)
    br2 = conv_bn_relu(br2, 64, 1, 7)
    br2 = conv_bn_relu(br2, 64, 7, 1)
    br2 = conv_bn_relu(br2, 96, 3, 3, padding='valid')

    x = concatenate([br1, br2])
    br1 = conv_bn_relu(x, 192, 3, 3, strides=(2, 2), padding='valid')
    br2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = concatenate([br1, br2])
    return x


def inception_a(inp):
    br1 = conv_bn_relu(inp, 96, 1, 1)

    br2 = conv_bn_relu(inp, 64, 1, 1)
    br2 = conv_bn_relu(br2, 96, 3, 3)

    br3 = conv_bn_relu(inp, 64, 1, 1)
    br3 = conv_bn_relu(br3, 96, 3, 3)
    br3 = conv_bn_relu(br3, 96, 3, 3)

    br4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inp)
    br4 = conv_bn_relu(br4, 96, 1, 1)

    out = concatenate([br1, br2, br3, br4])
    return out


def inception_b(inp):
    br1 = conv_bn_relu(inp, 384, 1, 1)

    br2 = conv_bn_relu(inp, 192, 1, 1)
    br2 = conv_bn_relu(br2, 224, 1, 7)
    br2 = conv_bn_relu(br2, 256, 7, 1)

    br3 = conv_bn_relu(inp, 192, 1, 1)
    br3 = conv_bn_relu(br3, 192, 7, 1)
    br3 = conv_bn_relu(br3, 224, 1, 7)
    br3 = conv_bn_relu(br3, 224, 7, 1)
    br3 = conv_bn_relu(br3, 256, 1, 7)

    br4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inp)
    br4 = conv_bn_relu(br4, 128, 1, 1)

    out = concatenate([br1, br2, br3, br4])
    return out


def inception_c(inp):
    br1 = conv_bn_relu(inp, 256, 1, 1)

    br2 = conv_bn_relu(inp, 384, 1, 1)
    br2_1 = conv_bn_relu(br2, 256, 1, 3)
    br2_2 = conv_bn_relu(br2, 256, 3, 1)
    br2 = concatenate([br2_1, br2_2])

    br3 = conv_bn_relu(inp, 384, 1, 1)
    br3 = conv_bn_relu(br3, 448, 3, 1)
    br3 = conv_bn_relu(br3, 512, 1, 3)
    br3_1 = conv_bn_relu(br3, 256, 1, 3)
    br3_2 = conv_bn_relu(br3, 256, 3, 1)
    br3 = concatenate([br3_1, br3_2])

    br4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inp)
    br4 = conv_bn_relu(br4, 256, 1, 1)

    out = concatenate([br1, br2, br3, br4])
    return out


def reduction_a(inp):
    br1 = conv_bn_relu(inp, 384, 3, 3, strides=(2, 2), padding='valid')
    br2 = conv_bn_relu(inp, 192, 1, 1)
    br2 = conv_bn_relu(br2, 224, 3, 3)
    br2 = conv_bn_relu(br2, 256, 3, 3, strides=(2, 2), padding='valid')
    br3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inp)
    out = concatenate([br1, br2, br3])
    return out


def reduction_b(inp):
    br1 = conv_bn_relu(inp, 192, 1, 1)
    br1 = conv_bn_relu(br1, 192, 3, 3, strides=(2, 2), padding='valid')
    br2 = conv_bn_relu(inp, 256, 1, 1)
    br2 = conv_bn_relu(br2, 256, 1, 7)
    br2 = conv_bn_relu(br2, 320, 7, 1)
    br2 = conv_bn_relu(br2, 320, 3, 3, strides=(2, 2), padding='valid')
    br3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inp)
    out = concatenate([br1, br2, br3])
    return out


def inception_v4(output_classes=1000, include_top_layer=False, dropout_prob=0.8):
    inp = Input((299, 299, 3))
    x = inception_base(inp)

    # 4 Inception-A blocks
    for _ in range(4):
        x = inception_a(x)
    # Reduction-A
    x = reduction_a(x)

    # 7 Inception-B blocks
    for _ in range(7):
        x = inception_b(x)
    # Reduction-B
    x = reduction_b(x)

    # 3 Inception-C blocks
    for _ in range(3):
        x = inception_c(x)

    # Add pool, dropout, fc.
    if include_top_layer:
        x = AveragePooling2D((8, 8), padding='valid')(x)
        x = Dropout(dropout_prob)(x)
        x = Flatten()(x)
        x = Dense(output_dim=output_classes, activation='softmax')(x)

    return Model(inp, x)
