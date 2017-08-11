from __future__ import division

from keras.layers import Input, Dense, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D, Conv2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.backend import int_shape

"""
Template for ResNet-34, ResNet-101 and ResNet-152 with keras 2 API, TF backend
ResNet-50 exists in keras.applications, can be constructed with layer_structure=[3, 4, 6 ,3], use_bottleneck=True
"""


def bn_relu(inp):
    """
    Basic bn - relu block
    """
    x = BatchNormalization()(inp)
    x = Activation("relu")(x)
    return x


def bn_relu_conv(inp, num_filters, kernel_size, strides=(1, 1), padding="same", kernel_initializer="he_normal"):
    """
    Basic bn - relu - conv block
    """
    x = bn_relu(inp)
    x = Conv2D(num_filters,
               kernel_size,
               strides=strides,
               padding=padding,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2(0.0001))(x)
    return x


def conv_bn_relu(inp, num_filters, kernel_size, strides, padding="same", kernel_initializer="he_normal"):
    """
    Basic conv - bn - relu block
    """
    x = Conv2D(num_filters,
               kernel_size,
               strides=strides,
               padding=padding,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2(0.0001))(inp)
    return bn_relu(x)


def bottleneck_block(inp, num_filters, strides=(1, 1), is_top=False):
    """
    Bottleneck block for deep residual networks
    """

    if is_top:
        x = Conv2D(num_filters,
                   (1, 1),
                   strides=strides,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4))(inp)
    else:
        x = bn_relu_conv(inp, num_filters=num_filters, kernel_size=(1, 1),
                         strides=strides)
    x = bn_relu_conv(x, num_filters=num_filters, kernel_size=(3, 3))
    residual = bn_relu_conv(x, num_filters=num_filters * 4, kernel_size=(1, 1))

    # Adding shortcut
    input_shape = int_shape(inp)
    residual_shape = int_shape(residual)
    input_w, input_h = input_shape[1], input_shape[2]
    residual_w, residual_h = residual_shape[1], residual_shape[2]

    shortcut = inp
    if input_w != residual_w \
            or input_h != residual_h \
            or input_shape[-1] != residual_shape[-1]:
        shortcut = Conv2D(residual_shape[-1], (1, 1),
                          strides=(int(round(input_w / residual_w)), int(round(input_h / residual_h))),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(inp)

    return add([shortcut, residual])


def basic_block(inp, num_filters, strides=(1, 1), is_top=False):
    """
    Basic block for shallower ResNets (< 50 layers)
    """

    if is_top:
        x = Conv2D(num_filters,
                   (3, 3),
                   strides=strides,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4))(inp)
    else:
        x = bn_relu_conv(inp, num_filters=num_filters, kernel_size=(3, 3), strides=strides)
    residual = bn_relu_conv(x, num_filters=num_filters, kernel_size=(3, 3))

    # Adding shortcut
    input_shape = int_shape(inp)
    residual_shape = int_shape(residual)
    input_w, input_h = input_shape[1], input_shape[2]
    residual_w, residual_h = residual_shape[1], residual_shape[2]

    shortcut = inp
    if input_w != residual_w \
            or input_h != residual_h \
            or input_shape[-1] != residual_shape[-1]:
        shortcut = Conv2D(residual_shape[-1], (1, 1),
                          strides=(int(round(input_w / residual_w)), int(round(input_h / residual_h))),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(inp)

    return add([shortcut, residual])


def resnet_constructor(input_shape, layer_structure, use_bottleneck, output_classes=1001, include_top_layer=False):
    """
    Basic constructor for deep residual networks

    Args:
        input_shape: Input shape in CHW
        layer_structure: A list describing the structure of layers
        use_bottleneck: Flag for using bottleneck block instead of basic block for deep architectures
        output_classes: Number of output classes for classification, only meaningful if include_top_layer is True
        include_top_layer: Include final classification layer

    """
    block = basic_block
    if use_bottleneck:
        block = bottleneck_block

    num_filters = 64
    inp = Input(input_shape)
    x = conv_bn_relu(inp, num_filters=num_filters, kernel_size=(7, 7), strides=(2, 2))
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Layered construction of ResNet based the given structure
    for layer, num_layers in enumerate(layer_structure):
        is_first_layer = not layer
        # Construct residual block
        for k in range(num_layers):
            if k == 0 and not is_first_layer:
                strides = (2, 2)
            else:
                strides = (1, 1)
            if k == 0 and is_first_layer:
                x = block(x, num_filters, strides=strides, is_top=True)
            else:
                x = block(x, num_filters, strides=strides, is_top=False)
        num_filters *= 2
    x = bn_relu(x)

    if include_top_layer:
        x = Flatten()(AveragePooling2D((int_shape(x)[1], int_shape(x)[2]), strides=(1, 1))(x))
        x = Dense(output_classes, kernel_initializer="he_normal", activation="softmax")(x)

    return Model(inputs=inp, outputs=x)


# ResNet-34
def resnet_34(input_shape, output_classes=1001):
    return resnet_constructor(input_shape,
                              layer_structure=[3, 4, 6, 3],
                              use_bottleneck=False,
                              output_classes=output_classes)


# ResNet-101
def resnet_101(input_shape, output_classes=1001):
    return resnet_constructor(input_shape,
                              layer_structure=[3, 4, 23, 3],
                              use_bottleneck=True,
                              output_classes=output_classes)


# ResNet-152
def resnet_152(input_shape, output_classes=1001):
    return resnet_constructor(input_shape,
                              layer_structure=[3, 8, 36, 3],
                              use_bottleneck=True,
                              output_classes=output_classes)
