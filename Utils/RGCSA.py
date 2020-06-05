# -*- coding: utf-8 -*-
# @Author  : Peida Wu
# @Time    : 20.3.23 023 9:29:20
# @Function:

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling3D, Dropout, Conv3D, Conv3DTranspose, \
    MaxPooling3D, Conv2D, Reshape, Conv2DTranspose, Cropping3D, Cropping2D, BatchNormalization, Activation, concatenate, \
    multiply, add, Lambda, Multiply, Flatten
from keras.regularizers import l2
from keras.utils import plot_model
from tensorflow.python.keras import backend as K


def space_attention_block(input, filters, kernel_size):
    output_trunk = input

    x = Conv3D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(input)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x_1 = Conv3D(filters, kernel_size=kernel_size, strides=(2, 2, 1), padding='same')(x)
    x_1 = Activation('relu')(x_1)

    x_2 = Conv3D(filters * 2, kernel_size=kernel_size, strides=(2, 2, 1), padding='same')(x_1)
    x_2 = Activation('relu')(x_2)

    x_3 = Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=(2, 2, 1), padding='same')(x_2)
    x_3 = Activation('relu')(x_3)

    x_4 = Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=(2, 2, 1), padding='same')(x_3)
    x_4 = Activation('sigmoid')(x_4)

    output = Multiply()([x_4, x])

    x_add = add([output, output_trunk])

    return x_add


def channel_attention_block(input, filters, kernel_size, padding, reduction_ratio, weight_decay=5e-4):
    x = Conv3D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    squeeze = GlobalAveragePooling3D()(x)

    excitation = Reshape((1, 1, 1, filters))(squeeze)
    excitation = Conv3D(filters=filters // reduction_ratio, kernel_size=1,
                        use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(excitation)
    excitation = Activation('relu')(excitation)
    excitation = Conv3D(filters=filters, kernel_size=1,
                        use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(excitation)
    excitation = Activation('sigmoid')(excitation)

    scale = multiply([x, excitation])

    x_add = add([scale, input])

    return x_add


def __grouped_convolution_block(input, grouped_channels, cardinality, strides,
                                weight_decay=5e-4):
    # TODO 分组卷积
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1  # -1

    group_list = []

    for c in range(cardinality):  # 8组
        # 根据channel维度进行分组
        x = Lambda(lambda z: z[:, :, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :, :])(input)

        # 分组后各自卷积
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=(strides, strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        x_channel = channel_attention_block(x, filters=grouped_channels, kernel_size=3, padding='same',
                                            reduction_ratio=4)

        group_list.append(x_channel)  # 将x存放在列表里，共8个

    # concat拼接
    group_merge = concatenate(group_list, axis=channel_axis)  # 将8个以channel维（最后一维）拼接

    x = BatchNormalization(axis=channel_axis)(group_merge)

    x = Activation('relu')(x)

    return x


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[CONV_DIM1] / residual_shape[CONV_DIM1]))
    stride_height = int(round(input_shape[CONV_DIM2] / residual_shape[CONV_DIM2]))
    stride_depth = int(round(input_shape[CONV_DIM3] / residual_shape[CONV_DIM3]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv3D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1, 1),
                          strides=(stride_width, stride_height, stride_depth),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def __bottleneck_block(input, filters=64, cardinality=8, strides=1,
                       spa_kernel_size=None, spa_attention=True,
                       weight_decay=5e-4):
    # TODO resnext模块

    init = input
    # （9，9，37，24） 128 s=1
    # （9，9，37，128） 256 s=2
    # （5，5，19，256） 512 s=2

    # 分组卷积的个数
    grouped_channels = int(filters / cardinality)  # 128/8=16  256/8=32  512/8=64
    channel_axis = -1

    # TODO 左边分组卷积之前的卷积，使用1x1改变卷积核个数
    x = Conv3D(filters, (1, 1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    # TODO 分组卷积
    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides,
                                    weight_decay)

    if spa_attention:
        x = space_attention_block(x, filters=filters, kernel_size=spa_kernel_size)

    # TODO 左右连接
    x = _shortcut(init, x)

    x = Activation('relu')(x)

    return x


def __initial_conv_block(input, weight_decay=5e-4):
    # 底层tf，channel最后
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1  # -1

    # 3x3x20，24，s=（1，1，5）
    x = Conv3D(32, (3, 3, 7), strides=(1, 1, 2), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), name="init_conv")(input)
    # 输入（11，11，200，1）  输出（9，9，37，24）
    x = BatchNormalization(axis=channel_axis, name="init_BN")(x)
    # x = Activation('relu', name="init_ReLU")(x)

    return x


def __create_res_next(nb_classes, img_input, cardinality=8, weight_decay=5e-4):
    # TODO 网络搭建
    # x_dense = __create_res_next(classes, input, cardinality, weight_decay)

    # 三层模块的滤波器个数
    # filters_list = [64, 128, 256, 512]  # 64, 128, 256, 512
    if cardinality == 6:
        filters_list = [48, 96, 192, 384]
    elif cardinality == 8:
        filters_list = [64, 128, 256, 512]
    elif cardinality == 10:
        filters_list = [80, 160, 320, 640]

    # TODO 初始化卷积层
    x = __initial_conv_block(img_input, weight_decay)

    # TODO 第一个模块
    x_1 = __bottleneck_block(x, filters_list[0], cardinality, strides=1,
                             spa_kernel_size=(3, 3, 1), spa_attention=True,
                             weight_decay=weight_decay)

    # TODO 第二个模块
    x_2 = __bottleneck_block(x_1, filters_list[1], cardinality, strides=2,
                             spa_kernel_size=(3, 3, 1), spa_attention=True,
                             weight_decay=weight_decay)

    # TODO 第三个模块
    x_3 = __bottleneck_block(x_2, filters_list[2], cardinality, strides=2,
                             spa_kernel_size=(1, 1, 1), spa_attention=True,
                             weight_decay=weight_decay)

    # TODO 第四个模块
    x_4 = __bottleneck_block(x_3, filters_list[3], cardinality, strides=2,
                             spa_attention=False,
                             weight_decay=weight_decay)

    x_gap = GlobalAveragePooling3D()(x_4)

    x_dense = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                    kernel_initializer='he_normal', activation='softmax')(x_gap)

    return x_dense


def _handle_dim_ordering():
    # TODO 处理维度
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    CONV_DIM1 = 1
    CONV_DIM2 = 2
    CONV_DIM3 = 3
    CHANNEL_AXIS = 4


def ResneXt_IN(input_shape=None, cardinality=8, weight_decay=5e-4, classes=None):
    # TODO 主函数
    # model = ResneXt_IN((1, 11, 11, 200), cardinality=8, classes=16)

    # 判断底层，tf，channel在最后
    _handle_dim_ordering()

    if len(input_shape) != 4:
        raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

    print('original input shape:', input_shape)
    # orignal input shape（1，11，11，200）

    if K.image_data_format() == 'channels_last':
        input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
    print('change input shape:', input_shape)

    # TODO 数据输入
    input = Input(shape=input_shape)

    # TODO 网络搭建
    x_dense = __create_res_next(classes, input, cardinality, weight_decay)

    model = Model(input, x_dense, name='resnext_IN')

    return model


def main():
    # TODO 程序入口
    # TODO 可变参数3，cardinality
    model = ResneXt_IN((1, 16, 16, 200), cardinality=8, classes=16)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()
    plot_model(model, show_shapes=True, to_file='./model_test.png')


if __name__ == '__main__':
    main()

# 736,064
