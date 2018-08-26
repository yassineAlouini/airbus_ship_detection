""" A customized Unet model
"""


from keras import layers, models

from asd.conf import IMG_SCALING


def _upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides,
                                  padding=padding)


def _upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)


upsample_dict = {"DECONV": _upsample_conv, "SIMPLE": _upsample_simple}


def build_u_net_model(input_shape, upsample_mode="DECONV", gaussian_noise=0.1,
                      padding="same", net_scaling=None, img_scaling=IMG_SCALING, *args, **kargs):

    upsample = upsample_dict.get(upsample_mode, _upsample_simple)

    input_img = layers.Input(input_shape, name='RGB_Input')
    pp_in_layer = input_img

    # TODO: Add dropout for regularization?

    # Some preprocessing
    # TODO: Add explanation of the different stes.
    if net_scaling is not None:
        pp_in_layer = layers.AvgPool2D(net_scaling)(pp_in_layer)

    pp_in_layer = layers.GaussianNoise(gaussian_noise)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding=padding)(pp_in_layer)
    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding=padding)(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding=padding)(p1)
    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding=padding)(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding=padding)(p2)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding=padding)(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding=padding)(p3)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding=padding)(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding=padding)(p4)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding=padding)(c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding=padding)(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding=padding)(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding=padding)(c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding=padding)(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding=padding)(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding=padding)(c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding=padding)(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding=padding)(u8)
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding=padding)(c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding=padding)(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding=padding)(u9)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding=padding)(c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    # TODO: Why is this commented
    # d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    # d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
    if net_scaling is not None:
        d = layers.UpSampling2D(net_scaling)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])
    if img_scaling is not None:
        fullres_model = models.Sequential()
        fullres_model.add(layers.AvgPool2D(img_scaling,
                                           input_shape=(None, None, 3)))
        fullres_model.add(seg_model)
        fullres_model.add(layers.UpSampling2D(img_scaling))
    else:
        fullres_model = seg_model
    fullres_model.summary()
    return fullres_model
