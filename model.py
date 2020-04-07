from keras import layers
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import MaxPooling1D
from keras.models import Model

import sincnet


def getModel(cfg):
    input_shape = (cfg.wlen, 1)
    #
    inputs = Input(input_shape)
    x = sincnet.SincConv1D(
        cfg.cnn_N_filt[0],
        cfg.cnn_len_filt[0],
        cfg.fs
    )(inputs)

    x = MaxPooling1D(pool_size=cfg.cnn_max_pool_len[0])(x)
    if cfg.cnn_use_batchnorm[0]:
        x = BatchNormalization(momentum=0.05)(x)
    if cfg.cnn_use_laynorm[0]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(
        cfg.cnn_N_filt[1],
        cfg.cnn_len_filt[1],
        strides=1,
        padding='valid'
    )(x)
    x = MaxPooling1D(pool_size=cfg.cnn_max_pool_len[1])(x)
    if cfg.cnn_use_batchnorm[1]:
        x = BatchNormalization(momentum=0.05)(x)
    if cfg.cnn_use_laynorm[1]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(
        cfg.cnn_N_filt[2],
        cfg.cnn_len_filt[2],
        strides=1,
        padding='valid'
    )(x)
    x = MaxPooling1D(pool_size=cfg.cnn_max_pool_len[2])(x)
    if cfg.cnn_use_batchnorm[2]:
        x = BatchNormalization(momentum=0.05)(x)
    if cfg.cnn_use_laynorm[2]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)

    # DNN
    x = Dense(cfg.fc_lay[0])(x)
    if cfg.fc_use_batchnorm[0]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    if cfg.fc_use_laynorm[0]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(cfg.fc_lay[1])(x)
    if cfg.fc_use_batchnorm[1]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    if cfg.fc_use_laynorm[1]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(cfg.fc_lay[2])(x)
    if cfg.fc_use_batchnorm[2]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    if cfg.fc_use_laynorm[2]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # DNN final
    prediction = layers.Dense(cfg.out_dim, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.summary()
    return model
