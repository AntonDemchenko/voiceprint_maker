import math

import numpy as np
import tensorflow as tf
from keras.utils import conv_utils
from tensorflow.keras import backend as K
from tensorflow.keras import layers


def to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


class SincConv(layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding='valid',
        dilation=1,
        min_low_hz=50,
        min_band_hz=50
    ):
        if in_channels != 1:
            raise ValueError('SinConv supports only one input channel')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        super().__init__()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'sample_rate': self.sample_rate,
            'min_low_hz': self.min_low_hz,
            'min_band_hz': self.min_band_hz
        })
        return config

    def build(self, input_shape):
        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = 0.5 * self.sample_rate - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            to_mel(low_hz),
            to_mel(high_hz),
            self.out_channels + 1
        )
        hz = to_hz(mel)

        # filter lower frequency (out_channels, in_channels)
        self.low_hz_ = self.add_weight(
            name='low_hz',
            shape=(self.out_channels, 1),
            initializer='uniform',
            trainable=True
        )

        # filter frequency band (out_channels, in_channels)
        self.band_hz_ = self.add_weight(
            name='band_hz',
            shape=(self.out_channels, 1),
            initializer='uniform',
            trainable=True,
        )
        self.set_weights([
            hz[:-1].reshape([-1, 1]),
            np.diff(hz).reshape([-1, 1])
        ])

        linspace = np.linspace(
            0,
            self.kernel_size // 2,
            self.kernel_size // 2
        )
        self.window_ = 0.54 - 0.46 * np.cos(2 * np.pi * linspace / self.kernel_size)

        n = 0.5 * (self.kernel_size - 1)
        self.n_ = 2 * np.pi * np.arange(-n, 0).reshape([1, -1]) / self.sample_rate
        self.n_ = self.n_.astype(np.float32)

        super().build(input_shape)

    def __call__(self, x):
        low = self.min_low_hz + tf.math.abs(self.low_hz_)
        high = tf.clip_by_value(
            low + self.min_band_hz + tf.math.abs(self.band_hz_),
            self.min_low_hz,
            0.5 * self.sample_rate
        )
        band = (high - low)[:, 0]

        f_times_t_low = tf.linalg.matmul(low, self.n_)
        f_times_t_high = tf.linalg.matmul(high, self.n_)

        band_pass_left = self.window_ * ((tf.sin(f_times_t_high) - tf.sin(f_times_t_low) / (0.5 * self.n_)))
        band_pass_center = tf.reshape(2 * band, [-1, 1])
        band_pass_right = tf.reverse(band_pass_left, axis=[1])
        band_pass = tf.concat([band_pass_left, band_pass_center, band_pass_right], axis=1)
        band_pass = band_pass / (2 * band[:, None])

        self.filters = tf.reshape(
            band_pass,
            [self.kernel_size, self.in_channels, self.out_channels]
        )

        return K.conv1d(
            x,
            self.filters,
            self.stride,
            self.padding,
            dilation_rate=self.dilation
        )

    def compute_output_shape(self, input_shape):
        new_size = conv_utils.conv_output_length(
            input_shape[1],
            self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation
        )
        return (input_shape[0],) + (new_size,) + (self.out_channels,)


class SincNetModelFactory:
    def __init__(self, options):
        self.options = options

        self.cnn_batch_norm_input = self.make_batch_norm()\
            if options.cnn_use_batchnorm_inp\
            else None

        self.cnn_layer_norm_input = self.make_layer_norm()\
            if options.cnn_use_laynorm_inp\
            else None

        sinc = SincConv(
            out_channels=options.cnn_N_filt[0],
            kernel_size=options.cnn_len_filt[0],
            sample_rate=options.fs
        )

        self.abs = layers.Lambda(lambda x: tf.math.abs(x))

        self.n_conv_layers = len(options.cnn_N_filt)
        self.conv = [sinc] + [
            layers.Conv1D(
                options.cnn_N_filt[i],
                options.cnn_len_filt[i],
                strides=1,
                padding='valid'
            )
            for i in range(1, self.n_conv_layers)
        ]
        self.maxpool = [
            layers.MaxPooling1D(pool_size=options.cnn_max_pool_len[i])
            for i in range(self.n_conv_layers)
        ]
        self.cnn_batch_norm = [
            self.make_batch_norm()
            if options.cnn_use_batchnorm[i]
            else None
            for i in range(self.n_conv_layers)
        ]
        self.cnn_layer_norm = [
            self.make_layer_norm()
            if options.cnn_use_laynorm[i]
            else None
            for i in range(self.n_conv_layers)
        ]
        self.cnn_activations = [
            self.make_activation(options.cnn_act[i])
            for i in range(self.n_conv_layers)
        ]
        self.cnn_dropout = [
            layers.Dropout(options.cnn_drop[i])
            if not np.isclose(options.cnn_drop[i], 0)
            else None
            for i in range(self.n_conv_layers)
        ]

        self.flatten = layers.Flatten()

        self.fc_batch_norm_input = self.make_batch_norm()\
            if options.fc_use_batchnorm_inp\
            else None

        self.fc_layer_norm_input = self.make_layer_norm()\
            if options.fc_use_laynorm_inp\
            else None

        self.n_dense_layers = len(options.fc_lay)
        self.dense = [
            layers.Dense(options.fc_lay[i])
            for i in range(self.n_dense_layers)
        ]
        self.fc_batch_norm = [
            self.make_batch_norm()
            if options.fc_use_batchnorm[i]
            else None
            for i in range(self.n_dense_layers)
        ]
        self.fc_layer_norm = [
            self.make_layer_norm()
            if options.fc_use_laynorm[i]
            else None
            for i in range(self.n_dense_layers)
        ]
        self.fc_activations = [
            self.make_activation(options.fc_act[i])
            for i in range(self.n_dense_layers)
        ]
        self.fc_dropout = [
            layers.Dropout(options.fc_drop[i])
            if not np.isclose(options.fc_drop[i], 0)
            else None
            for i in range(self.n_dense_layers)
        ]

    def make_layer_norm(self):
        return layers.LayerNormalization(epsilon=1e-6)

    def make_batch_norm(self):
        return layers.BatchNormalization(momentum=0.95, epsilon=1e-5)

    def make_activation(self, act):
        if act == 'leaky_relu':
            return layers.LeakyReLU(alpha=0.2)
        if act == 'relu':
            return layers.ReLU()

    def get_prediction(self, x):
        raise NotImplementedError

    def create(self):
        inputs = layers.Input(self.options.input_shape)

        x = inputs
        if self.cnn_batch_norm_input:
            x = self.cnn_batch_norm_input(x)
        if self.cnn_layer_norm_input:
            x = self.cnn_layer_norm_input(x)

        self.conv[0].build(self.options.input_shape) # SincConv build is not called automatically

        for i in range(self.n_conv_layers):
            x = self.conv[i](x)
            if i == 0 and self.cnn_layer_norm[i]:
                x = self.abs(x)
            x = self.maxpool[i](x)
            if self.cnn_batch_norm[i]:
                x = self.cnn_batch_norm[i](x)
            if self.cnn_layer_norm[i]:
                x = self.cnn_layer_norm[i](x)
            x = self.cnn_activations[i](x)
            if self.cnn_dropout[i]:
                x = self.cnn_dropout[i](x)

        x = self.flatten(x)

        if self.fc_batch_norm_input:
            x = self.fc_batch_norm_input(x)
        if self.fc_layer_norm_input:
            x = self.fc_layer_norm_input(x)

        for i in range(self.n_dense_layers):
            x = self.dense[i](x)
            if self.fc_batch_norm[i]:
                x = self.fc_batch_norm[i](x)
            if self.fc_layer_norm[i]:
                x = self.fc_layer_norm[i](x)
            x = self.fc_activations[i](x)
            if self.fc_dropout[i]:
                x = self.fc_dropout[i](x)

        prediction = self.get_prediction(x)

        model = tf.keras.Model(inputs=inputs, outputs=prediction)
        return model


class SincNetClassifierFactory(SincNetModelFactory):
    def __init__(self, options):
        super().__init__(options)
        self.class_batch_norm = self.make_batch_norm()\
            if options.class_use_batchnorm_inp\
            else None
        self.class_layer_norm = self.make_layer_norm()\
            if options.class_use_laynorm_inp\
            else None
        self.class_layer = layers.Dense(self.options.n_classes, activation='softmax')

    def get_prediction(self, x):
        if self.class_batch_norm:
            x = self.class_batch_norm(x)
        if self.class_layer_norm:
            x = self.class_layer_norm(x)
        x = self.class_layer(x)
        return x


class SincNetPrintMakerFactory(SincNetModelFactory):
    def __init__(self, options):
        super().__init__(options)

    def get_prediction(self, x):
        x = layers.Dense(self.options.out_dim)(x)
        x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        return x
