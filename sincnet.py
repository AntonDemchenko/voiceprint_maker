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


class SincConvFast(layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        min_low_hz=50,
        min_band_hz=50
    ):
        super().__init__()

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
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, in_channels)
        self.low_hz_ = self.add_weight(
            name='low_hz',
            shape=(self.N_filt,),
            initializer='uniform',
            trainable=True
        )

        # filter frequency band (out_channels, in_channels)
        self.band_hz_ = self.add_weight(
            name='band_hz',
            shape=(self.N_filt,),
            initializer='uniform',
            trainable=True,
        )
        self.set_weights([
            hz[:-1].reshape([-1, 1]),
            np.diff(hz).reshape([-1, 1])
        ])

        linspace = tf.linspace(
            0,
            0.5 * self.kernel_size - 1,
            self.kernel_size // 2
        )
        self.window_ = 0.54 - 0.46 * tf.math.cos(2 * tf.constant(np.pi) * linspace / self.kernel_size)

        n = 0.5 * (self.kernel_size - 1)
        self.n_ = 2 * tf.constant(np.pi) * tf.range(-n, 0).reshape([1, -1]) / self.sample_rate

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
        band_pass_center = 2 * band.reshape([-1, 1])
        band_pass_right = tf.reverse(band_pass_left, axis=1)
        band_pass = tf.concat([band_pass_left, band_pass_center, band_pass_right], axis=1)
        band_pass = band_pass / (2 * band[:, None])

        self.filters = band_pass.reshape([
            self.kernel_size,
            self.in_channels,
            self.out_channels,
        ])

        return K.conv1d(
            x,
            kernel=self.filters,
            stride=self.stride,
            padding=self.padding,
            dilations=self.dilation
        )


class SincConv1D(tf.keras.layers.Layer):
    def __init__(self, N_filt, Filt_dim, fs, **kwargs):
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

        super(SincConv1D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'N_filt': self.N_filt,
            'Filt_dim': self.Filt_dim,
            'fs': self.fs
        })
        return config

    def build(self, input_shape):
        # The filters are trainable parameters.
        self.filt_b1 = self.add_weight(
            name='filt_b1',
            shape=(self.N_filt,),
            initializer='uniform',
            trainable=True
        )
        self.filt_band = self.add_weight(
            name='filt_band',
            shape=(self.N_filt,),
            initializer='uniform',
            trainable=True,
        )

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = 2595 * np.log10(1 + (self.fs / 2) / 700)  # Convert Hz to Mel
        mel_points = np.linspace(
            low_freq_mel, high_freq_mel, self.N_filt
        )  # Equally spaced in Mel scale
        f_cos = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (self.fs / 2) - 100
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1 / self.freq_scale, (b2 - b1) / self.freq_scale])

        super(SincConv1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # filters = K.zeros(shape=(N_filt, Filt_dim))

        # Get beginning and end frequencies of the filters.
        min_freq = 50.0
        min_band = 50.0
        filt_beg_freq = K.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (
            K.abs(self.filt_band) + min_band / self.freq_scale
        )

        # Filter window (hamming).
        n = np.linspace(0, self.Filt_dim, self.Filt_dim)
        window = 0.54 - 0.46 * np.cos(2 * math.pi * n / self.Filt_dim)
        # window = K.cast(window, 'float32')
        # window = tf.Variable(window, name='sincnet_window', trainable=False)

        # TODO what is this?
        t_right_linspace = np.linspace(
            1, (self.Filt_dim - 1) / 2, int((self.Filt_dim - 1) / 2), dtype=np.float32
        )
        t_right = np.float32(t_right_linspace / self.fs)
        # t_right = tf.Variable(t_right, name='sincnet_t_right', trainable=False, dtype=tf.float32)

        # Compute the filters.
        output_list = []
        for i in range(self.N_filt):
            low_pass1 = (
                2 * filt_beg_freq[i] * sinc(filt_beg_freq[i] * self.freq_scale, t_right)
            )
            low_pass2 = (
                2 * filt_end_freq[i] * sinc(filt_end_freq[i] * self.freq_scale, t_right)
            )
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / K.max(band_pass)
            output_list.append(band_pass * window)
        filters = K.stack(output_list)  # (80, 251)
        filters = K.transpose(filters)  # (251, 80)
        filters = K.reshape(
            filters, (self.Filt_dim, 1, self.N_filt)
        )  # (251,1,80) in TF: (filter_width, in_channels, out_channels) in PyTorch (out_channels, in_channels, filter_width)

        """
        Given an input tensor of shape [batch, in_width, in_channels] if data_format is "NWC",
        or [batch, in_channels, in_width] if data_format is "NCW", and a filter / kernel tensor of shape [filter_width, in_channels, out_channels],
        this op reshapes the arguments to pass them to conv2d to perform the equivalent convolution operation.
        Internally, this op reshapes the input tensors and invokes tf.nn.conv2d. For example, if data_format does not start with "NC",
        a tensor of shape [batch, in_width, in_channels] is reshaped to [batch, 1, in_width, in_channels], and the filter is reshaped to
        [1, filter_width, in_channels, out_channels]. The result is then reshaped back to [batch, out_width, out_channels]
        (where out_width is a function of the stride and padding as in conv2d) and returned to the caller.
        """

        out = K.conv1d(x, kernel=filters)

        return out

    def compute_output_shape(self, input_shape):
        new_size = conv_utils.conv_output_length(
            input_shape[1], self.Filt_dim, padding='valid', stride=1, dilation=1
        )
        return (input_shape[0],) + (new_size,) + (self.N_filt,)


def sinc(band, t_right):
    y_right = tf.sin(2 * tf.constant(np.pi, dtype=tf.float32) * band * t_right) / (
        2 * tf.constant(np.pi, dtype=tf.float32) * band * t_right
    )
    # y_left = flip(y_right, 0) TODO remove if useless
    y_left = K.reverse(y_right, 0)
    y = K.cast(
        K.concatenate([y_left, tf.ones(1, dtype=tf.float32), y_right]), 'float32'
    )
    return y


class SincNetModelFactory:
    def __init__(self, options):
        self.options = options

        self.cnn_batch_norm_input = self.make_batch_norm()\
            if options.cnn_use_batchnorm_inp\
            else None

        self.cnn_layer_norm_input = self.make_layer_norm()\
            if options.cnn_use_laynorm_inp\
            else None

        sinc = SincConv1D(
            options.cnn_N_filt[0], options.cnn_len_filt[0], options.fs
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
