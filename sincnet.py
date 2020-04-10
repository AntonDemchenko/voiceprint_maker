import math

import numpy as np
import tensorflow as tf
from keras.utils import conv_utils
from tensorflow.keras import backend as K
from tensorflow.keras import layers


class SincConv1D(tf.keras.layers.Layer):
    def __init__(self, N_filt, Filt_dim, fs, **kwargs):
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

        super(SincConv1D, self).__init__(**kwargs)

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

        self.sinc_1 = SincConv1D(
            options.cnn_N_filt[0], options.cnn_len_filt[0], options.fs
        )
        self.maxpool_1 = layers.MaxPooling1D(pool_size=options.cnn_max_pool_len[0])
        if options.cnn_use_batchnorm[0]:
            self.batch_norm_1 = layers.BatchNormalization(momentum=0.05)
        if options.cnn_use_laynorm[0]:
            self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_1 = layers.LeakyReLU(alpha=0.2)

        self.conv_2 = layers.Conv1D(
            options.cnn_N_filt[1], options.cnn_len_filt[1], strides=1, padding='valid'
        )
        self.maxpool_2 = layers.MaxPooling1D(pool_size=options.cnn_max_pool_len[1])
        if options.cnn_use_batchnorm[1]:
            self.batch_norm_2 = layers.BatchNormalization(momentum=0.05)
        if options.cnn_use_laynorm[1]:
            self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_2 = layers.LeakyReLU(alpha=0.2)

        self.conv_3 = layers.Conv1D(
            options.cnn_N_filt[2], options.cnn_len_filt[2], strides=1, padding='valid'
        )
        self.maxpool_3 = layers.MaxPooling1D(pool_size=options.cnn_max_pool_len[2])
        if options.cnn_use_batchnorm[2]:
            self.batch_norm_3 = layers.BatchNormalization(momentum=0.05)
        if options.cnn_use_laynorm[2]:
            self.layer_norm_3 = layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_3 = layers.LeakyReLU(alpha=0.2)
        self.flatten = layers.Flatten()

        self.dense_4 = layers.Dense(options.fc_lay[0])
        if options.fc_use_batchnorm[0]:
            self.batch_norm_4 = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        if options.fc_use_laynorm[0]:
            self.layer_norm_4 = layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_4 = layers.LeakyReLU(alpha=0.2)

        self.dense_5 = layers.Dense(options.fc_lay[1])
        if options.fc_use_batchnorm[1]:
            self.batch_norm_5 = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        if options.fc_use_laynorm[1]:
            self.layer_norm_5 = layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_5 = layers.LeakyReLU(alpha=0.2)

        self.dense_6 = layers.Dense(options.fc_lay[2])
        if options.fc_use_batchnorm[2]:
            self.batch_norm_6 = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        if options.fc_use_laynorm[2]:
            self.layer_norm_6 = layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_6 = layers.LeakyReLU(alpha=0.2)

        self.prediction = layers.Dense(options.n_classes, activation='softmax')

    def create(self):
        inputs = layers.Input(self.options.input_shape)
        
        x = self.sinc_1(inputs)
        
        x = self.maxpool_1(x)
        if self.options.cnn_use_batchnorm[0]:
            x = self.batch_norm_1(x)
        if self.options.cnn_use_laynorm[0]:
            x = self.layer_norm_1(x)
        x = self.leaky_relu_1(x)

        x = self.conv_2(x)
        x = self.maxpool_2(x)
        if self.options.cnn_use_batchnorm[1]:
            x = self.batch_norm_2(x)
        if self.options.cnn_use_laynorm[1]:
            x = self.layer_norm_2(x)
        x = self.leaky_relu_2(x)

        x = self.conv_3(x)
        x = self.maxpool_3(x)
        if self.options.cnn_use_batchnorm[2]:
            x = self.batch_norm_3(x)
        if self.options.cnn_use_laynorm[2]:
            x = self.layer_norm_3(x)
        x = self.leaky_relu_3(x)
        x = self.flatten(x)

        x = self.dense_4(x)
        if self.options.fc_use_batchnorm[0]:
            x = self.batch_norm_4(x)
        if self.options.fc_use_laynorm[0]:
            x = self.layer_norm_4(x)
        x = self.leaky_relu_4(x)

        x = self.dense_5(x)
        if self.options.fc_use_batchnorm[1]:
            x = self.batch_norm_5(x)
        if self.options.fc_use_laynorm[1]:
            x = self.layer_norm_5(x)
        x = self.leaky_relu_5(x)

        x = self.dense_6(x)
        if self.options.fc_use_batchnorm[2]:
            x = self.batch_norm_6(x)
        if self.options.fc_use_laynorm[2]:
            x = self.layer_norm_6(x)
        x = self.leaky_relu_6(x)

        prediction = self.prediction(x)

        model = tf.keras.Model(inputs=inputs, outputs=prediction)
        return model