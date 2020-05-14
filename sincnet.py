import math

import numpy as np
import tensorflow as tf
from keras.utils import conv_utils
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras import regularizers


class SincConv1D(L.Layer):
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


class CosFace(L.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[0][-1], self.n_classes),
            initializer='glorot_uniform',
            trainable=True,
            regularizer=self.regularizer
        )

    def call(self, inputs):
        x, y = inputs
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        target_logits = logits - self.m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        })
        return config


def create_layers(options):
    layers = []

    for i in range(options.cnn_n_layers):
        conv = SincConv1D(
            options.cnn_n_filters[0],
            options.cnn_filter_len[0],
            options.sample_rate
        )\
        if i == 0\
        else L.Conv1D(
            options.cnn_n_filters[i],
            options.cnn_filter_len[i],
            strides=1,
            padding='valid'
        )
        layers.append(conv)
        layers.append(L.MaxPooling1D(pool_size=options.cnn_max_pool_len[i]))
        if options.cnn_use_batch_norm[i]:
            layers.append(L.BatchNormalization(momentum=0.95, epsilon=1e-5))
        if options.cnn_use_layer_norm[i]:
            layers.append(L.LayerNormalization(epsilon=1e-6))
        layers.append(L.LeakyReLU(alpha=0.2))
        if options.cnn_drop[i] > 1e-12:
            layers.append(L.Dropout(options.cnn_drop[i]))
    layers.append(L.Flatten())
    for i in range(options.fc_n_layers):
        layers.append(L.Dense(options.fc_size[i], kernel_initializer='he_normal'))
        if options.fc_use_batch_norm[i]:
            layers.append(L.BatchNormalization(momentum=0.95, epsilon=1e-5))
        if options.fc_use_layer_norm[i]:
            layers.append(L.LayerNormalization(epsilon=1e-6))
        layers.append(L.LeakyReLU(alpha=0.2))
        if options.fc_drop[i] > 1e-12:
            layers.append(L.Dropout(options.fc_drop[i]))
    return layers


def create_classifier(options):
    inputs = L.Input(shape=options.input_shape)
    labels = L.Input(shape=[options.n_classes,])

    layers = create_layers(options)

    x = inputs
    for layer in layers:
        x = layer(x)

    head = CosFace(n_classes=options.n_classes)
    prediction = head([x, labels])

    return tf.keras.Model(inputs=[inputs, labels], outputs=prediction)


def create_print_maker(options):
    inputs = L.Input(shape=options.input_shape)

    layers = create_layers(options)

    x = inputs
    for layer in layers:
        x = layer(x)

    head = L.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    prediction = head(x)

    return tf.keras.Model(inputs=inputs, outputs=prediction)