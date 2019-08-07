import tensorflow as tf
import sincnet


class SincNetModel(tf.keras.Model):
    def __init__(self, options):
        super(SincNetModel, self).__init__(name='SincNetModel')
        self.options = options

        self.sinc_1 = sincnet.SincConv1D(options.cnn_N_filt[0], options.cnn_len_filt[0], options.fs)
        self.maxpool_1 = tf.keras.layers.MaxPooling1D(pool_size=options.cnn_max_pool_len[0])
        if options.cnn_use_batchnorm[0]:
            self.batch_norm_1 = tf.keras.layers.BatchNormalization(momentum=0.05)
        if options.cnn_use_laynorm[0]:
            self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv_2 = tf.keras.layers.Conv1D(options.cnn_N_filt[1], options.cnn_len_filt[1], strides=1, padding='valid')
        self.maxpool_2 = tf.keras.layers.MaxPooling1D(pool_size=options.cnn_max_pool_len[1])
        if options.cnn_use_batchnorm[1]:
            self.batch_norm_2 = tf.keras.layers.BatchNormalization(momentum=0.05)
        if options.cnn_use_laynorm[1]:
            self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_2 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv_3 = tf.layers.Conv1D(options.cnn_N_filt[2], options.cnn_len_filt[2], strides=1, padding='valid')
        self.maxpool_3 = tf.layers.MaxPooling1D(pool_size=options.cnn_max_pool_len[2])
        if options.cnn_use_batchnorm[2]:
            self.batch_norm_3 = tf.layers.BatchNormalization(momentum=0.05)
        if options.cnn_use_laynorm[2]:
            self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_3 = tf.layers.LeakyReLU(alpha=0.2)
        self.flatten = tf.layers.Flatten()

        self.dense_4 = tf.layers.Dense(options.fc_lay[0])
        if options.fc_use_batchnorm[0]:
            self.batch_norm_4 = tf.layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        if options.fc_use_laynorm[0]:
            self.layer_norm_4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_4 = tf.layers.LeakyReLU(alpha=0.2)

        self.dense_5 = tf.layers.Dense(options.fc_lay[1])
        if options.fc_use_batchnorm[1]:
            self.batch_norm_5 = tf.layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        if options.fc_use_laynorm[1]:
            self.layer_norm_5 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_5 = tf.layers.LeakyReLU(alpha=0.2)

        self.dense_6 = tf.layers.Dense(options.fc_lay[2])
        if options.fc_use_batchnorm[2]:
            self.batch_norm_6 = tf.layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        if options.fc_use_laynorm[2]:
            self.layer_norm_6 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.leaky_relu_6 = tf.layers.LeakyReLU(alpha=0.2)

        self.prediction = tf.layers.Dense(options.out_dim, activation='softmax')

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
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

        x = self.prediction(x)

        return x
