import io
import os

import numpy as np
import soundfile as sf
import tensorflow as tf
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K
K.clear_session()

import conf
from test import Validation


def batchGenerator(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, fact_amp, out_dim):
    while True:
        sig_batch, lab_batch = create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, fact_amp, out_dim)
        yield sig_batch, lab_batch


def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, fact_amp, out_dim):
    # Initialization of the minibatch (batch_size, [0=>x_t, 1=>x_t+N, 1=>random_samp])
    sig_batch = np.zeros([batch_size, wlen])
    lab_batch = []
    snt_id_arr = np.random.randint(N_snt, size=batch_size)
    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)
    for i in range(batch_size):
        # select a random sentence from the list
        # [fs, signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        # signal=signal.astype(float)/32768
        fname = data_folder + wav_lst[snt_id_arr[i]]
        with tf.io.gfile.GFile(fname, 'rb') as f:
            [signal, fs] = sf.read(io.BytesIO(f.read()))
        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len-wlen - 1)  # randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg+wlen
        sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
        y = lab_dict[wav_lst[snt_id_arr[i]]]
        yt = to_categorical(y, num_classes=out_dim)
        lab_batch.append(yt)
    a, b = np.shape(sig_batch)
    sig_batch = sig_batch.reshape((a, b, 1))
    return sig_batch, np.array(lab_batch)


class ValidationCallback(Callback):
    def __init__(self, Batch_dev, data_folder, lab_dict, wav_lst_te, wlen, wshift, class_lay):
        self.wav_lst_te = wav_lst_te
        self.data_folder = data_folder
        self.wlen = wlen
        self.wshift = wshift
        self.lab_dict = lab_dict
        self.Batch_dev = Batch_dev
        self.class_lay = class_lay
    
    def on_epoch_end(self, epoch, logs={}):
        val = Validation(self.Batch_dev, self.data_folder, self.lab_dict, self.wav_lst_te, self.wlen, self.wshift, self.class_lay, self.model)
        val.validate(epoch)


def main():
    # np.random.seed(seed)
    # from tensorflow import set_random_seed
    # set_random_seed(seed)

    print('N_filt', conf.cnn_N_filt)
    print('N_filt len', conf.cnn_len_filt)
    print('FS', conf.fs)
    print('WLEN', conf.wlen)

    input_shape = (conf.wlen, 1)
    out_dim = conf.class_lay[0]
    from model import getModel

    model = getModel(input_shape, out_dim)
    optimizer = RMSprop(lr=conf.lr, rho=0.9, epsilon=1e-8)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    checkpoints_path = os.path.join(conf.output_folder, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    tb = TensorBoard(log_dir=os.path.join(conf.output_folder, 'logs', 'SincNet'))
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, 'SincNet.hdf5'),
        verbose=1,
        save_best_only=False
    )

    validation = ValidationCallback(conf.Batch_dev, conf.data_folder, conf.lab_dict, conf.wav_lst_te, conf.wlen, conf.wshift, conf.class_lay)
    callbacks = [tb, checkpointer, validation]

    if conf.pt_file != 'none':
        model.load_weights(conf.pt_file)

    train_generator = batchGenerator(conf.batch_size, conf.data_folder, conf.wav_lst_tr, conf.snt_tr, conf.wlen, conf.lab_dict, 0.2, out_dim)
    model.fit_generator(train_generator, steps_per_epoch=conf.N_batches, epochs=conf.N_epochs, verbose=1, callbacks=callbacks)


if __name__ == '__main__':
    main()
