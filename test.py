import io

import numpy as np
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm

from config import read_config
from model import getModel


class Validation:
    def __init__(self, cfg, model, debug=False):
        self.cfg = cfg
        self.model = model
        self.debug = debug

    def validate(self, epoch=None):

        if epoch is None or epoch % self.cfg.N_eval_epoch == 0:
            print('Valuating test set...')

            snt_te = len(self.cfg.wav_lst_te)

            err_sum = 0
            err_sum_snt = 0
            stn_sum = 0
            if self.debug:
                print('WLEN: ' + str(self.cfg.wlen))
                print('WSHIFT: ' + str(self.cfg.wshift))
                pbar = tqdm(total=snt_te)
            for i in range(snt_te):
                # [signal, fs] = sf.read(data_folder+wav_lst_te[i])
                fname = self.cfg.data_folder + self.cfg.wav_lst_te[i]
                with tf.io.gfile.GFile(fname, 'rb') as f:
                    [signal, fs] = sf.read(io.BytesIO(f.read()))

                signal = np.array(signal)
                lab_batch = self.cfg.lab_dict[self.cfg.wav_lst_te[i]]

                # split signals into chunck
                beg_samp = 0
                end_samp = self.cfg.wlen

                N_fr = int((signal.shape[0] - self.cfg.wlen) / (self.cfg.wshift))

                sig_arr = np.zeros([self.cfg.Batch_dev, self.cfg.wlen])

                lab = np.zeros(N_fr + 1) + lab_batch
                pout = np.zeros(shape=(N_fr + 1, self.cfg.class_lay[-1]))
                count_fr = 0
                count_fr_tot = 0

                while end_samp < signal.shape[0]:  # for each chunck
                    sig_arr[count_fr, :] = signal[beg_samp:end_samp]
                    beg_samp = beg_samp + self.cfg.wshift
                    end_samp = beg_samp + self.cfg.wlen
                    count_fr = count_fr + 1
                    count_fr_tot = count_fr_tot + 1
                    if count_fr == self.cfg.Batch_dev:
                        a, b = np.shape(sig_arr)
                        inp = sig_arr.reshape(a, b, 1)
                        inp = np.array(inp)
                        pout[
                            count_fr_tot - self.cfg.Batch_dev : count_fr_tot, :
                        ] = self.model.predict(inp, verbose=0)
                        count_fr = 0
                        sig_arr = np.zeros([self.cfg.Batch_dev, self.cfg.wlen])

                # Add the last items left
                if count_fr > 0:
                    inp = sig_arr[0:count_fr]
                    a, b = np.shape(inp)
                    inp = inp.reshape(a, b, 1)
                    inp = np.array(inp)
                    pout[
                        count_fr_tot - count_fr : count_fr_tot, :
                    ] = self.model.predict(inp, verbose=0)

                # Prediction for each chunkc  and calculation of average error
                pred = np.argmax(pout, axis=1)
                err = np.mean(pred != lab)

                # Calculate accuracy on the whole sentence
                best_class = np.argmax(np.sum(pout, axis=0))

                err_sum_snt = err_sum_snt + float((best_class != lab[0]))
                err_sum = err_sum + err

                stn_sum += 1

                temp_acc_stn = str(round(1 - (err_sum_snt / stn_sum), 4))
                temp_acc = str(round(1 - (err_sum / stn_sum), 4))
                if self.debug:
                    pbar.set_description(
                        'acc: {}, acc_snt: {}'.format(temp_acc, temp_acc_stn)
                    )
                    pbar.update(1)

            # average accuracy
            acc = 1 - (err_sum / snt_te)
            acc_snt = 1 - (err_sum_snt / snt_te)
            if self.debug:
                pbar.close()
            if epoch is None:
                print('acc_te: {}, acc_te_snt: {}\n'.format(acc, acc_snt))
            else:
                print(
                    'Epoch: {}, acc_te: {}, acc_te_snt: {}\n'.format(
                        epoch, acc, acc_snt
                    )
                )
                with open(self.cfg.output_folder + '/res.res', 'a') as res_file:
                    res_file.write(
                        'epoch %i, acc_te=%f acc_te_snt=%f\n' % (
                            epoch, acc, acc_snt
                        )
                    )
            return (acc, acc_snt)


def main():
    cfg = read_config()
    model = getModel(cfg)
    model.load_weights(cfg.pt_file)
    val = Validation(
        cfg,
        model,
        True,
    )
    val.validate()


if __name__ == '__main__':
    main()
