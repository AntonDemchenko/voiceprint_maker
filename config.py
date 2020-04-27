import configparser
import os
import re

import numpy as np


class SincNetConfigParser(configparser.ConfigParser):
    def getintlist(self, section, option):
        return list(map(int, self.getlist(section, option)))

    def getbooleanlist(self, section, option):
        return list(map(self._str_to_bool, self.getlist(section, option)))

    def getfloatlist(self, section, option):
        return list(map(float, self.getlist(section, option)))

    def getlist(self, section, option):
        value = self.get(section, option)
        return value.split(',')

    def _str_to_bool(self, s):
        if s == 'True':
            return True
        elif s == 'False':
            return False
        else:
            raise ValueError


class SincNetCfg:
    def __init__(self, cfg_file):
        if cfg_file is None:
            raise ValueError

        config = SincNetConfigParser()
        config.read(cfg_file)

        # [data]
        self.train_list_file = config.get('data', 'tr_lst')
        self.test_list_file = config.get('data', 'te_lst')
        self.validation_list_file = config.get('data', 'va_lst')
        self.labels_dict_file = config.get('data', 'lab_dict')
        self.data_folder = config.get('data', 'data_folder') + '/'
        self.output_folder = config.get('data', 'output_folder')
        self.pt_file = config.get('data', 'pt_file')

        # [windowing]
        self.fs = config.getint('windowing', 'fs')
        self.cw_len = config.getint('windowing', 'cw_len')
        self.cw_shift = config.getint('windowing', 'cw_shift')

        # [cnn]
        self.cnn_N_filt = config.getintlist('cnn', 'cnn_N_filt')
        self.cnn_len_filt = config.getintlist('cnn', 'cnn_len_filt')
        self.cnn_max_pool_len = config.getintlist('cnn', 'cnn_max_pool_len')
        self.cnn_use_laynorm_inp = config.getboolean('cnn', 'cnn_use_laynorm_inp')
        self.cnn_use_batchnorm_inp = config.getboolean('cnn', 'cnn_use_batchnorm_inp')
        self.cnn_use_laynorm = config.getbooleanlist('cnn', 'cnn_use_laynorm')
        self.cnn_use_batchnorm = config.getbooleanlist('cnn', 'cnn_use_batchnorm')
        self.cnn_act = config.getlist('cnn', 'cnn_act')
        self.cnn_drop = config.getfloatlist('cnn', 'cnn_drop')

        # [dnn]
        self.fc_lay = config.getintlist('dnn', 'fc_lay')
        self.fc_drop = config.getfloatlist('dnn', 'fc_drop')
        self.fc_use_laynorm_inp = config.getboolean('dnn', 'fc_use_laynorm_inp')
        self.fc_use_batchnorm_inp = config.getboolean('dnn', 'fc_use_batchnorm_inp')
        self.fc_use_batchnorm = config.getbooleanlist('dnn', 'fc_use_batchnorm')
        self.fc_use_laynorm = config.getbooleanlist('dnn', 'fc_use_laynorm')
        self.fc_act = config.getlist('dnn', 'fc_act')

        # [class]
        self.class_lay = config.getintlist('class', 'class_lay')
        self.class_use_laynorm_inp = config.getboolean('class', 'class_use_laynorm_inp')
        self.class_use_batchnorm_inp = config.getboolean(
            'class', 'class_use_batchnorm_inp'
        )

        # [optimization]
        self.optimizer = config.get('optimization', 'optimizer')
        self.lr = config.getfloat('optimization', 'lr')
        self.batch_size = config.getint('optimization', 'batch_size')
        self.N_epochs = config.getint('optimization', 'N_epochs')
        self.N_batches = config.getint('optimization', 'N_batches')
        self.N_eval_epoch = config.getint('optimization', 'N_eval_epoch')
        self.seed = config.getint('optimization', 'seed')
        self.N_val_windows_per_sample = config.getint('optimization', 'N_val_windows_per_sample')
        self.batch_size_test = config.getint('optimization', 'batch_size_test')

        # [callbacks]
        self.best_checkpoint_freq = config.getint('callbacks', 'best_checkpoint_freq')
        self.use_tensorboard_logger = config.getboolean('callbacks', 'use_tensorboard_logger')
        self.save_checkpoints = config.getboolean('callbacks', 'save_checkpoints')

        self.checkpoint_folder = os.path.join(self.output_folder, 'checkpoints')
        self.best_checkpoint_path = os.path.join(self.checkpoint_folder, 'SincNet-{epoch:04d}.hdf5')
        self.last_checkpoint_path = os.path.join(self.checkpoint_folder, 'last_checkpoint.hdf5')

        # Converting context and shift in samples
        self.wlen = int(self.fs * self.cw_len / 1000.00)
        self.wshift = int(self.fs * self.cw_shift / 1000.00)

        # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
        # self.sig_batch = np.zeros([self.batch_size, self.wlen])
        # self.lab_batch = np.zeros(self.batch_size)
        self.out_dim = 100
        self.n_classes = self.class_lay[0]
        self.input_shape = (self.wlen, 1)

        # Loading train list
        self.train_list = self._read_list_file(self.train_list_file)
        self.snt_tr = len(self.train_list)

        # Loading test list
        self.test_list = self._read_list_file(self.test_list_file)
        self.snt_te = len(self.test_list)

        # Loading validation list
        self.validation_list = self._read_list_file(self.validation_list_file)

        # Loading label dictionary
        self.lab_dict = np.load(self.labels_dict_file, allow_pickle=True).item()

        self.fact_amp = 0.2

        self.initial_epoch = self._get_initial_epoch()

    def _read_list_file(self, list_file):
        list_sig = []
        with open(list_file, 'r') as f:
            lines = f.readlines()
            for x in lines:
                list_sig.append(x.rstrip())
        return list_sig

    def _get_initial_epoch(self):
        result = 0
        log_path = os.path.join(self.output_folder, 'log.csv')
        match = re.compile(r'SincNet-(\d+)\.hdf5$').search(self.pt_file)
        if match:
            result = int(match.group(1))
        elif self.pt_file != 'none' and os.path.exists(log_path):
            s = ''
            with open(log_path, 'r') as f:
                for line in f:
                    s = line.split(',', 1)[0]
            try:
                result = int(s) + 1
            except:
                pass
        return result


def read_config():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--cfg')
    options, _ = parser.parse_args()
    cfg = SincNetCfg(options.cfg)
    return cfg
