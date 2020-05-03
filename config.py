import configparser
import os
import re

import numpy as np


class SincNetConfigParser(configparser.ConfigParser):
    def getintlist(self, section, option, size):
        return list(map(int, self.getlist(section, option, size)))

    def getbooleanlist(self, section, option, size):
        return list(map(self._str_to_bool, self.getlist(section, option, size)))

    def getfloatlist(self, section, option, size):
        return list(map(float, self.getlist(section, option, size)))

    def getlist(self, section, option, size):
        value = self.get(section, option)
        result = value.split(',')
        if len(result) == 1:
            result *= size
        if len(result) != size:
            raise ValueError(
                'Invalid length of {}.{} list ({} is expected, {} is found)'\
                .format(section, option, size, len(result))
            )
        return result

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
        self.train_list_file = config.get('data', 'train_list_file')
        self.test_list_file = config.get('data', 'test_list_file')
        self.val_list_file = config.get('data', 'val_list_file')
        self.path_to_label_file = config.get('data', 'path_to_label_file')
        self.dataset_folder = config.get('data', 'dataset_folder')
        self.output_folder = config.get('data', 'output_folder')
        self.checkpoint_file = config.get('data', 'checkpoint_file')

        # [windowing]
        self.sample_rate = config.getint('windowing', 'sample_rate')
        self.window_len_ms = config.getint('windowing', 'window_len_ms')
        self.window_shift_ms = config.getint('windowing', 'window_shift_ms')

        # [cnn]
        self.cnn_n_layers = config.getint('cnn', 'cnn_n_layers')
        self.cnn_n_filters = config.getintlist('cnn', 'cnn_n_filters', self.cnn_n_layers)
        self.cnn_filter_len = config.getintlist('cnn', 'cnn_filter_len', self.cnn_n_layers)
        self.cnn_max_pool_len = config.getintlist('cnn', 'cnn_max_pool_len', self.cnn_n_layers)
        self.cnn_use_layer_norm_before = config.getboolean('cnn', 'cnn_use_layer_norm_before')
        self.cnn_use_batch_norm_before = config.getboolean('cnn', 'cnn_use_batch_norm_before')
        self.cnn_use_layer_norm = config.getbooleanlist('cnn', 'cnn_use_layer_norm', self.cnn_n_layers)
        self.cnn_use_batch_norm = config.getbooleanlist('cnn', 'cnn_use_batch_norm', self.cnn_n_layers)
        self.cnn_act = config.getlist('cnn', 'cnn_act', self.cnn_n_layers)
        self.cnn_drop = config.getfloatlist('cnn', 'cnn_drop', self.cnn_n_layers)

        # [dnn]
        self.fc_n_layers = config.getint('dnn', 'fc_n_layers')
        self.fc_size = config.getintlist('dnn', 'fc_size', self.fc_n_layers)
        self.fc_use_layer_norm_before = config.getboolean('dnn', 'fc_use_layer_norm_before')
        self.fc_use_batch_norm_before = config.getboolean('dnn', 'fc_use_batch_norm_before')
        self.fc_use_batch_norm = config.getbooleanlist('dnn', 'fc_use_batch_norm', self.fc_n_layers)
        self.fc_use_layer_norm = config.getbooleanlist('dnn', 'fc_use_layer_norm', self.fc_n_layers)
        self.fc_act = config.getlist('dnn', 'fc_act', self.fc_n_layers)
        self.fc_drop = config.getfloatlist('dnn', 'fc_drop', self.fc_n_layers)

        # [class]
        self.n_classes = config.getint('class', 'n_classes')
        self.class_use_layer_norm_before = config.getboolean('class', 'class_use_layer_norm_before')
        self.class_use_batch_norm_before = config.getboolean(
            'class', 'class_use_batch_norm_before'
        )

        # [optimization]
        self.optimizer = config.get('optimization', 'optimizer')
        self.lr = config.getfloat('optimization', 'lr')
        self.batch_size = config.getint('optimization', 'batch_size')
        self.n_epochs = config.getint('optimization', 'n_epochs')
        self.n_batches = config.getint('optimization', 'n_batches')
        self.val_freq = config.getint('optimization', 'val_freq')
        self.seed = config.getint('optimization', 'seed')
        self.n_val_windows_per_sample = config.getint('optimization', 'n_val_windows_per_sample')
        self.batch_size_test = config.getint('optimization', 'batch_size_test')

        # [callbacks]
        self.best_checkpoint_freq = config.getint('callbacks', 'best_checkpoint_freq')
        self.use_tensorboard_logger = config.getboolean('callbacks', 'use_tensorboard_logger')
        self.save_checkpoints = config.getboolean('callbacks', 'save_checkpoints')

        self.checkpoint_folder = os.path.join(self.output_folder, 'checkpoints')
        self.best_checkpoint_path = os.path.join(self.checkpoint_folder, 'best_checkpoint.hdf5')
        self.last_checkpoint_path = os.path.join(self.checkpoint_folder, 'last_checkpoint.hdf5')

        self.window_len = int(self.sample_rate * self.window_len_ms / 1000.00)
        self.window_shift = int(self.sample_rate * self.window_len_ms / 1000.00)

        self.out_dim = 100
        self.input_shape = (self.window_len, 1)

        self.train_list = self._read_list_file(self.train_list_file)
        self.test_list = self._read_list_file(self.test_list_file)
        self.val_list = self._read_list_file(self.val_list_file)

        self.path_to_label = np.load(self.path_to_label_file, allow_pickle=True).item()

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
        match = re.compile(r'SincNet-(\d+)\.hdf5$').search(self.checkpoint_file)
        if match:
            result = int(match.group(1))
        elif self.checkpoint_file != 'none' and os.path.exists(log_path):
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
