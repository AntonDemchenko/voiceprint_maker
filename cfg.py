import configparser


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
        self.tr_lst = config.get('data', 'tr_lst')
        self.te_lst = config.get('data', 'te_lst')
        self.lab_dict = config.get('data', 'lab_dict')
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
        self.fc_use_batchnorm = [self._str_to_bool(e) for e in config.get('dnn', 'fc_use_batchnorm').split(',')]
        self.fc_use_laynorm = [self._str_to_bool(e) for e in config.get('dnn', 'fc_use_laynorm').split(',')]
        self.fc_act = [e for e in config.get('dnn', 'fc_act').split(',')]

        # [class]
        self.class_lay = config.getintlist('class', 'class_lay')
        self.class_drop = config.getfloatlist('class', 'class_drop')
        self.class_use_laynorm_inp = config.getboolean('class', 'class_use_laynorm_inp')
        self.class_use_batchnorm_inp = config.getboolean('class', 'class_use_batchnorm_inp')
        self.class_use_batchnorm = config.getbooleanlist('class', 'class_use_batchnorm')
        self.class_use_laynorm = config.getbooleanlist('class', 'class_use_laynorm')
        self.class_act = config.getlist('class', 'class_act')

        # [optimization]
        self.lr = config.getfloat('optimization', 'lr')
        self.batch_size = config.getint('optimization', 'batch_size')
        self.N_epochs = config.getint('optimization', 'N_epochs')
        self.N_batches = config.getint('optimization', 'N_batches')
        self.N_eval_epoch = config.getint('optimization', 'N_eval_epoch')
        self.seed = config.getint('optimization', 'seed')

        # training list
        self.train_list = self._read_list_file(self.tr_lst)

        # test list
        self.test_list = self._read_list_file(self.te_lst)

    def _str_to_bool(self, s):
        if s == 'True':
            return True
        elif s == 'False':
            return False
        else:
            raise ValueError

    def _read_list_file(self, list_file):
        list_sig = []
        with open(list_file, "r") as f:
            lines = f.readlines()
            for x in lines:
                list_sig.append(x.rstrip())
        return list_sig


