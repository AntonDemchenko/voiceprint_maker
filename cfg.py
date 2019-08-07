import configparser as ConfigParser


def read_cfg(cfg_file):
    if cfg_file is None:
        raise ValueError

    config = ConfigParser.ConfigParser()
    config.read(cfg_file)

    options = {}

    # [data]
    options.tr_lst = config.get('data', 'tr_lst')
    options.te_lst = config.get('data', 'te_lst')
    options.lab_dict = config.get('data', 'lab_dict')
    options.data_folder = config.get('data', 'data_folder')
    options.output_folder = config.get('data', 'output_folder')
    options.pt_file = config.get('data', 'pt_file')

    # [windowing]
    options.fs = config.get('windowing', 'fs')
    options.cw_len = config.get('windowing', 'cw_len')
    options.cw_shift = config.get('windowing', 'cw_shift')

    # [cnn]
    options.cnn_N_filt = config.get('cnn', 'cnn_N_filt')
    options.cnn_len_filt = config.get('cnn', 'cnn_len_filt')
    options.cnn_max_pool_len = config.get('cnn', 'cnn_max_pool_len')
    options.cnn_use_laynorm_inp = config.get('cnn', 'cnn_use_laynorm_inp')
    options.cnn_use_batchnorm_inp = config.get('cnn', 'cnn_use_batchnorm_inp')
    options.cnn_use_laynorm = config.get('cnn', 'cnn_use_laynorm')
    options.cnn_use_batchnorm = config.get('cnn', 'cnn_use_batchnorm')
    options.cnn_act = config.get('cnn', 'cnn_act')
    options.cnn_drop = config.get('cnn', 'cnn_drop')

    # [dnn]
    options.fc_lay = config.get('dnn', 'fc_lay')
    options.fc_drop = config.get('dnn', 'fc_drop')
    options.fc_use_laynorm_inp = config.get('dnn', 'fc_use_laynorm_inp')
    options.fc_use_batchnorm_inp = config.get('dnn', 'fc_use_batchnorm_inp')
    options.fc_use_batchnorm = config.get('dnn', 'fc_use_batchnorm')
    options.fc_use_laynorm = config.get('dnn', 'fc_use_laynorm')
    options.fc_act = config.get('dnn', 'fc_act')

    # [class]
    options.class_lay = config.get('class', 'class_lay')
    options.class_drop = config.get('class', 'class_drop')
    options.class_use_laynorm_inp = config.get('class', 'class_use_laynorm_inp')
    options.class_use_batchnorm_inp = config.get('class', 'class_use_batchnorm_inp')
    options.class_use_batchnorm = config.get('class', 'class_use_batchnorm')
    options.class_use_laynorm = config.get('class', 'class_use_laynorm')
    options.class_act = config.get('class', 'class_act')

    # [optimization]
    options.lr = config.get('optimization', 'lr')
    options.batch_size = config.get('optimization', 'batch_size')
    options.N_epochs = config.get('optimization', 'N_epochs')
    options.N_batches = config.get('optimization', 'N_batches')
    options.N_eval_epoch = config.get('optimization', 'N_eval_epoch')
    options.seed = config.get('optimization', 'seed')

    return parse_options(options)


def parse_options(options):
    assert options is not None

    return {
        # [data]
        'tr_lst': options.tr_lst,
        'te_lst': options.te_lst,
        'pt_file': options.pt_file,
        'class_dict_file': options.lab_dict,
        'data_folder': options.data_folder+'/',
        'output_folder': options.output_folder,

        # [windowing]
        'fs': int(options.fs),
        'cw_len': int(options.cw_len),
        'cw_shift': int(options.cw_shift),

        # [cnn]
        'cnn_N_filt': list(map(int, options.cnn_N_filt.split(','))),
        'cnn_len_filt': list(map(int, options.cnn_len_filt.split(','))),
        'cnn_max_pool_len': list(map(int, options.cnn_max_pool_len.split(','))),
        'cnn_use_laynorm_inp': str_to_bool(options.cnn_use_laynorm_inp),
        'cnn_use_batchnorm_inp': str_to_bool(options.cnn_use_batchnorm_inp),
        'cnn_use_laynorm': list(map(str_to_bool, options.cnn_use_laynorm.split(','))),
        'cnn_use_batchnorm': list(map(str_to_bool, options.cnn_use_batchnorm.split(','))),
        'cnn_act': list(map(str, options.cnn_act.split(','))),
        'cnn_drop': list(map(float, options.cnn_drop.split(','))),

        # [dnn]
        'fc_lay': list(map(int, options.fc_lay.split(','))),
        'fc_drop': list(map(float, options.fc_drop.split(','))),
        'fc_use_laynorm_inp': str_to_bool(options.fc_use_laynorm_inp),
        'fc_use_batchnorm_inp': str_to_bool(options.fc_use_batchnorm_inp),
        'fc_use_batchnorm': list(map(str_to_bool, options.fc_use_batchnorm.split(','))),
        'fc_use_laynorm': list(map(str_to_bool, options.fc_use_laynorm.split(','))),
        'fc_act ': list(map(str, options.fc_act.split(','))),

        # [class]
        'class_lay': list(map(int, options.class_lay.split(','))),
        'class_drop': list(map(float, options.class_drop.split(','))),
        'class_use_laynorm_inp': str_to_bool(options.class_use_laynorm_inp),
        'class_use_batchnorm_inp': str_to_bool(options.class_use_batchnorm_inp),
        'class_use_batchnorm': list(map(str_to_bool, options.class_use_batchnorm.split(','))),
        'class_use_laynorm': list(map(str_to_bool, options.class_use_laynorm.split(','))),
        'class_act': list(map(str, options.class_act.split(','))),

        # [optimization]
        'lr': float(options.lr),
        'batch_size': int(options.batch_size),
        'N_epochs': int(options.N_epochs),
        'N_batches': int(options.N_batches),
        'N_eval_epoch': int(options.N_eval_epoch),
        'seed': int(options.seed),

        # training list
        'wav_lst_tr': _read_list_file(options.tr_lst),

        # test list
        'wav_lst_te': _read_list_file(options.te_lst),

    }


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError


def _read_list_file(list_file):
    list_sig = []
    with open(list_file, "r") as f:
        lines = f.readlines()
        for x in lines:
            list_sig.append(x.rstrip())
    return list_sig


