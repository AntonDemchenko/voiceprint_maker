import os
from uuid import uuid4

import numpy as np

from config import read_config
from data_loader import ClassifierDataLoader
from sincnet import SincNetClassifierFactory
from training import train
from train_classifier import make_model
from test_classifier import test


def main():
    cfg = read_config()

    output_folder = cfg.output_folder

    max_iters = 1000
    for _ in range(max_iters):
        uid = str(uuid4())
        cfg.output_folder = os.path.join(output_folder, uid)

        options = dict(
            cnn_act=[np.random.choice(['relu', 'leaky_relu'])] * 3,
            cnn_drop=[np.random.uniform(0.0, 0.4)] * 3,
            cnn_use_laynorm=[np.random.choice([False, True])] * 3,
            cnn_use_laynorm_inp=np.random.choice([False, True]),
            cnn_use_batchnorm=[np.random.choice([False, True])] * 3,
            cnn_use_batchnorm_inp=np.random.choice([False, True]),
            fc_lay=[np.random.choice([128, 256, 512, 1024, 2048])] * 3,
            fc_drop=[np.random.uniform(0.0, 0.4)] * 3,
            fc_act=[np.random.choice(['relu', 'leaky_relu'])] * 3,
            fc_use_laynorm=[np.random.choice([False, True])] * 3,
            fc_use_laynorm_inp=np.random.choice([False, True]),
            fc_use_batchnorm=[np.random.choice([False, True])] * 3,
            fc_use_batchnorm_inp=np.random.choice([False, True]),
            class_use_laynorm_inp=np.random.choice([False, True]),
            class_use_batchnorm_inp=np.random.choice([False, True]),
            optimizer=np.random.choice(['adam', 'rmsprop'])
        )
        cfg.__dict__.update(options)
        model = make_model(cfg)
        data_loader = ClassifierDataLoader(cfg)
        train(cfg, model, data_loader)

        for layer in model.layers:
            layer.trainable = False

        accuracies = test(cfg, model, data_loader, cfg.validation_list)


if __name__ == '__main__':
    main()
