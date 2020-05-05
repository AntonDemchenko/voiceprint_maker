import csv
import json
import os
import random
from uuid import uuid4

from tensorflow.keras.callbacks import Callback

from config import read_config
from data_loader import DataLoader
from training import train
from train_classifier import make_model
from test_classifier import test


BATCH = 'batch'
LAYER = 'layer'

CNN_NORM_BEFORE = [BATCH, LAYER, None]
CNN_N_LAYERS = 3
CNN_NORM = [BATCH, LAYER]
CNN_DROP_RANGE = [0.0, 0.4]

FC_NORM_BEFORE = [BATCH, LAYER, None]
FC_N_LAYERS_RANGE = [1, 3]
FC_SIZES = [128, 256, 512, 1024, 2048]
FC_NORM = [BATCH, LAYER]
FC_DROP_RANGE = [0.0, 0.4]

CLASS_NORM_BEFORE = [BATCH, LAYER, None]
ACTIVATIONS = ['relu', 'leaky_relu']
LOG10_LEARNING_RATE_RANGE = [-3, -2.15]
OPTIMIZERS = ['adam', 'rmsprop']


def get_random_options():
    cnn_norm_before = random.choice(CNN_NORM_BEFORE)
    cnn_norm = random.choice(CNN_NORM)
    fc_norm_before = random.choice(FC_NORM_BEFORE)
    fc_norm = random.choice(FC_NORM)
    class_norm_before = random.choice(CLASS_NORM_BEFORE)
    fc_n_layers = random.randrange(*FC_N_LAYERS_RANGE)

    return {
        'cnn_use_layer_norm_before': (cnn_norm_before == LAYER),
        'cnn_use_batch_norm_before': (cnn_norm_before == BATCH),
        'cnn_n_layers': CNN_N_LAYERS,
        'cnn_use_layer_norm': [cnn_norm == LAYER] * CNN_N_LAYERS,
        'cnn_use_batch_norm': [cnn_norm == BATCH] * CNN_N_LAYERS,
        'cnn_act': [random.choice(ACTIVATIONS)] * CNN_N_LAYERS,
        'cnn_drop': [random.uniform(*CNN_DROP_RANGE)] * CNN_N_LAYERS,
        'fc_use_layer_norm_before': (fc_norm_before == LAYER),
        'fc_use_batch_norm_before': (fc_norm_before == BATCH),
        'fc_n_layers': fc_n_layers,
        'fc_size': [random.choice(FC_SIZES)] * fc_n_layers,
        'fc_use_layer_norm': [fc_norm == LAYER] * fc_n_layers,
        'fc_use_batch_norm': [fc_norm == BATCH] * fc_n_layers,
        'fc_act': [random.choice(ACTIVATIONS)] * fc_n_layers,
        'fc_drop': [random.uniform(*FC_DROP_RANGE)] * fc_n_layers,
        'class_use_layer_norm_before': (class_norm_before == LAYER),
        'class_use_batch_norm_before': (class_norm_before == BATCH),
        'lr': 10 ** random.uniform(*LOG10_LEARNING_RATE_RANGE),
        'optimizer': random.choice(OPTIMIZERS)
    }


def save_options(file_path, options):
    with open(file_path, 'w') as f:
        options_json = json.dumps(options, sort_keys=True, indent=4)
        print(options_json, file=f)


def reduce_repeated_values(options):
    for key in options:
        value = options[key]
        if isinstance(value, list) and all(v == value[0] for v in value):
            options[key] = value[0]
    return options


def save_tuning_result(file_path, uid, options, accuracies):
    options = reduce_repeated_values(options)

    tuning_result = dict()
    tuning_result.update(options)
    tuning_result.update(accuracies)
    tuning_result['uid'] = uid
    fieldnames = ['uid'] + sorted(options) + sorted(accuracies)

    is_first_row = not os.path.exists(file_path)
    with open(file_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_first_row:
            writer.writeheader()
        writer.writerow(tuning_result)


class EarlyStoppingMaxLoss(Callback):
    def __init__(self, max_loss, verbose=1):
        super().__init__()
        self.max_loss = max_loss
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        for name in ['loss', 'val_loss']:
            loss = logs.get(name)
            if loss is not None and self.max_loss < loss:
                self.model.stop_training = True
                if self.verbose:
                    print('Early stopping: {} exceeds max value ({} > {})'\
                        .format(name, loss, self.max_loss)
                    )


def make_early_stopping(cfg):
    from math import log
    max_loss = 3 * log(cfg.n_classes)
    return EarlyStoppingMaxLoss(max_loss)


def do_tune_step(cfg, output_folder):
    uid = str(uuid4())
    cfg.output_folder = os.path.join(output_folder, uid)
    os.makedirs(cfg.output_folder)

    options = get_random_options()

    options_file = os.path.join(cfg.output_folder, 'options.json')
    save_options(options_file, options)

    before = len(cfg.__dict__)
    cfg.__dict__.update(options)
    assert before == len(cfg.__dict__)

    callbacks = [make_early_stopping(cfg)]
    history = train(cfg, callbacks)

    # for layer in model.layers:
    #     layer.trainable = False
    # all_chunks_val_acc, sentence_val_acc = test(
    #     cfg, model, data_loader, cfg.val_list
    # )

    accuracies = dict(
        # sentence_val_acc=sentence_val_acc,
        # all_chunks_val_acc=all_chunks_val_acc,
        train_acc=history.history['accuracy'][-1],
        val_acc=history.history['val_accuracy'][-1]
    )

    tuning_result_file = os.path.join(output_folder, 'tuning_results.csv')
    save_tuning_result(tuning_result_file, uid, options, accuracies)


def main():
    cfg = read_config()
    output_folder = cfg.output_folder

    max_iters = 5000
    for _ in range(max_iters):
        do_tune_step(cfg, output_folder)


if __name__ == '__main__':
    main()
