import csv
import json
import os
import random
from uuid import uuid4

from tensorflow.keras.callbacks import Callback

from config import read_config
from data_loader import ClassifierDataLoader
from sincnet import SincNetClassifierFactory
from training import train
from train_classifier import make_model
from test_classifier import test


BATCH = 'batch'
LAYER = 'layer'


def random_norm():
    return random.choice([BATCH, LAYER])


def random_input_norm():
    return random.choice([BATCH, LAYER, None])


def get_random_options():
    cnn_input_norm = random_input_norm()
    cnn_norm = random_norm()
    fc_input_norm = random_input_norm()
    fc_norm = random_norm()
    class_input_norm = random_input_norm()
    return {
        'cnn_use_layer_norm_before': (cnn_input_norm == LAYER),
        'cnn_use_batch_norm_before': (cnn_input_norm == BATCH),
        'cnn_use_layer_norm': [cnn_norm == LAYER] * 3,
        'cnn_use_batch_norm': [cnn_norm == BATCH] * 3,
        'cnn_act': [random.choice(['relu', 'leaky_relu'])] * 3,
        'cnn_drop': [random.uniform(0.0, 0.4)] * 3,
        'fc_use_layer_norm_before': (fc_input_norm == LAYER),
        'fc_use_batch_norm_before': (fc_input_norm == BATCH),
        'fc_size': [random.choice([128, 256, 512, 1024, 2048])] * 3,
        'fc_use_layer_norm': [fc_norm == LAYER] * 3,
        'fc_use_batch_norm': [fc_norm == BATCH] * 3,
        'fc_act': [random.choice(['relu', 'leaky_relu'])] * 3,
        'fc_drop': [random.uniform(0.0, 0.4)] * 3,
        'class_use_layer_norm_before': (class_input_norm == LAYER),
        'class_use_batch_norm_before': (class_input_norm == BATCH),
        'lr': 10 ** random.uniform(-3, -2.15),
        'optimizer': random.choice(['adam', 'rmsprop'])
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

    model = make_model(cfg)
    data_loader = ClassifierDataLoader(cfg)
    callbacks = [make_early_stopping(cfg)]
    history = train(cfg, model, data_loader, callbacks)

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
