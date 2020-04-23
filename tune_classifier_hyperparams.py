import csv
import json
import os
import random
from uuid import uuid4

from config import read_config
from data_loader import ClassifierDataLoader
from sincnet import SincNetClassifierFactory
from training import train
from train_classifier import make_model
from test_classifier import test


def get_random_options():
    return dict(
        cnn_act=[random.choice(['relu', 'leaky_relu'])] * 3,
        cnn_drop=[random.uniform(0.0, 0.4)] * 3,
        cnn_use_laynorm=[random.choice([False, True])] * 3,
        cnn_use_laynorm_inp=random.choice([False, True]),
        cnn_use_batchnorm=[random.choice([False, True])] * 3,
        cnn_use_batchnorm_inp=random.choice([False, True]),
        fc_lay=[random.choice([128, 256, 512, 1024, 2048])] * 3,
        fc_drop=[random.uniform(0.0, 0.4)] * 3,
        fc_act=[random.choice(['relu', 'leaky_relu'])] * 3,
        fc_use_laynorm=[random.choice([False, True])] * 3,
        fc_use_laynorm_inp=random.choice([False, True]),
        fc_use_batchnorm=[random.choice([False, True])] * 3,
        fc_use_batchnorm_inp=random.choice([False, True]),
        class_use_laynorm_inp=random.choice([False, True]),
        class_use_batchnorm_inp=random.choice([False, True]),
        optimizer=random.choice(['adam', 'rmsprop'])
    )


def save_options(file_path, options):
    with open(file_path, 'w') as f:
        options_json = json.dumps(options, sort_keys=True, indent=4)
        print(options_json, file=f)


def save_tuning_result(file_path, options, accuracies):
    tuning_result = dict()
    tuning_result.update(options)
    tuning_result.update(accuracies)
    fieldnames = sorted(options) + sorted(accuracies)

    is_first_row = not os.path.exists(file_path)
    with open(file_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_first_row:
            writer.writeheader()
        writer.writerow(tuning_result)


def do_tune_step(cfg, output_folder):
    uid = str(uuid4())
    cfg.output_folder = os.path.join(output_folder, uid)
    os.makedirs(cfg.output_folder)

    options = get_random_options()

    options_file = os.path.join(cfg.output_folder, 'options.json')
    save_options(options_file, options)

    cfg.__dict__.update(options)
    model = make_model(cfg)
    data_loader = ClassifierDataLoader(cfg)
    history = train(cfg, model, data_loader)

    for layer in model.layers:
        layer.trainable = False
    all_chunks_val_acc, sentence_val_acc = test(
        cfg, model, data_loader, cfg.validation_list
    )

    accuracies = dict(
        sentence_val_acc=sentence_val_acc,
        all_chunks_val_acc=all_chunks_val_acc,
        train_acc=history.history['accuracy'][-1],
        val_acc=history.history['val_accuracy'][-1]
    )

    tuning_result_file = os.path.join(output_folder, 'tuning_results.csv')
    save_tuning_result(tuning_result_file, options, accuracies)


def main():
    cfg = read_config()
    output_folder = cfg.output_folder

    max_iters = 1000
    for _ in range(max_iters):
        do_tune_step(cfg, output_folder)


if __name__ == '__main__':
    main()
