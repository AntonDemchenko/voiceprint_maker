import numpy as np
from tqdm import tqdm

from config import read_config
from data_loader import DataLoader
from sincnet import SincNetModelFactory


def test(cfg, model, data_loader, path_list):
    dataset = data_loader.make_test_iterable(path_list)
    path_to_prediction_sum = dict()
    samples_count = 0
    error_count = 0
    for path_batch, signal_batch in tqdm(dataset):
        samples_count += len(path_batch)
        prediction_batch = model.predict(signal_batch)
        for path, prediction in zip(path_batch, prediction_batch):
            if path not in path_to_prediction_sum:
                path_to_prediction_sum[path] = np.zeros([cfg.n_classes])
            path_to_prediction_sum[path] += prediction
            predicted_label = np.argmax(prediction)
            expected_label = cfg.path_to_label[path]
            if expected_label != predicted_label:
                error_count += 1
    accuracy_sample = 1 - (error_count / samples_count)
    error_count = 0
    for path in path_list:
        prediction_sum = path_to_prediction_sum[path]
        expected_label = cfg.path_to_label[path]
        predicted_label = np.argmax(prediction_sum)
        if expected_label != predicted_label:
            error_count += 1
    accuracy_path = 1 - (error_count / len(path_list))
    return accuracy_sample, accuracy_path


def main():
    cfg = read_config()
    model = SincNetModelFactory(cfg).create()
    model.load_weights(cfg.checkpoint_file)
    for layer in model.layers:
        layer.trainable = False
    data_loader = DataLoader(cfg)
    accuracy = test(cfg, model, data_loader, cfg.test_list)
    print(accuracy)


if __name__ == '__main__':
    main()
