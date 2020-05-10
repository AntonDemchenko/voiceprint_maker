import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree
from tqdm import tqdm

from config import read_config
from data_loader import DataLoader
from sincnet import create_print_maker


def check_norms(vectors):
    norms = [np.linalg.norm(v) for v in vectors]
    assert abs(1 - min(norms)) < 1e-6
    assert abs(1 - max(norms)) < 1e-6


def make_window_voiceprints(model, dataset):
    paths = []
    voiceprints = []
    for path_batch, signal_batch, _ in tqdm(dataset):
        voiceprint_batch = model.predict(signal_batch)
        paths.extend(path_batch)
        voiceprints.extend(voiceprint_batch)
    check_norms(voiceprints)
    return paths, voiceprints


def make_path_voiceprints_from_window_voiceprints(paths, voiceprints):
    path_to_voiceprint = dict()
    path_to_chunk_cnt = dict()
    for path, voiceprint in zip(paths, voiceprints):
        if path not in path_to_voiceprint:
            path_to_chunk_cnt[path] = 0
            path_to_voiceprint[path] = np.zeros(voiceprint.shape)
        path_to_voiceprint[path] += voiceprint
        path_to_chunk_cnt[path] += 1
    for path in path_to_voiceprint.keys():
        path_to_voiceprint[path] /= path_to_chunk_cnt[path]
    paths = [path for path, _ in path_to_voiceprint.items()]
    voiceprints = [voiceprint for _, voiceprint in path_to_voiceprint.items()]
    voiceprints = tf.math.l2_normalize(voiceprints, axis=1).numpy()
    check_norms(voiceprints)
    return paths, voiceprints


def identify(cfg, voiceprints):
    voiceprints = np.array(voiceprints)
    kdtree = KDTree(voiceprints)
    closest_indexes = []
    for p in tqdm(voiceprints):
        closest = kdtree.query([p], k=2, return_distance=False, sort_results=True)[0][1]
        closest_indexes.append(closest)
    return closest_indexes


def calculate_accuracy(cfg, paths, voiceprints):
    assert len(paths) == len(voiceprints)
    
    labels = np.array([cfg.path_to_label[path] for path in paths])
    closest_indexes = identify(cfg, voiceprints)
    predicted_labels = np.array([labels[c] for c in closest_indexes])
    accuracy = np.mean(labels == predicted_labels)
    
    return accuracy


def test(cfg, model, dataset):
    paths, voiceprints = make_window_voiceprints(model, dataset)
    window_accuracy = calculate_accuracy(cfg, paths, voiceprints)

    paths, voiceprints = make_path_voiceprints_from_window_voiceprints(paths, voiceprints)
    path_accuracy = calculate_accuracy(cfg, paths, voiceprints)

    return window_accuracy, path_accuracy


def main():
    cfg = read_config()
    model = create_print_maker(cfg)
    # Skip mismatch enables to load weights of networks with other head
    model.load_weights(cfg.checkpoint_file, by_name=True, skip_mismatch=True)
    for layer in model.layers:
        layer.trainable = False
    dataset = DataLoader(cfg).make_test_iterable(cfg.test_list)
    accuracy = test(cfg, model, dataset)
    print(accuracy)


if __name__ == '__main__':
    main()
