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


def unite_equally_labeled_voiceprints(labels, voiceprints):
    label_to_voiceprints = dict()
    for l, v in zip(labels, voiceprints):
        if l not in label_to_voiceprints:
            label_to_voiceprints[l] = []
        label_to_voiceprints[l].append(v)
    labels = list(l for l, _ in label_to_voiceprints.items())
    voiceprints = list(np.mean(vs, axis=0) for _, vs in label_to_voiceprints.items())
    voiceprints = tf.math.l2_normalize(voiceprints, axis=1).numpy()
    check_norms(voiceprints)
    return labels, voiceprints


def make_path_voiceprints(model, dataset):
    path_to_voiceprint_sum = dict()
    path_to_voiceprint_count = dict()
    for path_batch, signal_batch, _ in tqdm(dataset):
        voiceprint_batch = model.predict(signal_batch)
        for p, v in zip(path_batch, voiceprint_batch):
            if p not in path_to_voiceprint_sum:
                path_to_voiceprint_sum[p] = np.zeros(v.shape)
                path_to_voiceprint_count[p] = 0
            path_to_voiceprint_sum[p] += v
            path_to_voiceprint_count[p] += 1
    paths = []
    voiceprints = []
    for path in path_to_voiceprint_sum:
        paths.append(path)
        v = path_to_voiceprint_sum[path] / path_to_voiceprint_count[path]
        voiceprints.append(v)
    voiceprints = tf.math.l2_normalize(voiceprints, axis=1).numpy()
    check_norms(voiceprints)
    return paths, voiceprints


def calculate_accuracies(base_labels, base_voiceprints, test_labels, test_voiceprints, max_top):
    assert len(base_labels) == len(base_voiceprints)
    assert len(test_labels) == len(test_voiceprints)

    base_labels = np.array(base_labels)
    test_labels = np.array(test_labels)
    base_voiceprints = np.array(base_voiceprints)
    kdtree = KDTree(base_voiceprints)
    top_to_accuracy = dict()

    for top in tqdm(range(1, max_top + 1)):
        closest_indexes = kdtree.query(test_voiceprints, k=top, return_distance=False, sort_results=True)
        predicted_labels = np.array([base_labels[c] for c in closest_indexes])
        accuracy = np.mean([test in predicted for test, predicted in zip(test_labels, predicted_labels)])
        top_to_accuracy[top] = accuracy

    return top_to_accuracy


def test(cfg, model, train_dataset, test_dataset):
    paths, voiceprints = make_path_voiceprints(model, train_dataset)
    labels = [cfg.path_to_label[p] for p in paths]
    base_labels, base_voiceprints = unite_equally_labeled_voiceprints(labels, voiceprints)
    paths, test_voiceprints = make_path_voiceprints(model, test_dataset)
    test_labels = [cfg.path_to_label[p] for p in paths]
    top_to_accuracy = calculate_accuracies(
        base_labels,
        base_voiceprints,
        test_labels,
        test_voiceprints,
        cfg.max_top
    )
    return top_to_accuracy


def main():
    cfg = read_config()
    model = create_print_maker(cfg)
    # Skip mismatch enables to load weights of networks with other head
    model.load_weights(cfg.checkpoint_file, by_name=True, skip_mismatch=True)
    for layer in model.layers:
        layer.trainable = False
    data_loader = DataLoader(cfg)
    train_dataset = data_loader.make_test_iterable(cfg.train_list)
    test_dataset = data_loader.make_test_iterable(cfg.test_list)
    top_to_accuracy = test(cfg, model, train_dataset, test_dataset)
    for top, acc in sorted(top_to_accuracy.items()):
        print(top, acc)


if __name__ == '__main__':
    main()
