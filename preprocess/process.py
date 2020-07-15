import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Process(object):

    def __init__(self, batch_size, pre_fetch):
        self.batch_size = batch_size
        self.pre_fetch = pre_fetch
        self.train_len = int(20400 * .8)

    def get_datasets(self):
        ref_frames, cur_frames = self._build_files()
        labels = self._build_labels()
        dataset = self._build_dataset(ref_frames, cur_frames, labels)
        return dataset

    def _build_files(self):
        file_names = []
        for i in range(20400):
            file_names.append(f'./data/raw/frames/frame-{i}.jpg')
        ref_frames = file_names[:-1]
        cur_frames = file_names[1:]
        return ref_frames, cur_frames

    def _build_labels(self):
        labels = []
        with open('./data/raw/train.txt', 'r') as f:
            for line in f:
                labels.append(np.float64(line.strip()))
        return labels[1:]

    def _build_dataset(self, ref_frames, cur_frames, labels):
        dataset = tf.data.Dataset.from_tensor_slices((ref_frames, cur_frames, labels))
        dataset = dataset.map(self._load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = dataset.take(self.train_len).shuffle(self.train_len).batch(self.batch_size)
        val_dataset = dataset.skip(self.train_len).batch(self.batch_size)
        return train_dataset, val_dataset

    @tf.function
    def _load_img(self, ref_frame, cur_frame, label):
        raw_ref_frame = tf.io.read_file(ref_frame)
        raw_cur_frame = tf.io.read_file(cur_frame)
        decoded_ref_frame = tf.io.decode_jpeg(raw_ref_frame)
        decoded_cur_frame = tf.io.decode_jpeg(raw_cur_frame)
        return decoded_ref_frame, decoded_cur_frame, label

    def _prepare(self, dataset, train):
        if train:
            dataset = dataset.cache().shuffle(data_len).batch(self.batch_size)
            dataset = dataset.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.batch(self.batch_size)
        return dataset

    def _normalize(self, input_, label):
        # Normalize input data:

        return input_, label

    def _augment_data(self, input_, label):
        # Augment training data here

        return input_, label

    @tf.function
    def _load_dataset(self, data_files):
        dataset = tf.data.Dataset.from_tensor_slices(data_files)
        dataset = dataset.map(self._load_datapoint,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    @tf.function
    def _load_datapoint(self, input_file, label_file):
        # Add code to load input and label file:

        return input_, label