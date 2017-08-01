import sys
import os

import tensorflow as tf

import dataset

class DatasetTfRecords(dataset.Dataset):
    def __init__(self):
        parameters_list = ["tfr_path", "input_size", "output_size"]
        self.open_config(parameters_list)
        self.tfr_path = self.config_dict["tfr_path"]
        self.train_file = [self.tfr_path + f for f in os.listdir(self.tfr_path)]
        self.num_file = len(self.train_file)
        self.validation_file = [self.tfr_path + f for f in os.listdir(self.tfr_path)]
        self.num_files_val = len(self.validation_file)

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([], tf.string),
            })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, self.config_dict["input_size"])

        depth = tf.decode_raw(features['depth'], tf.uint8)
        depth = tf.reshape(depth, self.config_dict["depth_size"])

        return image, depth

    def next_batch_train(self):
        """
        args:
            train:
                true: training
                false: validation
            batch_size:
                number of examples per returned batch
            num_epochs:
                number of time to read the input data

        returns:
            a tuple(image, depths) where:
                image is a float tensor with shape [batch size] + input_size
                depth is a float tensor with shape [batch size] + depth_size
        """

        filename = os.path.join(self.tfr_path, self.train_file[0])

        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=self.config_dict["num_epochs"])

            image, depth = self.read_and_decode(filename_queue)

            images, depths = tf.train.shuffle_batch(
                [image, depth], batch_size=self.config_dict["batch_size"],
                num_threads=self.config_dict["num_threads"],
                capacity=1000 + 3 * self.config_dict["batch_size"],
                min_after_dequeue=1000)
            return images, depths
