#pylint: disable=E0611,E0401,C0325
import os
import random

import tensorflow as tf

import Datasets.TransmissionNet.simulator as simulator

import dataset

class DatasetTransmission(dataset.Dataset):
    def __init__(self):
        parameters_list = ["tfr_path", "input_size", "output_size", "turbidity_path",
                           "turbidity_size", "patch_size"]
        self.open_config(parameters_list)
        self.batch_size = self.config_dict["batch_size"]
        self.input_size = self.config_dict["input_size"]
        self.patch_size = self.config_dict["patch_size"]
        self.input_size_prod = self.input_size[0] * self.input_size[1] * self.input_size[2]
        self.output_size = self.config_dict["output_size"]
        self.output_size_prod = self.output_size[0] * self.output_size[1] * self.output_size[2]
        self.tfr_path = self.config_dict["tfr_path"]
        self.train_file = [self.tfr_path + f for f in os.listdir(self.tfr_path)]
        self.num_file = len(self.train_file)
        self.validation_file = [self.tfr_path + f for f in os.listdir(self.tfr_path)]
        self.num_files_val = len(self.validation_file)
        #Simulator attributes
        self.turbidity_path = self.config_dict["turbidity_path"]
        self.turbidity_size = tuple(self.config_dict["turbidity_size"])
        self.sess = tf.Session()
        _, self.binf, _ = simulator.acquireProperties(
            self.turbidity_path, self.turbidity_size, self.batch_size, 0, 0, self.sess)

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
            })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self.input_size_prod])
        image = tf.cast(image, tf.float32) * (1. / 255)

        return image

    def next_batch_train(self):
        """
        args:
            batch_size:
                number of examples per returned batch
            num_epochs:
                number of time to read the input data

        returns:
            a tuple(image, transmissions) where:
                image is a float tensor with shape [batch size] + patch_size
                transmissions is a float tensor with shape [batch size]
        """

        filename = self.train_file
        print(filename)

        filename_queue = tf.train.string_input_producer(
            filename, num_epochs=self.config_dict["num_epochs"])

        image = self.read_and_decode(filename_queue)

        images = tf.train.shuffle_batch(
            [image], batch_size=self.config_dict["batch_size"],
            num_threads=self.config_dict["num_threads"],
            capacity=100+ 3 * self.config_dict["batch_size"],
            min_after_dequeue=100
            )
        images = tf.reshape(images, [self.batch_size] + self.input_size)

        size_x = self.config_dict['patch_size'][0]
        size_y = self.config_dict['patch_size'][1]
        offset_x = random.randint(0, self.input_size[0] - size_x - 1)
        offset_y = random.randint(0, self.input_size[0] - size_y - 1)

        images = images[:, offset_x:offset_x + size_x, offset_y:offset_y+size_y]
        #TODO(Rael): minval/maxval podem ficar no config
        transmissions = tf.random_uniform([self.batch_size], minval=0.05, maxval=1)
        images = simulator.applyTurbidityTransmission(images, self.binf, transmissions)
        print(images.shape)
        print(transmissions.shape)
        tf.summary.image("image", images)
        return images, transmissions
