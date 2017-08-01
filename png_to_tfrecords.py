from PIL import Image
import os
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

original_images = []

TF_RECORDS_FILENAME = 'Datasets/Data/nyu/tfrecords/nyu.tfrecords'

WRITER = tf.python_io.TFRecordWriter(TF_RECORDS_FILENAME)

IMAGE_PATH = 'Datasets/Data/nyu/images/'
DEPTH_PATH = 'Datasets/Data/nyu/depths/'

IMAGES = [IMAGE_PATH + f for f in os.listdir(IMAGE_PATH)]
DEPTHS = [DEPTH_PATH + f for f in os.listdir(DEPTH_PATH)]

FILENAME_PAIRS = zip(IMAGES, DEPTHS)

for img_path, depth_path in FILENAME_PAIRS:
    img = np.array(Image.open(img_path))
    depth = np.array(Image.open(depth_path))

    original_images.append((img, depth))

    image_raw = img.tostring()
    depth_raw = depth.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        'depth_raw': _bytes_feature(depth_raw)}))
    WRITER.write(example.SerializeToString())

WRITER.close()
