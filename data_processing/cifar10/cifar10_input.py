
"""Input utility functions for reading Cifar10 dataset.

Handles reading from Cifar10 dataset saved in binary original format. Scales and
normalizes the images as the preprocessing step. It can distort the images by
random cropping and contrast adjusting.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


def _read_input(filename_queue):
  """Reads a single record and converts it to a tensor.

  Each record consists the 3x32x32 image with one byte for the label.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
      image: a [32, 32, 3] float32 Tensor with the image data.
      label: an int32 Tensor with the label in the range 0..9.
  """
  label_bytes = 1
  height = 32
  depth = 3
  image_bytes = height * height * depth
  record_bytes = label_bytes + image_bytes

  reader = tf.compat.v1.FixedLengthRecordReader(record_bytes=record_bytes)
  _, byte_data = reader.read(filename_queue)
  uint_data = tf.io.decode_raw(byte_data, tf.uint8)

  label = tf.cast(tf.strided_slice(uint_data, [0], [label_bytes]), tf.int32)
  label.set_shape([1])

  depth_major = tf.reshape(
      tf.strided_slice(uint_data, [label_bytes], [record_bytes]),
      [depth, height, height])
  image = tf.cast(tf.transpose(a=depth_major, perm=[1, 2, 0]), tf.float32)

  return image, label


def _distort_resize(image, image_size):
  """Distorts input images for CIFAR training.

  Adds standard distortions such as flipping, cropping and changing brightness
  and contrast.

  Args:
    image: A float32 tensor with last dimmension equal to 3.
    image_size: The output image size after cropping.

  Returns:
    distorted_image: A float32 tensor with shape [image_size, image_size, 3].
  """
  distorted_image = tf.image.random_crop(image, [image_size, image_size, 3])
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(
      distorted_image, lower=0.2, upper=1.8)
  distorted_image.set_shape([image_size, image_size, 3])
  return distorted_image


def _batch_features(image, label, batch_size, split, image_size):
  """Constructs the batched feature dictionary.

  Batches the images and labels accourding to the split. Shuffles the data only
  if split is train. Formats the feature dictionary to be in the format required
  by experiment.py.

  Args:
    image: A float32 tensor with shape [image_size, image_size, 3].
    label: An int32 tensor with the label of the image.
    batch_size: The number of data points in the output batch.
    split: 'train' or 'test'.
    image_size: The size of the input image.

  Returns:
    batched_features: A dictionary of the input data features.
  """
  image = tf.transpose(a=image, perm=[2, 0, 1])
  features = {
      'images': image,
      'labels': tf.one_hot(label, 10),
      'recons_image': image,
      'recons_label': label,
  }
  if split == 'train':
    batched_features = tf.compat.v1.train.shuffle_batch(
        features,
        batch_size=batch_size,
        num_threads=16,
        capacity=10000 + 3 * batch_size,
        min_after_dequeue=10000)
  else:
    batched_features = tf.compat.v1.train.batch(
        features,
        batch_size=batch_size,
        num_threads=1,
        capacity=10000 + 3 * batch_size)
  batched_features['labels'] = tf.reshape(batched_features['labels'],
                                          [batch_size, 10])
  batched_features['recons_label'] = tf.reshape(
      batched_features['recons_label'], [batch_size])
  batched_features['height'] = image_size
  batched_features['width'] = image_size
  batched_features['depth'] = 3
  batched_features['num_targets'] = 1
  batched_features['num_classes'] = 10
  return batched_features


def inputs(split, data_dir, batch_size):
  """Constructs input for CIFAR experiment.

  Args:
    split: 'train' or 'test', which split of the data set to read from.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    batched_features: A dictionary of the input data features.
  """
  if split == 'train':
    filenames = [
        os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)
    ]
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]

  filename_queue = tf.compat.v1.train.string_input_producer(filenames)
  float_image, label = _read_input(filename_queue)

  image_size = 24

  if split == 'train':
    resized_image = _distort_resize(float_image, image_size)
  else:
    resized_image = tf.image.resize_with_crop_or_pad(
        float_image, image_size, image_size)
  image = tf.image.per_image_standardization(resized_image)

  return _batch_features(image, label, batch_size, split, image_size)
