"""Input utility functions for reading small norb dataset.

Handles reading from small norb dataset saved in binary original format. Scales and
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
      'labels': tf.one_hot(label, 5),
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
                                          [batch_size, 5])
  batched_features['recons_label'] = tf.reshape(
      batched_features['recons_label'], [batch_size])
  batched_features['height'] = image_size
  batched_features['width'] = image_size
  batched_features['depth'] = 3
  batched_features['num_targets'] = 1
  batched_features['num_classes'] = 5
  return batched_features

import os

def _parser(serialized_example):
    """Parse smallNORB example from tfrecord.

    Args:
      serialized_example: serialized example from tfrecord
    Returns:
      img: image
      lab: label
      cat:
        category
        the instance in the category (0 to 9)
      elv:
        elevation
        the elevation (0 to 8, which mean cameras are 30,
        35,40,45,50,55,60,65,70 degrees from the horizontal respectively)
      azi:
        azimuth
        the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in
        degrees)
      lit:
        lighting
        the lighting condition (0 to 5)
    """

    features = tf.compat.v1.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.compat.v1.FixedLenFeature([], tf.string),
            'label': tf.compat.v1.FixedLenFeature([], tf.int64),
            'category': tf.compat.v1.FixedLenFeature([], tf.int64),
            'elevation': tf.compat.v1.FixedLenFeature([], tf.int64),
            'azimuth': tf.compat.v1.FixedLenFeature([], tf.int64),
            'lighting': tf.compat.v1.FixedLenFeature([], tf.int64),
        })

    img = tf.compat.v1.decode_raw(features['img_raw'], tf.float64)
    img = tf.reshape(img, [96, 96, 1])
    img = tf.cast(img, tf.float32)  # * (1. / 255) # left unnormalized

    lab = tf.cast(features['label'], tf.int32)
    cat = tf.cast(features['category'], tf.int32)
    elv = tf.cast(features['elevation'], tf.int32)
    azi = tf.cast(features['azimuth'], tf.int32)
    lit = tf.cast(features['lighting'], tf.int32)

    return img, lab, cat, elv, azi, lit


def _train_preprocess(img, lab, cat, elv, azi, lit):
    """Preprocessing for training.

    Preprocessing from Hinton et al. (2018) "Matrix capsules with EM routing."
    Hinton2018: "We downsample smallNORB to 48 × 48 pixels and normalize each
    image to have zero mean and unit variance. During training, we randomly crop
    32 × 32 patches and add random brightness and contrast to the cropped images.
    During test, we crop a 32 × 32 patch from the center of the image and
    achieve..."

    Args:
      img: this fn only works on the image
      lab, cat, elv, azi, lit: allow these to pass through
    Returns:
      img: image processed
      lab, cat, elv, azi, lit: allow these to pass through
    """

    img = img / 255.
    img = tf.compat.v1.image.resize_images(img, [48, 48])
    img = tf.image.per_image_standardization(img)
    img = tf.compat.v1.random_crop(img, [32, 32, 1])
    img = tf.image.random_brightness(img, max_delta=2.0)
    # original 0.5, 1.5
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)

    # Original
    # image = tf.image.random_brightness(image, max_delta=32. / 255.)
    # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # image = tf.image.resize_images(image, [48, 48])
    # image = tf.random_crop(image, [32, 32, 1])

    return img, lab, cat, elv, azi, lit


def _val_preprocess(img, lab, cat, elv, azi, lit):
    """Preprocessing for validation/testing.

    Preprocessing from Hinton et al. (2018) "Matrix capsules with EM routing."
    Hinton2018: "We downsample smallNORB to 48 × 48 pixels and normalize each
    image to have zero mean and unit variance. During training, we randomly crop
    32 × 32 patches and add random brightness and contrast to the cropped
    images. During test, we crop a 32 × 32 patch from the center of the image
    and achieve..."

    Args:
      img: this fn only works on the image
      lab, cat, elv, azi, lit: allow these to pass through
    Returns:
      img: image processed
      lab, cat, elv, azi, lit: allow these to pass through
    """

    img = img / 255.
    img = tf.compat.v1.image.resize_images(img, [48, 48])
    img = tf.image.per_image_standardization(img)
    img = tf.slice(img, [8, 8, 0], [32, 32, 1])


    # Original
    # image = tf.image.resize_images(image, [48, 48])
    # image = tf.slice(image, [8, 8, 0], [32, 32, 1])

    return img, lab, cat, elv, azi, lit


def input_fn(path, is_train: bool, batch_size = 64, epochs=100):
    """Input pipeline for smallNORB using tf.data.

    Author:
      Ashley Gritzman 15/11/2018
    Args:
      is_train:
    Returns:
      dataset: image tf.data.Dataset
    """

    import re
    if is_train:
        CHUNK_RE = re.compile(r"train.*\.tfrecords")
    else:
        CHUNK_RE = re.compile(r"test.*\.tfrecords")

    chunk_files = [os.path.join(path, fname)
                   for fname in os.listdir(path)
                   if CHUNK_RE.match(fname)]

    # 1. create the dataset
    dataset = tf.data.TFRecordDataset(chunk_files)

    # 2. map with the actual work (preprocessing, augmentation…) using multiple
    # parallel calls
    dataset = dataset.map(_parser, num_parallel_calls=4)
    if is_train:
        dataset = dataset.map(_train_preprocess,
                              num_parallel_calls=4)
    else:
        dataset = dataset.map(_val_preprocess,
                              num_parallel_calls=4)

    # 3. shuffle (with a big enough buffer size)
    # In response to a question on OpenReview, Hinton et al. wrote the
    # following:
    # https://openreview.net/forum?id=HJWLfGWRb&noteId=rJgxonoNnm
    # "We did not have any special ordering of training batches and we random
    # shuffle. In terms of TF batch:
    # capacity=2000 + 3 * batch_size, ensures a minimum amount of shuffling of
    # examples. min_after_dequeue=2000."
    capacity = 2000 + 3 * batch_size
    dataset = dataset.shuffle(buffer_size=capacity)

    # 4. batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # 5. repeat
    dataset = dataset.repeat(count=epochs)

    # 6. prefetch
    dataset = dataset.prefetch(1)

    return dataset


def create_inputs_norb(path, is_train: bool,batch_size,epochs):
    """Get a batch from the input pipeline.

    Author:
      Ashley Gritzman 15/11/2018
    Args:
      is_train:
    Returns:
      img, lab, cat, elv, azi, lit:
    """

    # Create batched dataset
    dataset = input_fn(path, is_train,batch_size=batch_size, epochs=epochs)

    # Create one-shot iterator
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    img, lab, cat, elv, azi, lit = iterator.get_next()

    output_dict = {'image': img,
                   'label': lab,
                   'category': cat,
                   'elevation': elv,
                   'azimuth': azi,
                   'lighting': lit}

    return output_dict


def inputs(data_dir,
           batch_size,
           split,
           epochs=50):

    dict = create_inputs_norb(data_dir, split == "train",batch_size=batch_size, epochs=epochs)

    batched_features={}

    batched_features['height'] = 32
    batched_features['width'] = 32
    batched_features['depth'] = 1
    batched_features['num_targets'] = 1
    batched_features['num_classes'] = 5
    batched_features['recons_image'] = dict['image']
    batched_features['recons_label'] = dict['label']
    batched_features['images'] = dict['image']
    batched_features['labels'] = tf.one_hot(dict['label'], 5)

    return batched_features