"""Utility functions for declaring variables and adding summaries.

It adds all different scalars and histograms for each variable and provides
utility functions for weight and bias variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def weight_variable(shape, stddev=0.1, verbose=False, name = "", regularizer = None, reuse = False):
  """Creates a CPU variable with normal initialization. Adds summaries.

  Args:
    shape: list, the shape of the variable.
    stddev: scalar, standard deviation for the initilizer.
    verbose: if set add histograms.

  Returns:
    Weight variable tensor of shape=shape.
  """
  with tf.device('/cpu:0'):
    with tf.compat.v1.variable_scope('weights',reuse = reuse):
      weights = tf.compat.v1.get_variable(
          'weights'+name,
          shape,
          initializer=tf.compat.v1.truncated_normal_initializer(
              stddev=stddev, dtype=tf.float32),
          dtype=tf.float32,
          regularizer=regularizer)
  variable_summaries(weights, verbose)
  return weights


def bias_variable(shape, verbose=False, name =""):
  """Creates a CPU variable with constant initialization. Adds summaries.

  Args:
    shape: list, the shape of the variable.
    verbose: if set add histograms.

  Returns:
    Bias variable tensor with shape=shape.
  """
  with tf.device('/cpu:0'):
    with tf.compat.v1.variable_scope('biases'):
      biases = tf.compat.v1.get_variable(
          'biases'+name,
          shape,
          initializer=tf.compat.v1.constant_initializer(0.1),
          dtype=tf.float32)
  variable_summaries(biases, verbose)
  return biases


def variable_summaries(var, verbose):
  """Attaches a lot of summaries to a Tensor (for TensorBoard visualization).

  Args:
    var: tensor, statistic summaries of this tensor is added.
    verbose: if set add histograms.
  """
  if verbose:
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
  else:
    pass


def activation_summary(x, verbose):
  """Creates summaries for activations.

  Creates a summary that provides a histogram and sparsity of activations.

  Args:
    x: Tensor
    verbose: if set add histograms.
  """
  if verbose:
    tf.summary.histogram('activations', x)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))
  else:
    pass
