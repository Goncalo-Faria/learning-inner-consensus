# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convolutional Model class.

Uses only convolutional and fully connected layers for the inference ops.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .core.model import Model, Inferred
from .core import variables


class ConvModel(Model):
    """A baseline multi GPU Model without capsule layers.

    The inference graph includes ReLU convolution layers and fully connected
    layers. The last layer is linear and has 10 units.
    """

    def __init__(self, hparams, verbose=False, name="Convmodel"):

        super(ConvModel, self).__init__(
            name=name,
            hparams=hparams)
        self.verbose = verbose

    def _add_convs(self, input_tensor, channels):
        """Adds the convolution layers.

        Adds a series of convolution layers with ReLU nonlinearity and pooling
        after each of them.

        Args:
          input_tensor: a 4D float tensor as the input to the first convolution.
          channels: A list of channel sizes for input_tensor and following
            convolution layers. Number of channels in input tensor should be
            equal to channels[0].
        Returns:
          A 4D tensor as the output of the last pooling layer.
        """
        for i in range(1, len(channels)):
            with tf.compat.v1.variable_scope('conv{}'.format(i)) as scope:
                kernel = variables.weight_variable(
                    shape=[5, 5, channels[i - 1], channels[i]], stddev=5e-2,
                    verbose=self._hparams.verbose
                )
                conv = tf.nn.conv2d(
                    input_tensor,
                    kernel, [1, 1, 1, 1],
                    padding="VALID",
                    data_format='NHWC')
                biases = variables.bias_variable([channels[i]],
                                                 verbose=self._hparams.verbose)
                pre_activation = tf.nn.bias_add(
                    conv, biases, data_format='NHWC')
                relu = tf.nn.relu(pre_activation, name=scope.name)
                if self._hparams.verbose:
                    tf.summary.histogram('activation', relu)
                input_tensor = tf.compat.v1.keras.layers.MaxPool2D(
                     pool_size=(2, 2), strides=(2, 2), data_format='channels_last', padding='same')(relu)

        return input_tensor

    def inference(self, features):
        """Adds the inference graph ops.

        Builds the architecture of the neural net to drive logits from features.
        The inference graph includes a series of convolution and fully connected
        layers and outputs a [batch, 10] tensor as the logits.

        Args:
          features: Dictionary of batched feature tensors like images and labels.
        Returns:
          A model.Inferred named tuple of expected outputs of the model like
          'logits' and 'remakes' for the reconstructions (to be added).
        """
        image = features['images']
        image_dim = features['height']
        image_depth = features['depth']
        image_4d = tf.reshape(image, [-1, image_depth, image_dim, image_dim])

        image_4d = tf.transpose(image_4d, [0, 2, 3, 1])

        conv = self._add_convs(image_4d, [image_depth, 128, 64])
        hidden1 = tf.compat.v1.keras.layers.Flatten()(conv)

        with tf.compat.v1.variable_scope('fc1') as scope:
            dim = hidden1.shape.as_list()[-1]
            weights = variables.weight_variable(shape=[dim, 1024], stddev=0.1,
                                                verbose=self._hparams.verbose)
            biases = variables.bias_variable(shape=[1024],
                                             verbose=self._hparams.verbose)
            pre_activation = tf.matmul(hidden1, weights) + biases
            hidden2 = tf.nn.relu(pre_activation, name=scope.name)

        with tf.compat.v1.variable_scope('softmax_layer') as scope:
            weights = variables.weight_variable(
                shape=[1024, features['num_classes']], stddev=0.1,
                verbose=self._hparams.verbose
            )
            biases = variables.bias_variable(shape=[features['num_classes']],
                                             verbose=self._hparams.verbose)
            logits = tf.matmul(hidden2, weights) + biases

        return logits, None
