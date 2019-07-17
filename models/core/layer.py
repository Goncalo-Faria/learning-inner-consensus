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

"""Library for capsule layers.

This has the layer implementation for coincidence detection, routing and
capsule layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
    """Penalizes deviations from margin for each logit.

    Each wrong logit costs its distance to margin. For negative logits margin is
    0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
    margin is 0.4 from each side.

    Args:
      labels: tensor, one hot encoding of ground truth.
      raw_logits: tensor, model predictions in range [0, 1]
      margin: scalar, the margin after subtracting 0.5 from raw_logits.
      downweight: scalar, the factor for negative cost.

    Returns:
      A tensor with cost for each data point of shape [batch_size].
    """
    logits = raw_logits - 0.5
    positive_cost = labels * tf.cast(tf.less(logits, margin),
                                     tf.float32) * tf.pow(logits - margin, 2)
    negative_cost = (1 - labels) * tf.cast(
        tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
    return 0.5 * positive_cost + downweight * 0.5 * negative_cost


def evaluate(logits, labels, num_targets, scope, loss_type, reg_const=0.0):
    """Calculates total loss and performance metrics like accuracy.

    Args:
      logits: tensor, output of the model.
      labels: tensor, ground truth of the data.
      num_targets: scalar, number of present objects in the image,
        i.e. the number of 1s in labels.
      scope: The scope to collect losses of.
      loss_type: 'sigmoid' (num_targets > 1), 'softmax' or 'margin' for margin
        loss.

    Returns:
      The total loss of the model, number of correct predictions and number of
      cases where at least one of the classes is correctly predicted.
    Raises:
      NotImplementedError: if the loss_type is not softmax or margin loss.
    """
    with tf.name_scope('loss'):
        if loss_type == 'sigmoid':
            classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels / 2.0, logits=logits)
        elif loss_type == 'softmax':
            classification_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
        elif loss_type == 'margin':
            classification_loss = _margin_loss(
                labels=tf.stop_gradient(labels), raw_logits=logits) + reg_const * tf.reduce_sum(
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        else:
            raise NotImplementedError('Not implemented')

        with tf.name_scope('total'):
            batch_classification_loss = tf.reduce_mean(classification_loss)
            tf.compat.v1.add_to_collection('losses', batch_classification_loss)
    tf.compat.v1.summary.scalar('batch_classification_cost', batch_classification_loss)

    all_losses = tf.compat.v1.get_collection('losses', scope)
    total_loss = tf.add_n(all_losses, name='total_loss')
    tf.compat.v1.summary.scalar('total_loss', total_loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            _, targets = tf.nn.top_k(labels, k=num_targets)
            _, predictions = tf.nn.top_k(logits, k=num_targets)
            missed_targets = tf.compat.v1.sets.difference(
                targets, predictions)
            num_missed_targets = tf.compat.v1.sets.set_size(missed_targets)
            correct = tf.equal(num_missed_targets, 0)
            almost_correct = tf.less(num_missed_targets, num_targets)
            correct_sum = tf.reduce_sum(tf.cast(correct, tf.float32))
            almost_correct_sum = tf.reduce_sum(
                tf.cast(almost_correct, tf.float32))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.compat.v1.summary.scalar('accuracy', accuracy)
    tf.compat.v1.summary.scalar('correct_prediction_batch', correct_sum)
    tf.compat.v1.summary.scalar('almost_correct_batch', almost_correct_sum)
    return total_loss, correct_sum, almost_correct_sum


def reconstruction(capsule_mask, num_atoms, capsule_embedding, layer_sizes,
                   num_pixels, reuse, image, balance_factor):
    """Adds the reconstruction loss and calculates the reconstructed image.

    Given the last capsule output layer as input of shape [batch, 10, num_atoms]
    add 3 fully connected layers on top of it.
    Feeds the masked output of the model to the reconstruction sub-network.
    Adds the difference with reconstruction image as reconstruction loss to the
    loss collection.

    Args:
      capsule_mask: tensor, for each data in the batch it has the one hot
        encoding of the target id.
      num_atoms: scalar, number of atoms in the given capsule_embedding.
      capsule_embedding: tensor, output of the last capsule layer.
      layer_sizes: (scalar, scalar), size of the first and second layer.
      num_pixels: scalar, number of pixels in the target image.
      reuse: if set reuse variables.
      image: The reconstruction target image.
      balance_factor: scalar, downweight the loss to be in valid range.

    Returns:
      The reconstruction images of shape [batch_size, num_pixels].
    """
    first_layer_size, second_layer_size = layer_sizes
    capsule_mask_3d = tf.expand_dims(capsule_mask, -1)
    atom_mask = tf.tile(capsule_mask_3d, [1, 1, num_atoms])
    filtered_embedding = capsule_embedding * atom_mask
    filtered_embedding_2d = tf.compat.v1.layers.Flatten()(filtered_embedding)
    model = tf.keras.Sequential()
    for num in layer_sizes:
        model.add(
            tf.keras.layers.Dense(
                num,
                activation='relu',
                use_bias=True,
                kernel_initializer=tf.compat.v1.keras.initializers.truncated_normal(
                    stddev=0.1,
                    dtype=tf.float32
                ),
                bias_initializer=tf.compat.v1.keras.initializers.constant(0.1)
            )
        )
    model.add(
        tf.keras.layers.Dense(
            num_pixels,
            activation='sigmoid',
            use_bias=True,
            kernel_initializer=tf.compat.v1.keras.initializers.truncated_normal(
                stddev=0.1,
                dtype=tf.float32
            ),
            bias_initializer=tf.compat.v1.keras.initializers.constant(0.1)
        )
    )
    reconstruction_2d = model(filtered_embedding_2d)

    with tf.name_scope('loss'):
        image_2d = tf.compat.v1.layers.Flatten()(image)
        distance = tf.pow(reconstruction_2d - image_2d, 2)
        loss = tf.reduce_sum(distance, axis=-1)
        batch_loss = tf.reduce_mean(loss)
        balanced_loss = balance_factor * batch_loss
        tf.compat.v1.add_to_collection('losses', balanced_loss)
        tf.compat.v1.summary.scalar('reconstruction_error', balanced_loss)

    return reconstruction_2d
