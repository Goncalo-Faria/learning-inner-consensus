from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .core import layer
from .core import model
from .layers.capsule import CapsuleClassLayer, PrimaryCapsuleLayer, FullyConnectedCapsuleLayer


class CapsuleModel(model.Model):
    """
        A baseline Capsule multi GPU.
    """

    def __init__(self, hparams, name = "CapsuleModel"):
        self._pose_cache = []

        super(CapsuleModel, self).__init__(
            name,
            hparams)

    def _summarize_remakes(self, features, remakes):
        """Adds an image summary consisting original, target and remake images.

        Reshapes all images to 3D from flattened and transposes them to be in depth
        last order. For each target concats original, target and remake image
        vertically and concats all the target columns horizantally.
        Handles up to two targets.

        Args:
        features: A dictionary of input data containing the dimmension information
            and the input images.
        remakes: A list of network reconstructions.
        """
        image_dim = features['height']
        image_depth = features['depth']

        images = []
        images.append(features['images'])
        images.append(features['recons_image'])
        images += remakes
        if features['num_targets'] == 2:
            images.append(features['spare_image'])

        images_3d = []
        for image in images:
            image_3d = tf.reshape(
                image, [-1, image_depth, image_dim, image_dim])
            images_3d.append(tf.transpose(image_3d, [0, 2, 3, 1]))

        image_remake = tf.concat(images_3d[:3], axis=1)

        if features['num_targets'] == 2:
            # pylint: disable=unbalanced-tuple-unpacking
            original, _, _, remake, target = images_3d
            image_remake_2 = tf.concat([original, target, remake], axis=1)
            image_remake = tf.concat([image_remake, image_remake_2], axis=2)

        tf.summary.image('reconstruction', image_remake, 10)

    def _remake(self, features, capsule_embedding):
        """Adds the reconstruction subnetwork to build the remakes.

        This subnetwork shares the variables between different target remakes. It
        adds the subnetwork for the first target and reuses the weight variables
        for the second one.

        Args:
        features: A dictionary of input data containing the dimmension information
            and the input images and labels.
        capsule_embedding: A 3D tensor of shape [batch, 10, 16] containing
            network embeddings for each digit in the image if present.
        Returns:
        A list of network remakes of the targets.
        """
        num_pixels = features['depth'] * features['height'] * features['height']
        remakes = []
        targets = [(features['recons_label'], features['recons_image'])]
        if features['num_targets'] == 2:
            targets.append((features['spare_label'], features['spare_image']))

        for i in range(features['num_targets']):
            label, image = targets[i]
            remakes.append(
                layer.reconstruction(
                        capsule_mask=tf.one_hot(label, features['num_classes']),
                        num_atoms=np.prod(self._hparams.primary_parameters["pose_dim"]),
                        capsule_embedding=capsule_embedding,
                        layer_sizes=self._hparams.reconstruction_layer_sizes,
                        num_pixels=num_pixels,
                        reuse=(i > 0),
                        image=image,
                        balance_factor=0.0005))

        if self._hparams.verbose:
            self._summarize_remakes(features, remakes)

        return remakes

    def apply(self, features):
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

        if self._hparams.dataset != 'smallnorb':
            image_4d = tf.reshape(image, [-1, features['depth'], features['height'], features['width']])
            image_4d = tf.transpose(image_4d, [0, 2, 3, 1])
        else:
            image_4d = image

        lower_features = image_4d

        with tf.name_scope("derender/"):
            for i in range(len(self._hparams.derender_layers)):
                with tf.name_scope("layer" + str(i)):
                    higher_features = self._hparams.derender_layers[i](
                        lower_features)

                    lower_features = higher_features

        with tf.name_scope("bn/"):
            lower_features = tf.compat.v1.layers.BatchNormalization(
                center=False,
                trainable=self._hparams.bn_train)(
                    lower_features,
                    training=self._hparams.bn_train)

        with tf.name_scope("primarycapsules/"):
            primary_poses, primary_activations = PrimaryCapsuleLayer(
                self._hparams.primary_parameters["pose_dim"],
                self._hparams.primary_parameters["ksize"],
                self._hparams.primary_parameters["groups"],
            ).inference(lower_features)

        lower_poses, lower_activations = primary_poses, primary_activations

        for i in range(len(self._hparams.layers)):
            with tf.name_scope("layer" + str(i)+ "/") as scope:
                higher_poses, higher_activations = self._hparams.layers[i].inference(
                    (lower_poses, lower_activations))

                self._pose_cache.append(higher_poses)

                lower_poses = higher_poses
                lower_activations = higher_activations

        with tf.name_scope("fullyconnected/") as scope:
            fully_poses, fully_activations = FullyConnectedCapsuleLayer(
                transform=self._hparams.last_layer["transform"],
                routing=self._hparams.last_layer["routing"],
                name="last",
                activate=("softmax" != self._hparams.loss_type)
            ).inference((lower_poses, lower_activations))

        with tf.name_scope("classlayer/") as scope:
            final_poses, final_activations = CapsuleClassLayer(
                normalized=False,
                name=self.name + "Caps/"
            ).inference((fully_poses, fully_activations))

        if self._hparams.remake:
            final_shape = final_poses.shape.as_list()
            remake = self._remake(
                features,
                tf.reshape(
                    final_poses,
                    [-1, final_shape[1], np.prod(final_shape[2:])]
                )
            )
        else:
            remake = None

        return final_activations, remake

