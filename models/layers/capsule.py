from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import abc
import numpy as np

class CapsuleLayer(object):
    def __init__(
        self, 
        routing,
        transform, 
        ksizes,
        name,
        padding = "VALID",
        strides=[1,1,1,1]):        
        self._routing = routing
        self._ksizes = ksizes, 
        self._strides = strides
        self._transform = transform
        self._padding = padding
        self.name = name
        self._representation_dim = []

    def _receptivefield(self, input_tensor):
        ##  input_tensor == {  batch, w , h , depth } + repdim , {batch, w, h,  depth }

        poses , activations = input_tensor

        poses_shape = poses.shape.as_list()

        poses = tf.transpose(poses, [0, 4, 5, 1, 2, 3 ])

        ## poses = { batch } + repdim + { w , h , depth }

        condensed_poses = tf.reshape(poses, shape = [poses_shape[0], np.prod(self._representation_dim)] + poses_shape[1:4] )

        ## condensed_poses = { batch, np.prod(repdim), w , h, depth }

        patched_poses = tf.extract_volume_patches(
            input = condensed_poses, 
            ksizes = [self._ksizes[0], np.prod(self._representation_dim)] + self._ksizes[1:],
            strides = [self._strides[0], 1] + self._strides[1:],
            padding= self._padding
        )

        patched_activations = tf.compat.v1.extract_image_patches(
            input = activations, 
            ksizes = self._ksizes,
            strides = self._strides,
            padding= self._padding
        )
        ## patched_poses { batch, np.prod(repdim), new_w , new_h, depth * np.prod(ksizes) }
        ## patched_activations { batch, new_w , new_h, depth * np.prod(ksizes) }

        patched_poses = tf.transpose(patched_poses, [0, 2, 3, 4, 1])

        ## patched_poses { batch, new_w , new_h, depth * np.prod(ksizes), np.prod(repdim) }
        patched_poses_shape = patched_poses.shape.as_list()

        patched_poses = tf.reshape(patched_poses, patched_poses_shape[:5] + self._representation_dim)
        ## patched_poses { batch, new_w , new_h, depth * np.prod(ksizes)} + repdim }
    
        return patched_poses, patched_activations

    def inference(self, input_tensor):
        ##  input_tensor == {  batch, w , h , depth } + repdim , {batch, w, h,  depth }
        """Adds the inference graph ops."""

        self._representation_dim = input_tensor[0].shape.as_list()[4:]

        with tf.name_scope('CapsuleLayer' + self.name) as scope:
            poses, activations = self._receptivefield(input_tensor)
            """
                stacks the multiple possible receptive fields of capsules. 

            """
            ## poses { batch, new_w , new_h, depth * np.prod(ksizes)} + repdim
            ## activations { batch, new_w , new_h, depth * np.prod(ksizes) } 
      
            votes, activations = self._transform.translate(poses, activations)
            """
                transforms the poses to the output's capsule space.
            """
            ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
            ## activations { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

            higher_poses, higher_activations = self._routing.fit(votes, activations)
            """
                determines the pose and activation of the output capsules.
            """

            ## higher_poses :: { batch, new_w, new_h, output_atoms } + repdim
            ## higher_activations :: { batch, new_w, new_h,output_atoms }
            

            return higher_poses, higher_activations