from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..core.routing import SimplifiedRoutingProcedure
from ..core.variables import weight_variable, bias_variable


class KernelRouting(SimplifiedRoutingProcedure):
    def norm(self, vector,axis):
        return (vector/(tf.reduce_sum(vector, axis=axis,keepdims=True) + self._epsilon))

    def __init__(
            self,
            kernel,
            metric,
            iterations,
            name="",
            verbose=False):
        self._kernel = kernel
        self._agreement = None

        super(KernelRouting, self).__init__(
            name="KernelRouting" + name,
            metric=metric,
            design_iterations=iterations,
            initial_state=None,
            verbose=verbose,
            normalization = self.norm)

    def _compatibility(self, s, r, votes, poses, probabilities, activations, it):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        alpha = weight_variable([1, votes.shape[1], 1, 1, 1,1,1],
                                name= "lambda1",
                                verbose = self._verbose,
                                initializer=tf.compat.v1.keras.initializers.normal(mean=12))

        beta = weight_variable([1, votes.shape[1], 1, 1, 1,1,1],
                                name= "lambda2",
                                verbose = self._verbose,
                                initializer=tf.compat.v1.keras.initializers.constant(value=1))

        alpha = tf.abs(alpha)
        beta = tf.abs(beta)

        poses_tiled = tf.tile(poses, [1, 1, 1, 1, self.atoms, 1, 1])

        self._agreement = self._kernel.take(poses_tiled, votes)

        lambda_o = beta + alpha + self._epsilon
        r = tf.pow(activations + self._epsilon * 1e5, beta/lambda_o ) * tf.exp( 1/lambda_o * self._agreement )

        c = self._normalization(r, axis=4)
        
        return c, s

    def _activation(self, s, c, votes, poses, activations):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## c :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        raw = tf.reduce_sum(tf.multiply(c, self._agreement), axis=-3, keepdims=True)

        if self._verbose:
            tf.compat.v1.summary.histogram(self.name + "dist_" + str(self._it), self._agreement)

        ## raw :: { batch, output_atoms, new_w, new_h, 1 } 
        rs = [1, raw.shape[2], 1, 1, 1]

        theta1 = weight_variable(rs, name="theta1", verbose=self._verbose)
        theta2 = bias_variable(rs, name="theta2", verbose=self._verbose, initializer=tf.compat.v1.constant_initializer(1))
        theta3 = weight_variable(rs, name="theta3", verbose=self._verbose)

        extra = tf.reduce_sum(c * tf.math.log(activations), axis=-3, keepdims=True)
      
        if self._activate :
            activation = tf.sigmoid(theta1 * raw + tf.abs(theta3)*extra + theta2)
        else:
            activation = theta1 * raw + theta2
        ## activation :: { batch, output_atoms, new_w, new_h, 1 } 

        return activation
