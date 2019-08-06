from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf

from ..core.metric import Metric
from ..core.variables import bias_variable


class RoutingProcedure(object):
    """
        meta class for routing procedures.  
    """
    __metaclass__ = abc.ABCMeta
    count = 0
    def __init__(
            self,
            name,
            metric,
            initial_state,
            design_iterations,
            normalization = tf.nn.softmax,
            epsilon=1e-6,
            bias=False,
            verbose=False):
        self._iterations = design_iterations
        self._design_iterations = design_iterations
        self._verbose = verbose
        self._normalization = normalization
        self._epsilon = epsilon
        self._initial_state = initial_state
        self.name = name
        self._bias= bias
        self.metric = metric
        self.atoms = 0
        self._it = 0
        self._activate = True

        RoutingProcedure.count += 1
        assert isinstance(metric, Metric), \
            " metric must be instance of Metric metaclass. "

    @abc.abstractmethod
    def _compatibility(self, s, r, votes, poses, probabilities,activations, it):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }
        raise NotImplementedError('Not implemented')

    def compatibility(self, s, r, votes, poses, probabilities,activations, it):
        with tf.compat.v1.variable_scope('compatibility/', reuse=tf.compat.v1.AUTO_REUSE) as scope:
            return self._compatibility(s, r, votes, poses, probabilities, activations, it)

    @abc.abstractmethod
    def _activation(self, s, c, votes, poses):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## c :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }
        raise NotImplementedError('Not implemented')

    def activation(self, s, c, votes, poses):
        with tf.compat.v1.variable_scope('activation/', reuse=tf.compat.v1.AUTO_REUSE) as scope:
            return self._activation(s, c, votes, poses)

    @abc.abstractmethod
    def _initial_coefficients(self, r, activations):
        raise NotImplementedError('Not implemented')

    def _renormalizedDotProd(self, c, votes):
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim

        vshape = votes.shape.as_list()

        raw_poses = tf.reduce_sum(tf.multiply(c, votes), axis=4, keepdims=True)

        if self._bias:
            bvar = bias_variable(
                [1,vshape[1],vshape[2],vshape[3],1]+vshape[-2:],
                verbose=self._verbose,
                name="voting_bias"+str(RoutingProcedure.count)
            )
            raw_poses = raw_poses + bvar
        # raw_poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim

        poses = tf.divide(raw_poses, self._epsilon + self.metric.take(raw_poses))

        # poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim

        return poses

    def bound_activations(self):
        self._activate = True

    def unbound_activations(self):
        self._activate = False

    def fit(self, votes, activations, iterations = 0):
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## activations { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) } 
        self.atoms = votes.shape.as_list()[4]

        with tf.compat.v1.variable_scope('RoutingProcedure/' + self.name, reuse=tf.compat.v1.AUTO_REUSE):

            s = self._initial_state

            r = tf.zeros(shape=activations.shape.as_list() + [1, 1], dtype=tf.float32, name="compatibility_value")
            ## r { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

            activations = tf.reshape(activations, shape=activations.shape.as_list() + [1, 1])

            c = self._initial_coefficients(r, activations)
            ## c { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

            poses = self._renormalizedDotProd(c, votes)
            ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim

            probabilities = self.activation(s, c, votes, poses)
            ## probabilities :: { batch, output_atoms, new_w, new_h, 1 }

            if iterations == 0:
                self._iterations = self._design_iterations
            else :
                self._iterations = iterations

            for it in range(self._iterations):
                self._it = it

                r, s = self.compatibility(s, r, votes, poses, probabilities, activations, it)
                ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

                c = self._normalization(r, axis=-1)
                ## c :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

                poses = self._renormalizedDotProd(c, votes)
                ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim

                probabilities = self.activation(s, c, votes, poses)
                ## probabilities :: { batch, output_atoms, new_w, new_h, 1 }

            probabilities = tf.squeeze(probabilities, axis=[-2,-1])

            poses = tf.transpose(poses, [0, 4, 2, 3, 1, 5, 6])  ## output atoms become depth
            probabilities = tf.transpose(probabilities, [0, 4, 2, 3, 1])  ## output atoms become depth

            poses = tf.squeeze(poses, axis=[1])  ## remove output atoms dim
            probabilities = tf.squeeze(probabilities, axis=[1])  ## remove output atoms dim

            #if self._verbose:
                #tf.compat.v1.summary.histogram("RoutingProbabilities/" + self.name, probabilities)

            return poses, probabilities


class SimplifiedRoutingProcedure(RoutingProcedure):
    """
        meta class for routing procedures.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(
            self,
            name,
            metric,
            initial_state,
            design_iterations,
            normalization = tf.nn.softmax,
            epsilon=1e-6,
            bias=False,
            verbose=False):

        super(SimplifiedRoutingProcedure, self).__init__(
            name=name,
            metric=metric,
            initial_state=initial_state,
            design_iterations=design_iterations,
            normalization = normalization,
            epsilon=epsilon,
            bias=bias,
            verbose=verbose)

    def fit(self, votes, activations, iterations = 0):
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## activations { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }
        self.atoms = votes.shape.as_list()[4]

        with tf.compat.v1.variable_scope('SimplifiedRoutingProcedure/' + self.name, reuse=tf.compat.v1.AUTO_REUSE):

            s = self._initial_state

            r = tf.zeros(shape=activations.shape.as_list() + [1, 1], dtype=tf.float32, name="compatibility_value")
            ## r { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

            activations = tf.reshape(activations, shape=activations.shape.as_list() + [1, 1])

            c = self._initial_coefficients(r, activations)
            ## c { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

            poses = self._renormalizedDotProd(c, votes)
            ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim

            #probabilities = self.activation(s, c, votes, poses)
            ## probabilities :: { batch, output_atoms, new_w, new_h, 1 }

            if iterations == 0:
                self._iterations = self._design_iterations
            else :
                self._iterations = iterations

            for it in range(self._iterations):
                self._it = it

                r, s = self.compatibility(s, r, votes, poses, None, activations, it)
                ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

                c = self._normalization(r, axis=-1)
                ## c :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

                poses = self._renormalizedDotProd(c, votes)
                ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim


            probabilities = self.activation(s, c, votes, poses)
            ## probabilities :: { batch, output_atoms, new_w, new_h, 1 }

            probabilities = tf.squeeze(probabilities, axis=[-2,-1])

            poses = tf.transpose(poses, [0, 4, 2, 3, 1, 5, 6])  ## output atoms become depth
            probabilities = tf.transpose(probabilities, [0, 4, 2, 3, 1])  ## output atoms become depth

            poses = tf.squeeze(poses, axis=[1])  ## remove output atoms dim
            probabilities = tf.squeeze(probabilities, axis=[1])  ## remove output atoms dim

            #if self._verbose:
                #tf.compat.v1.summary.histogram("RoutingProbabilities/" + self.name, probabilities)

            return poses, probabilities
