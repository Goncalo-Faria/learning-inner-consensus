
"""Basic Model class which provides the optimization wrap around inference.

A general model class wrapps the loss and optimizer operations around the
inference graph. Therefore, it should define the learning rate, global step
and optimizer. The current version supports a multiple gpu scenario by
enforcing single gpu to select one gpu for calculations and reuse variables.

Different models will only have different inference graphs and they share the
training and evaluation ops. Therefore, we define the inference function
as an abstract method that each model should define specifically for itself.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import tensorflow as tf

from . import layer

TowerResult = collections.namedtuple('TowerResult', ('inferred', 'almost',
                                                     'correct', 'grads'))
JoinedResult = collections.namedtuple('JoinedResult', ('summary', 'train_op',
                                                       'correct', 'almost'))
Inferred = collections.namedtuple('Inferred',
                                  ('logits', 'remakes'))


class Model(object):
    """Base class for building a model and running inference on it."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, hparams):
        """Initializes the model parameters.

    Args:
      hparams: The hyperparameters for the model as tf.contrib.training.HParams.
    """
        self.name = hparams.model + name
        self._hparams = hparams
        with tf.device('/cpu:0'):
            self._global_step = tf.compat.v1.get_variable(
                'global_step', [],
                initializer=tf.compat.v1.constant_initializer(0),
                trainable=False)

            step = (self._global_step / hparams.max_steps)

            init_fact = tf.minimum( 10 * tf.sigmoid(-2 * (self._global_step / 2000) ) + 0.1, 1 )

            mom_sche = (0.85 - 0.95) * ( step - 1 ) / (0 - 1) + 0.95

            learning_rate = hparams.learning_rate * (1 - step ) * init_fact

            learning_rate = tf.maximum(learning_rate, 1e-8)

            lr = tf.compat.v1.train.exponential_decay(
                                        learning_rate=hparams.learning_rate,
                                        global_step=self._global_step,
                                        decay_steps=20000,
                                        decay_rate=0.96)

            #self._optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, mom_sche)
            #self._optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

            self._optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, epsilon=1e-9, )

    def inference(self, features):

        with tf.compat.v1.variable_scope('Capsule/' + self.name, reuse=tf.compat.v1.AUTO_REUSE):
            final_activations, remake = self.apply(features)

            return Inferred(final_activations, remake)

    @abc.abstractmethod
    def apply(self, features):
        """Adds the inference graph ops.

    Builds the architecture of the neural net to derive logits from features.
    The inference graph defined here should involve trainable variables
    otherwise the optimizer will raise a ValueError.

    Args:
    features: Dictionary of batched feature tensors like images and labels.
    Returns:
    An Inferred named tuple for expected outputs of the model like 'logits'
    and 'remakes' for the reconstructions.
    """
        raise NotImplementedError('Not implemented')

    def _single_tower(self, tower_ind, feature):
        """Calculates the model gradient for one tower.

    Adds the inference and loss operations to the graph. Calculates the
    gradients based on the loss. Appends all the output values of this tower to
    their respective lists.

    Args:
      tower_ind: The index number for this tower. Each tower is named as
                  tower_{tower_ind} and resides on gpu:{tower_ind}.
      feature: Dictionary of batched features like images and labels.
    Returns:
      A namedtuple TowerResult containing the inferred values like logits and
      reconstructions, gradients and evaluation metrics.
    """
        with tf.device('/gpu:%d' % tower_ind):## gpu
            with tf.name_scope('tower_%d' % (tower_ind)) as scope:
                inferred = self.inference(feature)
                losses, correct, almost = layer.evaluate(
                    logits=inferred.logits,
                    labels=feature['labels'],
                    num_targets=feature['num_targets'],
                    scope=scope,
                    loss_type=self._hparams.loss_type,
                    reg_const=self._hparams.regulizer_constant,
                    remake = self._hparams.remake,
                    max_steps = self._hparams.max_steps)
                tf.compat.v1.get_variable_scope().reuse_variables()
                grads = self._optimizer.compute_gradients(losses)

        return TowerResult(inferred, almost, correct, grads)

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each variable across all towers.

    Args:
      tower_grads: List of gradient lists for each tower. Each gradient list
        is a list of (gradient, variable) tuples for all variables.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been
      averaged across all towers.
    """
        average_grads = []
        for grads_and_vars in zip(*tower_grads):

            #print(grads_and_vars)
            grads = tf.stack([g for g, _ in grads_and_vars])
            grad = tf.reduce_mean(grads, 0)
            v = grads_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _summarize_towers(self, almosts, corrects, tower_grads):
        """Aggregates the results and gradients over all towers.

    Args:
      almosts: The number of almost correct samples for each tower.
      corrects: The number of correct samples for each tower.
      tower_grads: The gradient list for each tower.

    Returns:
      A JoinedResult of evaluation results, the train op and the summary op.
    """

        grads = self._average_gradients(tower_grads)
        train_op = self._optimizer.apply_gradients(
            grads, global_step=self._global_step)
        summaries = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)
        summary = tf.compat.v1.summary.merge(summaries)
        #summary = tf.compat.v1.summary.merge_all()
        stacked_corrects = tf.stack(corrects)
        stacked_almosts = tf.stack(almosts)
        summed_corrects = tf.reduce_sum(stacked_corrects, 0)
        summed_almosts = tf.reduce_sum(stacked_almosts, 0)
        return JoinedResult(summary, train_op, summed_corrects, summed_almosts)

    def multi_gpu(self, features, num_gpus):
        """Build the Graph and add the train ops on multiple GPUs.

    Divides the inference and gradient computation on multiple gpus.
    Then aggregates the gradients and return the resultant ops.

    Args:
      features: A list of dictionary of different features of input data.
                len(features) should be at least num_gpus.
      num_gpus: Number of gpus to be distributed on.
    Returns:
      A tuple of JoinedResult output Ops to be called in Session.run for
      training, evaluation or visualization, such as train_op and merged
      summary and a list of inferred outputs of each tower.
    """
        almosts = []
        corrects = []
        tower_grads = []
        inferred = []
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            for i in range(num_gpus):
                tower_output = self._single_tower(i, features[i])
                inferred.append(tower_output.inferred)
                almosts.append(tower_output.almost)
                corrects.append(tower_output.correct)
                tower_grads.append(tower_output.grads)

        summarized_results = self._summarize_towers(almosts, corrects, tower_grads)
        return summarized_results, inferred, corrects
