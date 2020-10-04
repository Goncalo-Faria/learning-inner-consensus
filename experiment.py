
"""Framework for training and evaluating models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import time

import numpy as np
import tensorflow as tf
import wandb

import architectures.cap_em as EM
import architectures.cap_block_net as BlockNet
import architectures.cap_kernel as KernelBaseline
import architectures.cap_mlp as CapsMLP
import architectures.cap_mlp_shared as CapsMLPShared
import architectures.cap_nin as CapNIN
import architectures.cap_dyn as CapDynamic
import models.capsulemodel as capm
from data_processing.cifar10 import cifar10_input
from data_processing.mnist import mnist_input_record
from data_processing.smallnorb import smallnorb_input_record
from models import convmodel
from models.coreimp.kernelmix import MonoKernelMix
from models.coreimp.commonKernels import GaussianKernel, SpectralMixture

# from models import conv_model
#import architectures.convnet as ConvNet

#wandb.init(project="Inner-consensus", sync_tensorboard=True)

parser = argparse.ArgumentParser(prog='Experiment', add_help=True)

parser.add_argument('--data_dir', default=None,
                    help='The data directory.',type=str)
parser.add_argument('--eval_size', default=10000,
                    help='Size of the test dataset.', type=int)
parser.add_argument('--learning_rate', default=0.001,
                    help='Size of the test dataset.', type=float)
parser.add_argument('--batch_size', default=16,
                    help='Batch size.', type=int)
parser.add_argument('--max_steps', default=1000,
                    help='Number of steps to train.', type=int)
parser.add_argument('--model', default='capsule',
                    help='The model to use for the experiment. capsule or baseline', type=str)
parser.add_argument('--dataset', default='mnist', type=str,
                    help='The dataset to use for the experiment.mnist, norb, cifar10.')
parser.add_argument('--num_gpus', default=1,
                    type=int,help='Number of gpus to use.')
parser.add_argument('--num_targets', default=1,
                    type=int,help='Number of targets to detect (1 or 2).')
parser.add_argument('--regulizer_constant',default=0.0,
                    type=float,help='scale of the sum of the regularizes.')
parser.add_argument('--num_trials', default=1,
                    type=int, help='Number of trials for ensemble evaluation.')
parser.add_argument('--num_saves', default=10,
                    type=int,help='number of checkpoints.')
parser.add_argument('--show_step',default=5,
                    type=int,help='How often to print.')
parser.add_argument('--summary_dir', default="",
                    type=str, help='Main directory for the experiments.')
parser.add_argument('--checkpoint', default=None,
                    type=str, help='The model checkpoint for evaluation.')
parser.add_argument('--remake', default=False,
                    type=bool,help='use reconstruction as regulizer.')
parser.add_argument('--train', default=False,
                    type=bool,help='Either train the model or test the model.')
parser.add_argument('--validate', default=False,
                    type=bool,help='Run trianing/eval in validation mode.')
parser.add_argument('--budget_threshold', default=0.9,
                    type=float,help='model saving threshold')
parser.add_argument('--num_classes',default=10,
                    type=int,help='number of classes in the dataset.')
parser.add_argument('--degree',default=None,
                    type=int,help='lstm degree.')
parser.add_argument('--verbose', default=False,
                    type=bool, help='Register model info.')
parser.add_argument('--loss_type', default='softmax',
                    type=str,help=' classfication head. ')
parser.add_argument('--track', default=False,
                    type=bool,help='plot history.')
parser.add_argument('--bn_train', default=False,
                    type=bool,help='adjust bn parameters.')
parser.add_argument('--min_history', default=0,
                    type=int,help='minimum plotted history.')

GLOBAL_HPAR = parser.parse_args()

models = {
    "CapsuleNin": capm.CapsuleModel,
    "ConvNet" : convmodel.ConvModel,
    "CapsuleBaseline": capm.CapsuleModel,
    "KernelNet" : capm.CapsuleModel,
    "KernelNetSpectral" : capm.CapsuleModel,
    "CapsMLPShared" : capm.CapsuleModel,
    "CapsMLP" : capm.CapsuleModel,
    "CapDynamic" : capm.CapsuleModel
}


def get_features(split, total_batch_size, num_gpus, data_dir, num_targets,
                 dataset, validate=False):
    """Reads the input data and distributes it over num_gpus GPUs.

    Each tower of data has 1/FLAGS.num_gpus of the total_batch_size.

    Args:
      split: 'train' or 'test', split of the data to read.
      total_batch_size: total number of data entries over all towers.
      num_gpus: Number of GPUs to distribute the data on.
      data_dir: Directory containing the input data.
      num_targets: Number of objects present in the image.
      dataset: The name of the dataset, either norb or mnist.
      validate: If set, subset training data into training and test.

    Returns:
      A list of batched feature dictionaries.

    Raises:
      ValueError: If dataset is not mnist or norb.
    """

    batch_size = total_batch_size // max(1, num_gpus)
    features = []
    for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
            if dataset == 'mnist':
                features.append(
                    mnist_input_record.inputs(
                        data_dir=data_dir,
                        batch_size=batch_size,
                        split=split,
                        num_targets=num_targets,
                        validate=validate,
                    ))
            elif dataset == 'cifar10':
                #data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
                features.append(
                    cifar10_input.inputs(
                        split=split, data_dir=data_dir, batch_size=batch_size))
            elif dataset == 'smallnorb':
                features.append(
                    smallnorb_input_record.inputs(
                        split=split, data_dir=data_dir, batch_size=batch_size))
            else:
                raise ValueError(
                    'Unexpected dataset {!r}, must be mnist, norb, or cifar10.'.format(
                        dataset))
    return features


def extract_step(path):
    """Returns the step from the file format name of Tensorflow checkpoints.

    Args:
      path: The checkpoint path returned by tf.train.get_checkpoint_state.
        The format is: {ckpnt_name}-{step}

    Returns:
      The last training step number of the checkpoint.
    """
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])


def load_training(saver, session, load_dir):
    """Loads a saved model into current session or initializes the directory.

    If there is no functioning saved model or FLAGS.restart is set, cleans the
    load_dir directory. Otherwise, loads the latest saved checkpoint in load_dir
    to session.

    Args:
      saver: An instance of tf.train.saver to load the model in to the session.
      session: An instance of tf.Session with the built-in model graph.
      load_dir: The directory which is used to load the latest checkpoint.

    Returns:
      The latest saved step.
    """
    if tf.io.gfile.exists(load_dir):
        ckpt = tf.train.get_checkpoint_state(load_dir)
        if ckpt and ckpt.model_checkpoint_path:

            saver.restore(session, ckpt.model_checkpoint_path)
            prev_step = extract_step(ckpt.model_checkpoint_path)

        else:
            tf.io.gfile.rmtree(load_dir)
            tf.io.gfile.makedirs(load_dir)
            prev_step = 0
    else:
        tf.io.gfile.makedirs(load_dir)
        prev_step = 0
    return prev_step


def train_experiment(session, result, writer, last_step, max_steps, saver,
                     summary_dir, num_saves, budget_threshold):
    """Runs training for up to max_steps and saves the model and summaries.

    Args:
      session: The loaded tf.session with the initialized model.
      result: The resultant operations of the model including train_op.
      writer: The summary writer file.
      last_step: The last trained step.
      max_steps: Maximum number of training iterations.
      saver: An instance of tf.train.saver to save the current model.
      summary_dir: The directory to save the model in it.
      num_saves: number of saved models model ckpt.
    """
    for i in range(last_step, max_steps):
        summary, _ , correts = session.run([result.summary, result.train_op, result.correct])

        if i % GLOBAL_HPAR.show_step == 0 :
            print(str(i) + " ------- "+ str(correts))

        #wandb.log({"correts": correts/ GLOBAL_HPAR.batch_size})

        writer.add_summary(summary, i)
        if ((i + 1) / max_steps) >= budget_threshold :
            if (i + 1) % round((1-budget_threshold)*max_steps/num_saves) == 0:
                saver.save(
                    session, os.path.join(summary_dir, 'model.ckpt'), global_step=i + 1)
                wandb.save( summary_dir + "model.ckpt-"+str(i+1)+".index")
                wandb.save( summary_dir + "model.ckpt-"+str(i+1)+".meta")
                wandb.save( summary_dir + "model.ckpt-"+str(i+1)+".data-00000-of-00001")
                wandb.save( summary_dir + "events.out.tfevents.*.*")
                wandb.save( summary_dir + "checkpoint")


def load_eval(saver, session, load_dir):
    """Loads the latest saved model to the given session.

    Args:
      saver: An instance of tf.train.saver to load the model in to the session.
      session: An instance of tf.Session with the built-in model graph.
      load_dir: The path to the latest checkpoint.

    Returns:
      The latest saved step.
    """
    saver.restore(session, load_dir)
    print('model loaded successfully')
    return extract_step(load_dir)


def eval_experiment(session, result, writer, last_step, max_steps, **kwargs):
    """Evaluates the current model on the test dataset once.

    Evaluates the loaded model on the test data set with batch sizes of 100.
    Aggregates the results and writes one summary point to the summary file.

    Args:
      session: The loaded tf.session with the trained model.
      result: The resultant operations of the model including evaluation metrics.
      writer: The summary writer file.
      last_step: The last trained step.
      max_steps: Maximum number of evaluation iterations.
      **kwargs: Arguments passed by run_experiment but not used in this function.
    """
    del kwargs

    total_correct = 0
    total_almost = 0
    summary_i=""

    ######################
    ######################
    ######################
    ######################

    max_steps = 1

    for _ in range(max_steps):
        summary_i, correct, almost = session.run(
            [result.summary, result.correct, result.almost])
        total_correct += correct
        total_almost += almost

    total_false = max_steps * GLOBAL_HPAR.batch_size - total_correct
    total_almost_false = max_steps * GLOBAL_HPAR.batch_size - total_almost
    summary = tf.compat.v1.Summary.FromString(summary_i)
    summary.value.add(tag='correct_prediction', simple_value=total_correct)
    summary.value.add(tag='wrong_prediction', simple_value=total_false)
    summary.value.add(
        tag='almost_wrong_prediction', simple_value=total_almost_false)
    print('Total wrong predictions: {}, wrong percent: {}%'.format(
        total_false, total_false /(max_steps*GLOBAL_HPAR.batch_size) ))
    tf.compat.v1.logging.info('Total wrong predictions: {}, wrong percent: {}%'.format(
        total_false, total_false / max_steps))
    writer.add_summary(summary, last_step)


def run_experiment(loader,
                   load_dir,
                   writer,
                   experiment,
                   result,
                   max_steps,
                   num_saves=0,
                   budget_threshold=0.6):
    """Starts a session, loads the model and runs the given experiment on it.

    This is a general wrapper to load a saved model and run an experiment on it.
    An experiment can be a training experiment or an evaluation experiment.
    It starts session, threads and queues and closes them before returning.

    Args:
      loader: A function of prototype (saver, session, load_dir) to load a saved
        checkpoint in load_dir given a session and saver.
      load_dir: The directory to load the previously saved model from it and to
        save the current model in it.
      writer: A tf.summary.FileWriter to add summaries.
      experiment: The function of prototype (session, result, writer, last_step,
        max_steps, saver, load_dir, save_step) which will execute the experiment
        steps from result on the given session.
      result: The resultant final operations of the built model.
      max_steps: Maximum number of experiment iterations.
      save_step: How often the training model should be saved.
    """
    session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                       tf.compat.v1.local_variables_initializer())
    session.run(init_op)
    saver = tf.compat.v1.train.Saver(max_to_keep=1000)
    last_step = loader(saver, session, load_dir)
    coord = tf.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=session, coord=coord)
    try:
        experiment(
            session=session,
            result=result,
            writer=writer,
            last_step=last_step,
            max_steps=max_steps,
            saver=saver,
            summary_dir=load_dir,
            num_saves=num_saves,
            budget_threshold=budget_threshold)
    except tf.errors.OutOfRangeError as excpt:
        print( excpt.message )
        tf.compat.v1.logging.info('Finished experiment.')
    finally:
        coord.request_stop()
    coord.join(threads)
    session.close()


def train(hparams, summary_dir, num_gpus, model_type, max_steps,
          data_dir, num_targets, dataset, validate):
    """Trains a model with batch sizes of 128 to FLAGS.max_steps steps.

    It will initialize the model with either previously saved model in the
    summary directory or start from scratch if FLAGS.restart is set or the
    directory is empty.
    The training is distributed on num_gpus GPUs. It writes a summary at every
    step and saves the model every 1500 iterations.

    Args:
      hparams: The hyper parameters to build the model graph.
      summary_dir: The directory to save model and write training summaries.
      num_gpus: Number of GPUs to use for reading data and computation.
      model_type: The model architecture category.
      max_steps: Maximum number of training iterations.
      save_step: How often the training model should be saved.
      data_dir: Directory containing the input data.
      num_targets: Number of objects present in the image.
      dataset: Name of the dataset for the experiments.
      validate: If set, use training-validation set for training.
    """
    summary_dir += '/train/' + "/" + hparams.model + "/"
    with tf.Graph().as_default():
        # Build model
        features = get_features('train', hparams.batch_size, num_gpus, data_dir, num_targets,
                                dataset, validate)
        model = models[model_type](hparams)
        result, _, _ = model.multi_gpu(features, num_gpus)
        # Print stats
        param_stats = tf.compat.v1.profiler.profile(
            graph=tf.compat.v1.get_default_graph())
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
        writer = tf.compat.v1.summary.FileWriter(summary_dir)
        run_experiment(load_training, summary_dir, writer, train_experiment, result,
                       max_steps, hparams.num_saves, hparams.budget_threshold)
        writer.close()


def find_checkpoint(load_dir, seen_step):
    """Finds the global step for the latest written checkpoint to the load_dir.

    Args:
      load_dir: The directory address to look for the training checkpoints.
      seen_step: Latest step which evaluation has been done on it.
    Returns:
      The latest new step in the load_dir and the file path of the latest model
      in load_dir. If no new file is found returns -1 and None.

    """
    ckpt = tf.train.get_checkpoint_state(load_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = extract_step(ckpt.model_checkpoint_path)
        if int(global_step) != seen_step:
            return int(global_step), ckpt.model_checkpoint_path
    return -1, None


def evaluate(hparams, summary_dir, num_gpus, model_type, eval_size, data_dir,
             num_targets, dataset, validate, checkpoint=None):
    """Continuously evaluates the latest trained model or a specific checkpoint.

    Regularly (every 2 min, maximum 6 hours) checks the training directory for
    the latest model. If it finds any new model, it outputs the total number of
    correct and wrong predictions for the test data set to the summary file.
    If a checkpoint is provided performs the evaluation only on the specific
    checkpoint.

    Args:
      hparams: The hyperparameters for building the model graph.
      summary_dir: The directory to load training model and write test summaries.
      num_gpus: Number of GPUs to use for reading data and computation.
      model_type: The model architecture category.
      eval_size: Total number of examples in the test dataset.
      data_dir: Directory containing the input data.
      num_targets: Number of objects present in the image.
      dataset: The name of the dataset for the experiment.
      validate: If set, use validation set for continuous evaluation.
      checkpoint: (optional) The checkpoint file name.
    """
    load_dir = summary_dir + '/train/' + "/" + hparams.model + "/"
    summary_dir += '/test/'
    with tf.Graph().as_default():
        features = get_features('test', hparams.batch_size, num_gpus, data_dir, num_targets,
                                dataset, validate)
        model = models[model_type](hparams)
        result, _ , _ = model.multi_gpu(features, num_gpus)
        test_writer = tf.compat.v1.summary.FileWriter(summary_dir)
        seen_step = -1
        paused = 0
        while paused < 360:
            print('start evaluation, model defined')
            if checkpoint:
                step = extract_step(checkpoint)
                last_checkpoint = checkpoint
            else:
                step, last_checkpoint = find_checkpoint(load_dir, seen_step)
            if step == -1:
                time.sleep(60)
                paused += 1
            else:
                paused = 0
                seen_step = step
                run_experiment(load_eval, last_checkpoint, test_writer, eval_experiment,
                               result, eval_size // hparams.batch_size)
                if checkpoint:
                    break

        test_writer.close()


def get_placeholder_data(num_steps, batch_size, features, session):
    """Reads the features into a numpy array and replaces them with placeholders.

    Loads all the images and labels of the features queue in memory. Replaces
    the feature queue reader handle with placeholders to switch input method from
    queue to placeholders. Using placeholders gaurantees the order of datapoints
    to stay exactly the same during each epoch.

    Args:
      num_steps: The number of times to read from the features queue.
      batch_size: The number of datapoints at each step.
      features: The dictionary containing the data queues such as images.
      session: The session handle to use for running tensors.

    Returns:
      data: List of numpy arrays containing all the queued data in features.
      targets: List of all the labels in range [0...num_classes].
    """
    image_size = features['height']
    depth = features['depth']
    num_classes = features['num_classes']
    data = []
    targets = []
    for i in range(num_steps):
        data.append(
            session.run({
                'recons_label': features['recons_label'],
                'labels': features['labels'],
                'images': features['images'],
                'recons_image': features['recons_image']
            }))
        targets.append(data[i]['recons_label'])
    image_shape = (batch_size, image_size, image_size, depth)
    features['images'] = tf.compat.v1.placeholder(tf.float32, shape=image_shape)
    features['labels'] = tf.compat.v1.placeholder(
        tf.float32, shape=(batch_size, num_classes))
    features['recons_image'] = tf.compat.v1.placeholder(tf.float32, shape=image_shape)
    features['recons_label'] = tf.compat.v1.placeholder(tf.int32, shape=batch_size)
    return data, targets


def infer_ensemble_logits(features, model, checkpoints, session, num_steps,
                          data):
    """Extracts the logits for the whole dataset and all the trained models.

    Loads all the checkpoints. For each checkpoint stores the logits for the whole
    dataset.

    Args:
      features: The dictionary of the input handles.
      model: The model operation graph.
      checkpoints: The list of all checkpoint paths.
      session: The session handle to use for running tensors.
      num_steps: The number of steps to run the experiment.
      data: The num_steps list of loaded data to be fed to placeholders.

    Returns:
      logits: List of all the final layer logits for different checkpoints.
    """
    _, inferred, _ = model.multi_gpu([features], 1)
    logits = []
    saver = tf.compat.v1.train.Saver()
    for checkpoint in checkpoints:
        saver.restore(session, checkpoint)
        for i in range(num_steps):
            logits.append(
                session.run(
                    inferred[0].logits,
                    feed_dict={
                        features['recons_label']: data[i]['recons_label'],
                        features['labels']: data[i]['labels'],
                        features['images']: data[i]['images'],
                        features['recons_image']: data[i]['recons_image']
                    }))
    return logits


def infer_ensemble_accuracy(features, model, checkpoints, session, num_steps,
                          data):
    """Extracts the logits for the whole dataset and all the trained models.

    Loads all the checkpoints. For each checkpoint stores the logits for the whole
    dataset.

    Args:
      features: The dictionary of the input handles.
      model: The model operation graph.
      checkpoints: The list of all checkpoint paths.
      session: The session handle to use for running tensors.
      num_steps: The number of steps to run the experiment.
      data: The num_steps list of loaded data to be fed to placeholders.

    Returns:
      logits: List of all the final layer logits for different checkpoints.
    """
    _, inferred, correct = model.multi_gpu([features], 1)
    corrects = []
    saver = tf.compat.v1.train.Saver()
    for checkpoint_info in checkpoints:

        step, checkpoint = checkpoint_info
        saver.restore(session, checkpoint)
        corrects_checkpoint = []
        for i in range(num_steps):
            corrects_checkpoint.append(
                session.run(
                    correct[0],
                    feed_dict={
                        features['recons_label']: data[i]['recons_label'],
                        features['labels']: data[i]['labels'],
                        features['images']: data[i]['images'],
                        features['recons_image']: data[i]['recons_image']
                    }))

        model_corrects = np.sum(corrects_checkpoint)

        corrects.append(
            model_corrects
        )

        wandb.log({"step": step, "corrects": model_corrects})

    return corrects


def evaluate_ensemble(hparams, model_type, eval_size, data_dir, num_targets,
                      dataset, checkpoint, num_trials):
    """Evaluates an ensemble of trained models.

    Loads a series of checkpoints and aggregates the output logit of them on the
    test data. Selects the class with maximum aggregated logit as the prediction.
    Prints the total number of wrong predictions.

    Args:
      hparams: The hyperparameters for building the model graph.
      model_type: The model architecture category.
      eval_size: Total number of examples in the test dataset.
      data_dir: Directory containing the input data.
      num_targets: Number of objects present in the image.
      dataset: The name of the dataset for the experiment.
      checkpoint: The file format of the checkpoints to be loaded.
      num_trials: Number of trained models to ensemble.
    """

    checkpointsname = []
    f = open(GLOBAL_HPAR.summary_dir + "/train/" + hparams.model + "/" + "checkpoint")
    for line in f:
        m = re.search('(?<=(?P<quote>["])).*(?P=quote)', line)
        checkpointsname.append(m.group(0)[:-1])

    checkpointsname = checkpointsname[::-1][:num_trials]

    checkpoints = []
    for file_name in checkpointsname:
        if tf.compat.v1.train.checkpoint_exists(GLOBAL_HPAR.summary_dir + "/train/" + hparams.model + "/" + file_name):
            checkpoints.append(GLOBAL_HPAR.summary_dir + "/train/" + hparams.model + "/" + file_name)

    with tf.Graph().as_default():
        features = get_features('test', hparams.batch_size, 1, data_dir, num_targets,
                                dataset)[0]
        model = models[model_type](hparams)

        session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=session, coord=coord)
        num_steps = eval_size // hparams.batch_size
        data, targets = get_placeholder_data(num_steps, hparams.batch_size, features,
                                             session)

        logits = infer_ensemble_logits(features, model, checkpoints, session,
                                       num_steps, data)
        coord.request_stop()
        coord.join(threads)
        session.close()

        logits = np.reshape(logits, (num_trials, num_steps, hparams.batch_size, -1))
        logits = np.sum(logits, axis=0)
        predictions = np.argmax(logits, axis=2)
        total_wrong = np.sum(np.not_equal(predictions, targets))
        print('Total wrong predictions: {}, wrong percent: {}%'.format(
            total_wrong, total_wrong / eval_size * 100))


def evaluate_history(hparams, model_type, eval_size, data_dir, num_targets,
                      dataset, checkpoint, num_trials, dataset_type="test"):
    """Evaluates an ensemble of trained models.

    Loads a series of checkpoints and aggregates the output logit of them on the
    test data. Selects the class with maximum aggregated logit as the prediction.
    Prints the total number of wrong predictions.

    Args:
      hparams: The hyperparameters for building the model graph.
      model_type: The model architecture category.
      eval_size: Total number of examples in the test dataset.
      data_dir: Directory containing the input data.
      num_targets: Number of objects present in the image.
      dataset: The name of the dataset for the experiment.
      checkpoint: The file format of the checkpoints to be loaded.
      num_trials: Number of trained models to ensemble.
    """

    checkpointsname = []
    f = open(GLOBAL_HPAR.summary_dir + "/train/" + hparams.model + "/" + "ls.txt")
    for line in f:
        model_number = int(line.split(" ", 2)[0])

        if model_number >= GLOBAL_HPAR.min_history :
            checkpointsname.append(model_number)

    checkpointsname = list(set(checkpointsname))
    checkpointsname.sort()

    checkpoints = []
    for model_number in checkpointsname:
        fname = GLOBAL_HPAR.summary_dir \
                + "/train/" + hparams.model + \
                "/" + "model.ckpt-" + str(model_number)

        if tf.compat.v1.train.checkpoint_exists(fname):
            checkpoints.append((model_number,fname))

    with tf.Graph().as_default():
        features = get_features(dataset_type, hparams.batch_size, 1, data_dir, num_targets,
                                dataset)[0]
        model = models[model_type](hparams)

        session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=session, coord=coord)
        num_steps = eval_size // hparams.batch_size
        data, targets = get_placeholder_data(num_steps, hparams.batch_size, features,
                                             session)

        corrects = infer_ensemble_accuracy(features, model, checkpoints, session,
                                       num_steps, data)
        coord.request_stop()
        coord.join(threads)
        session.close()

        #corrects_acc = corrects / eval_size * 100

        #for i in range(corrects):
        #    print(corrects_acc[i])


def main(_):

    global GLOBAL_HPAR

    if GLOBAL_HPAR.summary_dir == "" :
        GLOBAL_HPAR.summary_dir = wandb.run.dir

    if GLOBAL_HPAR.model == "CapsuleBlockNet":
        GLOBAL_HPAR = BlockNet.setup(GLOBAL_HPAR)

    elif GLOBAL_HPAR.model == "CapsuleBaseline":
        GLOBAL_HPAR = EM.setup(GLOBAL_HPAR)

    elif GLOBAL_HPAR.model == "ConvNet":
        print(GLOBAL_HPAR.model)

    elif GLOBAL_HPAR.model == "KernelNet":
        GLOBAL_HPAR = KernelBaseline.setup(GLOBAL_HPAR,
            GaussianKernel(GLOBAL_HPAR.verbose,
                           singular=False)
            )
    elif GLOBAL_HPAR.model == "KernelNetSpectral":
        GLOBAL_HPAR = KernelBaseline.setup(GLOBAL_HPAR,
            SpectralMixture(GLOBAL_HPAR.verbose)
            )

    elif GLOBAL_HPAR.model == "CapsMLP":
        GLOBAL_HPAR = CapsMLP.setup(GLOBAL_HPAR)

    elif GLOBAL_HPAR.model == "CapsMLPShared":
        GLOBAL_HPAR = CapsMLPShared.setup(GLOBAL_HPAR)

    elif GLOBAL_HPAR.model == "CapsuleNin":
        GLOBAL_HPAR = CapNIN.setup(GLOBAL_HPAR)

    elif GLOBAL_HPAR.model == "CapDynamic":
        GLOBAL_HPAR = CapDynamic.setup(GLOBAL_HPAR)

    print("Hyper Parameters")
    print(GLOBAL_HPAR)
    if GLOBAL_HPAR.train:
        wandb.init(project="Gulbenkian", name=GLOBAL_HPAR.model + "/train_experiment", sync_tensorboard=True, dir=".")
        train(GLOBAL_HPAR, GLOBAL_HPAR.summary_dir, GLOBAL_HPAR.num_gpus, GLOBAL_HPAR.model,
              GLOBAL_HPAR.max_steps, GLOBAL_HPAR.data_dir, GLOBAL_HPAR.num_targets,
              GLOBAL_HPAR.dataset, GLOBAL_HPAR.validate)
    else:
        if GLOBAL_HPAR.track:
            wandb.init(project="Gulbenkian", name=GLOBAL_HPAR.model + "/history", sync_tensorboard=True, dir=".")
            evaluate_history(GLOBAL_HPAR, GLOBAL_HPAR.model, GLOBAL_HPAR.eval_size, GLOBAL_HPAR.data_dir,
                              GLOBAL_HPAR.num_targets, GLOBAL_HPAR.dataset, GLOBAL_HPAR.checkpoint,
                              GLOBAL_HPAR.num_trials)
        else:
            wandb.init(project="Gulbenkian", name=GLOBAL_HPAR.model +"/evaluation", sync_tensorboard=True, dir=".")
            if GLOBAL_HPAR.num_trials == 1:
                evaluate(GLOBAL_HPAR, GLOBAL_HPAR.summary_dir, GLOBAL_HPAR.num_gpus, GLOBAL_HPAR.model,
                     GLOBAL_HPAR.eval_size, GLOBAL_HPAR.data_dir, GLOBAL_HPAR.num_targets,
                     GLOBAL_HPAR.dataset, GLOBAL_HPAR.validate, GLOBAL_HPAR.checkpoint)
            else:
                evaluate_ensemble(GLOBAL_HPAR, GLOBAL_HPAR.model, GLOBAL_HPAR.eval_size, GLOBAL_HPAR.data_dir,
                              GLOBAL_HPAR.num_targets, GLOBAL_HPAR.dataset, GLOBAL_HPAR.checkpoint,
                              GLOBAL_HPAR.num_trials)


if __name__ == '__main__':
    tf.compat.v1.app.run()
