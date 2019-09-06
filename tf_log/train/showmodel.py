
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

latest_ckp = tf.compat.v1.train.checkpoint_exists('CapsMLP/model.ckpt-65352')

print(latest_ckp)

print_tensors_in_checkpoint_file('CapsMLP/model.ckpt-65352', all_tensors=True, tensor_name='')
