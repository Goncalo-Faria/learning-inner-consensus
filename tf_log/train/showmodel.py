
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

latest_ckp = tf.compat.v1.train.checkpoint_exists('KernelNet/model.ckpt-14400')

print(latest_ckp)

print_tensors_in_checkpoint_file('KernelNet/model.ckpt-14400', all_tensors=True, tensor_name='')
