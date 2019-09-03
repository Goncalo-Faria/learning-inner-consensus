
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

latest_ckp = tf.compat.v1.train.checkpoint_exists('../mnist/train/KernelNet/model.ckpt-99990')

print(latest_ckp)

print_tensors_in_checkpoint_file('../mnist/train/KernelNet/model.ckpt-99990', all_tensors=True, tensor_name='')
