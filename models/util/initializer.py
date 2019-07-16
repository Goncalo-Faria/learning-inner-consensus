import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes


class IdentityRandomUniform(tf.initializers.Initializer):
  """Initializer that generates tensors with a uniform distribution.
  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate.
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate.  Defaults to 1 for float types.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer.
  """

  def __init__(self, minval=0, maxval=None, seed=None, dtype=dtypes.float32):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    id = tf.eye(num_rows=shape[-2],num_columns=shape[-1],batch_shape=shape[:5], dtype=dtype)
    init = id + random_ops.random_uniform(
        shape, self.minval, self.maxval, dtype, seed=self.seed)
    return init

  def get_config(self):
    return {
        "minval": self.minval,
        "maxval": self.maxval,
        "seed": self.seed,
        "dtype": self.dtype.name
    }
