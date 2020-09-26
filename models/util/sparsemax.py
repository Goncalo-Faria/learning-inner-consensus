import tensorflow as tf
import numpy as np


def smforward(z):
    """forward pass for sparsemax
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector.
    """

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return np.maximum(0, z - tau_z)


def smjacobian(z):
    """jacobian for sparsemax
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector.
    """

    # Construct S(z)
    # Possibly this could be reduced to just calculating k(z)
    p = forward(z)
    s = p > 0
    s_float = s.astype('float64')

    # row-wise outer product
    # http://stackoverflow.com/questions/31573856/theano-row-wise-outer-product-between-two-matrices
    jacobian = s_float[:, :, np.newaxis] * s_float[:, np.newaxis, :]
    jacobian /= - np.sum(s, axis=1)[:, np.newaxis, np.newaxis]

    # add delta_ij
    obs, index = s.nonzero()
    jacobian[obs, index, index] += 1

    return jacobian


def smRop(z, v):
    """Jacobian vector product (Rop) for sparsemax
    This calculates [J(z_i) * v_i, ...]. `z` is a 2d-array, where axis 1
    (each row) is assumed to be the the z-vector. `v` is a matrix where
    axis 1 (each row) is assumed to be the `v-vector`.
    """

    # Construct S(z)
    p = forward(z)
    s = p > 0

    # Calculate \hat{v}, which will be a vector (scalar for each z)
    v_hat = np.sum(v * s, axis=1) / np.sum(s, axis=1)

    # Calculates J(z) * v
    return s * (v - v_hat[:, np.newaxis])



def smlforward_loss(z, spm, q):
    """Calculates the sparsemax loss function
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector. q is a binary matrix of same shape, containing the labels
    """

    # Calculate q^T * z
    z_k = np.sum(q * z, axis=1)

    # calculate sum over S(z)
    p = spm
    s = p > 0
    # z_i^2 - tau(z)^2 = p_i (2 * z_i - p_i) for i \in S(z)
    S_sum = np.sum(s * p * (2 * z - p), axis=1)

    # because q is binary, sum([q_1^2, q_2^2, ...]) is just sum(q)
    q_norm = np.sum(q, axis=1)

    return -z_k + 0.5 * S_sum + 0.5 * q_norm


def smlgrad(z, q):
    return -q + smforward(z)


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        result = tf.compat.v1.py_func(func, inp, Tout, stateful=stateful, name=name)
        return result


def grad_sparsemax(op, grad):
    spm = op.outputs[0]
    support = tf.cast(spm > 0, spm.dtype)

    # Calculate \hat{v}, which will be a vector (scalar for each z)
    v_hat = tf.reduce_sum(tf.mul(grad, support), 1) / tf.reduce_sum(support, 1)

    # Calculates J(z) * v
    return [support * (grad - v_hat[:, np.newaxis])]


def grad_sparsemax_loss(op, grad):
    spm = op.inputs[1]
    labels = op.inputs[2]
    result = tf.transpose(grad * tf.transpose(-labels + spm))
    return [result, None, None]


def sparsemax(Z, name=None):
    with tf.compat.v1.keras.backend.name_scope(name, "SparseMaxGrad", [Z]) as name:
        # py_func takes a list of tensors and a
        # function that takes np arrays as inputs
        # and returns np arrays as outputs
        sparsemax_forward = py_func(
            smforward,
            [Z],
            [tf.float64],
            name=name,
            grad=grad_sparsemax)  # <-- here's the call to the gradient
    return sparsemax_forward[0]


def sparsemax_loss(Z, sparsemax, q, name=None):
    with tf.compat.v1.keras.backend.name_scope(name, "SparseMaxLossGrad", [Z, sparsemax, q]) as name:
        # py_func takes a list of tensors and a
        # function that takes np arrays as inputs
        # and returns np arrays as outputs
        sparsemax_forward_loss = py_func(
            smlforward_loss,
            [Z, sparsemax, q],
            [tf.float64],
            name=name,
            grad=grad_sparsemax_loss)  # <-- here's the call to the gradient

    return sparsemax_forward_loss[0]




#logits = tf.placeholder(tf.float64, name='z')
r = sparsemax(tf.zeros( (1,10) ) )
print(r)
#with tf.Session() as sess:
#    print(r.eval({logits: np.zeros((1, 10))}) ) 
