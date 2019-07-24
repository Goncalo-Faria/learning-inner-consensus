from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..core.kernel import Kernel
from ..core import variables


class KernelMix(Kernel):
    def __init__(
            self,
            kernel_list,
            normalization = tf.nn.softmax,
            name=""):
        self._kernel_list = kernel_list
        self._normalization = normalization

        for k in kernel_list:
            assert isinstance(k, Kernel), \
                " every element must be an instance of Kernel class. "

        super(KernelMix, self).__init__(
            'kernelmix' + str(len(kernel_list)) + name,
            False)

    def apply(self, a, b):
        ## a,b :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim

        b = variables.weight_variable(
            [len(self._kernel_list)],
            name="mixing_coeficients",
            initializer=tf.compat.v1.initializers.ones()
        )

        c = self._normalization(b)

        s = tf.zeros(a.shape.as_list()[:-2]+ [1,1],dtype=tf.float32)

        for i in range(len(self._kernel_list)):
            with tf.compat.v1.variable_scope('component' + str(i), reuse=tf.compat.v1.AUTO_REUSE):
                s = s + c[i] * self._kernel_list[i].take(a,b)

        return s


class MonoKernelMix(KernelMix):
    def __init__(
            self,
            kernel,
            degree,
            normalization=tf.nn.softmax,
            name=""):

        assert isinstance(kernel, Kernel), \
            " kernel must be an instance of Kernel class. "

        super(MonoKernelMix, self).__init__(
            kernel_list=[kernel] * degree,
            normalization=normalization,
            name="monokernel" + name)

