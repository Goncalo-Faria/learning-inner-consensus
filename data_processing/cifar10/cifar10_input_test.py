
"""Tests for cifar10_input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from . import cifar10_input


class CIFAR10InputTest(tf.test.TestCase):

  def _record(self, label, colors):
    image_size = 32 * 32
    record = bytes(
        bytearray([label] + [colors[0]] * image_size +
                  [colors[1]] * image_size + [colors[2]] * image_size))
    expected = [[colors] * 32] * 32
    return record, expected

  def testRead(self):
    """Tests if the records are read in the expected order and value."""
    labels = [0, 1, 9]
    colors = [[0, 0, 0], [255, 255, 255], [1, 100, 253]]
    records = []
    expecteds = []
    for i in range(3):
      record, expected = self._record(labels[i], colors[i])
      records.append(record)
      expecteds.append(expected)
    filename = os.path.join(self.get_temp_dir(), "cifar_test")
    open(filename, "wb").write(b"".join(records))

    with self.test_session() as sess:
      q = tf.queue.FIFOQueue(100, [tf.string], shapes=())
      q.enqueue([filename]).run()
      q.close().run()
      image_tensor, label_tensor = cifar10_input._read_input(q)

      for i in range(3):
        image, label = sess.run([image_tensor, label_tensor])
        self.assertEqual(labels[i], label)
        self.assertAllEqual(expecteds[i], image)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(image_tensor)

  def testBatchedOuput(self):
    """Tests if the final output of batching works properly."""
    record, _ = self._record(5, [255, 0, 128])
    batch_size = 10
    expected_labels = [5 for _ in range(batch_size)]
    data_dir = self.get_temp_dir()
    filename = os.path.join(data_dir, "test_batch.bin")
    open(filename, "wb").write(b"".join([record]))
    features = cifar10_input.inputs("test", data_dir, batch_size)

    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.compat.v1.train.start_queue_runners(coord=coord)
      labels = sess.run(features["recons_label"])
      self.assertAllEqual(expected_labels, labels)
      coord.request_stop()
      for thread in threads:
        thread.join()


if __name__ == "__main__":
  tf.test.main()
