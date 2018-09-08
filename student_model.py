import tensorflow as tf
import numpy as np


class StudentCifar:
    def _conv(self, _input, W, b, strides=1, padding='SAME'):
        output = tf.nn.conv2d(_input, W, strides=[1, strides, strides, 1], padding=padding)
        output = tf.nn.bias_add(output, b)
        return output

    def _lrelu(self, _input):
        return tf.maximum(_input, 0.1 * _input)
    
    def build(self, input):
        weights = {
            'conv1a': tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32, stddev=1e-1), name='s_conv1a_w'),
            'conv1b': tf.Variable(tf.truncated_normal([3, 3, 32, 32], dtype=tf.float32, stddev=1e-1),
                                  name='s_conv1b_w'),
            'conv1c': tf.Variable(tf.truncated_normal([3, 3, 32, 32], dtype=tf.float32, stddev=1e-1),
                                  name='s_conv1c_w'),
            'conv2a': tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=1e-1),
                                  name='s_conv2a_w'),
            'conv2b': tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                                  name='s_conv2b_w'),
            'conv2c': tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                                  name='s_conv2c_w'),
            'conv3a': tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                                  name='s_conv3a_w'),
            'conv3b': tf.Variable(tf.truncated_normal([1, 1, 64, 32], dtype=tf.float32, stddev=1e-1),
                                  name='s_conv3b_w'),
            'conv3c': tf.Variable(tf.truncated_normal([1, 1, 32, 32], dtype=tf.float32, stddev=1e-1),
                                  name='s_conv3c_w'),
            'fc': tf.Variable(tf.truncated_normal([32, 10], dtype=tf.float32, stddev=1e-1), name='s_fc_w')
        }

        biases = {
            'conv1a': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), name='s_conv1a_b'),
            'conv1b': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), name='s_conv1b_b'),
            'conv1c': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), name='s_conv1c_b'),
            'conv2a': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), name='s_conv2a_b'),
            'conv2b': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), name='s_conv2b_b'),
            'conv2c': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), name='s_conv2c_b'),
            'conv3a': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), name='s_conv3a_b'),
            'conv3b': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), name='s_conv3b_b'),
            'conv3c': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), name='s_conv3c_b'),
            'fc': tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), name='s_fc_b')
        }

        conv1a = self._conv(input, weights['conv1a'], biases['conv1a'])
        conv1a_relu = self._lrelu(conv1a)
        conv1b = self._conv(conv1a_relu, weights['conv1b'], biases['conv1b'])
        conv1b_relu = self._lrelu(conv1b)
        conv1c = self._conv(conv1b_relu, weights['conv1c'], biases['conv1c'])
        conv1c_relu = self._lrelu(conv1c)

        maxpool1 = tf.nn.max_pool(conv1c_relu, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

        conv2a = self._conv(maxpool1, weights['conv2a'], biases['conv2a'])
        conv2a_relu = self._lrelu(conv2a)
        conv2b = self._conv(conv2a_relu, weights['conv2b'], biases['conv2b'])
        conv2b_relu = self._lrelu(conv2b)
        conv2c = self._conv(conv2b_relu, weights['conv2c'], biases['conv2c'])
        self.conv2c_relu = self._lrelu(conv2c)

        maxpool2 = tf.nn.max_pool(self.conv2c_relu, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

        conv3a = self._conv(maxpool2, weights['conv3a'], biases['conv3a'], padding='SAME')
        conv3a_relu = self._lrelu(conv3a)
        conv3b = self._conv(conv3a_relu, weights['conv3b'], biases['conv3b'], padding='SAME')
        conv3b_relu = self._lrelu(conv3b)
        conv3c = self._conv(conv3b_relu, weights['conv3c'], biases['conv3c'], padding='SAME')
        conv3c_relu = self._lrelu(conv3c)

        h = tf.reduce_mean(conv3c_relu, reduction_indices=[1, 2])
        self.fc = tf.nn.bias_add(tf.matmul(h, weights['fc']), biases['fc'])

    def get_guide(self):
        return self.conv2c_relu

    def get_logits(self):
        return self.fc
