import tensorflow as tf
import numpy as np


class TeacherCifar:
    def __init__(self, para_npy_path=None):
        self._para = np.load(para_npy_path)
        self._para = self._para.item()
        print('Teacher parameter loaded...')

    def _conv(self, _input, W, b, strides=1, padding='SAME'):
        output = tf.nn.conv2d(_input, W, strides=[1, strides, strides, 1], padding=padding)
        output = tf.nn.bias_add(output, b)
        return output

    def _lrelu(self, _input):
        return tf.maximum(_input, 0.1 * _input)
    
    def build(self, input):
        weights = {
            'conv1a': tf.Variable(self._para['conv1a_w:0'], trainable=False, name='conv1a_w'),
            'conv1b': tf.Variable(self._para['conv1b_w:0'], trainable=False, name='conv1b_w'),
            'conv1c': tf.Variable(self._para['conv1c_w:0'], trainable=False, name='conv1c_w'),
            'conv2a': tf.Variable(self._para['conv2a_w:0'], trainable=False, name='conv2a_w'),
            'conv2b': tf.Variable(self._para['conv2b_w:0'], trainable=False, name='conv2b_w'),
            'conv2c': tf.Variable(self._para['conv2c_w:0'], trainable=False, name='conv2c_w'),
            'conv3a': tf.Variable(self._para['conv3a_w:0'], trainable=False, name='conv3a_w'),
            'conv3b': tf.Variable(self._para['conv3b_w:0'], trainable=False, name='conv3b_w'),
            'conv3c': tf.Variable(self._para['conv3c_w:0'], trainable=False, name='conv3c_w'),
            'fc': tf.Variable(self._para['fc_w:0'], trainable=False, name='fc_w')
        }

        biases = {
            'conv1a': tf.Variable(self._para['conv1a_b:0'], trainable=False, name='conv1a_b'),
            'conv1b': tf.Variable(self._para['conv1b_b:0'], trainable=False, name='conv1b_b'),
            'conv1c': tf.Variable(self._para['conv1c_b:0'], trainable=False, name='conv1c_b'),
            'conv2a': tf.Variable(self._para['conv2a_b:0'], trainable=False, name='conv2a_b'),
            'conv2b': tf.Variable(self._para['conv2b_b:0'], trainable=False, name='conv2b_b'),
            'conv2c': tf.Variable(self._para['conv2c_b:0'], trainable=False, name='conv2c_b'),
            'conv3a': tf.Variable(self._para['conv3a_b:0'], trainable=False, name='conv3a_b'),
            'conv3b': tf.Variable(self._para['conv3b_b:0'], trainable=False, name='conv3b_b'),
            'conv3c': tf.Variable(self._para['conv3c_b:0'], trainable=False, name='conv3c_b'),
            'fc': tf.Variable(self._para['fc_b:0'], trainable=False, name='fc_b')
        }

        conv1a = self._conv(input, weights['conv1a'], biases['conv1a'])
        conv1a_relu = self._lrelu(conv1a)
        conv1b = self._conv(conv1a_relu, weights['conv1b'], biases['conv1b'])
        conv1b_relu = self._lrelu(conv1b)
        conv1c = self._conv(conv1b_relu, weights['conv1c'], biases['conv1c'])
        conv1c_relu = self._lrelu(conv1c)

        self.maxpool1 = tf.nn.max_pool(conv1c_relu, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

        conv2a = self._conv(self.maxpool1, weights['conv2a'], biases['conv2a'])
        conv2a_relu = self._lrelu(conv2a)
        conv2b = self._conv(conv2a_relu, weights['conv2b'], biases['conv2b'])
        conv2b_relu = self._lrelu(conv2b)
        conv2c = self._conv(conv2b_relu, weights['conv2c'], biases['conv2c'])
        conv2c_relu = self._lrelu(conv2c)

        maxpool2 = tf.nn.max_pool(conv2c_relu, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

        conv3a = self._conv(maxpool2, weights['conv3a'], biases['conv3a'], padding='SAME')
        conv3a_relu = self._lrelu(conv3a)
        conv3b = self._conv(conv3a_relu, weights['conv3b'], biases['conv3b'], padding='SAME')
        conv3b_relu = self._lrelu(conv3b)
        conv3c = self._conv(conv3b_relu, weights['conv3c'], biases['conv3c'], padding='SAME')
        conv3c_relu = self._lrelu(conv3c)

        h = tf.reduce_mean(conv3c_relu, reduction_indices=[1, 2])
        self.fc = tf.nn.bias_add(tf.matmul(h, weights['fc']), biases['fc'])

    def get_hint(self):
        return self.maxpool1

    def get_logits(self):
        return self.fc
