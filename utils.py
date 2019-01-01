import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
import math


def batch_norm(input_op, is_training, epsilon=1e-5, momentum=0.99, name='batch_norm'):
    return tf.contrib.layers.batch_norm(input_op, decay=momentum, updates_collections=None,
                                        epsilon=epsilon, scale=True, is_training=is_training, scope=name)

def show_all_variables():
    all_variables = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(all_variables, print_info=True)

def leaky_relu(input_op, leak=0.2, name='linear'):
    return tf.maximum(input_op, leak*input_op, name=name)

def conv2d(input_op, n_out, name, kh=3, kw=3, dh=1, dw=1):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(conv, biases)

def conv2d_relu(input_op, n_out, name, is_training=True, kh=3, kw=3, dh=1, dw=1):
    n_in = input_op.get_shape()[-1].value
    if n_in is None:
        n_in = 3
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
        z_out = tf.nn.bias_add(conv, biases)
        return tf.nn.relu(batch_norm(z_out, is_training=is_training, name='conv_bn'))

def atrous_conv(input_op, n_out, rate, name, kh=3, kw=3, activate='relu'):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.atrous_conv2d(input_op, kernel, rate=rate, padding='SAME')
        biases = tf.get_variable('biases', (n_out), initializer=tf.constant_initializer(0.0))
        z_out = tf.nn.bias_add(conv, biases)
        if activate == 'relu':
            return tf.nn.relu(z_out, name='relu')
        elif activate == 'lrelu':
            return leaky_relu(z_out)
        else:
            return z_out

def pooling(input_op, name, kh=2, kw=2, dh=2, dw=2, pooling_type='max'):
    if 'max' in pooling_type:
        return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)
    else:
        return tf.nn.avg_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

def deconv2d(input_op, output_shape, kh=3, kw=3, dh=2, dw=2, name='deconv', bias_init=0.0):
    n_in = input_op.get_shape()[-1].value
    n_out = output_shape[-1]
    # filter : [height, width, output_channels, in_channels]
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernels',
                                 shape=(kh, kw, n_out, n_in),
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        deconv = tf.nn.conv2d_transpose(input_op, kernel,
                                        output_shape=output_shape,
                                        strides=(1, dh, dw, 1))
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        return tf.nn.relu(tf.nn.bias_add(deconv, biases), name='deconv_activate')

def fully_connect(input_op, n_out, name='fully_connected', bias_init=0.0, activate='lrelu', with_kernels=False):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='matrix',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        biases = tf.get_variable(name='bias', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        return tf.matmul(input_op, kernel) + biases

def dilated_block(input_op, n_out, rate=2, is_training=True, name='dilated_block'):
    # inception v3 with dilated convolution
    with tf.variable_scope(name):
        conv3x3 = conv2d(input_op, n_out=n_out / 2, name='conv1x1')
        atrous3x3 = atrous_conv(input_op, n_out=n_out / 2, rate=rate, name='atrous_3x3', activate='None')
        concat = tf.concat([conv3x3, atrous3x3], axis=3, name='concatenated')
        return tf.nn.relu(batch_norm(concat, is_training=is_training, name='dilated_bn'))