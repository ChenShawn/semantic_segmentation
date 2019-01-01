import tensorflow as tf
import matplotlib.pyplot as plt


def _build_iou(label, pred):
    # must input 2 tensors with shape (-1, 384, 384, 1)
    ious = list()
    for i in range(151):
        sub = tf.cast(tf.square(label - i), dtype=tf.float32)
        correct = 1.0 - tf.sign(sub)
        sub_ = tf.cast(tf.square(pred - i), dtype=tf.float32)
        correct_ = 1.0 - tf.sign(sub_)
        ious.append(tf.reduce_sum(tf.minimum(correct, correct_)) / tf.reduce_sum(tf.maximum(correct, correct_)))
    return ious