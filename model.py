import tensorflow as tf
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from utils import dilated_block, pooling, deconv2d, conv2d_relu, conv2d

class Network(object):
    def __init__(self, input_shape, data_dir, label_dir, batch_size=4, learning_rate=8e-4,
                 epoch=10, model_dir='./checkpoint/', log_dir='.\\logs', pre_train=True):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.epoch = epoch

        self.data = self._read_data(data_dir, label_dir)
        self.batch_xs, batch_ys = self.data.make_one_shot_iterator().get_next()
        self.batch_ys = tf.cast(batch_ys, tf.int32)

        self._build_model()
        self.loss = self._loss_function()
        self.optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=self.fcn_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        self.summary = tf.summary.scalar('loss', self.loss)
        map_summary = tf.summary.scalar('mAP', self.mAP)
        img_sum = tf.summary.image('image', self.batch_xs)
        label_sum = tf.summary.image('label', tf.cast(self.batch_ys, dtype=tf.uint8))
        pred_sum = tf.summary.image('prediction', tf.cast(self.prediction, dtype=tf.uint8))
        self.img_summary = tf.summary.merge([img_sum, label_sum, pred_sum, map_summary])
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        if pre_train and len(os.listdir(self.model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def _read_data(self, data_dir, label_dir):
        def _parse_function(filename, labelname):
            x_img_str = tf.read_file(filename)
            x_img_decoded = tf.image.convert_image_dtype(tf.image.decode_png(x_img_str), tf.float32)
            x_img_resized = tf.image.resize_images(x_img_decoded, size=[384, 384],
                                                   method=tf.image.ResizeMethod.BILINEAR)
            y_img_str = tf.read_file(labelname)
            y_img_decoded = tf.image.decode_png(y_img_str)
            y_img_resized = tf.image.resize_images(y_img_decoded, size=[384, 384],
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return x_img_resized, y_img_resized

        x_files = os.listdir(data_dir)
        y_files = os.listdir(label_dir)
        x_app = [os.path.join(data_dir, name) for name in x_files]
        y_app = [os.path.join(label_dir, name) for name in y_files]
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x_app), tf.constant(y_app)))
        data = data.map(_parse_function)
        return data.shuffle(buffer_size=1000).batch(self.batch_size).repeat(self.epoch)

    def _build_model(self):
        self.test_image = tf.placeholder(tf.float32, shape=[None, 384, 384, 3], name='test')
        self.fcnn, self.fcnn_logits = self._build_fcn(self.batch_xs, reuse=False, is_training=True)
        self.fcnn_, _ = self._build_fcn(self.test_image, reuse=True, is_training=False)
        self.prediction = tf.expand_dims(tf.argmax(self.fcnn, axis=3, output_type=tf.int32), axis=3)
        self.prediction_ = tf.expand_dims(tf.argmax(self.fcnn_, axis=3, output_type=tf.int32), axis=3)
        self.mAP = self._build_map(self.batch_ys, self.prediction)
        self.iou_idx = tf.placeholder(tf.int32, name='iou_idx')
        self.IoU = self._build_iou(self.batch_ys, self.prediction)

        t_vars = tf.trainable_variables()
        self.fcn_vars = [var for var in t_vars if 'FCN' in var.name]

    def _build_fcn(self, input_op, reuse=False, is_training=True):
        row, col = self.input_shape[0], self.input_shape[1]
        row_p1, col_p1 = int(row / 2), int(col / 2)
        row_p2, col_p2 = int(row_p1 /2), int(col_p1 / 2)

        with tf.variable_scope('FCNN', reuse=reuse):
            conv1_1 = conv2d_relu(input_op, n_out=64, name='conv1_1', is_training=is_training)
            conv1_2 = conv2d_relu(conv1_1, n_out=64, name='conv1_2', is_training=is_training)
            pool_1 = pooling(conv1_2, name='pool_1')

            conv2_1 = conv2d_relu(pool_1, n_out=128, name='conv2_1', is_training=is_training)
            conv2_2 = conv2d_relu(conv2_1, n_out=128, name='conv2_2', is_training=is_training)
            pool_2 = pooling(conv2_2, name='pool_2')

            conv3_1 = dilated_block(pool_2, n_out=256, is_training=is_training, name='conv3_1')
            conv3_2 = dilated_block(conv3_1, n_out=256, is_training=is_training, name='conv3_2')
            conv3_3 = dilated_block(conv3_2, n_out=256, is_training=is_training, name='conv3_3')
            pool_3 = pooling(conv3_3, name='pool_3')

            conv4_1 = dilated_block(pool_3, n_out=512, is_training=is_training, name='conv4_1')
            conv4_2 = dilated_block(conv4_1, n_out=512, is_training=is_training, name='conv4_2')
            conv4_3 = dilated_block(conv4_2, n_out=512, is_training=is_training, name='conv4_3')
            deconv_1 = deconv2d(conv4_3, output_shape=[self.batch_size, row_p2, col_p2, 256], name='deconv_1')

            concat_1 = tf.concat([conv3_3, deconv_1], axis=3, name='concat_1')
            conv5_1 = dilated_block(concat_1, n_out=256, is_training=is_training, name='conv5_1')
            conv5_2 = dilated_block(conv5_1, n_out=256, is_training=is_training, name='conv5_2')
            conv5_3 = dilated_block(conv5_2, n_out=256, is_training=is_training, name='conv5_3')
            deconv_2 = deconv2d(conv5_3, output_shape=[self.batch_size, row_p1, col_p1, 128], name='deconv_2')

            concat_2 = tf.concat([conv2_2, deconv_2], axis=3, name='concat_2')
            conv6_1 = conv2d_relu(concat_2, n_out=151, name='conv6_1', is_training=is_training)
            conv6_2 = conv2d_relu(conv6_1, n_out=151, name='conv6_2', is_training=is_training)
            deconv_3 = deconv2d(conv6_2, output_shape=[self.batch_size, row, col, 64], name='deconv_3')

            concat_3 = tf.concat([conv1_2, deconv_3], axis=3, name='concat_3')
            conv7_1 = conv2d_relu(concat_3, n_out=151, name='conv7_1', is_training=is_training)
            conv7_2 = conv2d(conv7_1, n_out=151, name='conv7_2')
            return tf.nn.softmax(conv7_2, axis=3), conv7_2

    def _loss_function(self):
        # logits = tf.reshape(self.fcnn_logits, shape=[384**2, 151])
        # labels = tf.reshape(self.batch_ys, shape=[384**2])
        labels = tf.squeeze(self.batch_ys, axis=[3])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fcnn_logits)
        return tf.reduce_sum(loss, name='loss')

    @staticmethod
    def _build_map(label, pred):
        # should inp0ut 2 tensors with the same shape
        correct = tf.cast(tf.equal(label, pred), dtype=tf.float32)
        return tf.reduce_mean(correct)

    def _build_iou(self, label, pred):
        # must input 2 tensors with shape (-1, 384, 384, 1)
        sub = tf.cast(tf.square(label - self.iou_idx), dtype=tf.float32)
        correct = 1.0 - tf.sign(sub)
        sub_ = tf.cast(tf.square(pred - self.iou_idx), dtype=tf.float32)
        correct_ = 1.0 - tf.sign(sub_)
        return tf.reduce_sum(tf.minimum(correct, correct_)) / tf.reduce_sum(tf.maximum(correct, correct_))

    def train(self, loop=None, log_iter=100):
        for iter in range(loop):
            try:
                _, summary_str = self.sess.run([self.optim, self.summary])
            except tf.errors.OutOfRangeError:
                print('Epoch has Ended!! Finish training!!')
                self.save()
                return None
            self.writer.add_summary(summary_str, self.counter)
            if iter % log_iter == 5:
                try:
                    loss, mAP, summary_str = self.sess.run([self.loss, self.mAP, self.img_summary])
                except tf.errors.OutOfRangeError:
                    print('Epoch has Ended!! Finish training!!')
                    self.save()
                    return None
                logging = ' --Iteration %d --FCN loss %g --Evaluated mAP %g' % (iter, loss, mAP)
                print(str(datetime.now()) + logging)
                self.writer.add_summary(summary_str, self.counter)
            self.counter += 1
        print('Training finished, ready to save...')
        self.save()

    def save(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        elif len(os.listdir(self.model_dir)) != 0:
            fs = os.listdir(self.model_dir)
            for f in fs:
                os.remove(self.model_dir + f)
        save_path = self.saver.save(self.sess, self.model_dir + 'CRFasRNN.model', global_step=self.counter)
        print('MODEL RESTORED IN: ' + save_path)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, self.model_dir + ckpt_name)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


if __name__ == '__main__':
    net = Network(input_shape=[384, 384], data_dir='E:\\datasets\\ADEChallengeData2016\\images\\training\\',
                  label_dir='E:\\datasets\\ADEChallengeData2016\\annotations\\training\\', batch_size=1,
                  learning_rate=8e-4, epoch=1, pre_train=True)

    net.train(50000)

    # while True:
    #     try:
    #         xs, ys = net.sess.run([net.batch_xs, net.batch_ys])
    #     except tf.errors.OutOfRangeError:
    #         break
    #     print(xs.shape, ys.shape)
    #
    #     plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(xs.reshape((384, 384, 3)))
    #     plt.subplot(122)
    #     plt.imshow(ys.reshape((384, 384)), cmap='gray')
    #     plt.show()
    #     plt.close()