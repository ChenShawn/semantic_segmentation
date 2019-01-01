from datetime import datetime
import matplotlib.pyplot as plt
from functools import reduce
import os

from utils import *
from cnn_read import Reader

class robustTraining(object):
    def __init__(self, input_shape, input_dim, batch_size=4, learning_rate=0.0002, gamma=10.0,
                 model_dir='./checkpoint/', log_dir='.\\logs', pre_train=True):
        # Copy parameters
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.model_dir = model_dir

        self._build_model()
        self.fcn_loss, self.fake_loss = self._loss_function(gamma=gamma)
        self.fcn_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.fcn_loss, var_list=self.fcn_vars)
        self.sub_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.fake_loss, var_list=self.fake_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Initialize summary
        self.fcn_summary = tf.summary.scalar('fcn_loss_summary', self.fcn_loss)
        self.fake_summary = tf.summary.scalar('fake_loss_summary', self.fcn_loss)
        self.adv_summary = tf.summary.scalar('adv_summary', self.fake_loss)
        img_summary = tf.summary.image('origin images', self.input_image)
        pred_summary = tf.summary.image('predictions', self.fcnn)
        adversarial_summary = tf.summary.image('adversarial examples', self.img_hat)
        adv_pred_summary = tf.summary.image('augmented prediction', self.sub_fcnn)
        self.img_summary = tf.summary.merge([img_summary, pred_summary, adversarial_summary, adv_pred_summary])
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        # load pre_trained model if checkpoint is not empty
        if pre_train and len(os.listdir(self.model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def _build_model(self):
        self.input_image_vector = tf.placeholder(tf.float32, shape=(None, self.input_dim))
        self.input_image = tf.reshape(self.input_image_vector, shape=[self.batch_size] + self.input_shape)
        self.label_vector = tf.placeholder(tf.float32, shape=[None, self.input_dim])

        self.img_hat = tf.get_variable('img_hat', dtype=tf.float32, shape=[self.batch_size] + self.input_shape)
        self.img_hat_assign = self.img_hat.assign(self.input_image)

        self.fcnn, self.fcnn_logits, self.hidden = self._build_fcn(self.input_image, is_training=True)
        self.sub_fcnn, self.sub_logits, _ = self._build_fcn(self.img_hat, is_training=False)

        trainable_variables = tf.trainable_variables()
        self.fcn_vars = [var for var in trainable_variables if 'FCNN' in var.name]
        self.fake_vars = [var for var in trainable_variables if 'FCNN' not in var.name]

    def _build_fcn(self, input_op, is_training):
        row, col = self.input_shape[0], self.input_shape[1]
        row_p1, col_p1 = int(row / 2), int(col / 2)
        with tf.variable_scope('FCNN', reuse=not is_training):
            conv1_1 = inception_block(input_op, n_out=32, is_training=is_training, name='inception1_1')
            conv1_2 = inception_block(conv1_1, n_out=32, is_training=is_training, name='conv1_2')
            conv1_3 = inception_block(conv1_2, n_out=32, is_training=is_training, name='conv1_3')
            pool_1 = pooling(conv1_3, name='pool_1')

            conv2_1 = inception_block(pool_1, n_out=128, is_training=is_training, name='conv2_1')
            conv2_2 = inception_block(conv2_1, n_out=128, is_training=is_training, name='conv2_2')
            conv2_3 = inception_block(conv2_2, n_out=128, is_training=is_training, name='conv2_3')
            pool_2 = pooling(conv2_3, name='pool_2')

            conv3_1 = inception_block(pool_2, n_out=512, is_training=is_training, name='conv3_1')
            conv3_2 = inception_block(conv3_1, n_out=512, is_training=is_training, name='conv3_2')
            deconv_1 = deconv2d(conv3_2, output_shape=[self.batch_size, row_p1, col_p1, 128])

            concat_1 = tf.concat([conv2_3, deconv_1], axis=3, name='concat_1')
            conv4_1 = conv2d_relu(concat_1, n_out=128, name='conv4_1', is_training=is_training)
            conv4_2 = conv2d_relu(conv4_1, n_out=128, name='conv4_2', is_training=is_training)
            deconv_2 = deconv2d(conv4_2, output_shape=[self.batch_size, row, col, 32], name='deconv_2')

            concat_2 = tf.concat([conv1_3, deconv_2], axis=3, name='concat_2')
            conv5_1 = conv2d_relu(concat_2, n_out=32, name='conv5_1', is_training=is_training)
            conv5_2 = conv2d(conv5_1, n_out=1, name='conv5_2')
            return tf.nn.sigmoid(conv5_2, name='sigmoid_fcn'), conv5_2, conv3_2

    def _build_penalty(self, input_op):
        # The implementation performs a simplification of CRFasRNN in binary classification
        with tf.variable_scope('Penalty'):
            # dx = const_conv2d(input_op, ink=sobel_3x3, n_out=1, name='dx')
            # dy = const_conv2d(input_op, ink=sobel_3x3.transpose(), n_out=1, name='dy')
            # grad = tf.sqrt(dx * dx + dy * dy, name='gradient')
            # return grad
            pass

    def _loss_function(self, gamma):
        logit_vector = tf.reshape(self.fcnn_logits, shape=(self.batch_size, self.input_dim))
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_vector, labels=self.label_vector)
        logits_hat = tf.reshape(self.sub_logits, shape=(self.batch_size, self.input_dim))
        cross_entropy_hat = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_hat, labels=self.label_vector)

        fcn_loss = tf.reduce_sum(cross_entropy, name='cross_entropy')
        loss_hat = tf.reduce_sum(-cross_entropy_hat) +\
                   gamma * tf.reduce_sum(tf.square(self.img_hat - self.input_image), name='loss_hat')
        return fcn_loss, loss_hat

    def train(self, reader, loop=20000, print_iter=100, fake_iter=20):
        for i in range(loop):
            batch_xs, batch_ys = reader.next_batch(self.batch_size)
            feed_dict = {self.input_image_vector: batch_xs, self.label_vector: batch_ys}

            self.sess.run(self.img_hat_assign, feed_dict=feed_dict)
            for k in range(fake_iter):
                _ = self.sess.run(self.sub_optim, feed_dict=feed_dict)
            fake_imgs = self.sess.run(self.img_hat, feed_dict=feed_dict)

            _, summary_str = self.sess.run([self.fcn_optim, self.fcn_summary], feed_dict=feed_dict)
            self.writer.add_summary(summary_str, self.counter)

            if i % print_iter == 5:
                _, summary_str = self.sess.run(self.img_summary, feed_dict=feed_dict)
                self.writer.add_summary(summary_str, self.counter)

            feed_dict[self.input_image_vector] = fake_imgs.reshape((self.batch_size, self.input_dim))
            _, summary_str = self.sess.run([self.fcn_optim, self.fake_summary], feed_dict=feed_dict)
            self.writer.add_summary(summary_str, self.counter)

            self.counter += 1
        print('Training finished, ready to save...')
        self.save()

    def predict(self, imgvec, as_list=False):
        pred = self.sess.run(self.fcnn, feed_dict={self.input_image_vector: imgvec})
        input_size = pred.shape[0]
        if as_list:
            return [pred[i, :, :, 0] for i in range(input_size)]
        else:
            return pred.reshape((input_size, self.input_dim))

    def predict_n_inference(self, imgvec, label_vec, as_list=True):
        pred = self.sess.run(self.sub_fcnn, feed_dict={self.input_image_vector: imgvec,
                                                       self.label_vector: label_vec})
        input_size = pred.shape[0]
        if as_list:
            return [pred[i, :, :, 0] for i in range(input_size)]
        else:
            return pred.reshape((input_size, self.input_dim))

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
    reader = Reader()
    crf_rnn = robustTraining(input_shape=[512,512,1], batch_size=1, gamma=10.0,
                             input_dim=262144, learning_rate=4e-4, pre_train=True)
    show_all_variables()
    crf_rnn.train(reader, loop=100)

    iter = 100
    acc_without_crf = 0.0
    acc_with_crf = 0.0
    for _ in range(iter):
        xs, ys = reader.next_batch(1)

        pred = crf_rnn.predict(xs, as_list=False)
        infe = crf_rnn.predict_n_inference(xs, ys, as_list=False)

        tmp_without_crf = get_accuracy(pred, ys)
        tmp_with_crf = get_accuracy(infe, ys)
        print('--Precision without CRF: %g --Precision with CRF: %g' % (tmp_without_crf, tmp_with_crf))
        acc_with_crf += tmp_with_crf
        acc_without_crf += tmp_without_crf

        plt.figure()
        plt.subplot(221)
        plt.imshow(pred.reshape((512, 512)), cmap='gray')
        plt.title('without CRF')
        plt.subplot(222)
        plt.imshow(infe.reshape((512, 512)), cmap='gray')
        plt.title('with CRF')
        plt.subplot(223)
        plt.imshow(xs.reshape((512, 512)), cmap='gray')
        plt.title('origin')
        plt.subplot(224)
        plt.imshow(ys.reshape((512, 512)), cmap='gray')
        plt.title('Ground Truth')
        plt.show()
        plt.close()

    print('\nTotal Precision:\n\t--Precision without CRF: %g\n\t--Precision with CRF: %g' %
          (acc_without_crf / float(iter), acc_with_crf / float(iter)))

    print('\nFinished!!!')