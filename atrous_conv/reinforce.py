from datetime import datetime
import matplotlib.pyplot as plt
from functools import reduce
import os

from utils import *
from cnn_read import Reader, staticReader

class reinforcementNetwork(object):
    def __init__(self, input_shape, input_dim, batch_size=4, learning_rate=0.0002, attempt=20,
                 model_dir='./checkpoint/', log_dir='.\\logs', pre_train=True):
        # Copy parameters
        # input_shape should be 3 dimensional list
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.attempt = attempt
        self.model_dir = model_dir

        self._build_model()
        self.fcn_loss, self.sub_loss, self.rl_loss = self._loss_function()
        self.fcn_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.fcn_loss, var_list=self.fcn_vars)
        self.sub_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.sub_loss, var_list=self.fcn_vars)
        self.rl_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.rl_loss, var_list=self.rl_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Initialize summary
        fcn_summary = tf.summary.scalar('fcn_loss_summary', self.fcn_loss)
        rl_summary = tf.summary.scalar('rl_loss_summary', self.rl_loss)
        img_summary = tf.summary.image('origin images', self.input_image)
        sub_img_summary = tf.summary.image('augmented images', self.sub_img)
        sub_label_summary = tf.summary.image('augmented labels', self.sub_label)
        sub_pred_summary = tf.summary.image('augmented prediction', self.sub_fcnn)
        pred_summary = tf.summary.image('predictions', self.fcnn)
        self.summary = tf.summary.merge([fcn_summary], name='fcn_summary')
        self.rl_summary  = tf.summary.merge([rl_summary], name='rl_summary')
        self.img_summary = tf.summary.merge([img_summary, sub_img_summary, sub_label_summary,
                                             pred_summary, sub_pred_summary])
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
        label_image = tf.reshape(self.label_vector, shape=[self.batch_size] + self.input_shape)
        # reinforcement learning
        self.offset_height = tf.placeholder(tf.int32, name='offset_height')
        self.offset_width = tf.placeholder(tf.int32, name='offset_width')

        pair_sample = tf.concat([self.input_image, label_image], axis=3)
        image_cropped = tf.image.crop_to_bounding_box(pair_sample, self.offset_height, self.offset_width,
                                                      target_height=128, target_width=128)
        image_resized = tf.image.resize_images(image_cropped, size=[512, 512], method=tf.image.ResizeMethod.BILINEAR)
        sub_img, sub_label = tf.split(image_resized, num_or_size_splits=[1, 1], axis=2)
        self.sub_img = tf.expand_dims(sub_img, dim=0)
        self.sub_label = tf.expand_dims(sub_label, dim=0)

        self.fcnn, self.fcnn_logits, self.hidden = self._build_fcn(self.input_image, is_training=True)
        self.sub_fcnn, self.sub_logits, _ = self._build_fcn(self.sub_img, is_training=False)
        # Reinforcement Learning
        self.rewards = tf.placeholder(tf.float32, name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=[self.batch_size, 256], name='actions')
        self.attention = self._build_attention(self.hidden)
        _, self.top_anchors = tf.nn.top_k(self.attention, k=self.attempt, name='top_k_anchors')

        trainable_variables = tf.trainable_variables()
        self.fcn_vars = [var for var in trainable_variables if 'FCNN' in var.name]
        self.rl_vars = [var for var in trainable_variables if 'ATTENTION' in var.name]

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

    def _build_crf(self, input_op, sigma=100.0):
        # The implementation performs a simplification of CRFasRNN in binary classification
        with tf.variable_scope('CRFasRNN'):
            # unaries = tf.sigmoid(input_op, name='unaries')
            # laplace = const_conv2d(self.input_image, ink=laplacian_5x5, n_out=1, name='laplacian')
            # low_pass = const_conv2d(tf.exp(-tf.square(laplace)), ink=gaussian_5x5, n_out=1, name='low_pass')
            # messages = low_pass * unaries
            pass

    def _build_attention(self, input_op):
        with tf.variable_scope('ATTENTION'):
            conv3_3 = inception_block(input_op, n_out=256, is_training=True, name='conv3_3')
            conv3_4 = inception_block(conv3_3, n_out=256, is_training=True, name='conv3_4')
            pool3 = pooling(conv3_4, name='pool_3')

            conv4_1 = conv2d_relu(pool3, n_out=128, is_training=True, name='conv4_1', dw=2, dh=2)
            conv4_2 = conv2d_relu(conv4_1, n_out=32, is_training=True, name='conv4_2', dw=2, dh=2)
            conv4_3 = conv2d(conv4_2, n_out=1, name='conv4_3', reuse=False)

            shape = conv4_3.get_shape()
            new_shape = [-1, shape[1].value * shape[2].value * shape[3].value]
            reshaped = tf.reshape(conv4_3, shape=new_shape, name='reshaped')
            return reshaped

    def _loss_function(self, l2_weight=0.01):
        logit_vector = tf.reshape(self.fcnn_logits, shape=(self.batch_size, self.input_dim))
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_vector, labels=self.label_vector)
        sub_label_vector = tf.reshape(self.sub_label, shape=(self.batch_size, self.input_dim))
        sub_logits_vector = tf.reshape(self.sub_logits, shape=(self.batch_size, self.input_dim))
        sub_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=sub_logits_vector, labels=sub_label_vector)

        policy_gradient = tf.nn.softmax_cross_entropy_with_logits(logits=self.attention, labels=self.actions)
        rl_loss = tf.reduce_sum(policy_gradient, name='rl_loss') * self.rewards

        return tf.reduce_sum(cross_entropy), tf.reduce_sum(sub_cross_entropy), rl_loss

    def train(self, reader, loop=20000, print_iter=50):
        for i in range(loop):
            batch_xs, batch_ys = reader.next_batch(self.batch_size)
            self.sess.run(self.fcn_optim, feed_dict={self.input_image_vector: batch_xs,
                                                     self.label_vector: batch_ys})
            self.sess.run(self.sub_optim, feed_dict={self.input_image_vector: batch_xs,
                                                     self.label_vector: batch_ys})
            # Log on screen
            if i % print_iter == 5:
                loss = self.sess.run(self.fcn_loss, feed_dict={self.input_image_vector: batch_xs,
                                                               self.label_vector: batch_ys})
                logging = ' --Iteration %d --FCN loss %g' % (i, loss)
                _, summary_str = self.sess.run([self.sub_optim, self.img_summary],
                                               feed_dict={self.input_image_vector: batch_xs,
                                                          self.label_vector: batch_ys})
                self.writer.add_summary(summary_str, self.counter)
                print(str(datetime.now()) + logging)
            # Log on tensorboard
            _, summary_str = self.sess.run([self.sub_optim, self.summary],
                                           feed_dict={self.input_image_vector: batch_xs,
                                                      self.label_vector: batch_ys})
            self.writer.add_summary(summary_str, self.counter)
            self.counter += 1
        print('Training finished, ready to save...')
        self.save()

    def reinforce(self, reader, loop=5000, print_iter=50):
        summary_str = None
        for i in range(loop):
            batch_xs, batch_ys = reader.next_batch(self.batch_size)
            top_ks, fcnn = self.sess.run([self.top_anchors, self.fcnn],
                                         feed_dict={self.input_image_vector: batch_xs})
            coordinates = self._convert_coordinates(top_ks)
            for idx, cd in enumerate(coordinates):
                sub_fcn, sub_label = self.sess.run([self.sub_fcnn, self.sub_label],
                                                   feed_dict={self.offset_height: cd[0],
                                                              self.offset_width: cd[1]})
                rewards = get_accuracy(sub_fcn, sub_label) - get_accuracy(fcnn, batch_ys) - 0.5
                actions = np.zeros((self.batch_size, 16**2), dtype=np.float32)
                actions[0, top_ks[idx]] = 1.0
                _, summary_str = self.sess.run([self.rl_optim, self.rl_summary],
                                               feed_dict={self.offset_height: cd[0],
                                                          self.offset_width: cd[1],
                                                          self.actions: actions,
                                                          self.rewards: rewards,
                                                          self.input_image_vector: batch_xs,
                                                          self.label_vector: batch_ys})
            self.writer.add_summary(summary_str, global_step=i)
            if i % print_iter == 2:
                _, summary_str = self.sess.run([self.sub_optim, self.img_summary],
                                               feed_dict={self.input_image_vector: batch_xs,
                                                          self.label_vector: batch_ys})
                self.writer.add_summary(summary_str, self.counter)

        if 'y' in str(input('save model??(y/n)')):
            self.save()

    def _convert_coordinates(self, top_ks):
        output = list()
        for idx in top_ks:
            rows = int(idx / 16) if int(idx / 16) < 12 else 8
            cols = int(idx % 16) if int(idx % 16) < 12 else 8
            output.append([rows, cols])
        return output

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
    crf_rnn = reinforcementNetwork(input_shape=[512,512,1], batch_size=1, attempt=20,
                                   input_dim=262144, learning_rate=8e-5, pre_train=True)
    show_all_variables()
    crf_rnn.reinforce(reader, loop=100)

    iter = 300
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