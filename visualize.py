import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def show_data_pairs():
    from model import Network
    net = Network(input_shape=[384, 384], data_dir='E:\\datasets\\ADEChallengeData2016\\images\\training\\',
                  label_dir='E:\\datasets\\ADEChallengeData2016\\annotations\\training\\', batch_size=1,
                  learning_rate=8e-4, epoch=1, pre_train=True)
    print('--Initialized Network')
    while True:
        try:
            xs, ys = net.sess.run([net.batch_xs, net.batch_ys])
        except tf.errors.OutOfRangeError:
            break

        plt.figure()
        plt.subplot(121)
        plt.imshow(xs.reshape((384, 384, 3)))
        plt.subplot(122)
        plt.imshow(ys.reshape((384, 384)), cmap='gray')
        plt.show()
        plt.close()

        ss = set()
        for i in range(384):
            for j in range(384):
                ss.add(ys[0, i, j, 0])
        print(ss)


if __name__ == '__main__':
    show_data_pairs()