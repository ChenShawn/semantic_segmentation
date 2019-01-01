import matplotlib.pyplot as plt

from network import CRF_RNN
from utils import show_all_variables, get_accuracy
from cnn_read import Reader, staticReader

def compare_crf():
    reader = Reader()
    crf_rnn = CRF_RNN(input_shape=[512,512,1], batch_size=1, l2_weight=0.001,
                      input_dim=262144, learning_rate=6e-4, pre_train=True)
    show_all_variables()

    iter = 30
    acc_without_crf = 0.0
    acc_with_crf = 0.0
    for _ in range(iter):
        xs, ys = reader.next_batch(1)

        pred = crf_rnn.predict(xs, as_list=False)
        infe = crf_rnn.predict_n_inference(xs, as_list=False)

        tmp_without_crf = get_accuracy(pred, ys)
        tmp_with_crf = get_accuracy(infe, ys)
        print('--Precision without CRF: %g --Precision with CRF: %g' % (tmp_without_crf, tmp_with_crf))
        acc_with_crf += tmp_with_crf
        acc_without_crf += tmp_without_crf

        plt.figure()
        plt.subplot(121)
        plt.imshow(pred.reshape((512, 512)), cmap='gray')
        plt.subplot(122)
        plt.imshow(infe.reshape((512, 512)), cmap='gray')
        plt.show()
        plt.close()
        plt.show()

    print('\nTotal Precision:\n\t--Precision without CRF: %g\n\t--Precision with CRF: %g' %
          (acc_without_crf / float(iter), acc_with_crf / float(iter)))

    print('\nFinished!!!')


if __name__ == '__main__':
    import tensorflow as tf

    reader = tf.WholeFileReader()
    key, value = reader.read(tf.train.string_input_producer(['test.jpg']))
    img = tf.image.convert_image_dtype(tf.image.decode_jpeg(value), dtype=tf.float32)

    resized = tf.image.resize_images(img, size=[256, 256], method=tf.image.ResizeMethod.BILINEAR)
    cropped = tf.random_crop(img, size=[256, 256, 3])
    flipped = tf.image.random_flip_left_right(tf.image.random_flip_up_down(img))
    brightten = tf.image.random_contrast(img, lower=0.1, upper=0.3)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        out0 = sess.run(resized)
        out1 = sess.run(cropped)
        out2 = sess.run(flipped)
        out3 = sess.run(brightten)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(221)
    plt.imshow(out0)
    plt.subplot(222)
    plt.imshow(out1)
    plt.subplot(223)
    plt.imshow(out2)
    plt.subplot(224)
    plt.imshow(out3)
    plt.show()