from __future__ import print_function
from data import DataLoader
import matplotlib.pyplot as plt
import cv2
import time
from bgsnet import BGSNet
import tensorflow as tf

train_dir = '/home/yzhq/data/CDNet2014/trainset/'
test_dir = '/home/yzhq/data/CDNet2014/testset/'
l = 320

DL = DataLoader(data_dir=train_dir, batch_size=40, image_shape=(l, l))

batch_x, batch_y = DL.next_batch()

sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, l, l, 3], name='input')
net = BGSNet('vgg16partial.npy', 'train/bgsnet30.npy')
net.build(x, l, 40)
sess.run(tf.global_variables_initializer())
batch_z = sess.run(net.output, feed_dict={x: batch_x})
sess.close()

for i in range(40):
    Ix = batch_x[i, :, :, :]
    Iy = batch_y[i, :, :, :]
    Iy = 255 * Iy
    Iy = Iy.astype('uint8')
    Iz = 255 * (batch_z[i, :, :, :])
    Iz = Iz.astype('uint8')

    Ix = cv2.cvtColor(Ix, cv2.COLOR_BGR2RGB)
    Iy = cv2.cvtColor(Iy, cv2.COLOR_GRAY2RGB)
    Iz = cv2.cvtColor(Iz, cv2.COLOR_GRAY2RGB)

    plt.subplot(131)
    plt.imshow(Ix)
    plt.subplot(132)
    plt.imshow(Iy)
    plt.subplot(133)
    plt.imshow(Iz)
    plt.show()

