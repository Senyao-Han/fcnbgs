import tensorflow as tf
import numpy as np
import cv2

VGG_MEAN = [103.939, 116.779, 123.68]

class BGSNet:

    def __init__(self, vgg_path, net_path=None):
        self.data_dict = None
        self.vgg_dict = None
        assert vgg_path is not None
        self.vgg_dict = np.load(vgg_path, encoding='latin1').item()
        if net_path is not None:
            self.data_dict = np.load(net_path, encoding='latin1').item()
        self.var_dict = {}

    def build(self, bgr, l, batch_size):
        # use input of l*l
        assert l % 32 == 0

        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=bgr)
        assert red.get_shape().as_list()[1:] == [l, l, 1]
        assert green.get_shape().as_list()[1:] == [l, l, 1]
        assert blue.get_shape().as_list()[1:] == [l, l, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        self.vgg_conv1_1 = self.vgg_conv_layer(bgr, 3, 64, 'vgg_conv1_1')
        self.vgg_conv1_2 = self.vgg_conv_layer(self.vgg_conv1_1, 64, 64, 'vgg_conv1_2')
        self.vgg_pool1 = self.pool2d(self.vgg_conv1_2, 'max', 'vgg_pool1')

        self.vgg_conv2_1 = self.vgg_conv_layer(self.vgg_pool1, 64, 128, 'vgg_conv2_1')
        self.vgg_conv2_2 = self.vgg_conv_layer(self.vgg_conv2_1, 128, 128, 'vgg_conv2_2')
        self.vgg_pool2 = self.pool2d(self.vgg_conv2_2, 'max', 'vgg_pool2')

        self.vgg_conv3_1 = self.vgg_conv_layer(self.vgg_pool2, 128, 256, 'vgg_conv3_1')
        self.vgg_conv3_2 = self.vgg_conv_layer(self.vgg_conv3_1, 256, 256, 'vgg_conv3_2')
        self.vgg_conv3_3 = self.vgg_conv_layer(self.vgg_conv3_2, 256, 256, 'vgg_conv3_3')
        self.vgg_pool3 = self.pool2d(self.vgg_conv3_3, 'max', 'vgg_pool3')

        self.vgg_conv4_1 = self.vgg_conv_layer(self.vgg_pool3, 256, 512, 'vgg_conv4_1')
        self.vgg_conv4_2 = self.vgg_conv_layer(self.vgg_conv4_1, 512, 512, 'vgg_conv4_2')
        self.vgg_conv4_3 = self.vgg_conv_layer(self.vgg_conv4_2, 512, 512, 'vgg_conv4_3')
        self.vgg_pool4 = self.pool2d(self.vgg_conv4_3, 'max', 'vgg_pool4')

        self.vgg_conv5_1 = self.vgg_conv_layer(self.vgg_pool4, 512, 512, 'vgg_conv5_1')
        self.bottleneck5 = self.bottleneck_layer(self.vgg_conv5_1, 512, 128, 'bottleneck5')
        self.conv5_2 = self.vgg_conv_layer(self.bottleneck5, 128, 64, 'conv5_2')
        self.conv5_3 = self.vgg_conv_layer(self.conv5_2, 64, 64, 'conv5_3')
        self.pool5 = self.pool2d(self.conv5_3, 'max', 'pool5')

        l6 = l / 16
        self.deconv6 = self.deconv_layer(self.pool5, name='deconv6',
                                         filter_shape=[3, 3, 64, 64],
                                         output_shape=[batch_size, l6, l6, 64])
        self.pool6 = tf.nn.max_pool(self.deconv6, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
        self.bottleneck6 = self.bottleneck_layer(self.pool6, 64, 32, 'bottleneck6')
        # output size: 16x16x32

        l7 = l6 * 2
        self.deconv7 = self.deconv_layer(self.bottleneck6, name='deconv7',
                                         filter_shape=[3, 3, 16, 32],
                                         output_shape=[batch_size, l7, l7, 16])
        self.pool7 = tf.nn.max_pool(self.deconv7, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
        # output size: 32x32x16

        l8 = l7 * 2
        self.deconv8 = self.deconv_layer(self.pool7, name='deconv8',
                                         filter_shape=[3, 3, 8, 16],
                                         output_shape=[batch_size, l8, l8, 8])
        self.pool8 = tf.nn.max_pool(self.deconv8, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
        # output size: 64x64x8

        l9 = l8 * 2
        self.deconv9 = self.deconv_layer(self.pool8, name='deconv9',
                                         filter_shape=[3, 3, 4, 8],
                                         output_shape=[batch_size, l9, l9, 4])
        self.pool9 = tf.nn.max_pool(self.deconv9, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
        # output size: 128x128x4

        l10 = l9 * 2
        self.deconv10 = self.deconv_layer(self.pool9, name='deconv10',
                                          filter_shape=[3, 3, 1, 4],
                                          output_shape=[batch_size, l10, l10, 1])
        self.pool10 = tf.nn.max_pool(self.deconv10, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')

        self.logits = self.pool10
        self.output = tf.nn.sigmoid(self.pool10)

    def pool2d(self, bottom, mode, name):
        if mode == 'max':
            return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        if mode == 'avg':
            return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def pool3d(self, bottom, ksize, strides, mode, name):
        pool = tf.transpose(bottom, perm=[0, 3, 1, 2])
        pool = tf.expand_dims(pool, 4)
        if mode == 'avg':
            pool = tf.nn.avg_pool3d(pool, ksize=ksize, strides=strides, padding='SAME', name=name)
        if mode == 'max':
            pool = tf.nn.max_pool3d(pool, ksize=ksize, strides=strides, padding='SAME', name=name)
        pool = tf.squeeze(pool, [4])
        pool = tf.transpose(pool, perm=[0, 2, 3, 1])
        return pool

    def vgg_conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_vgg_conv_var(3, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt = self.get_conv_var(3, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            return conv

    def bottleneck_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_vgg_conv_var(1, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            #prelu = self.prelu(bias, name)
            return relu

    def deconv_layer(self, bottom, filter_shape, output_shape, name):
        with tf.variable_scope(name):
            filt = self.get_deconv_var(filter_shape, name)
            deconv = tf.nn.conv2d_transpose(bottom, filt, output_shape=output_shape, strides=[1,2,2,1], padding='SAME')
            return deconv

    def get_vgg_conv_var(self, filter_size, in_channels, out_channels, name):
        initialize = tf.contrib.layers.xavier_initializer()
        filter_shape = [filter_size, filter_size, in_channels, out_channels]
        initial_value = initialize(filter_shape)
        filters = self.get_var(initial_value, name, 0, name + '_filters')

        initial_value = tf.zeros([out_channels])
        biases = self.get_var(initial_value, name, 1, name + '_biases')

        return filters, biases

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initialize = tf.contrib.layers.xavier_initializer()
        filter_shape = [filter_size, filter_size, in_channels, out_channels]
        initial_value = initialize(filter_shape)
        filters = self.get_var(initial_value, name, 0, name + '_filters')

        return filters

    def get_deconv_var(self, shape, name):
        initialize = tf.contrib.layers.xavier_initializer()
        initial_value = initialize(shape)
        filters = self.get_var(initial_value, name, 0, name + '_filters')
        return filters

    def prelu(self, _x, name):
        shape = [_x.get_shape()[-1]]
        initialize = tf.constant_initializer(0.0)
        initial_value = initialize(shape)
        alphas = self.get_var(initial_value, name, 2, name+'_alphas')
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def get_var(self, initial_value, name, idx, var_name):
        if name.startswith('vgg_'):
            vgg_name = name[4:]
            if self.vgg_dict is not None and vgg_name in self.vgg_dict:
                value = self.vgg_dict[vgg_name][idx]
            else:
                value = initial_value
                print('%s is a new variable' % var_name)
            var = tf.constant(value, dtype=tf.float32, name=var_name)
            assert var.get_shape() == initial_value.get_shape()
            return var
        else:
            if self.data_dict is not None and name in self.data_dict:
                value = self.data_dict[name][idx]
            else:
                value = initial_value
                print('%s is a new variable' % var_name)
            var = tf.Variable(value, name=var_name)
            self.var_dict[(name, idx)] = var
            assert var.get_shape() == initial_value.get_shape()
            return var

    def save_npy(self, sess, npy_path="./bgsnet.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

class detector:
    def __init__(self, vgg_path, net_path):
        self.net = BGSNet(vgg_path, net_path)
        self.l = 320
        self.x = tf.placeholder(tf.float32, shape=[None, self.l, self.l, 3], name='input')
        self.net.build(self.x, self.l, 1)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    def apply(self, frame):
        h, w, c = frame.shape
        input = cv2.resize(frame, (self.l, self.l))
        input = np.expand_dims(input, axis=0)
        output = self.net.output.eval(session=self.sess, feed_dict={self.x: input})
        output = output.reshape([self.l, self.l])
        output = (output>0.275).astype('uint8') * 255
        output = cv2.resize(output, (w, h))
        return output
