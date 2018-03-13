import tensorflow as tf
import numpy as np

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

        self.vgg_conv1_1 = self.conv_layer(bgr, 3, 64, 'vgg_conv1_1')
        self.vgg_conv1_2 = self.conv_layer(self.vgg_conv1_1, 64, 64, 'vgg_conv1_2')
        self.vgg_pool1 = self.pool2d(self.vgg_conv1_2, 'max', 'vgg_pool1')

        self.vgg_conv2_1 = self.conv_layer(self.vgg_pool1, 64, 128, 'vgg_conv2_1')
        self.vgg_conv2_2 = self.conv_layer(self.vgg_conv2_1, 128, 128, 'vgg_conv2_2')
        self.vgg_pool2 = self.pool2d(self.vgg_conv2_2, 'max', 'vgg_pool2')

        self.vgg_conv3_1 = self.conv_layer(self.vgg_pool2, 128, 256, 'vgg_conv3_1')
        self.vgg_conv3_2 = self.conv_layer(self.vgg_conv3_1, 256, 256, 'vgg_conv3_2')
        self.vgg_conv3_3 = self.conv_layer(self.vgg_conv3_2, 256, 256, 'vgg_conv3_3')
        self.vgg_pool3 = self.pool2d(self.vgg_conv3_3, 'max', 'vgg_pool3')

        self.vgg_conv4_1 = self.conv_layer(self.vgg_pool3, 256, 512, 'vgg_conv4_1')
        self.vgg_conv4_2 = self.conv_layer(self.vgg_conv4_1, 512, 512, 'vgg_conv4_2')
        self.vgg_conv4_3 = self.conv_layer(self.vgg_conv4_2, 512, 512, 'vgg_conv4_3')
        self.vgg_pool4 = self.pool2d(self.vgg_conv4_3, 'max', 'vgg_pool4')

        self.vgg_conv5_1 = self.conv_layer(self.vgg_pool4, 512, 512, 'vgg_conv5_1')
        self.bottleneck5 = self.bottleneck_layer(self.vgg_conv5_1, 512, 128, 'bottleneck5')
        self.conv5_2 = self.conv_layer(self.bottleneck5, 128, 128, 'conv5_2')
        #self.conv5_3 = self.conv_layer(self.conv5_2, 128, 128, 'conv5_3')
        self.pool5 = self.pool2d(self.conv5_2, 'max', 'pool5')

        l6 = l / 16
        self.deconv6 = self.deconv_layer(self.pool5, name='deconv6',
                                         filter_shape=[3, 3, 64, 128],
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

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def bottleneck_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(1, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def deconv_layer(self, bottom, filter_shape, output_shape, name):
        with tf.variable_scope(name):
            filt = self.get_deconv_var(filter_shape, name+'_filters')
            deconv = tf.nn.conv2d_transpose(bottom, filt, output_shape=output_shape, strides=[1,2,2,1], padding='SAME')
            return deconv

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.1)
        filters = self.get_var(initial_value, name, 0, name + '_filters')

        initial_value = tf.truncated_normal([out_channels], 0.0, 0.1)
        biases = self.get_var(initial_value, name, 1, name + '_biases')

        return filters, biases

    def get_deconv_var(self, shape, name):
        initial_value = tf.truncated_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
        filters = self.get_var(initial_value, name, 0, name)
        return filters

    def get_var(self, initial_value, name, idx, var_name):
        if name.startswith('vgg_'):
            vgg_name = name[4:]
            if self.vgg_dict is not None and vgg_name in self.vgg_dict:
                value = self.vgg_dict[vgg_name][idx]
            else:
                value = initial_value
            var = tf.constant(value, dtype=tf.float32, name=var_name)
            assert var.get_shape() == initial_value.get_shape()
            return var
        else:
            if self.data_dict is not None and name in self.data_dict:
                value = self.data_dict[name][idx]
            else:
                value = initial_value
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