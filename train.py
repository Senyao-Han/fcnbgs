from __future__ import print_function
from bgsnet import BGSNet
from data import DataLoader
import time
import tensorflow as tf

train_dir = '/home/yzhq/data/CDNet2014/trainset/'
test_dir = '/home/yzhq/data/CDNet2014/testset/'
l = 320
batch_size = 40

train_loader = DataLoader(data_dir=train_dir, batch_size=batch_size, image_shape=(l, l))
test_loader = DataLoader(data_dir=test_dir, batch_size=batch_size, image_shape=(l, l))

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, l, l, 3], name='input')
y = tf.placeholder(tf.float32, shape=[None, l, l, 1], name='groundtruth')

net = BGSNet('vgg16_conv.npy')
net.build(x, l, batch_size)
print('%d vars to train' % net.get_var_count())

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=net.logits, labels=y)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdadeltaOptimizer()
train = optimizer.minimize(loss)

num_epochs = 1
num_batches = train_loader.num_batches
num_test_batches = test_loader.num_batches
interval = 100
sess.run(tf.global_variables_initializer())

print('training started')
stime = time.time()
for epoch in range(num_epochs):
    for batch in range(num_batches):
        batch_x, batch_y = train_loader.next_batch()
        sess.run(train, feed_dict={x: batch_x, y: batch_y})

        if (batch+1) % interval == 0:
            ntime = time.time()
            cost_time = (ntime-stime)/interval
            stime = ntime

            total_loss = 0.0
            for tbatch in range(num_test_batches):
                batch_x, batch_y = test_loader.next_batch_threading()
                loss_val = loss.eval(session=sess, feed_dict={x: batch_x, y: batch_y})
                total_loss += loss_val
            total_loss = total_loss/num_test_batches

            print('epoch %d, batch %d/%d, test loss %f. average %f seconds per batch' % (epoch, batch+1, num_batches, total_loss, cost_time))

net.save_npy(sess)
sess.close()
