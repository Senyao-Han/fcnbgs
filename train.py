from __future__ import print_function
from bgsnet import BGSNet
from data import DataLoader
import time
import tensorflow as tf

def test(sess, testloader):
    stime = time.time()
    total_loss = 0.0
    test_loader.reset()
    for tbatch in range(num_test_batches):
        batch_x, batch_y = train_loader.next_batch()
        loss_val = loss.eval(session=sess, feed_dict={x: batch_x, y: batch_y})
        total_loss += loss_val
    total_loss = total_loss / num_test_batches
    cost_time = time.time() - stime
    print('test loss: %f obtained in %f seconds' % (total_loss, cost_time))


train_dir = '/home/yzhq/data/bgsnet/train/'
test_dir = '/home/yzhq/data/bgsnet/test/'
l = 320
batch_size = 40

train_loader = DataLoader(data_dir=train_dir, batch_size=batch_size, image_shape=(l, l))
test_loader = DataLoader(data_dir=test_dir, batch_size=batch_size, image_shape=(l, l))

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, l, l, 3], name='input')
y = tf.placeholder(tf.float32, shape=[None, l, l, 1], name='groundtruth')

net = BGSNet('vgg16partial.npy')
net.build(x, l, batch_size)
print('%d vars to train' % net.get_var_count())

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=net.logits, labels=y)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

num_epochs = 30
num_batches = train_loader.num_batches
num_test_batches = test_loader.num_batches
interval = 100
sess.run(tf.global_variables_initializer())
print('train batches: %d, test batches: %d' % (num_batches, num_test_batches))

test(sess, test_loader)

print('training started')
stime = time.time()
for epoch in range(1, num_epochs+1):

    train_loader.reset()
    for batch in range(num_batches):
        batch_x, batch_y = train_loader.next_batch()
        sess.run(train, feed_dict={x: batch_x, y: batch_y})
        if (batch+1) % interval == 0:
            ntime = time.time()
            cost_time = (ntime-stime)/interval
            stime = ntime
            print('epoch %d, batch %d/%d, average %f seconds per batch' %
                  (epoch, batch+1, num_batches, cost_time))

    test(sess, test_loader)

    net.save_npy(sess, 'train/bgsnet%d.npy' % (epoch))

sess.close()
