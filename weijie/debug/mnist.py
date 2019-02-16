import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.double(x_train)
x_test = np.double(x_test)

#
def data_preprocess(x, y):
    # 0-1 Normalization
    x = x - tf.reduce_min(x)
    x = x / tf.reduce_max(x)
    x = tf.expand_dims(x, -1)
    # one hot
    y = tf.one_hot(y, 10, dtype=tf.float64)

    return x, y


x_placeholder = tf.placeholder(dtype=tf.float64, shape=[None, 28, 28])
y_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices((x_placeholder, y_placeholder))
dataset = dataset.map(data_preprocess).shuffle(60000).batch(250).repeat().make_initializable_iterator()
x, y = dataset.get_next()


def network(input_):
    recon = tf.layers.conv2d(inputs=input_, filters=64, kernel_size=9, padding='same', activation=tf.nn.relu)
    recon = tf.layers.max_pooling2d(inputs=recon, pool_size=(2, 2), strides=(2, 2))
    recon = tf.layers.conv2d(inputs=recon, filters=32, kernel_size=9, padding='same', activation=tf.nn.relu)
    recon = tf.layers.max_pooling2d(inputs=recon, pool_size=(2, 2), strides=(2, 2))
    recon = tf.layers.flatten(recon)
    recon = tf.layers.dense(recon, 1024, activation=tf.nn.relu)
    recon = tf.layers.dense(recon, 10, activation=tf.nn.softmax)

    return recon


recon_ = network(x)
loss = -1*tf.reduce_sum(y*tf.log(recon_))
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(recon_,1), tf.argmax(y,1)), tf.float64))

with tf.Session() as sess:
    sess.run(dataset.initializer, feed_dict={x_placeholder: x_train, y_placeholder: y_train})
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        for i in range(200):
            _, loss_, accuracy_ = sess.run([train_op, loss, accuracy])
            if (i+1) % 25 == 0:
                print('Training... Epoch: [%d], Iter: [%d], Loss: [%.6f], Acc: [%.6f]' % (epoch, i, loss_, accuracy_))

        sess.run(dataset.initializer, feed_dict={x_placeholder: x_test, y_placeholder: y_test})

        accuracy_sum = 0
        for i in range(40):
            accuracy_sum += sess.run(accuracy)

        print('Testing... Epoch: [%d] Accuracy: [%.6f]' % (epoch, accuracy_sum / 40))
