import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data

class RNN:
    def __init__(self):
        pass

    def RNN(self, input, num_hidden):
        num_step = input.get_shape()[0]
        num_cell_loop = input.get_shape()[1]
        num_input = input.get_shape()[2]

        input = tf.transpose(input, [1, 0, 2])
        input = tf.reshape(input, [-1, num_input])
        input = tf.split(input, num_or_size_splits=num_cell_loop, axis=0)

        cell = rnn.BasicLSTMCell(num_hidden)

        outputs, _ = rnn.static_rnn(cell, input, dtype=tf.float32)

        return outputs[-1]

    def calc(self, input, labels, num_hidden):
        with tf.name_scope("Variable"):
            w1 = tf.get_variable("w1", (num_hidden, 10))
            b1 = tf.get_variable("b1", (10))
            
            output = tf.identity(input @ w1 + b1, name="output")

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels), name="loss")
            return output, loss

    def accuracy(self, y_, y):
        correctness = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        return tf.reduce_mean(tf.cast(correctness, tf.float32))

    def run(self):
        # Train data
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

        num_input = 28
        num_cell_loop = 28
        num_hidden = 32

        # Train
        lr = 0.001
        num_batch = 100
        num_step = int(1e3)
        num_print_step = num_step / 10

        # Test
        x = tf.placeholder(tf.float32, [None, num_cell_loop, num_input])
        y = tf.placeholder(tf.float32, [None, 10])

        data = mnist.test.images.reshape(-1, num_cell_loop, num_input)
        labels = mnist.test.labels
        feed_dict_test = { x: data, y: labels }

        cell = self.RNN(x, num_hidden)
        y_, loss = self.calc(cell, y, num_hidden)
        opt = tf.train.AdamOptimizer(lr).minimize(loss)
        acc = self.accuracy(y_, y)

        with tf.Session() as ss:
            ss.run(tf.global_variables_initializer())

            for i in range(num_step):
                batch_x, batch_y = mnist.train.next_batch(num_batch)
                batch_x = batch_x.reshape(-1, num_cell_loop, num_input)
                feed_dict = { x: batch_x, y: batch_y }

                ss.run(opt, feed_dict=feed_dict)

                if i % num_print_step == 0 and i != 0:
                    loss_val, acc_val = ss.run([loss, acc], feed_dict=feed_dict_test)
                    print("%d%% done - Loss: %f, accuracy: %f" % (100 * i / num_step, loss_val, acc_val))

            loss_val, acc_val = ss.run([loss, acc], feed_dict=feed_dict_test)
            print("100%% done - Loss: %f, accuracy: %f" % (loss_val, acc_val))

test = RNN()
test.run()