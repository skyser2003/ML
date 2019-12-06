import math

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

class Neuron:
    def init(self, num_data, num_dots, num_hidden, num_one_hot):
        # Tensorboard TODO

        # Data
        with tf.name_scope("Variable") as scope:
            self._variable_scope = scope
            self._w1 = tf.Variable(tf.zeros([num_dots, num_hidden]), name="w1")
            self._b1 = tf.Variable(tf.zeros([num_data, num_hidden]), name="b1")
            self._wDiff1 = tf.Variable(tf.zeros([num_dots, num_hidden]), name="wDiff1")
            self._bDiff1 = tf.Variable(tf.zeros([num_data, num_hidden]), name="bDiff1")

            self._t = tf.Variable(tf.zeros([num_data, num_hidden]), name="t")
            self._w2 = tf.Variable(tf.zeros([num_hidden, num_one_hot]), name="w2")
            self._b2 = tf.Variable(tf.zeros([num_data, num_one_hot]), name="b2")
            self._wDiff2 = tf.Variable(tf.zeros([num_hidden, num_one_hot]), name="wDiff2")
            self._bDiff2 = tf.Variable(tf.zeros([num_data, num_one_hot]), name="bDiff2")

            self._y_hat = tf.Variable(tf.zeros([num_data, num_one_hot]), name="y_hat")

        init = tf.global_variables_initializer()
        config = tf.ConfigProto(
           # device_count = {'GPU': 0}
        )
        self._session = tf.Session(config=config)
        self._session.run(init)

    def destroy(self):
        self._session.close()

    def set(self, w, b):
        with tf.name_scope("Set"):
            w1_u = self._w1.assign(w[0])
            b1_u = self._b1.assign(b[0])

            w2_u = self._w2.assign(w[1])
            b2_u = self._b2.assign(b[1])

        self._session.run([w1_u, b1_u, w2_u, b2_u])

    def get_result(self, x):
        self._session.run(self._forpass(x))

        return self._session.run(self._y_hat)

    def calc_accuracy(self, x, y):
        correct = 0
        wrong = 0

        num_data = y.shape[0]
        y_hat = self._session.run(self._y_hat)

        for i in range(num_data):
            min_y = np.argmax(y_hat[i], axis=0)
            min_answer = np.argmax(y[i], axis=0)

            if min_y == min_answer:
                correct += 1
            else:
                wrong += 1

        return [correct, wrong]

    def fit(self, x, y, n_iter, lr = 0.1, l2 = 0):
        ss = self._session

        percent = 0
        prev_percent = 0
        
        y_ = tf.constant(y, dtype = "float32", name="y")
        
        with tf.name_scope(self._variable_scope):
            err = tf.subtract(y_, self._y_hat, name="err")
   
        run_tensors = []

        run_tensors.append(self._forpass(x))
        run_tensors.append(self._backprop(x, err, lr))
        run_tensors.append(self._update_grad(l2))

        with tf.name_scope("Error_Graph"):
            mean, var = tf.nn.moments(err, axes=[0,1])
            accuracy = tf.placeholder("float")

            tf.summary.scalar("mean", mean)
            tf.summary.scalar("var", var)
            tf.summary.scalar("min", tf.reduce_min(err))
            tf.summary.scalar("max", tf.reduce_max(err))
            tf.summary.scalar("accuracy", accuracy)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log', self._session.graph)

        for i in range(n_iter):
             # Print percentage
            percent = int(i / n_iter * 100)

            if(math.floor(percent / 10) != math.floor(prev_percent / 10)):
                print(percent, "% done")

            prev_percent = percent

            # Tensorboard
            correct, wrong = self.calc_accuracy(x, y)
            train_writer.add_summary(self._session.run(merged, feed_dict = {accuracy: correct / (correct + wrong)}), i)

            # Calc
            ss.run(run_tensors)

        print("100% done")

    def _backprop(self, x, err, lr):
        with tf.name_scope("Backprop"):
            wDiff2_u = self._wDiff2.assign(lr / x.shape[0].value * (tf.transpose(self._t) @ err))
            bDiff2_u = self._bDiff2.assign(
                tf.tile(lr * tf.reduce_mean(err, axis=[0], keep_dims=True), [self._bDiff2.shape[0].value, 1])
                )

            err2 = (err @ tf.transpose(self._w2)) * self._t * (1 - self._t) # TODO * @ difference?

            wDiff1_u = self._wDiff1.assign(
                lr / x.shape[0].value * (tf.transpose(x) @ err2)
                )
            bDiff1_u = self._bDiff1.assign(
                tf.tile(lr * tf.reduce_mean(err2, axis=[0], keep_dims=True), [self._bDiff1.shape[0].value, 1])
                )

            return [wDiff2_u, bDiff2_u, wDiff1_u, bDiff1_u]

    def _update_grad(self, l2):
        with tf.name_scope("Update_Grad"):
            w1_u = self._w1.assign(self._w1 + self._wDiff1 - l2 * self._w1)
            b1_u = self._b1.assign(self._b1 + self._bDiff1 - l2 * self._b1)

            w2_u = self._w2.assign(self._w2 + self._wDiff2 - l2 * self._w2)
            b2_u = self._b2.assign(self._b2 + self._bDiff2 - l2 * self._b2)

            return [w1_u, b1_u, w2_u, b2_u]

    def _forpass(self, x):
        with tf.name_scope("Forpass"):
            t_u = self._t.assign(self._sigmoid(x @ self._w1 + self._b1))

            y_hat_temp = self._t @ self._w2 + self._b2
            y_hat_u = self._y_hat.assign(self._softmax(y_hat_temp))

            return [t_u, y_hat_u]

    def _sigmoid(self, y_hat):
        with tf.name_scope("Sigmoid"):
            return 1.0 / (1.0 + tf.exp(tf.negative(y_hat)))

    def _softmax(self, y_hat):
        with tf.name_scope("Softmax"):
            exp = tf.exp(y_hat - tf.reduce_max(y_hat, axis=[1], keep_dims=True))
            soft_u = exp / tf.reduce_sum(exp, axis=[1], keep_dims=True)

            return soft_u

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

images = mnist.test.images
answers = mnist.test.labels

num_data = images.shape[0]
num_dots = images.shape[1]
num_hidden = 100
num_one_hot = 10

n_iter = 10000

print("Begin")

# Initial data
bound = np.sqrt(1.0 / num_dots)
initW1 = np.random.uniform(-bound, bound, (num_dots, num_hidden))
initB1 = np.repeat(np.random.uniform(-bound, bound, (1, num_hidden)), num_data, axis=0)

bound = np.sqrt(1.0 / num_hidden)
initW2 = np.random.uniform(-bound, bound, (num_hidden, num_one_hot))
initB2 = np.repeat(np.random.uniform(-bound, bound, (1, num_one_hot)), num_data, axis=0)

# Neuron
with tf.name_scope("Input"):
    x = tf.constant(images, dtype="float32", name="x")

n = Neuron()
n.init(num_data, num_dots, num_hidden, num_one_hot)
n.set([initW1, initW2], [initB1, initB2])
n.fit(x, answers, n_iter)

correct, wrong = n.calc_accuracy(x, answers)
    
print(correct)
print(wrong)
print("ratio: ")
print(correct / (correct + wrong))

print ("End")

n.destroy()