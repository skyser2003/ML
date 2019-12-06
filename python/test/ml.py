import math

import tensorflow as tf
import numpy as np
from scipy.special import expit

from tensorflow.examples.tutorials.mnist import input_data

class Neuron:
    def __init__(self, **kwargs):
        # Private

        # Layer 1
        self._x = None

        self._w1 = None
        self._b1 = None

        self._wDiff1 = None
        self._bDiff1 = None

        # Layer 2
        self._t = None

        self._w2 = None
        self._b2 = None

        self._wDiff2 = None
        self._bDiff2 = None

    def set(self, w, b):
        self._w1, self._w2 = w[0], w[1]
        self._b1, self._b2 = b[0], b[1]

    def forpass(self, x):
        self._x = x
        self._t = self._sigmoid(x @ self._w1 + self._b1)
        y_hat = self._t @ self._w2 + self._b2

        return self._softmax(y_hat)

    def backprop(self, err, lr):
        self._wDiff2 = lr * (self._t.T @ err) / self._x.shape[0]
        self._bDiff2 = lr * np.average(err, axis=0)

        err2 = (err @ self._w2.T) * self._t * (1 - self._t)

        self._wDiff1 = lr * (self._x.T @ err2) / self._x.shape[0]
        self._bDiff1 = lr * np.average(err2, axis=0)

    def update(self, l2):
        w1 = self._w1 + self._wDiff1 - l2 * self._w1
        b1 = self._b1 + self._bDiff1 - l2 * self._b1

        w2 = self._w2 + self._wDiff2 - l2 * self._w2
        b2 = self._b2 + self._bDiff2 - l2 * self._b2

        self.set([w1, w2], [b1, b2])

    def fit(self, x, y, n_iter, lr=0.1, l2=0, cost_check=False):
        percent = 0
        prev_percent = 0

        for i in range(n_iter):
            # Print percentage
            percent = int(i / n_iter * 100)

            if(math.floor(percent / 10) != math.floor(prev_percent / 10)):
                print(percent, "% done")

            prev_percent = percent

            # Calc
            y_hat = self.forpass(x)
            err = y - y_hat
            self.backprop(err, lr)
            self.update(l2)

    def _sigmoid(self, y_hat):
        return expit(y_hat)

    def _softmax(self, y_hat):
        exp = np.exp(y_hat - y_hat.max(axis=1).reshape(-1,1))
        return exp / exp.sum(axis=1).reshape(-1, 1)

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

images = mnist.test.images
answers = mnist.test.labels

num_data = len(images)
num_dots = len(images[0])
num_hidden = 100
num_one_hot = 10

bound = np.sqrt(1./num_dots)
initW1 = np.random.uniform(-bound, bound, (num_dots, num_hidden))
initB1 = np.random.uniform(-bound, bound, num_hidden)

bound = np.sqrt(1./num_hidden)
initW2 = np.random.uniform(-bound, bound, (num_hidden, num_one_hot))
initB2 = np.random.uniform(-bound, bound, num_one_hot)

print("Begin run")

n_iter = 1000

n = Neuron()
n.set([initW1, initW2], [initB1, initB2])
n.fit(images, answers, n_iter)

final = n.forpass(images)

print("End run")

correct = 0
wrong = 0

for i in range(num_data):
    min_y = np.argmax(final[i], axis=0)
    min_answer = np.argmax(answers[i], axis=0)

    if min_y == min_answer:
        correct += 1
    else:
        wrong += 1

print(correct)
print(wrong)
print("ratio: ")
print(correct / (correct + wrong))