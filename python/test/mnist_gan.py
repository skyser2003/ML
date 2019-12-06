import random
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import util.plot_utils as plot_utils
import util.helper as helper

pixel_size = 28

class GAN:
    def __init__(self, num_hidden, num_cat_c):
        self.num_hidden = num_hidden
        self.num_cat_c = num_cat_c

    def generator(self, input, is_training):
        num_hidden = self.num_hidden

        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0)

        with tf.variable_scope("Generator"):
            w1 = tf.get_variable("w1", [input.get_shape()[1], num_hidden], initializer=w_init)
            b1 = tf.get_variable("b1", [num_hidden], initializer=b_init)

            w2 = tf.get_variable("w2", [num_hidden, pixel_size ** 2], initializer=w_init)
            b2 = tf.get_variable("b2", [pixel_size ** 2], initializer=b_init)

            h1 = input @ w1 + b1
            # h1 = tf.contrib.layers.batch_norm(h1, is_training=is_training)
            h1 = tf.nn.leaky_relu(h1, name="h1")

            h2 = h1 @ w2 + b2
            # h2 = tf.contrib.layers.batch_norm(h2, is_training=is_training)
            h2 = tf.nn.sigmoid(h2, name="h2")

            return h2, [w1, w2, b1, b2]

    def discriminator(self, input, reuse):
        num_hidden = self.num_hidden

        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0)

        with tf.variable_scope("Discriminator", reuse=reuse):
            w1 = tf.get_variable("w1", [input.get_shape()[1], num_hidden], initializer=w_init)
            b1 = tf.get_variable("b1", [num_hidden], initializer=b_init)

            w2 = tf.get_variable("w2", [num_hidden, 1 + self.num_cat_c], initializer=w_init)
            b2 = tf.get_variable("b2", [1 + self.num_cat_c], initializer=b_init)

            h1 = tf.nn.leaky_relu(input @ w1 + b1, name="h1")
            h2 = tf.identity(h1 @ w2 + b2, name="h2")

        cat_output = h2[:, 1:]
        h2 = h2[:, :1]

        return cat_output, h2, [w1, w2, b1, b2]

    def run(self, g_size):
        lr = 0.0002

        x_d = tf.placeholder(tf.float32, shape=[None, pixel_size ** 2], name="x_d")

        cat_input = tf.placeholder(tf.float32, shape=(None, self.num_cat_c), name="cat_c")
        real_z = tf.placeholder(tf.float32, shape=(None, g_size - self.num_cat_c), name="real_z")
        z = tf.concat([real_z, cat_input], 1)
        is_training = tf.placeholder(tf.bool, name="is_training")

        x_g, var_g = self.generator(z, is_training)

        _, disc_d, var_d = self.discriminator(x_d, False)
        cat_output, disc_g, _ = self.discriminator(x_g, True)

        loss_d = tf.identity(tf.reduce_mean((disc_d - 1) ** 2) + tf.reduce_mean(disc_g ** 2), name="loss_d")
        loss_g = tf.reduce_mean((disc_g - 1) ** 2, name="loss_g")

        opt_d = tf.train.AdamOptimizer(lr).minimize(loss_d, var_list=var_d)
        opt_g = tf.train.AdamOptimizer(lr).minimize(loss_g, var_list=var_g)

        if self.num_cat_c != 0:
            loss_cat = tf.nn.softmax_cross_entropy_with_logits(logits=cat_output, labels=cat_input)
            loss_cat = tf.reduce_mean(loss_cat, name="loss_cat")

            loss_d += loss_cat
            loss_g += loss_cat

        # Summary
        with tf.name_scope("Summary"):
            tf.summary.scalar("loss_d", loss_d)
            tf.summary.scalar("loss_g", loss_g)

            if self.num_cat_c != 0:
                tf.summary.scalar("loss_cat", loss_cat)

        merged = tf.summary.merge_all()

        with tf.Session() as ss:
            ss.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter("log", ss.graph)

            # Train
            num_d_step = 3
            batch_size = 128
            output_x = 10
            output_y = 10
            output_size = output_x * output_y

            print_step = int(1e3)
            i = 0

            # Output
            output_cat_val = helper.constant_z(0, [output_size, self.num_cat_c])

            if self.num_cat_c != 0:
                for i in range(output_size):
                    cat_index = int(i / output_x)

                    temp = output_cat_val[i]
                    temp[cat_index] = 1

            print(output_cat_val)

            output_dir = "mnist_output_gan"
            helper.clean_create_dir(output_dir)

            prr = plot_utils.Plot_Reproduce_Performance(output_dir, output_x, output_y)

            while True:
                for j in range(num_d_step):
                    batch_x, _ = mnist.train.next_batch(batch_size)
                    cat_input_val = helper.random_cat([batch_size, self.num_cat_c])
                    real_z_val = helper.random_normal(0, 1, [batch_size, g_size - self.num_cat_c])
                    feed_dict = {x_d: batch_x, cat_input: cat_input_val, real_z: real_z_val, is_training: True}

                    cur_loss_d, _ = ss.run([loss_d, opt_d], feed_dict=feed_dict)

                batch_x, _ = mnist.train.next_batch(batch_size)
                cat_input_val = helper.random_cat([batch_size, self.num_cat_c])
                real_z_val = helper.random_normal(0, 1, [batch_size, g_size - self.num_cat_c])
                feed_dict = {x_d: batch_x, cat_input: cat_input_val, real_z: real_z_val, is_training: True}

                summary, cur_loss_g, _, = ss.run([merged, loss_g, opt_g], feed_dict=feed_dict)

                if self.num_cat_c != 0:
                    cur_loss_cat = ss.run(loss_cat, feed_dict=feed_dict)

                writer.add_summary(summary, i)

                if i % print_step == 0 and i != 0:
                    output_i = int(i / print_step)

                    real_z_val = helper.random_normal(0, 1, [output_cat_val.shape[0], g_size - self.num_cat_c])
                    feed_dict = {cat_input: output_cat_val, real_z: real_z_val, is_training: False}

                    output_x = ss.run([x_g], feed_dict=feed_dict)
                    prr.save_images(np.array(output_x), name="result%d.jpg" % output_i)

                    if self.num_cat_c != 0:
                        print("%d step - Loss D: %f, Loss G: %f, Loss cat: %f" % (i, cur_loss_d, cur_loss_g, cur_loss_cat))
                    else:
                        print("%d step - Loss D: %f, Loss G: %f" % (i, cur_loss_d, cur_loss_g))

                i += 1


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

gan = GAN(256, 10)
gan.run(128)
