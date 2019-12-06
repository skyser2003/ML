import random, math
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
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

        init_w = tf.random_normal_initializer(stddev=0.02)
        init_b = tf.constant_initializer(0)

        with tf.variable_scope("Generator") as scope:
            fc1 = tc.layers.fully_connected(input, pixel_size ** 2 // 16 * 512,
                                            weights_initializer=init_w,
                                            biases_initializer=init_b,
                                            activation_fn=tf.identity)

            fc1 = tc.layers.batch_norm(fc1, is_training=is_training)
            fc1 = tf.nn.leaky_relu(fc1)
            fc1 = tf.reshape(fc1, [-1, 7, 7, 512])

            conv1 = tc.layers.conv2d_transpose(fc1, 256, 5, 2,
                                               weights_initializer=init_w,
                                               biases_initializer=init_b,
                                               activation_fn=tf.identity)

            conv1 = tc.layers.batch_norm(conv1, is_training=is_training)
            conv1 = tf.nn.leaky_relu(conv1)

            conv2 = tc.layers.conv2d_transpose(conv1, 1, 5, 2,
                                               weights_initializer=init_w,
                                               biases_initializer=init_b,
                                               activation_fn=tf.identity)

            conv2 = tc.layers.flatten(conv2)
            conv2 = tf.sigmoid(conv2)

            h2 = conv2

            vars_gen = [var for var in tf.global_variables() if var.name.startswith(scope.name)]

            return h2, vars_gen

    def discriminator(self, input, is_training, reuse):
        init_w = tf.random_normal_initializer(stddev=0.02)
        init_b = tf.constant_initializer(0)

        num_initial_filter = 64

        with tf.variable_scope("Discriminator", reuse=reuse) as scope:
            input_dc = tf.reshape(input, [-1, pixel_size, pixel_size, 1])

            conv1 = tc.layers.conv2d(input_dc, num_initial_filter, 5, 2,
                                     weights_initializer=init_w,
                                     biases_initializer=init_b,
                                     activation_fn=tf.identity)

            conv1 = tc.layers.batch_norm(conv1, is_training=is_training)
            conv1 = tf.nn.leaky_relu(conv1)

            conv2 = tc.layers.conv2d(conv1, num_initial_filter * 2, 5, 2,
                                     weights_initializer=init_w,
                                     biases_initializer=init_b,
                                     activation_fn=tf.identity)

            conv2 = tc.layers.batch_norm(conv2, is_training=is_training)
            conv2 = tf.nn.leaky_relu(conv2)

            conv3 = tc.layers.conv2d(conv2, num_initial_filter * 4, 5, 2,
                                     weights_initializer=init_w,
                                     biases_initializer=init_b,
                                     activation_fn=tf.identity)

            conv3 = tf.nn.leaky_relu(conv3)

            conv4 = tc.layers.conv2d(conv3, 1 + self.num_cat_c, (conv3.shape[1], conv3.shape[2]), 1,
                                     weights_initializer=init_w,
                                     biases_initializer=init_b,
                                     activation_fn=tf.identity)

            conv4 = tc.layers.flatten(conv4)
            h2 = conv4

        cat_output = h2[:, 1:]
        h2 = h2[:, :1]

        vars_disc = [var for var in tf.global_variables() if var.name.startswith(scope.name)]

        return cat_output, h2, vars_disc

    def run(self, g_size):
        lr = 0.0002

        x_d = tf.placeholder(tf.float32, shape=[None, pixel_size ** 2], name="x_d")

        cat_input = tf.placeholder(tf.float32, shape=(None, self.num_cat_c), name="cat_c")
        real_z = tf.placeholder(tf.float32, shape=(None, g_size - self.num_cat_c), name="real_z")
        z = tf.concat([real_z, cat_input], 1)
        is_training = tf.placeholder(tf.bool, name="is_training")

        x_g, var_g = self.generator(z, is_training)

        _, disc_d, var_d = self.discriminator(x_d, is_training, False)
        cat_output, disc_g, _ = self.discriminator(x_g, is_training, True)

        #loss_d = tf.identity(tf.reduce_mean((disc_d - 1) ** 2) + tf.reduce_mean(disc_g ** 2), name="loss_d")
        #loss_g = tf.reduce_mean((disc_g - 1) ** 2, name="loss_g")
        loss_d = tf.reduce_mean(disc_d - disc_g)
        loss_g = tf.reduce_mean(disc_g)

        eps = tf.random_uniform([], 0.0, 1.0)
        scale = 10.0

        x_pn = eps * x_d + (1 - eps) * x_g
        _, disc_pn, _ = self.discriminator(x_pn, is_training, True)

        grad = tf.gradients(disc_pn, x_pn)[0]
        grad = tf.norm(grad, axis=1)
        ddx = tf.reduce_mean(scale * tf.square(grad - 1.0))

        loss_d = loss_d + ddx

        opt_d = tf.train.AdamOptimizer(lr).minimize(loss_d, var_list=var_d)
        opt_g = tf.train.AdamOptimizer(lr).minimize(loss_g, var_list=var_g)

        if self.num_cat_c != 0:
            loss_cat = tf.nn.softmax_cross_entropy_with_logits(logits=cat_output, labels=cat_input)
            loss_cat = tf.reduce_mean(loss_cat, name="loss_cat")

            opt_cat = tf.train.AdamOptimizer(lr).minimize(loss_cat)

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
            batch_size = 64
            output_x = 8
            output_y = 8
            output_size = output_x * output_y

            print_step = int(1e2)
            i = 0

            # Output
            output_cat_val = helper.constant_z(0, [output_size, self.num_cat_c])

            if self.num_cat_c != 0:
                for i in range(output_size):
                    cat_index = int(i / output_x)

                    temp = output_cat_val[i]
                    temp[cat_index] = 1

            print(output_cat_val)

            output_dir = "mnist_output_dcgan"
            helper.clean_create_dir(output_dir)

            prr = plot_utils.Plot_Reproduce_Performance(output_dir, output_x, output_y)

            mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

            while True:
                for j in range(num_d_step):
                    batch_x, _ = mnist.train.next_batch(batch_size)
                    cat_input_val = helper.random_cat([batch_size, self.num_cat_c])
                    real_z_val = helper.random_normal(0, 1, [batch_size, g_size - self.num_cat_c])
                    feed_dict = {x_d: batch_x, cat_input: cat_input_val, real_z: real_z_val, is_training: True}

                    # ss.run(clip)
                    cur_loss_d, _ = ss.run([loss_d, opt_d], feed_dict=feed_dict)

                batch_x, _ = mnist.train.next_batch(batch_size)
                cat_input_val = helper.random_cat([batch_size, self.num_cat_c])
                real_z_val = helper.random_normal(0, 1, [batch_size, g_size - self.num_cat_c])
                feed_dict = {x_d: batch_x, cat_input: cat_input_val, real_z: real_z_val, is_training: True}

                summary, cur_loss_g, _, = ss.run([merged, loss_g, opt_g], feed_dict=feed_dict)

                if self.num_cat_c != 0:
                    _, cur_loss_cat = ss.run([opt_cat, loss_cat], feed_dict=feed_dict)

                writer.add_summary(summary, i)

                if i % print_step == 0 and i != 0:
                    output_i = int(i / print_step)

                    real_z_val = helper.random_normal(0, 1, [output_cat_val.shape[0], g_size - self.num_cat_c])
                    feed_dict = {cat_input: output_cat_val, real_z: real_z_val, is_training: False}

                    output_x = ss.run([x_g], feed_dict=feed_dict)
                    prr.save_images(np.array(output_x), name="result%d.jpg" % output_i)

                    if self.num_cat_c != 0:
                        print("%d step - Loss D: %f, Loss G: %f, Loss cat: %f" % (
                            i, cur_loss_d, cur_loss_g, cur_loss_cat))
                    else:
                        print("%d step - Loss D: %f, Loss G: %f" % (i, cur_loss_d, cur_loss_g))

                i += 1


gan = GAN(1024, 0)
gan.run(100)
