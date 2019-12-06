import tensorflow as tf
import numpy as np

import util.plot_utils as plot_utils
import util.helper as helper

from cq.cq_data import CqData, CqDataType

class CqLSGAN:
    def __init__(self, num_hidden, g_size):
        self.num_hidden = num_hidden
        self.g_size = g_size
        self.data = CqData(CqDataType.FACE)

    def random_z(self, m, n):
        return np.random.uniform(-1.0, 1.0, [m,n])

    def uniform_z(self, m, n):
        ret = np.arange(-m, m, 2)
        ret = np.tile(ret, [n, 1])
        ret = ret / m
        ret = np.transpose(ret)

        return ret

    def get_generator_vars(self, input_size, image_size):
        init_w = tf.contrib.layers.variance_scaling_initializer()
        init_b = tf.constant_initializer(0)

        num_hidden = self.num_hidden

        with tf.variable_scope("Generator_Variables"):
            w1 = tf.get_variable("w1", [input_size, num_hidden], initializer=init_w)
            b1 = tf.get_variable("b1", [num_hidden], initializer=init_b)

            w2 = tf.get_variable("w2", [num_hidden, image_size], initializer=init_w)
            b2 = tf.get_variable("b2", [image_size], initializer=init_b)

        return [w1, b1, w2, b2]

    def get_discriminator_vars(self, image_size):
        init_w = tf.contrib.layers.variance_scaling_initializer()
        init_b = tf.constant_initializer(0)

        num_hidden = self.num_hidden

        with tf.variable_scope("Discriminator_Variables"):
            w1 = tf.get_variable("w1", [image_size, num_hidden], initializer=init_w)
            b1 = tf.get_variable("b1", [num_hidden], initializer=init_b)

            w2 = tf.get_variable("w2", [num_hidden, 1], initializer=init_w)
            b2 = tf.get_variable("b2", [1], initializer=init_b)

        return [w1, b1, w2, b2]

    def generator(self, input, gen_vars):
        w1, b1, w2, b2 = gen_vars

        with tf.name_scope("Generator_Operations"):
            h1 = tf.nn.leaky_relu(input @ w1 + b1, name="h1")
            h2 = tf.sigmoid(h1 @ w2 + b2, name="h2")

        return h2

    def discriminator(self, input, disc_vars):
        w1, b1, w2, b2 = disc_vars

        with tf.name_scope("Discriminator_Operations"):
            h1 = tf.nn.leaky_relu(input @ w1 + b1, name="h1")
            h2 = tf.identity(h1 @ w2 + b2, name="h2")

        return h2

    def run(self):
        image_width = self.data.get_image_width()
        image_height = self.data.get_image_height()
        num_channel = self.data.get_channel_count()
        image_size = image_width * image_height * num_channel

        x_d = tf.placeholder(tf.float32, shape=[None, image_size], name="x_d")
        z = tf.placeholder(tf.float32, shape=[None, self.g_size], name="z")

        gen_vars = self.get_generator_vars(self.g_size, image_size)
        disc_vars = self.get_discriminator_vars(image_size)

        x_g = self.generator(z, gen_vars)
        disc_g = self.discriminator(x_g, disc_vars)
        disc_d = self.discriminator(x_d, disc_vars)

        with tf.name_scope("Loss"):
            loss_g = tf.multiply(0.5, tf.reduce_mean((disc_g - 1) ** 2), name="loss_g")
            loss_d = tf.multiply(0.5, tf.reduce_mean((disc_d - 1) ** 2) + tf.reduce_mean(disc_g ** 2), name="loss_d")

        # Optimize
        lr = 0.001
        opt_g = tf.train.AdamOptimizer(lr).minimize(loss_g, var_list=gen_vars)
        opt_d = tf.train.AdamOptimizer(lr).minimize(loss_d, var_list=disc_vars)

        # Summary
        with tf.name_scope("Summary"):
            tf.summary.scalar("disc_g", tf.reduce_mean(disc_g))
            tf.summary.scalar("disc_d", tf.reduce_mean(disc_d))
            tf.summary.scalar("loss_g", loss_g)
            tf.summary.scalar("loss_d", loss_d)

        merged = tf.summary.merge_all()

        # Debug variable
        with tf.name_scope("Debug"):
            disc_d_mean = tf.reduce_mean(disc_d)
            disc_g_mean = tf.reduce_mean(disc_g)

        # Run env
        num_step_d = 3
        num_step_g = 1
        num_input_x = 8
        num_input_y = 8
        num_input = num_input_x * num_input_y
        num_batch = 128
        print_tick = int(1e2)
        summary_tick = 10

        # Output
        output_dir = "cq_output_lsgan"
        helper.clean_create_dir(output_dir)

        test_input_x = self.data.get_batch(num_input)

        prr = plot_utils.Plot_Reproduce_Performance(output_dir, num_input_x, num_input_y, image_width, image_height)
        prr.save_pngs(test_input_x, num_channel, "input.png")

        # Config
        config = tf.ConfigProto(
            # allow_soft_placement = True,
            # log_device_placement = True
            )

        with tf.Session(config=config) as ss:
            ss.run(tf.global_variables_initializer())

            # Summary
            writer = tf.summary.FileWriter("cq_log", ss.graph)

            prev_per = 0
            cur_per = 0

            i = 0
            image_i = 0

            batch_x_d = test_input_x

            while True:
                for j in range(num_step_d):
                    # batch_x_d = self.data.get_batch(num_batch)
                    random_z_d = self.random_z(num_batch, self.g_size)

                    feed_dict_d = { x_d: batch_x_d, z: random_z_d }
                    summary, disc_d_val, loss_d_val, _ = ss.run([merged, disc_d_mean, loss_d, opt_d], feed_dict = feed_dict_d)

                for j in range(num_step_g):
                    random_z_g = self.random_z(num_batch, self.g_size)

                    feed_dict_g = { z: random_z_g }
                    disc_g_val, loss_g_val, _ = ss.run([disc_g_mean, loss_g, opt_g], feed_dict = feed_dict_g)

                if i % print_tick == 0:
                    i_div_k = i / 1000
                    print("%.1fk steps done - Disc G : %f, Disc D : %f, Loss G: %f, Loss D: %f" % (i_div_k, disc_g_val, disc_d_val, loss_g_val, loss_d_val))

                    feed_dict = { z: self.uniform_z(num_input, self.g_size) }
                    output_x = ss.run(x_g, feed_dict=feed_dict)
                    prr.save_pngs(output_x, num_channel, name="output%d.png" % (image_i))

                    image_i += 1

                prev_per = cur_per

                # Summary
                if i % summary_tick == 0:
                    writer.add_summary(summary, i)

                i += 1

            # Final output
            feed_dict = { z: self.random_z(num_input, self.g_size) }
            output_x = ss.run(x_g, feed_dict=feed_dict)
            prr.save_pngs(output_x, num_channel, name="final_output.png")

            print("100% done")

gan = CqLSGAN(1024, 1024)
gan.run()