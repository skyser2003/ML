import tensorflow as tf

import python.util.helper as helper
import python.util.plot_utils as plot_utils
from python.cq.cq_data import CqData, CqDataType, CqDataMode

# For FACE images
scale = 3


class CqDCGAN:
    def __init__(self):
        self.data = CqData(CqDataType.FACE, mode=CqDataMode.BG_BLACK, scale_down=scale)

    @staticmethod
    def discriminator_vars(num_hidden, image_size, num_latent_cat):
        init_w = tf.contrib.layers.variance_scaling_initializer()
        init_b = tf.constant_initializer(0)

        with tf.variable_scope("Discriminator_variables"):
            w1 = tf.get_variable("w1", (image_size, num_hidden), initializer=init_w)
            b1 = tf.get_variable("b1", (num_hidden), initializer=init_b)

            w2 = tf.get_variable("w2", (num_hidden, 1 + num_latent_cat), initializer=init_w)
            b2 = tf.get_variable("b2", (1 + num_latent_cat), initializer=init_b)

        return w1, b1, w2, b2

    @staticmethod
    def generator(input, hidden_size, image_size, dropout):
        init_w = tf.contrib.layers.variance_scaling_initializer()
        init_b = tf.constant_initializer(0)

        with tf.variable_scope("Generator_variables"):
            w1 = tf.get_variable("w1", (input.get_shape()[1], hidden_size), initializer=init_w)
            b1 = tf.get_variable("b1", (hidden_size), initializer=init_b)

            w2 = tf.get_variable("w2", (hidden_size, image_size), initializer=init_w)
            b2 = tf.get_variable("b2", (image_size), initializer=init_b)

        with tf.variable_scope("Generator_operations"):
            h1 = tf.nn.leaky_relu(input @ w1 + b1, name="h1")
            h1 = tf.nn.dropout(h1, dropout)
            h2 = tf.sigmoid(h1 @ w2 + b2, name="h2")

        return (w1, b1, w2, b2), h2

    @staticmethod
    def discriminator(input, discriminator_vars):
        w1, b1, w2, b2 = discriminator_vars

        with tf.variable_scope("Discriminator_operations"):
            h1 = tf.nn.leaky_relu(input @ w1 + b1, name="h1")
            h2 = tf.identity(h1 @ w2 + b2)

            output_code = tf.nn.softmax(h2[:, 1:], name="output_code")
            h2 = tf.identity(h2[:, :1], name="h2")

        return h2, output_code

    def run(self, num_z, num_hidden, num_latent_cat):
        # Input data
        image_width = self.data.get_image_width()
        image_height = self.data.get_image_height()
        num_channel = self.data.get_channel_count()
        image_size = image_width * image_height * num_channel

        with tf.name_scope("Placeholder"):
            real_z = tf.placeholder(tf.float32, (None, num_z - num_latent_cat), name="real_z")
            input_cat = tf.placeholder(tf.float32, (None, num_latent_cat), name="input_cat")
            z = tf.concat([real_z, input_cat], 1)
            x = tf.placeholder(tf.float32, (None, image_size), name="x")
            dropout = tf.placeholder(tf.float32, name="dropout")

        # Network
        vars_gen, x_g = self.generator(z, num_hidden, image_size, dropout)
        vars_disc = self.discriminator_vars(num_hidden, image_size, num_latent_cat)
        disc_g, output_cat_g = self.discriminator(x_g, vars_disc)
        disc_d, _ = self.discriminator(x, vars_disc)

        with tf.name_scope("Loss"):
            loss_d = tf.reduce_mean(disc_d - disc_g)
            loss_g = tf.reduce_mean(disc_g)

            eps = tf.random_uniform([], 0, 1)
            scale_fn = 10

            x_pn = eps * x + (1 - eps) * x_g
            disc_pn, _ = self.discriminator(x_pn, vars_disc)

            grad = tf.gradients(disc_pn, x_pn)[0]
            grad = tf.norm(grad, axis=1)
            ddx = scale_fn * tf.square(grad - 1)

            loss_d += tf.reduce_mean(ddx)

            loss_cat = tf.nn.softmax_cross_entropy_with_logits(labels=input_cat, logits=output_cat_g)
            loss_cat = tf.reduce_mean(loss_cat, name="loss_cat")

        # Train
        lr = 0.0002
        num_print = int(1e2)
        num_summary = 10
        num_batch = 256

        num_total_image = self.data.get_count()
        num_input_x = 8
        num_input_y = 8
        num_input = num_input_x * num_input_y

        num_step_d = 3
        num_step_g = 1

        opt_g = tf.train.AdamOptimizer(lr).minimize(loss_g, var_list=vars_gen)
        opt_d = tf.train.AdamOptimizer(lr).minimize(loss_d, var_list=vars_disc)
        opt_cat = tf.train.AdamOptimizer(lr).minimize(loss_cat)

        center = 0.0
        stddev = 1.0
        dropout_val = 0.7

        # Output
        output_dir = "cq_output_infogan"
        helper.clean_create_dir(output_dir)

        test_input_x = self.data.get_batch(num_input)

        prr = plot_utils.Plot_Reproduce_Performance(output_dir, num_input_x, num_input_y, image_width, image_height, scale)
        prr.save_pngs(test_input_x, num_channel, "input.png")

        output_cat_val = helper.constant_z(0, (num_input, num_latent_cat))

        if num_latent_cat != 0:
            for i in range(output_cat_val.shape[0]):
                output_cat_val[i] = 1

        # Debug
        with tf.name_scope("Debug"):
            disc_g_mean = tf.reduce_mean(disc_g, name="disc_g_mean")
            disc_d_mean = tf.reduce_mean(disc_d, name="disc_d_mean")

        # Summary
        with tf.name_scope("Summary"):
            tf.summary.scalar("disc_g", disc_g_mean)
            tf.summary.scalar("disc_d", disc_d_mean)
            tf.summary.scalar("loss_g", loss_g)
            tf.summary.scalar("loss_d", loss_d)

            if num_latent_cat != 0:
                tf.summary.scalar("loss_cat", loss_cat)

        merged = tf.summary.merge_all()

        with tf.Session() as ss:
            ss.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter("cq_log", ss.graph)

            i = 0
            print_i = 0

            while True:
                # Discriminator
                for j in range(num_step_d):
                    data_images = self.data.get_batch(num_batch)
                    real_z_val = helper.random_normal(center, stddev, (num_batch, num_z - num_latent_cat))
                    input_cat_val = helper.random_cat((num_batch, num_latent_cat))
                    feed_dict = {x: data_images, real_z: real_z_val, input_cat: input_cat_val, dropout: dropout_val}

                    # ss.run(clip)
                    disc_d_val, loss_d_val, _ = ss.run([disc_d_mean, loss_d, opt_d], feed_dict=feed_dict)

                # Generator
                for j in range(num_step_g):
                    real_z_val = helper.random_normal(center, stddev, (num_batch, num_z - num_latent_cat))
                    input_cat_val = helper.random_cat((num_batch, num_latent_cat))
                    feed_dict = {real_z: real_z_val, input_cat: input_cat_val, dropout: dropout_val}

                    disc_g_val, loss_g_val, _ = ss.run([disc_g_mean, loss_g, opt_g], feed_dict=feed_dict)

                    # Latent code
                    if num_latent_cat != 0:
                        loss_cat_val, _ = ss.run([loss_cat, opt_cat], feed_dict=feed_dict)

                # Outputs
                if i % num_summary == 0:
                    data_images = self.data.get_batch(num_batch)
                    feed_dict = {x: data_images, real_z: real_z_val, input_cat: input_cat_val, dropout: dropout_val}
                    summary = ss.run(merged, feed_dict=feed_dict)
                    writer.add_summary(summary, i)

                if i % num_print == 0 and i != 0:
                    i_div_k = i / 1000

                    debug_message = ""
                    debug_message += "%.1fk steps done - " % i_div_k
                    debug_message += "Disc G : %f, Disc D : %f, Loss G: %f, Loss D: %f" % (
                    disc_g_val, disc_d_val, loss_g_val, loss_d_val)

                    if num_latent_cat != 0:
                        debug_message += ", Loss cat: %f" % loss_cat_val

                    print(debug_message)

                    real_z_val = helper.random_normal(center, stddev, (num_input, num_z - num_latent_cat))
                    feed_dict = {real_z: real_z_val, input_cat: output_cat_val, dropout: dropout_val}

                    output_x = ss.run(x_g, feed_dict=feed_dict)
                    prr.save_pngs(output_x, num_channel, name="output%d.png" % print_i)

                    print_i += 1

                i += 1


gan = CqDCGAN()
gan.run(256, 4096, 0)
