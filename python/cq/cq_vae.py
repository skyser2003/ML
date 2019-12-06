import tensorflow as tf

import cq.cq_data as cq

import util.plot_utils as plot_utils
import util.helper as helper

scale=3

class VAE:
    def __init__(self, num_hidden, num_hidden2):
        self.num_hidden = num_hidden
        self.num_hidden2 = num_hidden2
        self.data = cq.CqData(cq.CqDataType.FACE, mode=cq.CqDataMode.BG_BLACK, scale_down=scale)
        
    def encoder(self, input, prob):
        num_hidden = self.num_hidden
        num_hidden2 = self.num_hidden2

        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0)

        with tf.variable_scope("Encoder_Variable"):
            w1 = tf.get_variable("w1", shape=[input.get_shape()[1], num_hidden], initializer=w_init)
            b1 = tf.get_variable("b1", shape=[num_hidden], initializer=b_init)

            w2 = tf.get_variable("w2", shape=[num_hidden, num_hidden2 * 2], initializer=w_init)
            b2 = tf.get_variable("b2", shape=[num_hidden2 * 2], initializer=b_init)

        with tf.variable_scope("Encoder_Operation"):
            t1 = tf.nn.dropout(tf.nn.relu(input @ w1 + b1), prob, name="t1")
            t2 = tf.identity(t1 @ w2 + b2, name="t2")

            mean = tf.identity(t2[:, :num_hidden2], name="mean")
            stddev = tf.identity(1e-6 + tf.nn.softplus(t2[:, num_hidden2:]), name="stddev")

        return mean, stddev

    def decoder(self, x, mean, stddev, prob, image_size):
        num_hidden = self.num_hidden

        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0)

        const_val = 1

        with tf.variable_scope("Decoder_Variable"):
            z = tf.identity(mean + stddev * tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32), name="z")

            w1 = tf.get_variable("w1", shape=[z.get_shape()[1], num_hidden], initializer=w_init)
            b1 = tf.get_variable("b1", shape=[num_hidden], initializer=b_init)

            w2 = tf.get_variable("w2", shape=[num_hidden, image_size], initializer=w_init)
            b2 = tf.get_variable("b2", shape=[image_size], initializer=b_init)

        with tf.variable_scope("Decoder_Operation"):
            t = tf.nn.dropout(tf.nn.elu(z @ w1 + b1), prob, name="t")

            x_ = tf.sigmoid(t @ w2 + b2)
            x_ = tf.clip_by_value(x_, 1e-5, const_val - 1e-5, name="x_")

            x = tf.clip_by_value(x, 1e-5, const_val - 1e-5, name="x")

            marginal_likelihood = tf.reduce_sum(x * tf.log(x_) + (const_val - x) * tf.log(const_val - x_), 1)
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(mean) + tf.square(stddev) - tf.log(1e-8 + tf.square(stddev)) - 1, 1)

            marginal_likelihood = tf.reduce_mean(marginal_likelihood, name="marginal_likelihood")
            KL_divergence = tf.reduce_mean(KL_divergence, name="kl_div")

            loss = tf.identity(KL_divergence - marginal_likelihood, name="loss")

        with tf.name_scope("Decoder_Summary"):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("marginal_likelihood", marginal_likelihood)
            tf.summary.scalar("kl_div", KL_divergence)
            tf.summary.scalar("x_", tf.reduce_mean(x_))
            tf.summary.scalar("z", tf.reduce_mean(z))
            tf.summary.scalar("mean", tf.reduce_mean(mean))
            tf.summary.scalar("stddev", tf.reduce_mean(stddev))

        return loss, x_, marginal_likelihood, KL_divergence

    def run(self):
        image_width = self.data.get_image_width()
        image_height = self.data.get_image_height()
        num_channel = self.data.get_channel_count()
        image_size = image_width * image_height * num_channel

        x = tf.placeholder(tf.float32, shape=[None, image_size], name="x")
        prob = tf.placeholder(tf.float32, name="prob")

        mean, stddev = self.encoder(x, prob)
        loss, x_, marginal, kl_div = self.decoder(x, mean, stddev, prob, image_size)

        opt = tf.train.AdamOptimizer(0.001).minimize(loss)

        num_batch_x = 10
        num_batch_y = 10
        num_batch = num_batch_x * num_batch_y

        merged = tf.summary.merge_all()

        config = tf.ConfigProto(device_count=
            {
            # "GPU":0
            })

        with tf.Session(config=config) as ss:
            ss.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter("cq_log", ss.graph)

            num_step = int(1e4)
            num_test_batch = 100
            num_print_percentage = 1

            prev_percentage = 0
            cur_percentage = 0

            # Output folder
            output_dir = "cq_output_vae"
            helper.clean_create_dir(output_dir)

            test_input_x = self.data.get_batch(num_batch)

            prr = plot_utils.Plot_Reproduce_Performance(output_dir, num_batch_x, num_batch_y, image_width, image_height, scale)
            prr.save_pngs(test_input_x, num_channel, "input.png")

            for i in range(num_step):
                feed_dict={x: test_input_x, prob: 0.9}

                summary, mean_val, stddev_val, loss_val, x_val, _ = ss.run([merged, mean, stddev, loss, x_, opt], feed_dict=feed_dict)
                writer.add_summary(summary, i)

                cur_percentage = helper.percentage(i, num_step)
                cur_print_percentage = int(cur_percentage / num_print_percentage)

                if(cur_print_percentage != int(prev_percentage / num_print_percentage)):
                    feed_dict={x: test_input_x, prob: 1.0}
                    output_x = ss.run(x_, feed_dict=feed_dict)
                    prr.save_pngs(output_x, num_channel, "output%d.png" % cur_print_percentage)

                    print("%d%% done - Loss : %f" % (cur_print_percentage, loss_val))

                prev_percentage = cur_percentage


            # Final output
            feed_dict={x: test_input_x, prob: 1.0}
            loss_val, output_x = ss.run([loss, x_], feed_dict=feed_dict)

            prr.save_pngs(output_x, num_channel, "final_output.png")


num_hidden = 256
num_hidden2 = 128

vae = VAE(num_hidden, num_hidden2)
vae.run()