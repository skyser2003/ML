import os
import glob

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import util.plot_utils as plot_utils
import util.helper as helper

class VAE:
    def __init__(self):
        with tf.variable_scope("Constant"):
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def get_encoder(self, num_pixels, num_hidden, num_output):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0)

        with tf.variable_scope("Encoder_Placeholder"):
            self.x = tf.placeholder(tf.float32, [None, num_pixels], name="x")

        with tf.variable_scope("Encoder_Variable"):
            w1 = tf.get_variable("w1", shape=[num_pixels, num_hidden], initializer=w_init)
            b1 = tf.get_variable("b1", shape=[num_hidden], initializer=b_init)

            w2 = tf.get_variable("w2", shape=[num_hidden, num_output * 2], initializer=w_init)
            b2 = tf.get_variable("b2", shape=[num_output * 2], initializer=b_init)

        with tf.variable_scope("Encoder_Operation"):
            t = tf.nn.dropout(tf.nn.relu(self.x @ w1 + b1), self.keep_prob, name="t")
            y_ = tf.identity(t @ w2 + b2, name="y_")

            mean = y_[:, :num_output]
            stddev = 1e-6 + tf.nn.softplus(y_[:, num_output:])

        self.y_ = y_

        return mean, stddev

    def get_decoder(self, num_pixels, num_hidden, mean, stddev):
        x = self.x

        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0)

        with tf.variable_scope("Decoder_Variable"):
            y_ = tf.identity(mean + stddev * tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32), name="z")

            w1 = tf.get_variable("w1", shape=[y_.get_shape()[1], num_hidden], initializer=w_init)
            b1 = tf.get_variable("b1", shape=[num_hidden], initializer=b_init)

            w2 = tf.get_variable("w2", shape=[num_hidden, num_pixels], initializer=w_init)
            b2 = tf.get_variable("b2", shape=[num_pixels], initializer=b_init)
        
        with tf.variable_scope("Decoder_Operation"):
            t = tf.nn.dropout(tf.nn.elu(y_ @ w1 + b1), self.keep_prob, name="t")

            x_ = tf.sigmoid(t @ w2 + b2)
            x_ = tf.clip_by_value(x_, 1e-8, 1 - 1e-8, name="x_")

            marginal_likelihood = tf.reduce_sum(x * tf.log(x_) + (1 - x) * tf.log(1 - x_), 1)
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(mean) + tf.square(stddev) - tf.log(1e-8 + tf.square(stddev)) - 1, 1)

            marginal_likelihood = tf.reduce_mean(marginal_likelihood, name="marginal_likelihood")
            KL_divergence = tf.reduce_mean(KL_divergence, name="kl_div")

            loss = tf.identity(KL_divergence - marginal_likelihood, name="loss")
            # loss = tf.reduce_sum(tf.losses.mean_squared_error(x, x_), name="loss")

            opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

        with tf.name_scope("Decoder_Summary"):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("marginal_likelihood", marginal_likelihood)
            tf.summary.scalar("kl_div", KL_divergence)
            tf.summary.scalar("x_", tf.reduce_mean(x_))
            tf.summary.scalar("z", tf.reduce_mean(y_))
            tf.summary.scalar("mean", tf.reduce_mean(mean))
            tf.summary.scalar("stddev", tf.reduce_mean(stddev))

        self.x_ = x_

        return loss, opt, KL_divergence

    def run(self, mnist, num_pixels, num_hidden, num_hidden2):
        images = mnist.train.images

        mean, stddev = self.get_encoder(num_pixels, num_hidden, num_hidden2)
        loss, opt2, kl_div = self.get_decoder(num_pixels, num_hidden, mean, stddev)

        num_iter = 1000

        merged = tf.summary.merge_all()

        config = tf.ConfigProto(device_count=
            {
            # "GPU":0
            })

        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            writer = tf.summary.FileWriter("log", sess.graph)

            # Output folder
            output_dir = "output"
            helper.clean_create_dir(output_dir)

            prr = plot_utils.Plot_Reproduce_Performance(output_dir)
            input_x, input_y = mnist.test.next_batch(prr.n_tot_imgs)

            prr.save_images(input_x, "input.jpg")


            prev_percentage = 0
            count = 0

            num_epochs = 20

            for epoch in range(num_epochs):
                percentage = int(epoch / num_epochs * 100)

                if(int(percentage / 10) != int(prev_percentage / 10)):
                    print(percentage, "% done")

                prev_percentage = percentage

                for i in range(num_iter):
                    batch_x, batch_y = mnist.train.next_batch(100)
                    # batch_x, batch_y = images, labels
                    feed_dict = { self.x: batch_x, self.keep_prob: 0.9 }

                    _, summary = sess.run([opt2, merged], feed_dict=feed_dict)

                    writer.add_summary(summary, count)
                    count += 1

                x_img = sess.run(self.x_, feed_dict={self.x: input_x, self.keep_prob: 1})
                prr.save_images(x_img, "result%02d.jpg" % epoch)

            print("100% done")

def main():
   mnist = input_data.read_data_sets("MNIST_data")
   num_pixels = mnist.train.images.shape[1]
   num_hidden = 100
   num_hidden2 = 100

   vae = VAE()
   vae.run(mnist, num_pixels, num_hidden, num_hidden2)

if __name__ == '__main__':
   main()