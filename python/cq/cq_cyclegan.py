import datetime
import random
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers, Sequential, Input
from tensorflow.python.keras.layers import Dense, Conv2DTranspose, Conv2D, Flatten, Reshape, AvgPool2D, Activation, \
    Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.summary_ops_v2 import graph

import numpy as np

import util.helper as helper
import util.plot_utils as plot_utils
from cq.cq_data import CqData, CqDataType, CqDataMode

# For FACE images
scale = 3


def _parse_images(filename: str):
    image_str = tf.io.read_file(filename)
    image_decoded = tf.image.decode_image(image_str)

    return image_decoded


class CqGAN:
    def __init__(self):
        self.cq_dataset_a = CqData(CqDataType.FACE, scale_down=scale)
        self.cq_dataset_b = CqData(CqDataType.WHOLE, scale_down=scale)

    def run(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        output_num_x = 8
        output_num_y = 8

        num_output_image = output_num_x * output_num_y

        a_width = self.cq_dataset_a.get_image_width()
        a_height = self.cq_dataset_a.get_image_height()
        a_channel = self.cq_dataset_a.get_channel_count()

        b_width = self.cq_dataset_b.get_image_width()
        b_height = self.cq_dataset_b.get_image_height()
        b_channel = self.cq_dataset_b.get_channel_count()

        a_size = (a_height, a_width, a_channel)
        b_size = (b_height, b_width, b_channel)

        output_dir = "cq_output_dcgan"
        helper.clean_create_dir(output_dir)
        prr = plot_utils.Plot_Reproduce_Performance(output_dir, output_num_x, output_num_y, a_width, a_height,
                                                    scale)

        test_input_images = self.cq_dataset_a.get_ordered_batch(num_output_image, False)
        prr.save_pngs(test_input_images, a_channel, "input.png")

        batch_size = 64
        num_iter = 10000000000000000000
        lr = 0.0002
        z_size = 256
        # num_cat = output_num_x * output_num_y
        num_cat = 0
        output_interval = 100

        opt_g = keras.optimizers.Adam(lr)
        opt_d = keras.optimizers.Adam(lr)

        gen_a = Generator(4, a_width, a_height, a_channel)
        disc_a = Discriminator(4, num_cat)
        gen_b = Generator(4, b_width, b_height, a_channel)
        disc_b = Discriminator(4, num_cat)

        adv_a = CycleLoss(disc_b)
        adv_b = CycleLoss(disc_a)
        consistency = CycleConsistencyLoss(gen_a, gen_b)

        lambd = 10

        real_images_a = Input(shape=a_size, name="input_b")
        real_images_b = Input(shape=b_size, name="input_a")

        gen_images_a = gen_a(real_images_b)
        gen_images_b = gen_b(real_images_a)

        adv_loss_a = adv_a(real_images_a, gen_images_a)
        adv_loss_b = adv_b(real_images_b, gen_images_b)

        consistency_loss = lambd * consistency(real_images_a, real_images_b, gen_images_a, gen_images_b)

        model = Model(inputs=[real_images_a, real_images_b], outputs=[adv_loss_a, adv_loss_b, consistency_loss])

        pos_y = np.ones((batch_size, 1), dtype=np.float32)
        neg_y = -pos_y

        gen.trainable = True
        disc.trainable = False
        model.compile(opt_g, loss=[self.loss_gan, self.loss_gan, self.loss_gan],
                      target_tensors={"cycle_consistency_loss": pos_y})

        # Summary
        summary_dir = "cq_log"
        summary_writer = tf.summary.create_file_writer(summary_dir)
        disc_gen_summary = keras.metrics.Mean()
        disc_real_summary = keras.metrics.Mean()
        loss_gen_summary = keras.metrics.Mean()
        loss_real_summary = keras.metrics.Mean()
        loss_cat_summary = keras.metrics.Mean()

        summary_writer.set_as_default()

        graph(K.get_graph(), step=0)

        # Test variables
        z_fixed, cat_fixed = self.__generate_fixed_z(num_output_image, self.cq_dataset_b, num_cat)

        # Begin training
        begin = datetime.datetime.now()

        for step in range(num_iter):
            loss_gen, loss_real, loss_cat = self.train_step(model_g, model_d, batch_size, a_size, b_size, z_size,
                                                            num_cat)

            # Summary
            # disc_gen_summary.update_state(disc_gen)
            # disc_real_summary.update_state(disc_real)
            loss_gen_summary.update_state(loss_gen)
            loss_real_summary.update_state(loss_real)
            loss_cat_summary.update_state(loss_cat)

            # Output
            if step % output_interval == 0 and step != 0:
                now = datetime.datetime.now()
                diff = now - begin
                begin = now

                output_count = step // output_interval
                output_filename = f"output{output_count}.png"
                output_images = gen(z_fixed)
                output_images = output_images.numpy()
                prr.save_pngs(output_images, a_channel, output_filename)

                print(f"{output_count * output_interval} times done: {diff.total_seconds()}s passed")

            # Summary
            tf.summary.scalar("loss/disc_gen", disc_gen_summary.result(), step)
            tf.summary.scalar("loss/disc_real", disc_real_summary.result(), step)
            tf.summary.scalar("loss/loss_gen", loss_gen_summary.result(), step)
            tf.summary.scalar("loss/loss_real", loss_real_summary.result(), step)
            if num_cat != 0:
                tf.summary.scalar("loss/loss_cat", loss_cat_summary.result(), step)

            disc_gen_summary.reset_states()
            disc_real_summary.reset_states()
            loss_gen_summary.reset_states()
            loss_real_summary.reset_states()
            loss_cat_summary.reset_states()

    def train_step(self, model_g: Model, model_d: Model, batch_size: int, a_size: Tuple[int, int, int],
                   b_size: Tuple[int, int, int], z_size: int, num_cat: int):
        # Discriminate
        for _ in range(3):
            batch_images = self.cq_dataset_a.get_batch(batch_size, False)
            eps = tf.random.uniform((batch_size, 1, 1, 1), 0, 1)

            z_input, cat_input = self.__generate_z(batch_size, self.cq_dataset_b, num_cat)
            losses_d = model_d.train_on_batch([z_input, batch_images, eps], [cat_input])

            loss_real = losses_d[0]

        # Generate
        z_input, cat_input = self.__generate_z(batch_size, self.cq_dataset_b, num_cat)
        losses_g = model_g.train_on_batch([z_input], [cat_input])

        loss_gen = losses_g[0]
        loss_cat = losses_g[2]

        return loss_gen, loss_real, loss_cat

    def __get_loss_cat(self, cat_input: tf.Tensor, cat_output: tf.Tensor):
        if cat_output.shape[1] == 0:
            return 0.0

        loss_cat = keras.backend.categorical_crossentropy(cat_input, cat_output, True, 1)
        return tf.reduce_mean(loss_cat)

    def __generate_z(self, batch_size: int, dataset: CqData, num_cat: int):
        real_z = dataset.get_batch(batch_size, False)
        cat_input = np.zeros([batch_size, num_cat])

        if num_cat == 0:
            return real_z, cat_input
        else:
            for cat in cat_input:
                rand_index = random.randint(0, num_cat - 1)
                cat[rand_index] = 1

            cat_input = tf.convert_to_tensor(cat_input, dtype=tf.float32)
            return tf.concat([real_z, cat_input], 1), cat_input

    def __generate_fixed_z(self, batch_size, dataset: CqData, num_cat: int):
        real_z = dataset.get_batch(batch_size, False)
        cat_input = np.zeros([batch_size, num_cat])

        if num_cat == 0:
            return real_z, cat_input
        else:
            np.fill_diagonal(cat_input, 1)

            return tf.concat([real_z, cat_input], 1), cat_input

    def loss_gan(self, y_label, y_pred):
        return keras.backend.mean(y_label * y_pred)


class Generator(Layer):
    def __init__(self, num_layer: int, width: int, height: int, num_channel: int):
        super().__init__()

        self.data_format = "channels_last"
        self.num_layer = num_layer
        self.width = width
        self.height = height
        self.num_channel = num_channel

        self.init_multiplier = 0.5

    def build(self, input_shape):
        num_channel = self.num_channel
        num_initial_filter = num_channel * 32
        kernel_size = 5
        stride_size = 2
        out_padding = 1

        kernel_init = keras.initializers.RandomNormal(stddev=0.02)
        bias_init = keras.initializers.zeros()

        self.conv2d_tps = Sequential()

        for i in range(self.num_layer - 1):
            self.conv2d_tps.add(
                Conv2DTranspose(num_initial_filter // (2 ** (i + 1)), kernel_size, stride_size, padding="same",
                                output_padding=out_padding, activation=tf.nn.leaky_relu,
                                kernel_initializer=kernel_init, bias_initializer=bias_init))

        self.conv2d_tps.add(
            layers.Conv2DTranspose(num_channel, kernel_size, stride_size, padding="same", output_padding=out_padding,
                                   kernel_initializer=kernel_init, bias_initializer=bias_init))

        pool_size = int(self.init_multiplier * stride_size ** self.num_layer)
        self.avgpool2d = AvgPool2D(pool_size)

        self.sigmoid = Activation(tf.nn.sigmoid)

    def call(self, inputs: keras.layers.Input, **kwargs):
        init_height = int(self.init_multiplier * self.height)
        init_width = int(self.init_multiplier * self.width)

        output = inputs
        output = tf.image.resize(output, (init_height, init_width))

        output = self.conv2d_tps(output)
        output = self.avgpool2d(output)
        output = self.sigmoid(output)

        return output


class Discriminator(Layer):
    def __init__(self, num_layer: int, num_cat: int):
        super().__init__()

        self.num_layer = num_layer
        self.num_cat = num_cat

    def build(self, input_shape):
        kernel_size = 5
        stride_size = 2

        width = input_shape[2]
        height = input_shape[1]
        num_channel = input_shape[3]
        next_channel = num_channel * 32

        kernel_init = keras.initializers.RandomNormal(stddev=0.02)
        bias_init = keras.initializers.zeros()

        self.conv2ds = Sequential()

        for i in range(self.num_layer - 1):
            conv = layers.Conv2D(next_channel, kernel_size, stride_size, padding="same",
                                 activation=tf.nn.leaky_relu, kernel_initializer=kernel_init,
                                 bias_initializer=bias_init, input_shape=(height, width, num_channel))
            self.conv2ds.add(conv)

            width = conv.output_shape[2]
            height = conv.output_shape[1]
            num_channel = next_channel
            next_channel *= stride_size

        conv = Conv2D(1 + self.num_cat, (height, width), (height, width), padding="valid",
                      kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.conv2ds.add(conv)

        self.flatten = Flatten()

    def call(self, inputs: keras.layers.Input, **kwargs):
        output = inputs

        output = self.conv2ds(output)
        output = self.flatten(output)

        classify_result: tf.Tensor = output[:, :1]
        cat_result: tf.Tensor = output[:, 1:]

        return classify_result, cat_result


class IWGanLoss(Layer):
    def __init__(self, disc: Discriminator):
        super().__init__()

        self.disc = disc

    def build(self, input_shape):
        super().build(input_shape)

        self.flatten = Flatten()

    def call(self, inputs: keras.layers.Input, **kwargs):
        x_pn = inputs
        disc_pn, _ = self.disc(x_pn)

        grad = keras.backend.gradients(disc_pn, x_pn)[0]
        grad = self.flatten(grad)

        grad = tf.norm(grad, axis=1)

        scale_fn = 10
        ddx = scale_fn * (grad - 1) ** 2
        ddx = tf.reduce_mean(ddx)

        return ddx


class CycleLoss(Layer):
    def __init__(self, disc: Discriminator):
        super().__init__()

        self.disc = disc

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs: List, **kwargs):
        real_images = inputs[0]
        gen_images = inputs[1]

        real_val = tf.reduce_mean(tf.math.log(self.disc(real_images)))
        gen_val = tf.reduce_mean(tf.math.log(1 - self.disc(gen_images)))

        return real_val + gen_val


class CycleConsistencyLoss(Layer):
    def __init__(self, gen_a: Generator, gen_b: Generator):
        super().__init__()

        self.gen_a = gen_a
        self.gen_b = gen_b

    def call(self, inputs: List, **kwargs):
        real_images_a = inputs[0]
        real_images_b = inputs[1]
        gen_images_a = inputs[2]
        gen_images_b = inputs[3]

        diff_a = tf.norm(gen_images_a - real_images_a, ord=1)
        diff_b = tf.norm(gen_images_b - real_images_b, ord=1)

        return keras.backend.mean(diff_a) + keras.backend.mean(diff_b)


gan = CqGAN()
gan.run()
