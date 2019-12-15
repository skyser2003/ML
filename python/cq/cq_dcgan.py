import os
import datetime
from typing import Dict, List

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

# Env variables
num_iter = int(os.getenv("num_iter", "10000000000000000000"))
strategy_type = os.getenv("strategy_type", "mirror")
mode = os.getenv("mode", "train")  # train, predict


@tf.function
def train_step(strategy: tf.distribute.Strategy, data_it, disc: Model, gen: Model, model_g: Model, model_d: Model,
               batch_size: int, z_size: int, num_cat: int, metrics: Dict[str, keras.metrics.Mean]):
    # Discriminate
    def discriminate(batch_images: tf.Tensor):
        train_vars = disc.trainable_variables

        batch_size = batch_images.shape[0]

        eps = tf.random.uniform((batch_size, 1, 1, 1), 0, 1)
        z_input, cat_input = CqGAN.generate_z(batch_size, z_size, num_cat)

        with tf.GradientTape() as tape:
            disc_gen, disc_real, iwgan_loss, cat_output = model_d((z_input, batch_images, eps), training=True)
            full_loss = -disc_gen + disc_real + iwgan_loss + CqGAN.get_loss_cat(cat_input, cat_output)

        grads = tape.gradient(full_loss, train_vars)
        disc.optimizer.apply_gradients(zip(grads, train_vars))

        loss_real = full_loss

        metrics["disc_gen"].update_state(disc_gen)
        metrics["disc_real"].update_state(disc_real)
        metrics["loss_real"].update_state(loss_real)
        metrics["iwgan_loss"].update_state(iwgan_loss)

    def generate():
        train_vars = gen.trainable_variables

        z_input, cat_input = CqGAN.generate_z(batch_size, z_size, num_cat)

        with tf.GradientTape() as tape:
            disc_gen, cat_output = model_g(z_input, training=True)

            loss_gen = disc_gen
            loss_cat = CqGAN.get_loss_cat(cat_input, cat_output)

            full_loss = loss_gen + loss_cat

        grads = tape.gradient(full_loss, train_vars)
        gen.optimizer.apply_gradients(zip(grads, train_vars))

        metrics["loss_gen"].update_state(loss_gen)
        metrics["loss_cat"].update_state(loss_cat)

    for _ in range(3):
        batch_images = next(data_it)
        strategy.experimental_run_v2(discriminate, args=(batch_images,))

    strategy.experimental_run_v2(generate)


@tf.function
def predict_image(strategy: tf.distribute.Strategy, gen: Model, input_z: tf.Tensor):
    return strategy.experimental_run_v2(lambda: gen(input_z))


class CqGAN:
    def run(self, batch_size: int, output_dir: str):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print(f"Number of GPUS: {len(gpus)}")

        cq_dataset = CqData(CqDataType.FACE, scale_down=scale)
        dataset = tf.data.Dataset.from_tensor_slices(cq_dataset.images) \
            .shuffle(len(cq_dataset.images)) \
            .repeat() \
            .batch(batch_size, drop_remainder=True)

        output_num_x = 8
        output_num_y = 8

        num_output_image = output_num_x * output_num_y

        img_width = cq_dataset.get_image_width()
        img_height = cq_dataset.get_image_height()
        num_channel = cq_dataset.get_channel_count()

        lr = 0.0002
        z_size = 256
        num_cat = cq_dataset.get_count()
        # num_cat = 0
        output_interval = 100

        opt_g = keras.optimizers.Adam(lr)
        opt_d = keras.optimizers.Adam(lr)

        opt_g = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt_g)
        opt_d = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt_d)

        if strategy_type == "mirror":
            strategy = tf.distribute.MirroredStrategy()
        elif strategy_type == "tpu":
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=f"grpc://{os.environ['COLAB_TPU_ADDR']}")
            tf.config.experimental_connect_to_host(resolver.master())
            tf.tpu.experimental.initialize_tpu_system(resolver)

            strategy = tf.distribute.experimental.TPUStrategy(resolver)
        else:
            raise Exception(f"Wrong strategy type: {strategy_type}")

        with strategy.scope():
            gen = Generator(4, img_width, img_height, num_channel)
            disc = Discriminator(4, num_cat)

            gen.optimizer = opt_g
            disc.optimizer = opt_d

            iwgan = IWGanLoss(disc)

            input_g = Input(batch_size=batch_size, shape=z_size + num_cat, name="z")
            input_d = Input(batch_size=batch_size, shape=(img_height, img_width, num_channel), name="real_images")
            input_eps = Input(batch_size=batch_size, shape=(1, 1, 1), name="eps")

            disc_gen: tf.Tensor
            disc_real: tf.Tensor
            cat_output: tf.Tensor

            gen_images: tf.Tensor = gen(input_g)

            x_pn = input_eps * input_d + (1 - input_eps) * gen_images
            iwgan_loss = iwgan(x_pn)

        disc_gen, cat_output = disc(gen_images)
        disc_real, _ = disc(input_d)

        model_g = Model(inputs=[input_g], outputs=[disc_gen, cat_output])
        model_d = Model(inputs=[input_g, input_d, input_eps], outputs=[disc_gen, disc_real, iwgan_loss, cat_output])

        dataset = strategy.experimental_distribute_dataset(dataset)
        data_it = iter(dataset)

        # Model
        model_dir = os.path.join(output_dir, "model_save")
        gen_model_dir = os.path.join(model_dir, "gen")
        disc_model_dir = os.path.join(model_dir, "disc")

        gen_ckpt = tf.train.Checkpoint(model=gen, optimizer=gen.optimizer)
        disc_ckpt = tf.train.Checkpoint(model=disc, optimizer=disc.optimizer)

        gen_ckpt_mgr = tf.train.CheckpointManager(gen_ckpt, gen_model_dir, max_to_keep=5)
        disc_ckpt_mgr = tf.train.CheckpointManager(disc_ckpt, disc_model_dir, max_to_keep=5)

        gen_latest = gen_ckpt_mgr.latest_checkpoint
        disc_latest = disc_ckpt_mgr.latest_checkpoint

        if gen_latest:
            gen_ckpt.restore(gen_latest)
        if disc_latest:
            disc_ckpt.restore(disc_latest)

        if mode == "train":
            # Image output
            image_dir = os.path.join(output_dir, "images")

            helper.clean_create_dir(image_dir)
            prr = plot_utils.Plot_Reproduce_Performance(image_dir, output_num_x, output_num_y, img_width, img_height,
                                                        scale)

            test_input_images = cq_dataset.get_ordered_batch(num_output_image, False)
            prr.save_pngs(test_input_images, num_channel, "input.png")

            # Summary
            summary_dir = "cq_log"
            summary_writer = tf.summary.create_file_writer(summary_dir)

            metrics = {
                "disc_gen": keras.metrics.Mean(),
                "disc_real": keras.metrics.Mean(),
                "loss_gen": keras.metrics.Mean(),
                "loss_real": keras.metrics.Mean(),
                "loss_cat": keras.metrics.Mean(),
                "iwgan_loss": keras.metrics.Mean()
            }

            summary_writer.set_as_default()

            graph(K.get_graph(), step=0)

            # Test variables
            z_fixed, _, cat_fixed = self.generate_fixed_z(num_output_image, z_size, num_cat)

            # Begin training
            begin = datetime.datetime.now()

            for step in range(num_iter):
                train_step(strategy, data_it, disc, gen, model_g, model_d, batch_size, z_size, num_cat, metrics)

                # Output
                if step % output_interval == 0 and step != 0:
                    now = datetime.datetime.now()
                    diff = now - begin
                    begin = now

                    output_count = step // output_interval
                    output_filename = f"output{output_count}.png"
                    output_images = gen(z_fixed)
                    output_images = output_images.numpy()
                    prr.save_pngs(output_images, num_channel, output_filename)

                    gen_ckpt_mgr.save()
                    disc_ckpt_mgr.save()

                    print(f"{output_count * output_interval} times done: {diff.total_seconds()}s passed, "
                          f"loss_gen: {metrics['loss_gen'].result()}, "
                          f"loss_real: {metrics['loss_real'].result()}, "
                          f"loss_cat: {metrics['loss_cat'].result()}")

                # Summary
                for metric_name, metric in metrics.items():
                    tf.summary.scalar(f"loss/{metric_name}", metric.result(), step)
                    metric.reset_states()
        elif mode == "predict":
            # Image output
            image_dir = os.path.join(output_dir, "predict_images")

            helper.clean_create_dir(image_dir)
            prr = plot_utils.Plot_Reproduce_Performance(image_dir, output_num_x, output_num_y, img_width, img_height,
                                                        scale)

            with strategy.scope():
                fixed_z, real_z, _ = self.generate_nocat_z(batch_size, z_size, num_cat)
                input_images = predict_image(strategy, gen, fixed_z).numpy()

                prr.save_pngs(input_images, num_channel, "input.png")

                for i in range(num_cat):
                    cat_input = self.generate_fixed_cat(batch_size, num_cat, [i])
                    input_z = tf.concat([real_z, cat_input], 1)

                    gen_images = predict_image(strategy, gen, input_z)
                    gen_images = gen_images.numpy()

                    output_filename = f"output{i + 1}.png"
                    prr.save_pngs(gen_images, num_channel, output_filename)

    @staticmethod
    def get_loss_cat(cat_input: tf.Tensor, cat_output: tf.Tensor):
        if cat_output.shape[1] == 0:
            return 0.0

        loss_cat = tf.nn.softmax_cross_entropy_with_logits(cat_input, cat_output, 1)
        return tf.reduce_mean(loss_cat)

    @staticmethod
    def generate_z(batch_size: int, z_size: int, num_cat: int, force_assign: List[int] = None):
        real_z = CqGAN.generate_real_z(batch_size, z_size)

        if num_cat == 0:
            cat_input = tf.zeros([batch_size, num_cat])
            return real_z, cat_input
        else:
            cat_input_np = np.zeros((batch_size, num_cat), dtype=np.float32)
            for i in range(batch_size):
                rand_index = np.random.randint(num_cat)
                cat_input_np[i][rand_index] = 1

            if force_assign is not None:
                CqGAN.force_assign_cat(cat_input_np, force_assign)

            cat_input = tf.convert_to_tensor(cat_input_np)

            return tf.concat([real_z, cat_input], 1), real_z, cat_input

    @staticmethod
    def generate_fixed_z(batch_size: int, z_size: int, num_cat: int):
        real_z = CqGAN.generate_real_z(batch_size, z_size)

        if num_cat == 0:
            cat_input = CqGAN.generate_fixed_cat(batch_size, num_cat)
            return real_z, cat_input
        else:
            cat_input = tf.eye(batch_size, num_cat)
            return tf.concat([real_z, cat_input], 1), real_z, cat_input

    @staticmethod
    def generate_nocat_z(batch_size: int, z_size: int, num_cat: int):
        real_z = CqGAN.generate_real_z(batch_size, z_size)
        cat_input = tf.zeros([batch_size, num_cat])

        if num_cat == 0:
            return real_z, cat_input
        else:
            return tf.concat([real_z, cat_input], 1), real_z, cat_input

    @staticmethod
    def generate_real_z(batch_size: int, z_size: int):
        return tf.random.normal([batch_size, z_size])

    @staticmethod
    def generate_fixed_cat(batch_size: int, num_cat: int, force_assign: List[int] = None):
        cat_input_np = np.eye(batch_size, num_cat, dtype=np.float32)

        if force_assign is not None:
            CqGAN.force_assign_cat(cat_input_np, force_assign)

        return tf.convert_to_tensor(cat_input_np)

    @staticmethod
    def force_assign_cat(cat_input_np: np.ndarray, force_assign: List[int]):
        for force_index in force_assign:
            for row in cat_input_np:
                row[force_index] = 1

    @staticmethod
    def loss_gan(y_label, y_pred):
        return tf.reduce_mean(y_label * y_pred)


class Generator(Model):
    def __init__(self, num_layer: int, width: int, height: int, num_channel: int):
        super().__init__()

        self.data_format = "channels_last"
        self.num_layer = num_layer
        self.width = width
        self.height = height
        self.num_channel = num_channel

    def build(self, input_shape):
        num_channel = self.num_channel
        num_initial_filter = num_channel * 32
        kernel_size = 5
        stride_size = 2

        init_multiplier = 0.5
        init_width = int(self.width * init_multiplier)
        init_height = int(self.height * init_multiplier)
        out_padding = 1

        kernel_init = keras.initializers.RandomNormal(stddev=0.02)
        bias_init = keras.initializers.zeros()

        self.dense = Dense(init_width * init_height * num_initial_filter, input_shape=(input_shape[1],),
                           kernel_initializer=kernel_init, bias_initializer=bias_init, activation=tf.nn.leaky_relu)

        self.reshape = Reshape([init_height, init_width, num_initial_filter])

        self.conv2d_tps = Sequential()

        for i in range(self.num_layer - 1):
            self.conv2d_tps.add(
                Conv2DTranspose(num_initial_filter // (2 ** (i + 1)), kernel_size, stride_size, padding="same",
                                output_padding=out_padding, activation=tf.nn.leaky_relu,
                                kernel_initializer=kernel_init, bias_initializer=bias_init))

        self.conv2d_tps.add(
            layers.Conv2DTranspose(num_channel, kernel_size, stride_size, padding="same", output_padding=out_padding,
                                   kernel_initializer=kernel_init, bias_initializer=bias_init))

        pool_size = int(init_multiplier * stride_size ** self.num_layer)
        self.avgpool2d = AvgPool2D(pool_size)

        self.sigmoid = Activation(tf.nn.sigmoid)

    def call(self, inputs: keras.layers.Input, **kwargs):
        output = inputs

        output = self.dense(output)
        output = self.reshape(output)
        output = self.conv2d_tps(output)
        output = self.avgpool2d(output)
        output = self.sigmoid(output)

        return output


class Discriminator(Model):
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
            conv = Conv2D(next_channel, kernel_size, stride_size, padding="same",
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

        with tf.GradientTape() as tape:
            tape.watch(x_pn)
            disc_pn, _ = self.disc(x_pn)

        grad = tape.gradient(disc_pn, x_pn)
        grad = self.flatten(grad)

        grad = tf.norm(grad, axis=1)

        scale_fn = 10
        ddx = scale_fn * (grad - 1) ** 2
        ddx = tf.expand_dims(ddx, 1)

        return ddx


if __name__ == "__main__":
    gan = CqGAN()
    gan.run(64, "cq_output_dcgan")
