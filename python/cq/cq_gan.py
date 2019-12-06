import tensorflow as tf


class BaseGanComponent:
    def get_input_placeholders(self) -> list:
        return None

    def get_generator_input_data(self):
        return None

    def get_discriminator_input_data(self):
        return None

    def generate(self, input: tf.Tensor) -> tf.Tensor:
        return None

    def discriminate(self, input: tf.Tensor, output_size: int):
        return None

    def get_optimizer(self):
        return [None, None]

    def get_vars(self):
        return [None, None]

    def train(self, x_g: tf.Tensor, output_fake, output_real, vars_gen: list, vars_disc: list):
        return None

    def get_output_size(self) -> int:
        return None


class CqGAN:
    def __init__(self, comp: BaseGanComponent):
        self.comp: BaseGanComponent = None

        self.init(comp)

    def init(self, comp: BaseGanComponent):
        self.comp = comp

    def run(self):
        z, x = self.comp.get_input_placeholders()
        x_g = self.comp.generate(z)

        output_size = self.comp.get_output_size()
        output_fake = self.comp.discriminate(x_g, output_size)
        output_real = self.comp.discriminate(x, output_size)

        vars_gen, vars_disc = self.comp.get_vars()

        self.comp.train(x_g, output_fake, output_real, vars_gen, vars_disc)

