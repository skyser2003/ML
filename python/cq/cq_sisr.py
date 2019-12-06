import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import tensorflow.nn as nn

from python.cq.cq_data import CqData, CqDataMode, CqDataType
import python.util.plot as plot_utils


class CqSisr:
    def __init__(self):
        self.ss = None

        self.init()

    def init(self):
        self.ss = tf.Session()

    def enhancement(self):
        pass

    def compression(self, input):
        fcl = tcl.fully_connected(input, activation_fn=nn.leaky_relu)
        # tcl.conv2d(fcl, )

    def run(self):
        pass


if __name__ == "__main__":
    sisr = CqSisr()
    sisr.run()
