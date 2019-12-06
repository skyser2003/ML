import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import datetime
import math
from typing import Dict, List, Callable

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tcl
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.mongoexp import MongoTrials

EndConditionCallback = Callable[[int, int, float], bool]


class ConfigManager:
    @staticmethod
    def parse_config(config):
        space = dict({
            "params": {},
            "types": {}
        })

        h_params = config["h_params"]

        for param_name, param_config in h_params.items():
            param_values = param_config["parameters"]
            param_type = param_config["type"]
            distribution = param_config["distribution"]

            dist_params = [param_name]
            dist_params.extend(param_values)

            param_elem = getattr(hp, distribution)(*dist_params)
            space["params"][param_name] = param_elem
            space["types"][param_name] = param_type

        return space

    @staticmethod
    def parse_end_condition(config):
        conditions: List[EndConditionCallback] = []

        config_end_condition = config["end_condition"]

        # Time
        run_time: int = config_end_condition.get("time")
        if run_time is not None:
            conditions.append(lambda time, step, threshold: run_time <= time)

        # Step
        step_count: int = config_end_condition.get("step")
        if step_count is not None:
            conditions.append(lambda time, step, threshold: step_count == step)

        # Threshold
        threshold_min: float = config_end_condition.get("threshold")
        if threshold_min is not None:
            conditions.append(lambda time, step, threshold: threshold_min <= threshold)

        return conditions

    @staticmethod
    def get_param_value(args_dict, param_name: str):
        params = args_dict["params"]
        types = args_dict["types"]

        return types[param_name](params[param_name])


class TestTrain:
    @classmethod
    def train(cls, lr: float, x_train: tf.Tensor, y_train: tf.Tensor, relu_alpha: tf.Tensor, num_hidden: int):
        x_train = tf.layers.Flatten()(x_train)
        y_train = tf.one_hot(y_train, 10, dtype=tf.float64)

        init_w = tf.random_normal_initializer(stddev=0.02)
        init_b = tf.constant_initializer(0)

        out = x_train
        out = tcl.fully_connected(out, int(x_train.shape[1]) // 2, activation_fn=tf.identity,
                                  weights_initializer=init_w, biases_initializer=init_b)
        out = tf.nn.leaky_relu(out, alpha=relu_alpha)

        out = tcl.fully_connected(out, num_hidden, activation_fn=tf.nn.leaky_relu,
                                  weights_initializer=init_w, biases_initializer=init_b)

        out = tcl.fully_connected(out, 10, activation_fn=tf.identity, weights_initializer=init_w,
                                  biases_initializer=init_b)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=out)
        loss = tf.reduce_mean(loss)

        acc = tf.equal(tf.argmax(out, 1), tf.argmax(y_train, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float64))

        opt = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        return opt, out, loss, acc

    @classmethod
    def run(cls, lr: float, num_iter: int, relu_alpha: float, num_hidden: int):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        num_train = x_train.shape[0]
        num_batch = 2 ** 10

        tf.reset_default_graph()

        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True

        with tf.Session(config=cfg) as ss:
            x_plh = tf.placeholder(tf.float64, (None, 28, 28), "x_plh")
            y_plh = tf.placeholder(tf.int64, None, "y_plh")

            opt, out, loss, acc = cls.train(lr, x_plh, y_plh, relu_alpha, num_hidden)

            ss.run(tf.global_variables_initializer())

            for _ in range(num_iter):
                samples_indices = np.random.choice(num_train, num_batch)

                batch_x_train = x_train[samples_indices]
                batch_y_train = y_train[samples_indices]

                ss.run(opt, feed_dict={x_plh: batch_x_train, y_plh: batch_y_train})

            out_val, acc_val = ss.run((out, acc), feed_dict={x_plh: x_test, y_plh: y_test})

        return acc_val


class Methods:
    @staticmethod
    def train_method(args_dict):
        lr: float = ConfigManager.get_param_value(args_dict, "lr")
        num_hidden: int = ConfigManager.get_param_value(args_dict, "num_hidden")

        return TestTrain.run(lr, 1000, 0.2, num_hidden)

    @staticmethod
    def eval_method(args_dict, accuracy: float):
        lr: float = ConfigManager.get_param_value(args_dict, "lr")
        num_hidden: int = ConfigManager.get_param_value(args_dict, "num_hidden")

        print(f"accuracy: {round(accuracy * 100, 2)}%, num_hidden: {num_hidden}, lr: {lr}")

        return 1 - accuracy


class Obj:
    def __init__(self, end_condition_methods):
        self.end_condition_methods = end_condition_methods

        self.time = 0
        self.step = 0
        self.now = datetime.datetime.now()

        self.status = STATUS_OK

    def objective(self, args_dict):
        ret_struct = {
            "loss": None,
            "status": self.status,
        }

        end_time = datetime.datetime.now()

        elapsed_sec = (end_time - self.now).total_seconds()
        self.now = end_time

        self.time += elapsed_sec
        self.step += 1

        if self.status == STATUS_OK:
            train_result = Methods.train_method(args_dict)
            ret_struct["loss"] = Methods.eval_method(args_dict, train_result)

            for end_condition_method in self.end_condition_methods:
                is_end = end_condition_method(self.time, self.step, train_result)

                if is_end is True:
                    self.status = STATUS_FAIL
                    break

        return ret_struct
