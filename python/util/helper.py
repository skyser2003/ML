import os, glob, random, shutil
import numpy as np


def percentage(cur, max):
    return int(cur / max * 100)


def clean_create_dir(dir):
    if os.path.exists(dir):
        files = glob.glob(os.path.join(dir, "*"))
        for file in files:
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)

    os.makedirs(dir, exist_ok=True)


def random_uniform(min, max, shape):
    return np.random.uniform(min, max, shape).astype(np.float32)


def random_normal(center, std, shape):
    return np.random.normal(center, std, shape).astype(np.float32)


def uniform_z(min, max, shape):
    m, n = shape

    diff = max - min

    ret = np.arange(0, m)
    ret = ret / m * diff
    ret += min
    ret = np.tile(ret, [n, 1])
    ret = np.transpose(ret)

    return ret


def constant_z(val, shape):
    return np.full(shape, val)


def random_cat(shape):
    ret = constant_z(0, shape)

    m, n = shape

    if 0 < n:
        for i in range(m):
            index = random.randint(0, n - 1)
            ret[i][index] = 1

    return ret
