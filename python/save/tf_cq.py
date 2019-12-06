import datetime

import tensorflow as tf
import numpy as np

import python.util.helper as helper
import python.util.plot_utils as plot_utils

num_image_width = 4
num_image_height = 4
num_batch = num_image_width * num_image_height
num_z = 256
num_latent_cat = 64
num_iter = int(1e3)

prr = plot_utils.Plot_Reproduce_Performance("result_tf", num_image_width, num_image_height, 22, 22, 1)

z_list = list()
cat_list = list()

for i in range(num_iter):
    real_z_val = helper.random_normal(0, 1, [num_batch, num_z - num_latent_cat])

    input_cat_val = np.zeros([num_batch, num_latent_cat], dtype=np.float32)

    for i_batch in range(num_batch):
        cat_index = np.random.randint(64)
        input_cat_val[i_batch][cat_index] = 1

    z_list.append(real_z_val)
    cat_list.append(input_cat_val)

with tf.Session() as sess:
    save_path = tf.train.latest_checkpoint("train")

    saver = tf.train.import_meta_graph(save_path + ".meta")
    saver.restore(sess, save_path)

    real_z = sess.graph.get_tensor_by_name("Placeholder/real_z:0")
    input_cat = sess.graph.get_tensor_by_name("Placeholder/input_cat:0")

    output = sess.graph.get_tensor_by_name("Generator_variables/gen_image:0")
    output_list = list()

    def generate(i):
        real_z_val = z_list[i]
        input_cat_val = cat_list[i]

        feed_dict = { real_z: real_z_val, input_cat: input_cat_val}
        output_val = sess.run(output, feed_dict=feed_dict)

        return output_val

    # Load model before actual running
    generate(0)

    begin_time = datetime.datetime.now()

    for i in range(num_iter):
        output_val = generate(i)
        output_list.append(output_val)

    end_time = datetime.datetime.now()

    diff = end_time - begin_time
    diff = int(diff.total_seconds() * 1000)

    print("Diff: %d" % diff)

    for i, output_val in enumerate(output_list):
        prr.save_pngs(output_val, 3, "output%d.png" % i)
