import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def generator(input):
    with tf.name_scope("Generator"):
        y = tf.add(input, tf.constant(1.0), name="y")
        tf.summary.scalar("y", y)
        return y

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x1 = tf.Variable(1.0)
x2 = tf.Variable(5.0)

y1 = generator(x1)
y2 = generator(x2)

with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())
    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log2", ss.graph)

    for i in range(1000):
        if i % 10 == 0:
            print("i: ", i)

        summary, _, _ = ss.run([merged, y1, y2])
        writer.add_summary(summary, i)

    print(ss.run([y1, y2]))