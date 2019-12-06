import tensorflow as tf

# arr = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
arr = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]

x = tf.placeholder("int32", [None, None])

reduce = tf.reduce_sum(x, 0)

sess = tf.Session()
result = sess.run(reduce, feed_dict={x: arr});

print(result)
