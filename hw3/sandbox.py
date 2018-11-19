import tensorflow as tf

x1 = tf.manip.reshape(tf.range(9), [3,3])

idx = tf.constant(list(zip(range(3), [1, 0, 2])))
x2 = tf.gather_nd(x1, idx)

x3 = tf.range(3)[:, tf.newaxis]
x4 = tf.range(3, 6)[:, tf.newaxis]
x5 = tf.concat((x3, x4), axis=1)
x6 = tf.random_uniform([1], maxval=10, dtype=tf.int32)
with tf.Session() as sess:
    print(sess.run(x1))
    print(sess.run(x2))
    print(sess.run(x5))
    print(sess.run(x6))
# print(list(zip(range(3), [1, 0, 2])))
