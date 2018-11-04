import tensorflow as tf
g = tf.Graph()
with g.as_default():
    x = tf.Variable(tf.random_normal([3]))
s1 = tf.Session(graph=g)
s2 = tf.Session(graph=g)
s1.run(x.initializer)
s2.run(x.initializer)
print(s1.run(x))
# array([ 0.3946251 , -0.72699219, 0.22102632], dtype=float32)
print(s2.run(x))
# array([-0.32520196, 0.75554067, 1.03024554], dtype=float32)