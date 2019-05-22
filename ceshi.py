import tensorflow as tf
label=tf.constant([[0.5,0.3,0.8],[0.2,0.4,0.6]])
one = tf.ones_like(label)
zero = tf.zeros_like(label)
label2 = tf.where(label <0.5, x=zero, y=one)

with tf.Session() as sess:
    print(sess.run(label))

    print(sess.run(label2))