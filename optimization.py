import tensorflow as tf


def sin_regression(gt, a=5., w=1., u=0., shift=0., fps=30):
    a = tf.Variable(initial_value=a, trainable=True, dtype=tf.float32)
    w = tf.Variable(initial_value=w, trainable=True, dtype=tf.float32)
    u = tf.Variable(initial_value=u, dtype=tf.float32)
    shift = tf.Variable(initial_value=shift, trainable=True, dtype=tf.float32)
    x = tf.Variable(initial_value=tf.linspace(0., 1., fps), trainable=False)
    out = a * tf.sin(w * x + u) + shift
    loss = tf.nn.l2_loss(gt - out)
    return loss, a, w, u, shift, x
