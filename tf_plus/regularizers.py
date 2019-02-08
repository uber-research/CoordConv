import tensorflow as tf

def l2reg(l2_strength):
    if l2_strength == 0:
        return lambda x: tf.zeros(())
    return lambda x: l2_strength * tf.reduce_sum(tf.square(x))
