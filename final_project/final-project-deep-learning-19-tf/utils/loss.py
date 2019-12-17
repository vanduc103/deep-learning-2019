import tensorflow as tf
import numpy as np

from miscc.config import cfg

def cosine_similarity(v1, v2):
    """
    Returns cosine similarity between v1 and v2
    """
    cost = tf.reduce_sum(tf.multiply(v1, v2), 1) / (tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1), 1)) * tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2), 1)))
    return cost