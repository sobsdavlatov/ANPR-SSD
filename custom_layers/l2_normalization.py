import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class L2Normalization(Layer):
    def __init__(self, gamma_init = 20, axis =- 1, **kwargs):
        self.axis = axis
        self.gamma_init = gamma_init
        super(L2Normalization, self).__init__(*kwargs)
    
    def build(self, input_shape):
        gamma = self.gamma_init * np.ones((input_shape[self.axis],),dtype=np.float32)
        self.gamma = tf.Variable(gamma, trainable=True)
        super(L2Normalization, self).build(input_shape)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, self.axis) * self.gamma
    
    def get_config(self, inputs):
        config = {"gamma_init": self.gamma_init, 'axis': self.axis}
        base_config = super(L2Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()) ) 
    
    @classmethod
    def from_config(csl, config):
        return csl(**config)