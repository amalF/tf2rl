import tensorflow as tf
from typing import List


class MLPFeatureExtractor(tf.keras.Model):
    def __init__(self, layer_sizes, activation, layer_norm=False, name="MLP"):
        super(MLPFeatureExtractor, self).__init__(name=name)
        self.hidden_layers = []
        for j, size in enumerate(layer_sizes):
            act = activation if not layer_norm else "linear"
            layer = tf.keras.layers.Dense(size, activation=act,
                                          name="{}/dense_{}".format(name, j))
            self.hidden_layers.append(layer)
            if layer_norm:
                norm_layer = tf.keras.layers.LayerNormalization()
                actv_layer = tf.keras.layers.Activation(activation)
                self.hidden_layers.append(norm_layer)
                self.hidden_layers.append(actv_layer)

    def call(self, inputs):
        net = inputs
        for layer in self.hidden_layers:
            net = layer(net)
        return net
