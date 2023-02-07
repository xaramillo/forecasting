import tensorflow as tf
class DenseDecoder(tf.keras.Model):
    def __init__(self, window_out=30, mlp_layer_sizes=None, activation=tf.nn.relu, **kwargs):
        super(DenseDecoder, self).__init__()

        if mlp_layer_sizes is None:
            mlp_layer_sizes = [256, 128]

        self.dense_layers = [tf.keras.layers.Dense(dim, activation=activation) for dim in mlp_layer_sizes]
        self.final_reshape_layer = tf.keras.layers.Dense(window_out)

    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return self.final_reshape_layer(x)