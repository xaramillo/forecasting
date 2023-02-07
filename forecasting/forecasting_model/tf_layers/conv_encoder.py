import tensorflow as tf

class Conv1DEncoder(tf.keras.layers.Layer):
    def __init__(self, conv_kernel_sizes=None, conv_filters=None, activation=tf.nn.leaky_relu, pooling=True, **kwargs):
        super(Conv1DEncoder, self).__init__()
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [32, 16]
        if conv_filters is None:
            conv_filters = [32, 64]

        if len(conv_filters) != len(conv_kernel_sizes):
            raise ValueError("conv_filters and conv_kernal_sizes must be the same length. Got {} and {}".format(len(conv_filters), len(conv_kernel_sizes)))

        self.conv_layers = []
        for i in range(len(conv_filters)):
            self.conv_layers.append(tf.keras.layers.Conv1D(conv_filters[i], conv_kernel_sizes[i], activation=activation))
            if pooling:
                self.conv_layers.append(tf.keras.layers.MaxPool1D())

    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x
