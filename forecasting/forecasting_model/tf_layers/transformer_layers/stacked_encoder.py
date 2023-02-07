import tensorflow as tf
from forecasting.utils import positional_encoding
from forecasting.forecasting_model.tf_layers.transformer_layers.encoder_layer import EncoderLayer

class Encoder(tf.keras.layers.Layer):
    def __init__(self, max_positional_encoding_input, num_layers=1, dim_model=256, num_heads=8, dim_ff=1024,
                 dropout_rate=0.1, **kwargs):
        super(Encoder, self).__init__()

        self.dim_model = dim_model
        self.num_layers = num_layers

        self.embedding_dense = tf.keras.layers.Dense(dim_model)
        self.pos_encoding = positional_encoding(max_positional_encoding_input,
                                                self.dim_model)

        self.enc_layers = [EncoderLayer(dim_model, num_heads, dim_ff, dropout_rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=True, mask=None):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding_dense(x)  # (batch_size, input_seq_len, dim_model)
        x *= tf.math.sqrt(tf.cast(self.dim_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)