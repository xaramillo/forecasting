import tensorflow as tf
from forecasting.utils import positional_encoding
from forecasting.forecasting_model.tf_layers.transformer_layers.decoder_layer import DecoderLayer


class Decoder(tf.keras.layers.Layer):
    def __init__(self,  window_out=30, num_layers=1, dim_model=256, num_heads=8, dim_ff=1024,
                 dropout_rate=0.1, **kwargs):
        super(Decoder, self).__init__()

        self.dim_model = dim_model
        self.num_layers = num_layers

        self.embedding_dense = tf.keras.layers.Dense(dim_model)
        self.pos_encoding = positional_encoding(window_out, dim_model)

        self.dec_layers = [DecoderLayer(dim_model, num_heads, dim_ff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, look_ahead_mask, training=True):
        seq_len = tf.shape(x)[1]

        x = self.embedding_dense(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.dim_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, look_ahead_mask, training)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x