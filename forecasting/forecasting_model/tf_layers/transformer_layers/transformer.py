import tensorflow as tf
from forecasting.forecasting_model.tf_layers.transformer_layers.stacked_encoder import Encoder
from forecasting.forecasting_model.tf_layers.transformer_layers.stacked_decoder import Decoder
from forecasting.utils import create_look_ahead_mask


class Transformer(tf.keras.Model):
    def __init__(self, batch_size, window_out=30, **kwargs):
        super(Transformer, self).__init__()

        self.encoder = Encoder(**kwargs)

        self.decoder = Decoder(window_out=window_out, **kwargs)

        self.final_layer = tf.keras.layers.Dense(1)
        self.batch_size = batch_size
        self.window_out = window_out

    def call(self, x, target=None, training=True, look_ahead_mask=None, **kwargs):

        if target is None:
            target = tf.zeros((self.batch_size, self.window_out))

        if look_ahead_mask is None:
            look_ahead_mask = create_look_ahead_mask(self.window_out)

        target = tf.expand_dims(target, axis=-1)

        enc_output = self.encoder(x, training, mask=None)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(
            target, enc_output, look_ahead_mask, training)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output