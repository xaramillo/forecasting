import tensorflow as tf
from forecasting.forecasting_model.tf_layers.stepping_lstm import SteppingLSTM

class DecoderLSTM(tf.keras.layers.Layer):
    def __init__(self, decoder_lstm_dims=None, decoder_embed_layer_dim=0, **kwargs):
        super(DecoderLSTM, self).__init__()
        if decoder_lstm_dims is None:
          decoder_lstm_dims = [128]

        self.lstm_hidden_dims = decoder_lstm_dims
        self.stepping_lstm = SteppingLSTM(lstm_hidden_dims=decoder_lstm_dims, bidirectional=False, **kwargs)
        self.encode_reshape_layers = [tf.keras.layers.Dense(dim) for dim in decoder_lstm_dims]

        self.embed_func = tf.keras.layers.Dense(decoder_embed_layer_dim) if decoder_embed_layer_dim > 0 else lambda x: x
        self.reduce_layer = tf.keras.layers.Dense(1)


    def call(self, encoded_input_series, final_x_val, window_out=30, correction_signal=None, **kwargs):
        correct_dec_input = correction_signal is not None
        predicted = []
        dec_input = self.embed_func(tf.expand_dims(final_x_val, axis=1))
        _, c = self.stepping_lstm.get_init_state()
        h = [reshape_layer(encoded_input_series) for reshape_layer in self.encode_reshape_layers]
        for t in range(window_out):
            dec_out, h, c = self.stepping_lstm.call_step(dec_input, h, c)
            dec_out = self.reduce_layer(dec_out)
            predicted.append(tf.squeeze(dec_out, axis=1))
            if correct_dec_input:
                dec_input = self.embed_func(tf.expand_dims(correction_signal[:, t], axis=-1))
            else:
                dec_input = self.embed_func(dec_out)
        return tf.stack(predicted, axis=1)