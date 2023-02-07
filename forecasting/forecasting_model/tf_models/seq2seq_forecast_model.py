import tensorflow as tf
from forecasting.forecasting_model.tf_layers.conv_encoder import Conv1DEncoder
from forecasting.forecasting_model.tf_layers.lstm_decdoder import DecoderLSTM

class Seq2SeqForecastModel(tf.keras.Model):
    def __init__(self, encoder_lstm_dim=128, **kwargs):
        super(Seq2SeqForecastModel, self).__init__()
        self.conv_encoder = Conv1DEncoder(**kwargs)

        self.seq_model = tf.keras.layers.LSTM(encoder_lstm_dim)

        self.decoder = DecoderLSTM(**kwargs)

    def call(self, x, **kwargs):
        final_x_val = x[:, -1]
        enc = self.conv_encoder(x)
        enc = self.seq_model(enc)
        prediction = self.decoder(enc, final_x_val, **kwargs)
        return prediction
