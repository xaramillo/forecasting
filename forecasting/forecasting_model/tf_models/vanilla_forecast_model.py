import tensorflow as tf
from forecasting.forecasting_model.tf_layers.conv_encoder import Conv1DEncoder
from forecasting.forecasting_model.tf_layers.dense_decoder import DenseDecoder

class VanillaForecastModel(tf.keras.Model):
    def __init__(self, lstm_dim=128, **kwargs):
        super(VanillaForecastModel, self).__init__()
        self.conv_encoder = Conv1DEncoder(**kwargs)
        self.seq_model = tf.keras.layers.LSTM(lstm_dim)
        self.decoder = DenseDecoder(**kwargs)

    def call(self, x):
        x = self.conv_encoder(x)
        x = self.seq_model(x)
        prediction = self.decoder(x)
        return prediction