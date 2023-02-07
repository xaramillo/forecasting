import tensorflow as tf
from forecasting.forecasting_model.tf_layers.transformer_layers.transformer import Transformer
from forecasting.forecasting_model.tf_layers.conv_encoder import Conv1DEncoder

class ForecastTransformer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ForecastTransformer, self).__init__()
        self.transformer = Transformer(**kwargs)
        self.conv_model = Conv1DEncoder(**kwargs)

    def call(self, x, **kwargs):
        conv_encoded = self.conv_model(x)
        pred = self.transformer(conv_encoded, **kwargs)
        return pred