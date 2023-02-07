import tensorflow as tf

from forecasting.forecasting_model.tf_layers.transformer_layers.multi_head_attention import MultiHeadAttention
from forecasting.forecasting_model.tf_layers.transformer_layers.feed_forward import FeedFoward

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim_model=256, num_heads=8, dim_ff=1024, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.self_mha = MultiHeadAttention(dim_model, num_heads)
        self.ffn = FeedFoward(dim_model, dim_ff)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_layer1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_layer2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=True, mask=None):

        # Self attention
        self_attn_output = self.self_mha(x, x, x, mask)  # (batch_size, input_seq_len, dim_model)
        self_attn_output = self.dropout_layer1(self_attn_output, training=training)

        # Add and norm 1
        attention_norm = self.layer_norm1(x + self_attn_output)  # (batch_size, input_seq_len, dim_model)

        ffn_output = self.ffn(attention_norm)  # (batch_size, input_seq_len, dim_model)
        ffn_output = self.dropout_layer2(ffn_output, training=training)

        # Add and norm 2
        encoded = self.layer_norm2(attention_norm + ffn_output)  # (batch_size, input_seq_len, dim_model)

        return encoded
