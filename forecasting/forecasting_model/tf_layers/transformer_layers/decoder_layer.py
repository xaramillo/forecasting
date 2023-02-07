import tensorflow as tf

from forecasting.forecasting_model.tf_layers.transformer_layers.multi_head_attention import MultiHeadAttention
from forecasting.forecasting_model.tf_layers.transformer_layers.feed_forward import FeedFoward


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim_model=256, num_heads=8, dim_ff=1024, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.self_mha = MultiHeadAttention(dim_model, num_heads)
        self.encoder_mha = MultiHeadAttention(dim_model, num_heads)

        self.ffn = FeedFoward(dim_model, dim_ff)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_layer1 = tf.keras.layers.Dropout(rate)
        self.dropout_layer2 = tf.keras.layers.Dropout(rate)
        self.dropout_layer3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask, training=True):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # Self attention on output sequence
        self_output_attention = self.self_mha(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        self_output_attention = self.dropout_layer1(self_output_attention, training=training)
        # Add and norm 1
        self_output_attention_norm = self.layer_norm1(self_output_attention + x)

        # Attention using encoder output for query and key
        encoder_mha_output = self.encoder_mha(enc_output, enc_output, self_output_attention_norm)  # (batch_size, target_seq_len, d_model)
        encoder_mha_output = self.dropout_layer2(encoder_mha_output, training=training)
        # Add and norm 2
        encoder_mha_output_norm = self.layer_norm2(encoder_mha_output + self_output_attention_norm)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(encoder_mha_output_norm)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout_layer3(ffn_output, training=training)
        # Add and norm 3
        decoded = self.layer_norm3(ffn_output + encoder_mha_output_norm)  # (batch_size, target_seq_len, d_model)

        return decoded
