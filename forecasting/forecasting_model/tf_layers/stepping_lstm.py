import tensorflow as tf

class SteppingLSTM(tf.keras.layers.Layer):
    def __init__(self, batch_size, lstm_hidden_dims=None, bidirectional=False, **kwargs):
        super(SteppingLSTM, self).__init__()
        if lstm_hidden_dims is None:
            lstm_hidden_dims = [256]

        # self.state_dims = []
        # for dim in lstm_hidden_dims:
        #     self.state_dims.append((batch_size, dim * (bidirectional + 1)))
        self.state_dims = [(batch_size, dim * (bidirectional + 1)) for dim in lstm_hidden_dims]
        #self.state_dims = [(batch_size, lstm_hidden_dims[0] * (bidirectional + 1)) for dim in lstm_hidden_dims]

        self.bidirectional = bidirectional
        if bidirectional:
            bidirectional_wrapper = tf.keras.layers.Bidirectional
        else:
            bidirectional_wrapper = lambda x: x

        self.stepping_layers = [
            bidirectional_wrapper(tf.keras.layers.LSTM(lstm_hidden_dim, return_sequences=True, return_state=True)) for
            lstm_hidden_dim in lstm_hidden_dims]

    def call_step(self, x, last_h, last_c):
        h = []
        c = []
        for i, layer in enumerate(self.stepping_layers):
            if self.bidirectional:
                last_h_fw, last_h_bw = tf.split(last_h[i], 2, axis=-1)
                last_c_fw, last_c_bw = tf.split(last_c[i], 2, axis=-1)
                x, h_fw, c_fw, h_bw, c_bw = layer(x, initial_state=(last_h_fw, last_c_fw, last_h_bw, last_c_bw))
                h.append(tf.concat([h_fw, h_bw], axis=-1))
                c.append(tf.concat([c_fw, c_bw], axis=-1))
            else:
                last_h_i = last_h[i]
                last_c_i = last_c[i]
                x, h_next, c_next = layer(x, initial_state=(last_h_i, last_c_i))
                h.append(h_next)
                c.append(c_next)

        encoded_context_word = x

        return encoded_context_word, h, c

    def get_init_state(self):
        h = [tf.zeros(state_dim) for state_dim in self.state_dims]
        c = [tf.zeros(state_dim) for state_dim in self.state_dims]
        return h, c

    def call(self, x, steps):
        encoded = []
        h, c = self.get_init_state()

        for i in range(steps):
            input_context = tf.expand_dims(x[:, i], -1)
            enc_i, h, c = self.call_step(input_context, h, c)
            encoded.append(tf.squeeze(enc_i))
        return tf.stack(encoded, axis=1)