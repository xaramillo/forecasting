import tensorflow as tf
import numpy as np

class Preprocessor():
    @staticmethod
    @tf.function
    def normalise_series(series):
      return (series - tf.math.reduce_mean(series)) / tf.math.reduce_std(series)

    @staticmethod
    @tf.function
    def _make_window_dataset(ds, window_in=400, window_out=30, window_shift=200):
        # Forms series windows of input and target
        window_size = window_in + window_out
        windows = ds.window(window_size, shift=window_shift)

        def sub_to_batch(sub):
            return sub.batch(window_size, drop_remainder=True)

        def split_window(window):
            return window[:window_in], window[window_in:]

        windows = windows.flat_map(sub_to_batch)
        windows = windows.map(split_window)
        return windows

    @tf.function
    def preprocess_series(self, series, **kwargs):
        # Form dataset of windows given a full length time series
        norm_series = self.normalise_series(series)
        norm_series_expanded = tf.expand_dims(norm_series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(norm_series_expanded)
        ds = self._make_window_dataset(ds, **kwargs)
        return ds

    @staticmethod
    def preprocess_predict_series(series, window_in, batch_size, **kwargs):
        # Preprocess time series for prediction on trained model
        series_window = series[-window_in:]
        mean = np.mean(series_window)
        std = np.std(series_window)
        norm_series = (series_window - mean) / std
        preprocessed_series = tf.cast(tf.expand_dims(tf.expand_dims(norm_series, axis=-1), axis=0), tf.float32)
        return tf.broadcast_to(preprocessed_series, [batch_size, window_in, 1]), mean, std

    def form_datasets(self, df, num_test_series=200, shuffle_buffer_size=2048, batch_size=128, **kwargs):
        # Form train, test datasets
        dataset = tf.data.Dataset.from_tensor_slices(df.values.astype(float)).shuffle(shuffle_buffer_size)

        test_dataset = dataset.take(num_test_series)
        train_dataset = dataset.skip(num_test_series)

        train_dataset = (train_dataset
                         # interleave for parallel processing on each time series
                         .interleave(self.preprocess_series,
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
                         # cache for faster access
                         .cache()
                         .shuffle(shuffle_buffer_size)
                         .batch(batch_size, drop_remainder=True)
                         # prefetch to execute different parts of pipeline in parallel
                         .prefetch(tf.data.experimental.AUTOTUNE))

        test_dataset = (test_dataset
                        .interleave(self.preprocess_series,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        .shuffle(shuffle_buffer_size)
                        .batch(batch_size, drop_remainder=True)
                        .prefetch(tf.data.experimental.AUTOTUNE))

        return train_dataset, test_dataset