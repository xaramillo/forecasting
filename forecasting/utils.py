import tensorflow as tf
import numpy as np
import yaml
import forecasting as f
import os
import matplotlib.pyplot as plt

def load_config(master_config_name='master_config.yaml', model_config_name='vanilla_model_config.yaml'):
    with open(os.path.join(f.CONFIG_DIR, master_config_name)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    with open(os.path.join(f.CONFIG_DIR, model_config_name)) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)
    config.update(model_config)
    return config

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def visualise_prediction(history, true_forecast, model_prediction, save_name=None):
    correct = np.concatenate((history, true_forecast), axis=-1)
    predicted = np.concatenate((history, model_prediction), axis=-1)

    fig = plt.figure()
    plt.plot(predicted, label='Predicted', alpha=0.5)
    plt.plot(correct, label='Correct', alpha=0.5)
    plt.axvline(len(history), c='black', label='Start Forecast', linestyle='dashed')
    plt.legend()
    plt.ylabel("Sold Units")
    plt.xlabel("Day")
    if save_name is not None:
        plt.savefig(os.path.join(f.VIS_DIR, save_name))
    plt.show()


def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

