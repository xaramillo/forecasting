import os
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
VIS_DIR = os.path.join(ROOT_DIR, "generated_forecasts")

VANILLA_MODEL_NAME = "VANILLA"
SEQ2SEQ_MODEL_NAME = "SEQ2SEQ"
TRANSFORMER_MODEL_NAME = "TRANSFORMER"

tf.random.set_seed(99)
