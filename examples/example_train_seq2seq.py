import pandas as pd
from forecasting.train import train_model
import forecasting as f
import os
from forecasting.utils import load_config

data = pd.read_csv(os.path.join(f.DATA_DIR, "wallmart_item_sales.csv"), index_col=0)

config = load_config(master_config_name='master_config.yaml', model_config_name='seq2seq_model_config.yaml')

model = train_model(f.SEQ2SEQ_MODEL_NAME, data, **config)