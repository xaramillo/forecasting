import pandas as pd
import forecasting as f
import os
from forecasting.utils import load_config
from forecasting.forecasting_model.tf_models.transformer_forecast_model import ForecastTransformer
from forecasting.predict import predict
from forecasting.utils import visualise_prediction

model_dir = os.path.join(f.MODEL_DIR, "transformer_final")

test_data = pd.read_csv(os.path.join(f.DATA_DIR, "sample_test_data.csv"), index_col=0)

config = load_config(master_config_name='master_config.yaml', model_config_name='transformer_model_config.yaml')

model = ForecastTransformer(**config)

model.load_weights(model_dir).expect_partial()

for i in range(len(test_data)):
    series = test_data.iloc[i].values
    prediction = predict(model, series, **config)
    history = series[-config['window_out'] - config['window_in']:-config['window_out']]
    true_forecast = series[-config['window_out']:]
    visualise_prediction(history, true_forecast, prediction, save_name="Transformer_forecast_{}.png".format(i))