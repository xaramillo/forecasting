from forecasting.preprocessor import Preprocessor
import numpy as np

def predict(trained_model, input_data, **kwargs):
    preprocessor = Preprocessor()
    model_input, mean, std = preprocessor.preprocess_predict_series(input_data, **kwargs)

    pred = np.squeeze(trained_model(model_input).numpy()[0])
    transformed_pred = pred * std + mean
    return transformed_pred