# Product Sales Forecasting

Three different deep learning architectures designed to forecast the daily sales of 
individual products a month into the future.

### Architectures:
- __Vanilla__: Convolution, max pooling and LSTM encoder, decoded by a MLP 
- __Seq2Seq__: Convolution, max pooling and LSTM encoder, decoded by a LSTM
- __Transformer__: Convolution, max pooling followed by Transformer encoder and decoder

## 1. Data & Preprocessing

The data consists of 1941 days of unit sales in Wallmart stores for over 3000 unique products. A handful
of products have been removed from `wallmart_item_sales.csv` into `sample_test_data.csv` to
illustrate predictions of trained models. During training, 200 random products are placed into a validation dataset.

The preprocessor creates multiple training examples per product by sliding a window over the history. The default
window size is 400 days of input to produce the 30 day forecast. This start of the window is shifted 200 days into the future
after each training example is generated. 

Prepocessing is done using the `tf.data` library to produce an optimised input 
pipeline. A Tensorflow graph is produced which can execute multiple parts of the 
pipeline in parallel for efficient processing. This code can be found in `preprocessor.py`.

`tf.data` Pipeline Performance Guide: https://www.tensorflow.org/guide/data_performance


## 2. Model Implementation Details

### 2.1 Vanilla Model

Multiple layers of convolution and max pooling are used to increase the dimensionality of the input 
by encoding some spatial features as well as reducing the length of the sequence. The shortened sequence
is then fed through a LSTM to produce an encoding of the history. A Multi-Layered Perceptron (MLP) is then used to decode
the sequence and produce the 30 day forecast.

### 2.2 Seq2Seq Model

Multiple layers of convolution and max pooling are used to increase the dimensionality of the input 
by encoding some spatial features as well as reducing the length of the sequence. The shortened sequence
is then fed through a LSTM to produce an encoding of the history. A decoder LSTM makes a prediction 
one time step at a time by taking in the previous prediction (embedded to a high dimension using a dense layer) as well
as the state of the LSTM from the previous step. During training, teacher forcing is used to improve optimisation;
instead of stepping using the previous prediction, the ground truth at the previous time step is used as input.

### 2.3 Transformer Model

Multiple layers of convolution and max pooling are used to increase the dimensionality of the input 
by encoding some spatial features as well as reducing the length of the sequence. A transformer encoder-decoder
network is then used to encode the shortened sequence using self attention and then decode the output into
a 30 day forecast. 

Commonly, teacher forcing is used in the decoder through the use of a look-ahead mask.
Through experimentation, I found this ineffective for product sales forecasting. 
The model struggled when producing forecasts one time step at a time when the future was unknown.
Most likely this was because the volatile nature of the time series made it difficult to produce a 
forecast recursively; this would explain why the vanilla approach which produces the whole forecast at once
performed better than the Seq2Seq model (see section 8). 

The implementation in this library feeds zeros as input to the decoder with positional
encodings then added. When forecasts are being performed for data with an unknown future,
the process of prediction is now the same as during the training procedure and this proved 
far more effective.

## 3. Install

Create Environment: `conda create --name forecasting python=3.6`

Activate Environment: `conda activate forecasting`

Install Requirements: `pip install -r requirements.txt`

## 4. Run

Train:
- `python -m examples.example_train_vanilla`
- `python -m examples.example_train_seq2seq`
- `python -m examples.example_train_transformer`

Predict:
- `python -m examples.example_predict_vanilla`
- `python -m examples.example_predict_seq2seq`
- `python -m examples.example_predict_transformer`

## 5. Produced Forecasts

A few examples of generated forecasts for unseen data using each architectures can be found in the generated_forecasts directory.

## 6. Project Structure

- __config__: Yaml files with configurations for models and input data
- __data__: Csv files of product sales data
- __examples__: Examples of how to effectively use the library
- __forecasting__: Library
- __generated_forecasts__: Saved visualisations of generated forecasts by the different architectures
- __logs__: Saved training logs
- __saved_models__: Saved model weights

## 7. Library Structure

- __forecasting_model__: Implementation of tensorflow models and their trainers
    - __tf_layers__: Reusable Tensorflow layers which are all instances of tf.keras.layers.Layer
        - __transformer_layers__: Layers specifically used for the Transformer architecture
    - __tf_model__: Instances of tf.keras.Model, the architectures used to predict the sales of products 
    - __trainers__: Wrappers around the models to train them and save the resulting weights and logs
- __predict__:  Function to forecast an individual time series using a trained model
- __preprocessor__:  Preprocesses training and test data
- __train__: Function to train models
- __utils__:  Extra utility functions

## 8. Performance

| Architecture  | Train MSE  | Validation MSE  |
|---|---|---|
| Vanilla  |  0.34 | 0.38  |
| Seq2Seq  |  0.34 (with teacher forcing) | 0.49 (no teacher forcing)|
| Transformer  |  0.41 | 0.42  | 
