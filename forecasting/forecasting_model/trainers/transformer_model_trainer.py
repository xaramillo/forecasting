import tensorflow as tf
import time
from forecasting.forecasting_model.tf_models.transformer_forecast_model import ForecastTransformer
from forecasting.utils import create_look_ahead_mask
import forecasting as f
import os
import datetime

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dim_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = dim_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class TransformerModelTrainer():
    def __init__(self, dim_model=128, window_out=30, loss_func=tf.keras.losses.mse, optimizer=tf.keras.optimizers.Adam, lr=None, warmup_steps=4000,
                 beta_1=0.9, beta_2=0.98, epsilon=1e-9, model_name='transformer_forecasting', **kwargs):
        if lr is None:
            lr = CustomSchedule(dim_model, warmup_steps)

        self.model = ForecastTransformer(dim_model=dim_model, **kwargs)
        self.loss_func = loss_func
        self.optimizer = optimizer(lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        self.window_out = window_out

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        self.model_name = model_name

    @tf.function
    def train_step(self, x, y):
        look_ahead_mask = create_look_ahead_mask(self.window_out)

        with tf.GradientTape() as tape:
            predictions = self.model(x,
                                         training=True,
                                         look_ahead_mask=look_ahead_mask)
            loss = self.loss_func(y,predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    @tf.function
    def eval_step(self, x, y):
        look_ahead_mask = create_look_ahead_mask(self.window_out)

        predictions = self.model(x,
                                    training=True,
                                    look_ahead_mask=look_ahead_mask)
        loss = self.loss_func(y, predictions)

        self.val_loss(loss)

    def train_loop(self, train_dataset, test_dataset, epochs=90, log=True, **kwargs):
        ckpt = tf.train.Checkpoint(model=self.model,
                                   optimizer=self.optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(f.MODEL_DIR, self.model_name), max_to_keep=5)

        if log:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(f.LOG_DIR, self.model_name) + current_time + '/train'
            val_log_dir =  os.path.join(f.LOG_DIR, self.model_name) + current_time + '/val'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for epoch in range(epochs):
            start = time.time()
            for batch, (x, y) in enumerate(train_dataset):
                self.train_step(x, y)

                if batch % 25 == 0:
                    print('Batch {} Loss {:.4f}'.format(
                        batch, self.train_loss.result()))

            print('Epoch {} Loss {:.4f}'.format(
                epoch + 1, self.train_loss.result()))

            for (batch, (x, y)) in enumerate(test_dataset.take(20)):
                self.eval_step(x, y)

            print('Epoch {} Val Loss {:.4f}'.format(
                epoch + 1, self.val_loss.result()))

            if (epoch + 1) % 3 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            if log:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                with val_summary_writer.as_default():
                    tf.summary.scalar('val_loss', self.val_loss.result(), step=epoch)

            self.train_loss.reset_states()
            self.val_loss.reset_states()
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        return self.model
