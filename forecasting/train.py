from forecasting.forecasting_model.trainers.seq2seq_model_trainer import Seq2SeqTrainer
from forecasting.forecasting_model.trainers.vanilla_model_trainer import VanillaTrainer
from forecasting.forecasting_model.trainers.transformer_model_trainer import TransformerModelTrainer
import forecasting as f
from forecasting.preprocessor import Preprocessor

def train_model(model_type, df, **kwargs):
    preprocessor = Preprocessor()
    train_dataset, test_dataset = preprocessor.form_datasets(df, **kwargs)

    trainer_obj_dict = {
        f.VANILLA_MODEL_NAME : VanillaTrainer,
        f.SEQ2SEQ_MODEL_NAME: Seq2SeqTrainer,
        f.TRANSFORMER_MODEL_NAME: TransformerModelTrainer
    }

    trainer = trainer_obj_dict[model_type](**kwargs)
    model = trainer.train_loop(train_dataset, test_dataset, **kwargs)
    return model