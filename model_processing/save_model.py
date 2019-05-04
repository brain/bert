import tensorflow as tf
import os
from training.siamese_modeling import inference_model_fn_builder
from training.input_fns import serving_input_receiver_fn_builder
from google_bert import modeling

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    'model_trial_name', None,
    'Trial name for model.')

flags.DEFINE_string(
    "bert_model", 'uncased_L-12_H-768_A-12',
    "The name of the BERT model.")

flags.DEFINE_string(
    "bert_pretrained_base_dir", 'gs://cloud-tpu-checkpoints/bert/',
    "Base directory of pretrained BERT model.")

BERT_MODEL = FLAGS.bert_model  # 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = FLAGS.bert_pretrained_base_dir + BERT_MODEL  # 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
TRAINED_MODEL_BUCKET = 'bert_output_bucket_mteoh'
TASK = FLAGS.model_trial_name  # 'FTM_BERT_DATA_009_tpu_trial_1'
TRAINED_MODEL_DIR = 'gs://{}/bert/models/{}'.format(TRAINED_MODEL_BUCKET, TASK)
PREDICT_BATCH_SIZE = 256
RANDOM_PROJECTION_OUTPUT_DIM = 128
MAX_SEQ_LENGTH = 30
USE_TPU = False
SAVE_CHECKPOINT_STEPS = 1000


def main(_):
    config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')

    checkpoint_path = tf.train.latest_checkpoint(TRAINED_MODEL_DIR)

    run_config = tf.estimator.RunConfig(
        model_dir=TRAINED_MODEL_DIR)
    model_fn = inference_model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(config_file),
        init_checkpoint=checkpoint_path,
        random_projection_output_dim=RANDOM_PROJECTION_OUTPUT_DIM)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    model_save_dir = os.path.join('models', FLAGS.model_trial_name)
    print(f'saving model in {model_save_dir}...')
    estimator.export_savedmodel(
        export_dir_base=model_save_dir,
        serving_input_receiver_fn=serving_input_receiver_fn_builder(MAX_SEQ_LENGTH)())
    print('done!')


if __name__ == "__main__":
    tf.app.run()
