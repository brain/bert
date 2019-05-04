import tensorflow as tf
import os

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    'model_trial_name', None,
    'Trial name for model.')


def main(_):
    """
        Example usage:

        ```
        python -m download_model --model_trial_name=FTM_BERT_DATA_008_tpu_trial_2
        ```

    """

    OUTPUT_BUCKET = 'bert_output_bucket_mteoh'
    OUTPUT_DIR = 'gs://{}/bert/models/{}'.format(OUTPUT_BUCKET, FLAGS.model_trial_name)
    MAIN_DIR = os.path.dirname(__file__)

    checkpoint_path = tf.train.latest_checkpoint(OUTPUT_DIR)
    print(checkpoint_path)

    model_save_dir = os.path.join(MAIN_DIR, 'models', FLAGS.model_trial_name)

    os.makedirs(model_save_dir, exist_ok=True)

    os.system(f'gsutil -m cp {checkpoint_path}.* {model_save_dir}/')


if __name__ == "__main__":
    tf.app.run()
