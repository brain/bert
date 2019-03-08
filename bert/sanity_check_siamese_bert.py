import tensorflow as tf
import modeling
import optimization
import run_classifier
import tokenization
import os
import datetime
import pickle
from siamese_bert import SiameseBert

flags = tf.flags
FLAGS = flags.FLAGS

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

# Remove the flags from `run_classifier` since we're not using them in this
# file
del_all_flags(tf.flags.FLAGS)

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("predict_batch_size", 128, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("use_debug", False, "Whether to use tf debugger.")
flags.DEFINE_string("task", "FTM_sanity_check", "Task name & output subdir name.")
flags.DEFINE_integer("which_gpu", None, "Which specific GPU to use.")
flags.DEFINE_integer("num_train_epochs", 20, "Number of epochs to train over.")

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    BERT_MODEL = 'uncased_L-12_H-768_A-12'
    BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
    # BERT_PRETRAINED_DIR = './models/' + BERT_MODEL
    BUCKET = 'bert_output_bucket_mteoh'
    TASK = FLAGS.task
    OUTPUT_DIR = 'gs://{}/bert/models/{}'.format(BUCKET, TASK)
    TASK_DATA_PATH = './example_data/DATA_EXAMPLE_train_pairs.pkl'

    if FLAGS.which_gpu:
        tf.logging.info(f'using GPU: {FLAGS.which_gpu}')
        os.environ["CUDA_VISIBLE_DEVICES"]=f"{FLAGS.which_gpu}"

    NUM_Q_REPEAT = 320

    # sb = SiameseBert(
    #     bert_model_type=BERT_MODEL,
    #     bert_pretrained_dir=BERT_PRETRAINED_DIR,
    #     output_dir=OUTPUT_DIR,
    #     use_tpu=FLAGS.use_tpu)

    sb = SiameseBert(
        bert_model_type=BERT_MODEL,
        bert_pretrained_dir=BERT_PRETRAINED_DIR,
        output_dir=OUTPUT_DIR,
        use_tpu=FLAGS.use_tpu,
        num_train_epochs=FLAGS.num_train_epochs,
        use_debug=FLAGS.use_debug)

    q1 = ["Display Jane Hill's Natural schedule for April 1st at 12pm.",
          "Figure out what Jerry's ETA is.",
          "Show me Dropbox file named past activities."]
    q2 = ["View schedules for Saturday with Gregory Thompson via Natural.",
          "Send, follow my account as well, to Danica using Twitter.",
          "View the contents of Today's Task on Google docs."]
    labels = [1, 1, 1]

    l_queries = q1 * NUM_Q_REPEAT
    r_queries = q2 * NUM_Q_REPEAT
    labels = labels * NUM_Q_REPEAT

    tf.logging.info('****evaluate on sanity check dataset...')
    res = sb.evaluate(l_queries, r_queries, labels)
    tf.logging.info(f'{res}')

    tf.logging.info('training on sanity check dataset...')
    sb.train(l_queries, r_queries, labels)
    tf.logging.info('done training')

    tf.logging.info('****evaluating again...')
    res = sb.evaluate(l_queries, r_queries, labels)
    tf.logging.info(f'{res}')


if __name__ == "__main__" :
    tf.app.run()
