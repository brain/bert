import tensorflow as tf
import pickle
import os
import tokenization
import featurization
import siamese_bert
import itertools
from tqdm import tqdm
from siamese_bert import SiameseBert
from ftm_processor import FtmProcessor

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

flags.DEFINE_string("dataset", None, "Data ID to use for this experiment.")
flags.DEFINE_integer("max_seq_length", 50, "Max token count for a query.")
flags.DEFINE_string("vdn_string", None, "String for val data.")
flags.DEFINE_string("DAr_string", None, "String for DAr data.")

def input_feature_pair_generator(vdn_queries, dar_queries,
                                 bert_pretrained_dir, bert_model_type):
    """Yields single InputFeaturesPair instances"""
    processor = FtmProcessor()

    # tokenizer
    vocab_file = os.path.join(bert_pretrained_dir, 'vocab.txt')
    config_file = os.path.join(bert_pretrained_dir, 'bert_config.json')
    init_checkpoint = os.path.join(bert_pretrained_dir, 'bert_model.ckpt')
    do_lower_case = bert_model_type.startswith('uncased')

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=do_lower_case)

    for l_q, r_q in itertools.product(vdn_queries, dar_queries):
        # convert to example objects
        inp_ex = processor._get_input_example(l_q, r_q)

        # convert to features
        inp_fe = featurization.convert_single_example(
            None, inp_ex, FLAGS.max_seq_length, tokenizer)

        yield inp_fe


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    TASK_DATA_DIR = f'./example_data/{FLAGS.dataset}/'
    TFRECORD_OUTPUT_DIR = os.path.join(TASK_DATA_DIR, 'devacc_tfrecords')
    os.makedirs(TFRECORD_OUTPUT_DIR, exist_ok=True)
    BERT_MODEL = 'uncased_L-12_H-768_A-12'
    BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL

    # load query pairs into memory
    tf.logging.info('Loading vdn and DAr data...')
    df_vdn = pickle.load(open(
        os.path.join(TASK_DATA_DIR, f'{FLAGS.dataset}_{FLAGS.vdn_string}.pkl'), 'rb'))
    df_DAr = pickle.load(open(
        os.path.join(TASK_DATA_DIR, f'{FLAGS.dataset}_{FLAGS.DAr_string}.pkl'), 'rb'))

    vdn_queries = df_vdn['query']
    dar_queries = df_DAr['query']
    tf.logging.info('Done loading vdn and DAr data...')

    tfrecord_save_path = os.path.join(
        TFRECORD_OUTPUT_DIR,
        f'{FLAGS.dataset}_devacc_pairs_{FLAGS.vdn_string}_{FLAGS.DAr_string}_{FLAGS.max_seq_length}.tfrecord')
    if not os.path.exists(tfrecord_save_path):
        inp_fe_generator = input_feature_pair_generator(
            list(vdn_queries), list(dar_queries), BERT_PRETRAINED_DIR,
            BERT_MODEL)
        with tf.io.TFRecordWriter(
            tfrecord_save_path) as writer:

            for inp_feat_pair in tqdm(inp_fe_generator, total=len(vdn_queries) * len(dar_queries)):
                features = tf.train.Features(
                    feature={
                        'l_input_ids': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=inp_feat_pair.l_input_ids)),
                        'r_input_ids': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=inp_feat_pair.r_input_ids)),
                        'l_input_mask': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=inp_feat_pair.l_input_mask)),
                        'r_input_mask': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=inp_feat_pair.r_input_mask))})
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

if __name__ == "__main__" :
    tf.app.run()
