import tensorflow as tf
import pickle
import os
import tokenization
import featurization
import siamese_bert
import numpy as np
import pandas as pd
from time import time as tt
from multiprocessing.pool import ThreadPool
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

flags.DEFINE_string("dataset", None,
    "Data ID to use for this experiment.")

flags.DEFINE_integer("max_seq_length", 50,
    "Max token count for a query.")

flags.DEFINE_integer("queries_per_file", 10000,
    "Number of queries to batch the tfrecord files in. None means no batching.")

flags.DEFINE_integer("num_cores", 32,
    "Number of cores to process the tfrecord files.")


def input_feature_pair_generator(l_queries, r_queries, labels, max_seq_length,
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

    for l_q, r_q, label in zip(l_queries, r_queries, labels):
        # convert to example objects
        inp_ex = processor._get_input_example(l_q, r_q)

        # convert to features
        inp_fe = featurization.convert_single_example(
            None, inp_ex, max_seq_length, tokenizer)

        yield inp_fe, label

def convert_to_tfrecords_helper(args):
    return convert_to_tfrecords(*args)

def convert_to_tfrecords(df_train_pairs, proc_number, max_seq_length, bert_pretrained_dir,
                         bert_model_type, queries_per_file, dataset,
                         train_tfrecord_dir):
    """Takes query pairs from `df` and stores them in tfrecord files"""
    l_queries = df_train_pairs['query']
    r_queries = df_train_pairs['query_compare']
    labels = df_train_pairs['y_class']
    filenames = []

    tf.logging.info(f'Proc_number: {proc_number}, {len(labels)} instances, processing tfrecords...')
    start = tt()
    input_feats_generator = input_feature_pair_generator(
        list(l_queries), list(r_queries), list(labels), max_seq_length,
        bert_pretrained_dir, bert_model_type)

    # iterate over given training pairs, step size is the `queries_per_file` limit
    for i in range(0, len(labels), queries_per_file):
        tf.logging.info(f'\t i = {i}, proc_number = {proc_number}, processing tfrecords...')
        inner_start = tt()
        tfrecord_filename = \
            '{}_train_pairs_{}_proc_{}_batch_{}.tfrecord'.format(
                dataset, max_seq_length, proc_number, i)
        tfrecord_save_path = os.path.join(train_tfrecord_dir,
                                          tfrecord_filename)

        with tf.io.TFRecordWriter(tfrecord_save_path) as writer:
            filenames.append(tfrecord_filename)

            ct = 0
            for inp_feat_pair, label in input_feats_generator:
                features = tf.train.Features(
                    feature={
                        'l_input_ids': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=inp_feat_pair.l_input_ids)),
                        'r_input_ids': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=inp_feat_pair.r_input_ids)),
                        'l_input_mask': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=inp_feat_pair.l_input_mask)),
                        'r_input_mask': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=inp_feat_pair.r_input_mask)),
                        'label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label]))})

                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

                ct += 1
                if ct == queries_per_file:
                    break

        tf.logging.info(f'\t i = {i}, proc_number = {proc_number}, done! time taken: {tt()-inner_start}')

    tf.logging.info(f'Proc_number: {proc_number}, done! Time taken: {tt() - start}')
    return filenames

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    TASK_DATA_DIR = f'./example_data/{FLAGS.dataset}/'
    BERT_MODEL = 'uncased_L-12_H-768_A-12'
    BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
    TRAIN_TFRECORD_DIR = os.path.join(TASK_DATA_DIR, 'train_tfrecords')
    TFRECORD_FILENAMES_PATH = os.path.join(TASK_DATA_DIR, 'tfrecord_filenames.txt')

    # load query pairs into memory
    tf.logging.info('Loading training pairs...')
    df_train_pairs = pickle.load(open(
        os.path.join(TASK_DATA_DIR, f'{FLAGS.dataset}_train_pairs.pkl'), 'rb'))
    tf.logging.info('Done loading training pairs!')

    train_df_split = np.array_split(df_train_pairs, FLAGS.num_cores)

    args_gen = ((train_df_split[i], i, FLAGS.max_seq_length,
                 BERT_PRETRAINED_DIR, BERT_MODEL, FLAGS.queries_per_file,
                 FLAGS.dataset, TRAIN_TFRECORD_DIR)
                for i in range(FLAGS.num_cores))
    os.makedirs(TRAIN_TFRECORD_DIR, exist_ok=True)

    with ThreadPool(FLAGS.num_cores) as pool:
        f_list_of_lists = pool.map(convert_to_tfrecords_helper, args_gen)

    # flatten
    filenames = [f for f_list in f_list_of_lists for f in f_list]

    with open(TFRECORD_FILENAMES_PATH, 'w') as f:
        for f_name in filenames:
           f.write(f_name + '\n')

if __name__ == "__main__" :
    tf.app.run()
