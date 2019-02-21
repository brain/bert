import tensorflow as tf
import modeling
import optimization
import run_classifier
import tokenization
import os
import datetime
import pickle

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

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(ex_index, example, max_seq_length,
                                         tokenizer)
        features.append(feature)
    return features

def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    """Converts single `InputExample` into a single `InputFeaturesPair`. This
    differs from `run_classifier.convert_single_example()` in that the two texts
    are not encoded into the same object"""
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = tokenizer.tokenize(example.text_b)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    if len(tokens_b) > max_seq_length - 2:
        tokens_b = tokens_b[0:(max_seq_length - 2)]

    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
    tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]

    input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
    input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

    input_mask_a = [1] * len(input_ids_a)
    input_mask_b = [1] * len(input_ids_b)

    while len(input_ids_a) < max_seq_length:
        input_ids_a.append(0)
        input_mask_a.append(0)

    while len(input_ids_b) < max_seq_length:
        input_ids_b.append(0)
        input_mask_b.append(0)

    assert len(input_ids_a) == max_seq_length
    assert len(input_mask_a) == max_seq_length
    assert len(input_ids_b) == max_seq_length
    assert len(input_mask_b) == max_seq_length

    feature = InputFeaturesPair(
        l_input_ids=input_ids_a,
        r_input_ids=input_ids_b,
        l_input_mask=input_mask_a,
        r_input_mask=input_mask_b,
        label_id=example.label)
    return feature


class InputFeaturesPair(object):
    """Single pair of features"""
    def __init__(self,
                 l_input_ids,
                 r_input_ids,
                 l_input_mask,
                 r_input_mask,
                 label_id):
        self.l_input_ids = l_input_ids
        self.r_input_ids = r_input_ids
        self.l_input_mask = l_input_mask
        self.r_input_mask = r_input_mask
        self.label_id = label_id

class FtmProcessor(run_classifier.DataProcessor):
    def get_labels(self):
        """See base class"""
        return [0, 1]

    def get_pred_examples(self, pair_cache_path):
        pairs_df = pickle.load(open(pair_cache_path, 'rb'))
        l_query_list = pairs_df['query'].values
        r_query_list = pairs_df['query_compare'].values
        labels = pairs_df['y_class'].values
        return self._create_examples(
            l_query_list, r_query_list, labels)

    def _create_examples(self, l_query_list, r_query_list, labels):
        """Creates example query pairs"""
        assert len(l_query_list) == len(r_query_list) \
            and len(labels) == len(r_query_list)

        examples = []
        for i, query_pair_info in enumerate(zip(l_query_list,
                                                r_query_list,
                                                labels)):
            l_query, r_query, label = query_pair_info
            guid = '%s' %(i)
            text_a = tokenization.convert_to_unicode(l_query)
            text_b = tokenization.convert_to_unicode(r_query)
            examples.append(
                run_classifier.InputExample(guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_l_input_ids = []
    all_r_input_ids = []
    all_l_input_masks = []
    all_r_input_masks = []
    all_label_ids = []

    for feature in features:
        all_l_input_ids.append(feature.l_input_ids)
        all_r_input_ids.append(feature.r_input_ids)
        all_l_input_masks.append(feature.l_input_mask)
        all_r_input_masks.append(feature.r_input_mask)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "l_input_ids":
                tf.constant(
                    all_l_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "r_input_ids":
                tf.constant(
                    all_r_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "l_input_mask":
                tf.constant(
                    all_l_input_masks,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "r_input_mask":
                tf.constant(
                    all_r_input_masks,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):

    def model_fn(features, labels, mode, params):

        # TODO: there's probably a few things i'm missing up here to make it work
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        l_input_ids = features["l_input_ids"]
        r_input_ids = features["r_input_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        sim_scores = create_model(
            bert_config, is_training, l_input_ids, r_input_ids,
            use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"sim_scores": sim_scores},
                scaffold_fn=scaffold_fn)
        else:
            tf.logging.error(f'mode `{mode}` not implemented yet')
        return output_spec

    return model_fn


def create_model(bert_config, is_training, l_input_ids, r_input_ids,
                 use_one_hot_embeddings):

    with tf.variable_scope('bert', reuse=tf.AUTO_REUSE) as bert_scope:
        l_model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=l_input_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            scope=bert_scope,
            reuse=tf.AUTO_REUSE)
        r_model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=r_input_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            scope=bert_scope,
            reuse=tf.AUTO_REUSE)

        l_output_layer = l_model.get_pooled_output()
        r_output_layer = r_model.get_pooled_output()

        with tf.variable_scope('similarity'):
            l1_norm = -tf.norm(l_output_layer - r_output_layer,
                               ord=1,
                               axis=-1,
                               name='abs_diff')
            sim_scores = tf.math.exp(l1_norm, name='exp')
            sim_scores = tf.clip_by_value(sim_scores, 1.0e-7, 1.0-1e-7,
                                          name='sim_scores')

        return sim_scores

def main(_):
    # This should be a minimum working example
    tf.logging.set_verbosity(tf.logging.INFO)
    print('Setting up model and TPU...')

    # Model and data path stuff
    BERT_MODEL = 'uncased_L-12_H-768_A-12'
    BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
    BUCKET = 'bert_output_bucket_mteoh'
    TASK = 'FTM'
    OUTPUT_DIR = 'gs://{}/bert/models/{}'.format(BUCKET, TASK)
    TASK_DATA_PATH = 'example_data/DATA_EXAMPLE_train_pairs.pkl'

    # Model Hyper Parameters
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 8
    PRED_BATCH_SIZE = 128
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    WARMUP_PROPORTION = 0.1
    MAX_SEQ_LENGTH = 128
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 1000
    ITERATIONS_PER_LOOP = 1000
    NUM_TPU_CORES = 8
    VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

    processor = FtmProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE,
                                           do_lower_case=DO_LOWER_CASE)

    # TODO: just prediction for now

    # run configuration for tpu if specified
    tpu_cluster_resolver = None
    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver('mteoh')
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=OUTPUT_DIR,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    # get prediction data
    pred_examples = processor.get_pred_examples(TASK_DATA_PATH)
    pred_features = convert_examples_to_features(
        pred_examples, MAX_SEQ_LENGTH, tokenizer)

    # model fn stuff
    model_fn = model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
        num_labels=len(label_list),
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=None,
        num_warmup_steps=None,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # create estimator
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        predict_batch_size=PRED_BATCH_SIZE)

    # predict input function
    predict_input_fn = input_fn_builder(
        features=pred_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    # get estimator to predict
    result = list(estimator.predict(input_fn=predict_input_fn))
    print(result)

if __name__ == "__main__" :
    flags.mark_flag_as_required("use_tpu")
    tf.app.run()