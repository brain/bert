import tensorflow as tf
import modeling
import optimization
import run_classifier
import tokenization
import os
import datetime
import pickle
import pandas as pd

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
flags.DEFINE_integer("predict_batch_size", 128, "Number of instances in a given batch.")

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
        r_input_mask=input_mask_b)
    return feature


class InputFeaturesPair(object):
    """Single pair of features"""
    def __init__(self,
                 l_input_ids,
                 r_input_ids,
                 l_input_mask,
                 r_input_mask):
        self.l_input_ids = l_input_ids
        self.r_input_ids = r_input_ids
        self.l_input_mask = l_input_mask
        self.r_input_mask = r_input_mask

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
            l_query_list, r_query_list)

    def _create_examples(self, l_query_list, r_query_list):
        """Creates example query pairs"""
        assert len(l_query_list) == len(r_query_list)

        examples = []
        for i, query_pair_info in enumerate(zip(l_query_list,
                                                r_query_list)):
            l_query, r_query = query_pair_info
            guid = '%s' %(i)
            text_a = tokenization.convert_to_unicode(l_query)
            text_b = tokenization.convert_to_unicode(r_query)
            examples.append(
                run_classifier.InputExample(guid, text_a=text_a, text_b=text_b))
        return examples


def input_fn_builder(features, seq_length, is_training, drop_remainder,
                     labels=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    assert labels is None or len(features) == len(labels), '''`features` and `labels` should be the same length'''

    all_l_input_ids = []
    all_r_input_ids = []
    all_l_input_masks = []
    all_r_input_masks = []

    for feature in features:
        all_l_input_ids.append(feature.l_input_ids)
        all_r_input_ids.append(feature.r_input_ids)
        all_l_input_masks.append(feature.l_input_mask)
        all_r_input_masks.append(feature.r_input_mask)
        # all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        feats_dict = {
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
                    dtype=tf.int32)}
        labels_input = None
        if labels:
            labels_input = tf.constant(
                labels,
                shape=[num_examples],
                dtype=tf.int32)
        tensor_slices_arg = (feats_dict, labels_input) \
            if labels_input is not None else feats_dict
        d = tf.data.Dataset.from_tensor_slices(tensor_slices_arg)

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
        l_input_mask = features["l_input_mask"]
        r_input_mask = features["r_input_mask"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, sim_scores, label_preds) = create_model(
            bert_config, is_training, l_input_ids, r_input_ids,
            l_input_mask, r_input_mask, labels, use_one_hot_embeddings)
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
                predictions={"sim_scores": sim_scores,
                             "label_preds": label_preds},
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, labels, label_preds):
                accuracy = tf.metrics.accuracy(
                    labels=labels, predictions=label_preds)
                loss = tf.metrics.mean(values=per_example_loss)
                return {
                    'eval_accuracy': accuracy,
                    'eval_loss': loss}

            eval_metrics = (metric_fn,
                            [per_example_loss, labels, label_preds])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            tf.logging.error(f'mode `{mode}` not implemented yet')
        return output_spec

    return model_fn


def create_model(bert_config, is_training, l_input_ids, r_input_ids,
                 l_input_mask, r_input_mask, labels, use_one_hot_embeddings):

    with tf.variable_scope('bert', reuse=tf.AUTO_REUSE) as bert_scope:
        l_model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=l_input_ids,
            input_mask=l_input_mask,
            use_one_hot_embeddings=use_one_hot_embeddings,
            scope=bert_scope,
            reuse=tf.AUTO_REUSE)
        r_model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=r_input_ids,
            input_mask=r_input_mask,
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
            label_preds = tf.math.round(sim_scores)


        with tf.variable_scope('loss'):
            logits = tf.math.add(tf.constant(1.0), -sim_scores)
            logits = tf.math.divide(sim_scores, logits)
            logits = tf.math.log(logits)
            per_example_loss, loss = None, None
            if labels is not None:
                per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(labels, dtype=tf.float32), logits=logits)
                loss = tf.reduce_mean(per_example_loss)

            return (loss, per_example_loss, logits, sim_scores, label_preds)


class SiameseBert(object):

    def __init__(self,
                 bert_model_type,
                 bert_pretrained_dir,
                 output_dir,
                 use_tpu=False,
                 tpu_name='mteoh',
                 learning_rate=2e-5,
                 num_train_epochs=3.0,
                 warmup_proportion=0.1,
                 train_batch_size=32,
                 eval_batch_size=8,
                 predict_batch_size=128,
                 max_seq_length=128,
                 save_checkpoints_steps=1000,
                 iterations_per_loop=1000,
                 num_tpu_cores=8):

        # set up relevant intermediate vars
        vocab_file = os.path.join(bert_pretrained_dir, 'vocab.txt')
        config_file = os.path.join(bert_pretrained_dir, 'bert_config.json')
        init_checkpoint = os.path.join(bert_pretrained_dir, 'bert_model.ckpt')
        do_lower_case = bert_model_type.startswith('uncased')

        # set up tokenizer
        self.processor = FtmProcessor()
        self.label_list = self.processor.get_labels()
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case)
        self.max_seq_length = max_seq_length

        # configs related to train, eval, predict
        # TODO: include the others that get passed into __init__()
        self.eval_batch_size = eval_batch_size

        # set up the RunConfig
        tpu_cluster_resolver = None
        if use_tpu:
            tpu_cluster_resolver = \
                tf.contrib.cluster_resolver.TPUClusterResolver(tpu_name)
        self.run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=output_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                num_shards=num_tpu_cores,
                per_host_input_for_training=\
                    tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

        # TODO: add something for training data here

        # make model function
        self.model_fn = model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(config_file),
            num_labels=len(self.label_list),
            init_checkpoint=init_checkpoint,
            learning_rate=learning_rate,
            # TODO: fill this in with something when we start training
            num_train_steps=None,
            num_warmup_steps=None,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_tpu)

        # make estimator
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=use_tpu,
            model_fn=self.model_fn,
            config=self.run_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            predict_batch_size=predict_batch_size)

    def predict_pairs(self, l_queries, r_queries):
        """Given pairs of queries, computes similarity scores for each pair.

        The ith query in `l_queries` is compared with the ith query in
        `r_queries` to produce the ith score in `result`.

        Args:
            l_queries (list or pd.Series): contains queries, to be compared
                with `r_queries`
            r_queries: same properties as `l_queries`

        Returns:
            sim_scores (pd.Series): ith score measures similarity of
                `l_queries[i]` and `r_queries[i]`
        """

        from time import time as tt
        # set up features
        print(f'preparing feats...')
        start = tt()
        pred_examples = self.processor._create_examples(
            list(l_queries),
            list(r_queries))
        pred_features = convert_examples_to_features(
            pred_examples, self.max_seq_length, self.tokenizer)
        print(f'done preparing feats. time taken: {tt()-start}')

        # make predict input function
        print(f'preparing input fn...')
        start = tt()
        input_fn = input_fn_builder(
            features=pred_features,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False)
        print(f'done preparing fn. time taken: {tt()-start}')

        # run prediction
        print(f'computing pred_results...')
        start = tt()
        pred_results = list(self.estimator.predict(input_fn=input_fn))
        pred_results = pd.DataFrame.from_records(pred_results)
        print(f'done computing pred_results. time taken: {tt()-start}')
        return pred_results

    def evaluate(self, l_queries, r_queries, labels):
        eval_examples = self.processor._create_examples(
            list(l_queries),
            list(r_queries))
        eval_features = convert_examples_to_features(
            eval_examples, self.max_seq_length, self.tokenizer)
        eval_steps = int(len(eval_examples) / self.eval_batch_size)
        eval_input_fn = input_fn_builder(
            features=eval_features,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False,
            labels=list(labels))
        eval_result = self.estimator.evaluate(
            input_fn=eval_input_fn,
            steps=eval_steps)
        return eval_result


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

    sb = SiameseBert(
        bert_model_type=BERT_MODEL,
        bert_pretrained_dir=BERT_PRETRAINED_DIR,
        output_dir=OUTPUT_DIR,
        use_tpu=FLAGS.use_tpu)
    df_pairs = pickle.load(open(TASK_DATA_PATH, 'rb'))
    l_queries = df_pairs['query']
    r_queries = df_pairs['query_compare']
    labels = df_pairs['y_class']

    # `pred_results` has the similarity scores of the query pairs
    from time import time as tt
    print(f'doing pred_results...')
    start = tt()
    pred_results = sb.predict_pairs(l_queries, r_queries)
    print(f'finished pred_results. time taken: {tt()-start}')

    # evaluate
    print(f'running evaluate...')
    start = tt()
    eval_result = sb.evaluate(l_queries, r_queries, labels)
    print(f'finished evaluating. time taken: {tt()-start}')


if __name__ == "__main__" :
    tf.app.run()
