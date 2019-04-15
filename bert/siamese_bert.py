import tensorflow as tf
import modeling
import optimization
import run_classifier
import tokenization
import os
import datetime
import pickle
import pandas as pd
import numpy as np
import itertools
from time import time as tt
from tqdm import tqdm
from tensorflow.python import debug as tf_debug

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in tqdm(enumerate(examples)):
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
        for i, query_pair_info in tqdm(enumerate(zip(l_query_list,
                                                r_query_list))):
            l_query, r_query = query_pair_info
            guid = '%s' %(i)
            text_a = tokenization.convert_to_unicode(l_query)
            text_b = tokenization.convert_to_unicode(r_query)
            examples.append(
                run_classifier.InputExample(guid, text_a=text_a, text_b=text_b))
        return examples

    def _get_input_example(self, l_query, r_query):
        text_a = tokenization.convert_to_unicode(l_query)
        text_b = tokenization.convert_to_unicode(r_query)

        return run_classifier.InputExample(None, text_a=text_a, text_b=text_b)


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
                    dtype=tf.int32,
                    name='l_input_ids'),
            "r_input_ids":
                tf.constant(
                    all_r_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32,
                    name='r_input_ids'),
            "l_input_mask":
                tf.constant(
                    all_l_input_masks,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32,
                    name='l_input_mask'),
            "r_input_mask":
                tf.constant(
                    all_r_input_masks,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32,
                    name='r_input_mask')}
        labels_input = None
        if labels:
            labels_input = tf.constant(
                labels,
                shape=[num_examples],
                dtype=tf.int32,
                name='labels')
        tensor_slices_arg = (feats_dict, labels_input) \
            if labels_input is not None else feats_dict
        d = tf.data.Dataset.from_tensor_slices(tensor_slices_arg)

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn

def input_fn_builder_tfrecords(is_training, drop_remainder, max_seq_length,
                               tfrecord_save_paths, pad_length=0):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # TODO: parameterize this better later on

        # TODO: very hacky to make backwards compatible; consider refactoring
        raw_dataset = tf.data.TFRecordDataset(tfrecord_save_paths)
        feature_description = {
            'l_input_ids': tf.FixedLenFeature([max_seq_length], tf.int64, default_value=None),
            'r_input_ids': tf.FixedLenFeature([max_seq_length], tf.int64, default_value=None),
            'l_input_mask': tf.FixedLenFeature([max_seq_length], tf.int64, default_value=None),
            'r_input_mask': tf.FixedLenFeature([max_seq_length], tf.int64, default_value=None)}
        if is_training:
            feature_description['label'] = tf.FixedLenFeature([], tf.int64, default_value=None)
        def _parse(example_proto):
            parsed_features = tf.parse_single_example(example_proto, feature_description)
            if is_training:
                return parsed_features, parsed_features['label']
            else:
                return parsed_features
        d = raw_dataset.map(_parse)

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        if pad_length and drop_remainder:
            # hacky padding for batching so nothing actually gets dropped
            d = d.concatenate(d.take(pad_length))

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


class SiameseBert(object):

    # TODO: check these args to see if we've `self.`'d the necessary ones
    def __init__(self,
                 bert_model_type,
                 bert_pretrained_dir,
                 output_dir,
                 use_tpu=False,
                 tpu_name='mteoh',
                 dataset_name=None,
                 learning_rate=2e-5,
                 num_train_epochs=3.0,
                 warmup_proportion=0.1,
                 train_batch_size=32,
                 eval_batch_size=8,
                 predict_batch_size=128,
                 max_seq_length=128,
                 save_checkpoints_steps=1000,
                 iterations_per_loop=1000,
                 num_tpu_cores=8,
                 use_debug=False,
                 feedforward_logging=False,
                 optimizer_logging=False,
                 random_projection_output_dim=128,
                 sum_loss=False):

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
        self.num_train_steps = None
        self.num_warmup_steps = None
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size

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

        # make model function
        self.model_fn = self.model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(config_file),
            num_labels=len(self.label_list),
            init_checkpoint=init_checkpoint,
            learning_rate=learning_rate,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_tpu)

        # make estimator
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=use_tpu,
            model_fn=self.model_fn,
            config=self.run_config,
            train_batch_size=self.train_batch_size,
            eval_batch_size=eval_batch_size,
            predict_batch_size=predict_batch_size)

        # other things
        self.use_debug = use_debug
        if use_tpu and (feedforward_logging or optimizer_logging):
            tf.logging.error('Cannot use `feedforward_logging` or `optimizer_logging` when on TPU.')
        self.feedforward_logging = feedforward_logging
        self.optimizer_logging = optimizer_logging
        self.dataset_name = dataset_name
        self.use_tpu = use_tpu

        self.random_projection_output_dim = random_projection_output_dim
        self.sum_loss = sum_loss

    def create_model(self, bert_config, is_training, l_input_ids, r_input_ids,
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
                l_output_layer = tf.layers.dense(
                    l_output_layer, self.random_projection_output_dim,
                    name='random_projection')
                r_output_layer = tf.layers.dense(
                    r_output_layer, self.random_projection_output_dim,
                    name='random_projection', reuse=True)

                l1_norm = -tf.norm(l_output_layer - r_output_layer,
                                   ord=1,
                                   axis=-1,
                                   name='abs_diff')
                sim_scores = tf.math.exp(l1_norm, name='exp')
                sim_scores = tf.clip_by_value(sim_scores, 1.0e-7, 1.0-1e-7,
                                              name='sim_scores')
                label_preds = tf.math.round(sim_scores, name='label_preds')


            with tf.variable_scope('loss'):
                logits = tf.math.add(tf.constant(1.0), -sim_scores)
                logits = tf.math.divide(sim_scores, logits)
                logits = tf.math.log(logits, name='logits')
                per_example_loss, loss = None, None
                if labels is not None:
                    per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.cast(labels, dtype=tf.float32), logits=logits,
                        name='per_example_loss')
                    if self.sum_loss:
                        loss = tf.reduce_sum(per_example_loss, name='total_loss')
                    else:
                        loss = tf.reduce_mean(per_example_loss, name='total_loss')

            merged_summaries = None

            if self.feedforward_logging:
                with tf.name_scope('summaries'):
                    tf.summary.histogram('sim_scores_hist', sim_scores)
                    if per_example_loss is not None:
                        tf.summary.histogram('per_example_loss_hist', per_example_loss)
                    if loss is not None:
                        tf.summary.scalar('average_loss', loss)

                    merged_summaries = tf.summary.merge_all(name='network_summaries')

            return (loss, per_example_loss, logits, sim_scores, label_preds,
                    merged_summaries)

    def model_fn_builder(self, bert_config, num_labels, init_checkpoint,
                         learning_rate, use_tpu, use_one_hot_embeddings):

        def model_fn(features, labels, mode, params):

            # TODO: there's probably a few things i'm missing up here to make it work
            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                pass
                # tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            l_input_ids = features["l_input_ids"]
            r_input_ids = features["r_input_ids"]
            l_input_mask = features["l_input_mask"]
            r_input_mask = features["r_input_mask"]

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, sim_scores, label_preds,
                merged_summaries) = self.create_model(
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
                    with tf.name_scope('assignments'):
                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                pass
                # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                #                 init_string)

            if mode == tf.estimator.ModeKeys.TRAIN:
                assert self.num_train_steps is not None and \
                    self.num_warmup_steps is not None, '''Please make sure that
                    `self.num_train_steps` and `self.num_warmup_steps` are
                    not None. They should be set based on the size of the
                    training data, in `self.train()`'''
                with tf.name_scope('optimization'):
                    train_op = optimization.create_optimizer(
                        total_loss, learning_rate, self.num_train_steps,
                        self.num_warmup_steps, use_tpu, self.optimizer_logging)
                    # TODO: fix summary logging since this appears in two places
                    # and doesn't seem to play nicely with TPU?
                    if self.optimizer_logging:
                        tf.summary.merge_all()

                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            elif mode == tf.estimator.ModeKeys.PREDICT:
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
                    precision = tf.metrics.precision(
                        labels=labels, predictions=label_preds)
                    recall = tf.metrics.recall(
                        labels=labels, predictions=label_preds)
                    return {
                        'eval_accuracy': accuracy,
                        'eval_loss': loss,
                        'eval_precision': precision,
                        'eval_recall': recall}

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

    def predict_single_pair(self, l_query, r_query):
        """Returns output score of one pair of queries"""

        return self.predict_pairs([l_query], [r_query])

    def predict_pairs(self, l_queries, r_queries, drop_remainder=True):
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

        # set up features
        tf.logging.info(f'Pair Predictions: preparing features...')
        start = tt()
        pred_examples = self.processor._create_examples(
            list(l_queries),
            list(r_queries))
        pred_features = convert_examples_to_features(
            pred_examples, self.max_seq_length, self.tokenizer)
        tf.logging.info(f'Pair Predictions: finished preparing features. Time taken: {tt()-start}')

        # make predict input function
        tf.logging.info(f'Pair Predictions: preparing input_fn...')
        start = tt()
        input_fn = input_fn_builder(
            features=pred_features,
            seq_length=self.max_seq_length,
            is_training=False,
            # TODO: do something about this since TPU does not play nice with
            # uneven batch sizes
            drop_remainder=drop_remainder)
            #drop_remainder=False)
        tf.logging.info(f'Pair Predictions: done preparing input_fn. time taken: {tt()-start}')

        # run prediction
        tf.logging.info(f'Pair Predictions: computing pred_results...')

        hooks = None
        if self.use_debug:
            hooks = [tf_debug.LocalCLIDebugHook()]

        start = tt()
        pred_results = list(self.estimator.predict(input_fn=input_fn, hooks=hooks))
        pred_results = pd.DataFrame.from_records(pred_results)
        tf.logging.info(f'Pair Predictions: done computing pred_results. time taken: {tt()-start}')
        return pred_results

    def predict_pairs_tfrecord(self, tfrecord_save_path, pad_length):
        tf.logging.info(f'Pair Predictions: preparing input_fn...')
        start = tt()
        input_fn = input_fn_builder_tfrecords(
            is_training=False,
            drop_remainder=self.use_tpu,
            max_seq_length=self.max_seq_length,
            tfrecord_save_paths=[tfrecord_save_path],
            pad_length=pad_length)
        tf.logging.info(f'Pair Predictions: done preparing input_fn. time taken: {tt()-start}')

        # run prediction
        tf.logging.info(f'Pair Predictions: computing pred_results...')

        hooks = None
        if self.use_debug:
            hooks = [tf_debug.LocalCLIDebugHook()]

        start = tt()
        pred_results = list(self.estimator.predict(input_fn=input_fn, hooks=hooks))
        pred_results = pd.DataFrame.from_records(pred_results)
        tf.logging.info(f'Pair Predictions: done computing pred_results. time taken: {tt()-start}')
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
            drop_remainder=True,
            # TODO: do something about this since TPU does not play nice with
            # uneven batch sizes
            # drop_remainder=False,
            labels=list(labels))
        tf.logging.info(f'Starting evaluation...')
        start = tt()
        eval_result = self.estimator.evaluate(
            input_fn=eval_input_fn,
            steps=eval_steps)
        tf.logging.info(f'Finished evaluation! Time taken: {tt() - start}')
        return eval_result

    def train_with_tfrecords(self, num_train_examples):
        # TODO: do we need to round up?
        self.num_train_steps = int(
            num_train_examples / self.train_batch_size * self.num_train_epochs)
        tf.logging.info(f'num_train_steps: {self.num_train_steps}')
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)

        #TODO: consider refactoring and parameterizing
        task_data_dir = f'gs://mteoh_siamese_bert_data/'
        tfrecord_filenames_path = f'./example_data/{self.dataset_name}/tfrecord_filenames.txt'
        raw_filenames = [line.rstrip('\n') for line in open(tfrecord_filenames_path)]
        tfrecord_save_paths = [
            os.path.join(task_data_dir, raw_filename)
            for raw_filename in raw_filenames]

        train_input_fn = input_fn_builder_tfrecords(
            is_training=True,
            drop_remainder=self.use_tpu,
            max_seq_length=self.max_seq_length,
            tfrecord_save_paths=tfrecord_save_paths)

        hooks = None
        if self.use_debug:
            hooks = [tf_debug.LocalCLIDebugHook()]

        # estimator.train
        tf.logging.info(f'Starting training...')
        start = tt()
        self.estimator.train(
            input_fn=train_input_fn,
            max_steps=self.num_train_steps,
            hooks=hooks)
        tf.logging.info(f'Training finished! Time take: {tt() - start}')

    def train(self, l_queries, r_queries, labels):

        # create train examples
        tf.logging.info(f'Preparing training features...')
        start = tt()

        # feats_path = './train_feats_cache_BERT_DATA_001.pkl'
        feats_path = f'./example_data/{self.dataset_name}/train_feats_cache_{self.dataset_name}.pkl'
        tf.logging.info(f'feats_path = {feats_path}')
        if os.path.exists(feats_path):
            tf.logging.info(f'Loading training features from: {feats_path}')
            train_features = pickle.load(open(feats_path, 'rb'))
        else:
            train_examples = self.processor._create_examples(
                list(l_queries),
                list(r_queries))
            # create train features
            train_features = convert_examples_to_features(
                train_examples, self.max_seq_length, self.tokenizer)
            pickle.dump(train_features, open(feats_path, 'wb'), -1)
        tf.logging.info(f'Done preparing training features. Time taken: {tt() - start}')

        # TODO: do we need to round up?
        self.num_train_steps = int(
            len(train_examples) / self.train_batch_size * self.num_train_epochs)
        tf.logging.info(f'num_train_steps: {self.num_train_steps}')
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)
        # create train input function
        train_input_fn = input_fn_builder(
            features=train_features,
            seq_length=self.max_seq_length,
            is_training=True,
            # TODO: do something about this since TPU does not play nice with
            # uneven batch sizes
            drop_remainder=True,
            #drop_remainder=False,
            labels=list(labels))

        hooks = None
        if self.use_debug:
            hooks = [tf_debug.LocalCLIDebugHook()]

        # estimator.train
        tf.logging.info(f'Starting training...')
        start = tt()
        self.estimator.train(
            input_fn=train_input_fn,
            max_steps=self.num_train_steps,
            hooks=hooks)
        tf.logging.info(f'Training finished! Time take: {tt() - start}')

    def _generate_query_pairs(self, input_q, ref_q):
        ordered_list_pairs = list(itertools.product(input_q, ref_q))
        # TODO: this is not very memory efficient, so we need a better way
        # to handle large numbers of pairs
        l_q, r_q = zip(*ordered_list_pairs)
        return list(l_q), list(r_q)

    def predict_labels_tfrecord(self, df_input, df_ref, tfrecord_save_path):
        num_pairs = len(df_input) * len(df_ref)
        batch_remainder = num_pairs % self.predict_batch_size
        pad_length = self.predict_batch_size - batch_remainder

        pred_res = self.predict_pairs_tfrecord(tfrecord_save_path, pad_length)

        sim_scores = pred_res.iloc[:num_pairs].sim_scores.values.reshape((len(df_input), -1))
        ref_q_top_idxs = np.argmax(sim_scores, axis=-1)
        ref_q_top_sim_scores = np.max(sim_scores, axis=-1)

        df_pred_labels = pd.DataFrame({'ref_q_top_idx': ref_q_top_idxs})
        df_pred_labels['label'] = df_pred_labels.ref_q_top_idx.apply(
            lambda idx: df_ref.iloc[idx]['id'])
        df_pred_labels['top_sim_score'] = ref_q_top_sim_scores
        df_pred_labels['ranked_indices'] = list(np.argsort(-sim_scores))

        return df_pred_labels


    def predict_labels(self, df_input, df_ref):
        l_q, r_q = self._generate_query_pairs(df_input['query'], df_ref['query'])
        num_pairs = len(df_input) * len(df_ref)

        assert len(l_q) == len(r_q) and len(l_q) == num_pairs
        # get number of queries to pad with
        batch_remainder = len(l_q) % self.predict_batch_size
        if batch_remainder:
            pad_length = self.predict_batch_size - batch_remainder
            l_q += [''] * pad_length
            r_q += [''] * pad_length
        pred_res = self.predict_pairs(l_q, r_q)
        sim_scores = pred_res.iloc[:num_pairs].sim_scores.values.reshape((len(df_input), -1))
        ref_q_top_idxs = np.argmax(sim_scores, axis=-1)
        ref_q_top_sim_scores = np.max(sim_scores, axis=-1)

        df_pred_labels = pd.DataFrame({'ref_q_top_idx': ref_q_top_idxs})
        df_pred_labels['label'] = df_pred_labels.ref_q_top_idx.apply(
            lambda idx: df_ref.iloc[idx]['id'])
        df_pred_labels['top_sim_score'] = ref_q_top_sim_scores
        df_pred_labels['ranked_indices'] = list(np.argsort(-sim_scores))

        return df_pred_labels

    def dev_accuracy(self, df_input, df_ref):
        tf.logging.info('Starting dev accuracy...')
        start = tt()
        # make the predictions
        df_pred_labels = self.predict_labels(df_input, df_ref)

        assert len(df_input) == len(df_pred_labels)

        # compute the fraction that are correct
        label_correctness = df_pred_labels['label'].reset_index(drop=True) \
            == df_input['id'].reset_index(drop=True)
        tf.logging.info(f'Finished computing dev accuracy. Time taken: {tt() - start}')
        return {'dev_accuracy': label_correctness.mean(),
                'df_pred_labels': df_pred_labels}

    def dev_accuracy_tfrecord(self, df_input, df_ref, tfrecord_save_path):
        tf.logging.info('Starting dev accuracy...')
        start = tt()
        # make the predictions
        df_pred_labels = self.predict_labels_tfrecord(
            df_input, df_ref, tfrecord_save_path)

        assert len(df_input) == len(df_pred_labels)

        # compute the fraction that are correct
        label_correctness = df_pred_labels['label'].reset_index(drop=True) \
            == df_input['id'].reset_index(drop=True)
        tf.logging.info(f'Finished computing dev accuracy. Time taken: {tt() - start}')
        return {'dev_accuracy': label_correctness.mean(),
                'df_pred_labels': df_pred_labels}
