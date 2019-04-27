# -*- coding: utf-8 -*-
from flask_restful import (
    Resource, reqparse
)
from operator import itemgetter
import time
import os
import tensorflow as tf
import pandas as pd
from training.siamese_modeling import create_model
from training.ftm_processor import FtmProcessor
from training.input_fns import input_fn_builder
from training import featurization
from google_bert import modeling
from google_bert import tokenization


BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
TRAINED_MODEL_BUCKET = 'bert_output_bucket_mteoh'
TASK = 'FTM_BERT_DATA_009_tpu_trial_1'
TRAINED_MODEL_DIR = 'gs://{}/bert/models/{}'.format(TRAINED_MODEL_BUCKET, TASK)
PREDICT_BATCH_SIZE = 256
RANDOM_PROJECTION_OUTPUT_DIM = 128
MAX_SEQ_LENGTH = 30
USE_TPU = False
SAVE_CHECKPOINT_STEPS = 1000


class SiameseBertSimilarityPlaceholder(Resource):
    def post(self):
        return dict(message='siamese bert similarity endpoint reached'), 200


class SiameseBertSimilarity(Resource):

    def __init__(self):
        super()

        # general BERT model related things
        vocab_file = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
        config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
        do_lower_case = BERT_MODEL.startswith('uncased')

        # checkpoint to initialize from
        checkpoint_path = tf.train.latest_checkpoint(TRAINED_MODEL_DIR)
        # checkpoint_path = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

        # featurization related tools
        self.processor = FtmProcessor()
        self.label_list = self.processor.get_labels()
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case)

        # estimator setup
        run_config = tf.estimator.RunConfig()
        model_fn = self._model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(config_file),
            init_checkpoint=checkpoint_path)
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

        # set up parser
        self.post_parser = reqparse.RequestParser()
        self.post_parser.add_argument("doc1", action='append', required=True)
        self.post_parser.add_argument("doc2", action='append', required=True)
        self.post_parser.add_argument("sort", required=False)

    def post(self):
        '''Calculates similarities between all combination of parameters

        "doc1": string or list of strings
        "doc2": string or list of strings
        "sort": whether to sort by score

        '''
        args = self.post_parser.parse_args()
        print(args)
        doc1 = args['doc1']
        doc2 = args['doc2']
        print(doc1, doc2)
        sort_opt = args.get('sort', True)

        if len(doc1) == 1:
            doc1 = doc1[0]

        results = []
        if isinstance(doc1, str):
            print(f'doc1: {doc1}; doc2: {doc2}')
            results = self._get_similarities(doc1, doc2, sort=sort_opt)
        else:
            for doc in doc1:
                result = self._get_similarities(doc, doc2, sort=sort_opt)
                results.append({"text": doc, "results": result})
        return results, 200

    def _get_similarities(self, doc, docs, sort=True):
        print(f'doc: {doc}; docs: {docs}')
        start = time.time()
        num_comparisons = len(docs)
        df_result = self._predict_pairs(
            [doc] * num_comparisons, docs)
        sim_scores = df_result.sim_scores.tolist()
        indices = df_result.index.values.tolist()
        results = [
            {'text': text, 'index': idx, 'score': score}
            for (text, idx, score) in zip(docs, indices, sim_scores)]
        print(f'done prediction: results: {results}')
        print(f'time taken: {time.time() - start}')
        if sort:
            results = sorted(results, key=itemgetter('score'), reverse=True)
        return results

    def _predict_pairs(self, l_queries, r_queries):
        pred_features = featurization.convert_examples_to_features(
            self.processor._create_examples(l_queries, r_queries),
            MAX_SEQ_LENGTH, self.tokenizer)
        input_fn = input_fn_builder(
            features=pred_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False,
            provided_batch_size=PREDICT_BATCH_SIZE)
        pred_results = list(self.estimator.predict(input_fn=input_fn))
        pred_results = pd.DataFrame.from_records(pred_results)

        return pred_results

    # TODO: is there a way we can have just one model fn for both training and
    # this API?
    def _model_fn_builder(self, bert_config, init_checkpoint):

        def model_fn(features, labels, mode, params):

            l_input_ids = features["l_input_ids"]
            r_input_ids = features["r_input_ids"]
            l_input_mask = features["l_input_mask"]
            r_input_mask = features["r_input_mask"]

            (total_loss, per_example_loss, logits, sim_scores, label_preds,
                merged_summaries) = create_model(
                bert_config, False, l_input_ids, r_input_ids,
                l_input_mask, r_input_mask, labels, True,
                RANDOM_PROJECTION_OUTPUT_DIM, False, False)

            # initialize variables
            print(f'---- initializing from: {init_checkpoint}')
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            assignment_map, initialized_variable_names = \
                modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            with tf.name_scope('assignments'):
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            # output specfication
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"sim_scores": sim_scores,
                             "label_preds": label_preds})

            return output_spec

        return model_fn
