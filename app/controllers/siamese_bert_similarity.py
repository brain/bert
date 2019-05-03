# -*- coding: utf-8 -*-
from flask_restful import (
    Resource, reqparse
)
from operator import itemgetter
import time
import os
import pandas as pd
import numpy as np
from tensorflow.contrib import predictor
from pathlib import Path
from training.ftm_processor import FtmProcessor
from training import featurization
from google_bert import tokenization
import model_configs


# SAVE_MODEL_BASE_DIR = 'gs://mteoh_bert_models/models/' + model_configs.TRAINED_MODEL_ID
SAVE_MODEL_BASE_DIR = 'models/' + model_configs.TRAINED_MODEL_ID


class SiameseBertSimilarityPlaceholder(Resource):
    def post(self):
        return dict(message='siamese bert similarity endpoint reached'), 200


class SiameseBertSimilarity(Resource):

    model_save_dirs = [_dir for _dir in Path(SAVE_MODEL_BASE_DIR).iterdir()
                       if _dir.is_dir() and 'temp' not in str(_dir)]
    model_save_dir = str(sorted(model_save_dirs)[-1])
    # model_save_dir = SAVE_MODEL_BASE_DIR + '/optimized'
    print(f'--- Loading the model saved at: {model_save_dir}')

    predict_fn = predictor.from_saved_model(model_save_dir)
    print(f'--- Done loading the model!')

    print('--- computing dummy input... ---')
    # run in dummy input since model has some overhead on the first prediction
    dummy_val = predict_fn({
        'l_input_ids': [[0] * model_configs.MAX_SEQ_LENGTH],
        'r_input_ids': [[0] * model_configs.MAX_SEQ_LENGTH],
        'l_input_mask': [[0] * model_configs.MAX_SEQ_LENGTH],
        'r_input_mask': [[0] * model_configs.MAX_SEQ_LENGTH]})
    print('--- done computing dummy input... ---')

    # general BERT model related things
    vocab_file = os.path.join(model_configs.BERT_PRETRAINED_DIR, 'vocab.txt')
    config_file = os.path.join(model_configs.BERT_PRETRAINED_DIR, 'bert_config.json')
    do_lower_case = model_configs.BERT_MODEL_TYPE.startswith('uncased')

    # featurization related tools
    processor = FtmProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=do_lower_case)

    # set up parser
    post_parser = reqparse.RequestParser()
    post_parser.add_argument("doc1", action='append', required=True)
    post_parser.add_argument("doc2", action='append', required=True)
    post_parser.add_argument("sort", required=False)

    print('----- API ready to serve!  ------')

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

        print('----- POST request: Starting similarities... ------')

        if len(doc1) == 1:
            doc1 = doc1[0]

        results = []
        if isinstance(doc1, str):
            results = self._get_similarities(doc1, doc2, sort=sort_opt)
        else:
            for doc in doc1:
                result = self._get_similarities(doc, doc2, sort=sort_opt)
                results.append({"text": doc, "results": result})

        print('----- POST request: done similarities!  ------')
        return results, 200

    def _get_similarities(self, doc, docs, sort=True):
        # print(f'doc: {doc}; docs: {docs}')
        start = time.time()
        results = []
        num_batches = len(docs) // model_configs.PREDICT_BATCH_SIZE + 1
        print(f'len(docs) = {len(docs)}')
        print(f'num_batches = {num_batches}')
        batches = np.array_split(docs, num_batches)
        for doc_batch in batches:
            num_comparisons = len(doc_batch)
            df_result = self._predict_pairs(
                [doc] * num_comparisons, doc_batch)
            sim_scores = df_result.sim_scores.tolist()
            indices = df_result.index.values.tolist()
            results += [
                {'text': text, 'index': idx, 'score': score}
                for (text, idx, score) in zip(docs, indices, sim_scores)]

        print(f'len(results) = {len(results)}')
        # print(f'done prediction: results: {results}')
        print(f'time taken: {time.time() - start}')
        if sort:
            results = sorted(results, key=itemgetter('score'), reverse=True)
        return results

    def _predict_pairs(self, l_queries, r_queries):
        pred_features = featurization.convert_examples_to_features(
            self.processor._create_examples(l_queries, r_queries),
            model_configs.MAX_SEQ_LENGTH, self.tokenizer)

        print('--- Prediction: Starting... ---')
        pred_results = self.predict_fn({
            'l_input_ids': list(map(lambda f: f.l_input_ids, pred_features)),
            'r_input_ids': list(map(lambda f: f.r_input_ids, pred_features)),
            'l_input_mask': list(map(lambda f: f.l_input_mask, pred_features)),
            'r_input_mask': list(map(lambda f: f.r_input_mask, pred_features))})
        print('--- Prediction: Done! ---')
        print(pred_results)

        pred_results = pd.DataFrame.from_records(pred_results)

        return pred_results
