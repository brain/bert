# -*- coding: utf-8 -*-
from flask_restful import (
    Resource, reqparse
)
from training.siamese_bert import SiameseBert
from operator import itemgetter
import time

class SiameseBertSimilarityPlaceholder(Resource):
    def post(self):
        return dict(message='siamese bert similarity endpoint reached'), 200


class SiameseBertSimilarity(Resource):
    BERT_MODEL = 'uncased_L-12_H-768_A-12'
    BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
    OUTPUT_BUCKET = 'bert_output_bucket_mteoh'
    BERT_TFRECORD_BUCKET = 'gs://mteoh_siamese_bert_data'
    TASK = 'FTM_BERT_DATA_009_tpu_trial_1'
    OUTPUT_DIR = 'gs://{}/bert/models/{}'.format(OUTPUT_BUCKET, TASK)
    PREDICT_BATCH_SIZE = 256
    MAX_SEQ_LENGTH = 30
    USE_TPU = False

    sb = SiameseBert(
        bert_model_type=BERT_MODEL,
        bert_pretrained_dir=BERT_PRETRAINED_DIR,
        output_dir=OUTPUT_DIR,
        predict_batch_size=PREDICT_BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_tpu=USE_TPU)

    post_parser = reqparse.RequestParser()
    post_parser.add_argument("doc1", action='append', required=True)
    post_parser.add_argument("doc2", action='append', required=True)
    post_parser.add_argument("sort", required=False)

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
        df_result = self.sb.predict_pairs(
            [doc] * num_comparisons,
            docs, drop_remainder=False)
        sim_scores = df_result.sim_scores.tolist()
        indices = df_result.index.values.tolist()
        results = [{'text': text, 'index': idx, 'score': score}
            for (text, idx, score) in zip(docs, indices, sim_scores)]
        print(f'done prediction: results: {results}')
        print(f'time taken: {time.time() - start}')
        if sort:
            results = sorted(results, key=itemgetter('score'), reverse=True)
        return results
