from google_bert import run_classifier
from google_bert import tokenization
from tqdm import tqdm

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
