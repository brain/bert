import pickle


def load_pair_data(df_path, l_col_name='query', r_col_name='query_name',
                   y_col_name='y_class'):
    """Loads pair data for ftm training

    Arguments:
        df_path (str): pickle path for DataFrame containing pair data
        l_col_name (str): column name for left queries
        r_col_name (str): column name for right queries
        y_col_name (str): column name for y_labels

    """

    df_pairs = pickle.load(open(df_path, 'rb'))

    l_queries = df_pairs['query']
    r_queries = df_pairs['query_compare']
    labels = df_pairs['y_class']

    return l_queries, r_queries, labels
