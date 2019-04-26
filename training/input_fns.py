import tensorflow as tf


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
