import tensorflow as tf
from google_bert import modeling


def create_model(bert_config, is_training, l_input_ids, r_input_ids,
                 l_input_mask, r_input_mask, labels, use_one_hot_embeddings,
                 random_projection_output_dim, sum_loss, feedforward_logging):

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
                l_output_layer, random_projection_output_dim,
                name='random_projection')
            r_output_layer = tf.layers.dense(
                r_output_layer, random_projection_output_dim,
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
                if sum_loss:
                    loss = tf.reduce_sum(per_example_loss, name='total_loss')
                else:
                    loss = tf.reduce_mean(per_example_loss, name='total_loss')

        merged_summaries = None

        if feedforward_logging:
            with tf.name_scope('summaries'):
                tf.summary.histogram('sim_scores_hist', sim_scores)
                if per_example_loss is not None:
                    tf.summary.histogram('per_example_loss_hist', per_example_loss)
                if loss is not None:
                    tf.summary.scalar('average_loss', loss)

                merged_summaries = tf.summary.merge_all(name='network_summaries')

        return (loss, per_example_loss, logits, sim_scores, label_preds,
                merged_summaries)


# TODO: find a better place to put this
# TODO: is there a way we can have just one model fn for both training and
# this API?
def inference_model_fn_builder(bert_config, init_checkpoint,
                               random_projection_output_dim):
    def model_fn(features, labels, mode, params):

        l_input_ids = features["l_input_ids"]
        r_input_ids = features["r_input_ids"]
        l_input_mask = features["l_input_mask"]
        r_input_mask = features["r_input_mask"]

        (total_loss, per_example_loss, logits, sim_scores, label_preds,
            merged_summaries) = create_model(
            bert_config, False, l_input_ids, r_input_ids,
            l_input_mask, r_input_mask, labels, True,
            random_projection_output_dim, False, False)

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
