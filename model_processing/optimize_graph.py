import tensorflow as tf
import os
from model_processing import utils


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    'saved_model_dir', None,
    'Dir for saved model.')

flags.DEFINE_string(
    'saved_models_base', None,
    'Base dir for saved model')


def main(_):
    # transforms = [
    #     # 'remove_nodes(op=Identity)',
    #     # 'merge_duplicate_nodes',
    #     'strip_unused_nodes',
    #     # 'fold_constants(ignore_errors=true)',
    #     'fold_batch_norms']
    print('--- Optimizing graph... ---')
    # utils.optimize_graph(FLAGS.saved_model_dir, 'frozen_model.pb', transforms, 'bert/similarity/sim_scores')
    utils.optimize_graph(FLAGS.saved_model_dir, 'frozen_model.pb', 'bert/similarity/sim_scores')

    print('--- Describing graph... ---')
    utils.describe_graph(utils.get_graph_def_from_file(
        os.path.join(FLAGS.saved_model_dir, 'optimized_model.pb')))
    print('--- Getting Size... ---')
    utils.get_size(FLAGS.saved_model_dir, 'optimized_model.pb')
    print('--- Converting to savemodel and saving... ---')
    utils.convert_graph_def_to_saved_model(
        os.path.join(FLAGS.saved_models_base, 'optimized'),
        os.path.join(FLAGS.saved_model_dir, 'optimized_model.pb'))


if __name__ == "__main__":
    tf.app.run()
