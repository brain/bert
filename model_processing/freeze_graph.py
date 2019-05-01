import tensorflow as tf
import os
from model_processing import utils


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    'saved_model_dir', None,
    'Dir for saved model.')


def main(_):
    print('--- Freezing graph... ---')
    utils.freeze_model(FLAGS.saved_model_dir,
                       'bert/similarity/sim_scores',
                       'frozen_model.pb')
    print('--- Describing graph... ---')
    utils.describe_graph(utils.get_graph_def_from_file(
        os.path.join(FLAGS.saved_model_dir, 'frozen_model.pb')))
    print('--- Getting Size... ---')
    utils.get_size(FLAGS.saved_model_dir, 'frozen_model.pb')


if __name__ == "__main__":
    tf.app.run()
