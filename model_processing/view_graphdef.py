import tensorflow as tf
from model_processing import utils

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    'saved_model_dir', None,
    'Dir for saved model.')


def main(_):
    print('--- Describing graph... ---')
    utils.describe_graph(
        utils.get_graph_def_from_saved_model(FLAGS.saved_model_dir))
    print('--- Getting Size... ---')
    utils.get_size(FLAGS.saved_model_dir)


if __name__ == "__main__":
    tf.app.run()
