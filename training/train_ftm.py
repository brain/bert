import tensorflow as tf
import pickle
import os
import re
import random
from training import train_utils
from training.siamese_bert import SiameseBert

flags = tf.flags
FLAGS = flags.FLAGS


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


# Flags for this script
# Remove the flags from `run_classifier` and instead use from this file
del_all_flags(tf.flags.FLAGS)

flags.DEFINE_bool(
    "use_tpu", False,
    "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer(
    "train_batch_size", 32,
    "Batch size to use for training.")

flags.DEFINE_integer(
    "predict_batch_size", 32,
    "Batch size to use for prediction.")

flags.DEFINE_integer(
    "eval_batch_size", 8,
    "Batch size to use for evaluation.")

flags.DEFINE_bool(
    "use_debug", False,
    "Whether to use tf debugger.")

flags.DEFINE_string(
    "task", "temp_task",
    "Task name & output subdir name.")

flags.DEFINE_integer(
    "which_gpu", None,
    "Which specific GPU to use.")

flags.DEFINE_integer(
    "num_train_epochs", 20,
    "Number of epochs to train over.")

flags.DEFINE_bool(
    "feedforward_logging", False,
    "Whether to log metrics in feedforward computation.")

flags.DEFINE_bool(
    "optimizer_logging", False,
    "Whether to log metrics in the optimizer.")

flags.DEFINE_string(
    "dataset", None,
    "Data ID to use for this experiment.")

flags.DEFINE_integer(
    "max_seq_length", 50,
    "Max token count for a query.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Number of TPU cores to use.")

flags.DEFINE_integer(
    "repeat_factor", 128,
    "Number of times to repeat query pairs.")

flags.DEFINE_string(
    "tpu_name", "mteoh",
    "Name of the tpu to use.")

flags.DEFINE_float(
    "learning_rate", 2e-5,
    "The initial learning rate for Adam.")

flags.DEFINE_integer(
    "train_epoch_increment", 500,
    "Number of epochs to increment model training.")

flags.DEFINE_bool(
    "compute_dev_accuracy", False,
    "Whether or not to compute dev accuracy.")

flags.DEFINE_bool(
    "use_train_tfrecords", False,
    "Whether to use tfrecords file for training data.")

flags.DEFINE_integer(
    "dev_acc_epoch_period", 2,
    "How many iterations in the epoch loop to wait before doing dev accuracy.")

flags.DEFINE_bool(
    "extended_compute_dev_acc", False,
    "Whether to use extended dev accuracy computations")

flags.DEFINE_bool(
    "do_preds", False,
    "Whether to do preds")

flags.DEFINE_bool(
    "do_eval", False,
    "Whether to evaluate pairwise accuracy and loss.")

flags.DEFINE_bool(
    "sum_loss", False,
    "Whether to sum (True) loss or average it (False).")

flags.DEFINE_string(
    "bert_model", 'uncased_L-12_H-768_A-12',
    "The name of the BERT model.")

flags.DEFINE_string(
    "bert_pretrained_base_dir", 'gs://cloud-tpu-checkpoints/bert/',
    "Base directory of pretrained BERT model.")

flags.DEFINE_bool(
    "initialize_weights", True,
    "Whether to initialize weights to ckpt.")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    BERT_MODEL = FLAGS.bert_model   # 'uncased_L-12_H-768_A-12'
    BERT_PRETRAINED_DIR = FLAGS.bert_pretrained_base_dir + BERT_MODEL   # 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
    OUTPUT_BUCKET = 'bert_output_bucket_mteoh'
    BERT_TFRECORD_BUCKET = 'gs://mteoh_siamese_bert_data'
    TASK = FLAGS.task
    OUTPUT_DIR = 'gs://{}/bert/models/{}'.format(OUTPUT_BUCKET, TASK)
    MAIN_DIR = os.path.dirname(__file__)
    TASK_DATA_DIR = os.path.join(MAIN_DIR, f'example_data/{FLAGS.dataset}/')
    DEV_ACC_DIR = os.path.join(MAIN_DIR, f'dev_accs/{TASK}')
    PREDS_DIR = os.path.join(MAIN_DIR, f'pred_results/{TASK}')
    RANDOM_SAMPLE_SEED = 0
    SAMPLE_SIZE = 2048
    random.seed(RANDOM_SAMPLE_SEED)

    if FLAGS.which_gpu is not None:
        tf.logging.info(f'using GPU: {FLAGS.which_gpu}')
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{FLAGS.which_gpu}"

    if FLAGS.do_preds:
        os.makedirs(PREDS_DIR, exist_ok=True)

    # all things re preparing data
    tf.logging.info('Loading training pairs...')
    l_queries, r_queries, labels = train_utils.load_pair_data(
        os.path.join(TASK_DATA_DIR, f'{FLAGS.dataset}_train_pairs.pkl'))
    tf.logging.info('Done loading training pairs!')

    int_idx_sample = random.sample(range(len(l_queries)), k=SAMPLE_SIZE)

    l_queries_sample = l_queries.iloc[int_idx_sample]
    r_queries_sample = r_queries.iloc[int_idx_sample]
    labels_sample = labels.iloc[int_idx_sample]

    df_vdn, df_ref = None, None
    if FLAGS.compute_dev_accuracy:
        df_vdn = pickle.load(open(
            os.path.join(TASK_DATA_DIR, f'{FLAGS.dataset}_vdn.pkl'), 'rb'))
        df_ref = pickle.load(open(
            os.path.join(TASK_DATA_DIR, f'{FLAGS.dataset}_DAr.pkl'), 'rb'))
        os.makedirs(DEV_ACC_DIR, exist_ok=True)

    df_vdn_disj, df_ref_large = None, None
    vdn_disj_exists, ref_large_exists = False, False
    if FLAGS.extended_compute_dev_acc:
        vdn_disj_path = os.path.join(TASK_DATA_DIR, f'{FLAGS.dataset}_vdn_disjoint.pkl')
        ref_large_path = os.path.join(TASK_DATA_DIR, f'{FLAGS.dataset}_DAr_large.pkl')
        if os.path.exists(vdn_disj_path):
            df_vdn_disj = pickle.load(open(vdn_disj_path, 'rb'))
            vdn_disj_exists = True
        if os.path.exists(ref_large_path):
            df_ref_large = pickle.load(open(ref_large_path, 'rb'))
            ref_large_exists = True

    if FLAGS.do_eval:
        sb = SiameseBert(
            bert_model_type=BERT_MODEL,
            bert_pretrained_dir=BERT_PRETRAINED_DIR,
            output_dir=OUTPUT_DIR,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            max_seq_length=FLAGS.max_seq_length,
            use_tpu=FLAGS.use_tpu,
            tpu_name=FLAGS.tpu_name,
            dataset_name=FLAGS.dataset,
            learning_rate=FLAGS.learning_rate,
            num_train_epochs=None,
            use_debug=FLAGS.use_debug,
            feedforward_logging=FLAGS.feedforward_logging,
            optimizer_logging=FLAGS.optimizer_logging,
            num_tpu_cores=FLAGS.num_tpu_cores,
            sum_loss=FLAGS.sum_loss,
            initialize_weights=FLAGS.initialize_weights)

        tf.logging.info('****evaluate on dataset...')
        res = sb.evaluate(l_queries_sample, r_queries_sample, labels_sample)
        tf.logging.info(f'{res}')

    for targ_epoch_num in range(FLAGS.train_epoch_increment,
                                FLAGS.num_train_epochs + FLAGS.train_epoch_increment,
                                FLAGS.train_epoch_increment):
        # Train in increments of `FLAGS.train_epoch_increment`, ending at
        #   `FLAGS.num_train_epochs`

        # If we've already trained past the number of steps given by
        # `targ_epoch_num`, then skip this iteration of the loop
        latest_checkpoint_path = tf.train.latest_checkpoint(OUTPUT_DIR)
        if latest_checkpoint_path is None:
            prior_steps_completed = 0
        else:
            step_num_idx = re.match('.*ckpt-', latest_checkpoint_path).end()
            prior_steps_completed = int(latest_checkpoint_path[step_num_idx:])

        current_num_steps = int(
            len(labels) / FLAGS.train_batch_size * targ_epoch_num)

        if current_num_steps <= prior_steps_completed:
            tf.logging.info(f'Current num steps: {current_num_steps}; latest checkpoint steps: {prior_steps_completed}; skipping...')
            continue

        sb = SiameseBert(
            bert_model_type=BERT_MODEL,
            bert_pretrained_dir=BERT_PRETRAINED_DIR,
            output_dir=OUTPUT_DIR,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            max_seq_length=FLAGS.max_seq_length,
            use_tpu=FLAGS.use_tpu,
            tpu_name=FLAGS.tpu_name,
            dataset_name=FLAGS.dataset,
            num_train_epochs=targ_epoch_num,
            use_debug=FLAGS.use_debug,
            feedforward_logging=FLAGS.feedforward_logging,
            optimizer_logging=FLAGS.optimizer_logging,
            num_tpu_cores=FLAGS.num_tpu_cores,
            sum_loss=FLAGS.sum_loss,
            initialize_weights=FLAGS.initialize_weights)

        tf.logging.info('training on dataset...')
        if FLAGS.use_train_tfrecords:
            sb.train_with_tfrecords(
                len(labels), BERT_TFRECORD_BUCKET,
                os.path.join(TASK_DATA_DIR, 'tfrecord_filenames.txt'))
        else:
            sb.train(l_queries, r_queries, labels)
        tf.logging.info('done training')

        if FLAGS.do_eval:
            tf.logging.info(f'****targ_epoch_num = {targ_epoch_num}: evaluating again...')
            res = sb.evaluate(l_queries_sample, r_queries_sample, labels_sample)
            tf.logging.info(f'{res}')

        if FLAGS.do_preds:
            tf.logging.info(f'****targ_epoch_num = {targ_epoch_num}: doing preds...')
            res = sb.predict_pairs(l_queries_sample, r_queries_sample)
            pred_save_path = os.path.join(PREDS_DIR, f'preds_epoch_{targ_epoch_num}.pkl')
            pickle.dump(res, open(pred_save_path, 'wb'), -1)

        # do dev accuracy
        if FLAGS.compute_dev_accuracy and not (targ_epoch_num / FLAGS.train_epoch_increment) % FLAGS.dev_acc_epoch_period:
            tf.logging.info(f'****targ_epoch_num = {targ_epoch_num}: computing dev accuracy...')

            dev_acc_tfrecord_bucket = os.path.join(
                BERT_TFRECORD_BUCKET,
                f'{FLAGS.dataset}_devacc_pairs_vdn_DAr_{FLAGS.max_seq_length}.tfrecord')
            dev_acc_res = sb.dev_accuracy_tfrecord(df_vdn, df_ref, dev_acc_tfrecord_bucket)
            dev_acc_save_path = os.path.join(
                DEV_ACC_DIR, f'dev_accs_epoch_{targ_epoch_num}.pkl')
            pickle.dump(dev_acc_res, open(dev_acc_save_path, 'wb'), -1)

            if FLAGS.extended_compute_dev_acc:
                if ref_large_exists:
                    tf.logging.info(f'****targ_epoch_num = {targ_epoch_num}: dev accuracy with larger DAr...')

                    dev_acc_tfrecord_bucket = os.path.join(
                        BERT_TFRECORD_BUCKET,
                        f'{FLAGS.dataset}_devacc_pairs_vdn_DAr_large_{FLAGS.max_seq_length}.tfrecord')
                    dev_acc_res = sb.dev_accuracy_tfrecord(df_vdn, df_ref_large, dev_acc_tfrecord_bucket)
                    dev_acc_save_path = os.path.join(
                        DEV_ACC_DIR, f'dev_accs_epoch_{targ_epoch_num}_vdn_DAr_large.pkl')
                    pickle.dump(dev_acc_res, open(dev_acc_save_path, 'wb'), -1)

                if vdn_disj_exists:
                    tf.logging.info(f'****targ_epoch_num = {targ_epoch_num}: dev acc unseen test data, DAr...')

                    dev_acc_tfrecord_bucket = os.path.join(
                        BERT_TFRECORD_BUCKET,
                        f'{FLAGS.dataset}_devacc_pairs_vdn_disjoint_DAr_{FLAGS.max_seq_length}.tfrecord')
                    dev_acc_res = sb.dev_accuracy_tfrecord(df_vdn_disj, df_ref, dev_acc_tfrecord_bucket)
                    dev_acc_save_path = os.path.join(
                        DEV_ACC_DIR, f'dev_accs_epoch_{targ_epoch_num}_vdn_disjoint_DAr.pkl')
                    pickle.dump(dev_acc_res, open(dev_acc_save_path, 'wb'), -1)

                if vdn_disj_exists and ref_large_exists:
                    tf.logging.info(f'****targ_epoch_num = {targ_epoch_num}: dev acc unseen test data, larger DAr...')

                    dev_acc_tfrecord_bucket = os.path.join(
                        BERT_TFRECORD_BUCKET,
                        f'{FLAGS.dataset}_devacc_pairs_vdn_disjoint_DAr_large_{FLAGS.max_seq_length}.tfrecord')
                    dev_acc_res = sb.dev_accuracy_tfrecord(df_vdn_disj, df_ref_large, dev_acc_tfrecord_bucket)
                    dev_acc_save_path = os.path.join(
                        DEV_ACC_DIR, f'dev_accs_epoch_{targ_epoch_num}_vdn_disjoint_DAr_large.pkl')
                    pickle.dump(dev_acc_res, open(dev_acc_save_path, 'wb'), -1)


if __name__ == "__main__":
    tf.app.run()
