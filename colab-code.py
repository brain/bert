import tensorflow as tf
import modeling
import optimization
import run_classifier
import tokenization
import os
import datetime


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    print('Setting up model and TPU...')

    # Model and data path stuff
    BERT_MODEL = 'uncased_L-12_H-768_A-12'
    BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
    BUCKET = 'bert_output_bucket_mteoh'
    TASK = 'MRPC'
    OUTPUT_DIR = 'gs://{}/bert/models/{}'.format(BUCKET, TASK)
    TASK_DATA_DIR = 'glue_data/' + TASK
    #TASK_DATA_DIR = 'gs://mteoh_bert_glue_data/' + TASK

    # Model Hyper Parameters
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    WARMUP_PROPORTION = 0.1
    MAX_SEQ_LENGTH = 128
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 1000
    ITERATIONS_PER_LOOP = 1000
    NUM_TPU_CORES = 8
    VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

    processors = {
        "cola": run_classifier.ColaProcessor,
        "mnli": run_classifier.MnliProcessor,
        "mrpc": run_classifier.MrpcProcessor,
    }
    processor = processors[TASK.lower()]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

    #tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver('grpc://10.240.1.2:8470')
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver('mteoh')
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=OUTPUT_DIR,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    ## TODO: will need an FTM version of this
    train_examples = processor.get_train_examples(TASK_DATA_DIR)
    num_train_steps = int(
        len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    # TODO: will need an FTM version of this
    model_fn = run_classifier.model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
        num_labels=len(label_list),
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=True,
        use_one_hot_embeddings=True)

    # TODO: will need an FTM version of this
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE)

    print('Training the model...')
    # TODO: will need an FTM version of this
    train_features = run_classifier.convert_examples_to_features(
        train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    print('***** Started training at {} *****'.format(datetime.datetime.now()))
    print('  Num examples = {}'.format(len(train_examples)))
    print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
    tf.logging.info("  Num steps = %d", num_train_steps)
    # TODO: will need an FTM version of this
    train_input_fn = run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print('***** Finished training at {} *****'.format(datetime.datetime.now()))

    # Evaluate model
    eval_examples = processor.get_dev_examples(TASK_DATA_DIR)
    eval_features = run_classifier.convert_examples_to_features(
        eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

    print('***** Started evaluation at {} *****'.format(datetime.datetime.now()))
    print('  Num examples = {}'.format(len(eval_examples)))
    print('  Batch size = {}'.format(EVAL_BATCH_SIZE))

    eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
    eval_input_fn = run_classifier.input_fn_builder(
        features=eval_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=True)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
    output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print('  {} = {}'.format(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__" :
    tf.app.run()
