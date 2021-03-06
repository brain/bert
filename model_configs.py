import os

# About the base BERT model
BERT_MODEL_TYPE = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL_TYPE

# About the trained Siamese BERT model
MAX_SEQ_LENGTH = 30
TRAINED_MODEL_ID = 'FTM_BERT_DATA_008_tpu_trial_2'
if os.getenv('ENVIRONMENT') == 'test':
    TRAINED_MODEL_ID = 'FTM_DATA_SANITY_CHECK_test_model'

# About the predictions
PREDICT_BATCH_SIZE = 2048
