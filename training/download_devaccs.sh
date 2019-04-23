mkdir ./dev_accs/${1}

gsutil -m cp gs://bert_output_bucket_mteoh/bert/models/${1}/dev_accs/* ./dev_accs/${1}/

