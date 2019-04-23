# (Siamese) BERT

For the README from the original BERT repo, see [this link](https://github.com/google-research/bert/blob/master/README.md).
This version of BERT is a siamese neural network implementation of function type matching. 
## How to train the model
In this guide, we'll use `dataset=BERT_DATA_000` for the running example, and the repo's base directory as the working directory unless otherwise specified.
### 1. Convert training/dev accuracy data to tfrecords
#### Training data
Training data are labelled pairs of query pairs. They should be in `./training/example_data/{dataset}/{dataset}_train_pairs.pkl`. For example, `./training/example_data/BERT_DATA_000/BERT_DATA_000_train_pairs.pkl`. Here are the required columns:
* `query` and `query_compare`: queries in the pair
* `y__class`: `1` if the two queries belong to the same action label, `0` otherwise.
To create the tfrecord files, run `create_train_tfrecord.py` from the base directory.
For example,
```
python -m training.create_train_tfrecord --dataset=BERT_DATA_000 --max_seq_length=30 --queries_per_file=2500 --num_cores=1
```
This will also generate a file called `tfrecord_filenames.txt` which gives all the names of the tfrecord files generated. (It's more efficient to store the training pairs across many files.)
#### Dev accuracy evaluation data
Dev accuracy requires two sets of data: the validation data (taken from `./training/example_data/{dataset}/{dataset}_{validation_suffix}.pkl`), and the DAr data (taken from `./training/example_data/{dataset}/{dataset}_{DAr_suffix}.pkl`). For example, `BERT_DATA_000_vdn.pkl` for the validation data, and `BERT_DATA_000_DAr.pkl` for the DAr data.
Both files should be pickled DataFrames with columns:
* `query`: text representing the query
* `id`: label for the action
To generate the tfrecord file for dev accuracy evaluation, run `create_devacc_tfrecord.py` from the base directory. For example:
```
python -m training.create_devacc_tfrecord --dataset=BERT_DATA_000 --max_seq_length=30 --vdn_string=vdn --DAr_string=DAr
```
### 2. Upload these to gcloud bucket
For the example, run:
```
sh upload_tfrecord_files.sh BERT_DATA_000
```
### 3. Start TPU
For how to set up cloud TPU usage, read [this guide](https://brain-team.quip.com/tIQSA4cakDCK/WIP-Using-cloud-tpu-from-the-command-line).
Run: 
```
ctpu up --name={tpu_name}
```
e.g. `tpu_name=mteoh`
### 4. Upload files to TPU
From your local machine, run:
```
gcloud compute config-ssh
rsync -r -v --exclude-from 'exclude-list.txt' . ${tpu_name}.us-central1-b.youtube-brain-ai-staging:~/bert
```
`exclude-list.txt` lets the `.pkl` files be uploaded but, none of the other data.
If you want to sync the files every time you make a local change, run:
```
while inotifywait -r -e modify,create,delete . ; do
    rsync -r -v --exclude-from 'exclude-list-all-data.txt' . ${tpu_name}.us-central1-b.youtube-brain-ai-staging:~/bert
done
```
`exclude-list-all-data.txt` prevents all data from being synced to speed things up.
### 5. Set up (Python) docker container
In the TPU machine, run:
```
cd ~/bert; docker run -it -e TPU_NAME=${TPU_NAME} -v${PWD}:/app -w/app python:3.6 bash
```
This starts a docker container that uses Python 3.6.
In the docker container, run:
```
pip install --upgrade google-api-python-client; pip install --upgrade oauth2client; pip install -r requirements.txt
```
This installs all the necessary requirements.
### 6. Run training script
Make sure you're in the container. Use the `train_ftm.py` script for training. For example:
```
export tpu_str=tpu_trial_testing; export dataset=BERT_DATA_000;
python -m training.train_ftm \
--task=FTM_${dataset}_${tpu_str} --dataset=${dataset} \
--use_tpu --num_train_epochs=30 --train_batch_size=2048 --eval_batch_size=2048 \
--predict_batch_size=2048 --max_seq_length=30 --tpu_name=${TPU_NAME} \
--train_epoch_increment=1 --dev_acc_epoch_period=1 --use_train_tfrecords --compute_dev_accuracy \
--extended_compute_dev_acc --sum_loss 
```
Once that's done, exit the docker container and go into the `bert` directory. Run `upload_devaccs.sh`. For example:
```
sh upload_devaccs.sh FTM_BERT_DATA_000_tpu_trial_testing
```
### 7. Download dev accuracy results
On your local machine, go to `./bert` directory, run `download_devaccs.sh`. For example:
```
sh download_devaccs.sh FTM_BERT_DATA_000_tpu_trial_testing
```

