# How to use model optimization scripts
The optimizations in these scripts are based on [a blog post](https://medium.com/google-cloud/optimizing-tensorflow-models-for-serving-959080e9ddbf) from Google Cloud.

For each of the commands listed here, run them in the base directory of the repo.

## Save the model given the checkpoint bucket:
```
model_trial_name=FTM_BERT_DATA_008_tpu_trial_2
python -m model_processing.save_model --model_trial_name=${model_trial_name}
saved_models_base=models/${model_trial_name}
saved_model_dir=${saved_models_base}/$(ls ${saved_models_base} | tail -n 1)
echo $saved_model_dir
```

## View details of the model:
```
saved_model_cli show --dir=${saved_model_dir} --all
python -m model_processing.view_graphdef --saved_model_dir=${saved_model_dir}
```

## Freeze the graph:
```
python -m model_processing.freeze_graph --saved_model_dir=${saved_model_dir}
```

## Optimize graph and save it back as a model:
```
python -m model_processing.optimize_graph --saved_model_dir=${saved_model_dir} --saved_models_base=${saved_models_base}
```
