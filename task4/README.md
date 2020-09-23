# BiLSTM-CRF for NER/POS

## Get Dataset
The provided dataset is a polished version in which we remove all "DOCSTART" sentences. To get the raw dataset and place into one destination, run the script in this folder.
```
sh download_dataset.sh <data_root>
```

## Implementation Details
The model is based on the paper [_"End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF"_](https://arxiv.org/pdf/1603.01354.pdf)

Chacter CNN doesn't help increase the accuracy. The set of parameters given in original paper cannot produce a satifying result, either. We tune both hyperparameters of both model and training for better performance. The choices of hyperparameters and corresponding performance are listed in the table below.

| Task | embed_size | hidden_size | lr | batch_size | dropout | val_acc | test_acc |
| ---: | ----: | ----: | ----: | ----: | ---: | ---: | --:|
| __POS__ | 300 | 300 | 0.005 | 32 | 0.2 | 94.20 | 93.02 |
| __NER__ | 300 | 300 | 0.005 | 64 | 0.2 | 98.02 | 97.16 |

## Usage
To train and evaluate the model, use the following command. The program will automatically save and test the model with the best performance.
You can also change default hyperparameter settings.
```
python --tag_type=pos --log_root=<path-to-store-log> --data_root=<path-to-store-data>
```
