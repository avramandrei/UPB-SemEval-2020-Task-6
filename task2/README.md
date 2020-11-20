This tutorial presents how to train and evaluate a modeul using our code for the 1st task of DeftEval competition.

Firstly, download the Deft corpus from [here](https://github.com/adobe-research/deft_corpus) and install the project [requirements](https://github.com/avramandrei/UPB-at-SemEval-2020-Task-6-Pretrained-Language-Models-for-DefinitionExtraction/blob/master/requirements.txt).

Then start training using the `train.py` script.

```
python3 train.py [-h] [--hidden_size HIDDEN_SIZE] [--device DEVICE] [--batch_size BATCH_SIZE] 
                 lang_model train_data dev_data fine_tune
```

To generate a file in submission's format run the `generate_evalfile.py` script.

```
 python3 generate_evalfile.py [-h] [--device DEVICE] model_path lang_model fine_tune input_path output_path
```
