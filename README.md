# UPB SemEval 2020 Task 6

This repository contains the source code for the first and the second task of [DeftEval 2020 competition](https://competitions.codalab.org/competitions/22759), used by the University Politehnica of Bucharest (UPB) team to train and evaluate the models.

We fined-tuned frozen and non-frozen Transformer-based language models using the [HuggingFace](https://github.com/huggingface/transformers) framework, together with a multi-task model that jointly predicts the outputs for the second and the third task.

The code for each task, with additional details on how to use it, can be found in task1 and task2 directories.

<p align="center">
  <img src="https://raw.githubusercontent.com/avramandrei/DeftEval/master/resources/multitask_system.png">
</p>

## Installation

Make sure you have Python3 and PyTorch installed.

```
pip install -r requirements.txt
```

## Task1 Results

| Model | Valid-Prec | Valid-Rec | Valid-F1c | Test-Prec | Test-Rec | Test-F1 |
--------| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
Frozen-BERT | 76.0 | 75.1 | 75.5 | - | - | - |
Frozen RoBERTa | 74.1 | 74.4 | 74.3 | - | - | - |
Frozen SciBERT | 71.7 | 80.6 | 75.9 | - | - | - |
Frozen XLNet | 66.8 | 70.8 | 68.7 | - | - | - |
Frozen ALBERT | 77.2 | 69.3 | 73.0 | - | - | - |
Fine-tuned BERT | 78.4 | 84.1 | 81.2 | - | - | - |
Fine-tuned RoBERTa | **78.2** | **84.5** | **81.3** | **75.0** | **80.6** | **77.7** |
Fine-tuned SciBERT | 79.4 | 79.7 | 79.6 | - | - | - |
Fine-tuned XLNet | 75.5 | 85.2 | 80.1 | - | - | - |
Fine-tuned ALBERT | 69.5 | 85.9 | 76.8 | - | - | - |

## Task2 Results

| Model | Valid-Prec | Valid-Rec | Valid-F1c | Test-Prec | Test-Rec | Test-F1 |
--------| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
Frozen BERT+CRF | 27.1 | 39.8 | 26.1 | - | - | - |
Frozen RoBERTa+CRF | 32.7 | 27.7 | 22.6 | - | - | - |
Frozen SciBERT+CRF | 29.0 | 37.5 | 26.2 | - | - | - |
Frozen XLNet+CRF | 29.6 | 33.4 | 26.8 | - | - | - |
Frozen Multi-task | 4.0 | 8.9 | 8.0 | - | - | - |
Fine-tuned BERT+CRF | 47.9 | 51.7 | 45.6 | - | - | - |
Fine-tuned RoBERTa+CRF | **41.4** | **66.4** | **46.0** | **39.4** | **55.6** | **43.9** |
Fine-tuned SciBERT+CRF | 46.7 | 46.6 | 41.7 | - | - | - |
Fine-tuned XLNet+CRF | 33.0 | 58.5 | 39.2 | - | - | - |
Fine-tuned Multi-task | 25.7 | 25.2 | 25.5 | - | - | - |

## Cite
Please consider citing the following [paper](https://www.aclweb.org/anthology/2020.fnp-1.8.pdf) as a thank you to the authors: 
```
@inproceedings{ionescu-etal-2020-upb,
    title = "{UPB} at {F}in{C}ausal-2020, Tasks 1 {\&} 2: Causality Analysis in Financial Documents using Pretrained Language Models",
    author = "Ionescu, Marius  and
      Avram, Andrei-Marius  and
      Dima, George-Andrei  and
      Cercel, Dumitru-Clementin  and
      Dascalu, Mihai",
    booktitle = "Proceedings of the 1st Joint Workshop on Financial Narrative Processing and MultiLing Financial Summarisation",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "COLING",
    url = "https://www.aclweb.org/anthology/2020.fnp-1.8",
    pages = "55--59",
    abstract = "Financial causality detection is centered on identifying connections between different assets from financial news in order to improve trading strategies. FinCausal 2020 - Causality Identification in Financial Documents {--} is a competition targeting to boost results in financial causality by obtaining an explanation of how different individual events or chain of events interact and generate subsequent events in a financial environment. The competition is divided into two tasks: (a) a binary classification task for determining whether sentences are causal or not, and (b) a sequence labeling task aimed at identifying elements related to cause and effect. Various Transformer-based language models were fine-tuned for the first task and we obtained the second place in the competition with an F1-score of 97.55{\%} using an ensemble of five such language models. Subsequently, a BERT model was fine-tuned for the second task and a Conditional Random Field model was used on top of the generated language features; the system managed to identify the cause and effect relationships with an F1-score of 73.10{\%}. We open-sourced the code and made it available at: https://github.com/avramandrei/FinCausal2020.",
}
```
