import tensorflow as tf
import tensorflow_datasets
from transformers import *

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
data = tensorflow_datasets.load('glue/mrpc')
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')

print(train_dataset)


