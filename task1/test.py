from transformers import *
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaForSequenceClassification.from_pretrained("roberta-base")

input_ids = torch.tensor([tokenizer.encode("Here is, some text to encode", add_special_tokens=True)])

with torch.no_grad():
    print(tokenizer.encode("government ’s", add_special_tokens=True))
    print(tokenizer.encode("government’s", add_special_tokens=True))
