import torch
from transformers import *
import argparse
import os


def cut_padding(x, y, mask, device, pad_id):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    stop_seq = seq_len

    stop_tensor = torch.ones(batch_size, dtype=torch.long).to(device) * pad_id

    for i in range(seq_len):
        if torch.equal(x[:, i], stop_tensor):
            stop_seq = i
            break

    return x[:, :stop_seq], y[:, :stop_seq], mask[:, :stop_seq]


def parse_lang_model(lang_model):
    if "roberta" in lang_model:
        model = RobertaModel.from_pretrained(lang_model)
        tokenizer = RobertaTokenizer.from_pretrained(lang_model)
    elif "scibert-base-cased" == lang_model:
        model = BertModel.from_pretrained(os.path.join("pretrained_models", "scibert-base-cased"))
        tokenizer = BertTokenizer.from_pretrained(os.path.join("pretrained_models", "scibert-base-cased"))
    elif "albert-base" in lang_model:
        model = AlbertModel.from_pretrained("albert-base-v2")
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    elif "xlnet" in lang_model:
        model = XLNetModel.from_pretrained(lang_model)
        tokenizer = XLNetTokenizer.from_pretrained(lang_model)
    elif "bert" in lang_model:
        model = BertModel.from_pretrained(lang_model)
        tokenizer = BertTokenizer.from_pretrained(lang_model)
    else:
        raise AttributeError("Incorrect language model")
        
    return model, tokenizer, 768 if "base" in lang_model else 1024


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')