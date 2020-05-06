import os
import pandas as pd
from transformers import RobertaTokenizer
import torch
import re
from torch.nn.utils.rnn import pad_sequence


def clean_data(df, tokenizer, fine_tune):
    df[0] = df[0].apply(lambda x: re.sub("\[ ?link ?\][a-z]?( \( [a-z] \))?", "<link>" if fine_tune else tokenizer.unk_token, x))

    df[0] = df[0].apply(lambda x: re.sub(r" ?https?:.+(\)|/|(\.pdf)|(\.PDF)|(\.html)|#| - U |aspx?|-[a-zA-z0-9]+|\.htm|\?.+)", "", x))
    df[0] = df[0].apply(lambda x: re.sub(r"www.+?( |\))", "", x))
    df[0] = df[0].apply(lambda x: x.replace(".  .", ".").replace(". .", ".").replace(", .", "."))

    df[0] = df[0].apply(lambda x: x.replace("“ ", "\"").replace(" ”", "\"").replace("’", "'").replace("‘", "'").replace(",", ",").replace("⋅", "*"))

    df[0] = df[0].apply(lambda x: re.sub(r" size 12.+}", "", x))
    df[0] = df[0].apply(lambda x: re.sub(r"5 \" MeV/\"c.+}", "", x))
    df[0] = df[0].apply(lambda x: re.sub(r" } { }", "", x))

    df[0] = df[0].apply(lambda x: re.sub(r"[^\s]+(\+|=|Δ|\*){1}[^\s]+", "<equation>" if fine_tune else tokenizer.unk_token, x))

    df[0] = df[0].apply(lambda x: re.sub(r"^ (\d+ . )?", "", x))

    df[0] = df[0].apply(lambda x: x.replace("do n't", "don't").replace("Do n't", "Don't"))

    df[0] = df[0].apply(lambda x: x.replace(" .", "."))
    df[0] = df[0].apply(lambda x: x.replace(" ,", ","))
    df[0] = df[0].apply(lambda x: x.replace(" ?", "?"))
    df[0] = df[0].apply(lambda x: x.replace(" - ", "-"))
    df[0] = df[0].apply(lambda x: x.replace("( ", "("))
    df[0] = df[0].apply(lambda x: x.replace(" )", ")"))
    df[0] = df[0].apply(lambda x: x.replace(" & ", "&"))
    df[0] = df[0].apply(lambda x: x.replace(" ;", ";"))
    df[0] = df[0].apply(lambda x: x.replace(" '", "'"))
    df[0] = df[0].apply(lambda x: x.replace(" :", ":"))
    df[0] = df[0].apply(lambda x: x.replace(" $", "$"))
    df[0] = df[0].apply(lambda x: x.replace(" %", "%"))
    df[0] = df[0].apply(lambda x: re.sub(r"(_ )+", "", x))
    df[0] = df[0].apply(lambda x: x.replace(",\"", "\""))
    
    return df


def process_data(data_path, tokenizer, device, train_data=False, fine_tune=False, batch_size=32):
    for i, filename in enumerate(os.listdir(data_path)):
        if i == 0:
            df = pd.read_csv(os.path.join(data_path, filename), header=None, sep="\t")
        else:
            df = df.append(pd.read_csv(os.path.join(data_path, filename), header=None, sep="\t"))

    if fine_tune:
        tokenizer.add_tokens(["<link>", "<equation>"])
    
    if train_data:
        true_df = df[df[1] == 1]
        false_df = df[df[1] == 0].sample(frac=0.75, random_state=13)
        df = true_df.append(false_df)

    df = df.sample(frac=1, random_state=13)

    df = clean_data(df, tokenizer, fine_tune)

    X, y = df.drop(columns=[1]).values, df.drop(columns=[0]).values

    mask = []
    token_list = []
    for i in range(X.shape[0]):
        tokens = torch.tensor(tokenizer.encode(X[i][0], add_special_tokens=True))
        token_list.append(tokens)
        mask.append(torch.ones_like(tokens))

    X = pad_sequence(token_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    mask = pad_sequence(mask, batch_first=True, padding_value=0).to(device)

    dataset = torch.utils.data.TensorDataset(X, y, mask)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def process_test_sentences(path, tokenizer, fine_tune, device):
    X_list = []

    for filename in os.listdir(path):
        df = pd.read_csv(os.path.join(path, filename), header=None, sep="\t")

        df = clean_data(df, tokenizer, fine_tune)[0]

        X = df.values

        tokens = []
        for i in range(X.shape[0]):
            tokens.append(torch.tensor(tokenizer.encode(X[i], add_special_tokens=True)))

        X = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

        X_list.append(torch.tensor(X))

    return X_list