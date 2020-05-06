import os
from transformers import RobertaTokenizer
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import torch
from sklearn.utils import shuffle
from collections import Counter

eval_ent_dict = {"O": 0,
                 "Term": 1,
                 "Definition": 2,
                 "Alias-Term": 3,
                 "Referential-Definition": 4,
                 "Referential-Term": 5,
                 "Qualifier": 6}

inv_eval_ent_dict = {0: "O",
                     1: "Term",
                     2: "Definition",
                     3: "Alias-Term",
                     4: "Referential-Definition",
                     5: "Referential-Term",
                     6: "Qualifier"}

REF_TERM_RESAMPLE = 32
REF_DEF_RESAMPLE = 16
ALIAS_TERM_RESAMPLE = 8
QUAL_RESAMPLE = 8
TERM_RESAMPLE = 2


def process_data(path, tokenizer, device, is_dev=False, use_scibert=False):
    X = []
    y = []
    ct = Counter()

    for filename in os.listdir(path):
        sentence = ""
        entities = []
        words = []

        with open(os.path.join(path, filename), "r", encoding="utf-8") as file:
            for line in file:
                if line is not "\n":
                    tokens = line.split()
                    word, entity = tokens[0], tokens[4] if tokens[4] == "O" else tokens[4][2:]

                    ct[entity] += 1

                    # preprocessing
                    word = word.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")

                    greek_alphabet = 'αβγδεζηθικλμνξοπρςστυφχψω'
                    greek_alphabet += greek_alphabet.upper()

                    for letter in greek_alphabet:
                        word = word.replace(letter, tokenizer.unk_token)

                    if use_scibert:
                        word = word.lower()
                        word = word.replace("á", "a").replace("ü", "u").replace("í", "i").\
                            replace("ê", "e").replace("é", "e").replace("ó", "o")

                    entities.append(entity)
                    words.append(word)

                    sentence += word + " "
                else:
                    if sentence == "":
                        continue

                    #print(entities)
                    tokens = tokenizer.encode(sentence, add_special_tokens=True)

                    new_entities = ['O']
                    #print(len(tokens), len(entities))

                    i = 0
                    token_i = 1

                    while i < len(words):
                        token_word = tokenizer.decode([tokens[token_i]]).replace(" ", "").replace("#", "")
                        if use_scibert:
                            token_word = token_word.lower()

                        print(token_word, token_i, words[i], entities[i], i)

                        if token_word == words[i]:
                            new_entities.append(entities[i])
                        else:
                            if len(token_word) < len(words[i]) and token_word in words[i]:
                                print("TOKEN IN WORD")
                                j = token_i
                                while j < len(tokens):
                                    token_word = tokenizer.decode([tokens[j]]).replace(" ", "").replace("#", "")
                                    if use_scibert:
                                        token_word = token_word.lower()

                                    if token_word in words[i]:

                                        new_entities.append(entities[i])
                                        words[i] = words[i][len(token_word):]

                                        print("token in word: {}  {}  {}  {}".format(token_word, words[i], entities[i], j))
                                    else:
                                        print(token_word, words[i])
                                        print("BIG ERROR")
                                        break
                                    j += 1

                                token_i = j - 1
                            elif len(token_word) >= len(words[i]) and words[i] in token_word:
                                print("WORD IN TOKEN")
                                j = i

                                while j < len(words):
                                    if words[j] in token_word:
                                        token_word = token_word[len(words[j]):]
                                        print("word in token: {}, {}, {}, {}".format(token_word, words[j], entities[i], j))
                                    else:
                                        break

                                    j += 1

                                i = j - 1

                                new_entities.append(entities[i])
                            else:
                                print("Unrecognized word")
                                new_entities.append(entities[i])


                        token_i += 1
                        i += 1

                    new_entities.append("O")
                    print(len(new_entities), len(tokens))
                    assert len(new_entities) == len(tokens)

                    X.append(torch.tensor(tokens))
                    y.append(torch.tensor([eval_ent_dict[ent] if ent in eval_ent_dict else 0 for ent in new_entities]))

                    if not is_dev:
                        resample_count = 0
                        # oversample if there are under-represented samples
                        if eval_ent_dict["Referential-Term"] in y[-1]:
                            resample_count = REF_TERM_RESAMPLE
                        elif eval_ent_dict["Referential-Definition"] in y[-1]:
                            resample_count = REF_DEF_RESAMPLE
                        elif eval_ent_dict["Alias-Term"] in y[-1]:
                            resample_count = ALIAS_TERM_RESAMPLE
                        elif eval_ent_dict["Qualifier"] in y[-1]:
                            resample_count = QUAL_RESAMPLE
                        elif eval_ent_dict["Term"] in y[-1]:
                            resample_count = TERM_RESAMPLE

                        for i in range(resample_count):
                            X.append(torch.tensor(tokens))
                            y.append(torch.tensor([eval_ent_dict[ent] if ent in eval_ent_dict else 0 for ent in new_entities]))

                    sentence = ""
                    entities = []
                    words = []

    X, y = shuffle(X, y, random_state=42)

    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0).to(device)

    # print(new_y.shape)

    assert X.shape[0] == y.shape[0] and X.shape[1] == y.shape[1]

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    return loader
