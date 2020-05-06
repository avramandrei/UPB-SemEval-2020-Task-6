import os
import torch
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
import random
from collections import Counter
import time

random.seed(42)

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


keep_def_prob = 1
keep_O_prob = 1


def decode_token(token, tokenizer, lang_model):
    if "roberta" in lang_model:
        return tokenizer.decode(token).replace(" ", "")
    elif "scibert" in lang_model:
        token_word = tokenizer.decode(token).lower()
        return token_word if token_word[:2] != "##" else token_word[2:]
    elif "xlnet" in lang_model:
        return tokenizer.decode(token)
    elif "albert-base" in lang_model:
        return tokenizer.decode(token)
    elif "bert" in lang_model:
        token_word = tokenizer.decode(token)
        return token_word if token_word[:2] != "##" else token_word[2:]
    

def remove_weird_letters(word, lang_model):
    replacements_chars = [("ö", "o"), ("é", "e"), ("ê", "e"), ("ü", "u"),
                          ("ó", "o"), ("â", "a"), ("ä", "a"), ("à", "a"),
                          ("ç", "c"), ("ï", "i"), ("ô", "o"), ("û", "u"),
                          ("ÿ", "y"), ("á", "a")]
    
    if "scibert" in lang_model:
        word = word.lower()
        
        for init_char, repl_char in replacements_chars:
            word = word.replace(init_char, repl_char)
    elif "albert" in lang_model:
        word = word.lower()
    elif "xlnet" in lang_model:
        for init_char, repl_char in replacements_chars:
            word = word.replace(init_char, repl_char)
            word = word.replace(init_char.upper(), repl_char.upper())
        
    return word


def correct_punctuation(word, lang_model):
    word = word.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'") \
                               .replace(",", ",").replace("⋅", "*").replace("—", "-").replace("`", "'")
                               # replace("…", "...").replace("º", "")
    
    if "scibert" in lang_model:
        word = word.replace("º", "*")
    elif "roberta" in lang_model:
        word = word.replace("º", "")
    elif "xlnet" in lang_model:
        word = word.replace("…", "...")
                               
    return word


def remove_greek(word, lang_model, tokenizer):
    greek_alphabet = 'αβγδεζηθικλμνξοπρςστυφχψω' + 'αβγδεζηθικλμνξοπρςστυφχψω'.upper() + "∆"
    
    for letter in greek_alphabet:
        word = tokenizer.unk_token if letter in word else word
        
    return word


def process_data(path, tokenizer, device, lang_model, is_dev=False, batch_size=32):
    X = []
    y = []
    mask = []
    max_labels = 0

    for file_ct, filename in enumerate(os.listdir(path)):
        sentence = ""
        last_sentence = ""
        entities = []
        words = []

        with open(os.path.join(path, filename), "r", encoding="utf-8") as file:
            new_context = False

            for line in file:
                if not new_context:
                    if line != "\n":
                        tokens = line.split()
                        word, entity = tokens[0], tokens[4] if tokens[4] == "O" else tokens[4][2:]

                        # # ########################### do some preprocessing ################################################

                        word = correct_punctuation(word, lang_model)
                        word = remove_greek(word, lang_model, tokenizer)
                        word = remove_weird_letters(word, lang_model)

                        if "http" in word or "www" in word:
                            word = tokenizer.unk_token

                        # ##################################################################################################

                        entities.append(entity)
                        words.append(word)

                        sentence += word + " "
                        last_sentence += word + " "
                    else:
                        if len(last_sentence.split()) == 2:
                            new_context = True
                            sentence = sentence[:-len(last_sentence)]
                            del words[-2:]
                            del entities[-2:]
                        else:
                            last_sentence = ""

                if new_context:
                    if sentence == "":
                        sentence = last_sentence
                        entities = ["O", "O"]
                        words = last_sentence.split()
                        new_context = False
                        continue

                    print("Filename: {}. File counter: {}/{}".format(filename, file_ct, len(os.listdir(path))))
                    print("Sentence: '{}'".format(sentence))
                    print("Entities: {}".format(entities))

                    tokens = tokenizer.encode(sentence, add_special_tokens=True)
                    print("Tokens: {}".format(tokens))
                    print("Decoded tokens: '{}'".format(tokenizer.decode(tokens)))
                    print()

                    new_entities = ['O'] if "xlnet" not in lang_model else []

                    word_id = 0
                    token_id = 1 if "xlnet" not in lang_model else 0

                    while word_id < len(words):
                        token_word = decode_token([tokens[token_id]], tokenizer, lang_model)

                        print("Current word: '{}' and current token_id: '{}'".format(words[word_id], token_word))

                        if token_word == words[word_id]:
                            new_entities.append(entities[word_id])

                            token_id += 1
                            word_id += 1
                        elif token_word in words[word_id]:
                            word_copy = words[word_id]

                            while len(word_copy) > 0:
                                print("\tToken in word: '{}' -> '{}".format(token_word, word_copy))
                                new_entities.append(entities[word_id])
                                
                                if token_word == tokenizer.unk_token:
                                    next_token_word = decode_token([tokens[token_id+1]], tokenizer, lang_model)
                                    
                                    first_appearance = word_copy.find(next_token_word)
                                    if first_appearance != -1 and next_token_word != '':
                                        word_copy = word_copy[first_appearance:]
                                    else:
                                        word_copy = ""
                                        break
                                else:
                                    word_copy = word_copy[len(token_word):]

                                token_id += 1
                                token_word = decode_token([tokens[token_id]], tokenizer, lang_model)

                            word_id += 1
                        else:
                            new_entities.append(entities[word_id])
                            
                            token_id += 1
                            word_id += 1

                    new_entities.append('O')
                    
                    if "xlnet" in lang_model:
                        new_entities.append('O')

                    assert len(new_entities) == len(tokens)

                    print("\nTokens: {}".format(tokens))
                    print("New entities: {}".format(new_entities))

                    keep = True

                    if not is_dev:
                        if "Alias-Term" not in new_entities and \
                            "Referential-Definition" not in new_entities and "Referential-Term" not in new_entities and \
                            "Qualifier" not in new_entities:
    
                            if "Definition" not in new_entities and "Term" not in new_entities:
                                keep = random.uniform(0, 1) < keep_O_prob
                            else:
                                keep = random.uniform(0, 1) < keep_def_prob

                    if keep or is_dev:
                        if "Referential-Term" in new_entities and not is_dev:
                            repeat_no = 16
                        elif "Referential-Definition" in new_entities and not is_dev:
                            repeat_no = 8
                        elif "Qualifier" in new_entities and not is_dev:
                            repeat_no = 4
                        elif "Alias-Term" in new_entities and not is_dev:
                            repeat_no = 2
                        else:
                            repeat_no = 1

                        for _ in range(repeat_no):
                            X.append(torch.tensor(tokens))
                            y.append(torch.tensor([eval_ent_dict[ent] if ent in eval_ent_dict else eval_ent_dict['O'] for ent in new_entities]))
                            mask.append(torch.ones_like(torch.tensor(tokens), dtype=torch.uint8))

                    print("Proposition keep: {}".format(keep))

                    print("\n" + "*" * 150 + "\n")

                    sentence = last_sentence
                    entities = ["O", "O"]
                    words = last_sentence.split()
                    new_context = False

        print("Sentence: '{}'".format(sentence))
        print("Entities: {}".format(entities))

        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        print("Tokens: {}".format(tokens))
        print("Decoded tokens: '{}'".format(tokenizer.decode(tokens)))
        print()

        new_entities = ['O'] if "xlnet" not in lang_model else []

        word_id = 0
        token_id = 1 if "xlnet" not in lang_model else 0

        while word_id < len(words):
            token_word = decode_token([tokens[token_id]], tokenizer, lang_model)

            print("Current word: '{}' and current token_id: '{}'".format(words[word_id], token_word))

            if token_word == words[word_id]:
                new_entities.append(entities[word_id])

                token_id += 1
                word_id += 1
            elif token_word in words[word_id]:
                word_copy = words[word_id]

                while len(word_copy) > 0:
                    print("\tToken in word: '{}' -> '{}".format(token_word, word_copy))
                    new_entities.append(entities[word_id])

                    if token_word == tokenizer.unk_token:
                        next_token_word = decode_token([tokens[token_id + 1]], tokenizer, lang_model)

                        first_appearance = word_copy.find(next_token_word)
                        if first_appearance != -1 and next_token_word != '':
                            word_copy = word_copy[first_appearance:]
                        else:
                            word_copy = ""
                            break
                    else:
                        word_copy = word_copy[len(token_word):]

                    token_id += 1
                    token_word = decode_token([tokens[token_id]], tokenizer, lang_model)

                word_id += 1
            else:
                new_entities.append(entities[word_id])

                token_id += 1
                word_id += 1

        new_entities.append('O')
        
        if "xlnet" in lang_model:
            new_entities.append('O')

        assert len(new_entities) == len(tokens)

        print("\nTokens: {}".format(tokens))
        print("New entities: {}".format(new_entities))

        keep = True

        if not is_dev:
            if "Alias-Term" not in new_entities and \
                "Referential-Definition" not in new_entities and "Referential-Term" not in new_entities and \
                "Qualifier" not in new_entities:

                if "Definition" not in new_entities and "Term" not in new_entities:
                    keep = random.uniform(0, 1) < keep_O_prob
                else:
                    keep = random.uniform(0, 1) < keep_def_prob

        if keep or is_dev:
            if "Referential-Term" in new_entities and not is_dev:
                repeat_no = 16
            elif "Referential-Definition" in new_entities and not is_dev:
                repeat_no = 8
            elif "Qualifier" in new_entities and not is_dev:
                repeat_no = 4
            elif "Alias-Term" in new_entities and not is_dev:
                repeat_no = 2
            else:
                repeat_no = 1

            for _ in range(repeat_no):
                X.append(torch.tensor(tokens))
                y.append(torch.tensor([eval_ent_dict[ent] if ent in eval_ent_dict else eval_ent_dict['O'] for ent in new_entities]))
                mask.append(torch.ones_like(torch.tensor(tokens), dtype=torch.uint8))

    ct = Counter()

    for seq in y:
        for ent in seq:
            ct[inv_eval_ent_dict[int(ent)]] += 1

    print("Class distribution after balancing: {}\n".format(ct))

    X, y, mask = shuffle(X, y, mask, random_state=42)
    
    print(len(X))

    X = pad_sequence(X, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    y = pad_sequence(y, batch_first=True, padding_value=0).to(device)
    mask = pad_sequence(mask, batch_first=True, padding_value=0).to(device)

    # print(new_y.shape)

    assert X.shape[0] == y.shape[0] and X.shape[1] == y.shape[1]

    dataset = torch.utils.data.TensorDataset(X, y, mask)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader
