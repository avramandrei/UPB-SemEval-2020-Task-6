import torch
import argparse
import os
from utils import parse_lang_model
from process import inv_eval_ent_dict, decode_token, remove_greek, remove_weird_letters, correct_punctuation
from collections import Counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("model_path")
    parser.add_argument("lang_model")
    parser.add_argument("output_path")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model = torch.load(args.model_path, map_location=args.device)
    model.fine_tune = False
    model.eval()

    _, tokenizer, _ = parse_lang_model(args.lang_model)

    for filename in os.listdir(args.data_path):
        sentence = ""
        last_sentence = ""
        words = []

        entities_list = []

        with open(os.path.join(args.data_path, filename), encoding="utf-8", mode="r") as in_file:

            new_context = False

            for i, line in enumerate(in_file):
                if not new_context:
                    if line is not "\n":
                        tokens = line.split("\t")
                        word = tokens[0]

                        word = correct_punctuation(word, args.lang_model)
                        word = remove_greek(word, args.lang_model, tokenizer)
                        word = remove_weird_letters(word, args.lang_model)

                        if "http" in word or "www" in word:
                            word = tokenizer.unk_token

                        words.append(word)
                        sentence += word + " "
                        last_sentence += word + " "
                    else:

                        if len(last_sentence.split()) == 2 and last_sentence.split()[0].isdigit():
                            new_context = True
                            sentence = sentence[:-len(last_sentence)]
                            del words[-2:]
                        else:
                            last_sentence = ""

                if new_context:
                    if sentence == "":
                        sentence = last_sentence
                        words = last_sentence.split()
                        new_context = False
                        continue
                    
                    print("Sentence: '{}'".format(sentence))

                    tokens = tokenizer.encode(sentence, add_special_tokens=True)
                    print("Tokens: {}".format(tokens))
                    print("Decoded tokens: '{}'".format(tokenizer.decode(tokens)))
                    print()

                    outputs = model.forward(torch.tensor([tokens]), torch.tensor([[1 for _ in range(len(tokens))]]))[0]
                    
                    outputs = outputs[1:-1] if "xlnet" not in args.lang_model else outputs[0:-2]

                    entities = [inv_eval_ent_dict[int(torch.argmax(output))] for output in outputs]
                    print("Predicted entities: {}".format(entities))

                    new_entities = []

                    word_idx = 0
                    ent_idx = 0
                    tok_idx = 1

                    while word_idx < len(words):
                        decoded_token = decode_token([tokens[tok_idx]], tokenizer, args.lang_model)

                        # if word is equal to token - go ahead
                        if words[word_idx] == decoded_token:
                            print("Equal words: '{}' and '{}'".format(words[word_idx], decoded_token))
                            new_entities.append(entities[ent_idx])

                            word_idx += 1
                            ent_idx += 1
                            tok_idx += 1
                        # if word is NOT equal to token
                        else:
                            print("Conflict bettween word '{}' and token '{}'".format(words[word_idx], decoded_token))

                            # case 1 - token is part of the word
                            if decoded_token in words[word_idx]:
                                part_ents = [entities[ent_idx]]
                                word_copy = words[word_idx][len(decoded_token):]

                                tok_idx += 1
                                ent_idx += 1

                                # while a token is part of the word
                                while word_copy != "":
                                    decoded_token = decode_token([tokens[tok_idx]], tokenizer, args.lang_model)
                                    part_ents.append(entities[ent_idx])

                                    word_copy = word_copy[len(decoded_token):]

                                    ent_idx += 1
                                    tok_idx += 1
                                    print(word_copy)

                                # now we have a list of possible entities for the word - we choose the majority
                                print("Entities resulted from solving conflict :{}".format(part_ents))

                                part_ent_counter = Counter()
                                for part_ent in part_ents:
                                    part_ent_counter[part_ent] += 1

                                max_count = 0
                                true_ent = None
                                for ent, count in part_ent_counter.items():
                                    if count > max_count:
                                        max_count = count
                                        true_ent = ent

                                print("New entity of word: {}".format(true_ent))
                                new_entities.append(true_ent)

                                word_idx += 1

                                print("-" * 100)
                                print()

                    assert len(new_entities) == len(words)

                    entities_list += new_entities

                    sentence = last_sentence
                    entities = ["O", "O"]
                    words = last_sentence.split()
                    new_context = False

            print("Sentence: '{}'".format(sentence))

            tokens = tokenizer.encode(sentence, add_special_tokens=True)
            print("Tokens: {}".format(tokens))
            print("Decoded tokens: '{}'".format(tokenizer.decode(tokens)))
            print()

            outputs = model.forward(torch.tensor([tokens]), torch.tensor([[1 for _ in range(len(tokens))]]))[0]

            outputs = outputs[1:-1] if "xlnet" not in args.lang_model else outputs[0:-2]

            entities = [inv_eval_ent_dict[int(torch.argmax(output))] for output in outputs]
            print("Predicted entities: {}".format(entities))

            new_entities = []

            word_idx = 0
            ent_idx = 0
            tok_idx = 1

            while word_idx < len(words):
                decoded_token = decode_token([tokens[tok_idx]], tokenizer, args.lang_model)

                # if word is equal to token - go ahead
                if words[word_idx] == decoded_token:
                    print("Equal words: '{}' and '{}'".format(words[word_idx], decoded_token))
                    new_entities.append(entities[ent_idx])

                    word_idx += 1
                    ent_idx += 1
                    tok_idx += 1
                # if word is NOT equal to token
                else:
                    print("Conflict bettween word '{}' and token '{}'".format(words[word_idx], decoded_token))

                    # case 1 - token is part of the word
                    if decoded_token in words[word_idx]:
                        part_ents = [entities[ent_idx]]
                        word_copy = words[word_idx][len(decoded_token):]

                        tok_idx += 1
                        ent_idx += 1

                        # while a token is part of the word
                        while word_copy != "":
                            decoded_token = decode_token([tokens[tok_idx]], tokenizer, args.lang_model)
                            part_ents.append(entities[ent_idx])

                            word_copy = word_copy[len(decoded_token):]

                            ent_idx += 1
                            tok_idx += 1
                            print(word_copy)

                        # now we have a list of possible entities for the word - we choose the majority
                        print("Entities resulted from solving conflict :{}".format(part_ents))

                        part_ent_counter = Counter()
                        for part_ent in part_ents:
                            part_ent_counter[part_ent] += 1

                        max_count = 0
                        true_ent = None
                        for ent, count in part_ent_counter.items():
                            if count > max_count:
                                max_count = count
                                true_ent = ent

                        print("New entity of word: {}".format(true_ent))
                        new_entities.append(true_ent)

                        word_idx += 1

                        print("-" * 100)
                        print()

            assert len(new_entities) == len(words)

            entities_list += new_entities

        with open(os.path.join(args.output_path, "task_2_" + filename), encoding="utf-8", mode="w") as out_file, \
            open(os.path.join(args.data_path, filename), encoding="utf-8", mode="r") as in_file:

            counter = 0
            prev_entity = "O"

            for line in in_file:
                if line != "\n":
                    tokens = line.split("\t")
                    word = tokens[0]
                    source = tokens[1]
                    start = tokens[2]
                    end = tokens[3]

                    if entities_list[counter] != "O" and prev_entity == "O":
                        entities_list[counter] = "B-" + entities_list[counter]
                    elif entities_list[counter] != "O" and prev_entity != "O":
                        entities_list[counter] = "I-" + entities_list[counter]

                    out_file.write("\"{}\"\t\"{}\"\t{}\t{}\t\"{}\"\n".format(word, source, start, end, entities_list[counter]))

                    prev_entity = entities_list[counter]
                    counter += 1
                else:
                    out_file.write("\n")
