import torch
import argparse
from process import process_test_sentences
from utils import parse_lang_model, str2bool
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("lang_model")
    parser.add_argument("fine_tune", type=str2bool)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device)
    lang_model, tokenizer, lm_emb_size = parse_lang_model(args.lang_model)

    print("Using model: {}".format(args.lang_model))
    print("Currently using {}".format(device))
    print("Using fine-tuning: {}".format(args.fine_tune))
    print()

    X_list = process_test_sentences(args.input_path, tokenizer, args.fine_tune, device)

    model = torch.load(args.model_path, map_location=device)
    model.eval()

    for filename, X in zip(os.listdir(args.input_path), X_list):
        pred = model.forward(X)

        with open(os.path.join(args.output_path, "task_1_" + filename), "w", encoding="utf-8") as out_file:
            with open(os.path.join(args.input_path, filename), "r", encoding='utf-8') as in_file:
                for line, y in zip(in_file, pred):
                    print("{}\t\"{}\"\n".format(line[:-5], 0 if y[0] < 0.5 else 1))
                    out_file.write("{}\t\"{}\"\n".format(line[:-5], 0 if y < 0.5 else 1))
