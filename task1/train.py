from process import process_data
from model import LangModelWithDense
from utils import parse_lang_model, str2bool, cut_padding
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
from transformers import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang_model")
    parser.add_argument("train_data")
    parser.add_argument("dev_data")
    parser.add_argument("save_path")
    parser.add_argument("--fine_tune", type=str2bool, default=False)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()

    lang_model, tokenizer, lm_emb_size = parse_lang_model(args.lang_model)
    vocab_size = len(tokenizer)

    device = torch.device(args.device)

    print("Using model: {}".format(args.lang_model))
    print("Using device: {}".format(device))
    print("Using fine-tuning: {}".format(args.fine_tune))
    print()

    train_loader = process_data(args.train_data, tokenizer, device, train_data=True, fine_tune=args.fine_tune, batch_size=args.batch_size)
    dev_loader = process_data(args.dev_data, tokenizer, device, fine_tune=args.fine_tune, batch_size=args.batch_size)

    model = LangModelWithDense(lang_model,
                               vocab_size,
                               lm_emb_size,
                               args.hidden_size,
                               args.fine_tune
                               ).to(device)

    print(model)
    
    epochs = 10
    total_steps = len(train_loader) * epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    #scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                #num_warmup_steps = 0, # Default value in run_glue.py
                                                #num_training_steps = total_steps)
    criterion = torch.nn.BCELoss()
    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        loss = 0
        acc = 0
        f1 = 0

        train_t = tqdm(train_loader)
        dev_t = tqdm(dev_loader)

        for i, (train_x, train_y, mask) in enumerate(train_t):
            optimizer.zero_grad()

            train_x, mask = cut_padding(train_x, mask, tokenizer.pad_token_id, device)

            output = model.forward(train_x, mask)
            curr_loss = criterion(output, train_y)
            curr_loss.backward()
            optimizer.step()
            #scheduler.step()

            loss = (loss * i + curr_loss.item()) / (i + 1)

            pred = torch.tensor([0 if x < 0.5 else 1 for x in output])
            curr_acc = accuracy_score(train_y.cpu(), pred.cpu())
            curr_f1 = f1_score(train_y.cpu(), pred.cpu())
            acc = (acc * i + curr_acc) / (i + 1)
            f1 = (f1 * i + curr_f1) / (i + 1)

            train_t.set_description(
                "Epoch: {}/{}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Train F1: {:.4f}".format(epoch, epochs, loss,
                                                                                                    acc, f1))
            train_t.refresh()

        del loss
        del acc
        del f1
        del output
        del pred

        model.eval()
        loss = 0
        acc = 0
        f1 = 0
        prec = 0
        rec = 0
        total_f1 = 0

        for i, (dev_x, dev_y, mask) in enumerate(dev_t):
            dev_x, mask = cut_padding(dev_x, mask, tokenizer.pad_token_id, device)

            output = model.forward(dev_x, mask)
            curr_loss = criterion(output, dev_y)
            loss = (loss * i + curr_loss.item()) / (i + 1)

            pred = torch.tensor([[0] if x[0] < 0.5 else [1] for x in output])
            curr_acc = accuracy_score(dev_y.cpu(), pred.cpu())
            curr_prec = precision_score(dev_y.cpu(), pred.cpu())
            curr_rec = recall_score(dev_y.cpu(), pred.cpu())
            curr_f1 = f1_score(dev_y.cpu(), pred.cpu())

            acc = (acc * i + curr_acc) / (i + 1)
            prec = (prec * i + curr_prec) / (i + 1)
            rec = (rec * i + curr_rec) / (i + 1)
            f1 = (f1 * i + curr_f1) / (i + 1)
            total_f1 += curr_f1

            dev_t.set_description(
                "Epoch: {}/{}, Dev Loss: {:.4f}, Dev Accuracy: {:.4f}, Dev F1: {:.4f}".format(epoch, epochs, loss, acc,
                                                                                              f1))
            dev_t.refresh()

        if f1 > best_f1:
            print("F1 score improved from {:.4f} -> {:.4f}. Saving model...".format(best_f1, f1))
            best_f1 = f1
            torch.save(model, args.save_path)
            with open(args.save_path[:-3] + ".txt", "w") as file:
                file.write("Acc: {}, Prec: {}, Rec: {}, F1: {}".format(acc, prec, rec, f1))

