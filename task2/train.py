from process import process_data
from tqdm import tqdm
from models import LangModelWithLSTM, LangModelWithDense
import torch
import argparse
from transformers import RobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import cut_padding, str2bool, parse_lang_model
from torchcrf import CRF
import numpy as np


np.set_printoptions(precision=3)


num_classes = 7
target_labels = [i for i in range(1, num_classes)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang_model")
    parser.add_argument("train_data")
    parser.add_argument("dev_data")
    parser.add_argument("fine_tune")
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()

    lang_model_names = ["scibert-base-cased", "xlnet-base-cased"]
    fine_tunes = [False]
            
    lang_model, tokenizer, lm_emb_size = parse_lang_model(args.lang_model_name)
    vocab_size = len(tokenizer)

    device = torch.device(args.device)

    train_loader = process_data(args.train_data, tokenizer, device, args.lang_model_name, batch_size=args.batch_size)
    dev_loader = process_data(args.dev_data, tokenizer, device, args.lang_model_name, is_dev=True, batch_size=args.batch_size)

    model = LangModelWithDense(lang_model,
                               lm_emb_size,
                               args.hidden_size,
                               num_classes,
                               args.fine_tune).to(device)

    print(model)
    print("Using model: {}".format(args.lang_model_name))
    print("Using device: {}".format(device))
    print("Using fine-tuning: {}".format(args.fine_tune))
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = CRF(num_classes, batch_first=True).to(device)

    epochs = 7 if args.fine_tune else 20
    best_macro_f1 = 0

    for epoch in range(epochs):
        model.train()
        loss = 0
        acc = 0
        micro_f1 = 0
        macro_f1 = 0
        macro_f1_all = np.zeros(num_classes-1)

        train_t = tqdm(train_loader)
        dev_t = tqdm(dev_loader)

        ct = 0

        for i, (train_x, train_y, mask) in enumerate(train_t):
            optimizer.zero_grad()

            train_x, train_y, mask = cut_padding(train_x, train_y, mask, device, tokenizer.pad_token_id)

            output = model.forward(train_x, mask)

            curr_loss = - criterion(output.to(device), train_y, reduction="token_mean", mask=mask)
            # curr_loss = criterion(output.reshape(-1, num_classes).to(device), train_y.reshape(-1))
            curr_loss.backward()
            optimizer.step()

            # --------------------------------------- Evaluate model ------------------------------------------------- #
            loss = (loss * i + curr_loss.item()) / (i + 1)

            pred = torch.tensor([torch.argmax(x) for x in output.view(-1, num_classes)])
            train_y = train_y.reshape(-1) # reshape to linear vector
            curr_acc = accuracy_score(train_y.cpu(), pred.cpu())
            curr_micro_f1 = f1_score(train_y.cpu(), pred.cpu(), labels=target_labels, average='micro')
            curr_macro_f1 = f1_score(train_y.cpu(), pred.cpu(), labels=target_labels, average='macro')
            curr_macro_f1_all = f1_score(train_y.cpu(), pred.cpu(), labels=target_labels, average=None)

            acc = (acc * i + curr_acc) / (i + 1)
            micro_f1 = (micro_f1 * i + curr_micro_f1) / (i + 1)
            macro_f1 = (macro_f1 * i + curr_macro_f1) / (i + 1)
            macro_f1_all = (macro_f1_all * i + curr_macro_f1_all) / (i + 1)

            train_t.set_description("Epoch: {}/{}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, "
                                    "Train Micro F1: {:.4f}, Train Macro F1: {:.4f}, Train Macro F1 all: {}".
                                    format(epoch, epochs, loss, acc, micro_f1, macro_f1, macro_f1_all))
            train_t.refresh()

        del loss
        del acc

        model.eval()
        loss = 0
        acc = 0

        micro_prec = 0
        micro_recall = 0
        micro_f1 = 0

        macro_prec = 0
        macro_recall = 0
        macro_f1 = 0
        macro_f1_all = np.zeros(num_classes-1)

        for i, (dev_x, dev_y, mask) in enumerate(dev_t):
            dev_x, dev_y, mask = cut_padding(dev_x, dev_y, mask, device, tokenizer.pad_token_id)
            # train_y = train_y.reshape(-1)

            output = model.forward(dev_x, mask)
            # curr_loss = criterion(output.reshape(-1, num_classes).to(device), dev_y.reshape(-1))
            curr_loss = - criterion(output.to(device), dev_y, reduction='token_mean', mask=mask)

            # --------------------------------------- Evaluate model ------------------------------------------------- #

            loss = (loss * i + curr_loss.item()) / (i + 1)

            pred = torch.tensor([torch.argmax(x) for x in output.view(-1, num_classes)])
            dev_y = dev_y.reshape(-1)  # reshape to linear vector
            curr_acc = accuracy_score(dev_y.cpu(), pred.cpu())

            curr_micro_prec = precision_score(dev_y.cpu(), pred.cpu(), labels=target_labels, average='micro')
            curr_micro_recall = recall_score(dev_y.cpu(), pred.cpu(), labels=target_labels, average='micro')
            curr_micro_f1 = f1_score(dev_y.cpu(), pred.cpu(), labels=target_labels, average='micro')

            curr_macro_prec = precision_score(dev_y.cpu(), pred.cpu(), labels=target_labels, average='macro')
            curr_macro_recall = recall_score(dev_y.cpu(), pred.cpu(), labels=target_labels, average='macro')
            curr_macro_f1 = f1_score(dev_y.cpu(), pred.cpu(), labels=target_labels, average='macro')

            acc = (acc * i + curr_acc) / (i + 1)
            micro_prec = (micro_prec * i + curr_micro_prec) / (i + 1)
            micro_recall = (micro_recall * i + curr_micro_recall) / (i + 1)
            micro_f1 = (micro_f1 * i + curr_micro_f1) / (i + 1)
            curr_macro_f1_all = f1_score(dev_y.cpu(), pred.cpu(), labels=target_labels, average=None)

            macro_prec = (macro_prec * i + curr_macro_prec) / (i + 1)
            macro_recall = (macro_recall * i + curr_macro_recall) / (i + 1)
            macro_f1 = (macro_f1 * i + curr_macro_f1) / (i + 1)
            macro_f1_all = (macro_f1_all * i + curr_macro_f1_all) / (i + 1)

            dev_t.set_description("Epoch: {}/{}, Dev Loss: {:.4f}, Dev Accuracy: {:.4f}, "
                                  "Dev Micro F1: {:.4f}, Dev Macro F1: {:.4f}, Dev Macro F1 all: {}".
                                  format(epoch, epochs, loss, acc, micro_f1, macro_f1, macro_f1_all))
            dev_t.refresh()

        if macro_f1 > best_macro_f1:
            print("Macro F1 score improved from {:.4f} -> {:.4f}. Saving model...".format(best_macro_f1, macro_f1))
            best_macro_f1 = macro_f1
            torch.save(model, "models/{}_{}.pt".format(args.lang_model_name, "finetune" if args.fine_tune else "frozen"))
            with open("models/{}_{}.txt".format(args.lang_model_name, "finetune" if args.fine_tune else "frozen"), "w") as file:
                file.write("Acc: {}, Macro Prec: {}, Macro Rec: {}, Macro F1: "
                           "{}, Micro Prec: {}, Micro Rec: {}, Micro F1: {}," 
                           "Macro F1 all:{}".format(acc,
                                                    macro_prec,
                                                    macro_recall,
                                                    macro_f1,
                                                    micro_prec,
                                                    micro_recall,
                                                    micro_f1,
                                                    macro_f1_all))
