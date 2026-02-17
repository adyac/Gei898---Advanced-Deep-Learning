import torch
import torchvision
from Devoir2_net import Devoir2_Net
from Helpers import *
from torch.utils.data import DataLoader

import torch.nn as nn
from madgrad import MADGRAD
import numpy as np
import matplotlib.pyplot as plt

def main():
    SAVE_PATH = "./models/best_model.pt"
    train = True
    generate = True
    test = True
    BATCH_SIZE = 128
    nb_epochs = 20
    lr = 0.00002
    SEQ_LEN = 64
    OVERLAP = 32
    EMBEDDING_DIM = 1024#512
    nb_heads = 8
    intermediary_dim = 512
    nb_transformer_layers = 4
    dropout = 0.1
    eos_token = "<EOS>"
    unk_token = "<UNKNOWN>"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ids, valid_ids, test_ids, vocab_size, vocab, mapping = get_data(SEQ_LEN, OVERLAP, unk_token, eos_token)

    train_loader = DataLoader(train_ids, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_ids, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_ids, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)


    loss_fn = nn.CrossEntropyLoss()

    train_loss_history = []
    valid_loss_history = []
    if train:
        best_valid = float('inf')
        model = Devoir2_Net(vocab_size=vocab_size, seq_lenN=SEQ_LEN, embedding_dimensionD=EMBEDDING_DIM,
                            nb_headsH=nb_heads, nb_transformerlayersK=nb_transformer_layers,
                            interdimF=intermediary_dim, dropout=dropout).to(device)
        optimizer = MADGRAD(model.parameters(), lr=lr)
        for epoch in range(nb_epochs):
            # training

            model.train()
            train_running_loss = 0
            for input_seq, target_seq in train_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                outputs = model(input_seq)
                target_seq = target_seq.contiguous().view(-1)
                outputs = outputs.view(-1, vocab_size)

                loss = loss_fn(outputs, target_seq.view(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_running_loss += loss.detach().cpu().numpy()
                #print(f"finished treating sequence {i}/{len(train_loader)}")

            epoch_train_loss = train_running_loss / len(train_loader)
            train_loss_history.append(epoch_train_loss)

            # validation
            with torch.no_grad():
                valid_running_loss = 0
                model.eval()
                for input_seq, target_seq in valid_loader:
                    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                    outputs = model(input_seq)
                    target_seq = target_seq.contiguous().view(-1)
                    outputs = outputs.view(-1, vocab_size)

                    loss = loss_fn(outputs, target_seq.view(-1))
                    valid_running_loss += loss.detach().cpu().numpy()

                epoch_valid_loss = valid_running_loss / len(valid_loader)
                valid_loss_history.append(epoch_valid_loss)
            print(f"Epoch {epoch + 1}\ntrain loss : {epoch_train_loss:.3f}\tvalid loss: {epoch_valid_loss:.3f}")
            if epoch_valid_loss < best_valid:
                best_valid = epoch_valid_loss

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_train_loss,
                    "valid_loss": epoch_valid_loss,
                }, SAVE_PATH)

        epochs = np.arange(1, nb_epochs + 1)
        plt.figure()
        plt.plot(epochs, train_loss_history, label="train loss")
        plt.plot(epochs, valid_loss_history, label="valid loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and validation losses")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./figures/loss_curves.png")
        plt.show()

    # test

    model = Devoir2_Net(
        vocab_size=vocab_size,
        seq_lenN=SEQ_LEN,  # must match training
        embedding_dimensionD=EMBEDDING_DIM,  # must match training
        nb_headsH=nb_heads,  # must match training
        nb_transformerlayersK=nb_transformer_layers,  # must match training
        interdimF=intermediary_dim,  # must match training
        dropout=dropout # usually same; can be anything for inference but keep consistent
    ).to(device)

    ckpt = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if test:
        with torch.no_grad():
            model.eval()
            correct = 0
            test_running_loss = 0
            total = 0
            for input_seq, target_seq in test_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                outputs = model(input_seq)
                target_seq = target_seq.contiguous().view(-1)
                outputs = outputs.view(-1, vocab_size)

                loss = loss_fn(outputs, target_seq.view(-1))
                targets = target_seq.view(-1)  # (B*63,)
                pred = outputs.argmax(dim=1)  # (B*63,)
                correct += (pred == targets).sum().item()
                test_running_loss += loss.detach().cpu().numpy()
                total += targets.numel()

            test_loss = test_running_loss / len(test_loader)
            test_token_acc = (correct / total)*100
            print(f"####################################\nTest loss: {test_loss:.3f}\ttokens accuracy: {test_token_acc:.3f}%\tvocabulary length : {vocab_size}\n####################################")
    if generate:
        while True:
            model.eval()
            eos_id = vocab[eos_token]

            print("Enter the start of sequence : ")
            prompt = input()

            _, generated_ids = prompt_to_model_input(prompt, eos_token, unk_token, vocab, max_len=SEQ_LEN - 1)

            # how many new tokens max (avoid infinite loops)
            max_new_tokens = 20

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # keep only the last context_len tokens (sliding window)
                    inp = torch.tensor(generated_ids[-(SEQ_LEN - 1):], dtype=torch.long, device=device).unsqueeze(0)

                    logits = model(inp)  # (1, L, vocab_size)
                    next_id = logits[0, -1].argmax().item()  # greedy decode

                    generated_ids.append(next_id)

                    #if next_id == eos_id:
                    #    break

            # Decode ids to tokens using mapping (id -> word)
            generated_tokens = [mapping[i] for i in generated_ids]

            # turn <EOS> back into .
            out = " ".join(generated_tokens).replace(" <EOS>", ".")
            print("\nGenerated:\n", out)


if __name__ == "__main__":
    main()




