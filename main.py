#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from decoder_model import DecoderOnlyTransformer, initialize_weights, count_parameters
from dataset import IWSLT
from training import train, evaluate
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import math


def epoch_time(start_time, end_time):
    """
    Compute the time it takes for one epoch.
    """
    time = end_time - start_time
    mins = int(time / 60)
    secs = int(time - (mins * 60))
    return mins, secs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sent_len", default=128, type=float,
                        help="Maximum character length of source/target sentence.")
    parser.add_argument("--seq_len", default=257, type=int,
                        help="Maximum total input sequence length after tokenization, sequences shorter will be padded.")
    parser.add_argument("--d_model", default=256, type=int,
                        help="The size of the hidden dimension.")
    parser.add_argument("--ff_dim", default=1024, type=int,
                        help="The dimension of the feed-forward network.")
    parser.add_argument("--n_heads", default=8, type=int,
                        help="The number of heads in the multi-headed attention.")
    parser.add_argument("--n_layers", default=9, type=int,
                        help="The number of decoder layers.")
    parser.add_argument("--dropout", default=0.1, type=float, 
                        help="The dropout value.")
    parser.add_argument("--learning_rate", default=0.00005, type=float,
                        help="The learning rate.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training/evaluation.")
    parser.add_argument("--n_epochs", default=50, type=int,
                        help="Number of training epochs to perform.")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_en_de.json")

    vocab_size = tokenizer.get_vocab_size()       # the size of the vocabulary
    pad_index = tokenizer.token_to_id("[PAD]")    # index of the padding token '[PAD]'

    print("Preparing dataset...")

    training = "train.de-en"
    validation = "IWSLT15.TEDX.dev2012.de-en"
    test = "IWSLT15.TED.tst2012.de-en"

    train_set = IWSLT(training, tokenizer, max_length=args.max_sent_len)
    validation_set = IWSLT(validation, tokenizer, max_length=args.max_sent_len)
    test_set = IWSLT(test, tokenizer, max_length=args.max_sent_len)

    print("data length: \n",
          f"length train_set: {len(train_set.de_en)} \n",
          f"length validation_set: {len(validation_set.de_en)} \n",
          f"length test_set: {len(test_set.de_en)}")

    print("Instantiating DecoderOnlyTransformer...")

    model = DecoderOnlyTransformer(pad_index, vocab_size, args.d_model, args.seq_len,
                                   args.n_heads, args.n_layers, args.ff_dim, args.dropout).to(device)
    model.apply(initialize_weights)
    model.to(device)
    print(f"Model has {count_parameters(model):,} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion.to(device)
    
    if torch.cuda.is_available(): 
        n_workers = 4
    else:
        n_workers = 0

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=n_workers, pin_memory=True)
    valid_loader = DataLoader(
        validation_set, batch_size=args.batch_size, num_workers=n_workers, pin_memory=True)

    print("Training...")

    best_valid_loss = float('inf')

    all_train_loss = []
    all_valid_loss = []

    for epoch in range(args.n_epochs):

        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, 1, device)
        all_train_loss.append(train_loss)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        all_valid_loss.append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("Saving trained model...")
            torch.save(model.state_dict(), 'model.pt')
            
            # plot train and validation loss
            plt.plot(all_train_loss, label='Training loss')
            plt.plot(all_valid_loss, label='Validation loss')
            plt.legend()
            plt.grid(True)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('losses_model.jpg')
            plt.close()

        print(f'Epoch: {epoch+1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} , PPL : {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} , PPL : {math.exp(valid_loss):7.3f}')

    print("Training finished")


if __name__ == "__main__":
    main()

