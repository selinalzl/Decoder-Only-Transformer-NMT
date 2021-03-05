#!/usr/bin/env python3

from tokenizers import Tokenizer
from decoder_model import DecoderOnlyTransformer, initialize_weights
from inference import greedy_decode
import argparse
import torch


def generate_sentence_translation(sentence, model, tokenizer, device, max_len=100):
    """
    Shows predicted translation of given sentence.

    :param senntence: Sentence in the source language the model was trained on
    :param model: Model to use for predictions
    :param tokenizer: Tokenizer to prepare dataset samples
    :param device: CPU or GPU
    :param max_len: Maximum length of source/target sentence
    :param n: Number of sentences to sample
    """

    translation, _, _ = greedy_decode(
        sentence, model, tokenizer, max_len, device)

    print("Predicted translation: \t\t" + translation, end='\n\n')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sent_len", default=128, type=float,
                        help="Maximum length of source/target sentence.")
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
    args = parser.parse_args()

    # get model
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_en_de.json")
    pad_index = tokenizer.token_to_id("[PAD]")
    vocab_size = tokenizer.get_vocab_size()
    model = DecoderOnlyTransformer(pad_index, vocab_size, args.d_model, args.seq_len,
                                   args.n_heads, args.n_layers, args.ff_dim, args.dropout).to(device)
    model.apply(initialize_weights)
    model.to(device)
    model.load_state_dict(torch.load('model.pt', map_location=device))

    while True:
        input_sentence = input("Input sentence to be translated: ")
        generate_sentence_translation(
            input_sentence, model, tokenizer, device, max_len=args.max_sent_len)


if __name__ == "__main__":
    main()

