#!/usr/bin/env python3

from tokenizers import Tokenizer
from decoder_model import DecoderOnlyTransformer, initialize_weights
from dataset import IWSLT
from training import evaluate
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from operator import itemgetter
from tqdm import tqdm
import math


def greedy_decode(sentence, model, tokenizer, max_len, device):
    """
    Predicts a translation of a given sentence using greedy decoding.

    :param sentence: Sentence to translate
    :param model: Model to use for predictions
    :param tokenizer: Tokenizer to prepare dataset sentences
    :param max_len: Maximum length of source/target sentence
    :param device: CPU or GPU
    :return: The predicted translation (str), encoded translation (list) and attention (tensor)
    """
    model.eval()
    sent_len = len(sentence)

    # encode sentence if not already encoded
    if isinstance(sentence, str):
        encoded = tokenizer.encode(sentence)
        encoded_sentence = torch.tensor(
            encoded.ids, dtype=torch.long, device=device)
    else:
        encoded_sentence = torch.tensor(
            sentence, dtype=torch.long, device=device)
    sentence = encoded_sentence.unsqueeze(0)

    generated = sentence
    with torch.no_grad():
        for i in range(max_len-1):
            output, attention = model(generated)

            next_token_logits = output[0, -1, :]
            next_token_probs = F.log_softmax(next_token_logits, dim=0)
            next_token = torch.argmax(next_token_probs).unsqueeze(0)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token == tokenizer.token_to_id("[EOS]"):
                break

    generated_text = generated[0, sent_len:].tolist()

    translation_list = []
    for index in generated_text:
        translation_list.append(tokenizer.id_to_token(index))
    translation = tokenizer.decode(generated_text, skip_special_tokens=True)

    return translation, translation_list, attention


def translate_sentence_beam_search(sentence, model, tokenizer, max_len, device, temperature=1, beam_size=3):
    """
    Translates a given sentence using beam search.

    :param sentence: Sentence to translate
    :param model: Model to use for predictions
    :param tokenizer: Tokenizer to prepare dataset sentences
    :param max_len: Maximum length of source/target sentence
    :param device: CPU or GPU
    :param temperature: Temperature value
    :param beam_size: Number of beams for beam search
    :return: The predicted translation (str), encoded translation (list) and attention (tensor)
    """

    model.eval()
    vocab_size = tokenizer.get_vocab_size()

    # encode sentence if not already encoded
    if isinstance(sentence, str):
        encoded = tokenizer.encode(sentence)
        encoded_sentence = torch.tensor(
            encoded.ids, dtype=torch.long, device=device)
    else:
        encoded_sentence = torch.tensor(
            sentence, dtype=torch.long, device=device)
    sentence = encoded_sentence.unsqueeze(0)

    final = []
    with torch.no_grad():
        output, attention = model(sentence)
        next_token_logits = output[0, -1, :] / temperature
        next_token_probs = F.log_softmax(next_token_logits, dim=0)
        scores, indices = torch.topk(next_token_probs, beam_size)
        # first hypotheses
        hyps = [[start_idx] for start_idx in indices.tolist()]

        for t in range(max_len):
            logits = torch.zeros(len(hyps) * vocab_size)
            indices_groups = []
            for h in range(len(hyps)):
                # add next token to sentence
                new_input = torch.cat((sentence, torch.tensor(
                    [hyps[h]], dtype=torch.long, device=device)), dim=1)
                output, attention = model(new_input)
                next_token_logits = output[0, -1, :] / temperature
                next_token_probs = F.log_softmax(next_token_logits, dim=0)
                # fill logits
                assert(vocab_size == len(next_token_logits))
                start, stop = h * vocab_size, (h + 1) * vocab_size
                indices_groups.append((start, stop))
                logits[start:stop] = next_token_probs

            new_scores, new_indices = torch.topk(logits, len(hyps))
            scores += new_scores.to(device)

            # get indices for hypotheses that preceded new tokens with highest probabilities
            hyps_indices = []
            for i, (start, stop) in enumerate(indices_groups):
                for index in new_indices:
                    if index in range(start, stop+1):
                        hyps_indices.append(i)

            # because logits size = [len(hyps) * vocab_size]
            logits = (new_indices % vocab_size).tolist()

            # extend hypothesis with next token
            next_hyps = []
            scores_indices = []
            for h in range(len(hyps)):
                hyps[h] = hyps[hyps_indices[h]] + [logits[h]]
                # hypothesis is complete when EOS token is reached
                if logits[h] == tokenizer.token_to_id("[EOS]"):
                    # length normalization
                    final.append((scores[h]/len(hyps[h]), hyps[h]))
                    scores_indices.append(h)
                else:
                    next_hyps.append(hyps[h])

            # stop when beam_size translations are generated
            if len(final) == beam_size:
                break

            hyps = next_hyps

            next_scores = np.delete(scores.cpu(), scores_indices).to(device)
            scores = next_scores

    # append hypotheses that have reached max_len, apply length normalization
    if len(final) != beam_size:
        for h in range(len(hyps)):
            final.append((scores[h]/len(hyps[h]), hyps[h]))

    final.sort(key=itemgetter(0), reverse=True)

    highest_score_translation = final[0][1]
    translation = tokenizer.decode(
        highest_score_translation, skip_special_tokens=True)

    return translation, highest_score_translation, attention


def generate_sample_translations(data, model, tokenizer, device, max_len=100, n=5):
    """
    Shows n sample sentences with predicted translation and actual translation.

    :param data: Data to sample the sentence(s) from
    :param model: Model to use for predictions
    :param tokenizer: Tokenizer to prepare dataset samples
    :param device: CPU or GPU
    :param max_len: Maximum length of source/target sentence
    :param n: Number of sentences to sample
    """
    for i in tqdm(range(n)):
        sample = data[i]
        source = sample['source'].tolist()
        target = sample['target'].tolist()

        translation, _, _ = greedy_decode(
            source, model, tokenizer, max_len, device)

        print('sentence', end='\n\n')
        print(tokenizer.decode(source), end='\n\n')
        print("predicted translation", end='\n\n')
        print(translation, end='\n\n')
        print('actual translation', end='\n\n')
        print(tokenizer.decode(target), end='\n\n')


def calculate_bleu(data, model, tokenizer, device, max_len=100):
    """
    Calculates the BLEU score.
    (Uses greedy search as implementation of beam search above with beam_size > 1 didn't provide good translations.)

    :param data: Source and target sentences
    :param model: Model to use for predictions
    :param tokenizer: Tokenizer to prepare dataset sentences
    :param device: CPU or GPU
    :param max_len: Maximum length of source/target sentence
    :return: BLEU score
    """
    targets = []
    pred_targets = []

    for sample in tqdm(data):
        source = sample['source'].tolist()
        target = sample['target'].tolist()

        target = tokenizer.decode(target)

        pred_target, _, _ = greedy_decode(
            source, model, tokenizer, max_len, device)

        pred_targets.append(pred_target.split())
        targets.append([target.split()])

    return bleu_score(pred_targets, targets)


def display_attention(sentence, translation, attention, max_len, n_heads=8, n_rows=4, n_cols=2):
    """
    Displays attention for each head in the multi-headed attention layer and saves plot in attention.jpg.

    :param sentence: Tokenized source sentence (list)
    :param translation: Predicted translation of the sentence
    :param attention: Attention tensor
    :param max_len: Maximum length of source/target sentence
    :param n_heads: Number of heads in the Multi-Head Attention
    :param n_rows: Number of rows to use to display attention
    :param n_cols: Number of columns to use to display attention
    """
    attention = attention[:, :, max_len:, max_len:]
    
    punctuation = ('.',',',';',':','!','?','-','"','$','[',']')
    sentence = [word if word.startswith(punctuation) else word[1:] for word in sentence]
    translation = [word if word.startswith(punctuation) else word[1:] for word in translation]

    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='viridis')
        fig.colorbar(cax)

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + sentence + [''],
                           rotation=90)
        ax.set_yticklabels([''] + translation + [''])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig('attention.jpg')
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

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
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training/evaluation.")
    args = parser.parse_args()

    # get test set
    test = "IWSLT15.TED.tst2012.de-en"
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_en_de.json")
    pad_index = tokenizer.token_to_id("[PAD]")
    test_set = IWSLT(test, tokenizer, max_length=args.max_sent_len)
    if torch.cuda.is_available(): 
        n_workers = 4
    else:
        n_workers = 0
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=n_workers, pin_memory=True)

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

    # instantiate and load model
    vocab_size = tokenizer.get_vocab_size()
    model = DecoderOnlyTransformer(pad_index, vocab_size, args.d_model, args.seq_len,
                                   args.n_heads, args.n_layers, args.ff_dim, args.dropout).to(device)
    model.apply(initialize_weights)
    model.to(device)
    model.load_state_dict(torch.load('model.pt', map_location=device))

    # look at some generated test set translations
    #generate_sample_translations(test_set, model, tokenizer, device, max_len=args.max_sent_len, n=15)

    # plot attention for a translated sentence
    #sentence = test_set[10]['source'].tolist()
    #translation, translation_list, attention = greedy_decode(sentence, model, tokenizer, 128, device)
    #sentence = [tokenizer.id_to_token(i) for i in sentence if i != pad_index ]
    #display_attention(sentence, translation_list, attention, max_len=128, n_heads=8, n_rows=4, n_cols=2)

    # compute test loss
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    # compute BLEU score
    bleu_score = calculate_bleu(test_set, model, tokenizer, device, max_len=args.max_sent_len)
    print(f'BLEU score = {bleu_score*100:.2f}')


if __name__ == "__main__":
    main()

