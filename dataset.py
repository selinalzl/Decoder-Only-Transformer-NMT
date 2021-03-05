#!/usr/bin/env python3

from torch.utils.data import Dataset
import torch


class IWSLT(Dataset):
    """ Class for IWSLT dataset. """

    def __init__(self, dataset_file, tokenizer, max_length=128):
        """
        :param dataset_file: Path to dataset file
        :param tokenizer: Tokenizer to prepare dataset samples
        :param max_length: Maximum character length of source/target sentence
        """

        self.dataset_file = dataset_file
        self.tokenizer = tokenizer

        self.de_en = []
        self.source = []
        self.target = []

        de_file = "iwslt/de-en/" + f"{dataset_file}.de"
        en_file = "iwslt/de-en/" + f"{dataset_file}.en"

        with open(de_file, 'r', encoding="utf-8") as f1, open(en_file, 'r', encoding="utf-8") as f2:
            line_f1 = f1.readlines()
            line_f2 = f2.readlines()
            for line1, line2 in zip(line_f1, line_f2):
                if 0 < len(line1) < max_length and 0 < len(line2) < max_length:
                    input = [line1.strip(), line2.strip()]
                    self.de_en.append(input)
                    self.source.append(line1.strip())
                    self.target.append(line2.strip())

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.de_en)

    def __getitem__(self, i):
        """
        Generates a dictionary for dataset sample at index i. The dictionary contains tensors of the sample,
        encoded source and target sentence and the index of the separator token. The sample consists of 
        encoded source sentence and target sentence that are connected by the separator token [SEP].

        :param i: Index of the dataset sample
        :return: Dictionary for dataset sample
        """

        source_encoded = self.tokenizer.encode([self.source[i]], is_pretokenized=True)
        target_encoded = self.tokenizer.encode([self.target[i]], is_pretokenized=True)
        encoded_sample = source_encoded.ids + [self.tokenizer.token_to_id("[SEP]")] + target_encoded.ids

        sep_token_index = encoded_sample.index(self.tokenizer.token_to_id("[SEP]"))

        return {'sample': torch.tensor(encoded_sample),
                'separator_token': sep_token_index,
                'source': torch.tensor(source_encoded.ids),
                'target': torch.tensor(target_encoded.ids)}

