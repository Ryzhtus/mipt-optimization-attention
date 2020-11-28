import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class ConLL2003Dataset(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.ner_tags = tags

    def len(self):
        return len(self.sentences)

    def __getitem__(self, item):
        words = self.sentences[item]
        tags = self.ner_tags[item]

        words_idx = []
        tags_idx = []

        for word, tag in zip(words, tags):
            tokens = tokenizer.tokenize(word) if word not in ("[CLS]", "[SEP]") else [word]
            tokens_idx = tokenizer.convert_tokens_to_ids(tokens)

            tag = [tag] + ["<PAD>"] * (len(tokens) - 1)
            tag_idx = [tag2idx[each_tag] for each_tag in tag]

            words_idx.extend(tokens_idx)
            tags_idx.extend(tag_idx)

        return words_idx, tags_idx


def read_data(filename):
    rows = open(filename, 'r').read().strip().split("\n\n")
    sentences, sentences_tags = [], []

    for sentence in rows:
        words = [line.split()[0] for line in sentence.splitlines()]
        tags = [line.split()[-1] for line in sentence.splitlines()] # возможно стоит обернуть в скобки ()
        sentences.append(["[CLS]"] + words + ["[SEP]"])
        sentences_tags.append(["<PAD>"] + tags + ["<PAD>"])

    return sentences, sentences_tags


def pad(batch):
    pass

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
TAGS = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')

tag2idx = {tag: idx for idx, tag in enumerate(TAGS)}
idx2tag = {idx: tag for idx, tag in enumerate(TAGS)}


