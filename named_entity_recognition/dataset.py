import torch
from torch.utils.data import Dataset, DataLoader


class ConLL2003Dataset(Dataset):
    def __init__(self, sentences, tags, tags_number, tokenizer, max_length):
        self.sentences = sentences
        self.ner_tags = tags
        self.tags_number = tags_number
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        words = str(self.sentences[item])
        tags = self.ner_tags[item]

        encoding = self.tokenizer.encode_plus(words, add_special_tokens=True, max_length=self.max_length,
                                              return_token_type_ids=False, pad_to_max_length=True,
                                              return_attention_mask=True, return_tensors='pt', truncation=True)

        tags_idx = [tag2idx[each_tag] for each_tag in tags]
        tags_idx_padded = [tags_idx + [0] * (self.max_length - len(tags_idx))]

        return {'words': words, 'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'tags': torch.tensor(tags_idx_padded, dtype=torch.long)}


def read_data(filename):
    rows = open(filename, 'r').read().strip().split("\n\n")
    sentences, sentences_tags = [], []

    for sentence in rows:
        words = [line.split()[0] for line in sentence.splitlines()]
        tags = [line.split()[-1] for line in sentence.splitlines()]
        sentences.append(words)
        sentences_tags.append(tags)

    tags_number = sum([len(tag) for tag in sentences_tags])

    return sentences, sentences_tags, tags_number


def create_dataset_and_dataloader(filename, batch_size, tokenizer, max_length):
    sentences, tags, tags_number = read_data(filename)
    dataset = ConLL2003Dataset(sentences, tags, tags_number, tokenizer, max_length)

    return dataset, DataLoader(dataset, batch_size, num_workers=4),


TAGS = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
MAX_LENGTH = 126
tag2idx = {tag: idx for idx, tag in enumerate(TAGS)}
idx2tag = {idx: tag for idx, tag in enumerate(TAGS)}
