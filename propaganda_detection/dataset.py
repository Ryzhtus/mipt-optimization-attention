import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class SentenceReader():
    def __init__(self, data):
        self.number_sentences = 0
        self.data = data
        self.grouped = data.groupby("sent_id").apply(lambda sentence: [(w, p) for w, p in zip(sentence["word_corrected"].values.tolist(), sentence["label"].values.tolist())])
        self.sentences = [sentence for sentence in self.grouped]

    def get_next(self):
        try:
            sentence = self.grouped[self.number_sentences]
            self.number_sentences += 1
            return sentence
        except:
            return None


def read_data(filename):
    data = pd.read_csv(filename)
    data = data.fillna(method="ffill")
    data['label'] = data['label'].apply(lambda x: str(list(x)[1]))

    return data


def encoder(sentences, labels, hash_token, tokenizer):
    tokens_all = []
    labels_all = []
    for (sentence, label) in zip(sentences, labels):
        tokens_per_sentence = []
        labels_per_sentence = []
        for word, label in zip(sentence.split(), label):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens_per_sentence.extend(word_tokens)
                labels_per_sentence.extend([label] + [hash_token] * (len(word_tokens) - 1))

        tokens_all.append(tokens_per_sentence)
        labels_all.append(labels_per_sentence)

    return tokens_all, labels_all


def prepare_data(data, max_length):
    reader = SentenceReader(data)
    sentences = [" ".join([idx[0].split()[0] for idx in sentence]) for sentence in reader.sentences]
    labels = [[idx[1] for idx in sentence] for sentence in reader.sentences]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokenized_texts, labels_ids = encoder(sentences, labels, 'X', tokenizer)

    tag2idx = {'0': 0, '1': 1, 'X': 2}
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(text) for text in tokenized_texts], maxlen=max_length,
                              dtype="long", truncating="post", padding="post")

    tags = pad_sequences([[tag2idx.get(idx) for idx in label] for label in labels_ids], maxlen=max_length, value=0, padding="post",
                         dtype="long", truncating="post")

    attention_masks = [[float(idx > 0) for idx in ids] for ids in input_ids]

    train_inputs, eval_inputs, train_tags, eval_tags = train_test_split(input_ids, tags, random_state=42, test_size=0.1)
    train_masks, eval_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.1)

    train_inputs = torch.tensor(train_inputs, dtype=torch.long)
    eval_inputs = torch.tensor(eval_inputs, dtype=torch.long)
    train_tags = torch.tensor(train_tags, dtype=torch.long)
    eval_tags = torch.tensor(eval_tags, dtype=torch.long)
    train_masks = torch.tensor(train_masks, dtype=torch.long)
    eval_masks = torch.tensor(eval_masks, dtype=torch.long)

    return train_inputs, train_tags, train_masks, eval_inputs, eval_tags, eval_masks


def create_dataloader(filename, batch_size, max_length):
    data = read_data(filename)

    train_inputs, train_tags, train_masks, eval_inputs, eval_tags, eval_masks = prepare_data(data, max_length)

    train_data = TensorDataset(train_inputs, train_masks, train_tags)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    eval_data = TensorDataset(eval_inputs, eval_masks, eval_tags)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size)

    return train_dataloader, eval_dataloader