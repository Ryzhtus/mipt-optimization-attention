from named_entity_recognition.dataset import ConLL2003Dataset, read_data

train_sentences, train_tags = read_data('data/named_entity_recognition/train.txt')
train_dataset = ConLL2003Dataset(train_sentences, train_tags)

for i in range(10):
    print(i, train_dataset[i])