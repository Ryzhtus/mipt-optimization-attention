import os
import csv


def read_articles(data_directory, article_directory):
    articles = []
    article_ids = []
    train_directory = os.path.join(data_directory, article_directory)

    for filename in sorted(os.listdir(train_directory)):
        file = open(os.path.join(train_directory, filename))
        document = file.read()
        articles.append(document)
        file.close()

    for filename in sorted(os.listdir(train_directory)):
        article_ids.append(filename[7:-4])

    return articles, article_ids


def read_spans(data_directory, span_directory=None):
    spans = []
    if span_directory is None:
        label_directory = os.path.join(data_directory, "train-labels-task-si")
    else:
        label_directory = os.path.join(data_directory, span_directory)
    for filename in sorted(os.listdir(label_directory)):
        file = open(os.path.join(label_directory, filename))
        tsv_reader = csv.reader(file, delimiter="\t")
        span = []
        for row in tsv_reader:
            span.append((int(row[1]), int(row[2])))
        file.close()
        spans.append(span)

    return spans


def is_whitespace(char):
    whitespaces = [" ", "\t", "\r", "\n"]
    if char in whitespaces or ord(char) == 0x202F:
        return True
    return False

def get_list_from_dict(number_sentences, word_offsets):
    data = []
    for _ in range(number_sentences):
        data.append([])
    for key in word_offsets:
        si = key[0]
        data[si].append(word_offsets[key])

    return data


def get_sentence_tokens_labels(article, span=None, article_index=None):
    doc_tokens = []
    char_to_word_offset = []
    current_sentence_tokens = []
    word_to_start_char_offset = {}
    word_to_end_char_offset = {}
    prev_is_whitespace = True
    prev_is_newline = True
    current_word_position = None
    for index, c in enumerate(article):
        if c == "\n":
            prev_is_newline = True
            # check for empty lists
            if doc_tokens:
                current_sentence_tokens.append(doc_tokens)
            doc_tokens = []
        if is_whitespace(c):
            prev_is_whitespace = True
            if current_word_position is not None:
                word_to_end_char_offset[current_word_position] = index
                current_word_position = None
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
                current_word_position = (len(current_sentence_tokens), len(doc_tokens) - 1)
                word_to_start_char_offset[current_word_position] = index  # start offset of word
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append((len(current_sentence_tokens), len(doc_tokens) - 1))
    if doc_tokens:
        current_sentence_tokens.append(doc_tokens)
    if current_word_position is not None:
        word_to_end_char_offset[current_word_position] = index
        current_word_position = None
    if span is None:
        return current_sentence_tokens, (word_to_start_char_offset, word_to_end_char_offset)

    current_propaganda_labels = []
    for doc_tokens in current_sentence_tokens:
        current_propaganda_labels.append([0] * len(doc_tokens))

    start_positions = []
    end_positions = []

    for sp in span:
        if (char_to_word_offset[sp[0]][0] != char_to_word_offset[sp[1] - 1][0]):
            l1 = char_to_word_offset[sp[0]][0]
            l2 = char_to_word_offset[sp[1] - 1][0]
            start_positions.append(char_to_word_offset[sp[0]])
            end_positions.append((l1, len(current_sentence_tokens[l1]) - 1))
            l1 += 1
            while (l1 < l2):
                start_positions.append((l1, 0))
                end_positions.append((l1, len(current_sentence_tokens[l1]) - 1))
                l1 += 1
            start_positions.append((l2, 0))
            end_positions.append(char_to_word_offset[sp[1] - 1])
            continue
        start_positions.append(char_to_word_offset[sp[0]])
        end_positions.append(char_to_word_offset[sp[1] - 1])

    for i, s in enumerate(start_positions):
        assert start_positions[i][0] == end_positions[i][0]

        current_propaganda_labels[start_positions[i][0]][start_positions[i][1]] = 2  # Begin label
        if start_positions[i][1] < end_positions[i][1]:
            current_propaganda_labels[start_positions[i][0]][start_positions[i][1] + 1: end_positions[i][1]] = \
                [1] * (end_positions[i][1] - start_positions[i][1] - 1)
            current_propaganda_labels[start_positions[i][0]][end_positions[i][1]] = 3  # End label


    num_sentences = len(current_sentence_tokens)

    start_offset_list = get_list_from_dict(num_sentences, word_to_start_char_offset)
    end_offset_list = get_list_from_dict(num_sentences, word_to_end_char_offset)
    sentences = []
    for i in range(num_sentences):
        sentence = Sentence(current_sentence_tokens[i], current_propaganda_labels[i],
                            article_index, i, start_offset_list[i], end_offset_list[i])

        num_words = len(sentence.tokens)
        assert len(sentence.labels) == num_words
        assert len(sentence.word_to_start_char_offset) == num_words
        assert len(sentence.word_to_end_char_offset) == num_words
        sentences.append(sentence)

    return current_sentence_tokens, current_propaganda_labels, (
    word_to_start_char_offset, word_to_end_char_offset), sentences