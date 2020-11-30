import pandas as pd
import numpy as np
import os
import time
import re
from tqdm import tqdm

"""
train_articles_path = 'data/propaganda_detection/datasets/train-articles/'
dev_articles_path = 'data/propaganda_detection/datasets/dev-articles/'
y_path_train = 'data/propaganda_detection/datasets/train-task-si.labels'
y_path_dev = 'data/propaganda_detection/datasets/dev-task-si.labels'"""

train_articles_path = r'C:\Users\shkun\Documents\Учеба\МФТИ\Методы оптимизации\mipt-opt\data\propaganda_detection\datasets\train-articles/'
dev_articles_path = r'C:\Users\shkun\Documents\Учеба\МФТИ\Методы оптимизации\mipt-opt\data\propaganda_detection\datasets\dev-articles/'
y_path_train = r'C:\Users\shkun\Documents\Учеба\МФТИ\Методы оптимизации\mipt-opt\data\propaganda_detection\datasets\train-task-si.labels'
y_path_dev = r'C:\Users\shkun\Documents\Учеба\МФТИ\Методы оптимизации\mipt-opt\data\propaganda_detection\datasets\dev-task-si.labels'


def indices_sentence(article_id, article_path):
    file = open(article_path + 'article' + str(article_id) + '.txt', "r")
    indices = {}
    start_index = 0
    for i, line in enumerate(file):
        indices[i] = {}
        indices[i]['article_id'] = article_id
        indices[i]['span_present'] = 0
        indices[i]['sentence'] = line
        indices[i]['start_index'] = start_index
        indices[i]['end_index'] = start_index + len(line)
        start_index = indices[i]['end_index']

    return indices


def get_label_dataframe(path):
    label_si = pd.read_csv(path, delimiter='\t')
    label_si.loc[-1, :] = [int(i) for i in label_si.columns.tolist()]
    label_si.columns = ['article_id', 'begin_offset', 'end_offset']
    label_si.sort_values(by=['article_id', 'begin_offset'], inplace=True)
    label_si.reset_index(drop=True, inplace=True)
    return label_si


def get_label_test(path):
    label_si = pd.read_csv(path, delimiter='\t')
    _dum = label_si.columns.tolist()
    label_si.loc[-1, :] = [int(_dum[0]), _dum[1], int(_dum[2]), int(_dum[3])]
    label_si.columns = ['article_id', 'tc_class', 'begin_offset', 'end_offset']
    label_si.sort_values(by=['article_id', 'begin_offset'], inplace=True)
    label_si.reset_index(drop=True, inplace=True)
    return label_si[['article_id', 'begin_offset', 'end_offset']]


def get_wordchar_indicies(sent):
    k_l = list(sent)
    k_l_b = [0 if i == ' ' else 1 for i in k_l]
    k_df = pd.DataFrame({'char': k_l, 'space_mark': k_l_b})
    k_df = k_df.reset_index()
    k_df['u1'] = k_df['space_mark'].diff()
    k_df['u1'].fillna(1, inplace=True)
    k_df.loc[k_df['u1'] == 1, 'u2'] = k_df.loc[k_df['u1'] == 1, 'u1'].cumsum()
    k_df.loc[k_df['u1'] == -1, 'u2'] = k_df.loc[k_df['u1'] == -1, 'u1']
    k_df['u2'] = k_df['u2'].ffill(axis=0)
    k_df = k_df[k_df['u2'] != -1]
    k_df_gb = pd.DataFrame(k_df.groupby(['u2'])['index'].min())
    k_df_gb['last_index_word'] = k_df.groupby(['u2'])['index'].max()
    k_df_gb = k_df_gb.reset_index().rename(columns={'u2': 'word_index', 'index': 'first_index_word'})
    try:
        k_df_gb['words'] = sent.split()
    except:
        print(k_df_gb, sent)
    return k_df_gb


def get_sentence_data(article_id, data_path, article_path):
    se_dict = indices_sentence(article_id, article_path)
    label_df = get_label_dataframe(data_path)
    req_labels = label_df.loc[label_df['article_id'] == article_id, :]
    req_labels.reset_index(drop=True, inplace=True)

    sentence_ids = []
    for key, value in se_dict.items():

        if value['sentence'] == '\n':
            se_dict[key]['word_st_index'] = [0]
            se_dict[key]['word_en_index'] = [0]
        else:
            wordchar_df = get_wordchar_indicies(value['sentence'])
            se_dict[key]['word_st_index'] = list(wordchar_df['first_index_word'])
            se_dict[key]['word_en_index'] = list(wordchar_df['last_index_word'])

        se_dict[key]['Y'] = np.zeros((len(value['sentence'].split()), 1))

        k = []
        for b, e in zip(req_labels['begin_offset'], req_labels['end_offset']):
            _value = value.copy()
            if ((value['start_index'] <= b) & (value['end_index'] >= e)):
                sentence_ids.append([key])

            if (((_value['start_index'] <= b) & (_value['end_index'] > b)) & (_value['end_index'] < e)):

                k.append(key)
                key = key + 1
                _value = se_dict[key]
                if (((_value['start_index'] <= e) & (_value['end_index'] > e)) & (_value['start_index'] > b)):
                    k.append(key)
                else:
                    while ((_value['start_index'] > b) & (_value['end_index'] < e)):
                        k.append(key)
                        key = key + 1
                        _value = se_dict[key]
                    else:
                        k.append(key)

                sentence_ids.append(k)

    if len(sentence_ids) == len(req_labels):

        req_labels.loc[:, 'sentence_id'] = sentence_ids

        try:
            req_labels.loc[:, 'len_s_id'] = req_labels.loc[:, 'sentence_id'].apply(lambda x: len(x))
        except:
            req_labels.loc[:, 'len_s_id'] = 1

        multiple_spans = req_labels[req_labels['len_s_id'] > 1]['sentence_id'].tolist()
        for _span_id in multiple_spans:
            _sentence = se_dict[_span_id[0]]['sentence']
            _st_in = se_dict[_span_id[0]]['start_index']

            _Y = se_dict[_span_id[0]]['Y']
            _st_word = se_dict[_span_id[0]]['word_st_index']
            _en_word = se_dict[_span_id[0]]['word_en_index']

            for _id in _span_id[1:]:
                se_dict[_id]['only_part_span'] = 1
                _sentence += se_dict[_id]['sentence']
                _en_in = se_dict[_id]['end_index']
                _Y = np.concatenate([_Y, se_dict[_id]['Y']], axis=0)
                _st_word += se_dict[_id]['word_st_index']
                _en_word += se_dict[_id]['word_en_index']

            se_dict[_span_id[0]]['sentence'] = _sentence
            se_dict[_span_id[0]]['end_index'] = _en_in
            se_dict[_span_id[0]]['Y'] = _Y
            se_dict[_span_id[0]]['word_st_index'] = _st_word
            se_dict[_span_id[0]]['word_en_index'] = _en_word
    else:
        print(sentence_ids, "Length of sentence ids is not same as req_labels")

    for i, s_id_list in enumerate(req_labels['sentence_id']):
        try:
            s_id = s_id_list[0]
        except:
            s_id = s_id_list

        if 'only_part_span' in se_dict[s_id].keys():
            se_dict[s_id].pop('only_part_span')

        se_dict[s_id]['span_present'] = 1
        sentence = se_dict[s_id]['sentence']
        span_start = req_labels.loc[i, 'begin_offset'] - se_dict[s_id]['start_index']
        span_end = req_labels.loc[i, 'end_offset'] - se_dict[s_id]['start_index']
        span_start = int(span_start)
        span_end = int(span_end)
        s1 = sentence[0:span_start]
        s2 = sentence[span_start:span_end]
        tail_len = len(s1.split())
        tail_san_len = len(s1.split()) + len(s2.split())
        if tail_san_len > len(sentence.split()):
            tail_san_len = tail_san_len - 1
            tail_len = tail_len - 1
        for n in range(tail_len, tail_san_len):
            se_dict[s_id]['Y'][n, 0] = 1
    return se_dict, req_labels


def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    text = ' '.join(text.split())
    return (text)


def get_prop_df(dict_all):
    start = time.time()
    print(time.asctime())
    X = []
    Y =[]
    Z = []
    st = []
    en = []
    sn_st = []
    sn_en = []
    for _dict in dict_all:
        se_dict = _dict
        for i in range(len(se_dict)):
            X.append(se_dict[i]['sentence'])
            Y.append(se_dict[i]['Y'])
            Z.append(se_dict[i]['article_id'])
            st.append(se_dict[i]['word_st_index'])
            en.append(se_dict[i]['word_en_index'])
            sn_st.append(se_dict[i]['start_index'])
            sn_en.append(se_dict[i]['end_index'])

    df = pd.DataFrame()
    df['sent'] = X
    df['label'] = Y
    df['article'] = Z
    df['word_st_index'] = st
    df['word_en_index'] = en
    df['sent_start_index'] = sn_st
    df['sent_end_index'] = sn_en
    df1 = df.copy()
    df1.index = range(0,len(df1))
    word = []
    sent_id = []
    article_sent_id = []
    art_id = []
    lab_id = []
    word_st = []
    word_en = []
    sent_start_index = []
    sent_end_index = []

    for i in tqdm(range(0, len(df1))):
        sent_ls = df1['sent'].iloc[i].split()
        art = df1['article'].iloc[i]
        sent_start = df1['sent_start_index'].iloc[i]
        sent_end = df1['sent_end_index'].iloc[i]
        lab = df1['label'].iloc[i]
        wst = df1['word_st_index'].iloc[i]
        wen = df1['word_en_index'].iloc[i]
        article_sent_id += list(range(len(sent_ls)))

        for j in range(0, len(sent_ls)):
            word.append(sent_ls[j])
            lab_id.append(lab[j])
            sent_id.append(i)
            sent_start_index.append(sent_start)
            sent_end_index.append(sent_end)
            art_id.append(art)
            word_st.append(wst[j])
            word_en.append(wen[j])

    df_final = pd.DataFrame()
    df_final['article_id'] = art_id
    df_final['sent_id'] = sent_id
    df_final['word'] = word
    df_final['label'] = lab_id
    df_final['word_st_index'] = word_st
    df_final['word_en_index'] = word_en
    df_final['sent_start_index'] = sent_start_index
    df_final['sent_end_index'] = sent_end_index
    df_final['word_corrected'] = df_final['word'].apply(lambda x: text_preprocessing(x))
    df_final_corr = df_final[df_final['word_corrected'] != '']
    print('shape of df_final_corr: ', df_final_corr.shape)
    print(round((time.time()-start)/60, 2), "Minutes lapsed")
    return df_final_corr


train_article_ids = [int(file.replace('article', '').replace('.txt', '')) for file in os.listdir(train_articles_path)]
dev_article_ids = [int(file.replace('article', '').replace('.txt', '')) for file in os.listdir(dev_articles_path)]

start = time.time()

train_dict_all = []
for req_id in train_article_ids:
    print(req_id)
    se_dict, req_labels = get_sentence_data(req_id, y_path_train, train_articles_path)
    train_dict_all.append(se_dict)
print(round((time.time() - start) / 60, 2), "Minutes lapsed")

start = time.time()
dev_dict_all = []
for req_id in dev_article_ids:
    print(req_id)
    se_dict, req_labels = get_sentence_data(req_id, y_path_dev, dev_articles_path)
    dev_dict_all.append(se_dict)
print(round((time.time() - start) / 60, 2), "Minutes lapsed")

get_prop_df(train_dict_all).to_csv('data/propaganda_detection/datasets/train_articles.csv', index=False)
get_prop_df(dev_dict_all).to_csv('data/propaganda_detection/datasets/eval_articles.csv', index=False)