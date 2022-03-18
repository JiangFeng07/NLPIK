#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2022-03-16 23:07 
# @Author : Leo
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import re

base_path = '../../data/SemEval'


class WordEmbeddingLoader(object):
    def __init__(self, embedding_path, word_dim):
        self.embedding_path = embedding_path
        self.word_dim = word_dim

    def load_embedding(self):
        word2id = dict()
        word_vec = list()

        word2id['PAD'] = len(word2id)
        pad_emb = np.zeros(shape=[self.word_dim], dtype=np.float32)
        word_vec.append(pad_emb)
        word2id['UNK'] = len(word2id)
        pad_emb = np.zeros(shape=[self.word_dim], dtype=np.float32)
        word_vec.append(pad_emb)
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split(' ')
                if len(fields) != self.word_dim + 1:
                    continue
                word, embed = fields[0], fields[1:]
                word2id[word] = len(word2id)
                word_vec.append(np.asarray(embed, dtype=np.float32))
        word_vec = np.asarray(word_vec)
        return word2id, word_vec


# class SemEvalDataset(object):
#     def __init__(self, train_path, test_path):
#         self.train_path = train_path
#         self.test_path = test_path
def map_label_to_id(label_file):
    label2id = dict()
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            label2id[line.strip()] = len(label2id)
    return label2id


def entity_context(entity_idx, words):
    context = []
    if entity_idx >= 1:
        context.append(words[entity_idx - 1])
    else:
        context.append(words[entity_idx])
    context.append(words[entity_idx])

    if entity_idx < len(words) - 1:
        context.append(words[entity_idx + 1])
    else:
        context.append(words[entity_idx])

    return context


def distance(n):
    if n < -60:
        return 0
    elif -60 <= n <= 60:
        return n + 61

    return 122


def process_batch_data(sentences, label2id, word2id, max_len=128):
    texts, pos1, pos2, labels, contexts = [], [], [], [], []
    for ele in sentences:
        fields = ele.strip().split('\t')
        if len(fields) != 2:
            continue
        sentence = fields[-1].lower()
        words = sentence.split(' ')
        entity1_start, entity1_end, entity2_start, entity2_end = -1, -1, -1, -1
        for i, word in enumerate(words):
            if '<e1>' in word and entity1_start == -1:
                entity1_start = i
            if '<e2>' in word and entity2_start == -1:
                entity2_start = i
            if '</e1>' in word and entity1_end == -1:
                entity1_end = i
            if '</e2>' in word and entity2_end == -1:
                entity2_end = i
        word_list = []
        for word in words:
            word = re.sub('(?:<e1>|</e1>|</e2>|<e2>)', '', word)
            word_list.append(word)

        entity_contexts = []
        for word in entity_context(entity1_start, word_list):
            entity_contexts.append(word2id.get(word, word2id['UNK']))
        for word in entity_context(entity2_start, word_list):
            entity_contexts.append(word2id.get(word, word2id['UNK']))
        contexts.append(entity_contexts)
        if len(word_list) > max_len:
            word_list = word_list[:max_len]
        while len(word_list) < max_len:
            word_list.append('PAD')

        _texts, _pos1, _pos2 = [], [], []
        for i, word in enumerate(word_list):
            _texts.append(word2id.get(word, word2id['UNK']))
            if i < entity1_start:
                _pos1.append(i - entity1_start)
                _pos2.append(i - entity2_start)
            elif entity1_start <= i <= entity1_end:
                _pos1.append(0)
                _pos2.append(i - entity1_start)
            elif entity1_end < i < entity2_start:
                _pos1.append(i - entity1_end)
                _pos2.append(i - entity2_start)
            elif entity2_start <= i <= entity2_end:
                _pos1.append(i - entity1_end)
                _pos2.append(0)
            elif i > entity2_end:
                _pos1.append(i - entity1_end)
                _pos2.append(i - entity2_end)

        texts.append(_texts)
        pos1.append([distance(ele) for ele in _pos1])
        pos2.append([distance(ele) for ele in _pos2])
        labels.append(label2id[fields[0].strip()])

    return texts, pos1, pos2, labels, contexts


if __name__ == '__main__':
    wordEmbed = WordEmbeddingLoader(os.path.join(base_path, 'vector_50.txt'), word_dim=50)
    word2id, word_vec = wordEmbed.load_embedding()
    train_texts = pd.read_csv(os.path.join(base_path, 'train/train.txt'), sep='\t', header=None)
    train_labels = pd.read_csv(os.path.join(base_path, 'train/train_result_full.txt'), sep='\t', header=None)

    texts, pos1, pos2, labels, contexts = [], [], [], [], []
    pos_num = -1
    label2id = map_label_to_id(os.path.join(base_path, 'labels.csv'))

    sentences, tags = [], []

    with open(os.path.join(base_path, 'test_data.csv'), 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(line.strip())
    texts, pos1, pos2, labels, contexts = process_batch_data(sentences[:1], label2id)

    for i in range(len(texts)):
        print(texts[i])
        print(pos1[i])
        print(pos2[i])
        print(labels[i])
        print(contexts[i])
        print('\n')
