#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/4/9 9:56 AM
# @Author: lionel
import os


class Config(object):
    def __init__(self, data_path, vocab_path, log_path, model_path):
        self.train_path = os.path.join(data_path, 'train.csv')
        self.dev_path = os.path.join(data_path, 'dev.csv')
        self.test_path = os.path.join(data_path, 'test.csv')
        self.label_path = os.path.join(data_path, 'label.csv')
        self.vocab_path = vocab_path
        self.vocab = self.load_vocab()
        self.vocab_size = len(self.vocab)
        self.labels = self.load_labels()
        self.num_class = len(self.labels)
        self.embed_dim = 256
        self.hidden_dim = 128
        self.lr = 1e-4
        self.batch_size = 200
        self.epochs = 20
        self.num_layers = 10
        self.log_path = log_path
        self.model_path = model_path
        self.drop_out = 0.5
        self.max_seq_length = 200

    def load_labels(self):
        with open(self.label_path, 'r', encoding='utf-8') as f:
            return f.read().strip().split('\n')

    def load_vocab(self):
        vocab = dict()
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                vocab[line.strip()] = len(vocab)
        return vocab
