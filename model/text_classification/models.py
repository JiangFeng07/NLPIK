#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/4/10 9:56 AM
# @Author: lionel
import tensorflow as tf
from tensorflow import keras


class TextRnnModel(keras.Model):
    def __init__(self, config, model='bi_direct'):
        super(TextRnnModel, self).__init__()
        self.embedding = keras.layers.Embedding(config.vocab_size, config.embed_dim, mask_zero=True)
        if model == 'bi_direct':
            self.rnn = keras.layers.Bidirectional(keras.layers.LSTM(config.hidden_dim))
        elif model == 'lstm':
            self.rnn = keras.layers.LSTM(config.hidden_dim)
        elif model == 'gru':
            self.rnn = keras.layers.GRU(config.hidden_dim)
        self.fc = keras.layers.Dense(config.num_class, activation=tf.nn.softmax)

    def call(self, inputs):
        inputs = self.embedding(inputs)
        inputs = self.rnn(inputs)
        inputs = keras.layers.Dropout(rate=0.5)(inputs)
        out = self.fc(inputs)
        return out
