#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/4/10 9:56 AM
# @Author: lionel
import os

import tensorflow as tf
from tensorflow import keras

from model.text_classification.config import Config


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


class TextCnnModel(keras.Model):
    def __init__(self, config):
        super(TextCnnModel, self).__init__()
        self.embedding = keras.layers.Embedding(config.vocab_size, config.embed_dim, mask_zero=True)
        self.fc = keras.layers.Dense(config.num_class, activation=tf.nn.softmax)

    def call(self, inputs):
        inputs = self.embedding(inputs)
        cnn_list = []
        for ele in [3, 4, 5]:
            conv = keras.layers.Conv1D(filters=256, kernel_size=ele, padding='same', strides=1,
                                       activation='relu')(inputs)
            max_poll = keras.layers.MaxPool1D(pool_size=4)(conv)
            cnn_list.append(max_poll)
        cnn = keras.layers.concatenate(cnn_list, axis=-1)
        flatten = keras.layers.Flatten(name='flat')(cnn)
        inputs = keras.layers.Dropout(rate=0.5)(flatten)
        out = self.fc(inputs)
        return out


if __name__ == '__main__':
    data_path = '/Users/jiangfeng/Workspace/Data/Classifier/cnews'
    config = Config(data_path, vocab_path=os.path.join("/Users/jiangfeng/Downloads", 'vocab.txt'), log_path='log.csv',
                    model_path='cnn_model')
    text_cnn_model = TextCnnModel(config=config)
