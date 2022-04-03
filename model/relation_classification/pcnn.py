#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2022-03-18 17:31 
# @Author : Leo
import re
import os
import numpy as np
import tensorflow as tf

## Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks 论文实现
from model.relation_classification.utils import WordEmbeddingLoader, map_label_to_id

base_path = '../../data'
train_file = os.path.join(base_path, 'SemEval/train_data.csv')
test_file = os.path.join(base_path, 'SemEval/test_data.csv')
model_path = os.path.join(base_path, 'relation_model/rc_model')
summary_path = os.path.join(base_path, 'relation_model/summary')

wordEmbed = WordEmbeddingLoader(os.path.join(base_path, 'SemEval/vector_50.txt'), word_dim=50)
word2id, word_vec = wordEmbed.load_embedding()
label2id = map_label_to_id(os.path.join(base_path, 'SemEval/labels.csv'))


class DataProcess(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def data_process(self):
        labels = []
        word_ids_left, pos1_ids_left, pos2_ids_left = [], [], []
        word_ids_mid, pos1_ids_mid, pos2_ids_mid = [], [], []
        word_ids_right, pos1_ids_right, pos2_ids_right = [], [], []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 2:
                    continue
                label, sentence = tuple(fields)
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
                word_ids = []
                for word in words:
                    word = re.sub('(?:<e1>|</e1>|</e2>|<e2>)', '', word)
                    word_ids.append(word2id.get(word, word2id['UNK']))
                    word_list.append(word)

                pos1_ids, pos2_ids = [], []
                for i, word in enumerate(words):
                    if i < entity1_start:
                        pos1_ids.append(i - entity1_start)
                        pos2_ids.append(i - entity2_start)
                    elif entity1_start <= i <= entity1_end:
                        pos1_ids.append(0)
                        pos2_ids.append(i - entity1_start)
                    elif entity1_end < i < entity2_start:
                        pos1_ids.append(i - entity1_end)
                        pos2_ids.append(i - entity2_start)
                    elif entity2_start <= i <= entity2_end:
                        pos1_ids.append(i - entity1_end)
                        pos2_ids.append(0)
                    elif i > entity2_end:
                        pos1_ids.append(i - entity1_end)
                        pos2_ids.append(i - entity2_end)

                word_ids_left.append(word_ids[:entity1_end + 1])
                pos1_ids_left.append(pos1_ids[:entity1_end + 1])
                pos2_ids_left.append(pos2_ids[:entity1_end + 1])

                word_ids_mid.append(word_ids[entity1_end + 1:entity2_end + 1])
                pos1_ids_mid.append(pos1_ids[entity1_end + 1:entity2_end + 1])
                pos2_ids_mid.append(pos2_ids[entity1_end + 1:entity2_end + 1])

                word_ids_right.append(word_ids[entity2_end + 1:])
                pos1_ids_right.append(pos1_ids[entity2_end + 1:])
                pos2_ids_right.append(pos2_ids[entity2_end + 1:])

                labels.append(label2id[label.strip()])

        return labels, word_ids_left, pos1_ids_left, pos2_ids_left, \
               word_ids_mid, pos1_ids_mid, pos2_ids_mid, \
               word_ids_right, pos1_ids_right, pos2_ids_right


class PCNNModel(object):
    def __init__(self, vocab_size, word_dim, pos_num, pos_dim, max_len, num_filters, lr_rate, dropout, label_num,
                 word_vec=None):
        self.word_vec = word_vec
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.pos_num = pos_num
        self.pos_dim = pos_dim
        self.num_filters = num_filters
        self.max_len = max_len
        self.lr_rate = lr_rate
        self.dropout = dropout
        self.label_num = label_num
        self.word_ids_left = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words_ids_left')
        self.pos1_ids_left = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos1_ids_left')
        self.pos2_ids_left = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos2_ids_left')
        self.word_ids_mid = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words_ids_mid')
        self.pos1_ids_mid = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos1_ids_mid')
        self.pos2_ids_mid = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos2_ids_mid')
        self.word_ids_right = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words_ids_right')
        self.pos1_ids_right = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos1_ids_right')
        self.pos2_ids_right = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos2_ids_right')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.label_num], name='labels')
        self.forward()

    def forward(self):
        sentence_embedding_left = self.add_sentence_embedding_op(self.word_ids_left, self.pos1_ids_left,
                                                                 self.pos2_ids_left)
        sentence_embedding_mid = self.add_sentence_embedding_op(self.word_ids_mid, self.pos1_ids_mid,
                                                                self.pos2_ids_mid)
        sentence_embedding_right = self.add_sentence_embedding_op(self.word_ids_right, self.pos1_ids_right,
                                                                  self.pos2_ids_right)

        conv_left = self.add_convolution_op(sentence_embedding_left)
        conv_mid = self.add_convolution_op(sentence_embedding_mid)
        conv_right = self.add_convolution_op(sentence_embedding_right)

        conv = tf.concat([conv_left, conv_mid, conv_right], axis=2)
        conv = tf.reshape(conv, [-1, 3 * self.num_filters])
        _gvector = tf.tanh(conv)
        self.gvector = tf.nn.dropout(_gvector, self.dropout)

        with tf.variable_scope('predict'):
            W = tf.get_variable('W', dtype=tf.float32, shape=[3 * self.num_filters, self.label_num])

            b = tf.get_variable('b', dtype=tf.float32, shape=[self.label_num], initializer=tf.zeros_initializer)

        pred = tf.matmul(self.gvector, W) + b
        self.logits = tf.reshape(pred, [-1, self.label_num])

        self.relations_pred = tf.reshape(tf.cast(tf.argmax(self.logits, axis=-1), tf.int32), [-1])

        """Defines the loss"""
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.labels)
        self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

        global_step = tf.Variable(0, trainable=False, name='step', dtype=tf.int32)
        optimizer = tf.train.AdamOptimizer(self.lr_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # for batch_norm
            self.train_op = optimizer.minimize(self.loss, global_step)
        self.global_step = global_step

    def add_sentence_embedding_op(self, word_ids, pos1_ids, pos2_ids):
        with tf.variable_scope('words', reuse=tf.AUTO_REUSE):
            if not self.word_vec:
                _word_embed = tf.get_variable(name='_word_embed', shape=[self.vocab_size, self.word_dim],
                                              dtype=tf.float32)
            else:
                _word_embed = tf.get_variable(name='_word_embed', initializer=self.word_vec, dtype=tf.float32)
            word_embed = tf.nn.embedding_lookup(_word_embed, word_ids, name='word_embed')

        with tf.variable_scope('pos1', reuse=tf.AUTO_REUSE):
            _pos1_embed = tf.get_variable(name='_pos1_embed', dtype=tf.float32, shape=[self.pos_num, self.pos_dim])

            pos1_embed = tf.nn.embedding_lookup(_pos1_embed, pos1_ids, name='pos1_embed')

        with tf.variable_scope('pos2', reuse=tf.AUTO_REUSE):
            _pos2_embed = tf.get_variable(name='_pos2_embed', dtype=tf.float32, shape=[self.pos_num, self.pos_dim])

            pos2_embed = tf.nn.embedding_lookup(_pos1_embed, pos2_ids, name='pos2_embed')

        sentence_embed = tf.concat([word_embed, pos1_embed, pos2_embed], axis=2)

        sentence_embed = tf.expand_dims(sentence_embed, -1)

        return sentence_embed

    def add_convolution_op(self, sentence_embed):
        input_dim = sentence_embed.shape.as_list()[2]
        pool_outputs = []
        for filter_size in [3, 4, 5]:
            with tf.variable_scope('conv-%s' % filter_size):
                conv_weight = tf.get_variable('W1',
                                              [filter_size, input_dim, 1, self.num_filters],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv_bias = tf.get_variable('b1', [self.num_filters],
                                            initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(sentence_embed,
                                    conv_weight,
                                    strides=[1, 1, input_dim, 1],
                                    padding="SAME")
                conv = tf.nn.relu(conv + conv_bias)  # batch_size, max_len, 1, num_filters
                conv = tf.nn.relu(conv + conv_bias)
                pool = tf.nn.max_pool(conv,
                                      ksize=[1, self.max_len, 1, 1],
                                      strides=[1, self.max_len, 1, 1],
                                      padding="SAME")  # batch_size, 1, 1, num_filters
                pool_outputs.append(pool)
            sentence_feature = tf.reshape(tf.concat(pool_outputs, 3), [-1, 3 * self.num_filters])

        return sentence_feature


if __name__ == '__main__':
    dp = DataProcess(train_file)
    train_labels, train_word_ids_left, train_pos1_ids_left, train_pos2_ids_left, \
    train_word_ids_mid, train_pos1_ids_mid, train_pos2_ids_mid, \
    train_word_ids_right, train_pos1_ids_right, train_pos2_ids_right = dp.data_process()

    dp = DataProcess(test_file)
    test_labels, test_word_ids_left, test_pos1_ids_left, test_pos2_ids_left, \
    test_word_ids_mid, test_pos1_ids_mid, test_pos2_ids_mid, \
    test_word_ids_right, test_pos1_ids_right, test_pos2_ids_right = dp.data_process()

