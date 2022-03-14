#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2022-03-14 11:42 
# @Author : Leo

import warnings
import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib.layers import xavier_initializer

from bert.tokenization import load_vocab
from ner.data_utils import process_batch_data, cut_text, texts_digit


class Bilstm_Crf(object):
    def __init__(self, flags, mode='train'):
        self.flags = flags
        self.input = tf.placeholder(tf.int32, shape=[None, None])
        self.target = tf.placeholder(tf.int32, shape=[None, None])
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None, ])
        self.tags, self.id2tag = self.load_tags()
        self.vocab = self.load_vocab()
        if mode != 'train':
            embed = self.embedding_layer()
        else:
            embed = tf.nn.dropout(self.embedding_layer(), keep_prob=0.5)
        if mode != 'train':
            output = self.bilstm_layer(embed)
        else:
            output = tf.nn.dropout(self.bilstm_layer(embed), keep_prob=0.5)
        # 在这里设置一个无偏置的线性层
        self.W = tf.get_variable('W', shape=[2 * self.flags.units, self.flags.num_tags], dtype=tf.float32)
        self.b = tf.get_variable('b', shape=[self.flags.num_tags], dtype=tf.float32)
        matricized_output = tf.reshape(output, [-1, 2 * self.flags.units])
        if mode != 'train':
            matricized_unary_scores = tf.matmul(matricized_output, self.W) + self.b
        else:
            matricized_unary_scores = tf.nn.dropout(tf.matmul(matricized_output, self.W) + self.b, keep_prob=0.5)
        self.scores = tf.reshape(matricized_unary_scores,
                                 [-1, self.flags.max_seq_length, self.flags.num_tags])

        ## softmax
        # self.prediction = tf.cast(tf.argmax(self.scores, axis=-5), tf.int32)
        # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.target)
        # mask = tf.sequence_mask(self.seq_lengths)
        # losses = tf.boolean_mask(losses, mask)
        # self.loss = tf.reduce_mean(losses)
        # self.correct_num = tf.cast(tf.equal(self.target, self.prediction), tf.float32)

        ## crf
        # 计算log-likelihood并获得transition_params
        self.log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.target,
                                                                                   self.sequence_lengths)
        # 进行解码（维特比算法），获得解码之后的序列viterbi_sequence和分数viterbi_score
        self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
            self.scores, transition_params, self.sequence_lengths)

        self.loss = tf.reduce_mean(-self.log_likelihood)

        if mode == 'train':
            self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

        self.correct_num = tf.cast(tf.equal(self.target, self.viterbi_sequence), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_num)

    def load_tags(self):
        tags, id2tag = dict(), dict()

        with open(self.flags.tags_file, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split(':')
                tags[fields[0]] = int(fields[1])
                id2tag[int(fields[1])] = fields[0]
        return tags, id2tag

    def load_vocab(self):
        return load_vocab(self.flags.vocab_file)

    def embedding_layer(self):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.get_variable(name='embedding', hape=[self.flags.vocab_size,
                                                                self.flags.embedding_size],
                                        dtype=tf.float32,
                                        initializer=xavier_initializer())

            embed = tf.nn.embedding_lookup(embedding, self.input)
            return embed

    def bilstm_layer(self, embed):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.flags.units)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.flags.units)
        outputs, final_states = nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=embed,
                                                             dtype=tf.float32)
        # output = tf.concat(outputs, axis=2)
        output_fw, output_bw = outputs

        output = tf.concat([output_fw, output_bw], axis=-1)
        return output

    def train(self):
        def gen_train_data():
            with tf.io.gfile.GFile(self.flags.train_file, 'r') as f:
                for line in f:
                    yield line.strip()

        def gen_evaluate_data():
            with tf.io.gfile.GFile(self.flags.evaluate_file, 'r') as f:
                for line in f:
                    yield line.strip()

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.merge_all()
        saver = tf.train.Saver()
        i = 1
        max_acc = 0.0
        train_dataset = tf.data.Dataset.from_generator(gen_train_data, (tf.string)).shuffle(
            self.flags.batch_size).repeat(2)
        train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset.batch(self.flags.batch_size))
        train_next_element = train_iterator.get_next()

        log_file = tf.io.gfile.GFile(self.flags.log_file, 'w')
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            while True:
                try:
                    texts = sess.run(train_next_element)
                    texts = [text.decode('utf-8') for text in texts]
                    seqs, labels = process_batch_data(texts, self.tags, self.vocab, max_len=self.flags.max_seq_length)
                    train_op, accuracy, loss = sess.run([self.train_op, self.accuracy, self.loss],
                                                        feed_dict={self.input: seqs, self.target: labels,
                                                                   self.sequence_lengths: len(seqs) * [
                                                                       self.flags.max_seq_length]})
                    if i % 100 == 0:
                        evaluate_accuracy, evaluate_loss = 0.0, 0.0
                        batch_num = 0
                        evaluate_dataset = tf.data.Dataset.from_generator(gen_evaluate_data, (tf.string))
                        evaluate_iterator = tf.compat.v1.data.make_one_shot_iterator(
                            evaluate_dataset.batch(self.flags.batch_size))
                        evaluate_next_element = evaluate_iterator.get_next()

                        while True:
                            try:
                                evaluate_texts = sess.run(evaluate_next_element)
                                evaluate_texts = [text.decode('utf-8') for text in evaluate_texts]
                                evaluate_seqs, evaluate_labels = process_batch_data(evaluate_texts, self.tags,
                                                                                    self.vocab,
                                                                                    max_len=self.flags.max_seq_length)
                                _accuracy, _loss = sess.run([self.accuracy, self.loss],
                                                            feed_dict={self.input: evaluate_seqs,
                                                                       self.target: evaluate_labels,
                                                                       self.sequence_lengths: len(evaluate_seqs) * [
                                                                           self.flags.max_seq_length]})
                                evaluate_accuracy += _accuracy
                                evaluate_loss += _loss
                                batch_num += 1
                            except tf.errors.OutOfRangeError:
                                break
                        evaluate_accuracy, evaluate_loss = round((evaluate_accuracy / batch_num), 2), round(
                            (evaluate_loss / batch_num), 2)
                        if evaluate_accuracy > max_acc and i >= 400:
                            saver.save(sess=sess, save_path=self.flags.model_path)
                            tf.summary.FileWriter(self.flags.summary_path, graph=sess.graph)
                            max_acc = evaluate_accuracy
                        log_file.write(
                            '%d batch: train accuracy is %f, train loss is %f; evaluate accuracy is %f, evaluate loss is %f\n' % (
                                i, accuracy, loss, evaluate_accuracy, evaluate_loss))
                    i += 1
                except tf.errors.OutOfRangeError:
                    break

    def predict(self, texts, max_length):
        entity_list = []

        for text in texts:
            _texts = cut_text(text)
            predict_data = texts_digit(_texts, self.vocab)
            sequence_lengths = []
            for ele in predict_data:
                sequence_lengths.append(len(ele))
            predict_data = tf.keras.preprocessing.sequence.pad_sequences(predict_data, value=self.vocab.get('[PAD]'),
                                                                         padding='post', maxlen=max_length)
            tags = []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess=sess,
                              save_path=self.flags.model_path)
                # self.export_model(sess, '../../export_model/10')
                result = sess.run(self.viterbi_sequence,
                                  feed_dict={self.input: predict_data, self.target: np.zeros(shape=predict_data.shape),
                                             self.sequence_lengths: len(predict_data) * [max_length]})
                for row in result:
                    tags.append([self.id2tag[ele] for ele in row])
                entities = []
                for text, tag in zip(_texts, tags):
                    entities.append(self.parse_entities(text, tag))
                entity_tags = set()
                for entity in entities:
                    for ele in entity:
                        entity_tags.add(ele)
                entity_list.append(list(entity_tags))
            return entity_list

    def parse_entities(self, text, tag):
        entities = []

        i = 0
        start = 0
        val = ""
        while i < len(tag):
            if 'B-' in tag[i]:
                val = tag[i][2:]
                start = i
                i += 1
            elif 'I-' in tag[i]:
                while i < len(tag) and 'I-' in tag[i]:
                    i += 1
                words = []
                for ele in text[start:i]:
                    words.append(ele)
                entities.append((''.join(words), val))
            else:
                i += 1
        return entities

    def export_model(self, sess, export_path):
        input = tf.saved_model.utils.build_tensor_info(self.input)

        viterbi_sequence = tf.saved_model.utils.build_tensor_info(self.viterbi_sequence)
        sequence_lengths = tf.saved_model.utils.build_tensor_info(self.sequence_lengths)
        predict_signature = (tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input': input, "sequence_lengths": sequence_lengths},
            outputs={'viterbi_sequence': viterbi_sequence},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={'predict_label': predict_signature},
                                             legacy_init_op=legacy_init_op)
        builder.save()

    def predict2(self, text_list, export_path):
        texts = []

        for text in text_list: texts.extend(cut_text(text))
        predict_data = texts_digit(texts, self.vocab)
        predict_data = tf.keras.preprocessing.sequence.pad_sequences(predict_data, value=self.vocab.get('[PAD]'),
                                                                     padding='post', maxlen=self.flags.max_seq_length)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(graph=tf.Graph(), config=sess_config) as sess:
            meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                                  export_path)
            signature = meta_graph_def.signature_def
            input = signature['predict_label'].inputs['input'].name
            sequence_lengths = signature['predict_label'].inputs["sequence_lengths"].name
            viterbi_sequence = signature['predict_label'].outputs['viterbi_sequence'].name
            input = sess.graph.get_tensor_by_name(input)
            viterbi_sequence = sess.graph.get_tensor_by_name(viterbi_sequence)
            sequence_lengths = sess.graph.get_tensor_by_name(sequence_lengths)
            result = sess.run(viterbi_sequence, feed_dict={input: predict_data, sequence_lengths: len(predict_data) * [
                self.flags.max_seq_length]})
            tags = []
            for row in result:
                tags.append([self.id2tag[ele] for ele in row])
        entities = []
        for text, tag in zip(texts, tags):
            entities.append(self.parse_entities(text, tag))
        return entities
