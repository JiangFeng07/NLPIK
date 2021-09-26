#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2021-09-23 11:11 
# @Author : Leo
import re
import tensorflow as tf
from tensorflow import nn

from bert import modeling, tokenization
from bert.run_classifier import DataProcessor
from bert.tokenization import load_vocab


class InputExample(object):
    """A single training/test example for simple sequence name entity recognition."""

    def __init__(self, guid, text, labels=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class NerProcessor(DataProcessor):
    def get_train_examples(self, texts):
        i = 0
        examples = []
        for text in texts:
            fields = text.decode('utf-8').split('\t')
            if len(fields) != 2:
                continue
            label, text = tuple(fields)
            guid = "train-%d" % (i)
            text = tokenization.convert_to_unicode(text)
            label = tokenization.convert_to_unicode(label)
            examples.append(InputExample(guid, text, label))
        return examples

    def get_dev_examples(self, data_dir):
        pass

    def get_test_examples(self, data_dir):
        pass

    def get_labels(self):
        pass


def convert_batch_example(examples, max_seq_length, vocab, label_dicts):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    input_ids, input_mask, segment_ids, label_ids = [], [], [], []
    for example in examples:
        labels, text = example.labels, example.text

        _input_ids, _label_ids, tokens = [], [], []
        tokens.append('[CLS]')
        for token in list(text):
            tokens.append(token)
        tokens.append('[SEP]')
        _input_ids.append(vocab['[CLS]'])
        _label_ids.append(label_dicts['O'])
        tmp_input_ids, tmp_label_ids = convert_tokens_labels_to_ids(text, labels, vocab, label_dicts)
        _input_ids.extend(tmp_input_ids)
        _label_ids.extend(tmp_label_ids)
        _input_ids.append(vocab['[SEP]'])
        _label_ids.append(label_dicts['O'])
        _input_mask = [1] * len(_input_ids)
        while len(_input_ids) < max_seq_length:
            _input_ids.append(0)
            _input_mask.append(0)
            _label_ids.append(label_dicts['O'])

        _segment_ids = [0] * len(_input_ids)

        assert len(_input_ids) == max_seq_length
        assert len(_input_mask) == max_seq_length
        assert len(_segment_ids) == max_seq_length
        assert len(_label_ids) == max_seq_length
        input_ids.append(_input_ids)
        input_mask.append(_input_mask)
        segment_ids.append(_segment_ids)
        label_ids.append(_label_ids)
    return input_ids, input_mask, segment_ids, label_ids


def convert_tokens_labels_to_ids(text, labels, vocab, label_dicts):
    input_ids = convert_by_vocab(vocab, text)
    label_list = labels.split(';')
    label_map = {}
    for label in label_list:
        fields = label.split(':')
        if len(fields) != 2:
            continue
        label_map[fields[0]] = fields[1]
    label_ids = len(input_ids) * [label_dicts['O']]

    for key, val in label_map.items():
        for ele in re.finditer(key, text):
            start_index = ele.span()[0]
            end_index = ele.span()[1]
            label_ids[start_index] = label_dicts['B-' + val]
            while start_index + 1 < end_index:
                label_ids[start_index + 1] = label_dicts['I-' + val]
                start_index += 1
    return input_ids, label_ids


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab.get(item, vocab.get('[UNK]')))
    return output


def cut_text(context, max_seq_length=350):
    text_list = []
    if not context:
        return text_list
    tags = '。！!？?；;'
    while len(context) > max_seq_length - 2:
        if context[max_seq_length - 2] in tags:
            text_list.append(context[:max_seq_length - 2])
            context = context[max_seq_length - 2:]
        else:
            tmp_text = context[:max_seq_length - 2]
            arr = re.split('[%s]' % tags, tmp_text)
            if len(arr) > 1:
                index = len(''.join(arr[:-1])) + len(arr) - 1
            else:
                index = max_seq_length - 2
            text_list.append(context[:index])
            context = context[index:]
    if len(context) > 0:
        text_list.append(context)
    return text_list


class BertCRF(object):
    def __init__(self, is_training=True, units=100, num_labels=3, max_seq_length=128, log_file_path='log.csv',
                 model_path='./model/ber_ner_model', summary_path='tensorboard'):
        self.units = units
        self.is_training = is_training
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self.log_file_path = log_file_path
        self.model_path = model_path
        self.summary_path = summary_path
        self.bert_config = modeling.BertConfig.from_json_file(
            '/Users/jiangfeng/workspace_me/NLPIK/bert-chinese/chinese_L-12_H-768_A-12/bert_config.json')

        self.vocab = load_vocab('/Users/jiangfeng/workspace_me/NLPIK/bert-chinese/chinese_L-12_H-768_A-12/vocab.txt')
        self.label_dicts = {'B-PER': 0, 'I-PER': 1, 'O': 2}

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
        self.label_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='label_id')
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')

        self.lengths = tf.reduce_sum(self.input_mask, axis=1)
        self.batch_size = tf.shape(self.input_ids)[0]

        bert_embedding = self.bert_embedding()
        bilstm_layer = self.bilstm_layer(bert_embedding)
        dense_layer = tf.keras.layers.Dense(num_labels)(bilstm_layer)

        self.scores = tf.reshape(dense_layer, [-1, self.max_seq_length, self.num_labels])

        ## crf
        # 计算log-likelihood并获得transition_params
        self.log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.label_ids,
                                                                                   self.lengths)
        # 进行解码（维特比算法），获得解码之后的序列viterbi_sequence和分数viterbi_score
        self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.scores, transition_params,
                                                                         self.lengths)
        self.loss = tf.reduce_mean(-self.log_likelihood)
        if is_training:
            self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        self.correct_num = tf.cast(tf.equal(self.label_ids, self.viterbi_sequence), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_num)

    def bert_embedding(self):
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        embedding = bert_model.get_sequence_output()
        return embedding

    def bilstm_layer(self, embed):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.units)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.units)
        outputs, final_states = nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=embed,
                                                             dtype=tf.float32)
        output_fw, output_bw = outputs
        output = tf.concat([output_fw, output_bw], axis=-1)
        return output

    def train(self):
        def gen_train_data():
            with tf.io.gfile.GFile('train.csv', 'r') as f:
                for line in f:
                    yield line.strip()

        def gen_evaluate_data():
            with tf.io.gfile.GFile('evaluate.csv', 'r') as f:
                for line in f:
                    yield line.strip()

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.merge_all()
        saver = tf.train.Saver()

        ner_processor = NerProcessor()

        i = 1
        max_acc = 0.0
        train_dataset = tf.data.Dataset.from_generator(gen_train_data, (tf.string)).repeat(2)
        train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset.batch(10))
        train_next_element = train_iterator.get_next()
        log_file = tf.io.gfile.GFile(self.log_file_path, 'w')
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            while True:
                try:
                    texts = sess.run(train_next_element)
                    examples = ner_processor.get_train_examples(texts)
                    input_ids, input_mask, segment_ids, label_ids = convert_batch_example(examples,
                                                                                          max_seq_length=self.max_seq_length,
                                                                                          vocab=self.vocab,
                                                                                          label_dicts=self.label_dicts)

                    train_op, accuracy, loss = sess.run([self.train_op, self.accuracy, self.loss],
                                                        feed_dict={self.input_ids: input_ids,
                                                                   self.input_mask: input_mask,
                                                                   self.segment_ids: segment_ids,
                                                                   self.label_ids: label_ids,
                                                                   self.dropout: 0.5})

                    if i % 1 == 0:
                        evaluate_accuracy, evaluate_loss = 0.0, 0.0
                        batch_num = 0
                        evaluate_dataset = tf.data.Dataset.from_generator(gen_evaluate_data, (tf.string))
                        evaluate_iterator = tf.compat.v1.data.make_one_shot_iterator(
                            evaluate_dataset.batch(10))
                        evaluate_next_element = evaluate_iterator.get_next()
                        while True:
                            try:
                                texts = sess.run(evaluate_next_element)
                                examples = ner_processor.get_train_examples(texts)
                                input_ids, input_mask, segment_ids, label_ids = convert_batch_example(examples,
                                                                                                      max_seq_length=self.max_seq_length,
                                                                                                      vocab=self.vocab,
                                                                                                      label_dicts=self.label_dicts)

                                _accuracy, _loss = sess.run([self.accuracy, self.loss],
                                                            feed_dict={self.input_ids: input_ids,
                                                                       self.input_mask: input_mask,
                                                                       self.segment_ids: segment_ids,
                                                                       self.label_ids: label_ids,
                                                                       self.dropout: 1.0})
                                evaluate_accuracy += _accuracy
                                evaluate_loss += _loss
                                batch_num += 1
                            except tf.errors.OutOfRangeError:
                                break
                        evaluate_accuracy, evaluate_loss = round((evaluate_accuracy / batch_num), 2), round(
                            (evaluate_loss / batch_num), 2)
                        if evaluate_accuracy > max_acc and i >= 400:
                            saver.save(sess=sess, save_path=self.model_path)
                            tf.summary.FileWriter(self.summary_path, graph=sess.graph)
                            max_acc = evaluate_accuracy
                        log_file.write(
                            '%d batch: train accuracy is %f, train loss is %f; evaluate accuracy is %f, evaluate loss is %f\n' % (
                                i, accuracy, loss, evaluate_accuracy, evaluate_loss))
                    i += 1
                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    bert_crf = BertCRF()
    bert_crf.train()
