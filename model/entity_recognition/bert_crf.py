#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2021-09-23 11:11 
# @Author : Leo
import os
import re
import tensorflow as tf
from tensorflow import nn

from model.bert import tokenization
from model.bert import modeling
# from model.bert import AdamWeightDecayOptimizer
from model.bert.optimization import AdamWeightDecayOptimizer
from model.bert.run_classifier import DataProcessor
from model.bert.tokenization import load_vocab

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("train_file", '', "SQuAD json for training. E.g., train_data.csv")
flags.DEFINE_string("evaluate_file", '', "SQuAD json for predictions. E.g., train_data.csv")
flags.DEFINE_string("bert_config_file",
                    r'D:\judgement-nlp-server\bert-chinese\chinese_L-12_H-768_A-12\bert_config.json',
                    "The config json file corresponding to the pre-trained BERT model. " "This specifies the model architecture.")
flags.DEFINE_string("vocab_file", r'D:\judgement-nlp-server\bert-chinese\chinese_L-12_H-768_A-12\vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("max_seq_length", 128,
                     "The maximum total input sequence length after WordPiece tokenization. " "Sequences longer than this will be truncated, and sequences shorter " "than this will be padded.")
flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10% of training.")
flags.DEFINE_string("init_checkpoint", r'D:\judgement-nlp-server\bert-chinese\chinese_L-12_H-768_A-12\bert_model.ckpt',
                    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string("output_dir", '../../', "The output directory where the model checkpoints will be written.")


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
            # fields = text.decode('utf-8').split('\t')
            fields = text.split('\t')
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

    def get_test_examples(self, texts):
        i = 0
        examples = []
        for text in texts:
            guid = "test-%d" % (i)
            text = tokenization.convert_to_unicode(text)
            label = tokenization.convert_to_unicode('')
            examples.append(InputExample(guid, text, label))
        return examples

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
        try:
            tmp_input_ids, tmp_label_ids = convert_tokens_labels_to_ids(text, labels, vocab, label_dicts)
        except:
            continue
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


def count_train_file(file_path):
    count = 0
    with tf.io.gfile.GFile(file_path, 'r') as f:
        for _ in f:
            count += 1
    return count


class BertCRF(object):

    def __init__(self, is_training=True, units=100, log_file_path='log.csv'):
        self.units = units

        self.is_training = is_training
        self.log_file_path = log_file_path
        self.global_step = tf.Variable(0, trainable=False)
        self.model_path = os.path.join(FLAGS.output_dir, 'model/bert_ner_model_final')
        self.summary_path = os.path.join(FLAGS.output_dir, 'tensorboard')
        self.bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        self.vocab = load_vocab(FLAGS.vocab_file)
        self.label_dicts = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-COM': 3, 'I-COM': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7,
                            'I-LOC': 8}
        self.id_to_label = {val: key for key, val in self.label_dicts.items()}
        self.num_labels = len(self.label_dicts)
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
        self.label_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='label_id')
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
        self.lengths = tf.reduce_sum(self.input_mask, axis=1)
        self.batch_size = tf.shape(self.input_ids)[0]

        # [batch_size, max_seq_length, 768]
        self.bert_embedding = self.bert_embedding()
        self.bilstm_layer = self.bilstm_layer(self.bert_embedding)
        self.dense_layer = tf.keras.layers.Dense(self.num_labels)(self.bilstm_layer)

        self.scores = tf.reshape(self.dense_layer, [-1, FLAGS.max_seq_length, self.num_labels])

        ## crf
        # 计算log-likelihood并获得transition_params
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.label_ids,
                                                                                        self.lengths)
        self.loss = tf.reduce_mean(-self.log_likelihood)
        # 进行解码（维特比算法），获得解码之后的序列viterbi_sequence和分数viterbi_score
        self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.scores, self.transition_params,
                                                                         self.lengths)

        if is_training:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       FLAGS.init_checkpoint)

            tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            train_vars = []
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                else:
                    train_vars.append(var)
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            with tf.variable_scope('optimizer'):
                # grads = tf.gradients(self.loss, train_vars)
                # (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
                #
                # self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).apply_gradients(
                #     zip(grads, train_vars), global_step=self.global_step)
                self.train_data_count = count_train_file(FLAGS.train_file)
                num_train_steps = int(self.train_data_count / FLAGS.batch_size)
                num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

                self.train_op = self.create_optimizer(self.loss, init_lr=5e-5, num_train_steps=num_train_steps,
                                                      num_warmup_steps=num_warmup_steps)

                self.correct_num = tf.cast(tf.equal(self.label_ids, self.viterbi_sequence), tf.float32)
                self.accuracy = tf.reduce_mean(self.correct_num)

    def create_optimizer(self, loss, init_lr, num_train_steps, num_warmup_steps):
        """Creates an optimizer training op."""

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
        # Implements linear decay of the learning rate.
        learning_rate = tf.train.polynomial_decay(learning_rate, global_step, num_train_steps, end_learning_rate=0.0,
                                                  power=1.0, cycle=False)
        # Implements linear warmup. I.e., if global_step < num_warmup_steps, the # learning rate will be `global_step/num_warmup_steps * init_lr`. if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done
        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
        # It is recommended that you use this optimizer for fine tuning, since this
        # is how the model was trained (note that the Adam m/v variables are NOT
        # loaded from init_checkpoint.)
        optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate, weight_decay_rate=0.01, beta_1=0.9,
                                             beta_2=0.999, epsilon=1e-6,
                                             exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        # This is how the model was pre-trained.
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        # Normally the global step update is done inside of `apply_gradients`.
        # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
        # a different optimizer, you should probably take this line out.
        new_global_step = global_step + 1
        train_op = tf.group(train_op, [global_step.assign(new_global_step)])
        return train_op

    def bert_embedding(self):
        bert_model = modeling.BertModel(config=self.bert_config, is_training=self.is_training, input_ids=self.input_ids,
                                        input_mask=self.input_mask, token_type_ids=self.segment_ids,
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

    def train_step(self, sess, records, ner_processor):
        examples = ner_processor.get_train_examples(records)

        input_ids, input_mask, segment_ids, label_ids = convert_batch_example(examples,
                                                                              max_seq_length=FLAGS.max_seq_length,
                                                                              vocab=self.vocab,
                                                                              label_dicts=self.label_dicts)
        train_op, accuracy, loss = sess.run([self.train_op, self.accuracy, self.loss],
                                            feed_dict={self.input_ids: input_ids, self.input_mask: input_mask,
                                                       self.segment_ids: segment_ids, self.label_ids: label_ids,
                                                       self.dropout: 0.5})
        return train_op, accuracy, loss

    def evaluate_step(self, sess, records, ner_processor):
        evaluate_examples = ner_processor.get_train_examples(records)

        evaluate_input_ids, evaluate_input_mask, evaluate_segment_ids, evaluate_label_ids = convert_batch_example(
            evaluate_examples, max_seq_length=FLAGS.max_seq_length, vocab=self.vocab, label_dicts=self.label_dicts)
        evaluate_accuracy, evaluate_loss, correct_num = sess.run([self.accuracy, self.loss, self.correct_num],
                                                                 feed_dict={self.input_ids: evaluate_input_ids,
                                                                            self.input_mask: evaluate_input_mask,
                                                                            self.segment_ids: evaluate_segment_ids,
                                                                            self.label_ids: evaluate_label_ids,
                                                                            self.dropout: 1.0})
        return evaluate_accuracy, evaluate_loss, correct_num

    def train(self):
        tf.summary.scalar("loss", self.loss)

        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.merge_all()
        ner_processor = NerProcessor()
        saver = tf.train.Saver()
        max_acc = 0.0
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
        with tf.io.gfile.GFile(FLAGS.train_file, 'r') as f:
            train_records, train_nums_batch = [], 0
        for line in f:
            if len(train_records) == FLAGS.batch_size:
                train_op, accuracy, loss = self.train_step(sess, train_records, ner_processor)
                train_nums_batch += 1
                if train_nums_batch % 10 == 0:
                    with tf.io.gfile.GFile(FLAGS.evaluate_file, 'r') as evaluate_f:
                        evaluate_nums_batch = 0
                        evaluate_accuracy, evaluate_loss = 0.0, 0.0
                        evaluate_records = []
                        for evaluate_line in evaluate_f:
                            if len(evaluate_records) == FLAGS.batch_size:
                                _accuracy, _loss, correct_num = self.evaluate_step(
                                    sess, evaluate_records, ner_processor)
                                evaluate_nums_batch += 1
                                evaluate_accuracy += _accuracy
                                evaluate_loss += _loss
                                evaluate_records = []
                                evaluate_records.append(evaluate_line.strip())
                                if len(evaluate_records) > 0: _accuracy, _loss, correct_num = self.evaluate_step(sess,
                                                                                                                 evaluate_records,
                                                                                                                 ner_processor)
                                evaluate_nums_batch += 1
                                evaluate_accuracy += _accuracy
                                evaluate_loss += _loss
                                evaluate_accuracy, evaluate_loss = round((evaluate_accuracy / evaluate_nums_batch),
                                                                         2), round(
                                    (evaluate_loss / evaluate_nums_batch), 2)
                                if evaluate_accuracy > max_acc: saver.save(sess=sess, save_path=self.model_path)
                                tf.summary.FileWriter(self.summary_path, graph=sess.graph)
                                max_acc = evaluate_accuracy
                                tf.logging.info(
                                    '%d batch: train accuracy is %f, train loss is %f; evaluate accuracy is %f, evaluate loss is %f\n' % (
                                        train_nums_batch, round(accuracy, 6), round(loss, 6),
                                        round(evaluate_accuracy, 6),
                                        round(evaluate_loss, 6)))
                                train_records = []
                                train_records.append(line.strip())
                                if len(train_records) > 0:
                                    train_op, accuracy, loss = self.train_step(sess, train_records, ner_processor)
                                    train_nums_batch += 1
                                    with tf.io.gfile.GFile(FLAGS.evaluate_file, 'r') as evaluate_f:
                                        evaluate_nums_batch = 0
                                        evaluate_accuracy, evaluate_loss = 0.0, 0.0
                                        evaluate_records = []
                                        for evaluate_line in evaluate_f:
                                            if len(evaluate_records) == FLAGS.batch_size:
                                                _accuracy, _loss, correct_num = self.evaluate_step(sess,
                                                                                                   evaluate_records,
                                                                                                   ner_processor)
                                                evaluate_nums_batch += 1
                                                evaluate_accuracy += _accuracy
                                                evaluate_loss += _loss
                                                evaluate_records = []
                                            evaluate_records.append(evaluate_line.strip())
                                            if len(evaluate_records) > 0:
                                                _accuracy, _loss, correct_num = self.evaluate_step(sess,
                                                                                                   evaluate_records,
                                                                                                   ner_processor)
                                                evaluate_nums_batch += 1
                                                evaluate_accuracy += _accuracy
                                                evaluate_loss += _loss
                                            evaluate_accuracy, evaluate_loss = round(
                                                (evaluate_accuracy / evaluate_nums_batch), 2), round(
                                                (evaluate_loss / evaluate_nums_batch), 2)
                                            if evaluate_accuracy >= max_acc:
                                                max_acc = evaluate_accuracy
                                                saver.save(sess=sess, save_path=self.model_path)
                                                tf.summary.FileWriter(self.summary_path, graph=sess.graph)
                                            tf.logging.info(
                                                '%d batch: train accuracy is %f, train loss is %f; evaluate accuracy is %f, evaluate loss is %f\n' % (
                                                    train_nums_batch, round(accuracy, 6), round(loss, 6),
                                                    round(max_acc, 6),
                                                    round(evaluate_loss, 6)))
                                saver.save(sess=sess,
                                           save_path=os.path.join(FLAGS.output_dir, 'model/bert_ner_model_final'))
                                tf.summary.FileWriter(self.summary_path, graph=sess.graph)

    def predict(self, records):
        ner_processor = NerProcessor()

        examples = ner_processor.get_test_examples(records)
        texts = []
        for example in examples: texts.append(' ' + example.text)
        input_ids, input_mask, segment_ids, label_ids = convert_batch_example(examples,
                                                                              max_seq_length=FLAGS.max_seq_length,
                                                                              vocab=self.vocab,
                                                                              label_dicts=self.label_dicts)
        tags = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess,
                      save_path=self.model_path)
        # self.export_model(sess, '../../export_model/21')
        viterbi_sequence = sess.run(self.viterbi_sequence,
                                    feed_dict={self.input_ids: input_ids, self.input_mask: input_mask,
                                               self.segment_ids: segment_ids, self.label_ids: label_ids,
                                               self.dropout: 1.0})
        for row in viterbi_sequence:
            tags.append([self.id_to_label[ele] for ele in row])
            entities = []
            entity_list = []
            for text, tag in zip(texts, tags):
                entities.append(self.parse_entities(text, tag))
            for entity in entities:
                entity_tags = set()
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
                while 'I-' in tag[i]:
                    i += 1
                words = []
                for ele in text[start:i]:
                    words.append(ele)
                entities.append('%s,%s' % (''.join(words), val))
            else:
                i += 1
        return entities

    def predict2(self, records, export_path):
        ner_processor = NerProcessor()

        examples = ner_processor.get_test_examples(records)
        texts = []
        for example in examples:
            texts.append(' ' + example.text)
        _input_ids, _input_mask, _segment_ids, _label_ids = convert_batch_example(examples,
                                                                                  max_seq_length=FLAGS.max_seq_length,
                                                                                  vocab=self.vocab,
                                                                                  label_dicts=self.label_dicts)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        tags = []
        with tf.Session(graph=tf.Graph(), config=sess_config) as sess:
            meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                                  export_path)
            signature = meta_graph_def.signature_def
            input_ids = signature['predict_label'].inputs['input_ids'].name
            input_mask = signature['predict_label'].inputs["input_mask"].name
            segment_ids = signature['predict_label'].inputs[
                'segment_ids'].name
            # label_ids = signature['predict_label'].inputs['label_ids'].name
            # dropout = signature['predict_label'].inputs['dropout'].name
            input_ids = sess.graph.get_tensor_by_name(input_ids)
            input_mask = sess.graph.get_tensor_by_name(input_mask)
            segment_ids = sess.graph.get_tensor_by_name(segment_ids)
            # label_ids = sess.graph.get_tensor_by_name(label_ids)
            # dropout = sess.graph.get_tensor_by_name(dropout)
            viterbi_sequence = signature['predict_label'].outputs['viterbi_sequence'].name
            viterbi_sequence = sess.graph.get_tensor_by_name(viterbi_sequence)
            predicts = sess.run(viterbi_sequence, feed_dict={input_ids: _input_ids, input_mask: _input_mask,
                                                             segment_ids: _segment_ids,
                                                             # label_ids: _label_ids, # dropout: 1.0,
                                                             })
            for row in predicts:
                tags.append([self.id_to_label[ele] for ele in row])
        entities = []
        for text, tag in zip(texts, tags):
            entities.append(self.parse_entities(text, tag))
        return entities

    def export_model(self, sess, export_path):
        input_ids = tf.saved_model.utils.build_tensor_info(self.input_ids)

        input_mask = tf.saved_model.utils.build_tensor_info(self.input_mask)
        segment_ids = tf.saved_model.utils.build_tensor_info(self.segment_ids)
        label_ids = tf.saved_model.utils.build_tensor_info(self.label_ids)
        dropout = tf.saved_model.utils.build_tensor_info(self.dropout)
        viterbi_sequence = tf.saved_model.utils.build_tensor_info(self.viterbi_sequence)
        predict_signature = (tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input_ids': input_ids, "input_mask": input_mask, "segment_ids": segment_ids,
                    "label_ids": label_ids,
                    'dropout': dropout}, outputs={'viterbi_sequence': viterbi_sequence},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={'predict_label': predict_signature},
                                             legacy_init_op=legacy_init_op)
        builder.save()


if __name__ == '__main__':
    # bert_crf = BertCRF()
    # bert_crf.train()
    pass
