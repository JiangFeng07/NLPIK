#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2022-03-17 17:09
# @Author : Leo
import os
import tensorflow as tf

from model.relation_classification.cdnn import CDNNModel
from model.relation_classification.utils import process_batch_data, WordEmbeddingLoader, map_label_to_id

base_path = '../../data'
train_file = os.path.join(base_path, 'SemEval/train_data.csv')
test_file = os.path.join(base_path, 'SemEval/test_data.csv')
model_path = os.path.join(base_path, 'relation_model/rc_model')
summary_path = os.path.join(base_path, 'relation_model/summary')

wordEmbed = WordEmbeddingLoader(os.path.join(base_path, 'SemEval/vector_50.txt'), word_dim=50)
word2id, word_vec = wordEmbed.load_embedding()
label2id = map_label_to_id(os.path.join(base_path, 'SemEval/labels.csv'))


def build_train_valid_model():
    """Relation Classification via Convolutional Deep Neural Network"""
    with tf.name_scope("Train"):
        with tf.variable_scope('CNNModel', reuse=None):
            m_train = CDNNModel(word_vec=word_vec, is_train=True)
    with tf.name_scope('Valid'):
        with tf.variable_scope('CNNModel', reuse=True):
            m_valid = CDNNModel(word_vec=word_vec, is_train=False)
    return m_train, m_valid


def evaluate(sess, texts, saver, m_valid, batch_size=100):
    checkpoint = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, checkpoint.model_checkpoint_path)

    def generate_batch():
        text_list = []
        while len(text_list) < batch_size:
            try:
                data_next = next(texts)
                text_list.append(data_next)
            except StopIteration:
                break
        return text_list

    text_list = generate_batch()
    batch_num = 0
    evaluate_accuracy, evaluate_loss = 0.0, 0.0
    while len(text_list) > 0:
        batch_num += 1
        evaluate_texts = [text.decode('utf-8') for text in text_list]
        texts, pos1, pos2, labels, contexts = process_batch_data(evaluate_texts, label2id,
                                                                 word2id)
        _accuracy, _loss = sess.run([m_valid.accuracy, m_valid.loss],
                                    feed_dict={m_valid.input: texts, m_valid.pos1: pos1,
                                               m_valid.pos2: pos2, m_valid.lexical: contexts,
                                               m_valid.labels: labels})
        evaluate_accuracy += _accuracy
        evaluate_loss += _loss
        text_list = generate_batch()
    evaluate_accuracy, evaluate_loss = round((evaluate_accuracy / batch_num), 2), round(
        (evaluate_loss / batch_num), 2)
    return evaluate_accuracy, evaluate_loss


def train():
    def gen_train_data():
        with tf.io.gfile.GFile(train_file, 'r') as f:
            for line in f:
                yield line.strip()

    evaluate_texts = []
    with tf.io.gfile.GFile(test_file, 'r') as f:
        for line in f:
            evaluate_texts.append(line.strip())

    eveluate_batch_num = 5
    # model = CDNNModel(word_vec=word_vec, is_train=True)

    m_train, m_valid = build_train_valid_model()

    tf.summary.scalar("loss", m_train.loss)
    tf.summary.scalar("accuracy", m_train.accuracy)
    tf.summary.merge_all()
    saver = tf.train.Saver()
    i = 1
    max_acc = 0.0
    train_dataset = tf.data.Dataset.from_generator(gen_train_data, (tf.string)).shuffle(m_train.batch_size).repeat(100)
    train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset.batch(m_train.batch_size))
    train_next_element = train_iterator.get_next()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        while True:
            try:
                texts = sess.run(train_next_element)
                texts = [text.decode('utf-8') for text in texts]
                texts, pos1, pos2, labels, contexts = process_batch_data(texts, label2id, word2id)
                train_op, accuracy, loss = sess.run([m_train.train_op, m_train.accuracy, m_train.loss],
                                                    feed_dict={m_train.input: texts, m_train.pos1: pos1,
                                                               m_train.pos2: pos2, m_train.lexical: contexts,
                                                               m_train.labels: labels})

                if i % 10 == 0:
                    evaluate_accuracy, evaluate_loss = 0.0, 0.0
                    text_list = []
                    for evaluate_text in evaluate_texts[:500]:
                        if len(text_list) == 100:
                            texts, pos1, pos2, labels, contexts = process_batch_data(evaluate_texts, label2id,
                                                                                     word2id)
                            _accuracy, _loss = sess.run([m_valid.accuracy, m_valid.loss],
                                                        feed_dict={m_valid.input: texts, m_valid.pos1: pos1,
                                                                   m_valid.pos2: pos2, m_valid.lexical: contexts,
                                                                   m_valid.labels: labels})
                            evaluate_accuracy += _accuracy
                            evaluate_loss += _loss
                            text_list = []
                            continue
                        text_list.append(evaluate_text)
                    if len(text_list) > 0:
                        _accuracy, _loss = sess.run([m_valid.accuracy, m_valid.loss],
                                                    feed_dict={m_valid.input: texts, m_valid.pos1: pos1,
                                                               m_valid.pos2: pos2, m_valid.lexical: contexts,
                                                               m_valid.labels: labels})
                        evaluate_accuracy += _accuracy
                        evaluate_loss += _loss

                    evaluate_accuracy, evaluate_loss = round((evaluate_accuracy / eveluate_batch_num), 2), round(
                        (evaluate_loss / eveluate_batch_num), 2)
                    if evaluate_accuracy > max_acc:
                        saver.save(sess=sess, save_path=model_path)
                        tf.summary.FileWriter(summary_path, graph=sess.graph)
                        max_acc = evaluate_accuracy
                    print(
                        '%d batch: train accuracy is %f, train loss is %f; evaluate accuracy is %f, evaluate loss is %f' % (
                            i, accuracy, loss, evaluate_accuracy, evaluate_loss))
                i += 1
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    train()
