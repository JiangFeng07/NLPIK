#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2022-03-16 11:45 
# @Author : Leo

# Relation Classification via Convolutional Deep Neural Network 论文实现
import os
import tensorflow as tf


class CDNNModel(object):
    def __init__(self, word_vec, num_relations=19, num_filters=100, max_len=128, pos_num=123, window_size=3,
                 word_dim=50, distance_dim=5, lrn_rate=0.01, is_train=False, batch_size=128):
        self.num_filters = num_filters
        self.max_len = max_len
        self.batch_size = batch_size

        self.input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos2')

        # Lexical Level Features
        self.lexical = tf.placeholder(dtype=tf.int32, shape=[None, 2 * window_size], name='lexical')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, ], name='label')
        self.labels2 = tf.one_hot(self.labels, num_relations)
        word_embed = tf.get_variable(name='word_embed', initializer=word_vec, dtype=tf.float32)

        lexical_feature = tf.nn.embedding_lookup(word_embed, self.lexical)
        lexical_feature = tf.reshape(lexical_feature, [-1, 6 * word_dim])

        sentence = tf.nn.embedding_lookup(word_embed, self.input)

        pos1_embed = tf.get_variable(name='pos1_embed', shape=[pos_num, distance_dim])
        pos1 = tf.nn.embedding_lookup(pos1_embed, self.pos1)

        pos2_embed = tf.get_variable(name='pos2_embed', shape=[pos_num, distance_dim])
        pos2 = tf.nn.embedding_lookup(pos2_embed, self.pos2)

        # cnn model
        sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
        if is_train:
            sent_pos = tf.nn.dropout(sent_pos, keep_prob=0.5)
        sentence_feature = self.cnn_layer(sent_pos)

        feature = tf.concat([lexical_feature, sentence_feature], axis=1)

        if is_train:
            feature = tf.nn.dropout(feature, keep_prob=0.5)

        feature_size = feature.shape.as_list()[1]
        logits, loss_l2 = self.linear_layer(feature, feature_size, num_relations, is_regularize=True)
        prediction = tf.nn.softmax(logits)
        prediction = tf.argmax(prediction, axis=1)
        accuracy = tf.equal(prediction, tf.argmax(self.labels2, axis=1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        loss_ce = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels2, logits=logits))

        self.logits = logits
        self.prediction = prediction
        self.accuracy = accuracy
        self.loss = loss_ce + 0.01 * loss_l2

        if not is_train:
            return

        # global_step = tf.train.get_or_create_global_step()
        global_step = tf.Variable(0, trainable=False, name='step', dtype=tf.int32)
        optimizer = tf.train.AdamOptimizer(lrn_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # for batch_norm
            self.train_op = optimizer.minimize(self.loss, global_step)
        self.global_step = global_step

    def cnn_layer(self, sent_pos):
        with tf.variable_scope('cnn'):
            input_data = tf.expand_dims(sent_pos, axis=-1)
            input_dim = input_data.shape.as_list()[2]

            # convolution layer
            pool_outputs = []
            for filter_size in [3, 4, 5]:
                with tf.variable_scope('conv-%s' % filter_size):
                    conv_weight = tf.get_variable('W1',
                                                  [filter_size, input_dim, 1, self.num_filters],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
                    conv_bias = tf.get_variable('b1', [self.num_filters],
                                                initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(input_data,
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

    def linear_layer(self, x, in_size, out_size, is_regularize=False):
        with tf.variable_scope('linear'):
            loss_l2 = tf.constant(0, dtype=tf.float32)
            w = tf.get_variable('linear_W', [in_size, out_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('linear_b', [out_size],
                                initializer=tf.constant_initializer(0.1))
            o = tf.nn.xw_plus_b(x, w, b)  # batch_size, out_size
            if is_regularize:
                loss_l2 += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
            return o, loss_l2


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
