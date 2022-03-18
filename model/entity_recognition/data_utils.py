#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2022-03-14 13:29 
# @Author : Leo
import re
import tensorflow as tf


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


def text_digit(text, vocab):
    digits = []
    if not text or len(text) == 0:
        return digits
    for word in text:
        digits.append(vocab.get(word, vocab.get('[UNK]')))
    return digits


def texts_digit(texts, vocab):
    digits_list = []
    if not texts or len(texts) == 0:
        return digits_list
    for text in texts:
        digits_list.append(text_digit(text, vocab))
    return digits_list


def process_batch_data(texts, tag_dict, vocab, max_len=350):
    def _process_data(label, content):
        text_list = []
        for word in content:
            text_list.append(vocab.get(word, vocab.get('[UNK]')))
        entity_list = label.split(';')
        entity_property = dict()
        for entity in entity_list:
            fields = entity.split(':')
            if len(fields) != 2:
                continue
            entity_property[fields[0]] = fields[1]
        entity_property = sorted(entity_property.items(), key=lambda row: len(row[0]))
        tag_list = len(text_list) * [tag_dict['O']]
        for key, val in entity_property:
            try:
                for ele in re.finditer(key, content):
                    start = ele.span()[0]
                    end = ele.span()[1]
                    tag_list[start] = tag_dict['B-' + val]
                    while start + 1 < end:
                        tag_list[start + 1] = tag_dict['I-' + val]
                        start = start + 1
            except:
                continue
        return text_list, tag_list

    if not texts or len(texts) == 0:
        return None
    text_data, tag_data = [], []
    for text in texts:
        fields = text.split('\t')
        if len(fields) != 2:
            continue
        label, content = tuple(fields)
        _contents = cut_text(content)
        for _content in _contents:
            text_list, tag_list = _process_data(label, _content)
            text_data.append(text_list)
            tag_data.append(tag_list)
    text_data = tf.keras.preprocessing.sequence.pad_sequences(text_data, value=vocab.get('[PAD]'), padding='post',
                                                              maxlen=max_len)
    tag_data = tf.keras.preprocessing.sequence.pad_sequences(tag_data, value=tag_dict.get('O'), padding='post',
                                                             maxlen=max_len)
    return text_data, tag_data
