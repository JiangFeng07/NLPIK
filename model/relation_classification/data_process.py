#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2022-03-17 10:14 
# @Author : Leo
import os
import pandas as pd

base_path = '../../data/SemEval'


def label_process():
    data = pd.read_csv(os.path.join(base_path, 'train/train_result_full.txt'), sep='\t', header=None)
    with open(os.path.join(base_path, 'labels.csv'), 'w', encoding='utf-8') as f:
        for line in set(data[1].values):
            f.write(line + '\n')
