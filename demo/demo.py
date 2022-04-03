#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2022-04-02 22:37 
# @Author : Leo
import pickle

if __name__ == '__main__':
    path = '/Users/jiangfeng/workspace_me/data/SQuAD/train-v2.0.json'
    # with open(path, 'r', encoding='utf-8') as f:
    import json

    data = json.load(open(path, 'r', encoding='utf-8'))
    print(data.keys())
    for ele in data['data']:
        print(ele.keys())
        for ele1 in ele['paragraphs']:
            print(ele1.keys())
            break
        break
