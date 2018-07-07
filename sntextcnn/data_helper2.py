import jieba
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

def load_data_and_labels(file):
    label_index = {}
    labels = []
    data = []
    max_sequence_length = 0
    with open(file,'r') as f:
        lines_list=[]
        #devide lines into items
        for line in f.readlines():
            line_list=line.split("\t")[0].split(" ")
            label=line.split("\t")[-1]
            if label not in label_index.keys():
                label_id = len(label_index)
                label_index[label] = label_id
            else:
                label_id=label_index[label]
            tmp=""
            for item in line_list:
                tmp+=" ".join(jieba.cut(item))
            if len(tmp.split()) > max_sequence_length:
                max_sequence_length = len(tmp.split())
            data.append(tmp)
            labels.append(label_id)
    # print(data[0])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    data = pad_sequences(sequences, max_sequence_length, padding='post')
    return np.array(data), to_categorical(labels), len(tokenizer.word_index)

if  __name__ == '__main__':
    file='/erp/CLOUD_DISK/notebook/Me/taxCode.txt'
    load_data_and_labels(file)