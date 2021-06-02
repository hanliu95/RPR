# -*- coding: utf-8 -*-
import tensorflow as tf
import pickle
import re
import numpy as np
from collections import Counter
import itertools
import os

def clean_str(string):
    '''
    string cleaning
    '''
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def load_data_and_labels(train_data, valid_data, user_pos_review, user_neg_review, item_pos_review, item_neg_review):
    '''
    Load polarity data from files, split the data into words and generate label.
    Return split sentences and labels
    '''
    f_train = open(train_data, 'r')
    f1 = open(user_pos_review, 'rb')
    f2 = open(user_neg_review, 'rb')
    f3 = open(item_pos_review, 'rb')
    f4 = open(item_neg_review, 'rb')
    
    user_pos_reviews = pickle.load(f1)
    user_neg_reviews = pickle.load(f2)
    item_pos_reviews = pickle.load(f3)
    item_neg_reviews = pickle.load(f4)

    uid_train = []
    iid_train = []
    y_train = []
    u_pos_text = {}
    u_neg_text = {}
    i_pos_text = {}
    i_neg_text = {}
    
    i = 0
    for line in f_train:
        i = i + 1
        line = line.split(',')
        uid = int(line[0])
        iid = int(line[1])
        uid_train.append(uid)
        iid_train.append(iid)

        if u_pos_text.__contains__(uid):
            a = 1
        else:
            u_pos_text[uid] = '<PAD/>'
            if uid in user_pos_reviews:
                for s in user_pos_reviews[uid]:
                    u_pos_text[uid] = u_pos_text[uid] + ' ' + s.strip()
            u_pos_text[uid] = clean_str(u_pos_text[uid])
            u_pos_text[uid] = u_pos_text[uid].split(" ")


        if u_neg_text.__contains__(uid):
            a = 1
        else:
            u_neg_text[uid] = '<PAD/>'
            if uid in user_neg_reviews:
                for s in user_neg_reviews[uid]:
                    u_neg_text[uid] = u_neg_text[uid] + ' ' + s.strip()
            u_neg_text[uid] = clean_str(u_neg_text[uid])
            u_neg_text[uid] = u_neg_text[uid].split(" ")

        if i_pos_text.__contains__(iid):
            a = 1
        else:
            i_pos_text[iid] = '<PAD/>'
            if iid in item_pos_reviews:
                for s in item_pos_reviews[iid]:
                    i_pos_text[iid] = i_pos_text[iid] + ' ' + s.strip()
            i_pos_text[iid] = clean_str(i_pos_text[iid])
            i_pos_text[iid] = i_pos_text[iid].split(" ")

        if i_neg_text.__contains__(iid):
            a = 1
        else:
            i_neg_text[iid] = '<PAD/>'
            if iid in item_neg_reviews:
                for s in item_neg_reviews[iid]:
                    i_neg_text[iid] = i_neg_text[iid] + ' ' + s.strip()
            i_neg_text[iid] = clean_str(i_neg_text[iid])
            i_neg_text[iid] = i_neg_text[iid].split(" ")
            
        y_train.append(float(line[2]))
    print('***')
    print(i_neg_text)
        
    # valid
    uid_valid = []
    iid_valid = []
    y_valid = []
    f_valid = open(valid_data)
    
    for line in f_valid:
        line = line.split(',')
        uid = int(line[0])
        iid = int(line[1])
        uid_valid.append(uid)
        iid_valid.append(iid)
        y_valid.append(float(line[2]))
        
    # length
    u_pos = np.array([len(x) for x in u_pos_text.values()])
    x = np.sort(u_pos)
    u_pos_len = x[int(0.5 * len(u_pos)) - 1]

    u_neg = np.array([len(x) for x in u_neg_text.values()])
    x = np.sort(u_neg)
    u_neg_len = x[int(0.85 * len(u_neg)) - 1]

    i_pos = np.array([len(x) for x in i_pos_text.values()])
    x = np.sort(i_pos)
    i_pos_len = x[int(0.5 * len(i_pos)) - 1]

    i_neg = np.array([len(x) for x in i_neg_text.values()])
    x = np.sort(i_neg)
    i_neg_len = x[int(0.85 * len(i_neg)) - 1]

    user_num = len(u_pos_text)
    item_num = len(i_pos_text)
    print("user_num", user_num)
    print("item_num", item_num)
    return [u_pos_text, u_neg_text, i_pos_text, i_neg_text, y_train, y_valid, u_pos_len, u_neg_len,\
            i_pos_len, i_neg_len, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num]



def pad_sentences(u_text, u_len, padding_word = '<PAD/>'):
    '''
    Pad all sentences to the same length. The length is defined by the longest sentence.
    Return padded sentences.
    '''
    sequence_length = u_len
    u_text2 = {}
    print(len(u_text))
    for i in u_text.keys():
        sentence = u_text[i]
        if sequence_length > len(sentence):
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            u_text2[i] = new_sentence
        else:
            new_sentence = sentence[:sequence_length]
            u_text2[i] = new_sentence
    return u_text2

    
    
def build_vocab(sentences1, sentences2):
    '''
    Build a vocabulary mapping from word to index and based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    '''
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}
    
    word_counts2 = Counter(itertools.chain(*sentences2))
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]



def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    '''
    Map sentences and labels to vectors based on a vocabulary.
    '''
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([vocabulary_u[word] for word in u_reviews])
        u_text2[i] = u
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([vocabulary_i[word] for word in i_reviews])
        i_text2[j] = i
        
    return u_text2, i_text2
    


def load_data(train_data, valid_data, user_pos_review, user_neg_review, item_pos_review, item_neg_review):
    '''
    Load and preprocess data for the dataset.
    Return input vectors, labels, vocabulary, and inverse vocabulary.
    '''
    print('load data start')
    u_pos_text, u_neg_text, i_pos_text, i_neg_text, y_train, y_valid, u_pos_len, u_neg_len, \
    i_pos_len, i_neg_len, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num=\
        load_data_and_labels(train_data, valid_data, user_pos_review, user_neg_review, item_pos_review, item_neg_review)
    print('load data done')

    u_pos_text = pad_sentences(u_pos_text, u_pos_len)
    u_neg_text = pad_sentences(u_neg_text, u_neg_len)
    i_pos_text = pad_sentences(i_pos_text, i_pos_len)
    i_neg_text = pad_sentences(i_neg_text, i_neg_len)

    user_pos_voc = [x for x in u_pos_text.values()]
    user_neg_voc = [x for x in u_neg_text.values()]
    item_pos_voc = [x for x in i_pos_text.values()]
    item_neg_voc = [x for x in i_neg_text.values()]

    vocabulary_user_pos, vocabulary_inv_user_pos, vocabulary_item_pos, vocabulary_inv_item_pos = \
        build_vocab(user_pos_voc, item_pos_voc)
    vocabulary_user_neg, vocabulary_inv_user_neg, vocabulary_item_neg, vocabulary_inv_item_neg = \
        build_vocab(user_neg_voc, item_neg_voc)

    u_pos_text, i_pos_text = build_input_data(u_pos_text, i_pos_text, vocabulary_user_pos, vocabulary_item_pos)
    u_neg_text, i_neg_text = build_input_data(u_neg_text, i_neg_text, vocabulary_user_neg, vocabulary_item_neg)

    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)

    return [u_pos_text, u_neg_text, i_pos_text, i_neg_text, y_train, y_valid,
            vocabulary_user_pos, vocabulary_inv_user_pos, vocabulary_item_pos, vocabulary_inv_item_pos,
            vocabulary_user_neg, vocabulary_inv_user_neg, vocabulary_item_neg, vocabulary_inv_item_neg,
            uid_train, iid_train, uid_valid, iid_valid, user_num, item_num]



if __name__ == '__main__':
    DIR = 'pro_data/music'
    print('main start')
    train_data = "pro_data/music/yelp_train.csv"
    valid_data = "pro_data/music/yelp_valid.csv"
    test_data = "pro_data/music/yelp_test.csv"
    user_pos_review = "pro_data/music/user_pos_reviews"
    user_neg_review = "pro_data/music/user_neg_reviews"
    item_pos_review = "pro_data/music/item_pos_reviews"
    item_neg_review = "pro_data/music/item_neg_reviews"


    u_pos_text, u_neg_text, i_pos_text, i_neg_text, y_train, y_valid,\
    vocabulary_user_pos, vocabulary_inv_user_pos, vocabulary_item_pos, vocabulary_inv_item_pos,\
    vocabulary_user_neg, vocabulary_inv_user_neg, vocabulary_item_neg, vocabulary_inv_item_neg,\
    uid_train, iid_train, uid_valid, iid_valid, user_num, item_num = \
        load_data(train_data, valid_data, user_pos_review, user_neg_review, item_pos_review, item_neg_review)

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    
    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    
    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]
    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]
    
    batches_train=list(zip(userid_train, itemid_train, y_train))
    batches_test=list(zip(userid_valid, itemid_valid, y_valid))
    output = open(os.path.join(DIR, 'music.train'), 'wb')
    pickle.dump(batches_train,output)
    output = open(os.path.join(DIR, 'music.test'), 'wb')
    pickle.dump(batches_test,output)
    
    para={}
    para['user_num'] = user_num
    para['item_num'] = item_num
    para['train_length'] = len(y_train)
    para['test_length'] = len(y_valid)
    para['user_pos_length'] = u_pos_text[0].shape[0]
    para['user_neg_length'] = u_neg_text[0].shape[0]
    para['item_pos_length'] = i_pos_text[0].shape[0]
    para['item_neg_length'] = i_neg_text[0].shape[0]
    para['user_pos_vocab'] = vocabulary_user_pos
    para['user_neg_vocab'] = vocabulary_user_neg
    para['item_pos_vocab'] = vocabulary_item_pos
    para['item_neg_vocab'] = vocabulary_item_neg
    para['u_pos_text'] = u_pos_text
    para['u_neg_text'] = u_neg_text
    para['i_pos_text'] = i_pos_text
    para['i_neg_text'] = i_neg_text

    output = open(os.path.join(DIR, 'music.para'), 'wb')
    pickle.dump(para, output)
    print("data_pro.py END")