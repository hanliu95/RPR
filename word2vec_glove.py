import numpy as np
import pickle

glove_dict = {}
embedding_dim = 50
fname = 'glove_embeddings/glove.6B.%sd.txt' % embedding_dim
with open(fname, 'r') as f:
    lines = f.readlines()
    for line in lines:
        tokens = line.rstrip().split(' ')
        glove_dict[tokens[0]] = np.array(tokens[1:])

pkl_file = open("pro_data/music/music.para", 'rb')

para = pickle.load(pkl_file)
vocabulary_user_pos = para['user_pos_vocab']
vocabulary_user_neg = para['user_neg_vocab']
vocabulary_item_pos = para['item_pos_vocab']
vocabulary_item_neg = para['item_neg_vocab']
vocab = list(glove_dict.keys())

u_pos = 0
Wu_pos = np.random.uniform(-1.0, 1.0, (len(vocabulary_user_pos), embedding_dim))

for word in vocab:
    if word in vocabulary_user_pos:
        u_pos = u_pos + 1
        idx = vocabulary_user_pos[word]
        Wu_pos[idx] = glove_dict[word]

np.save('pro_data/music/pre_Wu_pos.npy', Wu_pos)

u_neg = 0
Wu_neg = np.random.uniform(-1.0, 1.0, (len(vocabulary_user_neg), embedding_dim))

for word in vocab:
    if word in vocabulary_user_neg:
        u_neg = u_neg + 1
        idx = vocabulary_user_neg[word]
        Wu_neg[idx] = glove_dict[word]

np.save('pro_data/music/pre_Wu_neg.npy', Wu_neg)

i_pos = 0
Wi_pos = np.random.uniform(-1.0, 1.0, (len(vocabulary_item_pos), embedding_dim))

for word in vocab:
    if word in vocabulary_item_pos:
        i_pos = i_pos + 1
        idx = vocabulary_item_pos[word]
        Wi_pos[idx] = glove_dict[word]

np.save('pro_data/music/pre_Wi_pos.npy', Wi_pos)

i_neg = 0
Wi_neg = np.random.uniform(-1.0, 1.0, (len(vocabulary_item_neg), embedding_dim))

for word in vocab:
    if word in vocabulary_item_neg:
        i_neg = i_neg + 1
        idx = vocabulary_item_neg[word]
        Wi_neg[idx] = glove_dict[word]

np.save('pro_data/music/pre_Wi_neg.npy', Wi_neg)
