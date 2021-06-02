# -*- coding: utf-8 -*-
import os 
import json
import pandas as pd
import numpy as np
import pickle
import time

startTime =time.clock()
DIR = 'raw_data'
FILE = 'Musical_Instruments_5.json'
f = open(os.path.join(DIR, FILE))

users_id = []
items_id = []
ratings = []
reviews = []

count = 0
for line in f:
    js = json.loads(line)
    
    if str(js['reviewerID']) == 'unknown':
        print('unknown')
        continue
    
    if str(js['asin']) == 'unknown':
        print('unknown')
        continue
    
    reviews.append(js['reviewText'])
    users_id.append(str(js['reviewerID']) + ',')
    items_id.append(str(js['asin']) + ',')
    ratings.append(str(js['overall']))
    
    count = count + 1
    if count == 300000:
        break

print('-------------------------------------------------------------------------')
print('json load finish')

data = pd.DataFrame({'user_id': pd.Series(users_id),
                     'item_id': pd.Series(items_id),
                     'ratings': pd.Series(ratings),
                     'reviews': pd.Series(reviews)})[['user_id', 'item_id', 'ratings', 'reviews']]

def get_count(tp, key):
    count_groupbykey = tp[[key, 'ratings']].groupby(key, as_index = False)
    count = count_groupbykey.size()
    return count

MIN_USER_COUNT = 5
MIN_ITEM_COUNT = 5

def filter_triplets(tp, min_uc=MIN_USER_COUNT, min_ic=MIN_ITEM_COUNT):
    '''Only keep the triplets for items which were rated by at least min_ic users.'''
    itemcount = get_count(tp, 'item_id')
    tp = tp[tp['item_id'].isin(itemcount.index[itemcount >= min_ic])]
    
    '''Only keep the triplets for users who listened to at least min_uc songs
    After doing this, some of the songs will have less than min_uc users, 
    but should only be a small proportion'''
    usercount = get_count(tp, 'user_id')
    tp = tp[tp['user_id'].isin(usercount.index[usercount >= min_uc])]
    
    itemcount = get_count(tp, 'item_id')
    usercount = get_count(tp, 'user_id')
    
    return tp, itemcount, usercount

data, itemcount, usercount = filter_triplets(data)

unique_uid = usercount.index
unique_iid = itemcount.index

user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
item2id = dict((iid, i) for (i, iid) in enumerate(unique_iid))

def numerize(tp):
    uid = map(lambda x: user2id[x], tp['user_id'])
    iid = map(lambda x: item2id[x], tp['item_id'])
    tp['user_id'] = list(uid)
    tp['item_id'] = list(iid)
    return tp

data = numerize(data)
print('numerize finish')

tp_rating = data[['user_id', 'item_id', 'ratings']]

n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size = int(0.20*n_ratings), replace = False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]
tp_train = tp_rating[~test_idx]

n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size=int(0.0*n_ratings), replace = False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]

train_userid = pd.DataFrame(tp_train['user_id'].drop_duplicates())
a = pd.merge(tp_valid, train_userid, on=['user_id'], how='inner')
a = a.iloc[:, 0: 3].drop_duplicates()
a.columns = ['user_id', 'item_id', 'ratings']
train_itemid = pd.DataFrame(tp_train['item_id'].drop_duplicates())
b = pd.merge(a, train_itemid, on=['item_id'], how='inner')
b = b.iloc[:, 0:3].drop_duplicates()
b.columns = ['user_id', 'item_id', 'ratings']
tp_valid = tp_valid.append(b)
tp_valid = tp_valid.append(b)
c = tp_valid.drop_duplicates(keep=False)
tp_train = pd.merge(tp_train, c, how='outer')
tp_valid = b

train_userid = pd.DataFrame(tp_train['user_id'].drop_duplicates())
a = pd.merge(tp_test, train_userid, on=['user_id'], how='inner')
a = a.iloc[:, 0: 3].drop_duplicates()
a.columns = ['user_id', 'item_id', 'ratings']
train_itemid = pd.DataFrame(tp_train['item_id'].drop_duplicates())
b = pd.merge(a, train_itemid, on=['item_id'], how='inner')
b = b.iloc[:, 0:3].drop_duplicates()
b.columns = ['user_id', 'item_id', 'ratings']
tp_test = tp_test.append(b)
tp_test = tp_test.append(b)
c = tp_test.drop_duplicates(keep=False)
tp_train = pd.merge(tp_train, c, how='outer')
tp_test = b

tp_train.to_csv('pro_data/music/music_train.csv', index=False, header=None)
tp_valid.to_csv('pro_data/music/music_valid.csv', index=False, header=None)
tp_test.to_csv('pro_data/music/music_test.csv', index=False, header=None)
data = pd.merge(data, tp_train, on=['user_id', 'item_id', 'ratings'], how='inner')

user_reviews = {}
item_reviews = {}
user_rid = {}
item_rid = {}

print(data)
for i in data.values:
    if user_reviews.__contains__(i[0]):
        user_reviews[i[0]].append(i[3])
        user_rid[i[0]].append(i[1])
    else:
        user_rid[i[0]] = [i[1]]
        user_reviews[i[0]] = [i[3]]
        
    if item_reviews.__contains__(i[1]):
        item_reviews[i[1]].append(i[3])
        item_rid[i[1]].append(i[0])
    else:
        item_reviews[i[1]] = [i[3]]
        item_rid[i[1]] = [i[0]]

pickle.dump(user_reviews, open('pro_data/music/user_reviews', 'wb'))
pickle.dump(item_reviews, open('pro_data/music/item_reviews', 'wb'), -1)
pickle.dump(user_rid, open('pro_data/music/user_rid', 'wb'), -1)
pickle.dump(item_rid, open('pro_data/music/item_rid', 'wb'), -1)


user_pos_reviews = {}
user_pos_rid = {}
item_pos_reviews = {}
item_pos_rid = {}
user_neg_reviews = {}
user_neg_rid = {}
item_neg_reviews = {}
item_neg_rid = {}

for i in data.values:
    if float(i[2]) >= 4.0:
        if user_pos_reviews.__contains__(i[0]):
            user_pos_reviews[i[0]].append(i[3])
            user_pos_rid[i[0]].append(i[1])
        else:
            user_pos_reviews[i[0]] = [i[3]]
            user_pos_rid[i[0]] = [i[1]]

        if item_pos_reviews.__contains__(i[1]):
            item_pos_reviews[i[1]].append(i[3])
            item_pos_rid[i[1]].append(i[0])
        else:
            item_pos_reviews[i[1]] = [i[3]]
            item_pos_rid[i[1]] = [i[0]]

    else:
        if user_neg_reviews.__contains__(i[0]):
            user_neg_reviews[i[0]].append(i[3])
            user_neg_rid[i[0]].append(i[1])
        else:
            user_neg_reviews[i[0]] = [i[3]]
            user_neg_rid[i[0]] = [i[1]]

        if item_neg_reviews.__contains__(i[1]):
            item_neg_reviews[i[1]].append(i[3])
            item_neg_rid[i[1]].append(i[0])
        else:
            item_neg_reviews[i[1]] = [i[3]]
            item_neg_rid[i[1]] = [i[0]]

pickle.dump(user_pos_reviews, open('pro_data/music/user_pos_reviews', 'wb'), -1)
pickle.dump(user_neg_reviews, open('pro_data/music/user_neg_reviews', 'wb'), -1)
pickle.dump(item_pos_reviews, open('pro_data/music/item_pos_reviews', 'wb'), -1)
pickle.dump(item_neg_reviews, open('pro_data/music/item_neg_reviews', 'wb'), -1)
pickle.dump(user_pos_rid, open('pro_data/music/user_pos_rid', 'wb'), -1)
pickle.dump(user_neg_rid, open('pro_data/music/user_neg_rid', 'wb'), -1)
pickle.dump(item_pos_rid, open('pro_data/music/item_pos_rid', 'wb'), -1)
pickle.dump(item_neg_rid, open('pro_data/music/item_neg_rid', 'wb'), -1)
print('========================================================')
print(tp_train.shape[0])
print(tp_valid.shape[0])
print(tp_test.shape[0])

endTime = time.clock()
print('time is %s s' % (endTime - startTime))




















