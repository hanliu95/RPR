import pickle
import tensorflow as tf
import Model
import numpy as np

PARA_DATA = "pro_data/music/music.para"
TRAIN_DATA = "pro_data/music/music.train"
VALID_DATA = "pro_data/music/music.test"

EMBEDDING_DIM = 50
FILTER_SIZE = "5"
NUM_FILTERS = 16  # out_channels
n_pos_aspect = 32
n_neg_aspect = 32
word2vec = True
batch_size = 500
factor_dim = 32

def train_step_1(u_pos_batch, u_neg_batch, i_pos_batch, i_neg_batch, uid, iid, y_batch):
    feed_dict = {
        model.input_u_pos: u_pos_batch,
        model.input_u_neg: u_neg_batch,
        model.input_i_pos: i_pos_batch,
        model.input_i_neg: i_neg_batch,
        model.input_uid: uid,
        model.input_iid: iid,
        model.input_y: y_batch
    }
    _, loss = sess.run([train_op_1, model.loss], feed_dict)
    accuracy, mae = sess.run([model.accuracy, model.mae], feed_dict)
    return accuracy, mae

def train_step_2(u_pos_batch, u_neg_batch, i_pos_batch, i_neg_batch, uid, iid, y_batch):
    feed_dict = {
        model.input_u_pos: u_pos_batch,
        model.input_u_neg: u_neg_batch,
        model.input_i_pos: i_pos_batch,
        model.input_i_neg: i_neg_batch,
        model.input_uid: uid,
        model.input_iid: iid,
        model.input_y: y_batch
    }
    _, loss = sess.run([train_op_2, model.loss], feed_dict)
    accuracy, mae = sess.run([model.accuracy, model.mae], feed_dict)
    return accuracy, mae

def train_step_3(u_pos_batch, u_neg_batch, i_pos_batch, i_neg_batch, uid, iid, y_batch):
    feed_dict = {
        model.input_u_pos: u_pos_batch,
        model.input_u_neg: u_neg_batch,
        model.input_i_pos: i_pos_batch,
        model.input_i_neg: i_neg_batch,
        model.input_uid: uid,
        model.input_iid: iid,
        model.input_y: y_batch
    }
    _, loss = sess.run([train_op_3, model.loss], feed_dict)
    accuracy, mae = sess.run([model.accuracy, model.mae], feed_dict)
    return accuracy, mae



def dev_step(u_pos_valid, u_neg_valid, i_pos_valid, i_neg_valid, userid_valid, itemid_valid, y_valid):
    """
    Evaluates model
    """
    feed_dict = {
        model.input_u_pos: u_pos_valid,
        model.input_u_neg: u_neg_valid,
        model.input_i_pos: i_pos_valid,
        model.input_i_neg: i_neg_valid,
        model.input_uid: userid_valid,
        model.input_iid: itemid_valid,
        model.input_y: y_valid
    }

    def clip_labels(x):
        if x > 5:
            return 5
        elif x < 1:
            return 1
        else:
            return x

    preds, loss, accuracy, mae = sess.run([model.predictions, model.loss, model.accuracy, model.mae], feed_dict)
    acc_preds = [clip_labels(x) for x in preds]
    labels = [x[0] for x in y_valid]
    mse = calculateMSE(acc_preds, labels)**0.5
    return loss, mse, mae

def calculateMSE(X, Y):
    return sum([(x - y)**2 for x, y in zip(X, Y)])/len(X)

if __name__ == '__main__':
    print('loading data...')
    pkl_file = open(PARA_DATA, 'rb')
    para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    user_pos_length = para['user_pos_length']
    user_neg_length = para['user_neg_length']
    item_pos_length = para['item_pos_length']
    item_neg_length = para['item_neg_length']
    vocabulary_user_pos = para['user_pos_vocab']
    vocabulary_user_neg = para['user_neg_vocab']
    vocabulary_item_pos = para['item_pos_vocab']
    vocabulary_item_neg = para['item_neg_vocab']
    u_pos_text = para['u_pos_text']
    u_neg_text = para['u_neg_text']
    i_pos_text = para['i_pos_text']
    i_neg_text = para['i_neg_text']
    train_length = para['train_length']
    test_length = para['test_length']

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = Model.Model(
                user_num=user_num,
                item_num=item_num,
                f=factor_dim,
                user_pos_length=user_pos_length,
                user_neg_length=user_neg_length,
                item_pos_length=item_pos_length,
                item_neg_length=item_neg_length,
                user_pos_vocab_size=len(vocabulary_user_pos),
                user_neg_vocab_size=len(vocabulary_user_neg),
                item_pos_vocab_size=len(vocabulary_item_pos),
                item_neg_vocab_size=len(vocabulary_item_neg),
                embedding_size=EMBEDDING_DIM,
                filter_sizes=list(map(int, FILTER_SIZE.split(','))),
                num_filters=NUM_FILTERS,
                n_pos_aspect=n_pos_aspect,
                n_neg_aspect=n_neg_aspect
            )

            optimizer = tf.train.AdamOptimizer(3e-4, beta1=0.9, beta2=0.999, epsilon=1e-8)
            train_op_1 = optimizer.minimize(loss=model.loss, var_list=[model.user_Matrix])
            train_op_2 = optimizer.minimize(loss=model.loss, var_list=[model.item_Matrix])
            train_op_3 = optimizer.minimize(loss=model.loss, var_list=model.variables + [model.pos_W, model.neg_W])

            sess.run(tf.initialize_all_variables())

            if word2vec:
                sess.run(model.Wu_pos.assign(np.load('data/yelp/pre_Wu_pos.npy')))
                sess.run(model.Wu_neg.assign(np.load('data/yelp/pre_Wu_neg.npy')))
                sess.run(model.Wi_pos.assign(np.load('data/yelp/pre_Wi_pos.npy')))
                sess.run(model.Wi_neg.assign(np.load('data/yelp/pre_Wi_neg.npy')))

            l = (train_length / batch_size) + 1
            print(l)
            ll = 0
            epoch = 1
            best_mae = 5
            best_rmse = 5
            best_mse = 25
            train_mae = 0
            train_rmse = 0

            pkl_file = open(TRAIN_DATA, 'rb')
            train_data = pickle.load(pkl_file)
            train_data = np.array(train_data)
            pkl_file.close()

            pkl_file = open(VALID_DATA, 'rb')
            valid_data = pickle.load(pkl_file)
            valid_data = np.array(valid_data)
            pkl_file.close()

            data_size_train = len(train_data)
            data_size_valid = len(valid_data)
            ll = int(len(train_data) / batch_size) + 1

            for epoch in range(2):
                # Shuffle the data at each epoch
                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]
                # training for user_Matrix
                for iter in range(5):
                    for batch_num in range(ll):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size_train)
                        data_train = shuffled_data[start_index: end_index]

                        uid, iid, y_batch = zip(*data_train)

                        u_pos_batch = []
                        u_neg_batch = []
                        i_pos_batch = []
                        i_neg_batch = []

                        for i in range(len(uid)):
                            u_pos_batch.append(u_pos_text[uid[i][0]])
                            u_neg_batch.append(u_neg_text[uid[i][0]])
                            i_pos_batch.append(i_pos_text[iid[i][0]])
                            i_neg_batch.append(i_neg_text[iid[i][0]])
                        u_pos_batch = np.array(u_pos_batch)
                        u_neg_batch = np.array(u_neg_batch)
                        i_pos_batch = np.array(i_pos_batch)
                        i_neg_batch = np.array(i_neg_batch)

                        t_rmse, t_mae = train_step_1(u_pos_batch, u_neg_batch, i_pos_batch, i_neg_batch, uid, iid, y_batch)

                # training for item_Matrix
                for iter in range(5):
                    for batch_num in range(ll):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size_train)
                        data_train = shuffled_data[start_index: end_index]
                        uid, iid, y_batch = zip(*data_train)

                        u_pos_batch = []
                        u_neg_batch = []
                        i_pos_batch = []
                        i_neg_batch = []

                        for i in range(len(uid)):
                            u_pos_batch.append(u_pos_text[uid[i][0]])
                            u_neg_batch.append(u_neg_text[uid[i][0]])
                            i_pos_batch.append(i_pos_text[iid[i][0]])
                            i_neg_batch.append(i_neg_text[iid[i][0]])
                        u_pos_batch = np.array(u_pos_batch)
                        u_neg_batch = np.array(u_neg_batch)
                        i_pos_batch = np.array(i_pos_batch)
                        i_neg_batch = np.array(i_neg_batch)

                        t_rmse, t_mae = train_step_2(u_pos_batch, u_neg_batch, i_pos_batch, i_neg_batch, uid, iid, y_batch)

                # training for other variables
                for iter in range(5):
                    for batch_num in range(ll):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size_train)
                        data_train = shuffled_data[start_index: end_index]
                        uid, iid, y_batch = zip(*data_train)

                        u_pos_batch = []
                        u_neg_batch = []
                        i_pos_batch = []
                        i_neg_batch = []

                        for i in range(len(uid)):
                            u_pos_batch.append(u_pos_text[uid[i][0]])
                            u_neg_batch.append(u_neg_text[uid[i][0]])
                            i_pos_batch.append(i_pos_text[iid[i][0]])
                            i_neg_batch.append(i_neg_text[iid[i][0]])
                        u_pos_batch = np.array(u_pos_batch)
                        u_neg_batch = np.array(u_neg_batch)
                        i_pos_batch = np.array(i_pos_batch)
                        i_neg_batch = np.array(i_neg_batch)

                        t_rmse, t_mae = train_step_3(u_pos_batch, u_neg_batch, i_pos_batch, i_neg_batch, uid, iid, y_batch)

                # train loss
                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train = shuffled_data[start_index: end_index]
                    uid, iid, y_batch = zip(*data_train)

                    u_pos_batch = []
                    u_neg_batch = []
                    i_pos_batch = []
                    i_neg_batch = []

                    for i in range(len(uid)):
                        u_pos_batch.append(u_pos_text[uid[i][0]])
                        u_neg_batch.append(u_neg_text[uid[i][0]])
                        i_pos_batch.append(i_pos_text[iid[i][0]])
                        i_neg_batch.append(i_neg_text[iid[i][0]])
                    u_pos_batch = np.array(u_pos_batch)
                    u_neg_batch = np.array(u_neg_batch)
                    i_pos_batch = np.array(i_pos_batch)
                    i_neg_batch = np.array(i_neg_batch)
                    loss, t_rmse, t_mae = dev_step(u_pos_batch, u_neg_batch, i_pos_batch, i_neg_batch,
                                                   uid, iid, y_batch)
                    train_rmse = train_rmse + len(uid) * np.square(t_rmse)
                    train_mae = train_mae + len(uid) * t_mae
                print('Epoch' + str(epoch))
                print("train_rmse, mae:", train_rmse / data_size_train, train_mae / data_size_train)

                train_rmse = 0
                train_mae = 0

                loss_s = 0
                accuracy_s = 0
                mae_s = 0

                ll_test = int(len(valid_data) / batch_size) + 1
                for batch_num2 in range(ll_test):
                    start_index = batch_num2 * batch_size
                    end_index = min((batch_num2 + 1) * batch_size, data_size_valid)
                    data_valid = valid_data[start_index: end_index]

                    userid_valid, itemid_valid, y_valid = zip(*data_valid)

                    u_pos_valid = []
                    u_neg_valid = []
                    i_pos_valid = []
                    i_neg_valid = []
                    for i in range(len(userid_valid)):
                        u_pos_valid.append(u_pos_text[userid_valid[i][0]])
                        u_neg_valid.append(u_neg_text[userid_valid[i][0]])
                        i_pos_valid.append(i_pos_text[itemid_valid[i][0]])
                        i_neg_valid.append(i_neg_text[itemid_valid[i][0]])
                    u_pos_valid = np.array(u_pos_valid)
                    u_neg_valid = np.array(u_neg_valid)
                    i_pos_valid = np.array(i_pos_valid)
                    i_neg_valid = np.array(i_neg_valid)

                    loss, accuracy, mae = dev_step(u_pos_valid, u_neg_valid, i_pos_valid, i_neg_valid,
                                                   userid_valid, itemid_valid, y_valid)
                    loss_s = loss_s + loss
                    accuracy_s = accuracy_s + len(userid_valid) * np.square(accuracy)
                    mae_s = mae_s + len(userid_valid) * mae
                print("loss_valid {:g}, mse_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / test_length, accuracy_s / test_length,
                                                                                np.sqrt(accuracy_s / test_length),
                                                                                mae_s / test_length))
                mse = accuracy_s / test_length
                rmse = np.sqrt(accuracy_s / test_length)
                mae = mae_s / test_length
                if best_mse > mse:
                    best_mse = mse
                if best_rmse > rmse:
                    best_rmse = rmse
                if best_mae > mae:
                    best_mae = mae
                print("")
            print('best mse:', best_mse)
            print('best rmse:', best_rmse)
            print('best mae:', best_mae)

        print('end')



