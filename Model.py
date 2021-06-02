import tensorflow as tf
from tensorflow.contrib import layers

class Model(object):
    def __init__(self, user_num, item_num, f, user_pos_length, user_neg_length, item_pos_length, item_neg_length,
                 user_pos_vocab_size, user_neg_vocab_size, item_pos_vocab_size, item_neg_vocab_size, embedding_size,
                 filter_sizes, num_filters, n_pos_aspect, n_neg_aspect):

        self.input_u_pos = tf.placeholder(tf.int32, [None, user_pos_length], name='input_u_pos')
        self.input_u_neg = tf.placeholder(tf.int32, [None, user_neg_length], name='input_u_neg')
        self.input_i_pos = tf.placeholder(tf.int32, [None, item_pos_length], name='input_i_pos')
        self.input_i_neg = tf.placeholder(tf.int32, [None, item_neg_length], name='input_i_neg')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("user_pos_embedding"):
            self.Wu_pos = tf.Variable(tf.random_uniform([user_pos_vocab_size, embedding_size], -1.0, 1.0), trainable=False, name='Wu_pos')
            self.embedded_user_pos = tf.nn.embedding_lookup(self.Wu_pos, self.input_u_pos)
            self.embedded_users_pos = tf.expand_dims(self.embedded_user_pos, -1)

        with tf.name_scope("user_neg_embedding"):
            self.Wu_neg = tf.Variable(tf.random_uniform([user_neg_vocab_size, embedding_size], -1.0, 1.0), trainable=False, name='Wu_neg')
            self.embedded_user_neg = tf.nn.embedding_lookup(self.Wu_neg, self.input_u_neg)
            self.embedded_users_neg = tf.expand_dims(self.embedded_user_neg, -1)

        with tf.name_scope("item_pos_embedding"):
            self.Wi_pos = tf.Variable(tf.random_uniform([item_pos_vocab_size, embedding_size], -1.0, 1.0), trainable=False, name='Wi_pos')
            self.embedded_item_pos = tf.nn.embedding_lookup(self.Wi_pos, self.input_i_pos)
            self.embedded_items_pos = tf.expand_dims(self.embedded_item_pos, -1)

        with tf.name_scope("item_neg_embedding"):
            self.Wi_neg = tf.Variable(tf.random_uniform([item_neg_vocab_size, embedding_size], -1.0, 1.0), trainable=False, name='Wi_neg')
            self.embedded_item_neg = tf.nn.embedding_lookup(self.Wi_neg, self.input_i_neg)
            self.embedded_items_neg = tf.expand_dims(self.embedded_item_neg, -1)

        with tf.name_scope("user_latent_factors"):
            self.user_Matrix = tf.Variable(tf.random_uniform([user_num, f], -1.0, 1.0), name='user_Matrix')
            self.user_latent = tf.nn.embedding_lookup(self.user_Matrix, self.input_uid)
            self.user_latent = tf.reshape(self.user_latent, [-1, f])

        with tf.name_scope("item_latent_factors"):
            self.item_Matrix = tf.Variable(tf.random_uniform([item_num, f], -1.0, 1.0), name='item_Matrix')
            self.item_latent = tf.nn.embedding_lookup(self.item_Matrix, self.input_iid)
            self.item_latent = tf.reshape(self.item_latent, [-1, f])

        with tf.name_scope("pos_aspect_weight"):
            self.pos_W = tf.Variable(tf.random_uniform([n_pos_aspect, f], -1.0, 1.0), name='pos_W')

        with tf.name_scope("neg_aspect_weight"):
            self.neg_W = tf.Variable(tf.random_uniform([n_neg_aspect, f], -1.0, 1.0), name='neg_W')

        output_u_pos = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_pos_conv-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_users_pos,
                    W,
                    strides=[1, 1, embedding_size, 1],
                    padding="SAME",
                    name="conv")  # batch_size * user_pos_length * 1 * num_filters
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h1 = tf.reshape(h, [-1, user_pos_length, num_filters])
                output_u_pos.append(h1)

        num_filters_total = num_filters * len(filter_sizes)
        self.output_u_pos_con = tf.concat(output_u_pos, 2)
        self.output_u_pos_res = tf.reshape(self.output_u_pos_con, [-1, num_filters_total])
        # Layer 1
        Wu_pos_1 = tf.get_variable("Wu_pos_1", shape=[num_filters_total, n_pos_aspect],
                                   initializer=tf.contrib.layers.xavier_initializer())
        bu_pos_1 = tf.Variable(tf.constant(0.1, shape = [n_pos_aspect]))
        self.u_pos_l1 = tf.nn.softmax(tf.nn.relu(tf.matmul(self.output_u_pos_res, Wu_pos_1) + bu_pos_1))


        self.pos_asp = tf.reduce_sum(tf.reshape(self.u_pos_l1, [-1, user_pos_length, n_pos_aspect]), axis=1)
        self.pos_asp_imp = tf.nn.softmax(self.pos_asp)  # batch_size * n_pos_aspect


        output_u_neg = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_neg_conv-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1,
                                num_filters]  # [filter_height, filter_width, in_channels, out_channels]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_users_neg,
                    W,
                    strides=[1, 1, embedding_size, 1],
                    padding="SAME",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h1 = tf.reshape(h, [-1, user_neg_length, num_filters])
                output_u_neg.append(h1)

        self.output_u_neg_con = tf.concat(output_u_neg, 2)
        self.output_u_neg_res = tf.reshape(self.output_u_neg_con, [-1, num_filters_total])
        # Layer 1
        Wu_neg_1 = tf.get_variable("Wu_neg_1", shape=[num_filters_total, n_neg_aspect],
                                   initializer=tf.contrib.layers.xavier_initializer())
        bu_neg_1 = tf.Variable(tf.constant(0.1, shape=[n_neg_aspect]))
        self.u_neg_l1 = tf.nn.softmax(tf.nn.relu(tf.matmul(self.output_u_neg_res, Wu_neg_1) + bu_neg_1))


        self.neg_asp = tf.reduce_sum(tf.reshape(self.u_neg_l1, [-1, user_neg_length, n_neg_aspect]), axis=1)
        self.neg_asp_imp = tf.nn.softmax(self.neg_asp)  # batch_size * n_neg_aspect



        neg_asp_imp_add = []
        with tf.name_scope("pos2neg_imp"):
            W = tf.Variable(tf.truncated_normal(shape=[f, f], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[f]), name='b')
            h = tf.Variable(tf.truncated_normal(shape=[f, 1], stddev=0.1), name="h")
            for i in range(n_neg_aspect):
                neg_Wi = self.neg_W[i]
                mul = tf.multiply(self.pos_W, neg_Wi)
                rel = tf.nn.relu(tf.matmul(mul, W) + b)
                attn = tf.nn.softmax(tf.matmul(rel, h), dim=0)  # n_pos_aspect * 1
                neg_asp_imp_i = tf.matmul(self.pos_asp_imp, attn)  # batch_size * 1
                neg_asp_imp_add.append(neg_asp_imp_i)

        pos_asp_imp_add = []
        with tf.name_scope("neg2pos_imp"):
            W = tf.Variable(tf.truncated_normal(shape=[f, f], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[f]), name='b')
            h = tf.Variable(tf.truncated_normal(shape=[f, 1], stddev=0.1), name="h")
            for i in range(n_pos_aspect):
                pos_Wi = self.pos_W[i]
                mul = tf.multiply(self.neg_W, pos_Wi)
                rel = tf.nn.relu(tf.matmul(mul, W) + b)
                attn = tf.nn.softmax(tf.matmul(rel, h), dim=0)
                pos_asp_imp_i = tf.matmul(self.neg_asp_imp, attn)
                pos_asp_imp_add.append(pos_asp_imp_i)

        with tf.name_scope("prediction"):
            # print(self.user_latent.shape())
            self.interaction = tf.multiply(self.user_latent, self.item_latent)
            self.pos_asp_r = tf.matmul(self.interaction, tf.transpose(self.pos_W))  # batch_size * n_pos_asp
            self.pos_imp = self.pos_asp_imp + tf.concat(pos_asp_imp_add, -1)
            self.pos_r = tf.reduce_sum(tf.multiply(self.pos_asp_r, self.pos_imp), axis=-1)

            self.neg_asp_r = tf.matmul(self.interaction, tf.transpose(self.neg_W))
            self.neg_imp = self.neg_asp_imp + tf.concat(neg_asp_imp_add, -1)
            self.neg_r = tf.reduce_sum(tf.multiply(self.neg_asp_r, self.neg_imp), axis=-1)

            self.predictions = self.pos_r - self.neg_r

        regularizer = layers.l2_regularizer(scale=1.0)
        Var_list_1 = [Wu_pos_1, bu_pos_1, Wu_neg_1, bu_neg_1]

        for i, filter_size in enumerate(filter_sizes):
            Var_list_1 += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="user_pos_conv-%s" % filter_size)
            Var_list_1 += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="user_neg_conv-%s" % filter_size)

        reg_1 = layers.apply_regularization(regularizer, weights_list=Var_list_1)

        Var_list_2 = []

        Var_list_3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pos2neg_imp") \
                     + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="neg2pos_imp")

        reg_3 = layers.apply_regularization(regularizer, weights_list=Var_list_3)

        self.variables = Var_list_1 + Var_list_2 + Var_list_3

        reg_4 = layers.apply_regularization(regularizer, weights_list=[self.user_Matrix, self.item_Matrix, self.pos_W, self.neg_W])


        with tf.name_scope("loss"):
            beta_1 = 1e-4
            beta_2 = 0.001
            losses = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            self.loss = losses + beta_2 * (reg_1 + reg_3 + reg_4)


        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))


