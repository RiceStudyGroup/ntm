import tensorflow as tf
import numpy as np


class MANNCell():
    def __init__(self, rnn_size, memory_size, memory_vector_dim, head_num, gamma=0.95,
                 reuse=False, k_strategy='separate'):
        self.rnn_size = rnn_size
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.head_num = head_num  # #(read head) == #(write head)
        self.reuse = reuse
        self.controller = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        self.step = 0
        self.gamma = gamma
        self.k_strategy = k_strategy

    def __call__(self, x, prev_state):
        prev_read_vector_list = prev_state[
            'read_vector_list']  # read vector (the content that is read out, length = memory_vector_dim)
        prev_controller_state = prev_state['controller_state']  # state of controller (LSTM hidden state)
        # batch_size = 16, every line of memory is 40, controller is lstm cell, and rnn size is 200
        # x + prev_read_vector -> controller (RNN) -> controller_output

        controller_input = tf.concat([x] + prev_read_vector_list,
                                     axis=1)  # x's shape is batch_size by 405, is input concat label one hot,
        with tf.variable_scope('controller', reuse=self.reuse):  # batch_size by 405 + 40*4 = 565
            controller_output, controller_state = self.controller(controller_input, prev_controller_state)
        # output batch_size by 200,
        # controller_output     -> k (dim = memory_vector_dim, compared to each vector in M)
        #                       -> a (dim = memory_vector_dim, add vector, only when k_strategy='separate')
        #                       -> alpha (scalar, combination of w_r and w_lu)

        if self.k_strategy == 'summary':
            num_parameters_per_head = self.memory_vector_dim + 1
        elif self.k_strategy == 'separate':
            num_parameters_per_head = self.memory_vector_dim * 2 + 1  # 81,
        total_parameter_num = num_parameters_per_head * self.head_num  # 81 * 4 = 324
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            o2p_w = tf.get_variable('o2p_w', [controller_output.get_shape()[1], total_parameter_num],
                                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            o2p_b = tf.get_variable('o2p_b', [total_parameter_num],
                                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            parameters = tf.nn.xw_plus_b(controller_output, o2p_w, o2p_b)  # (16, 200) matmul with (200, 324)

        ## seems like in the paper, from equation (9) to (16) happens inside the LSTMCell, and we just use the output of the lstm cell
        head_parameter_list = tf.split(parameters, self.head_num, axis=1)  # split 16,324 to be 4 (16,81)

        # k, prev_M -> w_r
        # alpha, prev_w_r, prev_w_lu -> w_w
        # prev w r list is a list of 4 tensor whose shape is (16, 128)
        prev_w_r_list = prev_state['w_r_list']  # vector of weightings (blurred address) over locations
        prev_M = prev_state['M']  # batch_size, 128 rows and each row has 40
        prev_w_u = prev_state['w_u']  # 16, batch_size by 128
        prev_indices, prev_w_lu = self.least_used(prev_w_u)  # prev_indices , 16by128, so as prev_w_lu
        w_r_list = []
        w_w_list = []
        k_list = []
        a_list = []
        # p_list = []   # For debugging
        for i, head_parameter in enumerate(head_parameter_list):
            with tf.variable_scope('addressing_head_%d' % i):
                k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim],
                            name='k')  # the first half is used to generate k
                if self.k_strategy == 'separate':
                    a = tf.tanh(head_parameter[:, self.memory_vector_dim:self.memory_vector_dim * 2], name='a')
                sig_alpha = tf.sigmoid(head_parameter[:, -1:],
                                       name='sig_alpha')
                # one number for each batch, [:, -1:], : means each batch, -1: means the last one
                # if use head_parameter[:, -1], the shape is (16,) rather than (16,1), equal to tf.expand_dims(head_parameter[:, -1],dim=1)
                w_r = self.read_head_addressing(k, prev_M)  # eq (17) and (18) in this method, (16, 128)
                w_w = self.write_head_addressing(sig_alpha, prev_w_r_list[i], prev_w_lu)  # (16, 128)
            w_r_list.append(w_r)
            w_w_list.append(w_w)
            k_list.append(k)
            if self.k_strategy == 'separate':
                a_list.append(a)
            # p_list.append({'k': k, 'sig_alpha': sig_alpha, 'a': a})   # For debugging

        w_u = self.gamma * prev_w_u + tf.add_n(w_r_list) + tf.add_n(w_w_list)  # eq (20) (16, 128)

        # Set least used memory location computed from w_(t-1)^u to zero

        M_ = prev_M * tf.expand_dims(1. - tf.one_hot(prev_indices[:, -1], depth=self.memory_size), dim=2)
        # prev_indices[:,-1] is one index for each batch,
        # Writing

        M = M_
        with tf.variable_scope('writing'):
            for i in range(self.head_num):
                w = tf.expand_dims(w_w_list[i], axis=2)
                if self.k_strategy == 'summary':
                    k = tf.expand_dims(k_list[i], axis=1)
                elif self.k_strategy == 'separate':
                    k = tf.expand_dims(a_list[i], axis=1)
                M = M + tf.matmul(w, k)

        # Reading

        read_vector_list = []
        with tf.variable_scope('reading'):
            for i in range(self.head_num):
                read_vector = tf.reduce_sum(tf.expand_dims(w_r_list[i], dim=2) * M, axis=1)
                read_vector_list.append(read_vector)

        # controller_output -> NTM output

        NTM_output = tf.concat([controller_output] + read_vector_list, axis=1)

        state = {
            'controller_state': controller_state,
            'read_vector_list': read_vector_list,
            'w_r_list': w_r_list,
            'w_w_list': w_w_list,
            'w_u': w_u,
            'M': M,
        }

        self.step += 1
        return NTM_output, state

    def read_head_addressing(self, k, prev_M):
        """

        :param k: (batch_size, memory_col) = (16,40)
        :param prev_M: (batch_size, memory_row, memory_col) = (16,128,40)
        :return:
        """
        with tf.variable_scope('read_head_addressing'):
            # Cosine Similarity

            k = tf.expand_dims(k, axis=2)  # (16,40) -> (16, 40, 1)
            inner_product = tf.matmul(prev_M, k)  # (16, 128,1) the numerator of equation(17)
            k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True))
            """
            tf.square(k) same as k, (16,40,1)
            tf.reduce_sum(tf.square(k), axis=1, keep_dims=True), (16,1,1)
            tf.sqrt(.. ) = (16,1,1)
            corresponding to ||k_t|| in equation (17) in the paper
            """
            M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keep_dims=True))
            """
            prev_M is (16,128,40)
            after tf square, it's same,
            after reduce sum, it's (16,128,1)
            same after tf sqrt
            M_norm ||M_t(i)||, (16,128,1)
            
            for kt, it only has one norm for each batch
            for memory, it has 128 rows, and each row has a norm. 
            
            """
            norm_product = M_norm * k_norm
            # (batch_size, memory_row, 1) = (16,128,1)
            K = tf.squeeze(inner_product / (norm_product + 1e-8))  # eq (17)
            # squeeze will Removes dimensions of size 1 from the shape of a tensor.
            # inner_product / (norm_product + 1e-8) is (16,128,1)
            # after squeezing, it's (16,128)
            # Calculating w^c

            K_exp = tf.exp(K)
            w = K_exp / tf.reduce_sum(K_exp, axis=1, keep_dims=True)  # eq (18)

            return w

    def write_head_addressing(self, sig_alpha, prev_w_r, prev_w_lu):
        """

        :param sig_alpha: (16, 1)
        :param prev_w_r: (16, 128)
        :param prev_w_lu: (16, 128)
        :return:
        """
        with tf.variable_scope('write_head_addressing'):
            # Write to (1) the place that was read in t-1 (2) the place that was least used in t-1

            return sig_alpha * prev_w_r + (1. - sig_alpha) * prev_w_lu  # eq (22)

    def least_used(self, w_u):
        """
        firstly, pick the smallest 4 from w_u in each batch, then use one-hot
        to convert them into [0 ,0 ,. ... 1,0 .. 0 ] ,128
                             [0 ,0 ,. .1. 0,0 .. 0 ] ,128
                             [0 ,0 1. .0. 0,0 .. 0 ] ,128
                             [0 ,0 ,. .0. 0,0 1. 0 ] ,128
        and then sum them up to be
                             [0 ,0 1. .1. 1,0 1. 0 ] ,128

        and this 16 by 128 will be the least used weights.
        :param w_u: (16,128), batch_size, memory_row
        :return:
        """
        _, indices = tf.nn.top_k(w_u, k=self.memory_size)  # for each batch, sort the w_u,
        w_lu = tf.reduce_sum(tf.one_hot(indices=indices[:, -self.head_num:], depth=self.memory_size), axis=1)
        # indices[:, -self.head_num:] is used to pick the last 4 rows that is least used
        return indices, w_lu  # shape of w_lu is still batch_size by 128, but it's one hot

    def zero_state(self, batch_size, dtype):
        one_hot_weight_vector = np.zeros([batch_size, self.memory_size])
        one_hot_weight_vector[..., 0] = 1
        one_hot_weight_vector = tf.constant(one_hot_weight_vector, dtype=tf.float32)
        with tf.variable_scope('init', reuse=self.reuse):
            state = {
                'controller_state': self.controller.zero_state(batch_size, dtype),
                'read_vector_list': [tf.zeros([batch_size, self.memory_vector_dim])
                                     for _ in range(self.head_num)],
                'w_r_list': [one_hot_weight_vector for _ in range(self.head_num)],
                'w_u': one_hot_weight_vector,
                'M': tf.constant(np.ones([batch_size, self.memory_size, self.memory_vector_dim]) * 1e-6,
                                 dtype=tf.float32)
            }
            return state
