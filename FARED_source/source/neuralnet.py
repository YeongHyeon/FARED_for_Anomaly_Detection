import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

class LSTM_Model(object):

    def __init__(self, data_dim=None):

        print("\n** Initialize the LSTM Model")

        print(data_dim)
        self.inputs = tf.placeholder(tf.float32, [None, None, data_dim])
        self.outputs = tf.placeholder(tf.float32, [None, None, data_dim])

        with tf.variable_scope('lstm'):
            # Activated by tanh
            self.cell_1 = tf.contrib.rnn.LSTMCell(num_units=data_dim, initializer=tf.contrib.layers.xavier_initializer(), forget_bias=1.0)
            self.cell_2 = tf.contrib.rnn.LSTMCell(num_units=data_dim, initializer=tf.contrib.layers.xavier_initializer(), forget_bias=1.0)
            self.cell_3 = tf.contrib.rnn.LSTMCell(num_units=data_dim, initializer=tf.contrib.layers.xavier_initializer(), forget_bias=1.0)
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell_1, self.cell_2, self.cell_3])

            self.logits, states = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.inputs, dtype=tf.float32)

        self.vars_lstm = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm')
        self.loss = tf.sqrt(tf.reduce_sum(tf.square(self.logits - self.outputs)))
        self.train = tf.train.RMSPropOptimizer(learning_rate=0.005).minimize(loss=self.loss, var_list=self.vars_lstm)

        w_cell_1, b_cell_1 = self.cell_1.variables
        tf.summary.histogram("w_cell_1", w_cell_1)
        tf.summary.histogram("b_cell_1", b_cell_1)
        w_cell_2, b_cell_2 = self.cell_2.variables
        tf.summary.histogram("w_cell_2", w_cell_2)
        tf.summary.histogram("b_cell_2", b_cell_2)
        w_cell_3, b_cell_3 = self.cell_3.variables
        tf.summary.histogram("w_cell_3", w_cell_3)
        tf.summary.histogram("b_cell_3", b_cell_3)

        tf.summary.scalar('loss', self.loss)
        self.summaries = tf.summary.merge_all()
