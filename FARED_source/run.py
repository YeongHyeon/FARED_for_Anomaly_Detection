import argparse

import tensorflow as tf

import source.neuralnet as nn
import source.datamanager as dman
import source.tf_process as tfp
import source.developer as developer
developer.print_stamp()

def main():
    training_keys = []
    training_keys.append(FLAGS.trkey)

    dataset = dman.DataSet(key_tr=training_keys, cycle=FLAGS.cycle)
    lstm = nn.LSTM_Model_TF1_14(data_dim=dataset.data_dim)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    tfp.training(sess=sess, neuralnet=lstm, saver=saver, dataset=dataset, batch_size=FLAGS.batch, sequence_length=FLAGS.trainlen, iteration=FLAGS.iter)
    tfp.validation(sess=sess, neuralnet=lstm, saver=saver, dataset=dataset, sequence_length=FLAGS.testlen)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=100, help='Mini-batch size') # 100 batch * 36 sequence = 10 sec
    parser.add_argument('--trainlen', type=int, default=30, help='Sequence length for contruct test bunch. Mini-batch is constructed by bunchs. (Bunch is not Batch)')
    parser.add_argument('--testlen', type=int, default=30, help='Sequence length for contruct test bunch. Mini-batch is constructed by bunchs. (Bunch is not Batch)')
    parser.add_argument('--iter', type=int, default=1000, help='Number of iteration for training')
    parser.add_argument('--trkey', type=str, default='None', help='Class name for training. Must prepare at least more than 2 .wav files.')
    parser.add_argument('--cycle', type=int, default=1, help='Number of cycle for training. One cycle means one .wav file. Before training .wav file it should be splited cycle unit. But test .wav file does not need splitting procedure')

    FLAGS, unparsed = parser.parse_known_args()

    main()
