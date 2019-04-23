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
    lstm = nn.LSTM_Model(data_dim=dataset.data_dim)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    tfp.training(sessl=sess, neuralnet=lstm, saver=saver, dataset=dataset, batch_size=FLAGS.batch, sequence_length=FLAGS.trainlen, iteration=FLAGS.iter)
    tfp.validation(sess=sess, neuralnet=lstm, saver=saver, dataset=dataset, sequence_length=FLAGS.testlen)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=100, help='-') # 100 batch * 36 sequence = 10 sec
    parser.add_argument('--trainlen', type=int, default=30, help='-')
    parser.add_argument('--testlen', type=int, default=30, help='-')
    parser.add_argument('--iter', type=int, default=1000, help='-')
    parser.add_argument('--trkey', type=str, default='AT2-IN88-SINK', help='-')
    parser.add_argument('--cycle', type=int, default=7, help='-')

    FLAGS, unparsed = parser.parse_known_args()

    main()
