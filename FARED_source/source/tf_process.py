import os, inspect, time, scipy.misc, pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):
    try: os.mkdir(path)
    except: print("%s is already exists." %(path))

def data2canvas(data, dy, dx):
    nx, ny = 10, 10
    i, j = 0, 0
    canvas = np.zeros((dy*ny+10*(ny-1), dx*nx+10*(nx-1)))
    for da in data:
        if(np.min(da) < 0):
            da += abs(np.min(da))
        canvas[(nx-i-1)*dy+(nx-i-1)*10:(nx-i)*dy+(nx-i-1)*10, j*dx+j*10:(j+1)*dx+(j)*10] = da.reshape((dy, dx))
        j += 1
        if(j >= 10):
            i += 1
            j = 0
        if(i >= 10):
            break
    return canvas

def training(sess, neuralnet, saver,
             dataset, batch_size, sequence_length,
             iteration):

    start_time = time.time()
    loss_tr = 0
    list_loss = []
    print("\n** Training of the LSTM model to %d iterations | Batch size: %d" %(iteration, batch_size))

    make_dir(path=os.path.join(PACK_PATH, 'results'))

    train_writer = tf.summary.FileWriter(PACK_PATH+'/logs')
    for it in range(iteration):

        D_tr = dataset.next_batch(batch_size=batch_size, sequence_length=sequence_length)
        summaries, _ = sess.run([neuralnet.summaries, neuralnet.train], feed_dict={neuralnet.inputs:D_tr, neuralnet.outputs:D_tr})
        loss_tr = sess.run(neuralnet.loss, feed_dict={neuralnet.inputs:D_tr, neuralnet.outputs:D_tr})
        list_loss.append(loss_tr)
        train_writer.add_summary(summaries, it)

        if(it % 100 == 0):
            Y_pred = sess.run(neuralnet.logits, feed_dict={neuralnet.inputs:D_tr, neuralnet.outputs:D_tr})
            scipy.misc.imsave(os.path.join(PACK_PATH, 'results', str(it)+"_pred.png"), data2canvas(data=Y_pred, dy=sequence_length, dx=dataset.data_dim))
            scipy.misc.imsave(os.path.join(PACK_PATH, 'results', str(it)+"_gt.png"), data2canvas(data=D_tr, dy=sequence_length, dx=dataset.data_dim))
        print("Iteration [%d / %d] | Loss: %f" %(it, iteration, loss_tr))

        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("Final iteration | Loss: %f" %(loss_tr))

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    list_loss = np.asarray(list_loss)
    np.save("loss", list_loss)

    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(list_loss, color='blue', linestyle="-", label="loss")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("loss.png")
    plt.close()

    if(list_loss.shape[0] > 100):
        sparse_loss_x = np.zeros((100))
        sparse_loss = np.zeros((100))

        unit = int(list_loss.shape[0]/100)
        for i in range(100):
            sparse_loss_x[i] = i * unit
            sparse_loss[i] = list_loss[i * unit]

        plt.clf()
        plt.rcParams['font.size'] = 15
        plt.plot(sparse_loss_x, sparse_loss, color='blue', linestyle="-", label="loss")
        plt.ylabel("loss")
        plt.xlabel("iteration")
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.savefig("loss_sparse-ver.png")
        plt.close()

def validation(sess, neuralnet, saver,
               dataset, sequence_length):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    make_dir(path=os.path.join(PACK_PATH, 'valids'))

    print("\n** Validation of the LSTM model with %d sets." %(dataset.am_tot))
    plt.clf()
    plt.rcParams['font.size'] = 15
    for key in dataset.key_tot:

        list_dists = []
        list_bunchs = []

        print("* Validation of %s" %(key))
        valcnt = 0
        start_time = time.time()
        while(True):

            D_te, b_list = dataset.next_batch(batch_size=1, sequence_length=sequence_length, v_key=key)
            if(D_te is None):
                break

            l2dist = sess.run(neuralnet.loss, feed_dict={neuralnet.inputs:D_te, neuralnet.outputs:D_te})
            list_dists.append(l2dist)
            list_bunchs.append(b_list)

            if(valcnt % 100 == 0):
                print("%d-th bunch of %s | L2 Dist: %f" %(valcnt, key, l2dist))
            valcnt += 1

        elapsed_time = time.time() - start_time
        try: print("Avg propagation time: %.3f sec" %(elapsed_time/valcnt))
        except: print("Avg propagation time: 0 sec")

        list_dists = np.asarray(list_dists)
        plt.plot(list_dists, label=key)
        plt.ylabel("loss")

        np.save("%s/valids/%s_l2dist" %(PACK_PATH, key), list_dists)
        with open("%s/valids/%s_bunchs" %(PACK_PATH, key), 'wb') as fp:
            pickle.dump(list_bunchs, fp)

    plt.legend(loc="best")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("compare_loss.png")
    plt.close()
