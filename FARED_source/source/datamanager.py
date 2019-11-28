import os, inspect, glob, sys

import numpy as np

class DataSet(object):

    def __init__(self, key_tr, cycle):

        print("\n** Prepare the Dataset")

        self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/../../preprocessing_source/dataset_mfcc"

        self.keylist = glob.glob(os.path.join(self.data_path, "*"))
        self.keylist.sort() # sorting the subdir list is optional.
        for idx, sd in enumerate(self.keylist):
            self.keylist[idx] = sd.split('/')[-1]
        print(self.keylist)

        # List
        self.key_tot = self.keylist
        if('None' in key_tr): self.key_tr = [self.key_tot[0]]
        else: self.key_tr = key_tr

        # Dictionary
        self.sublist_total = {}
        self.sublist_train = {}

        for ktot in self.key_tot: # sub of subdir lsit
            self.sublist_total["%s" %(ktot)] = glob.glob(os.path.join(self.data_path, "%s" %(ktot), "data", "*"))
            self.sublist_total["%s" %(ktot)].sort()

            self.sublist_train["%s" %(ktot)] = self.sublist_total["%s" %(ktot)][:cycle]
            self.sublist_train["%s" %(ktot)].sort()
            self.sublist_total["%s" %(ktot)] = self.sublist_total["%s" %(ktot)][cycle:]
            self.sublist_total["%s" %(ktot)].sort()

            for idx, path in enumerate(self.sublist_total["%s" %(ktot)]):
                self.sublist_total["%s" %(ktot)][idx] = glob.glob(os.path.join(path, "*.npy"))
                self.sublist_total["%s" %(ktot)][idx].sort()

            for idx, path in enumerate(self.sublist_train["%s" %(ktot)]):
                self.sublist_train["%s" %(ktot)][idx] = glob.glob(os.path.join(path, "*.npy"))
                self.sublist_train["%s" %(ktot)][idx].sort()

        # Information of dataset
        self.am_tot = len(self.keylist)
        self.am_tr = len(self.key_tr)
        try: self.data_dim = np.load(self.sublist_total[self.key_tr[0]][0][0]).shape[0]
        except:
            print("\n\n!!! ERROR !!!")
            print("You must prepare at least more than 2 \'.wav\' files.")
            print("Or set the \'--cycle\' argument less than the number of \'.wav\' files\n\n")
            sys.exit()
        # Variable for using dataset
        self.kidx_tr = 0 # key
        self.kidx_tot = 0
        self.sidx_tr = 0 # subdir
        self.sidx_tot = 0
        self.nidx_tr = 0 # numpy
        self.nidx_tot = 0

        print("Total Record : %d" %(self.am_tot))
        print("Training Set  : %d" %(self.am_tr))
        print("Each data has %d dimension." %(self.data_dim))
        print("Training Key: " + str(self.key_tr))
        print("Test Key: " + str(self.key_tot))

    def next_batch(self, batch_size, sequence_length, v_key=None):

        data_bat = np.zeros((0, sequence_length, self.data_dim), float)
        data_bunch = np.zeros((0, self.data_dim), float)

        if(v_key is None): # training batch
            index_bank = self.nidx_tr
            while(True): # collect mini batch set
                while(True): # collect sequence set

                    if(data_bunch.shape[0] >= sequence_length): # break the loop when sequences are collected
                        data_bunch = data_bunch[:sequence_length]
                        break

                    nplsit = self.sublist_train[self.key_tr[self.kidx_tr]][self.sidx_tr]
                    np_data = np.load(nplsit[self.nidx_tr])

                    self.nidx_tr = self.nidx_tr + 1
                    if(self.nidx_tr >= len(nplsit)):
                        index_bank = 0
                        self.nidx_tr = 0
                        self.sidx_tr += 1

                        data_bunch = np.zeros((0, self.data_dim), float)

                        if(self.sidx_tr >= len(self.sublist_train[self.key_tr[self.kidx_tr]])):
                            self.sidx_tr = 0
                            self.kidx_tr = (self.kidx_tr + 1) % (self.am_tr)

                    data_tmp = np_data.reshape((1, self.data_dim))
                    data_bunch = np.append(data_bunch, data_tmp, axis=0)

                data_tmp = data_bunch.reshape((1, sequence_length, self.data_dim))
                data_bat = np.append(data_bat, data_tmp, axis=0)

                if(data_bat.shape[0] >= batch_size):  # break the loop when mini batch is collected
                    break
            self.nidx_tr = (index_bank + 1) % len(nplsit)
            return np.nan_to_num(data_bat) # replace nan to zero using np.nan_to_num

        else: # Usually used with 1 of the batch size.
            list_seqname = [] # it used for confirm the anomaly.

            index_bank = self.nidx_tot
            while(True): # collect mini batch set

                while(True): # collect sequence set
                    if(data_bunch.shape[0] >= sequence_length): # break the loop when sequences are collected
                        data_bunch = data_bunch[:sequence_length]
                        break

                    nplsit = self.sublist_total[v_key][self.sidx_tot]

                    list_seqname.append(nplsit[self.nidx_tot])
                    np_data = np.load(nplsit[self.nidx_tot])

                    self.nidx_tot = self.nidx_tot + 1
                    if(self.nidx_tot >= len(nplsit)):
                        index_bank = 0
                        self.nidx_tot = 0
                        self.sidx_tot += 1

                        data_bunch = np.zeros((0, self.data_dim), float)

                        if(self.sidx_tot >= len(self.sublist_total[v_key])):
                            self.sidx_tot = 0
                            print("Cannot make bunch anymore. (length %d at %d)" %(sequence_length, self.nidx_tot))
                            return None, None

                    data_tmp = np_data.reshape((1, self.data_dim))
                    data_bunch = np.append(data_bunch, data_tmp, axis=0)

                data_tmp = data_bunch.reshape((1, sequence_length, self.data_dim))
                data_bat = np.append(data_bat, data_tmp, axis=0)

                if(data_bat.shape[0] >= batch_size):  # break the loop when mini batch is collected
                    break
            self.nidx_tot = (index_bank + 1) % len(nplsit)
            return np.nan_to_num(data_bat), list_seqname # replace nan to zero using np.nan_to_num
