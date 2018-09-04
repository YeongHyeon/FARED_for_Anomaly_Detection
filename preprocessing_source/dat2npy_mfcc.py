import glob, os, librosa

import scipy.io.wavfile

import numpy as np
import matplotlib.pyplot as plt

def makedir(path):
    try:
        os.mkdir(path)
        print("Make dir: %s" %(path))
    except:
        print("Already exist: %s" %(path))

def spectrums2timeaverage(spects=None):

    spect_sum = np.zeros((spects.shape[0]), float)

    spects_tr = np.transpose(spects)
    for sp in spects_tr:
        spect_sum = np.sum([spect_sum, sp], axis=0)
    spect_avg = spect_sum / spects.shape[1]

    return np.transpose(spect_avg)

def spectrum_normalizing(spectrum=None):
    ref = (np.max(spectrum) - np.min(spectrum))

    if(ref == 0):
        return spectrum

    spectrum = (spectrum + 0.0001) / ref

    ref = np.max(spectrum) - 1
    spectrum = spectrum - ref

    return spectrum

def main(smdname):

    smd = smdname.split('/')[-1]
    makedir("./dataset_mfcc/%s" %(smd))
    makedir("./dataset_mfcc/%s/data" %(smd))
    makedir("./dataset_mfcc/%s/plot" %(smd))
    makedir("./dataset_mfcc/%s/spectra" %(smd))

    list_wav = glob.glob(os.path.join(smdname, "*.wav"))
    list_wav.sort()
    for lw in list_wav:
        print(lw)
        sr, signal = scipy.io.wavfile.read(lw)

        window_len = int(sr/2)
        window_start = 0
        window_end = window_start + window_len

        wavname = lw.split('/')[len(lw.split('/'))-1].split('.')[0]
        makedir("./dataset_mfcc/%s/data/%s" %(smd, wavname))
        makedir("./dataset_mfcc/%s/plot/%s" %(smd, wavname))
        makedir("./dataset_mfcc/%s/spectra/%s" %(smd, wavname))

        while(True):
            data = signal[window_start:window_end].astype(float)
            melspectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048, hop_length=int(2048/4))
            mfcc_spectrum = spectrums2timeaverage(spects=melspectrogram)
            mfcc_spectrum = mfcc_spectrum / (np.max(mfcc_spectrum) - np.min(mfcc_spectrum))

            # mfcc_spectrum = mfcc_spectrum[:40]

            savename = "%s_%08d" %(wavname, window_start)
            # print(savename)

            if(window_end >= signal.shape[0]):
                break

            plt.clf()
            plt.rcParams['font.size'] = 15
            plt.imshow(np.rot90(melspectrogram, k=1), cmap='nipy_spectral_r')
            plt.ylabel("Time")
            plt.xlabel("Mel-scale Frequency")
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.savefig("./dataset_mfcc/%s/spectra/%s/%s.png" %(smd, wavname, savename))
            plt.close()

            plt.clf()
            plt.rcParams['font.size'] = 15
            plt.plot(mfcc_spectrum, linewidth=1, color='navy')
            plt.ylabel("MFCC")
            plt.xlabel("Mel-scale Frequency")
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.savefig("./dataset_mfcc/%s/spectra/%s/%s_avg.png" %(smd, wavname, savename))
            plt.close()

            plt.clf()
            plt.subplot(211)
            plt.title("Original Signal")
            plt.plot(signal[window_start:window_end], linewidth=1, color='black')
            plt.ylabel("Amplitude")
            plt.xlabel("Time (Sampling Rate: %d)" %(sr))
            plt.subplot(212)
            plt.title("MFCC Spectrum")
            plt.plot(mfcc_spectrum, linewidth=1, color='red')
            plt.ylabel("MFCC")
            plt.xlabel("Mel-scale Frequency")
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.savefig("./dataset_mfcc/%s/plot/%s/%s.png" %(smd, wavname, savename))
            plt.close()

            np.save("./dataset_mfcc/%s/data/%s/%s" %(smd, wavname, savename), mfcc_spectrum)

            window_start += int(window_len/2)
            window_end = window_start + window_len


if __name__ == '__main__':

    data_path = "./sample_data"

    smdlist = glob.glob(os.path.join(data_path, "*"))
    smdlist.sort()

    makedir("./dataset_mfcc")

    for smd in smdlist:
        print(smd)
        main(smdname=smd)
