Fast Adaptive RNN Encoder-Decoder for Anomaly Detection in SMD Assembly Machine
=====

## Introduction
This repository provides the source code of the paper <a href="https://www.mdpi.com/1424-8220/18/10/3573">"Fast Adaptive RNN Encoder-Decoder for Anomaly Detection in SMD Assembly Machine"</a>.

<div align="center">
  <img src="./figures/microphone.png" width="500">  
  <p>The SMD assembly machine with microphone (red box)</p>
</div>


## Requirements
* Python 3.5.2  
* Tensorflow 1.4.0  
* Numpy 1.13.3  
* Scipy 1.2.0  
* Matplotlib 3.0.2  
* Librosa 0.5.1


## Usage
### Preparing the dataset
First, Organize the audio dataset and keep as below.  
```
Dataset
├── AT2-IN88-SINK
│   ├── data_1.wav
│   ├── data_2.wav
│   ├── data_3.wav
│   │     ...
│   └── data_n.wav
├── NA-9473
│     ...
└── ST-4214-GE
```
Then, run the python script as following.  
```
$ cd preprocessing_source
$ python dat2npy_mfcc.py
```
Use `dat2npy_stft.py` instead of `dat2npy_mfcc.py` if you want to train FARED with Short Time Fourier Transform (STFT). The sample dataset is available at [sample_data](https://github.com/YeongHyeon/FARED_for_Anomaly_Detection/tree/master/sample_data).

### Training and Test
```
$ cd FARED_source
$ python run.py
```
:exclamation: The result of the experiment may differ to paper because we provide only sample audio data.

<div align="center">
  <img src="./figures/model.png" width="500">  
  <p>Architecture of Fast Adaptive RNN Encoder-Decoder</p>  
</div>


### BibTeX
```
@Article{s18103573,
  AUTHOR = {Park, YeongHyeon and Yun, Il Dong},
  TITLE = {Fast Adaptive RNN Encoder–Decoder for Anomaly Detection in SMD Assembly Machine},
  JOURNAL = {Sensors},
  VOLUME = {18},
  YEAR = {2018},
  NUMBER = {10},
  ARTICLE-NUMBER = {3573},
  URL = {http://www.mdpi.com/1424-8220/18/10/3573},
  ISSN = {1424-8220},
  DOI = {10.3390/s18103573}
}
```
