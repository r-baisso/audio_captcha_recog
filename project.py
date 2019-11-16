#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import math
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sg
import re
from scipy.fftpack import dct
from scipy.fftpack import fft
from scipy.fftpack import ifft
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# In[2]:


#calculo o do MFCC medio para cada audio de entrada
def calc_mfcc(segmento, sr):
    av = segmento.to_numpy(copy=True)
    return np.mean(librosa.feature.mfcc(y=av, sr=sr, n_mfcc=24).T,axis=0)


# In[3]:


#divisao dos audios de entrada para separacao de cada letra
def extract_intervals(signal, cut):
    data_interval = []
    interval = int(len(signal) // cut)
    for i in range(0, cut) :
        data_interval.append(pd.Series(data[i*interval : (i+1)*interval]))
    return data_interval


# In[4]:


#calculo do sinal filtrado po filtro passa-baixas de ButterWorth
def lowPassFilter(signal):
    N  = 5   # Filter order
    Wn = 0.5 # Cutoff frequency
    B, A = sg.butter(N, Wn, output='ba')
    smooth_data = sg.filtfilt(B,A, signal)
    return smooth_data


# In[5]:


#calculo da serie discreta de cosenos do sinal
def calc_dct(segmento):
    av = segmento.to_numpy(copy=True)
    return dct(av, type=3, n=24)


# In[6]:


#calcula o Mel Spectrogram do sinal
def calc_mel(segmento):
    av = segmento.to_numpy(copy=True)
    return np.mean(librosa.feature.melspectrogram(av, sr=sr).T,axis=0)


# In[7]:


#calcula a FFT do sinal
def calc_fft(segmento):
    av = segmento.to_numpy(copy=True)
    return abs(fft(av, n=24))


# In[8]:


#faz a aquisicao do caminho de todos os arquivos na pasta path
def get_files(path):
    files = glob.glob(path + "*.wav")
    return files
#quebra o nome do arquivo em letras para categorizacao dos audios
def get_labels(path_file):
    path_file = re.sub("[ (1)]", "", path_file)
    return list(path_file[-8:-4])


# In[88]:


#realiza a extracao de features do sinal
def feature_extraction(segmento, sr):

    features = []
    np.array(features)
    
    features=np.append(features, calc_mfcc(segmento, sr))
    features=np.append(features, calc_mel(segmento))
    features=np.append(features, calc_dct(segmento))
    features=np.append(features, calc_fft(segmento))

    return features


# In[90]:


def get_x_y(path, seed=4):
    files = get_files(path)

    Xt = []
    yt = []
    segmentos = []
    random.seed(seed)
    random.shuffle(files)
    for f in files:
        data, sr = librosa.load(f, mono=True)
        segmentos += (extract_intervals(data, 4))
        yt+=get_labels(f)    

    for segmento in segmentos:
        Xt.append(feature_extraction(segmento, sr))
    
    return Xt,yt


# In[91]:


def unique_values(l):
    unique_list = []
    for x in l: 
        if x not in unique_list: 
            unique_list.append(x)
    return unique_list


# In[93]:


def print_conf_mtx(yt , y_pred, labels):
    cm = confusion_matrix(yt, y_pred, labels)
    
    print(cm)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.locator_params(nbins=len(labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[10]:


#amostras = data.shape[0]
#print(f"frequencia de amostragem = {sr}")
#print(f"quantidade de amostras = {amostras}")


# In[32]:


path = 'TREINAMENTO/'
files = get_files(path)


# In[33]:


X,y = get_x_y(path)


# In[80]:


svm = SVC(kernel='linear', class_weight='balanced',probability=True, gamma='auto')


# In[81]:


svm.fit(X,y)


# In[37]:


test_path = 'VALIDACAO/'
Xt, yt = get_x_y(test_path)


# In[82]:


y_pred = svm.predict(Xt)


# In[66]:


yt == y_pred


# In[41]:


from sklearn.metrics import confusion_matrix


# In[83]:


svm.score(Xt, yt)


# In[94]:


labels = unique_values(y)
print_conf_mtx(yt, y_pred, labels)

