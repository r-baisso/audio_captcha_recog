print("\n-- Bem vindo ao Projeto de Reconhecimento de Audio --")
print("\nRealizando Imports de Libs necessarias...")

import librosa
import math
import glob
import random
import re
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sg
from scipy.fftpack import dct
from scipy.fftpack import fft
# from scipy.fftpack import ifft
from sklearn.metrics import confusion_matrix
# from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# calculo do sinal filtrado por filtro passa-baixas de ButterWorth
def lowPassFilter(signal):
    N  = 5   # Filter order
    Wn = 0.5 # Cutoff frequency
    B, A = sg.butter(N, Wn, output='ba')
    smooth_data = sg.filtfilt(B,A, signal)
    return smooth_data

def calc_chroma(segmento, sr):
    av = segmento.to_numpy(copy=True)
    return np.mean(librosa.feature.chroma_cens(y=av, sr=sr, n_chroma=24).T, axis=0)

# calculo da serie discreta de cosenos do sinal
def calc_dct(segmento):
    av = segmento.to_numpy(copy=True)
    return dct(av, type=3, n=24)

# calcula a FFT do sinal
def calc_fft(segmento):
    av = segmento.to_numpy(copy=True)
    return abs(fft(av, n=24))

# calcula o Mel Spectrogram do sinal
def calc_mel(segmento, sr):
    av = segmento.to_numpy(copy=True)
    return np.mean(librosa.feature.melspectrogram(av, sr=sr).T, axis=0)

# calculo do MFCC medio para cada audio de entrada
def calc_mfcc(segmento, sr):
    av = segmento.to_numpy(copy=True)
    return np.mean(librosa.feature.mfcc(y=av, sr=sr, n_mfcc=24).T, axis=0)

# realiza a extracao de features do sinal
def feature_extraction(segmento, sr):
    features = []
    np.array(features)
    
    features = np.append(features, calc_mfcc(segmento, sr))
    features = np.append(features, calc_mel(segmento, sr))
  
    """
    features = np.append(features, calc_chroma(segmento, sr))
    features = np.append(features, calc_dct(segmento))
    features = np.append(features, calc_fft(segmento))
    """
    return features

# divisao dos audios de entrada para separacao de cada letra
def extract_intervals(signal, cut):
    data_interval = []
    interval = int(len(signal) // cut)
    for i in range(0, cut) :
        data_interval.append(pd.Series(signal[i*interval : (i+1)*interval]))
    return data_interval

# quebra o nome do arquivo em letras para categorizacao dos audios
def get_labels(path_file):
    path_file = re.sub("[ (1)]", "", path_file)
    return list(path_file[-8:-4])

# faz a aquisicao do caminho de todos os arquivos na pasta path
def get_files(path):
    files = glob.glob(path + "*.wav")
    return files

def get_x_y(path, seed=4):
    files = get_files(path)
    Xt = []
    yt = []
    segmentos = []
    random.seed(seed)
    random.shuffle(files)

    for f in files:
        data, sr = librosa.load(f, mono=True)
        segmentos += extract_intervals(data, 4)
        yt += get_labels(f)    

    for segmento in segmentos:
        Xt.append(feature_extraction(segmento, sr))
    
    return Xt,yt


def unique_values(l):
    unique_list = []
    for x in l: 
        if x not in unique_list: 
            unique_list.append(x)
    return unique_list


# gera e imprime a matriz de confusao do modelo
def print_conf_mtx(yt, y_pred, labels):
    print(confusion_matrix(yt, y_pred, labels))

    """
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
    """

# MAIN

"""
data, fs = librosa.load("TREINAMENTO/xbcb.wav", sr=8000)
amostras = data.shape[0]
print(f"frequencia de amostragem = {fs}")
print(f"quantidade de amostras = {amostras}")
"""

path = 'TREINAMENTO/'
test_path = 'VALIDACAO/'
print(f"\nBuscando dados nas Pastas {path} e {test_path} internas do projeto...")

print("\nExtraindo DataSet para Treino")
X, y = get_x_y(path)
print("Extraindo DataSet para Teste/Validacao")
Xt, yt = get_x_y(test_path)


# Classificador SVC

print("\nInstanciando Modelo SVC (kernel linear)")
svm = SVC(kernel='linear', probability=True, gamma='auto')

print("\nTreinando Modelo...")
svm.fit(X, y)

print("Realizando Classificacao...")
y_pred = svm.predict(Xt)

svm_score = svm.score(Xt, yt)
print(f"\nAcuracia do modelo SVC = {svm_score:{4}.{4}}\n")

print("Matriz de Confusao do Modelo SVC\n")
labels = unique_values(y)
print_conf_mtx(yt, y_pred, labels)


# Classificador KNN

print("\n\nInstanciando Modelo KNN (3 neighbours)")
knc = KNeighborsClassifier(n_neighbors = 3, n_jobs = 3, p = 1,  weights='uniform')

print("\nTreinando Modelo...")
knc.fit(X, y)

print("Realizando Classificacao...")
y_knc = knc.predict(Xt)

knn_score = knc.score(Xt, yt)
print(f"\nAcuracia do modelo KNN = {knn_score:{4}.{4}}\n")

print("Matriz de Confusao do Modelo KNN\n")
print_conf_mtx(yt, y_knc, labels)

