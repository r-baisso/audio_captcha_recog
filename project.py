print("\n--- Bem vindo ao Projeto de Reconhecimento de Audio ---")
print("\nRealizando Imports de Libs necessarias...")

import sys
import librosa
import glob
import random
import re
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from pprint import pprint
# import scipy.signal as sg
# from scipy.fftpack import dct
# from scipy.fftpack import fft
# from scipy.fftpack import ifft
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV


"""
# features descartadas ate o momento

# calculo do sinal filtrado por filtro passa-baixas de ButterWorth
def lowPassFilter(signal):
    N  = 5   # Filter order
    Wn = 0.5 # Cutoff frequency
    B, A = sg.butter(N, Wn, output='ba')
    smooth_data = sg.filtfilt(B,A, signal)
    return smooth_data

# calculo da serie discreta de cosenos do sinal
def calc_dct(segmento):
    av = segmento.to_numpy(copy=True)
    return dct(av, type=3, n=24)

# calcula a FFT do sinal
def calc_fft(segmento):
    av = segmento.to_numpy(copy=True)
    return abs(fft(av, n=24))
"""

# calculo do Chroma Energy Normalized Statistics (CENS) do sinal
def calc_chroma(segmento, sr):
    av = segmento.to_numpy(copy=True)
    return np.mean(librosa.feature.chroma_cqt(y=av, sr=sr, n_chroma=24).T, axis=0)

# calculo do Mel Spectrogram do sinal
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
    features = np.append(features, calc_chroma(segmento, sr))
    """
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

def get_x_y(path, reduce=False):
    files = get_files(path)
    Xt = []
    yt = []
    segmentos = []
    seed = random.randint(1, 1000)
    random.seed(seed)
    random.shuffle(files)
    # print(f"Semente gerada: {seed}\n")
    i = 0
    for f in files:
        data, sr = librosa.load(f, mono=True)
        segmentos += extract_intervals(data, 4)
        yt += get_labels(f)
        # print(i)
        if i > 30 and reduce:
            break    
        i += 1
        

    for segmento in segmentos:
        Xt.append(feature_extraction(segmento, sr))
    
    return Xt,yt

# lista os elementos (caracteres) unicos de um array
def unique_values(l):
    unique_list = []
    for x in l: 
        if x not in unique_list: 
            unique_list.append(x)
    return unique_list


# gera e imprime a matriz de confusao geral do modelo
def print_conf_mtx(yt, y_pred, labels, classifier):
    print(labels)
    print()
    cm = confusion_matrix(yt, y_pred, labels)
    print(cm)
    """
    print("\nMatrizes de Confusao Individuais")
    print(multilabel_confusion_matrix(yt, y_pred))
    """
    
    print("\nPlotando Confusion Matrix geral usando o matplotlib...")
    file_name = 'confusion_matrix_'+ classifier +'.png'
    print(f"\nPlot {file_name} salvo na pasta do projeto")
    print("Feche o arquivo de saída para continuar a execução")

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the ' + classifier + ' classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.locator_params(nbins=len(labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(file_name)
    plt.show()
    

# MAIN

train_path = 'TREINAMENTO/'
validation_path= 'VALIDACAO/'
test_path = 'TESTE/'

is_test = bool(int(sys.argv[1]))

if is_test:
    print(f"\nBuscando dados nas Pastas {train_path}, {validation_path} e {test_path} internas do projeto...")

    print("\nPreparando DataSet para Treino")
    X, y = get_x_y(train_path)
    Xv, yv = get_x_y(validation_path)

    X += (Xv)
    y += (yv)

    print("Preparando DataSet para Teste")
    Xt, yt = get_x_y(test_path)
else:
    print(f"\nBuscando dados nas Pastas {train_path} e {validation_path} internas do projeto...")

    print("\nPreparando DataSet para Treino")
    X, y = get_x_y(train_path)
    # X, y = get_x_y(train_path, True)
    print("Preparando DataSet para Teste/Validacao")
    Xt, yt = get_x_y(validation_path)
    # Xt, yt = get_x_y(validation_path, True)

labels = unique_values(y)


n_estimators = [int(x) for x in np.linspace(start = 500, stop = 2000, num = 10)]
# max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
bootstrap = [True]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()

print("\nPreparando para Validação Cruzada para Random Forest")
print("\nParametros de busca:")
pprint(random_grid)

print("\nRealizando Busca por Validação Cruzada")
rf_gcv = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 3, verbose=1, n_jobs = -1)
rf_gcv.fit(X, y)

print("\nMelhores parametros:")
pprint(rf_gcv.best_params_)

# Classificador Random Forest parte 1
# print("\nInstanciando Modelo Random Forest (n_estimators = 500 e max_depth = 50)")
# rfc = RandomForestClassifier(n_estimators = 500, max_depth = 50, random_state = 0)

rfc = rf_gcv.best_estimator_

# print("\nTreinando Modelo...")
# rfc.fit(X, y)

print("Realizando Classificacao...")
y_rfc = rfc.predict(Xt)

rfc_score = rfc.score(Xt, yt)
print(f"\nAcuracia do modelo Random Forest = {rfc_score:{4}.{4}}\n")

print("Relatorio de Classificacao")
print(classification_report(yt, y_rfc))

print("Matriz de Confusao Geral do Modelo Random Forest\n")
print_conf_mtx(yt, y_rfc, labels, "Random Forest")


#Classificador SVC

# print("\n\nInstanciando Modelo SVC (kernel linear)")
# svm = SVC(kernel='linear', probability=True, gamma='auto')

# print("\nTreinando Modelo...")
# svm.fit(X, y)

# print("Realizando Classificacao...")
# y_pred = svm.predict(Xt)

# svm_score = svm.score(Xt, yt)
# print(f"\nAcuracia do modelo SVC = {svm_score:{4}.{4}}\n")


# print("Relatorio de Classificacao")
# print(classification_report(yt, y_pred))

# print("Matriz de Confusao Geral do Modelo SVC\n")
# print_conf_mtx(yt, y_pred, labels, "SVC")

