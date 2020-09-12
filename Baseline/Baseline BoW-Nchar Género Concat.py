#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:43:32 2020

@author: root
"""
import os,codecs,sys
import re,time,shutil
import numpy as np
import pandas as pd
import string, warnings
import sklearn

from sklearn import model_selection, preprocessing, linear_model, metrics, svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_selection import VarianceThreshold
from joblib import dump, load
from sklearn.base import clone


from datetime import datetime

type_features = 2 # 0 = BoW / 1 = Nword / 2 = Nchar
input_path= "Documentos/Bases/sinSlangs"
input_pathT = "Documentos/Bases"
file = "Nuevo Sexo sinSlangs"
input_pathImp = "Documentos/ResultadosSS"





####### elige y procesa las características, bow, char, word #####
def extract_features(features,type_features,set_to_fit, set_to_transform):
    type_feature = int(type_features)
    features[type_feature].fit(set_to_fit)
    set_tranformed =  features[type_feature].transform(set_to_transform)
    return set_tranformed

#############################################################################
##########****** MAIN PARA ENTRENAR ALGORITMOS  *********##########
######*** Read in the raw text ***######


############****** ENTRENA MODELOS Y OBTIENE LAS METRICAS ******############
def baseline(classifiers, X_train, y_train_label, features,type_features,carpeta, prepro, numUsuarios,fecha):
    start_time2 = time.time()
    output.write('Fecha de creación del archivo: '+str(fecha))
    output.write('\nCarpeta usada: '+str(carpeta))
    output.write('\nPreprocesamiento de la info: '+str(prepro))
    output.write('\nNúmero de archivos analizados: '+str(numUsuarios))
    
    output.write('\n \n ---------------------------')
    start_time = time.time()
    warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
    #############*** fit the training dataset on the classifier ***#################
    #classifier.fit(X_train, y_train_lable)
    ####*** guardan los resulados de cada una de las 50 iteraciones de RepeatedStratifiedKFold****######
    scores_rskf = []
    f1,precision,accuracy,recall = [],[],[],[]
    scores_rskfx = []
    f1x,precisionx,accuracyx,recallx = [],[],[],[]

    ######*** se Elijió RepeatedStratifiedKFold porque nos da una muestra representativa de c/u de las clases
    #### y así evitar sesgos
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,random_state=36851234)
    i=0
    for train_index, test_index in rskf.split(X_train, y_train_label):
        #clone_clf = clone(classifier)
        X_train_foldsTxt, X_test_foldsTxt = X_train[train_index], X_train[test_index]
        y_train_folds, y_test_folds = y_train_label[train_index], y_train_label[test_index]
        print("***** EXTRAYENDO CARACTERISTICAS *******")
        X_train_folds =extract_features(features,type_features,X_train_foldsTxt, X_train_foldsTxt)
        X_test_folds = extract_features(features,type_features,X_train_foldsTxt, X_test_foldsTxt)
        #print("***** NUMERO DE CARACTERISTICAS *******")
        car = X_train_folds.shape
        i+=1
        print('*** Entrenando modelo ***',i,'\n')
        
        classifiers[0].fit(X_train_folds,y_train_folds)
        print('*** Predecir modelo ***',i,'\n')
        y_pred = classifiers[0].predict(X_test_folds)
        n_correct = sum(y_pred == y_test_folds)
        print('*** Entrenando modelo 2 ***',i,'\n')
        classifiers[1].fit(X_train_folds,y_train_folds)
        print('*** Predecir modelo 2 ***',i,'\n')
        y_predx = classifiers[1].predict(X_test_folds)
        n_correctx = sum(y_predx == y_test_folds)

        ########**** Métricas de evaluación ****#########
        scores_rskf.append(n_correct/len(y_pred))
        f1.append(metrics.f1_score(y_pred, y_test_folds, average='weighted', labels=np.unique(y_test_folds)))
        precision.append(metrics.precision_score(y_pred, y_test_folds,average='binary'))
        accuracy.append(metrics.accuracy_score(y_pred, y_test_folds))
        recall.append(metrics.recall_score(y_pred, y_test_folds,average='binary'))
        
        scores_rskfx.append(n_correctx/len(y_predx))
        f1x.append(metrics.f1_score(y_predx, y_test_folds, average='weighted', labels=np.unique(y_test_folds)))
        precisionx.append(metrics.precision_score(y_predx, y_test_folds,average='binary'))
        accuracyx.append(metrics.accuracy_score(y_predx, y_test_folds))
        recallx.append(metrics.recall_score(y_predx, y_test_folds,average='binary'))
            
        print("*** Termino el entrenamiento del modelo ***", i)
        print("--- %s seconds ---" % (time.time() - start_time2))
            
        #if i==1:
        #   break
            

    ##############*** mean y std de los scores obtenidos de  RepeatedStratifiedKFold***###################
    mean_score = np.mean(scores_rskf)
    standar_score =np.std(scores_rskf)
    
    mean_scorex = np.mean(scores_rskfx)
    standar_scorex =np.std(scores_rskfx)
        
    output.write('\n *** '+str(classifiers[0])+'\n ---------------------- ')
    output.write('\n ---------Validación RepeatedStratifiedKFold------------------')
    output.write('\n  Scores:' + str(scores_rskf))
    output.write('\n  Mean:' + str(mean_score))
    output.write('\n  Standard deviation:' + str(standar_score))
    output.write('\n  F1:  '+ str(np.mean(f1)))
    output.write('\n  precision:  ' + str(np.mean(precision)))
    output.write('\n  recall:  ' + str(np.mean(recall)))
    output.write('\n  exactitud (accuracy):  ' + str(np.mean(accuracy)))
    output.write('\n  elapsed time:' + str(time.time() - start_time) )
    output.write('\n  #características:' + str(car))
    
    output.write('\n \n ---------------------------')
    output.write('\n *** '+str(classifiers[1])+'\n ---------------------- ')
    output.write('\n ---------Validación RepeatedStratifiedKFold------------------')
    output.write('\n  Scores:' + str(scores_rskfx))
    output.write('\n  Mean:' + str(mean_scorex))
    output.write('\n  Standard deviation:' + str(standar_scorex))
    output.write('\n  F1:  '+ str(np.mean(f1x)))
    output.write('\n  precision:  ' + str(np.mean(precisionx)))
    output.write('\n  recall:  ' + str(np.mean(recallx)))
    output.write('\n  exactitud (accuracy):  ' + str(np.mean(accuracyx)))
    output.write('\n  elapsed time:' + str(time.time() - start_time) )
    output.write('\n  #características:' + str(car))
    
    
    
    
    
    print("*** Termino el split ***")
    print("--- %s seconds --- \n" % (time.time() - start_time2))


##############################################################################

print ('Comenzamos \nLeemos el path: ', input_path)

with open(input_pathT+"/"+"truth.txt","r") as lf:
    truth = {}
    txtfiles=[]
    labels_gen=[]
    labels_leng=[]
    for line in lf.readlines():
        line = line.split(":::")   
        truth[line[0]]=[]        
        for item in line[1:]:            
            truth[line[0]].append(item.rstrip())
#####**** Guarda en una lista los txtfile, las etiquetas ****########       
for item in os.listdir(input_path):
    if os.path.splitext(item)[1] == ".txt" and os.path.splitext(item)[0][:-1] in truth.keys():   
        with open(input_path+"/"+item) as fileLeer:
            txtfiles.append(fileLeer.read())
            labels_gen.append(truth[os.path.splitext(item)[0][:-1]][0])
            labels_leng.append(truth[os.path.splitext(item)[0][:-1]][1])


    
######*** Crea el dataframe y guarda el texto y las etiquetas ***######
trainDF = pd.DataFrame()
trainDF['text'] = txtfiles
trainDF['labels_gen'] = labels_gen
trainDF['labels_leng']=labels_leng

######***  Set de entrenamiento y etiquetas de entrenamiento  ***######   
X_train=trainDF['text']
y_train_gen=trainDF['labels_gen']
#y_train_leng = trainDF['labels_leng']

######*** Hacemos el encoder de los paises ***#######
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train_gen)

#print("Estas son las etiquetas: ",np.unique(y_train),"\nY estos el significado: ", encoder.inverse_transform(np.unique(y_train)))

################*******  PESADO  *******###########################
#######***  Parámmetros para vectorización ***######

ngram_range_tfidf_min= 3
ngram_range_tfidf_max= 3
max_features_tfidf_ngram= 10000
ngram_range_min_tfidf_char= 3
ngram_range_max_tfidf_char= 3
max_features_ngram_tfidf_char = 10000

###############  **PESADO**  Crea vectores con TF IDF   ##########
######*** BoW ***######
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',smooth_idf=False)

######*** N-gramas de palabras ***######
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(ngram_range_tfidf_min,ngram_range_tfidf_max), max_features=max_features_tfidf_ngram,smooth_idf=False)

######*** N-gramas de caracteres ***######
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char',  ngram_range=(ngram_range_min_tfidf_char,ngram_range_max_tfidf_char), smooth_idf=False)

#######*** Característica a utilizar
features =[]
features.append(tfidf_vect)
features.append(tfidf_vect_ngram)
features.append(tfidf_vect_ngram_chars)

######***   Clasificador a utilizar  ***######
classifiers = []
classifiers.append(linear_model.LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='sag', tol=0.0001, verbose=0,
                   warm_start=False))
classifiers.append(svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0))

#----------------------------------------------------------------------

if type_features ==0:
    nameExt = "BoW"
elif type_features ==1:
    nameExt = "Nword"
else:
    nameExt = "Nchar"


now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H-%M-%S")

rutaFileFinal = input_pathImp+"/"+nameExt+" "+file+" "+dt_string+".txt"
with open(rutaFileFinal,'w') as output:
    baseline(classifiers,X_train,y_train,features,type_features,input_path,
             nameExt, X_train.shape[0], dt_string) 
#    shutil.move(file,'./Resultado_leng/')
print('**** Ver resultados en el archivo generado ***')


