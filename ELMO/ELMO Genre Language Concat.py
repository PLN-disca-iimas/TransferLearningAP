#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:58:49 2020

@author: root
"""

#importamos los paquetes que usaremos

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
#import torch
import random
from itertools import chain
from datetime import datetime
from datetime import timedelta



nombre = str(sys.argv[1])
input_path= "/001/usuarios/palacios.alc/archivos/basesResumidas/df/"
input_pathImp = "/001/usuarios/palacios.alc/archivos/resultadosELMO/"
pathWeights = "/001/usuarios/palacios.alc/archivos/ELMOmodel/145/"



def Evaluacion(classifiers,input_pathImp,columnaLabel,nombre, nameExt,df, type_label,columnasTexto):
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    dt_stringPuntos = now.strftime("%d-%m-%Y %H:%M:%S")
    #declaramos el path para guardar el archivo junto con el nombre del file
    filepath = input_pathImp+nameExt+" "+columnaLabel+" "+nombre+" "+dt_string+".txt"
      
    #comenzamos a escribir el archibo
    with open(filepath, "w") as output:
        
        #comenzamos a contar el tiempo
        start_time = time.time()
        #imprimimos datos relevantes para el txt
        output.write('Fecha de creación del archivo: '+str(dt_stringPuntos))
        output.write('\nPreprocesamiento usado: '+str(nombre))
        output.write('\nPreprocesamiento de la info: '+str(nameExt))
        output.write('\nNúmero de usuarios analizados: '+str(df.shape[0]))
        #
        output.write('\n ---------------------------')
        #warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
    
        ####*** guardan los resulados de cada una de las 50 iteraciones de RepeatedStratifiedKFold****######
        scores_rskf = []
        f1,precision,accuracy,recall = [],[],[],[]
        scores_rskfx = []
        f1x,precisionx,accuracyx,recallx = [],[],[],[]
    
        ######*** se Elijió RepeatedStratifiedKFold porque nos da una muestra representativa de c/u de las clases
        #### y así evitar sesgos
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,random_state=36851234)
        i=0
        # creamos una base donde aparecezcan solo los ids
        
        
        for train_index, test_index in rskf.split(df["text"],df["label"]):
            
            i+=1
            #hacemos el split entre train y test
            X_train_folds, X_test_folds = df.iloc[train_index][columnasTexto], df.iloc[test_index][columnasTexto]
            y_train_folds, y_test_folds =df.iloc[train_index]["label"], df.iloc[test_index]["label"]
            
            
            
            print('*** Entrenando modelo I ***',i)
            classifiers[0].fit(X_train_folds,y_train_folds)
            print('*** Predecir modelo I ***',i)
            y_pred = classifiers[0].predict(X_test_folds)
            
            print('*** Entrenando modelo II ***',i)
            classifiers[1].fit(X_train_folds,y_train_folds)
            print('*** Predecir modelo II ***',i)
            y_predx = classifiers[1].predict(X_test_folds)
            
            
            n_correct = sum(y_pred == y_test_folds)
            n_correctx = sum(y_predx == y_test_folds)
            ########**** Métricas de evaluación ****#########
            #utilizamos la misma métrica, ya sea si predecimos sexo o variante de lenguaje, pero cambiamos el 
            #parametro binary por weighted
            if type_label == 0:
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
            else:
                scores_rskf.append(n_correct/len(y_pred))
                f1.append(metrics.f1_score(y_pred, y_test_folds, average='weighted', labels=np.unique(y_test_folds)))
                precision.append(metrics.precision_score(y_pred, y_test_folds,average='weighted'))
                accuracy.append(metrics.accuracy_score(y_pred, y_test_folds))
                recall.append(metrics.recall_score(y_pred, y_test_folds,average='weighted'))
                
                scores_rskfx.append(n_correctx/len(y_predx))
                f1x.append(metrics.f1_score(y_predx, y_test_folds, average='weighted', labels=np.unique(y_test_folds)))
                precisionx.append(metrics.precision_score(y_predx, y_test_folds,average='weighted'))
                accuracyx.append(metrics.accuracy_score(y_predx, y_test_folds))
                recallx.append(metrics.recall_score(y_predx, y_test_folds,average='weighted'))
            #Fin del if
            #if i ==1:
            #    break
            print("*** Termino el entrenamiento de la capa ",i," ***")
            print("--- %s seconds ---" % (time.time() - start_time), "\n")
        #Fin del for
        
        
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
        
        print("Terminamos.")
### Fin de la función
#-----------------------------------------------------------------------------


if __name__ == "__main__":
    ######*** Crea el dataframe y guarda el texto y las etiquetas ***######
    trainDF = pd.read_csv(input_path+nombre+".csv", encoding="utf-8-sig")
    ######***  Set de entrenamiento y etiquetas de entrenamiento  ***######   
    print("Shape base ", trainDF.shape)
    df = trainDF[["text","labels_gen","labels_leng"]] #.head(20)
    
    y_train_gen=df["labels_gen"]
    y_train_len=df["labels_leng"]
    
    ######*** Hacemos el encoder de los paises ***#######
    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train_gen)
    
    encoderL = preprocessing.LabelEncoder()
    y_trainL = encoderL.fit_transform(y_train_len)
    
    print("Estas son las etiquetas: ",np.unique(y_train),"\nY estos el significado: ", encoder.inverse_transform(np.unique(y_train)))
    print("Estas son las etiquetas: ",np.unique(y_trainL),"\nY estos el significado: ", encoderL.inverse_transform(np.unique(y_trainL)))
    df["labels_gen"]=y_train
    df["labels_leng"]=y_trainL
    print("df shape ", df.shape)
    print(df.head())
    
    #importamos ELMO
    from elmoformanylangs import Embedder
    
    e = Embedder(pathWeights)
    
    ELMOvalues = []
    
    start_timeTF = time.time()
    print("\nComenzamos el cálculo")
    
    for i in range(df.shape[0]):
      leido = df["text"].iloc[i]
      output = e.sents2elmo([leido])
      ELMOvalues.append(output)
      if i%10 == 1:
        print("\nStep:",i,"  Avance:", "{:.2%}".format(i/df.shape[0]))
        print("--- %s seconds ---" % (time.time() - start_timeTF), "\n")
    print("Terminamos de calcular \n","--- %s seconds ---" % (time.time() - start_timeTF))
    
    final = list(chain.from_iterable(ELMOvalues))
    train_embeddings= []
    
    for k in final:
      valores = k.mean(axis=0)
      train_embeddings.append(valores)
    
    dfprov = pd.DataFrame(train_embeddings, index = df.index)
    
    columnasDF = [ str(x) for x in dfprov.columns.to_list()]
    dfprov.columns = columnasDF
    dfFIn = pd.concat([df, dfprov],axis=1, sort=False)
    
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
    
    # Género
    baseGen = dfFIn[["text","labels_gen"]+columnasDF]
    baseGen.columns = ["text","label"]+columnasDF
    
    Evaluacion(classifiers,input_pathImp,"labels_gen",nombre, "ELMO",baseGen, 0, columnasDF)
    
    # Lenguaje
    baseLen = dfFIn[["text","labels_leng"]+columnasDF]
    baseLen.columns = ["text","label"]+columnasDF
    #baseLen.head()
    Evaluacion(classifiers,input_pathImp,"labels_leng",nombre, "ELMO",baseLen, 1, columnasDF)