#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:15:36 2020

@author: root
"""
#---versiones
# python v 3.6.9
# tensorflow v 1.13.1
# tensorflow-hub v 0.7.0
# tf-sentencepiece v. 0.1.85


import os, codecs, sys
import time, re
import string,sklearn
import numpy as np
import pandas as pd
import string, warnings
import time

import tensorflow as tf
import tensorboard
import tensorflow_hub as hub
import matplotlib.pyplot as plt

import tf_sentencepiece  # Not used directly but needed to import TF ops.


from sklearn import model_selection, linear_model, metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import preprocessing, svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve
from sklearn.base import clone

from itertools import chain
from datetime import datetime


##############################################################################
#Declaramos las carpetas en donde están las cosas

#carpeta donde lee los archivos
input_path='/home/josue/Documentos/Bases/sinURLs/'
#carpeta donde está el archivo truth.txt
input_pathT='/home/josue/Documentos/Bases/'
#carpeta donde guardamos el txt output de resultados
input_pathImprimir = "Documentos/ResultadosSS/USE"
#Nombre del archivo output de resultados
nombreFile = "USE Género sinURLs "

###################################################

def plot_roc_curve(fpr, tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1], [1,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
########****** Método para entrenar los modelos ******##########
def Modelo(classifiers, X_train, y_train_label, carpeta, prepro, numUsuarios,fecha):
    #comenzamos a contar el tiempo
    start_time2 = time.time()
    #imprimimos datos relevantes para el txt
    output.write('Fecha de creación del archivo: '+str(fecha))
    output.write('\nCarpeta usada: '+str(carpeta))
    output.write('\nPreprocesamiento de la info: '+str(prepro))
    output.write('\nNúmero de archivos analizados: '+str(numUsuarios))
    
    #For por cada uno de los algoritmos
    for classifier in classifiers:
        output.write('\n ---------------------------')
        output.write('\n *** '+str(classifier)+'\n ---------------------- ')

        start_time = time.time()
        warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

        ####*** guardan los resulados de cada una de las 50 iteraciones de RepeatedStratifiedKFold****######
        scores_rskf = []
        f1,precision,accuracy,recall = [],[],[],[]

        ######*** se Elijió RepeatedStratifiedKFold porque nos da una muestra representativa de c/u de las clases
        #### y así evitar sesgos
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,random_state=36851234)
        i=0
        for train_index, test_index in rskf.split(X_train, y_train_label):
            
            #hacemos el split entre train y test
            X_train_foldsTxt, X_test_foldsTxt = X_train[train_index], X_train[test_index]
            y_train_folds, y_test_folds = y_train_label[train_index], y_train_label[test_index]
            
            #Entrenamos y predecimos
            i+=1
            print('*** Entrenando modelo ***',i,'\n')
            classifier.fit(X_train_foldsTxt,y_train_folds)
            y_pred = classifier.predict(X_test_foldsTxt)
            n_correct = sum(y_pred == y_test_folds)

            ########**** Métricas de evaluación ****#########
            scores_rskf.append(n_correct/len(y_pred))
            f1.append(metrics.f1_score(y_pred, y_test_folds, average='weighted', labels=np.unique(y_test_folds)))
            precision.append(metrics.precision_score(y_pred, y_test_folds,average='binary'))
            accuracy.append(metrics.accuracy_score(y_pred, y_test_folds))
            recall.append(metrics.recall_score(y_pred, y_test_folds,average='binary'))
            
            print("*** Termino el entrenamiento del modelo ***", i)
            print("--- %s seconds ---" % (time.time() - start_time2))
            
            

        ##############*** mean y std de los scores obtenidos de  RepeatedStratifiedKFold***###################
        mean_score = np.mean(scores_rskf)
        standar_score =np.std(scores_rskf)
        
        output.write('\n ---------Validación RepeatedStratifiedKFold------------------')
        output.write('\n  Scores:' + str(scores_rskf))
        output.write('\n  Mean:' + str(mean_score))
        output.write('\n  Standard deviation:' + str(standar_score))
        output.write('\n  F1:  '+ str(np.mean(f1)))
        output.write('\n  precision:  ' + str(np.mean(precision)))
        output.write('\n  recall:  ' + str(np.mean(recall)))
        output.write('\n  exactitud (accuracy):  ' + str(np.mean(accuracy)))
        output.write('\n  elapsed time:' + str(time.time() - start_time) )
        #output.write('\n  #características:' + str(car))
        print("*** Termino el split ***")
        print("--- %s seconds --- \n" % (time.time() - start_time2))


##############################################################################

##########****** MAIN PARA ENTRENAR ALGORITMOS  *********##########

#leemos todas las  lineas del arhivo truth
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

#######*** Guarda en una lista los txtfile, las etiquetas ***######
#leemos los archivos de la carpeta que tiene los files
#si el titulo del archivo está en truth guardamos su contenido
# junto con el sexo y variante del lenguaje      
for item in os.listdir(input_path):
    if os.path.splitext(item)[1] == ".txt" and os.path.splitext(item)[0][:-1] in truth.keys():   
        with open(input_path+"/"+item) as fileLeer:
            txtfiles.append(fileLeer.read())
            labels_gen.append(truth[os.path.splitext(item)[0][:-1]][0])
            labels_leng.append(truth[os.path.splitext(item)[0][:-1]][1])


#######*** Crea el dataframe y guarda el texto y las etiquetas ***######
trainDF = pd.DataFrame()
trainDF['text'] = txtfiles
trainDF['labels_gen'] = labels_gen
trainDF['labels_leng']=labels_leng

######*** Elegimos con qué variables vamos a correr  ***######  
X_train=trainDF['text']
y_train=trainDF['labels_gen']

#######*** label encode the target variable ***######
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)

#print("Estas son las etiquetas: ",np.unique(y_train), "\nEste es el encoder: ", encoder.inverse_transform(np.unique(y_train)))

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
# Reduce logging #output.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
#### tensors
textd = X_train.tolist()
print("TIPO DE textd",type(textd), len(textd))

#cargamos la gráfica que usaremos
print("Cargando modelo USE") 
g = tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
    embedded_text = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

print("Calculando embeddings usuarios") 
# Initialize session.
session = tf.Session(graph=g)
print("Cargó la grafica")
session.run(init_op)
print("paso session.run")

# creamos particiones de 10 en 10 para procesar nuestro texto
longitud =len(textd)
iterable =list(range(0,longitud-1,10))
iterable.append(longitud)

#comenzamos a procesar el texto
start_timeTF = time.time()
print("Comenzamos el cálculo")
guardado = []
for i in range(len(iterable)-1):
    en_result = session.run(embedded_text, feed_dict={text_input: textd[iterable[i]:iterable[i+1]]})
    guardado.append(en_result)
    print("--- %s seconds ---" % (time.time() - start_timeTF))
    print("calculamos ",[iterable[i],iterable[i+1]])
print("Terminamos de calcular \n","--- %s seconds ---" % (time.time() - start_timeTF))

# el output lo convertimos en un formato que sí podemos usar
final = list(chain.from_iterable(guardado))
train_embeddings=np.array(final) #np.asarray(train_embeddings)

print("SHAPE DE X_train_emb", train_embeddings.shape)    


######*** Modelo de entrenamiento ***######

#obtenemos la fecha actual
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H-%M-%S")


rutaFile = input_pathImprimir+"/"+nombreFile+" "+dt_string+".txt"
with open(rutaFile, "w") as output:
    Modelo(classifiers, train_embeddings, y_train, input_path,
           "Transfer Learning", train_embeddings.shape[0],dt_string)
print('**** Ver resultados en el archivo generado ***')





