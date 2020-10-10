#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:50:37 2020

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


type_label = int(sys.argv[1]) #0 = Género  / 1 = Lenguaje
nombre = str(sys.argv[2])
input_path= "/001/usuarios/palacios.alc/archivos/bases/"
input_pathImp = "/001/usuarios/palacios.alc/archivos/resultadosUSEavg/"
#input_path= "/home/josue/Documentos/Bases/Wiki corpus es/df Extendido/"
#input_pathImp = "Documentos/ResultadosSS/USEExtendidoPromedio/"


###################################################
##########****** MAIN PARA ENTRENAR ALGORITMOS  *********##########
#-----------------------------------------------------------------------------

def Evaluacion(classifiers,input_pathImp,columnaLabel,nombre, nameExt,df, type_label,columnasTexto):
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    dt_stringPuntos = now.strftime("%d-%m-%Y %H:%M:%S")
    #declaramos el path para guardar el archivo junto con el nombre del file
    filepath = input_pathImp+nameExt+" "+columnaLabel+" "+nombre+"Promedio "+dt_string+".txt"
      
    #comenzamos a escribir el archibo
    with open(filepath, "w") as output:
        
        #comenzamos a contar el tiempo
        start_time = time.time()
        #imprimimos datos relevantes para el txt
        output.write('Fecha de creación del archivo: '+str(dt_stringPuntos))
        output.write('\nCarpeta usada: '+str(nombre))
        output.write('\nPreprocesamiento de la info: '+str(nameExt))
        output.write('\nProcesamiento de los tweets de cada persona: '+str("Promedio"))
        output.write('\nNúmero de usuarios analizados: '+str(len(df["ID"].unique())))
        #
        output.write('\n ---------------------------')
        warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
    
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
        
        
        for train_index, test_index in rskf.split(df["ID"],df["label"]):
            
            i+=1
            #hacemos el split entre train y test
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
            n_correctx = sum(y_predx == y_test_folds)########**** Métricas de evaluación ****#########
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


##############################################################################

if __name__ == "__main__":
    ######*** Crea el dataframe y guarda el texto y las etiquetas ***######
    trainDF = pd.read_csv(input_path+"Extendido "+nombre+".csv", encoding="utf-8-sig")
    ######***  Set de entrenamiento y etiquetas de entrenamiento  ***######   

    if type_label == 0:
        columnaLabel = "labels_gen"
    else:
        columnaLabel = "labels_leng"

    df = trainDF[["text", "ID", columnaLabel]] #.head(5000)
    df.columns = ["text", "ID", "label"]
    y_train_gen=df['label']

    ######*** Hacemos el encoder de los paises ***#######
    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train_gen)
    print("Estas son las etiquetas: ",np.unique(y_train),"\nY estos el significado: ", encoder.inverse_transform(np.unique(y_train)))
    df["label"]=y_train
    df["text"].fillna(" ", inplace=True)

    ##############################################################################


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
    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
    #### tensors
    textd = df["text"].values.tolist()
    print("TIPO DE texto",type(textd), len(textd))

    #cargamos la gráfica que usaremos
    print("Cargando modelo USE") 
    g = tf.Graph()
    print("Antes de entrar")
    with g.as_default():
        print("Entró")
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        print("Después de placeholder")
        os.environ['TFHUB_CACHE_DIR'] = '/001/usuarios/palacios.alc/archivos/cache/'
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
        print("Después de module")
        embedded_text = embed(text_input)
        print("Después de embed")
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        print("después de tf.group")
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
        if i % 100 == 0:
            print(i/(len(iterable)-1))
            print("--- %s seconds ---" % (time.time() - start_timeTF))
            #print("calculamos ",[iterable[i],iterable[i+1]])
    print("Terminamos de calcular \n","--- %s seconds ---" % (time.time() - start_timeTF))

    # el output lo convertimos en un formato que sí podemos usar
    final = list(chain.from_iterable(guardado))
    train_embeddings=np.array(final) #np.asarray(train_embeddings)

    print("SHAPE DE X_train_emb", train_embeddings.shape)    

    columnasDF = list(range(train_embeddings.shape[1]))
    dfFIn = pd.concat([df,
                    pd.DataFrame(train_embeddings,
                    columns = columnasDF)], 
                    axis=1, sort=False)
    
    dfPromedio = dfFIn[["ID",
                        "label"]+columnasDF].groupby("ID").mean().reset_index()

    ######*** Modelo de entrenamiento ***######
    #hacemos la evaluación
    Evaluacion(classifiers,input_pathImp,columnaLabel,nombre, "USE",dfPromedio, type_label, columnasDF)





