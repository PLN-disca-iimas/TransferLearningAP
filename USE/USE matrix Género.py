#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 19:40:05 2020

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
from sklearn.model_selection import train_test_split


from itertools import chain
from datetime import datetime


##############################################################################
#Declaramos las carpetas en donde están las cosas



#nombre = "es_clean" #str(sys.argv[1])
input_path= "/001/usuarios/palacios.alc/archivos/basesResumidas/df/"
input_pathImp = "/001/usuarios/palacios.alc/archivos/resultadosUSEMatrix/"
#input_path= "/home/josue/Documentos/Bases/Wiki corpus es/df/"
#input_pathImp = "Documentos/ResultadosSS/PruebaMatrix/"


###################################################
##########****** MAIN PARA ENTRENAR ALGORITMOS  *********##########
#-----------------------------------------------------------------------------

def Evaluacion(classifiers,input_pathImp,nombre,dfFinx, lista,columnasDFx):
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    #declaramos el path para guardar el archivo junto con el nombre del file
    filepath = input_pathImp+"USE Matrix Género "+nombre+" "+dt_string+".csv"
    
    scores_rskf = []
    NombreTrain, NombreTest, NombreModelo = [],[],[]
    f1,precision,accuracy,recall = [],[],[],[]
    scores_rskfx = []
    NombreTrainx, NombreTestx, NombreModelox = [],[],[]
    f1x,precisionx,accuracyx,recallx = [],[],[],[]
    
    for i in lista:
        for k in lista:
            if i==k:
                #iguales
                dfUsar = dfFinx[dfFinx["labels_leng"]==k]
                X_train, X_test, y_train, y_test = train_test_split(dfUsar[columnasDFx],
                                        dfUsar["labelNum"], test_size=0.2, random_state=42)
                NomTrain= k
                NomTest = k
                
            else:
                #diferentes
                X_train = dfFinx.loc[dfFinx["labels_leng"]==i, columnasDFx]
                X_test = dfFinx.loc[dfFinx["labels_leng"]==k, columnasDFx]
                y_train = dfFinx.loc[dfFinx["labels_leng"]==i, "labelNum"]
                y_test = dfFinx.loc[dfFinx["labels_leng"]==k, "labelNum"]
                NomTrain= i
                NomTest = k
    
            print('*** Entrenando modelo I ***',i,k)
            classifiers[0].fit(X_train,y_train)
            print('*** Predecir modelo I ***',i,k)
            y_pred = classifiers[0].predict(X_test)
    
            print('*** Entrenando modelo II ***',i,k)
            classifiers[1].fit(X_train,y_train)
            print('*** Predecir modelo II ***',i,k)
            y_predx = classifiers[1].predict(X_test)
            
            n_correct = sum(y_pred == y_test)
            n_correctx = sum(y_predx == y_test)
            #calculamos métricas
            scores_rskf.append(n_correct/len(y_pred))
            f1.append(metrics.f1_score(y_pred, y_test, average='weighted', labels=np.unique(y_test)))
            precision.append(metrics.precision_score(y_pred, y_test,average='binary'))
            accuracy.append(metrics.accuracy_score(y_pred, y_test))
            recall.append(metrics.recall_score(y_pred, y_test,average='binary'))
            NombreTrain.append(i)
            NombreTest.append(k)
            NombreModelo.append("linear model") 
    
            scores_rskfx.append(n_correctx/len(y_predx))
            f1x.append(metrics.f1_score(y_predx, y_test, average='weighted', labels=np.unique(y_test)))
            precisionx.append(metrics.precision_score(y_predx, y_test,average='binary'))
            accuracyx.append(metrics.accuracy_score(y_predx, y_test))
            recallx.append(metrics.recall_score(y_predx, y_test,average='binary'))
            NombreTrainx.append(i)
            NombreTestx.append(k)
            NombreModelox.append("SVM") 
        #fin segundo for
    #fin primer for
    dfModel1 = pd.DataFrame({"Train":NombreTrain, "Test":NombreTest,
                             "Modelo":NombreModelo,"Score":scores_rskf,
                             "F1":f1, "Precision":precision,
                             "Accuracy":accuracy,"Recall":recall})
    dfModel2 = pd.DataFrame({"Train":NombreTrainx, "Test":NombreTestx,
                             "Modelo":NombreModelox,"Score":scores_rskfx,
                             "F1":f1x, "Precision":precisionx,
                             "Accuracy":accuracyx,"Recall":recallx})
    dfModelos = pd.concat([dfModel1,dfModel2], sort=False)
    dfModelos["Preprocesamiento"] = nombre
    dfModelos.to_csv(filepath, index=False, encoding="utf-8-sig")
    print("Terminamos.")
### Fin de la función
#-----------------------------------------------------------------------------


##############################################################################

if __name__ == "__main__":
    
    listaFiles = ["es_clean","sinEmojis","sinEmoticones","sinHastag",
                  "sinMenciones","sinNada","sinSlangs","sinURLs"]
    
    for h in listaFiles:
        nombre = str(h)
        ######*** Crea el dataframe y guarda el texto y las etiquetas ***######
        trainDF = pd.read_csv(input_path+nombre+".csv", encoding="utf-8-sig")
        ######***  Set de entrenamiento y etiquetas de entrenamiento  ***######   
        columnaLabel = "labels_gen"
    
        df = trainDF[["text","labels_leng",  columnaLabel]] #.head(5000)
        df.columns = ["text","labels_leng","label"]
        y_train_gen=df['label']

        ######*** Hacemos el encoder de los paises ***#######
        encoder = preprocessing.LabelEncoder()
        y_train = encoder.fit_transform(y_train_gen)
        print("Estas son las etiquetas: ",np.unique(y_train),"\nY estos el significado: ", encoder.inverse_transform(np.unique(y_train)))
        df["labelNum"]=y_train
        df["text"].fillna(" ", inplace=True)
    
        listaLen = df["labels_leng"].unique().tolist()

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
            if i % 10 == 0:
                print(i/(len(iterable)-1))
                print("--- %s seconds ---" % (time.time() - start_timeTF))
                #print("calculamos ",[iterable[i],iterable[i+1]])
        print("Terminamos de calcular \n","--- %s seconds ---" % (time.time() - start_timeTF))

        # el output lo convertimos en un formato que sí podemos usar
        final = list(chain.from_iterable(guardado))
        train_embeddings=np.array(final) #np.asarray(train_embeddings)

        print("SHAPE DE X_train_emb", train_embeddings.shape)    

        columnasDF = list(range(train_embeddings.shape[1]))
        dfFin = pd.concat([df,
                    pd.DataFrame(train_embeddings,
                    columns = columnasDF)], 
                    axis=1, sort=False)
    
    
        ######*** Modelo de entrenamiento ***######
        #hacemos la evaluación
        Evaluacion(classifiers,input_pathImp,nombre,dfFin, listaLen,columnasDF)