# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:09:32 2019

@author: daniel.lopez
"""
# Cargar librarías necesarias
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import scipy.stats as stt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
from sklearn.preprocessing import MinMaxScaler

#Definir directorio
os.chdir('C:/Users/daniel.lopez/Desktop/IE')

#Cargar datos 
df_org = pd.read_csv('baobab_prueba.csv', sep = ';')

#Convert target variable from good/bad to 1/0
df_org['Type'] = df_org['Type'].replace('good',0)
df_org['Type'] = df_org['Type'].replace('bad',1)

# Transform other variables 
df_org['Telephone'] = df_org['Telephone'].replace('none',0)
df_org['Telephone'] = df_org['Telephone'].replace('yes',1)

df_org['foreign worker'] = df_org['foreign worker'].replace('no',0)
df_org['foreign worker'] = df_org['foreign worker'].replace('yes',1)


#Transformar variables en dummies 
dummies = df_org.select_dtypes(include='object')

dummies_df = pd.get_dummies(dummies)

# Delete these columns from df
df_org = df_org.select_dtypes(exclude=['object'])

# Reescalamos las variables entre 0-1 para que sean comparativas
scaler = MinMaxScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df_org),columns=df_org.columns)

#Merge dummies df with original df 

df = pd.concat([df_scaled,dummies_df], axis=1)

#Ver cuántos good y bad hay de cada clase
df[df['Type'] == 1].count()
df[df['Type'] == 0].count()
"""
Encontramos un data set poco equlibrado en el que tenemmos 
700 casos de clientes 'good' y 300 clientes 'bad'
"""
# Visualizar
sns.countplot('Type', data=df)
plt.title('Distribución por clases')

# Definición de la correlación entre las variables 
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r',xticklabels=True,yticklabels=True)
plt.title("Matriz de Correlation", fontsize=14)

# Aplicar algoritmos de clusterización para ver patrones
X = df.drop('Type', axis=1)
y = df['Type']

# Algoritmo TSNE
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

# leyenda

azul = mpatches.Patch(color = '#0A0AFF', label = 'Good')
rojo =  mpatches.Patch(color = '#AF0000', label = 'Bad')                     
                      
# t-SNE scatter plot
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Good', linewidths=2)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Bad', linewidths=2)
plt.legend(handles = [azul,rojo])
plt.title('t-SNE', fontsize=14)
"""
No se ve ningún patrón interesante en la gráfica
"""

# Otro método
X_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)

plt.scatter(X_pca[:,0], X_pca[:,1], c=(y == 0), cmap='coolwarm', label='Good', linewidths=2)
plt.scatter(X_pca[:,0], X_pca[:,1], c=(y == 1), cmap='coolwarm', label='Bad', linewidths=2)
plt.legend(handles = [azul,rojo])
plt.title('PCA', fontsize=14)
""" 
No parece una clara relación tampoco
"""
# TruncatedSVD
X_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)

plt.scatter(X_svd[:,0], X_svd[:,1], c=(y == 0), cmap='coolwarm', label='Good', linewidths=2)
plt.scatter(X_svd[:,0], X_svd[:,1], c=(y == 1), cmap='coolwarm', label='Bad', linewidths=2)
plt.legend(handles = [azul,rojo])
plt.title('SVD', fontsize=14)
"""
Tampoco vemos una clara relación
"""

###############################################################################
# Modelos
###############################################################################

# Separar en test y train
from sklearn.model_selection import train_test_split

# Hagamos el modelo 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
"""
Hemos probado con 20 variables a elegir pero no sabemos si es el número 
óptimo por ello lo hacemos de manera iterativa
"""
n_vars = np.arange(1,len(df.columns))  
resultado = 0
nof=0           
lista_resultados =[]

for n in range(len(n_vars)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    model = LogisticRegression()
    rfe = RFE(model,n_vars[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    var = model.score(X_test_rfe,y_test)
    lista_resultados.append(var)
    if(var > resultado):
        resultado = var
        nof = n_vars[n]

print('El número optimo de variables es: %d' %nof)

print("Resultado con %d varibales: %f" % (nof, resultado))

# Buscamos variables a aplicar
cols = list(X.columns)
model = LogisticRegression()

#Iniciamos el RFE (recursive feature elimination)
rfe = RFE(model, 24)             
#Entremos el modelo y vemos las variables a elegir
rfe.fit(X_train,y_train)              

variables_usar = list(X_train.columns[rfe.support_])

# Generamos una nueva matriz de correlaciones con las columnas elegidas

sns.set(style="white")

# Matriz de correlacion
corr = X_train[variables_usar].corr()

sns.heatmap(corr, cmap='coolwarm_r',xticklabels=True,yticklabels=True,annot=True)
plt.title("Matriz de Correlation", fontsize=14)
"""
Existen algunas correlaciones pero no demasiado altas
"""
# Entranamos el modelo para predecir

model.fit(X_train[variables_usar],y_train)

# Predicciones 

y_pred = model.predict_proba(X_test[variables_usar])[:,1]

"""
Teniendo en cuenta que contamos con un test bastante desquilibrado y que si,
de manera aleatoria decidiesemos clasificar a los clientes obtendríamos una 
precisión del 70% (700 buenos 300 malos)definiremos el threshold como 0.7
"""

y_predict = np.where(y_pred > 0.7,1,0)

# Matriz de confusiones 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_recall_curve,average_precision_score

matriz = confusion_matrix(y_test,y_predict)
print(classification_report(y_test, y_predict))

#create a heat map
sns.heatmap(pd.DataFrame(matriz), annot = True, cmap = 'Blues', fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()

###############################################################################
# Probamos otros modelos 
###############################################################################

# Los tranformamos en format array para meterlos en los algoritmos 
X_train_mat = X_train[variables_usar].values
X_test_mat = X_test[variables_usar].values
y_train_mat = y_train.values
y_test_mat = y_test.values

# Definimos los algoritmos a utlizar
algoritmos = {
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

# Definimos la función para sacar la precisión

for key, classifier in algoritmos.items():
    classifier.fit(X_train_mat, y_train_mat)
    training_score = cross_val_score(classifier, X_train_mat, y_train_mat, cv=5)
    print("Algoritmo: ", classifier.__class__.__name__, "Precisión", round(training_score.mean(), 2) * 100)

