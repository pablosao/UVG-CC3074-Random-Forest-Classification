# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:08:02 2020

@author: pablo sao
"""
# Random Forest Classifier

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def label_race(df):
    
    for i in range(len(nombres)):
        df.loc[df['class']==nombres[i],'codigo_clase'] = codigoClass.get(nombres[i])


# Importing the datasets
datasets = pd.read_csv('iris_clean.csv')

nombres = datasets['class'].unique().tolist()
codigoClass = {}

for i in range(len(nombres)):
    codigoClass[nombres[i]] = i + 1


label_race(datasets)

#print(datasets.to_string())

X = datasets.iloc[:, [1,4]].values
Y = datasets.iloc[:, 6].values

## Agregar columna con codigo para el tipo de Iris-setosa y virginicola, versicolor


# Convirtiendo a float los datos numericos
datasets['sepal_length_cm'] = datasets['sepal_length_cm'].astype(float)
datasets['sepal_width_cm'] = datasets['sepal_width_cm'].astype(float)
datasets['petal_length_cm'] = datasets['petal_length_cm'].astype(float)
datasets['petal_width_cm'] = datasets['petal_width_cm'].astype(float)
datasets['codigo_clase'] = datasets['codigo_clase'].astype(int)


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)




# Fitting the classifier into the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_Train,Y_Train)

# Predicting the test set results

Y_Pred = classifier.predict(X_Test)

# Making the Confusion Matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Visualising the Training set results

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    name_label = '{} - {}'.format(int(j), nombres[int(j) - 1])
    
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = name_label)
plt.title('Random Forest Classifier (Training set)')
plt.xlabel('Petalos y Sepalos')
plt.ylabel('Clasificacion')
plt.legend()
plt.show()

# Visualising the Test set results

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Test, Y_Test
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    name_label = '{} - {}'.format(int(j), nombres[int(j) - 1])
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = name_label)
plt.title('Random Forest Classifier (Test set)')
plt.xlabel('Petalos y Sepalos')
plt.ylabel('Clasificacion')
plt.legend()
plt.show()

