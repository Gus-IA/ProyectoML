import requests
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

"""
Script para la descarga y preparación inicial del dataset de precios de viviendas en California.

Este script:
- Descarga automáticamente un archivo `.tgz` desde una URL remota.
- Extrae el contenido y carga los datos en un DataFrame.
- Realiza una primera división en conjuntos de entrenamiento y prueba.
- Aplica estratificación por categorías de ingreso.
- Guarda los datos preparados en archivos CSV.
"""

URL = "https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/housing.tgz"
PATH = "housing.tgz"

# obtenemos los datos
def getData(url=URL, path=PATH):
  r = requests.get(url)
  with open(path, 'wb') as f:
    f.write(r.content)  
  housing_tgz = tarfile.open(path)
  housing_tgz.extractall()
  housing_tgz.close()

PATH = "housing.csv"

# cargamos los datos en un path
def loadData(path=PATH):
  return pd.read_csv(path)

data = loadData()

# usamos train_test_split para separar del conjunto de datos, un 20% aleatorios para entrenar el modelo
train, test = train_test_split(data, test_size=0.2, random_state=42)

# mostramos el gráfico con la mediana de train y test para verificar que las distribuciones están correctas
train['median_income'].hist(bins=50)
test['median_income'].hist(bins=50)

plt.show()

# creamos una nueva columna income_cat y la dividimos con pd.cut para que cada una de las filas tenga una categoría entre 1 y 5
data['income_cat'] = pd.cut(data['median_income'],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
data['income_cat'].hist()

# con stratify distribuimos en train y test de forma equitativa
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['income_cat'])

# mostramos el gráfico con la mediana de train y test para verificar que las distribuciones están correctas
train['median_income'].hist(bins=50)
test['median_income'].hist(bins=50)

plt.show()


# eliminamos la columna creada para el stratify
for set_ in (train, test):
    set_.drop("income_cat", axis=1, inplace=True)

# guardamos los datos de train y test en un csv
train.to_csv('housing_train.csv', index=False)
test.to_csv('housing_test.csv', index=False)