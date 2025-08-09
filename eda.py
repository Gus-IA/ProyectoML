import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PATH = "housing_train.csv"

# cargamos los datos en un path
def loadData(path=PATH):
  return pd.read_csv(path)

# cargamos los datos de entrenamiento
data = loadData('housing_train.csv')

# muestra información sobre el dataset como el número de columnas o el peso total
data.info()

# muestra las instancias de cada clase
print(data['ocean_proximity'].value_counts())

# muestra diferentes estadísticas para las columnas numéricas
print(data.describe())

# mostramos esas estadísticas en gráficos
data.hist(bins=50, figsize=(20,15))
plt.show()


# mostramos el gráfico en uno de puntos 

data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=data["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.show()

# calculamos la correlación menos con la columna no numérica 
# y la ordenamos por el valor medio 
corr_matrix = data.drop(columns=['ocean_proximity']).corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))