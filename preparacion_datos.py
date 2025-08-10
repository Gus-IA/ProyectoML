import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

"""
Script de preparación de datos, entrenamiento y evaluación de modelos.

Este script:
- Imputa valores nulos y escala los datos.
- Codifica variables categóricas.
- Entrena modelos de regresión (lineal, árbol de decisión, random forest).
- Evalúa modelos con validación cruzada.
- Realiza ajuste de hiperparámetros con `GridSearchCV`.
- Guarda el modelo y pipeline final usando `joblib`.

Utiliza los datos de entrenamiento y prueba generados por `obtencion_datos.py`.
"""

# cargamos los datos de entrenamiento
data = pd.read_csv("housing_train.csv")

# separamos la variable que queremos para predecir del resto de datos
data, labels = data.drop(['median_house_value'], axis=1), data['median_house_value'].copy()

# mostramos los datos que están a null
print(data[data.isnull().any(axis=1)])

# hay varias opciones según el caso
# eliminar todos los null de la columna total_bedrooms
#data.dropna(subset=["total_bedrooms"]) 

# eliminar la columna
#data.drop("total_bedrooms", axis=1)

# hacer la media para rellenar
data.fillna(data["total_bedrooms"].median())

print(data.head())


# separamos los datos numéricos de los categóricos
data_num = data.drop(['ocean_proximity'], axis=1)
print(data_num.head())


data_cat = data[['ocean_proximity']]
print(data_cat.head())



# una forma con sklearn para rellenar los datos que están a null
imputer = SimpleImputer(strategy="median")
imputer.fit(data_num)
imputer.statistics_

# mostramos estos datos
print(data_num.median().values)

# reemplazamos los datos null y lo guardamos en una variable
X = imputer.transform(data_num)
X

# convertimos las columnas a valores numéricos 
ordinal_encoder = OrdinalEncoder()
data_cat_encoded = ordinal_encoder.fit_transform(data_cat)
print(data_cat_encoded[:10])

print(ordinal_encoder.categories_)


# en lugar de transformar todo a valores numéricos diferentes
# transformamos en un vector y todos los valores están a 0 menos la clase que está a 1
cat_encoder = OneHotEncoder()
data_cat_1hot = cat_encoder.fit_transform(data_cat)
print(data_cat_1hot)

# lo pasamos a array
print(data_cat_1hot.toarray())



# normalizamos los datos que estén todos a la misma escala en un pipeline para encadenar los pasos
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        # ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

data_num_tr = num_pipeline.fit_transform(data_num)


# separamos columnas numéricas y columnas categóricas
# aplicamos el pipeline para escalar
num_attribs = list(data_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

data_prepared = full_pipeline.fit_transform(data)


# ---- SELECCIÓN MODELO ----

# entrenamos el modelo con LinearRegressión
lin_reg = LinearRegression()
lin_reg.fit(data_prepared, labels)

# printamos los valores de las 5 primeras líneas de las predicciones
some_data = data.iloc[:5]
some_labels = labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))


# printamos también las categorías
print("Labels:", list(some_labels))


# entrenamos el modelo con el error métrico cuadrado
predictions = lin_reg.predict(data_prepared)
lin_mse = mean_squared_error(labels, predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


# entrenamos el modelo con un árbol de decisión
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(data_prepared, labels)

# evaluamos el modelo de árbol de decisión
predictions = tree_reg.predict(data_prepared)
tree_mse = mean_squared_error(labels, predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


# ---- VALIDACIÓN CRUZADA ----

# divide los datos en 10 partes y se entrenarán y validarán
scores = cross_val_score(tree_reg, data_prepared, labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

# mostramos los datos de la validación cruzada
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# entrenamos el modelo con Random Forest Regressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(data_prepared, labels)
predictions = forest_reg.predict(data_prepared)
forest_mse = mean_squared_error(labels, predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# mostramos los datos de la validación cruzada del Random Forest Regressor
forest_scores = cross_val_score(forest_reg, data_prepared, labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)



# ---- FINETUNING ----

# hacemos una búsqueda de hiperparámetros con GridSearchCV para encontrar la mejor combinación de parámetros en un RandomForestRegressor
param_grid = [
    # probamos 12 combinaciones 3 del primero y 4 del segundo. 
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # probamos 6 combinaciones 2 del primero por 3 del segundo. Boostrap a false para que se entrene con todo el conjunto
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

# creamos el modelo Rndom Forest Regressor
forest_reg = RandomForestRegressor(random_state=42)
# entrenamos con una validación cruzada de 5 con el error métrico cuadrado 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(data_prepared, labels)


# mostramos los mejores parámetros
print(grid_search.best_params_)


# recuperamos la importancia de las características
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

# y las mostramos por importancia en la influencia de cada una de ellas en la predición
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))


# calculamos con los datos de test nuestro mejor modelo, calcular las predicciones y calcular las métricas
test_data = pd.read_csv('housing_test.csv')

final_model = grid_search.best_estimator_

X_test = test_data.drop("median_house_value", axis=1)
y_test = test_data["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

# Crear la carpeta housing-app si no existe
dest_folder = f'housing-app'
os.makedirs(dest_folder, exist_ok=True)

# guardamos el modelo
joblib.dump(final_model, "housing-app/my_model.pkl")
joblib.dump(full_pipeline, "housing-app/my_pipeline.pkl")