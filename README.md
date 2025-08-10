# 🏡 Housing Price Prediction Project

Este proyecto implementa un flujo completo de ciencia de datos para predecir el valor medio de viviendas en California. Abarca la adquisición de datos, análisis exploratorio, ingeniería de características, preparación de datos, entrenamiento de modelos y evaluación final.

---

## 📌 Objetivos Aprendidos

- Descarga y extracción de datasets automáticamente con `requests` y `tarfile`.
- Separación de datos de entrenamiento y prueba utilizando `train_test_split` con y sin `stratify`.
- Análisis exploratorio de datos (EDA), incluyendo:
  - Histogramas de distribución.
  - Gráficos geográficos con variables como población y valor de viviendas.
  - Matriz de correlaciones.
- Ingeniería de características:
  - Creación de nuevas columnas derivadas de los datos originales para mejorar el rendimiento del modelo.
- Preparación de datos con `Pipeline` y `ColumnTransformer`:
  - Imputación de valores nulos.
  - Normalización y codificación de variables categóricas (`OneHotEncoder`).
- Entrenamiento de modelos:
  - Regresión lineal.
  - Árboles de decisión.
  - Random Forest con búsqueda de hiperparámetros (`GridSearchCV`).
- Validación cruzada (`cross_val_score`) y evaluación final del modelo.
- Serialización de modelos con `joblib`.

---

pip install -r requirements.txt y ejecutar el main.py

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
