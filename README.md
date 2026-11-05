# Clasificación de Dry Beans: comparación de modelos con y sin PCA

Trabajo de inferencia estadística (Universidad Nacional Guillermo Brown) cuyo objetivo es predecir la clase de poroto seco a partir de sus características morfológicas, usando el dataset **Dry Beans**.

## Qué hace
- Análisis exploratorio de la base (distribuciones, visualizaciones, PCA en 2D: PC1 vs PC2).
- Entrenamiento y evaluación de cuatro clasificadores: **Bayes ingenuo, LDA, QDA y SVM**.
- Comparación de cada modelo **con y sin PCA** (reducción de dimensionalidad).
- Métricas y **matrices de confusión** para cada algoritmo.
- Se trabaja con todas las filas de la base (sin eliminar duplicados).
- Revisión de datos faltantes (NA) y análisis del balanceo de las clases.

## Herramientas
Python, pandas, NumPy, matplotlib, scikit-learn. Desarrollado en Google Colab.

## Estructura
- `Examen_inferencia_estadistica.ipynb`: notebook principal, con comentarios en cada celda.
- `base de datos parcial.xlsx`: dataset (hoja `Dry_Beans_Dataset`).

## Autor
Lorenzo · Estudiante de Ciencia de Datos
