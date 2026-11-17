[README.md](https://github.com/user-attachments/files/34996592/README.md)
# Clasificación de porotos secos (Dry Beans): QDA, LDA, Naive Bayes y SVM, y el efecto de PCA

**Autor:** Lorenzo Ciprés

Proyecto de clasificación supervisada y reconocimiento de patrones. El objetivo es predecir el tipo de poroto seco (`Class`) a partir de 16 medidas de forma y tamaño, comparando cuatro clasificadores (QDA, LDA, Naive Bayes gaussiano y SVM) con y sin reducción de dimensionalidad mediante PCA.

## Resumen

Criterios de diseño del proyecto:

- Los registros duplicados se analizan, pero no se eliminan.
- Todo el preprocesamiento (estandarización y PCA) se ajusta únicamente con los datos de entrenamiento, usando `Pipeline`, para evitar fuga de información.
- Se usa siempre la misma partición entrenamiento/prueba, para que los modelos sean comparables.

Hallazgos principales:

1. **SVM es el mejor modelo**, y la ventaja excede el ruido de la partición.
2. **Los errores se concentran en clases parecidas** (DERMASON/SIRA y BARBUNYA/CALI), no en las clases chicas.
3. **PCA con el criterio del 95 % de varianza (4 componentes) empeoró a los cuatro clasificadores.**
4. **El problema no es PCA en sí, sino la cantidad de componentes**: con 6 componentes, la validación cruzada iguala o casi iguala a las versiones sin PCA.

## 1. Dataset

La base (hoja `Dry_Beans_Dataset`) contiene **13.611 observaciones** y **17 variables**: 16 variables numéricas de tamaño y forma y la variable objetivo `Class`, con 7 variedades de poroto. No hay valores nulos, infinitos ni etiquetas de clase vacías.

**Tabla 1. Variables del dataset (n = 13.611).**

| Variable | Tipo | Media | Desvío | Mínimo | Máximo |
|---|---|---|---|---|---|
| Area | Entero | 53.048,285 | 29.324,096 | 20.420,000 | 254.616,000 |
| Perimeter | Decimal | 855,283 | 214,290 | 524,736 | 1.985,370 |
| MajorAxisLength | Decimal | 320,142 | 85,694 | 183,601 | 738,860 |
| MinorAxisLength | Decimal | 202,271 | 44,970 | 122,513 | 460,198 |
| AspectRation | Decimal | 1,583 | 0,247 | 1,025 | 2,430 |
| Eccentricity | Decimal | 0,751 | 0,092 | 0,219 | 0,911 |
| ConvexArea | Entero | 53.768,200 | 29.774,916 | 20.684,000 | 263.261,000 |
| EquivDiameter | Decimal | 253,064 | 59,177 | 161,244 | 569,374 |
| Extent | Decimal | 0,750 | 0,049 | 0,555 | 0,866 |
| Solidity | Decimal | 0,987 | 0,005 | 0,919 | 0,995 |
| roundness | Decimal | 0,873 | 0,060 | 0,490 | 0,991 |
| Compactness | Decimal | 0,800 | 0,062 | 0,641 | 0,987 |
| ShapeFactor1 | Decimal | 0,007 | 0,001 | 0,003 | 0,010 |
| ShapeFactor2 | Decimal | 0,002 | 0,001 | 0,001 | 0,004 |
| ShapeFactor3 | Decimal | 0,644 | 0,099 | 0,410 | 0,975 |
| ShapeFactor4 | Decimal | 0,995 | 0,004 | 0,948 | 1,000 |
| Class | Cualitativo nominal | 7 categorías: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA | | | |

**Tabla 2. Distribución de la variable objetivo.**

| Clase | Registros | Porcentaje | Prueba (n) |
|---|---|---|---|
| DERMASON | 3.546 | 26,05 % | 1.064 |
| SIRA | 2.636 | 19,37 % | 791 |
| SEKER | 2.027 | 14,89 % | 608 |
| HOROZ | 1.928 | 14,17 % | 578 |
| CALI | 1.630 | 11,98 % | 489 |
| BARBUNYA | 1.322 | 9,71 % | 397 |
| BOMBAY | 522 | 3,84 % | 157 |

## 2. Análisis exploratorio (EDA)

**Duplicados.** Se identificaron 68 registros duplicados (0,50 % de la base), todos de la clase HOROZ (3,53 % de esa clase) y ubicados en un tramo del archivo (filas 5.505 a 7.342). No existe ninguna combinación de medidas idénticas asociada a clases distintas. Se analizaron pero no se eliminaron; como consecuencia, 31 de los 4.084 registros de prueba (0,76 %) tienen una copia exacta en el conjunto de entrenamiento.

**Balance de clases.** DERMASON es la clase más grande (26 %) y BOMBAY la más chica (3,8 %), una relación de 6,8 a 1. El desbalance es moderado, pero la accuracy puede esconder malos resultados en las clases chicas, y los modelos que usan la frecuencia de cada clase (NB, LDA y QDA) tienden a favorecer a las más frecuentes. No se aplicó ninguna técnica de balanceo, porque la clase menor tiene más de 500 ejemplos y todos los modelos se comparan con la misma partición. Para controlar el efecto se usó una partición estratificada y se informaron métricas macro y el recall de cada clase.

**Distribución y correlación.** Las variables de tamaño separan bien a BOMBAY (granos mucho más grandes) y a DERMASON (los más chicos), y se solapan entre las otras clases. Hay 15 pares de variables con correlación mayor a 0,95 y varias variables claramente asimétricas (por ejemplo, `Area`, `Solidity` y `ShapeFactor4`). Esto justifica el uso de PCA y explica que la covarianza de QDA sea casi singular, por lo que se redujo su tolerancia (`tol = 1e-10`).

**Preprocesamiento.** Las variables se estandarizaron con `StandardScaler` y, cuando correspondió, se aplicó PCA. Ambos pasos se ajustaron solo con los datos de entrenamiento mediante `Pipeline`.

![Figura 1. Cantidad de registros y porcentaje de cada clase](figures/fig1_distribucion_clases.png)

*Figura 1. Cantidad de registros (izquierda) y porcentaje (derecha) de cada clase.*

## 3. Modelado

Se implementaron cuatro clasificadores: **QDA**, **LDA**, **Naive Bayes** gaussiano y **SVM con kernel RBF** (`C = 1`, `gamma = "scale"`), cada uno en dos versiones: con las 16 variables estandarizadas y precedido por PCA. Todos se entrenaron sobre la misma partición 70/30 estratificada (`random_state = 42`), con **9.527 registros de entrenamiento y 4.084 de prueba**, y con los mismos hiperparámetros con y sin PCA.

La cantidad de componentes se fijó con el criterio del 95 % de la varianza, que dio **4 componentes** (PC1 = 55,46 %, PC2 = 26,43 %, PC3 = 8,10 % y PC4 = 5,09 %; acumulado = 95,09 %):

- **PC1:** combina el tamaño en oposición a la redondez.
- **PC2:** refleja qué tan alargado es el grano (`AspectRation`, `Eccentricity` y `ShapeFactor1` frente a `MinorAxisLength` y `Compactness`).
- **PC3:** dominado por `Solidity` y `ShapeFactor4`.
- **PC4:** casi solo por `Extent`.

El desempeño se midió en el conjunto de prueba con accuracy y precision, recall y F1-score macro, y se complementó con validación cruzada estratificada de 5 grupos sobre el conjunto de entrenamiento.

```python
# Fragmento del procedimiento de partición, escalado, PCA y modelado SVM
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

svm = Pipeline([("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1, gamma="scale"))])
svm_pca = Pipeline([("scaler", StandardScaler()),
                    ("pca", PCA(n_components=4)),  # 95 % de la varianza
                    ("svm", SVC(kernel="rbf", C=1, gamma="scale"))])
# Accuracy en prueba: SVM = 0,9251 | SVM + PCA = 0,8962
```

![Figura 2. Varianza explicada por componente y acumulada](figures/fig2_varianza_explicada.png)

*Figura 2. Varianza explicada por componente (izquierda) y acumulada (derecha). El criterio del 95 % selecciona 4 componentes.*

## 4. Resultados

**Tabla 3. Desempeño de las ocho configuraciones** (conjunto de prueba, n = 4.084; validación cruzada de 5 grupos sobre entrenamiento), ordenadas por F1-score macro.

| Modelo | Accuracy | Precision | Recall | F1-score | Accuracy CV (media ± desvío) |
|---|---|---|---|---|---|
| SVM | 0,9251 | 0,9390 | 0,9359 | 0,9373 | 0,9312 ± 0,0049 |
| QDA | 0,9084 | 0,9254 | 0,9259 | 0,9245 | 0,9137 ± 0,0051 |
| LDA | 0,9040 | 0,9298 | 0,9173 | 0,9205 | 0,9046 ± 0,0055 |
| Naive Bayes | 0,8969 | 0,9094 | 0,9083 | 0,9078 | 0,8973 ± 0,0068 |
| SVM + PCA | 0,8962 | 0,9040 | 0,8967 | 0,8989 | 0,8926 ± 0,0031 |
| QDA + PCA | 0,8866 | 0,8923 | 0,8910 | 0,8906 | 0,8856 ± 0,0057 |
| LDA + PCA | 0,8825 | 0,8930 | 0,8850 | 0,8868 | 0,8783 ± 0,0070 |
| Naive Bayes + PCA | 0,8639 | 0,8753 | 0,8680 | 0,8694 | 0,8625 ± 0,0076 |

### Hallazgo 1. SVM es el mejor modelo, con una ventaja que excede el ruido de la partición

SVM tiene el mejor valor en las cuatro métricas de prueba (accuracy 0,9251; F1 macro 0,9373) y también el mejor accuracy en validación cruzada (0,9312 ± 0,0049). Le siguen QDA (F1 0,9245), LDA (0,9205) y Naive Bayes (0,9078). La diferencia entre SVM y QDA (cerca de 1,7 puntos de accuracy) es mayor que el desvío de la validación cruzada (0,5 puntos), por lo que no parece producto del azar de la partición.

El tipo de frontera importa: SVM (curva y flexible) supera a QDA (curva, pero con suposición de normalidad) y a LDA (recta), y los modelos más simples (NB y LDA) quedan por debajo. En el caso de Naive Bayes, su suposición de variables independientes no se cumple, porque muchas variables están muy correlacionadas.

### Hallazgo 2. Los errores están en clases parecidas, no en las clases chicas

Todos los modelos confunden DERMASON con SIRA (QDA clasifica como SIRA a 143 de los 1.064 granos de DERMASON; LDA, a 152; Naive Bayes, a 119; SVM mantiene la confusión pero más repartida entre ambos sentidos, con 75 y 71 granos) y BARBUNYA con CALI (28 granos en QDA y 59 en Naive Bayes). En cambio, BOMBAY, la clase más chica, se acierta siempre (recall = 1,00 en los cinco modelos), porque sus granos son mucho más grandes que los del resto.

### Hallazgo 3. PCA con 4 componentes empeoró a los cuatro algoritmos

Al aplicar PCA (95 % de la varianza), el accuracy bajó entre 2,2 y 3,3 puntos y el F1-score macro entre 3,4 y 3,8 puntos (Tabla 4). La validación cruzada muestra la misma caída (entre 2,6 y 3,9 puntos), así que no depende de la partición. El orden de los modelos casi no cambia: SVM sigue siendo el mejor con y sin PCA, Naive Bayes el peor, y ninguna versión con PCA supera a su versión original.

Naive Bayes fue el que más perdió (-3,3 puntos), aunque en teoría es el que más se beneficia de componentes decorrelacionados; además, su recall de BARBUNYA cayó a 0,63 (99 de 397 granos se predicen como CALI). Acá pesó más la información perdida que la mejora en la suposición de independencia.

**Tabla 4. Efecto de PCA (4 componentes) en cada algoritmo, conjunto de prueba.**

| Algoritmo | Accuracy sin PCA | Accuracy con PCA | Diferencia | F1 sin PCA | F1 con PCA |
|---|---|---|---|---|---|
| QDA | 0,9084 | 0,8866 | -2,2 pts | 0,9245 | 0,8906 |
| SVM | 0,9251 | 0,8962 | -2,9 pts | 0,9373 | 0,8989 |
| Naive Bayes | 0,8969 | 0,8639 | -3,3 pts | 0,9078 | 0,8694 |
| LDA | 0,9040 | 0,8825 | -2,2 pts | 0,9205 | 0,8868 |

![Figura 3. Accuracy y F1-score macro, sin PCA y con PCA](figures/fig3_efecto_pca.png)

*Figura 3. Accuracy (izquierda) y F1-score macro (derecha) de cada algoritmo, sin PCA y con PCA (4 componentes).*

### Hallazgo 4. El problema no es PCA en sí, sino la cantidad de componentes

En el barrido de validación cruzada sobre entrenamiento (Figura 4), todos los algoritmos mejoran mucho al pasar de 4 a 6 componentes: SVM llega a 0,9296, casi igual que con las 16 variables (0,9312), y QDA (0,9181), Naive Bayes (0,8998) y LDA (0,9069) igualan o superan levemente a sus versiones sin PCA (0,9137, 0,8973 y 0,9046); en QDA y LDA la diferencia está dentro del desvío de la validación cruzada.

Las componentes 5 y 6 explican poca varianza (2,69 % y 1,13 %), pero aportan información que sirve para separar clases parecidas como DERMASON/SIRA. El criterio del 95 % de varianza resultó demasiado agresivo para este problema. Este barrido es exploratorio; la comparación principal del proyecto sigue siendo la de 4 componentes.

![Figura 4. Accuracy en validación cruzada según la cantidad de componentes de PCA](figures/fig4_cv_componentes.png)

*Figura 4. Accuracy en validación cruzada según la cantidad de componentes de PCA. La línea punteada marca los 4 componentes elegidos.*

## 5. Trabajo futuro

- **Definir la cantidad de componentes por validación cruzada:** si se usara PCA en la práctica, conviene fijar la cantidad de componentes por validación cruzada y no por el porcentaje de varianza.
- **Repetir la comparación con más componentes:** evaluar también sobre el conjunto de prueba las versiones con PCA con 6 componentes, ya que el barrido se realizó solo con validación cruzada sobre entrenamiento.
- **Cuantificar el efecto de los duplicados:** medir cuánto influyen los 31 registros de prueba con una copia exacta en entrenamiento sobre las métricas reportadas.

## Tecnologías

Python, pandas, scikit-learn (`Pipeline`, `StandardScaler`, `PCA`, `QDA`, `LDA`, `GaussianNB`, `SVC`), matplotlib / seaborn.
