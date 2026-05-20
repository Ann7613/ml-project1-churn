# Análisis de Modelos - Predicción de Churn

## Preprocesamiento

Antes de entrenar los modelos, se realizó un proceso de limpieza y preparación de datos.

Las principales etapas fueron:

- Eliminación de la columna `customerID`, ya que no aporta información útil para la predicción.
- Conversión de `TotalCharges` a valores numéricos.
- Reemplazo de valores nulos usando la mediana.
- Transformación de variables categóricas mediante One Hot Encoding.
- Escalamiento de variables numéricas utilizando StandardScaler.
- División del dataset en entrenamiento y validación usando `train_test_split` con estratificación.

---

# Regresión Logística

Se eligió Regresión Logística como modelo baseline debido a que:

- funciona bien en problemas de clasificación binaria,
- es rápida computacionalmente,
- y permite interpretar fácilmente los resultados.

Resultados obtenidos:

- Accuracy: 0.815
- F1-score: 0.612

El modelo mostró un buen desempeño general y buena capacidad de generalización.

---

# Random Forest

Se eligió Random Forest porque:

- puede capturar relaciones no lineales,
- es robusto frente al ruido,
- y maneja adecuadamente interacciones complejas entre variables.

Resultados obtenidos:

- Accuracy: 0.783
- F1-score: 0.531

Aunque inicialmente tuvo menor rendimiento que Regresión Logística, mejoró luego del ajuste de hiperparámetros.

---

# Ajuste de Hiperparámetros (GridSearchCV)

Se utilizó GridSearchCV junto con validación cruzada estratificada (`StratifiedKFold`) para encontrar mejores hiperparámetros.

Para Regresión Logística se ajustaron:

- C
- solver
- class_weight

Para Random Forest se ajustaron:

- n_estimators
- max_depth
- min_samples_leaf
- class_weight

Mejores resultados luego del tuning:

## Tuned Logistic Regression

- Accuracy: 0.743
- F1-score: 0.624

## Tuned Random Forest

- Accuracy: 0.776
- F1-score: 0.642

El Random Forest ajustado obtuvo el mejor F1-score, mejorando la detección de la clase minoritaria (clientes que abandonan el servicio).

---

# Bias-Variance Tradeoff

La Regresión Logística presenta mayor sesgo pero menor varianza, ya que asume relaciones aproximadamente lineales entre las variables.

Por otro lado, Random Forest reduce el sesgo al aprender patrones más complejos y no lineales, aunque puede aumentar la varianza si no se controla adecuadamente.

El ajuste de hiperparámetros permitió mejorar el equilibrio entre bias y variance, aumentando la capacidad de generalización de ambos modelos.