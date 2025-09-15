# Predicción de Precios de Diamantes 💎

Este repositorio contiene la implementación y análisis de distintos modelos de **Machine Learning** para predecir el **precio de diamantes** a partir de sus características físicas (`carat`, `depth`, `x`, `y`, `z`).  
Se trabaja con el dataset [Diamonds de Kaggle](https://www.kaggle.com/datasets/shivam2503/diamonds), que incluye más de 50,000 registros.

---

## Archivos del repositorio

### Implementaciones
- **`Gradient_Descent_Diamantes.md`** → Explicación del algoritmo de **Regresión Lineal con Descenso de Gradiente**, implementado desde cero en Python, con pasos de preprocesamiento, estandarización y entrenamiento.  
- **`RandomForest_Diamantes_Diamantes.md`** → Implementación del modelo **Random Forest Regressor**, incluyendo justificación teórica, preprocesamiento, entrenamiento, métricas y generación de gráficas de evaluación.  

### Análisis de resultados
- **`Análisis_Resultados_Gradient_Descent.md`** → Resultados de la regresión lineal con descenso de gradiente (R², MSE, interpretación de pesos y sesgo).  
- **`Análisis_Resultados_Random_Forest.md`** → Análisis detallado de varias iteraciones del Random Forest, con ajuste de hiperparámetros, diagnóstico de bias/varianza, gráficas de residuales, calibración y comparación entre train/validation/test.  

---

## Modelos implementados

### Regresión Lineal con Descenso de Gradiente
- Implementado sin librerías de ML, solo con **Python y NumPy**.  
- R² ≈ **0.82** en train y test.  
- `carat` y dimensiones físicas (`x`, `y`, `z`) son las variables más influyentes.  
- Buen balance entre entrenamiento y prueba → sin overfitting ni underfitting.

### Random Forest Regressor
- Modelo de ensamble de árboles, robusto a no linealidades y outliers.  
- R² en validación y prueba ≈ **0.88–0.89** tras ajuste de hiperparámetros.  
- Ligero overfitting en la primera configuración, corregido en iteraciones posteriores.  
- Mejor desempeño en diamantes pequeños y medianos; mayor error en diamantes grandes.  

---

## Objetivo
Comparar enfoques de **modelos lineales** (Gradient Descent) vs. **modelos de ensamble** (Random Forest), evaluando:
- Precisión predictiva (R², MSE, MAE).
- Sesgo (bias).
- Capacidad de generalización.  
- Comportamiento del error en función del tamaño del diamante.

---

## Resultados clave
- **Gradient Descent:** simple, interpretable, y buen ajuste con R²≈0.82.  
- **Random Forest:** más complejo, mejor rendimiento con R²≈0.89 y menor sesgo.  
- Ambos modelos muestran que los **diamantes grandes** son los más difíciles de predecir con exactitud.  

---

## Tecnologías usadas
- Python (NumPy, Pandas, Matplotlib, Scikit-learn).  
- Dataset: Diamonds (Kaggle).  
- Visualizaciones: Parity plots, histogramas de residuales, curvas de error.  

---

## Conclusión
Este proyecto demuestra cómo distintos enfoques de Machine Learning capturan la relación entre las características físicas de un diamante y su precio.  
- El **Gradient Descent** ofrece interpretabilidad y simplicidad.  
- El **Random Forest** logra mayor precisión y estabilidad, a costa de menor interpretabilidad.  

---

## Evaluación del desempeño

- **Conjuntos usados:**  
Ambos modelos fueron evaluados con un **conjunto de validación** y un **conjunto de prueba externo**, confirmando su capacidad de generalización.

- **Train (`train_diamonds.csv`)**  
  Dataset principal de entrenamiento.  
  - Limpieza: `x, y, z > 0` y `price < 20000` para eliminar valores inválidos y outliers.  
  - División: **80% Train** y **20% Validation** mediante `train_test_split`.  

- **Validation (20% del train)**  
  Subconjunto interno usado para ajustar hiperparámetros y controlar el sobreajuste.  

- **Test (`test_diamonds.csv`)**  
  Conjunto externo e independiente para la **evaluación final**.  
  - Mismos filtros que en train.  
  - Se genera `rf_predictions_test.csv` con precios reales y predichos.  
  

- **Bias (sesgo):**  
  - *Gradient Descent:* Bajo → errores centrados en 0, sin sobre/subestimación clara.  
  - *Random Forest:* Bajo → sesgo prácticamente nulo en validación y prueba.  

- **Varianza:**  
  - *Gradient Descent:* Media → estable entre train y test, sin grandes caídas de rendimiento.  
  - *Random Forest:* Media-Alta en la primera iteración (ligero overfit), reducida a **media** tras ajustes de hiperparámetros.  

- **Nivel de ajuste:**  
  - *Gradient Descent:* **Fit balanceado**, sin underfitting ni overfitting.  
  - *Random Forest:* Inicialmente **ligero overfitting**, corregido en iteraciones posteriores → modelo más **equilibrado**.  

- **Regularización aplicada:**  
  - *Gradient Descent:* La estandarización de variables ayuda a estabilizar el entrenamiento.  
  - *Random Forest:* Se aplicaron técnicas de regularización ajustando hiperparámetros (`max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha`), lo que redujo el sobreajuste y mejoró la estabilidad.  
