# RandomForest_Diamantes

# Predicción de precio con Random Forest – Dataset Diamonds (Kaggle)

Este proyecto entrena un **Random Forest Regressor** (bosque aleatorio) para predecir el **precio de diamantes** a partir de sus características físicas, generando además un set de **gráficas de evaluación** listo para reportes.

## ¿Qué es Random Forest y por qué usarlo aquí?

**Random Forest** es un **ensamble** de muchos árboles de decisión. Cada árbol aprende un conjunto de **reglas if-else** sobre los datos; el bosque promedia (en regresión) las predicciones de todos para **reducir varianza** y mejorar la **generalización**.

**Por qué es útil en este problema:**
- **Relaciones no lineales e interacciones**: el precio no crece estrictamente lineal con `carat` ni con dimensiones; los árboles capturan no linealidades y umbrales.
- **Robustez a escalas y outliers**: no exige estandarizar y tolera valores atípicos razonables.
- **Buen rendimiento con poco *tuning***: con hiperparámetros razonables suele dar R² altos en este dataset.

**Cómo “aprende” un Random Forest (idea rápida):**
1. **Bootstrap**: para cada árbol se toma una muestra aleatoria con reemplazo del *train*.
2. **Aleatoriedad en *features***: en cada división, el árbol considera solo un subconjunto aleatorio de variables (`max_features`).
3. **Crecimiento y poda ligera**: se controlan profundidad (`max_depth`) y tamaños mínimos de split/hoja para evitar sobreajuste.
4. **Agregación**: la predicción final es el **promedio** de todos los árboles.

**Diferencias vs. Gradient Descent**  
- *Gradient Descent* minimiza un costo ajustando **parámetros continuos** (p. ej., pesos de una regresión).  
- *Random Forest* aprende **estructuras de árbol** (regras de partición) y no usa gradientes.  
- Aquí elegimos Random Forest para capturar **no linealidad** y **interacciones** sin ingeniería de features compleja.

---

## 1. Objetivo del proyecto
- Entrenar un modelo basado en **árboles de decisión en conjunto** (Random Forest).
- Predecir el **precio** (`price`) usando `carat`, `depth`, `x`, `y`, `z`.
- Evaluar con **R²**, **MSE** y **MAE** en *train*, *validation* y *test*.
- Visualizar: *parity plots*, residuales, histogramas, **bias**, **calibración** y **error por iteración**.

---

## 2. Dataset elegido y justificación
**Dataset:** [Diamonds (Kaggle)](https://www.kaggle.com/datasets/shivam2503/diamonds)

**¿Por qué este dataset?**
- >50,000 observaciones ⇒ buena potencia estadística.
- Variables físicas con **relación fuerte** con el precio (peso y dimensiones).

---

## 3. Descarga, carga y preprocesamiento
- El dataset se carga con **pandas**.
- Limpieza mínima y **filtros físicos**:
  - `x`, `y`, `z` > 0 para descartar dimensiones inválidas.
  - `price < 20000` para reducir outliers extremos.
- **Sin estandarización**: los árboles no requieren escalado de features.

---

## 4. Selección de variables
- **Features (`X`)**: `carat`, `depth`, `x`, `y`, `z`.
- **Target (`y`)**: `price`.

---

## 5. Hiperparámetros clave (usados en el script)
- `n_estimators` → número de árboles; más árboles ⇒ menor varianza (mayor costo computacional).
- `max_depth` → limita profundidad para evitar sobreajuste.
- `min_samples_split`, `min_samples_leaf` → controlan tamaño mínimo de nodos.
- `max_features="sqrt"` → aleatoriza *features* por split (buena práctica en bosques).
- `random_state=42` → reproducibilidad.
- `n_jobs=-1` → usa todos los núcleos disponibles.

**Señales de *overfitting***: R² de *train* ≫ R² de *validation/test* o residuales con estructura marcada. En ese caso, baja `max_depth` o sube `min_samples_leaf`, `min_samples_split`.

---

## 6. Entrenamiento, validación y test
- **Split interno** con `train_test_split` (80/20) para *validation*.
- **Test externo** con `test_diamonds.csv` para estimar desempeño fuera de muestra.
- **Métricas impresas** en consola:
  - **R²**: proporción de varianza explicada (0–1; más alto es mejor).
  - **MSE**: error cuadrático medio (penaliza más los grandes errores).
  - **MAE**: error absoluto medio (en unidades de precio).

---

## 7. Gráficas generadas (archivos .png)
El script guarda imágenes listas para insertar en reportes:

Ejemplo:

1. **`rf_parity_all.png`** – *Parity plots* (Train/Validation/Test)  
   - Dispersión *precio real vs. predicho* + diagonal ideal.  
   - Puntos cerca de la diagonal ⇒ buen ajuste. Dispersión amplia ⇒ error alto.

2. **`rf_residuals_combo.png`** – *Residuales vs. predicho* (Validation/Test)  
   - Residual = real − predicho.  
   - Patrón horizontal alrededor de 0 ⇒ errores no sistemáticos.  
   - Forma de abanico ⇒ heterocedasticidad (error crece con el precio).

3. **`rf_residual_hist_combo.png`** – Histogramas de residuales (Validation/Test)  
   - Centrado en 0 y simétrico ⇒ sin sesgo fuerte.  
   - Cola pesada ⇒ algunos outliers.

4. **`rf_metrics_r2_bias.png`** – Barras de **R²** y **Bias**  
   - **Bias (MBE)** = promedio de (pred − real).  
   - Bias ≈ 0 ⇒ sin sobre/subestimación sistemática.

5. **`rf_mae_bias_by_carat_test.png`** – **MAE** y **Bias por rangos de `carat`** (Test)  
   - Muestra para qué tamaños el modelo **se equivoca más** o **se sesga**.  
   - Útil para priorizar mejoras (p. ej., agregar *features* para rangos problemáticos).

6. **`rf_calibration_valid_test.png`** – **Calibración** (pendiente/intercepto)  
   - Compara predicho vs. real con línea ideal (y=x) y una recta ajustada.  
   - Pendiente <1 ⇒ tendencia a **subestimar** valores altos; >1 ⇒ **sobreestimar**.

7. **`rf_error_per_iteration.png`** – **Error por iteración** (agregando árboles)  
   - Con `warm_start`, el bosque crece y graficamos MSE/MAE en Train/Val/Test vs. número de árboles.  
   - Esperable: **disminución** y luego **estabilización** del error. Si *validation/test* empeoran, es señal de sobreajuste o *leakage*.

---

## 8. Salida de predicciones
- Se guarda `rf_predictions_test.csv` con: `carat, depth, x, y, z, price, pred_price`.  
- Útil para análisis posteriores y *auditoría* del modelo.

---

## 9. Buenas prácticas y próximos pasos
- Validar sensibilidad a hiperparámetros (p. ej., `max_depth`, `min_samples_leaf`, `max_features`).
- Probar más *features*: razones (`x/y`, `y/z`), volumen aproximado (`x*y*z`), calidad/corte/claridad si están disponibles.
- Evaluar **intervalos de predicción** (cuantiles por árbol) y **feature importance** (Gini/Permutation).
- Considerar **Gradient Boosting/XGBoost** para potenciales mejoras de error y calibración.

---
