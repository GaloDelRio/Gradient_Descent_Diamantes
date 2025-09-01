# Gradient_Descent_Diamantes

#  Regresión Lineal con Descenso de Gradiente – Dataset Diamonds (Kaggle)

Este proyecto implementa un  algoritmo sin usar ninguna biblioteca o framework de aprendizaje máquina, ni de estadística avanzada. El objetivo es códificar **Gradient Descent** en Python, para predecir el **precio de diamantes** a partir de sus características físicas.

---

##  1. Objetivo del proyecto
El objetivo es:
- Entrenar un modelo de **regresión lineal múltiple** utilizando **Gradient Descent**.
- Predecir el **precio de diamantes** (`price`) con base en variables como `carat`, `depth`, `x`, `y`, `z`.
- Evaluar el modelo con **Error Cuadrático Medio (MSE)** y **Coeficiente de Determinación (R²)**.
- Visualizar la convergencia del algoritmo (historial del costo).

---

##  2. Dataset elegido y justificación
**Dataset:** [Diamonds (Kaggle)](https://www.kaggle.com/datasets/shivam2503/diamonds)

### ¿Por qué este dataset?
- Tiene más de **50,000 registros**, lo cual da robustez estadística.
- Variables predictivas (`carat`, `depth`, `x`, `y`, `z`) tienen **relación fuerte con el precio**.

---

## 3. Descarga y carga de datos
- El dataset ya viene en el repositosrio pero se puede cargar desde kaggle de ser necesario con una API.
- Se carga con **pandas** en un DataFrame.


---

##  4. Limpieza y preprocesamiento
1. **Eliminar duplicados** → evitar sesgos en el entrenamiento.  
2. **Revisar valores nulos** → confirmación de que el dataset está completo.  
3. **Filtrar valores inválidos** (`x`, `y`, `z` > 0) → descartar diamantes con dimensiones físicas imposibles.  
4. **Eliminar outliers** (`price < 20000`) → evitar que valores extremos distorsionen el ajuste.  

---

## 5. Selección de variables
- **Features (`X`) seleccionadas:**
  - `carat` → peso en quilates.  
  - `depth` → profundidad relativa.  
  - `x`, `y`, `z` → dimensiones físicas (longitud, ancho, alto).  

- **Variable objetivo (`y`):**
  - `price` → precio del diamante.  

---

## 6. Estandarización
Para mejorar la convergencia del Descenso de Gradiente:  
- Se **estandarizan las features (`X`)**: media = 0, desviación estándar = 1.  
- Se **estandariza también la variable objetivo (`y`)** para centrarla en 0.  

Esto asegura que todas las variables tengan la misma escala y el bias del modelo tienda a 0.

---

## 7. Implementación del Gradient Descent
Se implementa una función personalizada `gradient_descent_` que:
1. Inicializa los pesos y bias en 0.  
2. Iterativamente actualiza parámetros en base a:  
   - **Predicción:**  y_pred = X \cdot w + b  
   - **Costo (MSE):** frac{1}{n}\sum (y - y_pred)^2   
   - **Gradientes:** cálculo de derivadas parciales para `w` y `b`.  
   - **Actualización:**  
     w := w - \alpha \cdot dw, \quad b := b - \alpha \cdot db
   - donde alpha es la tasa de aprendizaje (`learning_rate`).  

3. Registra el historial de costos para graficar la convergencia.

---

## 8. Entrenamiento y evaluación
- **Hiperparámetros usados:**
  - `learning_rate = 0.001`
  - `iterations = 2000`

- **Evaluación:**
  - **MSE (costo final)** → mide el error promedio al cuadrado.  
  - **R² (Coeficiente de Determinación)** → mide qué porcentaje de la varianza en el precio es explicado por el modelo.  

En los experimentos realizados:  
- **MSE final ≈ 0.18 (en datos estandarizados).**  
- **R² ≈ 0.82**, lo que indica un ajuste fuerte del modelo.  

---

## 9. Visualización
Se grafica el **historial del costo (MSE) vs. número de iteraciones** para verificar la convergencia del algoritmo de Descenso de Gradiente.

Ejemplo de salida esperada:

<img width="661" height="452" alt="image" src="https://github.com/user-attachments/assets/8992bead-9942-4d4b-951b-ecde9271e687" />


