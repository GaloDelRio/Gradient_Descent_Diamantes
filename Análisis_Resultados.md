# Análisis_Resultados

## Resultados en entrenamiento (train)

- **Pesos aprendidos (w):**
  - `carat`: 0.356 → variable más influyente, más quilates = mayor precio.
  - `depth`: –0.028 → impacto negativo, cortes más profundos reducen valor.
  - `x`, `y`, `z`: ~0.16–0.20 → dimensiones mayores = más precio.
- **Bias (b):** ≈ 0 (por estandarización).
- **MSE (train):** 0.181 (estandarizado).
- **R² (train):** 0.8185 → el modelo explica el 81.9% de la variabilidad del precio en los datos de entrenamiento.

---

## Resultados en prueba (test)
- **R² (test):** 0.8178 → el modelo explica el 81.8% de la variabilidad del precio en datos no vistos.
- **MSE (test):** 0.184 (estandarizado), muy cercano al error en entrenamiento.

---

## Interpretación del modelo
- **Importancia de variables:**
  - `carat` es el factor más determinante en el precio.
  - Las dimensiones (`x`, `y`, `z`) son también muy relevantes.
  - `depth` aporta un ajuste pequeño y negativo.
- **Consistencia train vs. test:**
  - Resultados muy similares → el modelo generaliza bien.
- **Diagnóstico:**
  - No hay overfitting: el desempeño en test ≈ train.
  - No hay underfitting: el modelo explica más del 80% de la variabilidad.
  - Conclusión: el modelo tiene un buen ajuste balanceado.

---

## Conclusión
El modelo de **Regresión Lineal con Descenso de Gradiente** sobre el dataset de diamantes logra un desempeño sólido, explicando más del 80% de la variabilidad del precio usando solo características físicas.  
Los resultados son consistentes en train y test, lo que confirma que el modelo no sufre de overfitting ni underfitting y generaliza de manera adecuada.
