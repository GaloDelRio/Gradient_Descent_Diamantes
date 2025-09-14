# Análisis_Resultados – Random Forest

## Primera iteración e hiperparámetros iniciales
**Hiperparámetros usados:**

- `bootstrap`: **True**  
- `ccp_alpha`: **0.0**  
- `criterion`: **'squared_error'**  
- `max_depth`: **None**  
- `max_features`: **'sqrt'**  
- `max_leaf_nodes`: **None**  
- `max_samples`: **None**  
- `min_impurity_decrease`: **0.0**  
- `min_samples_leaf`: **1**  
- `min_samples_split`: **2**  
- `min_weight_fraction_leaf`: **0.0**  
- `n_estimators`: **300**  
- `n_jobs`: **-1**  
- `oob_score`: **False**  
- `random_state`: **42**  
- `verbose`: **0**  
- `warm_start`: *(no especificado)*  

## Resultados en entrenamiento (train)  
- **R² (train):** 0.9807 → el modelo explica el **98.1%** de la variabilidad del precio en los datos de entrenamiento.  
- **MSE (train):** bajo, en el rango de ~0.3e6.  
- **MAE (train):** < 800 (consistente con el buen ajuste).  

## Resultados en validación  
- **R² (validation):** 0.8709 → el modelo explica el **87.1%** de la variabilidad en datos no vistos.  
- **MSE (validation):** 2,036,741.01  
- **MAE (validation):** 799.32  

## Resultados en prueba (test)  
- **R² (test):** 0.8786 → el modelo explica el **87.9%** de la variabilidad del precio en datos externos.  
- **MSE (test):** 1,946,537.24  
- **MAE (test):** 802.73  

---

## Interpretación del modelo  
- **Precisión general:** El modelo captura gran parte de la variabilidad, con métricas sólidas en validación y prueba.  
- **Consistencia:** Validación y test muestran resultados muy similares → buena capacidad de generalización.  
- **Gap train–valid/test:** El desempeño en train es mucho más alto que en valid/test, lo que indica un ajuste casi perfecto en entrenamiento pero con ligera pérdida de generalización.  

---

## Diagnóstico  
- **Bias (sesgo): Bajo.**  
  El error sistemático es mínimo (mean bias error cercano a cero). El modelo no presenta una tendencia clara a sobreestimar o subestimar precios.  

- **Varianza: Media-Alta.**  
  La diferencia entre train (R² ≈ 0.98) y valid/test (R² ≈ 0.87–0.88) indica cierta sensibilidad del modelo a los datos de entrenamiento.  

- **Nivel de ajuste:** **Ligeramente overfit.**  
  El modelo está muy ajustado en entrenamiento, pero la caída de ~0.11 puntos de R² al pasar a valid/test refleja sobreajuste leve. Sin embargo, no es severo porque los resultados en test son bastante buenos (>87%).  

---

## Interpretación de las gráficas

### 1. Calibración (Validation & Test)  
- Pendiente cercana a **0.89**, menor a la ideal (=1).  
- Subestima precios altos y sobreestima precios bajos.  
- Intercepto (~435–443 USD) bajo → sesgo sistemático mínimo.

<img width="3570" height="1459" alt="image" src="https://github.com/user-attachments/assets/51967057-1ded-4867-a650-07256de8ac7a" />


 **Conclusión:** El modelo está bien calibrado, con ligeras dificultades en valores extremos.  

---

### 2. Error por iteración (Random Forest, warm_start)  
- **Train:** error cae rápido y se estabiliza en valores muy bajos (~0.3e6).  
- **Validation/Test:** error se estabiliza alrededor de ~2.0e6 tras ~100 árboles.  
- Gap estable entre train y valid/test → **leve overfit, no severo**.

<img width="2970" height="1766" alt="image" src="https://github.com/user-attachments/assets/d6bb9c35-8a3f-4896-928c-44a8b8c21b8a" />


 **Conclusión:** Más de 100 árboles no aportan mejoras sustanciales.  

---

### 3. MAE y Bias por rango de carat (Test)  
- **MAE** muy bajo (<200 USD) en diamantes <0.5 carat, pero supera los **2000 USD** en diamantes grandes (>1.5 carat).  
- **Bias** cercano a 0 salvo:  
  - 1.5–2.0 carat → sobreestimación (+62 USD).  
  - 2.0–3.0 carat → subestimación clara (–197 USD).
 
  <img width="3570" height="1166" alt="image" src="https://github.com/user-attachments/assets/36151555-2366-455b-a96d-ae49fb53030e" />


 **Conclusión:** Excelente rendimiento en diamantes pequeños/medianos; errores crecen con carat.  

---

### 4. Comparativa R² y Bias  
- **R²:**  
  - Train: 0.98  
  - Validation: 0.87  
  - Test: 0.88  
- **Bias:** bajo en todos los conjuntos (máx. 20 USD en validación).

  <img width="3570" height="1166" alt="image" src="https://github.com/user-attachments/assets/0f957026-c29e-4d99-b18b-71b3e87cf9fb" />


**Conclusión:** Buen poder predictivo y sesgo mínimo.  

---

### 5. Parity Plots (Train/Validation/Test)  
- **Train:** puntos sobre la diagonal → ajuste casi perfecto.  
- **Validation/Test:** mayor dispersión en precios altos.

  <img width="4770" height="1466" alt="image" src="https://github.com/user-attachments/assets/91022d71-1596-4741-bc07-5bbc9e4fe367" />


 **Conclusión:** Ligero overfit, con pérdida de precisión en valores altos.  

---

### 6. Histogramas de residuales (Validation & Test)  
- Distribución centrada en 0, simétrica.  
- Colas largas → outliers con errores >5000 USD.

  <img width="3570" height="1316" alt="image" src="https://github.com/user-attachments/assets/de9d287f-50e1-4f27-97e9-84395413a23a" />


**Conclusión:** Normalidad en la mayoría de casos, pero errores extremos en diamantes atípicos.  

---

### 7. Residuales vs Predicted (Validation & Test)  
- Forma de embudo: residuales crecen con el precio.  
- Evidencia de **heterocedasticidad**.

  <img width="3564" height="1316" alt="image" src="https://github.com/user-attachments/assets/8dcc3185-f95a-48eb-b927-b748105747e7" />


 **Conclusión:** Modelo más preciso en diamantes baratos, menos en los caros.  

---

## Propuesta de mejoras en hiperparámetros

Con base en los resultados obtenidos y el diagnóstico de ligera heterocedasticidad y overfitting, estos son cambios que se le realizaran a los hiperparámetros del modelo:

- **`n_estimators`:** Reducir de **300** a **100–150**.  
  Justificación: después de ~100 árboles el error en validación y test ya no mejora, por lo que más árboles solo aumentan el costo computacional sin beneficio.

- **`max_depth`:** Limitar a **15** en lugar de `None`.  
  Justificación: evita que los árboles crezcan indefinidamente y memoricen outliers, reduciendo la varianza.

- **`min_samples_leaf`:** Aumentar de **1** a **5**.  
  Justificación: obliga a que cada hoja tenga suficientes muestras, lo que suaviza las predicciones y mejora la estabilidad en precios altos.

- **`min_samples_split`:** Aumentar de **2** a **10**.  
  Justificación: previene divisiones demasiado tempranas, disminuyendo el sobreajuste.

- **`max_features`:** Mantener en `"sqrt"`.  
  Justificación: favorece la diversidad entre árboles y reduce la correlación de predicciones.

- **`bootstrap`:** Mantener en **True**.  
  Justificación: la técnica de remuestreo ya está funcionando bien para reducir varianza.

- **`ccp_alpha`:** Considerar un valor pequeño >0  **0.001**.  
  Justificación: introduce poda mínima que ayuda a controlar la complejidad de los árboles.

---

 
