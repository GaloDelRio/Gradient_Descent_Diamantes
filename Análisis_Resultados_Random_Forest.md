# Análisis_Resultados – Random Forest

# Primera iteración e hiperparámetros iniciales
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

- **`n_estimators`:** Reducir de **300** a **100**.  
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

# Segunda Iteración – Random Forest (con ajustes de hiperparámetros)

## Hiperparámetros usados
- `bootstrap`: **True**  
- `ccp_alpha`: **0.001**  
- `criterion`: **'squared_error'**  
- `max_depth`: **15**  
- `max_features`: **'sqrt'**  
- `max_leaf_nodes`: **None**  
- `max_samples`: **None**  
- `min_impurity_decrease`: **0.0**  
- `min_samples_leaf`: **5**  
- `min_samples_split`: **10**  
- `min_weight_fraction_leaf`: **0.0**  
- `n_estimators`: **100**  
- `n_jobs`: **-1**  
- `oob_score`: **False**  
- `random_state`: **42**  
- `verbose`: **0**  
- `warm_start`: **False**  

---

## Resultados en entrenamiento (train)  
- **R² (train):** 0.9139 → el modelo explica el **91.4%** de la variabilidad en entrenamiento.  
- **MSE (train):** ~1.37e6.  
- **MAE (train):** ~730.  

## Resultados en validación  
- **R² (validation):** 0.8816 → el modelo explica el **88.2%** de la variabilidad.  
- **MSE (validation):** 1,867,141.84  
- **MAE (validation):** 758.31  

## Resultados en prueba (test)  
- **R² (test):** 0.8881 → el modelo explica el **88.8%** de la variabilidad.  
- **MSE (test):** 1,793,699.29  
- **MAE (test):** 769.52  

---

## Comparación con la primera iteración
- **Generalización mejorada:** El gap entre train (0.91) y test (0.89) se redujo notablemente → menor sobreajuste.  
- **R² en valid/test aumentó levemente** respecto a la primera configuración (de 0.87–0.88 a 0.88–0.89).  
- **MSE/MAE en valid/test bajaron ligeramente**, mostrando más estabilidad.  
- **Train R² bajó de 0.98 a 0.91**, lo cual confirma que el modelo dejó de memorizar tanto y ganó capacidad de generalizar.  

---

## Interpretación de las gráficas

### 1. Calibración (Validation & Test)  
- **Pendiente:** ~0.89, intercepto ~430–440.  
- **Validation/Test:** menor dispersión que en la primera iteración.

  <img width="3570" height="1459" alt="image" src="https://github.com/user-attachments/assets/a4ef9392-864b-4e37-800d-4255446d639b" />
 

**Conclusión:** Mejor calibración, con sesgo bajo y alineación más cercana a la diagonal ideal.  

---

### 2. Error por iteración (Random Forest, warm_start)  
- **Train:** error converge en ~1.37e6.  
- **Validation/Test:** ambos se estabilizan tras ~50 árboles, sin incremento posterior.

   <img width="2970" height="1766" alt="image" src="https://github.com/user-attachments/assets/8d70fb31-8594-4e59-bf83-64afb7724684" />


**Conclusión:** Modelo estable y eficiente; 100 árboles son suficientes.  

---

### 3. MAE y Bias por rango de carat (Test)  
- **MAE:** crece con el tamaño del diamante, de ~100 USD (<0.3 carat) a >2000 USD (2–3 carats).  
- **Bias:** valores cercanos a 0 en la mayoría de los rangos, con ligera subestimación en 2–3 carats (–144 USD).

  <img width="3570" height="1166" alt="image" src="https://github.com/user-attachments/assets/76dfb1c2-f77d-4e1d-b977-04ff2f3b57d0" />


**Conclusión:** Buen desempeño en diamantes pequeños y medianos; menor sesgo negativo que en la primera iteración.  

---

### 4. Comparativa R² y Bias  
- **R²:** 0.91 (train), 0.88 (valid), 0.89 (test).  
- **Bias:** prácticamente nulo (entre –5 y +1 USD).

  <img width="3570" height="1166" alt="image" src="https://github.com/user-attachments/assets/8af16a4f-c1ef-4b7b-99a0-d7d476b51b8b" />


**Conclusión:** Métricas equilibradas entre conjuntos, con sesgo casi inexistente.  

---

### 5. Parity Plots (Train/Validation/Test)  
- **Train:** mayor dispersión que en la primera iteración → menos ajuste perfecto.  
- **Validation/Test:** mejor alineación con la diagonal, especialmente en precios altos.

  <img width="4770" height="1466" alt="image" src="https://github.com/user-attachments/assets/5c8300c3-57c6-4568-a3e8-e5381a5db2be" />


**Conclusión:** Reducción de overfitting y mejor consistencia en valid/test.  

---

### 6. Histogramas de residuales (Validation & Test)  
- **Distribución:** centrada en cero y simétrica.  
- **Colas:** menos extremas que en la primera iteración.4

  <img width="3570" height="1316" alt="image" src="https://github.com/user-attachments/assets/bba4300b-0f5d-4159-a8fa-0b2df7f279c1" />


**Conclusión:** Reducción de outliers, residuales más compactos.  

---

### 7. Residuales vs Predicted (Validation & Test)  
- **Forma:** persiste la heterocedasticidad (más error en precios altos).  
- **Dispersión:** más compacta que en la primera configuración.

  <img width="3564" height="1316" alt="image" src="https://github.com/user-attachments/assets/168fd554-01ce-460b-9ec6-ee1bb832339e" />


**Conclusión:** Aunque los errores crecen con el precio, el modelo ahora es más estable.  

---


 
