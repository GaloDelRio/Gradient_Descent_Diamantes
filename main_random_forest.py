# ---------------------------------------
# Random Forest para predicción de precio
# ---------------------------------------

# 0. Importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Cargar los datos de entrenamiento en un DataFrame de pandas
dataset_file = "train_diamonds.csv"  # Archivo de entrenamiento
print(f"Loading dataset from: {dataset_file}")
df = pd.read_csv(dataset_file)

# 2. Preprocesamiento de datos
# df = df.drop_duplicates()  # opcional ya que no hay en este dataset
df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]
df = df[df["price"] < 20000]

# 3. Seleccionar features y target (sin estandarizar para árboles)
feature_cols = ["carat", "depth", "x", "y", "z"]
X_full = df[feature_cols].values
y_full = df["price"].values

# 4. Separación Train/Validation interna
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# 5. Definir y entrenar el modelo RandomForest
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=10,
    max_features="log2",
    ccp_alpha=0.001,
    n_jobs=-1,
    random_state=42

)

print("\nEntrenando RandomForest...")
rf.fit(X_train, y_train)

# 6. Métricas en TRAIN y VALIDATION
y_pred_train = rf.predict(X_train)
y_pred_val = rf.predict(X_val)

r2_train = r2_score(y_train, y_pred_train)
r2_val = r2_score(y_val, y_pred_val)
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)

print(f"\nR-squared (TRAIN): {r2_train:.4f}")
print(f"R-squared (VALIDATION): {r2_val:.4f}")
print(f"MSE (VALIDATION): {mse_val:.2f}")
print(f"MAE (VALIDATION): {mae_val:.2f}")

# ------------------------------
# TEST: Evaluación con archivo separado
# ------------------------------
dataset_file_test = "test_diamonds.csv"  # Archivo de prueba
print(f"\nLoading dataset from: {dataset_file_test}")
df_test = pd.read_csv(dataset_file_test)

# Preprocesamiento igual al de train
# df_test = df_test.drop_duplicates()  # opcional
df_test = df_test[(df_test["x"] > 0) & (df_test["y"] > 0) & (df_test["z"] > 0)]
df_test = df_test[df_test["price"] < 20000]

# Preparar datos de test (sin estandarizar)
X_test = df_test[feature_cols].values
y_test = df_test["price"].values

# Predicciones en test
y_pred_test = rf.predict(X_test)

# Calcular R², MSE, MAE en test
r2_test = r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\nR-squared (TEST): {r2_test:.4f}")
print(f"MSE (TEST): {mse_test:.2f}")
print(f"MAE (TEST): {mae_test:.2f}")

# 8. Guardar predicciones de TEST a CSV
out = df_test.copy()
out["pred_price"] = y_pred_test
out[["carat", "depth", "x", "y", "z", "price", "pred_price"]].to_csv(
    "rf_predictions_test.csv", index=False
)
print("Predicciones de TEST guardadas en 'rf_predictions_test.csv'")

# 9. (Opcional) Imprimir hiperparámetros finales (para trazabilidad)
print("\nHiperparámetros del modelo:")
print(rf.get_params())

# ---------------------------------------
#  GRÁFICAS COMBINADAS PARA REPORTE
# ---------------------------------------
from sklearn.metrics import mean_absolute_error

# Métricas de TRAIN para barras combinadas
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

# Helpers
def _diagonal_limits(y_true, y_pred):
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    return lo, hi

def _calibration_params(y_true, y_pred):
    a, b = np.polyfit(y_true, y_pred, deg=1)
    return a, b

# ---------- (A) Parity: TRAIN / VALID / TEST en una sola imagen ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=False, sharey=False)
splits = [
    ("TRAIN", y_train, y_pred_train),
    ("VALIDATION", y_val, y_pred_val),
    ("TEST", y_test, y_pred_test),
]
for ax, (name, y_t, y_p) in zip(axes, splits):
    lo, hi = _diagonal_limits(y_t, y_p)
    ax.scatter(y_t, y_p, s=8, alpha=0.5)
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_title(f"Parity - {name}")
    ax.set_xlabel("Precio real")
    ax.set_ylabel("Precio predicho")
plt.tight_layout()
plt.savefig("rf_parity_all.png", dpi=300, bbox_inches="tight")
plt.close()
print("Plot saved as 'rf_parity_all.png'")

# ---------- (B) Residuales vs predicho: VALID y TEST combinados ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (name, y_t, y_p) in zip(
    axes, [("VALIDATION", y_val, y_pred_val), ("TEST", y_test, y_pred_test)]
):
    resid = y_t - y_p
    ax.scatter(y_p, resid, s=8, alpha=0.5)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"Residuales vs predicho - {name}")
    ax.set_xlabel("Precio predicho")
    ax.set_ylabel("Residual (real - predicho)")
plt.tight_layout()
plt.savefig("rf_residuals_combo.png", dpi=300, bbox_inches="tight")
plt.close()
print("Plot saved as 'rf_residuals_combo.png'")

# ---------- (C) Histogramas de residuales: VALID y TEST combinados ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (name, y_t, y_p) in zip(
    axes, [("VALIDATION", y_val, y_pred_val), ("TEST", y_test, y_pred_test)]
):
    resid = y_t - y_p
    ax.hist(resid, bins=50)
    ax.set_title(f"Histograma residuales - {name}")
    ax.set_xlabel("Residual (real - predicho)")
    ax.set_ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("rf_residual_hist_combo.png", dpi=300, bbox_inches="tight")
plt.close()
print("Plot saved as 'rf_residual_hist_combo.png'")

# ---------- (D) Métricas globales: R² y Bias en una sola figura ----------
labels = ["TRAIN", "VALID", "TEST"]
r2_vals  = [r2_train, r2_val, r2_test]
mae_vals = [mae_train, mae_val, mae_test]  # (por si lo usas aparte)
mbe_train = float(np.mean(y_pred_train - y_train))
mbe_val   = float(np.mean(y_pred_val   - y_val))
mbe_test  = float(np.mean(y_pred_test  - y_test))
bias_vals = [mbe_train, mbe_val, mbe_test]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Subplot 1: R²
axes[0].bar(labels, r2_vals)
for i,v in enumerate(r2_vals):
    axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("R²")
axes[0].set_title("Comparativa R²")
# Subplot 2: Bias
axes[1].bar(labels, bias_vals)
for i,v in enumerate(bias_vals):
    axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom" if v>=0 else "top")
axes[1].axhline(0, linewidth=1)
axes[1].set_ylabel("Bias (USD)  mean(pred − real)")
axes[1].set_title("Comparativa Bias")
plt.tight_layout()
plt.savefig("rf_metrics_r2_bias.png", dpi=300, bbox_inches="tight")
plt.close()
print("Plot saved as 'rf_metrics_r2_bias.png'")

# ---------- (E) Error por tamaño (carat): MAE y Bias en TEST (combinado) ----------
bins = [0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
labels_bins = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
test_df_plot = df_test.copy()
test_df_plot["abs_err"] = np.abs(test_df_plot["price"] - y_pred_test)
test_df_plot["bias"] = y_pred_test - test_df_plot["price"]
test_df_plot["carat_bin"] = pd.cut(test_df_plot["carat"], bins=bins, labels=labels_bins, include_lowest=True)

mae_by_bin = test_df_plot.groupby("carat_bin")["abs_err"].mean()
bias_by_bin = test_df_plot.groupby("carat_bin")["bias"].mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# MAE por bin
axes[0].bar(mae_by_bin.index.astype(str), mae_by_bin.values)
for i,v in enumerate(mae_by_bin.values):
    axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom")
axes[0].set_ylabel("MAE (USD)")
axes[0].set_xlabel("Rango de carat")
axes[0].set_title("MAE por rango de carat (TEST)")
# Bias por bin
axes[1].bar(bias_by_bin.index.astype(str), bias_by_bin.values)
for i,v in enumerate(bias_by_bin.values):
    axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom" if v>=0 else "top")
axes[1].axhline(0, linewidth=1)
axes[1].set_ylabel("Bias (USD)")
axes[1].set_xlabel("Rango de carat")
axes[1].set_title("Bias por rango de carat (TEST)")
plt.tight_layout()
plt.savefig("rf_mae_bias_by_carat_test.png", dpi=300, bbox_inches="tight")
plt.close()
print("Plot saved as 'rf_mae_bias_by_carat_test.png'")

# ---------- (F) Calibración: VALID y TEST combinados ----------
def _plot_calib(ax, y_true, y_pred, split):
    a, b = _calibration_params(y_true, y_pred)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.scatter(y_true, y_pred, s=8, alpha=0.5)
    ax.plot([lo, hi], [lo, hi], linewidth=1, label="Ideal")
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, a*xs + b, linewidth=1, label="Ajuste")
    ax.set_xlabel("Precio real")
    ax.set_ylabel("Precio predicho")
    ax.set_title(f"Calibración {split}\n(pendiente={a:.3f}, intercep.={b:.0f})")
    ax.legend()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
_plot_calib(axes[0], y_val,  y_pred_val,  "VALIDATION")
_plot_calib(axes[1], y_test, y_pred_test, "TEST")
plt.tight_layout()
plt.savefig("rf_calibration_valid_test.png", dpi=300, bbox_inches="tight")
plt.close()
print("Plot saved as 'rf_calibration_valid_test.png'")

# ---------- (G) Error por iteración (agregando árboles uno a uno) ----------
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Elegir la métrica que quieres graficar: "mse" o "mae"
metric = "mse"  # cambia a "mae" si prefieres
def _err(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) if metric == "mse" else mean_absolute_error(y_true, y_pred)

max_trees = rf.get_params()["n_estimators"]  # 300 en tu caso
step = 1  # 1 = una iteración por árbol; sube a 5/10 si quieres acelerar

rf_iter = clone(rf)
rf_iter.set_params(warm_start=True, n_estimators=0)

iters = []
err_tr, err_va, err_te = [], [], []

while rf_iter.get_params()["n_estimators"] < max_trees:
    rf_iter.set_params(n_estimators=rf_iter.get_params()["n_estimators"] + step)
    rf_iter.fit(X_train, y_train)

    yp_tr = rf_iter.predict(X_train)
    yp_va = rf_iter.predict(X_val)
    yp_te = rf_iter.predict(X_test)

    err_tr.append(_err(y_train, yp_tr))
    err_va.append(_err(y_val, yp_va))
    err_te.append(_err(y_test, yp_te))
    iters.append(rf_iter.get_params()["n_estimators"])

plt.figure(figsize=(10, 6))
plt.plot(iters, err_tr, marker="o", linewidth=1, label=f"Train {metric.upper()}")
plt.plot(iters, err_va, marker="o", linewidth=1, label=f"Validation {metric.upper()}")
plt.plot(iters, err_te, marker="o", linewidth=1, label=f"Test {metric.upper()}")
plt.xlabel("Iteración (número de árboles)")
plt.ylabel(metric.upper())
plt.title("Error por iteración (Random Forest, warm_start)")
plt.legend()
plt.tight_layout()
plt.savefig("rf_error_per_iteration.png", dpi=300, bbox_inches="tight")
plt.close()
print("Plot saved as 'rf_error_per_iteration.png'")
