# Importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error 

# 1. Cargar los datos de entrenamiento en un DataFrame de pandas
dataset_file = "train_diamonds.csv" # Archivo de entrenamiento
print(f"Loading dataset from: {dataset_file}")
df = pd.read_csv(dataset_file)

# 2. Preprocesamiento de datos
# Eliminar filas duplicadas del DataFrame (en este caso no hay, pero es buena práctica buscarlos y tratarlo cuando se encuentren los mismos)
# En este caso lo comente para no hacer ruido en los resultados
# df = df.drop_duplicates()

# Verificar si hay valores nulos en el DataFrame después de eliminar duplicados
# print(df.isnull().sum())

# Eliminar filas donde las dimensiones (x, y, z) sean cero o negativas, ya que no tienen sentido físico, encontre en este dataset que hay algunos registros con estas caracteristicas 
df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]
# Eliminar outliers de precio (precios superiores a 20000), ya que son muy pocos y pueden distorsionar el modelo
df = df[df["price"] < 20000]

# 3. Definir la función de Descenso de Gradiente

def gradient_descent_(X, y, learning_rate, iterations):
    n_samples, n_features = X.shape # Obtiene el número de muestras y características

    # Inicializa los pesos (w) y el bias (b) con ceros
    w = np.zeros(n_features)
    b = 0
    cost_history = []  # Lista para almacenar el costo (MSE) en cada iteración

    for i in range(iterations):
        # 1. Predicción: Calcula las predicciones (y_pred) usando el modelo lineal actual
        y_pred = np.dot(X, w) + b

        # 2. Costo (MSE): Calcula el Error Cuadrático Medio (MSE) como medida de costo
        cost = (1/n_samples) * np.sum((y - y_pred) ** 2)
        cost_history.append(cost)

        # 3. Gradientes: Calcula los gradientes de la función de costo con respecto a w y b
        dw = -(2/n_samples) * np.dot(X.T, (y - y_pred))
        db = -(2/n_samples) * np.sum(y - y_pred)

        # 4. Actualizar parámetros: Actualiza los pesos y el bias usando la tasa de aprendizaje
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 5. (Opcional) Imprime el costo, w y b de cada iteración
        # if i % 100 == 0:
            # print(f"Iteración {i+1}/{iterations}, Costo (MSE): {cost}, Pesos (w): {w}, Bias (b): {b}")

    # Imprime los pesos finales, el bias final y el costo final después del entrenamiento
    print("Pesos finales (w):", w)
    print("Bias final (b):", b)
    print("Costo final (MSE):", cost)

    return w, b, cost_history

# 4. Preparar los datos para el modelo
# Seleccionar las columnas de características (features) y la columna objetivo (target)
X = df[["carat", "depth", "x", "y", "z"]].values
y = df["price"].values

# Normalizar las características (X) usando estandarización (media 0, desviación estándar 1)
mu_X, std_X = X.mean(axis=0), X.std(axis=0)
X = (X - mu_X) / std_X

# Estandarizar la variable objetivo (y)
mu_y, std_y = y.mean(), y.std()
y = (y - mu_y) / std_y

# 5. Definir hiperparámetros para el Descenso de Gradiente
learning_rate = 0.001   # Tasa de aprendizaje
iterations = 2000       # Número de iteraciones

# 6. Aplicar la función de Descenso de Gradiente
w, b, cost_history = gradient_descent_(X, y, learning_rate, iterations)

# 7. Calcular el coeficiente de determinación (R-cuadrado) en TRAIN
y_pred = np.dot(X, w) + b
r2 = r2_score(y, y_pred)
print(f"R-squared (TRAIN): {r2}")

# 8. Visualizar el historial del costo
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), cost_history)
plt.xlabel("Número de Iteraciones")
plt.ylabel("Costo (MSE)")
plt.title("Historial del Costo durante el Descenso de Gradiente (TRAIN)")
plt.grid(True)
plt.savefig('cost_history_train.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'cost_history_train.png'")
plt.close()


# ------------------------------
#  TEST: Evaluación con archivo separado
# ------------------------------
dataset_file_test = "test_diamonds.csv" # Archivo de prueba
print(f"\nLoading dataset from: {dataset_file_test}")
df_test = pd.read_csv(dataset_file_test)

# Preprocesamiento igual al de train
df_test = df_test.drop_duplicates()
df_test = df_test[(df_test["x"] > 0) & (df_test["y"] > 0) & (df_test["z"] > 0)]
df_test = df_test[df_test["price"] < 20000]

# Preparar datos de test
X_test = df_test[["carat", "depth", "x", "y", "z"]].values
y_test = df_test["price"].values

# Normalizar usando medias y desviaciones del TRAIN 
X_test = (X_test - mu_X) / std_X
y_test_std = (y_test - mu_y) / std_y

# Predicciones en test
y_pred_test_std = np.dot(X_test, w) + b

# Calcular R² y MSE en test
r2_test = r2_score(y_test_std, y_pred_test_std)
mse_test = mean_squared_error(y_test_std, y_pred_test_std)
print(f"R-squared (TEST): {r2_test}")
print(f"MSE (TEST, estandarizado): {mse_test}")


