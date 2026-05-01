import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# El MSE esperado para regresion lineal sobre Wine Quality esta entre 0.4 y 0.7.
# Se define 1.0 como umbral maximo aceptable para aprobar la validacion.
THRESHOLD = 1.0

# Cargar el mismo dataset externo que se uso en entrenamiento
# Es importante usar los mismos parametros de division (random_state=42)
# para que X_test corresponda al mismo subconjunto que vio el modelo
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
print("Descargando dataset desde:", DATA_URL)
df = pd.read_csv(DATA_URL, sep=";")
print("Filas:", df.shape[0], "| Columnas:", df.shape[1])

X = df.drop(columns=["quality"])
y = df["quality"]

# Usar la misma semilla que en train.py para obtener el mismo X_test
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dimensiones de X_test:", X_test.shape)

# Conectar al tracking de MLflow local para recuperar el modelo registrado
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
mlflow.set_tracking_uri(tracking_uri)

# Buscar el experimento por nombre
experiment = mlflow.get_experiment_by_name("CI-CD-Lab2")
if experiment is None:
    print("Error: no se encontro el experimento 'CI-CD-Lab2'. Ejecuta primero make train.")
    sys.exit(1)

# Obtener el run mas reciente del experimento
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)
if runs.empty:
    print("Error: no hay runs registrados. Ejecuta primero make train.")
    sys.exit(1)

# Cargar el modelo desde el run mas reciente de MLflow
latest_run_id = runs.iloc[0].run_id
model_uri = "runs:/" + latest_run_id + "/model"
print("Cargando modelo desde MLflow:", model_uri)

try:
    model = mlflow.sklearn.load_model(model_uri)
    print("Modelo cargado correctamente.")
except Exception as e:
    print("Error al cargar el modelo desde MLflow:", e)
    sys.exit(1)

# Generar predicciones sobre el conjunto de prueba
try:
    y_pred = model.predict(X_test)
except ValueError as e:
    print("Error en la prediccion:", e)
    sys.exit(1)

# Calcular MSE y comparar contra el umbral definido
mse = mean_squared_error(y_test, y_pred)
print("MSE del modelo:", round(mse, 4), "| Umbral:", THRESHOLD)

if mse <= THRESHOLD:
    print("El modelo cumple los criterios de calidad. Pipeline aprobado.")
    sys.exit(0)
else:
    print("El modelo no cumple el umbral. Pipeline detenido.")
    sys.exit(1)
