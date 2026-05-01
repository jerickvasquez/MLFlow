import os
import sys
import traceback
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

# Rutas base del proyecto
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)

print("Directorio de trabajo:", workspace_dir)
print("Directorio mlruns:", mlruns_dir)

# Crear carpeta mlruns si no existe
os.makedirs(mlruns_dir, exist_ok=True)

# Configurar MLflow para guardar localmente en mlruns/
mlflow.set_tracking_uri(tracking_uri)

# Cargar dataset externo desde el repositorio UCI
# Se usa Wine Quality (vino tinto) porque tiene variables numericas continuas
# adecuadas para regresion lineal y no depende de sklearn.datasets
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
print("Descargando dataset desde:", DATA_URL)
df = pd.read_csv(DATA_URL, sep=";")
print("Filas:", df.shape[0], "| Columnas:", df.shape[1])

# Separar caracteristicas y variable objetivo
X = df.drop(columns=["quality"])
y = df["quality"]

# Dividir en entrenamiento y prueba con semilla fija para reproducibilidad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresion lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Calcular el error cuadratico medio sobre el conjunto de prueba
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print("MSE en conjunto de prueba:", round(mse, 4))

# Guardar el modelo en disco para uso local (joblib es mas eficiente que pickle para sklearn)
model_path = os.path.join(workspace_dir, "model.pkl")
joblib.dump(model, model_path)
print("Modelo guardado en:", model_path)

# Crear el experimento en MLflow o recuperarlo si ya existe
experiment_name = "CI-CD-Lab2"
experiment_id = None

try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=tracking_uri
    )
    print("Experimento creado con ID:", experiment_id)
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print("Experimento ya existia, ID:", experiment_id)
    else:
        raise e

if experiment_id is None:
    print("Error: no se pudo obtener el ID del experimento.")
    sys.exit(1)

# Iniciar un run de MLflow para registrar parametros, metricas y el modelo
run = None
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        print("Run ID:", run.info.run_id)

        # Inferir la firma del modelo a partir de los datos de entrenamiento
        # La firma documenta los tipos de entrada y salida esperados
        signature = infer_signature(X_train, model.predict(X_train))

        # Ejemplo de entrada que se guardara junto al modelo para referencia
        input_example = X_train.iloc[:3]

        # Registrar parametros del experimento
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("dataset", "winequality-red")
        mlflow.log_param("random_state", 42)

        # Registrar la metrica principal
        mlflow.log_metric("mse", mse)

        # Guardar el modelo en MLflow con su firma y ejemplo de entrada
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        print("Modelo registrado en MLflow correctamente. MSE:", round(mse, 4))

except Exception as e:
    print("Error durante el registro en MLflow:")
    traceback.print_exc()
    sys.exit(1)
