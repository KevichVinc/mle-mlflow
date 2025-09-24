import os

import mlflow

from dotenv import load_dotenv

load_dotenv()

# определяем основные credentials, которые нужны для подключения к MLflow
# важно, что credentials мы передаём для себя как пользователей Tracking Service
# у вас должен быть доступ к бакету, в который вы будете складывать артефакты
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net" #endpoint бакета от YandexCloud
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") # получаем id ключа бакета, к которому подключён MLFlow, из .env
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") # получаем ключ бакета, к которому подключён MLFlow, из .env

# определяем глобальные переменные
# поднимаем MLflow локально
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

YOUR_NAME = "testing_loging" # введите своё имя для создания уникального эксперимента
assert YOUR_NAME, "введите своё имя в переменной YOUR_NAME для создания уникального эксперимента"

# название тестового эксперимента и запуска (run) внутри него
EXPERIMENT_NAME = f"test_connection_experiment_{YOUR_NAME}"
RUN_NAME = "test_connection_run_1"

# тестовые данные
METRIC_NAME = "test_metric"
METRIC_VALUE = 0

# устанавливаем host, который будет отслеживать наши эксперименты
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

# создаём тестовый эксперимент и записываем в него тестовую информацию
experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    
    mlflow.log_metric(METRIC_NAME, METRIC_VALUE)

##### 4. Проверяем себя, что в MLflow:
# - создался `experiment` с нашим именем
# - внутри эксперимента появился запуск `run`
# - внутри `run` записалась наша тестовая `metric`
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
run = mlflow.get_run(run_id)

assert "active" == experiment.lifecycle_stage
assert mlflow.get_run(run_id)
assert METRIC_VALUE == run.data.metrics[METRIC_NAME]

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    
mlflow.log_metric(METRIC_NAME, METRIC_VALUE)