import mlflow
import numpy as np

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

# напишите код, который подключает tracking и registry uri
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}") 

experiment_id = '0'

# указываем путь до окружения
pip_requirements="../requirements.txt"

# формируем сигнатуру, дополнительно передавая параметры применения модели
signature = mlflow.models.infer_signature(
  np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
  np.array([0.1, 0.2])
)
# формируем пример входных данных
input_example = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
# предположим, мы хотим указать на то, что модель предсказывает на месяц вперёд
metadata = {"target_name": "churn"}
# путь до скрипта или ноутбука, который осуществляет обучение модели и валидацию
code_paths = ["train.py", "val_model.py"]


with mlflow.start_run(run_name="model_reg", experiment_id=experiment_id) as run:
    run_id = run.info.run_id
  
    model_info = mlflow.catboost.log_model( 
        cb_model=model,
        artifact_path="models",
        pip_requirements=pip_requirements,
        signature=signature,
        input_example=input_example,
        metadata=metadata,
        code_paths=code_paths,
        registered_model_name=REGISTRY_MODEL_NAME,
  )