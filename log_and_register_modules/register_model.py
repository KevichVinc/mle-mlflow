import os

import mlflow


EXPERIMENT_NAME = "churn_vinc"
RUN_NAME = "virtual_churn_check"
REGISTRY_MODEL_NAME = "churn_model_vinc"


os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY", "")

# ваш код здесь
import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# сформируем requirements.txt и сохраним окружение проекта
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(
        "\n".join(
            [
                "mlflow",
                "scikit-learn",
                "pandas",
                "numpy",
                # при необходимости добавьте свои пакеты ниже
            ]
        )
    )

pip_requirements = '../requirements.txt'
signature = mlflow.models.infer_signature(X_test, prediction)
input_example = X_test[:10]
metadata = {'model_type': 'monthly'}

experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    # ваш код здесь
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("requirements.txt")

    model_info = mlflow.sklearn.log_model(
        cb_model=model,
        artifact_path="model",
        registered_model_name=REGISTRY_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        metadata=metadata,
        artifact_path='models',
        await_registration_for=60
    )