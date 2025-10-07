# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import boto3
import joblib
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, log_loss
)
import mlflow
import mlflow.sklearn

# ==== 0) Конфиг из .env ====
load_dotenv()

S3_BUCKET_NAME   = os.environ["S3_BUCKET_NAME"]
S3_MODEL_KEY     = os.environ["S3_MODEL_KEY"]
S3_TEST_KEY      = os.environ["S3_TEST_KEY"]
TARGET_COL       = os.environ["TARGET_COL"]
S3_ENDPOINT_URL  = os.getenv("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
THRESHOLD        = float(os.getenv("PRED_THRESHOLD", "0.5"))
PRED_TRANSFORM   = os.getenv("PRED_TRANSFORM", "clip_0_1")   # clip_0_1 | sigmoid | identity

# ==== 1) Клиент S3 и локальные пути ====
s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT_URL)
MODEL_LOCAL_PATH = "/tmp/model.pkl"
TEST_LOCAL_PATH  = "/tmp/test.csv"

# ==== 2) Скачиваем из S3 ====
s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, MODEL_LOCAL_PATH)
s3.download_file(S3_BUCKET_NAME, S3_TEST_KEY, TEST_LOCAL_PATH)

# ==== 3) Загружаем модель ====
model = joblib.load(MODEL_LOCAL_PATH)  # sklearn.Pipeline и т.п.
if not hasattr(model, "predict"):
    raise TypeError("Загруженный объект не имеет метода .predict — проверьте S3_MODEL_KEY")

# ==== 4) Читаем тест ====
df_test = pd.read_csv(TEST_LOCAL_PATH)
if TARGET_COL not in df_test.columns:
    raise ValueError(f"TARGET_COL={TARGET_COL} не найден в {TEST_LOCAL_PATH}")

y_test = df_test[TARGET_COL].to_numpy()
X_test = df_test.drop(columns=[TARGET_COL])

# ==== 5) Предсказания ====
raw = np.asarray(model.predict(X_test), dtype=float).ravel()

def to_proba(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-arr))
    if mode == "identity":
        return arr
    return np.clip(arr, 0.0, 1.0)

proba = to_proba(raw, PRED_TRANSFORM)            # (n_samples,)
prediction = (proba >= THRESHOLD).astype(int)    # бинаризация

print("Shapes:", X_test.shape, y_test.shape, proba.shape, prediction.shape)
print("Unique y_test values:", sorted(pd.Series(y_test).dropna().unique()))

# ==== 6) Метрики ====
# Нормализуем y_test к {0,1}
y_ser = pd.Series(y_test)

if set(y_ser.dropna().unique()).issubset({0, 1}):
    y_bin = y_ser.astype(int).to_numpy()
elif set(y_ser.dropna().unique()).issubset({"No", "Yes"}):
    y_bin = (y_ser == "Yes").astype(int).to_numpy()
elif set(y_ser.dropna().unique()).issubset({False, True}):
    y_bin = y_ser.astype(bool).astype(int).to_numpy()
else:
    if np.issubdtype(y_ser.dropna().dtype, np.number):
        y_bin = (y_ser.fillna(0) > 0).astype(int).to_numpy()
    else:
        raise ValueError(f"Неподдерживаемые значения y_test: {sorted(y_ser.dropna().unique())}")

# Матрица ошибок строго для меток 0/1
cm = confusion_matrix(y_bin, prediction, labels=[0, 1])
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

# Ошибки I/II рода
err1 = fp / (fp + tn) if (fp + tn) else 0.0   # FPR
err2 = fn / (fn + tp) if (fn + tp) else 0.0   # FNR

# Наличие обоих классов
has_both = (len(np.unique(y_bin)) == 2)

# AUC: определён только если есть оба класса
auc = roc_auc_score(y_bin, proba) if has_both else float("nan")

# Для устойчивости logloss
proba_bounded = np.clip(proba, 1e-15, 1 - 1e-15)

# log_loss: при одном классе надо явно указать labels=[0,1]
try:
    logloss = log_loss(y_bin, proba_bounded, labels=[0, 1])
except ValueError:
    # ручной бинарный кросс-энтропий как fallback
    yb = y_bin.astype(float)
    logloss = float(-np.mean(yb * np.log(proba_bounded) + (1 - yb) * np.log(1 - proba_bounded)))

precision = precision_score(y_bin, prediction, zero_division=0)
recall    = recall_score(y_bin, prediction, zero_division=0)
f1        = f1_score(y_bin, prediction, zero_division=0)

metrics = {
    "err1": err1,
    "err2": err2,
    "auc": auc,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "logloss": logloss,
}

# ==== 7) Настройки MLflow ====
EXPERIMENT_NAME = "churn_vinc_2"
RUN_NAME = "virtual_churn_check_2"
REGISTRY_MODEL_NAME = "churn_model_vinc_2"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY", "")

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = exp.experiment_id

# ==== 8) requirements.txt и сигнатура ====
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(["mlflow", "scikit-learn", "pandas", "numpy"]))

pip_requirements = "requirements.txt"
signature = mlflow.models.infer_signature(X_test, proba)  # сигнатура под вероятности
input_example = X_test.head(5)
metadata = {"model_type": "monthly"}

# ==== 9) Логирование в MLflow ====
with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id):
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("requirements.txt")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="models",
        registered_model_name=REGISTRY_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        metadata=metadata,
        await_registration_for=60
    )
